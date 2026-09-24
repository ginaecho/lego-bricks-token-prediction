"""Tests use only synthetic fixtures; no provider, network, or scratch files."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_fixture(self):
        return app.run_pipeline(self.fixture)

    def test_full_pipeline_and_order(self):
        output = self.run_fixture()
        self.assertEqual(list(output["stages"]), list(app.ORDER))
        self.assertIs(app.validate(output, 4), output)

    def test_research_quotes_source(self):
        result = app.research(self.fixture)["stages"]["research"]
        self.assertEqual(result["findings"][0]["evidence"][0]["excerpt"],
                         self.fixture["input"]["sources"][0]["content"])

    def test_research_unknown_question(self):
        self.fixture["input"]["questions"][0]["text"] = "Quantum zebra?"
        finding = app.research(self.fixture)["stages"]["research"]["findings"][0]
        self.assertEqual(finding["status"], "insufficient_evidence")
        self.assertEqual(finding["evidence"], [])

    def test_onboarding_personalized(self):
        onboard = self.run_fixture()["stages"]["onboard"]
        self.assertIn("Synthetic Casey", onboard["steps"][0]["action"])
        self.assertEqual(onboard["goal"], self.fixture["input"]["customer"]["goal"])
        self.assertEqual(onboard["research_question_id"], "q-setup")
        self.assertTrue(onboard["needs_assistance"])

    def test_experienced_customer(self):
        self.fixture["input"]["customer"]["experience"] = "experienced"
        self.assertFalse(self.run_fixture()["stages"]["onboard"]["needs_assistance"])

    def test_no_sources_fallback(self):
        self.fixture["input"]["sources"] = []
        result = self.run_fixture()["stages"]
        self.assertTrue(result["onboard"]["needs_assistance"])
        self.assertEqual(result["onboard"]["source_ids"], [])
        self.assertTrue(all(r["status"] == "escalated" for r in result["support"]["responses"]))

    def test_feedback_grouping_and_counts(self):
        themes = {t["name"]: t for t in self.run_fixture()["stages"]["insights"]["themes"]}
        self.assertEqual(themes["onboarding"]["feedback_ids"], ["f-1", "f-3"])
        self.assertEqual(themes["onboarding"]["negative_count"], 1)
        self.assertEqual(themes["billing"]["count"], 1)

    def test_feedback_unknown_theme(self):
        self.fixture["input"]["feedback"][0]["text"] = "Wonderful colors."
        names = [t["name"] for t in self.run_fixture()["stages"]["insights"]["themes"]]
        self.assertIn("other", names)

    def test_empty_feedback_and_tickets(self):
        self.fixture["input"]["feedback"] = []
        self.fixture["input"]["tickets"] = []
        result = self.run_fixture()["stages"]
        self.assertEqual(result["insights"]["themes"], [])
        self.assertEqual(result["support"]["responses"], [])

    def test_grounded_support(self):
        response = self.run_fixture()["stages"]["support"]["responses"][1]
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["citations"], ["s-refund"])
        self.assertEqual(response["answer"], self.fixture["input"]["sources"][1]["content"])

    def test_unknown_support_escalation(self):
        response = self.run_fixture()["stages"]["support"]["responses"][2]
        self.assertEqual(response["status"], "escalated")
        self.assertEqual(response["citations"], [])
        self.assertIn("human review", response["answer"])

    def test_cross_stage_propagation(self):
        stages = self.run_fixture()["stages"]
        self.assertEqual(stages["onboard"]["source_ids"],
                         [e["source_id"] for e in stages["research"]["findings"][0]["evidence"]])
        self.assertEqual(stages["insights"]["onboarding_next_step_id"], stages["onboard"]["next_step_id"])
        response = stages["support"]["responses"][0]
        theme = next(t for t in stages["insights"]["themes"] if t["name"] == response["theme"])
        self.assertEqual(response["follow_up"], theme["action"])
        self.assertEqual(response["next_step_id"], stages["insights"]["onboarding_next_step_id"])

    def test_unresearched_source_cannot_ground_support(self):
        self.fixture["input"]["questions"] = self.fixture["input"]["questions"][:1]
        response = self.run_fixture()["stages"]["support"]["responses"][1]
        self.assertEqual(response["status"], "escalated")

    def test_deterministic_and_nonmutating(self):
        before = copy.deepcopy(self.fixture)
        self.assertEqual(self.run_fixture(), self.run_fixture())
        self.assertEqual(self.fixture, before)

    def test_stage_cannot_skip_predecessor(self):
        for stage in (app.onboard, app.insights, app.support):
            with self.subTest(stage=stage.__name__), self.assertRaises(app.ValidationError):
                stage(self.fixture)

    def test_corrupted_evidence_rejected(self):
        document = app.research(self.fixture)
        document["stages"]["research"]["findings"][0]["evidence"][0]["excerpt"] = "Fabrication"
        with self.assertRaises(app.ValidationError):
            app.onboard(document)

    def test_corrupted_onboarding_reference_rejected(self):
        document = app.onboard(app.research(self.fixture))
        document["stages"]["onboard"]["source_ids"] = ["invented"]
        with self.assertRaises(app.ValidationError):
            app.insights(document)

    def test_corrupted_insights_rejected(self):
        document = app.insights(app.onboard(app.research(self.fixture)))
        document["stages"]["insights"]["themes"][0]["count"] = 999
        with self.assertRaises(app.ValidationError):
            app.support(document)

    def test_corrupted_answer_rejected(self):
        document = self.run_fixture()
        document["stages"]["support"]["responses"][0]["answer"] = "Invented answer"
        with self.assertRaises(app.ValidationError):
            app.validate(document, 4)

    def test_invalid_customer_reference(self):
        self.fixture["input"]["feedback"][0]["customer_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            self.run_fixture()

    def test_invalid_inputs(self):
        for field, value in (("questions", []), ("sources", None), ("feedback", "text"),
                             ("tickets", [{}]), ("customer", [])):
            with self.subTest(field=field):
                document = copy.deepcopy(self.fixture)
                document["input"][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(document)

    def test_duplicate_ids(self):
        self.fixture["input"]["sources"].append(copy.deepcopy(self.fixture["input"]["sources"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_fixture()

    def test_blank_and_oversized_text(self):
        for value in ("   ", "x" * 2001, 12, None):
            with self.subTest(value_type=type(value).__name__):
                self.fixture["input"]["customer"]["goal"] = value
                with self.assertRaises(app.ValidationError):
                    self.run_fixture()

    def test_invalid_schema_and_unknown_fields(self):
        for field, value in (("schema_version", "9"), ("synthetic", "yes"), ("extra", 1)):
            with self.subTest(field=field):
                document = copy.deepcopy(self.fixture)
                document[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(document)

    def test_keyword_tie_deterministic(self):
        sources = [{"id": "z", "title": "setup", "content": "guide"},
                   {"id": "a", "title": "setup", "content": "guide"}]
        self.assertEqual([s["id"] for s in app.source_matches("setup", sources)], ["a", "z"])

    def test_unicode_customer_name(self):
        self.fixture["input"]["customer"]["name"] = "合成顧客"
        self.assertIn("合成顧客", self.run_fixture()["stages"]["onboard"]["steps"][0]["action"])

    def test_maximum_valid_customer_text(self):
        self.fixture["input"]["customer"]["name"] = "N" * 2000
        self.fixture["input"]["customer"]["goal"] = "G" * 2000
        self.assertEqual(self.run_fixture()["status"], "ok")

    def test_tampered_theme_action_rejected(self):
        document = app.insights(app.onboard(app.research(self.fixture)))
        document["stages"]["insights"]["themes"][0]["action"] = "Invent a policy"
        with self.assertRaises(app.ValidationError):
            app.support(document)

    def test_tampered_follow_up_rejected(self):
        document = self.run_fixture()
        document["stages"]["support"]["responses"][0]["follow_up"] = "Invent a policy"
        with self.assertRaises(app.ValidationError):
            app.validate(document, 4)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                 str(HERE / "example_input.json")],
                                capture_output=True, text=True, cwd=HERE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                 str(HERE / "does-not-exist.json")],
                                capture_output=True, text=True, cwd=HERE)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for raw in ("{", "null", "[]", '{"x":1,"x":2}', '{"x": NaN}'):
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch.object(app.Path, "read_text", return_value=raw), contextlib.redirect_stdout(output):
                    code = app.main(["mock-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_unreadable_file(self):
        for error in (PermissionError("denied"), UnicodeError("invalid encoding")):
            output = io.StringIO()
            with patch.object(app.Path, "read_text", side_effect=error), contextlib.redirect_stdout(output):
                code = app.main(["mock-input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
