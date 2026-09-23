import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    return json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))


def agreed():
    data = fixture()
    data["documents"][1]["claims"][0].update(
        position="30_days", statement="Synthetic policy allows returns within 30 days.")
    return data


class PipelineTests(unittest.TestCase):
    def test_grounded_answer(self):
        result = app.run_pipeline(agreed())
        faq = result["faq"]
        self.assertEqual(faq["response"]["status"], "answered")
        self.assertEqual(faq["response"]["answer"],
                         "\n".join(e["statement"] for e in faq["evidence"]))
        self.assertEqual(len(faq["response"]["citations"]), 4)

    def test_conflict_abstention_and_research(self):
        result = app.run_pipeline(fixture())
        self.assertEqual(result["faq"]["response"]["status"], "abstained")
        self.assertEqual(result["deep"]["disagreements"][0]["topic"], "return_window")
        self.assertEqual(len(result["deep"]["disagreements"][0]["positions"]), 2)
        self.assertEqual(result["deep"]["findings"][1]["source_count"], 2)
        self.assertTrue(result["deep"]["unresolved_questions"])

    def test_answer_citations_propagate(self):
        result = app.run_pipeline(agreed())
        self.assertEqual(result["deep"]["faq_citations"], result["faq"]["response"]["citations"])
        self.assertEqual(result["deep"]["faq_status"], "answered")
        self.assertEqual(result["deep"]["unresolved_questions"], [])

    def test_empty_documents(self):
        data = fixture()
        data["documents"] = []
        result = app.run_pipeline(data)
        self.assertEqual(result["faq"]["response"]["answer"], "")
        self.assertEqual(result["deep"]["findings"][0]["positions"], [])
        self.assertEqual(len(result["deep"]["unresolved_questions"]), 3)

    def test_missing_topic(self):
        data = agreed()
        data["topics"].append("warranty")
        result = app.run_pipeline(data)
        self.assertIn("No evidence for topic: warranty", result["faq"]["blocking_reasons"])

    def test_distinct_source_threshold(self):
        data = agreed()
        data["documents"] = data["documents"][:1]
        duplicate = copy.deepcopy(data["documents"][0]["claims"][0])
        duplicate["id"] = "repeated"
        data["documents"][0]["claims"].append(duplicate)
        result = app.run_pipeline(data)
        self.assertEqual(result["faq"]["response"]["status"], "abstained")
        self.assertEqual(result["deep"]["findings"][0]["source_count"], 1)

    def test_irrelevant_claim_filtered(self):
        data = agreed()
        data["documents"][0]["claims"].append(
            {"id": "irrelevant", "topic": "shipping", "position": "free",
             "statement": "Synthetic return window shipping unrelated."})
        self.assertEqual(len(app.run_pipeline(data)["faq"]["evidence"]), 4)

    def test_invalid_inputs(self):
        for key, value in [("schema_version", True), ("min_sources", True),
                           ("min_sources", 0), ("question", " "), ("topics", []),
                           ("topics", ["x", "x"]), ("documents", {}), ("synthetic", False)]:
            with self.subTest(key=key, value=value):
                data = fixture()
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_ids(self):
        data = fixture()
        data["documents"].append(copy.deepcopy(data["documents"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)
        data = fixture()
        data["documents"][0]["claims"].append(copy.deepcopy(data["documents"][0]["claims"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)

    def test_handoff_tamper_rejected(self):
        data = agreed()
        faq = app.faq_stage(data)
        faq["evidence"][0]["statement"] = "Invented evidence"
        with self.assertRaises(app.ValidationError):
            app.deep_stage(faq, data)

    def test_valid_injected_fixture(self):
        data = agreed()
        expected = app.faq_stage(data)["response"]
        def injected(context):
            context["evidence"].clear()
            return copy.deepcopy(expected)
        self.assertEqual(app.run_pipeline(data, injected)["faq"]["response"], expected)

    def test_ungrounded_injected_answer_rejected(self):
        data = agreed()
        response = app.faq_stage(data)["response"]
        response["answer"] = "Refunds are guaranteed instantly."
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data, lambda _: response)

    def test_injected_unknown_citation_rejected(self):
        response = app.faq_stage(agreed())["response"]
        response["citations"][0]["claim_id"] = "not-present"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(agreed(), lambda _: response)

    def test_injected_answer_cannot_override_conflict(self):
        response = app.faq_stage(agreed())["response"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(fixture(), lambda _: response)

    def test_determinism_and_no_input_mutation(self):
        data = fixture()
        before = copy.deepcopy(data)
        first = app.run_pipeline(data)
        self.assertEqual(data, before)
        data["documents"].reverse()
        self.assertEqual(first, app.run_pipeline(data))

    def test_output_validation_rejects_research_tamper(self):
        data = fixture()
        result = app.run_pipeline(data)
        result["deep"]["disagreements"] = []
        with self.assertRaises(app.ValidationError):
            app.validate("output", result, data)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["does-not-exist.json"]):
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        for payload in ("{", "[]", '{"schema_version":1,"schema_version":1}', '{"x":NaN}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
