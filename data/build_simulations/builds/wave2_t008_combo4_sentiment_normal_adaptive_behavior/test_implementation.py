"""Tests use exclusively synthetic local fixtures; no network or scratch files."""
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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def stages(self):
        return app.run_pipeline(self.data)["stages"]

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), ["sentiment", "research", "onboarding", "behavior"])

    def test_negative_evidence(self):
        issue = self.stages()["sentiment"]["issues"][0]
        self.assertEqual(issue["score"], -1)
        self.assertEqual([e["token"] for e in issue["evidence"]], ["broken", "confusing"])

    def test_negation(self):
        self.data["issues"][0]["text"] = "not good but never bad"
        issue = self.stages()["sentiment"]["issues"][0]
        self.assertEqual(issue["score"], 0)
        self.assertEqual([e["contribution"] for e in issue["evidence"]], [-1, 1])

    def test_severity_precedes_sentiment(self):
        self.data["issues"][0]["text"] = "excellent"
        self.data["issues"][1]["text"] = "terrible"
        self.assertEqual(self.stages()["sentiment"]["priority_order"], ["issue-1", "issue-2"])

    def test_unknown_words_neutral(self):
        self.data["issues"][0]["text"] = "xyz"
        self.assertEqual(self.stages()["sentiment"]["issues"][0]["label"], "neutral")

    def test_exact_citations(self):
        self.data["sources"][0]["text"] = "  Checkout needs an address!  Broken checkout fails."
        sources = {s["id"]: s["text"] for s in self.data["sources"]}
        for finding in self.stages()["research"]["findings"]:
            self.assertEqual(sources[finding["source_id"]][finding["start"]:finding["end"]], finding["quote"])

    def test_research_priority_propagation(self):
        stages = self.stages()
        self.assertEqual(stages["research"]["findings"][0]["issue_id"],
                         stages["sentiment"]["priority_order"][0])

    def test_newline_delimited_passages(self):
        self.data["sources"][0]["text"] = "Checkout help\nUnrelated"
        findings = self.stages()["research"]["findings"]
        self.assertEqual(findings[0]["quote"], "Checkout help")

    def test_issue_change_propagates_end_to_end(self):
        self.data["issues"] = [self.data["issues"][0]]
        before = self.stages()
        self.data["issues"][0]["text"] = "The building guide is confusing"
        after = self.stages()
        self.assertEqual(before["onboarding"]["focus_categories"], ["checkout"])
        self.assertEqual(after["onboarding"]["focus_categories"], ["building"])
        scores = lambda stages: {r["product_id"]: r["components"]["onboarding"]
                                 for r in stages["behavior"]["ranking"]}
        self.assertEqual(scores(before)["kit"], 0)
        self.assertEqual(scores(after)["kit"], 0.5)

    def test_more_recent_event_has_more_weight(self):
        self.data["events"][1]["kind"] = "browse"
        weights = self.stages()["behavior"]["event_weights"]
        self.assertGreater(weights[0]["weight"], weights[1]["weight"])

    def test_no_retrieval_fabrication(self):
        self.data["sources"] = []
        research = self.stages()["research"]
        self.assertEqual(research["findings"], [])
        self.assertEqual(research["unanswered_issue_ids"], ["issue-1", "issue-2"])

    def test_prerequisites_and_format(self):
        plan = self.stages()["onboarding"]["steps"]
        ids = [row["step_id"] for row in plan]
        self.assertLess(ids.index("profile"), ids.index("checkout"))
        self.assertEqual(ids[0], "building")
        self.assertNotIn("advanced", ids)
        self.assertIn("prerequisite", plan[ids.index("profile")]["explanation"])

    def test_experience(self):
        self.data["customer"]["experience"] = "expert"
        self.assertIn("advanced", [s["step_id"] for s in self.stages()["onboarding"]["steps"]])

    def test_completed_steps_not_repeated(self):
        self.data["customer"]["completed_steps"] = ["profile"]
        self.assertNotIn("profile", [s["step_id"] for s in self.stages()["onboarding"]["steps"]])

    def test_research_onboarding_behavior_handoffs(self):
        stages = self.stages()
        self.assertEqual(stages["research"]["focus_categories"], stages["onboarding"]["focus_categories"])
        self.assertEqual(stages["onboarding"]["focus_categories"], stages["behavior"]["focus_categories"])
        self.assertIn("issue-1", stages["onboarding"]["research_issue_ids"])
        ranked = {r["product_id"]: r for r in stages["behavior"]["ranking"]}
        self.assertEqual(ranked["checkout-help"]["components"]["onboarding"], 0.5)
        self.assertEqual(ranked["display"]["components"]["onboarding"], 0)

    def test_recency_half_life_and_purchase(self):
        weights = self.stages()["behavior"]["event_weights"]
        self.assertAlmostEqual(weights[1]["weight"], 1.5)
        self.assertAlmostEqual(weights[0]["weight"], 2 ** (-1 / 30))

    def test_cold_start(self):
        self.data["events"] = []
        output = self.stages()["behavior"]
        self.assertTrue(output["cold_start"])
        self.assertEqual(output["ranking"][0]["product_id"], "kit")
        self.assertEqual(output["ranking"][0]["components"]["history"], 0)

    def test_popularity_only_fallback(self):
        self.data["events"] = []
        self.data["sources"] = []
        self.data["customer"]["interests"] = []
        self.assertEqual(self.stages()["behavior"]["ranking"][0]["product_id"], "display")

    def test_empty_collections(self):
        for key in ("issues", "sources", "steps", "products", "events"):
            self.data[key] = []
        stages = self.stages()
        self.assertEqual(stages["behavior"]["ranking"], [])
        self.assertEqual(stages["onboarding"]["steps"], [])

    def test_deterministic_and_nonmutating(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.stages(), self.stages())
        self.assertEqual(self.data, original)

    def test_tie_breaking(self):
        self.data["events"] = []
        self.data["sources"] = []
        self.data["customer"]["interests"] = []
        for product in self.data["products"]:
            product["popularity"] = 0
        self.assertEqual([r["product_id"] for r in self.stages()["behavior"]["ranking"]],
                         ["checkout-help", "display", "kit"])

    def test_missing_field(self):
        del self.data["customer"]
        self.invalid()

    def test_unknown_field(self):
        self.data["unexpected"] = True
        self.invalid()

    def test_duplicate_id(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        self.invalid()

    def test_invalid_severity(self):
        self.data["issues"][0]["severity"] = "urgent"
        self.invalid()

    def test_invalid_experience(self):
        self.data["customer"]["experience"] = "wizard"
        self.invalid()

    def test_invalid_numbers(self):
        for value in (True, -1, 2, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.data["products"][0]["popularity"] = value
                self.invalid()

    def test_future_event(self):
        self.data["events"][0]["at"] = "2027-01-01T00:00:00Z"
        self.invalid()

    def test_timezone_required(self):
        self.data["now"] = "2026-09-23T12:00:00"
        self.invalid()

    def test_unknown_event_product(self):
        self.data["events"][0]["product_id"] = "missing"
        self.invalid()

    def test_unknown_prerequisite(self):
        self.data["steps"][0]["prerequisites"] = ["missing"]
        self.invalid()

    def test_cycle(self):
        self.data["steps"][0]["prerequisites"] = ["checkout"]
        self.invalid()

    def test_synthetic_label_required(self):
        self.data["synthetic"] = False
        self.invalid()

    def test_tampered_citation_rejected(self):
        insights = app.sentiment(self.data)
        findings = app.research(self.data, insights)
        findings["findings"][0]["quote"] = "Invented!"
        with self.assertRaises(app.ValidationError):
            app.validate_stage("research", findings, self.data, insights)

    def test_tampered_prerequisite_order_rejected(self):
        stages = self.stages()
        plan = stages["onboarding"]
        plan["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate_stage("onboarding", plan, self.data, stages["research"])

    def test_boundary_validation_stops_pipeline(self):
        with patch.object(app, "research", return_value={}), patch.object(app, "onboarding") as following:
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data)
            following.assert_not_called()

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "does-not-exist.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_bad_json_and_schema(self):
        for payload in ('{', '{"x":1,"x":2}', '[]', '{"schema_version":1}', '\ufeff{}'):
            with self.subTest(payload=payload), patch("builtins.open", mock_open(read_data=payload)):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
