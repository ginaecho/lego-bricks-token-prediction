import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["onboarding"]["recommended_product_id"], "starter")
        self.assertEqual(result["onboarding"]["next_step"]["id"], "learn_basics")
        app.validate(result, "onboarding")

    def test_insight_evidence_and_actions(self):
        state = app.customer_insights(self.data)
        themes = {t["id"]: t for t in state["insights"]["themes"]}
        self.assertEqual(themes["usability"]["feedback_ids"], ["f1", "f3"])
        self.assertEqual(themes["usability"]["negative_count"], 1)
        self.assertTrue(themes["pricing"]["action"])

    def test_unknown_feedback(self):
        self.data["feedback"] = [{"id": "f", "customer_id": "c", "text": "Purple branding."}]
        self.assertEqual(app.customer_insights(self.data)["insights"]["themes"][0]["id"], "other")

    def test_search_synonyms(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["search"]["query_terms"], ["affordable", "easy", "workspace"])
        self.assertIn("affordable", result["search"]["results"][0]["matched_terms"])

    def test_search_typo(self):
        self.data["query"] = "workspce"
        self.assertEqual(len(app.run_pipeline(self.data)["search"]["results"]), 3)

    def test_cross_stage_feedback_changes_ranking(self):
        self.data["query"] = "affordable"
        self.data["feedback"] = []
        no_feedback = app.run_pipeline(self.data)
        self.assertEqual(no_feedback["search"]["results"][0]["product_id"], "lite")
        self.data["feedback"] = [{"id": "f", "customer_id": "c", "text": "Need a tutorial"}]
        with_feedback = app.run_pipeline(self.data)
        self.assertEqual(with_feedback["search"]["results"][0]["product_id"], "starter")
        self.assertEqual(with_feedback["onboarding"]["recommended_product_id"], "starter")
        self.assertEqual(with_feedback["onboarding"]["theme_ids"], ["support"])

    def test_cross_stage_query_changes_onboarding(self):
        self.data["query"] = "automation"
        result = app.run_pipeline(self.data)
        self.assertEqual(result["onboarding"]["recommended_product_id"], "advanced")
        self.assertIn("Advanced", result["onboarding"]["next_step"]["instruction"])

    def test_no_matches_does_not_force_theme_result(self):
        self.data["query"] = "telescope"
        result = app.run_pipeline(self.data)
        self.assertEqual(result["search"]["results"], [])
        self.assertEqual(result["onboarding"]["next_step"]["id"], "choose_product")

    def test_empty_catalog_and_feedback(self):
        self.data["products"] = []
        self.data["feedback"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["insights"]["themes"], [])
        self.assertIsNone(result["onboarding"]["recommended_product_id"])

    def test_blank_query_uses_goal(self):
        self.data["query"] = " "
        self.data["customer"]["goal"] = "automation"
        self.assertEqual(app.run_pipeline(self.data)["search"]["results"][0]["product_id"], "advanced")

    def test_blank_query_and_goal_use_insights(self):
        self.data["query"] = ""
        self.data["customer"]["goal"] = ""
        result = app.run_pipeline(self.data)
        self.assertTrue(result["search"]["results"])
        self.assertEqual(result["onboarding"]["next_step"]["id"], "set_goal")

    def test_no_signals_no_recommendation(self):
        self.data["query"] = ""
        self.data["customer"]["goal"] = ""
        self.data["feedback"] = []
        self.assertEqual(app.run_pipeline(self.data)["search"]["results"], [])

    def test_experience_and_completed_steps(self):
        self.data["customer"]["experience"] = "experienced"
        self.data["customer"]["completed_steps"] = ["configure_product"]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["onboarding"]["next_step"]["id"], "complete_first_task")
        self.assertIn("Organize team tasks", result["onboarding"]["next_step"]["instruction"])

    def test_completed_onboarding(self):
        self.data["customer"]["completed_steps"] = sorted(app.STEPS)
        result = app.run_pipeline(self.data)
        self.assertEqual(result["onboarding"]["status"], "complete")
        self.assertIsNone(result["onboarding"]["next_step"])

    def test_limit_and_determinism_and_no_mutation(self):
        self.data["limit"] = 1
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(len(first["search"]["results"]), 1)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_invalid_root_and_types(self):
        for value in (None, [], {}, "input"):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(value)
        for key, value in (("limit", True), ("limit", 0), ("query", 4),
                           ("feedback", {}), ("schema_version", True), ("synthetic", False)):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_duplicate_ids_and_unknown_fields(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data["products"].pop()
        self.data["unknown"] = "field"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_tampered_insights_rejected_by_search(self):
        state = app.customer_insights(self.data)
        state["insights"]["themes"][0]["feedback_ids"][0] = "nonexistent"
        with self.assertRaises(app.ValidationError):
            app.product_search(state)

    def test_tampered_search_rejected_by_onboarding(self):
        state = app.product_search(app.customer_insights(self.data))
        state["search"]["results"][0]["product_id"] = "nonexistent"
        with self.assertRaises(app.ValidationError):
            app.guided_onboarding(state)

    def test_nonfinite_score_rejected(self):
        state = app.product_search(app.customer_insights(self.data))
        state["search"]["results"][0]["score"] = float("nan")
        with self.assertRaises(app.ValidationError):
            app.guided_onboarding(state)

    def test_invalid_customer_and_tags(self):
        for field, value in (("experience", "expert"), ("completed_steps", ["unknown"]),
                             ("id", " ")):
            data = copy.deepcopy(self.data)
            data["customer"][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)
        self.data["products"][0]["tags"] = ["same", "same"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "nonexistent.json")]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_invalid_and_duplicate_json(self):
        for content in ("{", "[]", '{"x":1,"x":2}', '{"x":NaN}'):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)):
                output = io.StringIO()
                with patch("sys.stdout", output):
                    code = app.main(["synthetic-memory-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
