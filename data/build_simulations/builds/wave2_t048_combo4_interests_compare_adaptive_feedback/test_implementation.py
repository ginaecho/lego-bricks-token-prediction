"""Tests use only labeled synthetic fixtures and standard-library tooling."""

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

    def run_data(self):
        return app.run_pipeline(self.data)

    def test_full_pipeline(self):
        result = self.run_data()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["synthetic"])
        for stage in ("compare", "adaptive", "feedback"):
            self.assertEqual(result[stage]["selected_product_id"], "camera-a")

    def test_grounded_interests(self):
        row = self.run_data()["interests"]["recommendations"][0]
        self.assertEqual(row["score"], 5)
        self.assertEqual(row["matched_interests"], ["photography", "travel"])
        self.assertTrue(any("'travel'" in reason for reason in row["explanations"]))

    def test_tag_exclusion(self):
        result = self.run_data()
        self.assertEqual(result["interests"]["excluded_ids"], ["camera-x"])
        self.assertNotIn("camera-x", [r["id"] for r in result["compare"]["ranking"]])

    def test_id_exclusion_propagates(self):
        self.data["preferences"]["excluded_ids"] = ["camera-a"]
        result = self.run_data()
        self.assertEqual(result["adaptive"]["selected_product_id"], "camera-b")
        self.assertEqual(result["feedback"]["unique_count"], 1)
        self.assertIn("f1", result["feedback"]["ignored_ids"])

    def test_unit_normalization(self):
        row = self.run_data()["compare"]["ranking"][0]
        self.assertEqual(row["attributes"], {"price": 120, "weight": 400})
        self.assertEqual(row["interest_score"], 5)
        self.assertEqual(row["attribute_score"], 6)
        self.assertEqual(row["score"], 11)

    def test_preferences_can_change_winner(self):
        self.data["products"][1]["attributes"]["price"]["value"] = 10
        self.data["preferences"]["priorities"] = {"price": 20}
        result = self.run_data()
        self.assertEqual(result["interests"]["recommendations"][0]["id"], "camera-a")
        self.assertEqual(result["compare"]["selected_product_id"], "camera-b")
        self.assertEqual(result["feedback"]["selected_product_id"], "camera-b")

    def test_limit_preserves_candidates(self):
        self.data["preferences"]["limit"] = 1
        result = self.run_data()
        self.assertEqual(len(result["compare"]["ranking"]), 1)
        self.assertEqual(result["compare"]["ranking"][0]["id"],
                         result["interests"]["recommendations"][0]["id"])

    def test_missing_attributes_remain_unknown(self):
        self.data["products"][0]["attributes"] = {}
        result = self.run_data()
        row = next(r for r in result["compare"]["ranking"] if r["id"] == "camera-a")
        self.assertEqual(row["attributes"], {"price": None, "weight": None})
        self.assertEqual(row["attribute_score"], 0)

    def test_equal_attributes(self):
        self.data["products"][1]["attributes"] = copy.deepcopy(self.data["products"][0]["attributes"])
        rows = self.run_data()["compare"]["ranking"]
        self.assertEqual([r["attribute_score"] for r in rows], [6, 6])

    def test_deterministic_tie_breaking(self):
        self.data["preferences"]["interests"] = {}
        self.data["preferences"]["priorities"] = {}
        self.data["products"].reverse()
        result = self.run_data()
        self.assertEqual([r["id"] for r in result["compare"]["ranking"]], ["camera-a", "camera-b"])

    def test_prerequisite_closure_and_explanation(self):
        plan = self.run_data()["adaptive"]["steps"]
        self.assertEqual([s["id"] for s in plan], ["safety", "setup"])
        self.assertIn("prerequisite", plan[0]["reason"])

    def test_experience_changes_plan_and_feedback_scope(self):
        self.data["preferences"]["experience"] = "expert"
        result = self.run_data()
        self.assertEqual([s["id"] for s in result["adaptive"]["steps"]], ["safety", "advanced"])
        self.assertIn("f1", result["feedback"]["ignored_ids"])
        self.assertNotIn("f5", result["feedback"]["ignored_ids"])

    def test_preference_orders_ready_onboarding_steps(self):
        self.data["onboarding_steps"].append({
            "id": "photo", "title": "Synthetic photography", "instructions": "Take fixture photo.",
            "tags": ["photography"], "experience_levels": ["beginner"], "prerequisites": ["safety"]})
        self.assertEqual([s["id"] for s in self.run_data()["adaptive"]["steps"]],
                         ["safety", "setup", "photo"])
        self.data["preferences"]["interests"]["photography"] = 10
        self.assertEqual([s["id"] for s in self.run_data()["adaptive"]["steps"]],
                         ["safety", "photo", "setup"])

    def test_deduplication_and_verbatim_evidence(self):
        result = self.run_data()["feedback"]
        self.assertEqual(result["unique_count"], 2)
        self.assertEqual(result["duplicate_count"], 1)
        usability = next(t for t in result["themes"] if t["id"] == "usability")
        self.assertEqual(usability["count"], 1)
        self.assertEqual(usability["support"][0]["source_ids"], ["f1", "f2"])
        self.assertEqual(usability["support"][0]["text"], self.data["feedback"][0]["text"])
        self.assertEqual(result["ignored_ids"], ["f4", "f5"])

    def test_dedup_does_not_merge_different_steps(self):
        self.data["feedback"].append({"id": "f6", "product_id": "camera-a",
                                     "step_id": "safety", "text": self.data["feedback"][0]["text"]})
        self.assertEqual(self.run_data()["feedback"]["unique_count"], 3)

    def test_other_theme(self):
        self.data["feedback"] = [{"id": "other", "product_id": "camera-a", "text": "Hello fixture."}]
        self.assertEqual(self.run_data()["feedback"]["themes"][0]["id"], "other")

    def test_multi_label_themes(self):
        labels = {t["id"] for t in self.run_data()["feedback"]["themes"]}
        self.assertEqual(labels, {"usability", "quality", "value"})

    def test_no_candidates(self):
        self.data["preferences"]["excluded_ids"] = ["camera-a", "camera-b", "camera-x"]
        result = self.run_data()
        self.assertEqual(result["compare"]["ranking"], [])
        self.assertIsNone(result["adaptive"]["selected_product_id"])
        self.assertEqual(result["adaptive"]["steps"], [])
        self.assertEqual(result["feedback"]["unique_count"], 0)
        self.assertEqual(len(result["feedback"]["ignored_ids"]), 5)

    def test_empty_catalog_and_no_steps(self):
        self.data["products"] = []
        self.data["feedback"] = []
        self.data["onboarding_steps"] = []
        self.assertEqual(self.run_data()["feedback"]["themes"], [])

    def test_input_unchanged_and_repeatable(self):
        before = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(first, self.run_data())
        self.assertEqual(before, self.data)

    def test_invalid_input_variations(self):
        for key, value in (("schema_version", True), ("synthetic", False), ("products", {}),
                           ("feedback", None), ("onboarding_steps", "bad")):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.data)
                bad[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)

    def test_invalid_preference_variations(self):
        for key, value in (("limit", True), ("limit", 0), ("experience", "new"),
                           ("interests", {"travel": -1}), ("interests", {"travel": float("nan")}),
                           ("priorities", {"unknown": 2}), ("excluded_ids", ["missing"])):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.data)
                bad["preferences"][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)

    def test_duplicate_ids_rejected(self):
        for key in ("products", "feedback", "onboarding_steps"):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.data)
                bad[key].append(copy.deepcopy(bad[key][0]))
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)

    def test_invalid_measurements(self):
        for measurement in ({"value": -1, "unit": "USD"}, {"value": True, "unit": "USD"},
                            {"value": 1, "unit": "EUR"}, {"value": float("inf"), "unit": "USD"}):
            with self.subTest(measurement=measurement):
                self.data["products"][0]["attributes"]["price"] = measurement
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_unknown_prerequisite_and_cycle(self):
        self.data["onboarding_steps"][0]["prerequisites"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            self.run_data()
        self.data["onboarding_steps"][0]["prerequisites"] = ["setup"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_unknown_feedback_references(self):
        for key in ("product_id", "step_id"):
            bad = copy.deepcopy(self.data)
            bad["feedback"][0][key] = "missing"
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(bad)

    def test_interest_handoff_rejects_tampering(self):
        previous = app.interests(self.data)
        previous["recommendations"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.compare(self.data, previous)

    def test_compare_handoff_rejects_tampering(self):
        discovery = app.interests(self.data)
        comparison = app.compare(self.data, discovery)
        comparison["ranking"][0]["attributes"]["price"] = 0
        with self.assertRaises(app.ValidationError):
            app.adaptive(self.data, comparison, discovery)

    def test_adaptive_handoff_rejects_tampering(self):
        result = self.run_data()
        result["adaptive"]["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.analyze_feedback(self.data, result["adaptive"], result["compare"])

    def test_feedback_validator_rejects_fabricated_excerpt(self):
        result = self.run_data()
        result["feedback"]["themes"][0]["support"][0]["text"] = "Fabrication"
        with self.assertRaises(app.ValidationError):
            app.validate("feedback", result["feedback"],
                         {"input": self.data, "adaptive": result["adaptive"]})

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), self.run_data())
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "does-not-exist.json")],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_bad_argument_count(self):
        for arguments in ([], ["one", "two"]):
            output = io.StringIO()
            with redirect_stdout(output):
                code = app.main(arguments)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for contents in ('{bad', '{"x": NaN}', '{"x": 1}', '[]', 'null'):
            with self.subTest(contents=contents):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=contents)), redirect_stdout(output):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
