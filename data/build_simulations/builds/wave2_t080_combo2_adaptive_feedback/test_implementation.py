"""Tests use only labeled synthetic records; no network or external dependencies."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_prerequisite_closure_and_explanations(self):
        result = app.adaptive_onboarding(self.data)
        self.assertEqual([step["id"] for step in result["steps"]],
                         ["account", "listings", "analytics"])
        self.assertEqual(result["steps"][2]["prerequisites"], ["listings"])
        self.assertIn("beginner", result["steps"][1]["explanation"])
        self.assertEqual(len(result["steps"][1]["instructions"]), 3)

    def test_experience_and_preference_adaptation(self):
        for experience, preference, expected in [
            ("beginner", "concise", 1), ("advanced", "guided", 1),
            ("intermediate", "guided", 2), ("beginner", "guided", 3),
        ]:
            with self.subTest(experience=experience, preference=preference):
                self.data["profile"].update(experience=experience, preference=preference)
                result = app.run_pipeline(self.data)
                self.assertEqual(len(result["onboarding"]["steps"][1]["instructions"]),
                                 expected)
                recommendation = result["feedback"]["themes"][0]["recommendation"]
                self.assertIn("a concise checklist" if expected == 1
                              else "step-by-step guidance", recommendation)

    def test_completion_propagates_to_feedback(self):
        result = app.run_pipeline(self.data)
        account = next(group for group in result["feedback"]["groups"]
                       if group["step_id"] == "account")
        self.assertEqual(account["step_status"], "completed")
        self.assertIn("no repeat", result["onboarding"]["steps"][0]["explanation"])

    def test_deduplication_preserves_all_exact_excerpts(self):
        result = app.run_pipeline(self.data)["feedback"]
        self.assertEqual((result["source_count"], result["unique_count"],
                          result["duplicate_count"]), (4, 3, 1))
        self.assertEqual(result["groups"][0]["excerpts"], self.data["feedback"][:2])
        self.assertEqual(result["groups"][0]["source_ids"],
                         ["synthetic-001", "synthetic-002"])

    def test_themes_trace_to_groups_and_source_records(self):
        result = app.run_pipeline(self.data)["feedback"]
        sources = {source["id"]: source for source in self.data["feedback"]}
        groups = {group["id"]: group for group in result["groups"]}
        self.assertEqual({theme["name"] for theme in result["themes"]},
                         {"learning", "pricing", "navigation", "performance", "setup"})
        for theme in result["themes"]:
            expected = [excerpt for group_id in theme["group_ids"]
                        for excerpt in groups[group_id]["excerpts"]]
            self.assertEqual(theme["supporting_excerpts"], expected)
            for excerpt in theme["supporting_excerpts"]:
                self.assertEqual(excerpt, sources[excerpt["id"]])

    def test_same_text_on_different_steps_is_not_merged(self):
        self.data["feedback"] = [
            {"id": "synthetic-a", "step_id": "account", "text": "Needs help"},
            {"id": "synthetic-b", "step_id": "listings", "text": "Needs help"},
        ]
        self.assertEqual(app.run_pipeline(self.data)["feedback"]["unique_count"], 2)

    def test_unicode_normalization_and_other_theme(self):
        self.data["feedback"] = [
            {"id": "synthetic-a", "step_id": "account", "text": "ＧＲＥＡＴ!"},
            {"id": "synthetic-b", "step_id": "account", "text": " great "},
        ]
        result = app.run_pipeline(self.data)["feedback"]
        self.assertEqual(result["unique_count"], 1)
        self.assertEqual(result["themes"][0]["name"], "other")

    def test_empty_feedback(self):
        self.data["feedback"] = []
        self.assertEqual(app.run_pipeline(self.data)["feedback"], {
            "source_count": 0, "unique_count": 0, "duplicate_count": 0,
            "groups": [], "themes": [],
        })

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)
        handoff = app.adaptive_onboarding(self.data)
        self.data["feedback"][0]["text"] = "Changed source"
        self.assertEqual(handoff["source_feedback"][0]["text"],
                         original["feedback"][0]["text"])

    def test_invalid_inputs(self):
        cases = []
        for key, value in [("schema_version", True), ("schema_version", 2),
                           ("synthetic", False), ("feedback", {}), ("extra", 1)]:
            value_doc = copy.deepcopy(self.data)
            value_doc[key] = value
            cases.append(value_doc)
        for key, value in [("goals", []), ("goals", ["sell", "sell"]),
                           ("experience", "expert"), ("preference", "video"),
                           ("completed_steps", ["analytics"])]:
            value_doc = copy.deepcopy(self.data)
            value_doc["profile"][key] = value
            cases.append(value_doc)
        for field, value in [("text", " "), ("text", "!!!"), ("step_id", "unknown")]:
            value_doc = copy.deepcopy(self.data)
            value_doc["feedback"][0][field] = value
            cases.append(value_doc)
        duplicate = copy.deepcopy(self.data)
        duplicate["feedback"][1]["id"] = duplicate["feedback"][0]["id"]
        cases.append(duplicate)
        cases.extend([None, [], {"schema_version": 1}])
        for document in cases:
            with self.subTest(document=document):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(document)

    def test_feedback_outside_selected_plan_rejected(self):
        self.data["profile"]["goals"] = ["start"]
        with self.assertRaisesRegex(app.ValidationError, "outside onboarding"):
            app.run_pipeline(self.data)

    def test_invalid_handoff_rejected_before_feedback(self):
        for change in ("order", "prerequisites", "status", "instructions"):
            handoff = app.adaptive_onboarding(self.data)
            if change == "order":
                handoff["steps"].reverse()
            elif change == "prerequisites":
                handoff["steps"][1]["prerequisites"] = []
            elif change == "status":
                handoff["steps"][0]["status"] = "planned"
            else:
                handoff["steps"][1]["instructions"] = []
            with self.subTest(change=change):
                with self.assertRaises(app.ValidationError):
                    app.analyze_feedback(handoff)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), app.run_pipeline(self.data))
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "absent-synthetic-input.json")],
                     ["one", "two"]]:
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_validation_errors(self):
        for content in ["{", '{"x": NaN}', '{"x": 1, "x": 2}', "null",
                        json.dumps({**self.data, "schema_version": 77})]:
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", return_value=io.StringIO(content)):
                    with patch("sys.stdout", output):
                        status = app.main(["synthetic-fixture.json"])
                self.assertEqual(status, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
