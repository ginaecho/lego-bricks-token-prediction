import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def plan_ids(self, output):
        return [row["step_id"] for row in output["adaptive"]["plan"]]

    def test_integrated_example(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["behavior"]["ranking"][0]["item_id"], "synthetic-bricks")
        self.assertEqual(self.plan_ids(result), ["setup", "build-basics"])
        self.assertTrue(all(row["explanation"] for row in result["adaptive"]["plan"]))

    def test_half_life_and_purchase_multiplier(self):
        self.data["profile"]["interests"] = []
        self.data["catalog"] = [
            {"id": "old", "title": "Old", "categories": [], "popularity": 0},
            {"id": "new", "title": "New", "categories": [], "popularity": 0}]
        self.data["events"] = [
            {"item_id": "old", "kind": "purchase", "at": "2026-08-24T12:00:00Z"},
            {"item_id": "new", "kind": "browse", "at": self.data["now"]}]
        rows = app.personalize(self.data)["ranking"]
        self.assertEqual([(r["item_id"], r["score"]) for r in rows],
                         [("old", 1.5), ("new", 1.0)])

    def test_cold_start_preferences_then_popularity(self):
        self.data["events"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(result["behavior"]["cold_start"])
        self.assertTrue(result["adaptive"]["source_cold_start"])
        self.assertEqual(result["behavior"]["ranking"][0]["item_id"], "synthetic-bricks")
        self.data["profile"]["interests"] = []
        self.assertEqual(app.personalize(self.data)["ranking"][0]["item_id"], "synthetic-paints")

    def test_behavior_changes_onboarding(self):
        self.data["profile"]["interests"] = []
        self.data["limit"] = 1
        self.data["events"] = [{"item_id": "synthetic-paints", "kind": "purchase",
                                "at": self.data["now"]}]
        output = app.run_pipeline(self.data)
        self.assertEqual(output["adaptive"]["focus_categories"], ["art"])
        self.assertEqual(self.plan_ids(output), ["setup", "art-basics"])

    def test_expert_experience(self):
        self.data["profile"]["experience"] = "expert"
        self.assertEqual(self.plan_ids(app.run_pipeline(self.data)), ["setup", "build-advanced"])

    def test_completed_prerequisite_not_repeated(self):
        self.data["profile"]["completed_steps"] = ["setup"]
        self.assertEqual(self.plan_ids(app.run_pipeline(self.data)), ["build-basics"])

    def test_prerequisite_included_despite_experience_filter(self):
        self.data["steps"][0]["experiences"] = ["expert"]
        output = app.run_pipeline(self.data)
        self.assertEqual(self.plan_ids(output), ["setup", "build-basics"])
        self.assertIn("Required prerequisite", output["adaptive"]["plan"][0]["explanation"])

    def test_empty_catalog_with_preferences(self):
        self.data["catalog"] = []
        self.data["events"] = []
        output = app.run_pipeline(self.data)
        self.assertEqual(output["behavior"]["ranking"], [])
        self.assertEqual(self.plan_ids(output), ["setup", "build-basics"])

    def test_empty_everything(self):
        self.data["catalog"] = []
        self.data["events"] = []
        self.data["steps"] = []
        self.data["profile"]["interests"] = []
        self.assertEqual(self.plan_ids(app.run_pipeline(self.data)), [])

    def test_ties_and_no_mutation(self):
        self.data["events"] = []
        self.data["profile"]["interests"] = []
        for row in self.data["catalog"]:
            row["popularity"] = 0
        original = copy.deepcopy(self.data)
        output = app.run_pipeline(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(output, app.run_pipeline(self.data))
        self.assertEqual([r["item_id"] for r in output["behavior"]["ranking"]],
                         ["synthetic-bricks", "synthetic-paints"])

    def test_invalid_input_cases(self):
        mutations = [
            lambda d: d.update(limit=True),
            lambda d: d.update(limit=0),
            lambda d: d.update(now="2026-09-23"),
            lambda d: d.update(synthetic=False),
            lambda d: d["profile"].update(experience="novice"),
            lambda d: d["profile"].update(completed_steps=["missing"]),
            lambda d: d["catalog"][0].update(popularity=float("nan")),
            lambda d: d["catalog"].append(copy.deepcopy(d["catalog"][0])),
            lambda d: d["events"][0].update(item_id="missing"),
            lambda d: d["events"][0].update(kind="click"),
            lambda d: d["events"][0].update(at="2027-01-01T00:00:00Z"),
            lambda d: d["steps"][0].update(prerequisites=["missing"]),
            lambda d: d["steps"][0].update(prerequisites=["build-basics"]),
            lambda d: d.update(unexpected=True),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                mutation(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_validated_handoff_rejects_corruption(self):
        behavior = app.personalize(self.data)
        behavior["ranking"][0]["score"] = -1
        with self.assertRaises(app.ValidationError):
            app.onboard(behavior, self.data["steps"])

    def test_adaptive_schema_rejects_wrong_order(self):
        result = app.run_pipeline(self.data)["adaptive"]
        result["plan"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate(result, "adaptive")

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                capture_output=True, text=True, cwd=ROOT, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_invalid_json_without_scratch_files(self):
        for payload in ("{broken", '{"x": NaN}', '{"x": 1, "x": 2}',
                        "[]", '{"schema_version": 1}'):
            with self.subTest(payload=payload):
                stream = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), redirect_stdout(stream):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
