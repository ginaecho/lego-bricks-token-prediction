"""Standard-library tests; all data is synthetic and no provider is called."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_purchase_weight_and_recency(self):
        result = app.run_pipeline(self.data)
        ranked = result["behavior"]["recommendations"]
        self.assertEqual([r["product_id"] for r in ranked], ["camera", "printer"])
        self.assertEqual([r["score"] for r in ranked], [4.0, 1.0])
        self.assertFalse(result["behavior"]["cold_start"])

    def test_half_life_decay(self):
        self.data["events"] = [
            {"product_id": "camera", "action": "purchase", "at": "2026-09-16T10:00:00Z"}
        ]
        self.assertEqual(app.run_pipeline(self.data)["behavior"]["recommendations"][0]["score"], 2.0)

    def test_cold_start_popularity(self):
        self.data["events"] = []
        behavior = app.run_pipeline(self.data)["behavior"]
        self.assertTrue(behavior["cold_start"])
        self.assertEqual(behavior["strategy"], "popularity")
        self.assertEqual([r["product_id"] for r in behavior["recommendations"]],
                         ["speaker", "printer"])

    def test_tie_break_is_stable(self):
        self.data["events"] = []
        for product in self.data["catalog"]:
            product["popularity"] = 1
        self.data["catalog"].reverse()
        self.assertEqual([r["product_id"] for r in app.run_pipeline(self.data)["behavior"]["recommendations"]],
                         ["camera", "printer"])

    def test_guided_closure_and_blocking(self):
        guided = app.run_pipeline(self.data)["guided"]
        by_id = {step["step_id"]: step for step in guided["steps"]}
        self.assertEqual(set(by_id), {"account", "network", "camera-ready", "printer-ready"})
        self.assertEqual(by_id["account"]["status"], "completed")
        self.assertEqual(by_id["network"]["status"], "available")
        self.assertEqual(by_id["camera-ready"]["missing_prerequisite_ids"], ["network"])
        self.assertEqual(by_id["camera-ready"]["status"], "blocked")
        self.assertEqual(guided["next_step_ids"], ["network"])
        self.assertEqual(guided["progress"], {"completed": 1, "total": 4,
                                            "fraction": 0.25, "is_complete": False})

    def test_cross_stage_personalized_handoff(self):
        self.data["settings"]["top_n"] = 1
        warm = app.run_pipeline(self.data)
        self.assertEqual(warm["guided"]["source_product_ids"], ["camera"])
        self.data["events"] = []
        cold = app.run_pipeline(self.data)
        self.assertEqual(cold["guided"]["source_product_ids"], ["speaker"])
        self.assertEqual([s["step_id"] for s in cold["guided"]["steps"]],
                         ["account", "speaker-ready"])
        self.assertTrue(all(s["product_ids"] == ["speaker"] for s in cold["guided"]["steps"]))

    def test_shared_prerequisites_are_deduplicated(self):
        guided = app.run_pipeline(self.data)["guided"]
        network = [s for s in guided["steps"] if s["step_id"] == "network"]
        self.assertEqual(len(network), 1)
        self.assertEqual(network[0]["product_ids"], ["camera", "printer"])

    def test_progress_unlocks_and_finishes(self):
        self.data["completed_step_ids"].append("network")
        guided = app.run_pipeline(self.data)["guided"]
        self.assertEqual(guided["next_step_ids"], ["camera-ready", "printer-ready"])
        self.assertEqual(guided["progress"]["fraction"], 0.5)
        self.data["completed_step_ids"].extend(["camera-ready", "printer-ready"])
        guided = app.run_pipeline(self.data)["guided"]
        self.assertTrue(guided["progress"]["is_complete"])
        self.assertEqual(guided["next_step_ids"], [])

    def test_unselected_completions_not_counted(self):
        self.data["completed_step_ids"].append("speaker-ready")
        self.assertEqual(app.run_pipeline(self.data)["guided"]["progress"]["completed"], 1)

    def test_empty_catalog_and_plan(self):
        self.data.update(catalog=[], steps=[], events=[], completed_step_ids=[])
        result = app.run_pipeline(self.data)
        self.assertEqual(result["behavior"]["recommendations"], [])
        self.assertEqual(result["guided"]["progress"], {
            "completed": 0, "total": 0, "fraction": 1.0, "is_complete": True
        })

    def test_products_without_setup(self):
        for product in self.data["catalog"]:
            product["setup_step_ids"] = []
        self.assertEqual(app.run_pipeline(self.data)["guided"]["steps"], [])

    def test_invalid_prerequisite_graph(self):
        for prerequisites in (["missing"], ["network"], ["camera-ready"]):
            with self.subTest(prerequisites=prerequisites):
                data = copy.deepcopy(self.data)
                data["steps"][1]["prerequisites"] = prerequisites
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_completion(self):
        for completed in (["unknown"], ["network"], ["account", "account"]):
            with self.subTest(completed=completed):
                self.data["completed_step_ids"] = completed
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_invalid_events(self):
        changes = (("at", "2027-01-01T00:00:00Z"), ("at", "2026-01-01"),
                   ("at", "bad"), ("action", "click"), ("product_id", "unknown"))
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["events"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_types_and_numbers(self):
        for value in (0, -1, True, "2", float("nan"), float("inf")):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["settings"]["half_life_days"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        self.data["schema_version"] = True
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_duplicate_catalog_and_unknown_fields(self):
        data = copy.deepcopy(self.data)
        data["catalog"].append(copy.deepcopy(data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)
        self.data["unexpected"] = 1
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_handoff_tampering_rejected(self):
        artifact = app.behavior_stage(self.data)
        artifact["behavior"]["recommendations"][0]["product_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.guided_stage(artifact)

    def test_final_validation_rejects_progress_tampering(self):
        result = app.run_pipeline(self.data)
        result["guided"]["progress"]["fraction"] = 0.99
        with self.assertRaises(app.ValidationError):
            app.Schema.artifact(result, final=True)

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(self.data, original)
        handoff = app.behavior_stage(self.data)
        snapshot = copy.deepcopy(handoff)
        app.guided_stage(handoff)
        self.assertEqual(handoff, snapshot)

    def test_equivalent_timezone_offsets(self):
        baseline = app.run_pipeline(self.data)["behavior"]
        self.data["now"] = "2026-09-23T12:00:00+02:00"
        self.assertEqual(app.run_pipeline(self.data)["behavior"], baseline)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        app.Schema.artifact(json.loads(process.stdout), final=True)

    def test_cli_file_json_and_usage_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], [str(ROOT / "implementation.py")]):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    capture_output=True, text=True, cwd=ROOT, check=False,
                )
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stderr, "")
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_validation_error_and_strict_json(self):
        from unittest.mock import mock_open, patch
        for payload in ('{"schema_version":1}', '{"a":1,"a":2}', '{"a":NaN}', '[]'):
            with self.subTest(payload=payload):
                stream = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), contextlib.redirect_stdout(stream):
                    self.assertEqual(app.main(["synthetic-input.json"]), 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
