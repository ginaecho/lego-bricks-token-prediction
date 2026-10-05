import copy
import json
import pathlib
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
import contextlib
import io

import implementation as app


ROOT = pathlib.Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def reject(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_integrated_deterministic(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_guided_progress(self):
        guided = app.guided_setup(self.data)["guided"]
        self.assertEqual(guided["progress"]["completed"], 1)
        self.assertEqual(guided["available_steps"], ["verify_telemetry"])

    def test_empty_onboarding(self):
        self.data["completed_steps"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["guided"]["progress"]["fraction"], 0)
        self.assertEqual([s["action"] for s in result["journey"]["steps"]],
                         ["verify_work_order", "verify_telemetry"])

    def test_completed_onboarding(self):
        self.data["completed_steps"] = [s for s, _ in app.STEPS]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["guided"]["progress"]["fraction"], 1)
        self.assertEqual([s["action"] for s in result["journey"]["steps"]],
                         ["review_maintenance", "request_human_release"])

    def test_prerequisites(self):
        self.data["completed_steps"] = ["review_inspections"]
        self.reject()

    def test_duplicate_steps(self):
        self.data["completed_steps"] *= 2
        self.reject()

    def test_two_step_journey(self):
        state = app.run_pipeline(self.data)
        completed = set(state["guided"]["completed_steps"])
        for step in state["journey"]["steps"]:
            self.assertLessEqual(set(step["prerequisites"]), completed)
            completed.add(step["action"])
        self.assertEqual(len(state["journey"]["steps"]), 2)

    def test_guided_tamper(self):
        state = app.guided_setup(self.data)
        state["guided"]["progress"]["completed"] = 3
        with self.assertRaises(app.ValidationError):
            app.recommend_journey(state)

    def test_journey_tamper(self):
        state = app.recommend_journey(app.guided_setup(self.data))
        state["journey"]["steps"][0]["action"] = "request_human_release"
        with self.assertRaises(app.ValidationError):
            app.prioritize_sentiment(state)

    def test_cross_stage_traceability(self):
        state = app.run_pipeline(self.data)
        self.assertEqual(state["guided"]["quality_decision_ids"], state["sentiment"]["quality_decision_ids"])
        self.assertEqual(state["journey"]["based_on_progress"], state["guided"]["progress"])
        inspection = next(i for i in state["sentiment"]["issues"] if i["source"] == "inspection")
        self.assertEqual(inspection["decision_id"], self.data["inspections"][0]["decision_id"])
        self.assertEqual(inspection["work_order_id"], self.data["work_order"]["id"])

    def test_sentiment_transparent(self):
        value = app.score_sentiment("Good, stable; failed.")
        self.assertAlmostEqual(value["score"], 1 / 3)
        self.assertEqual(value["positive_matches"], ["good", "stable"])
        self.assertEqual(app.score_sentiment("routine check")["score"], 0)
        self.assertEqual(app.score_sentiment("BAD fault")["score"], -1)

    def test_safety_overrides_positive_low_severity(self):
        self.data["maintenance_logs"][0].update(text="good excellent", severity="low", safety_critical=True)
        issue = app.run_pipeline(self.data)["sentiment"]["issues"][0]
        self.assertEqual(issue["severity"], "critical")
        self.assertEqual(issue["route"], "human_safety_review")
        self.assertEqual(issue["sentiment"]["label"], "positive")

    def test_severity_dominates_sentiment(self):
        self.data["maintenance_logs"][0].update(text="bad fault failed", severity="low", safety_critical=False)
        issues = app.run_pipeline(self.data)["sentiment"]["issues"]
        self.assertEqual(issues[0]["source"], "inspection")

    def test_safety_sensor_escalates(self):
        lines = self.data["telemetry_csv"].splitlines()
        row = lines[1].split(",")
        row[4] = "110"
        lines[1] = ",".join(row)
        self.data["telemetry_csv"] = "\n".join(lines)
        issue = app.run_pipeline(self.data)["sentiment"]["issues"][0]
        self.assertEqual(issue["source"], "telemetry")
        self.assertTrue(issue["human_required"])

    def test_unit_validation(self):
        self.data["telemetry_csv"] = self.data["telemetry_csv"].replace("degC", "degF")
        self.reject()

    def test_bad_tolerance(self):
        self.data["sensor"]["tolerance_min"] = 80
        self.data["sensor"]["tolerance_max"] = 40
        self.reject()

    def test_inspection_units(self):
        self.data["inspections"][0]["unit"] = "inch"
        self.reject()

    def test_decision_requires_evidence(self):
        self.data["inspections"][0]["rationale"] = ""
        self.reject()

    def test_decision_consistency(self):
        self.data["inspections"][0]["decision"] = "pass"
        self.reject()

    def test_work_order_link(self):
        self.data["inspections"][0]["work_order_id"] = "OTHER"
        self.reject()

    def test_nan_reading(self):
        lines = self.data["telemetry_csv"].splitlines()
        row = lines[1].split(",")
        row[4] = "NaN"
        lines[1] = ",".join(row)
        self.data["telemetry_csv"] = "\n".join(lines)
        self.reject()

    def test_time_order(self):
        lines = self.data["telemetry_csv"].splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        self.data["telemetry_csv"] = "\n".join(lines)
        self.reject()

    def test_empty_telemetry(self):
        self.data["telemetry_csv"] = self.data["telemetry_csv"].splitlines()[0]
        self.reject()

    def test_no_logs_and_passing_quality(self):
        self.data["maintenance_logs"] = []
        self.data["inspections"][0].update(measured=20, decision="pass")
        state = app.run_pipeline(self.data)
        self.assertEqual(state["sentiment"]["issues"], [])
        self.assertFalse(state["sentiment"]["human_review_required"])

    def test_tolerance_inclusive(self):
        self.data["inspections"][0].update(measured=20.1, decision="pass")
        self.assertEqual(app.run_pipeline(self.data)["status"], "ok")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "does_not_exist.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_no_arguments(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_validation(self):
        for value in ('{', '{"x":1,"x":2}', '{"schema_version":1}', '[]'):
            with self.subTest(value=value), patch("builtins.open", mock_open(read_data=value)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_synthetic_required(self):
        self.data["synthetic"] = False
        self.reject()


if __name__ == "__main__":
    unittest.main()
