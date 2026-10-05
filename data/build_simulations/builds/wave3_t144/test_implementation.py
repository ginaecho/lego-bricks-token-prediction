import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


class JourneyTests(unittest.TestCase):
    def setUp(self):
        self.data = app.synthetic_fixture()

    def actions(self):
        return [s["action"] for s in app.recommend(self.data)["journey"]["steps"]]

    def test_normal_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.recommend(self.data), app.recommend(self.data))
        self.assertEqual(self.data, before)
        self.assertEqual(self.actions(), ["review_quality_traceability", "request_human_release_review"])

    def test_seeded_plausible_telemetry(self):
        self.assertEqual(app.synthetic_fixture(2), app.synthetic_fixture(2))
        self.assertNotEqual(app.synthetic_fixture(2)["telemetry_csv"], app.synthetic_fixture(3)["telemetry_csv"])
        self.assertEqual(len(app.validate(self.data)), 6)

    def test_personalization(self):
        self.data["profile"]["priority"] = "maintenance"
        self.assertEqual(self.actions()[0], "review_maintenance_history")

    def test_critical_alert_overrides_preference(self):
        self.data["telemetry_csv"] += "SYN-MILL-044,2026-09-24T08:04:00Z,temperature,85,degC\n"
        result = app.recommend(self.data)
        self.assertTrue(result["human_escalation"]["required"])
        self.assertEqual(result["human_escalation"]["delivery_status"], "not_sent")
        self.assertEqual(self.actions(), ["escalate_to_human", "recommend_safety_hold"])

    def test_warning_priority(self):
        self.data["telemetry_csv"] += "SYN-MILL-044,2026-09-24T08:04:00Z,vibration,7,mm/s\n"
        self.assertEqual(self.actions()[0], "review_maintenance_history")

    def test_failed_quality(self):
        self.data["inspections"][0].update(value=20.1, decision="fail")
        self.assertEqual(self.actions(), ["investigate_nonconformance", "schedule_reinspection"])

    def test_missing_inspection(self):
        self.data["inspections"] = []
        self.assertEqual(self.actions(), ["schedule_inspection", "assign_quality_reviewer"])

    def test_tolerance_boundaries(self):
        for value in (19.95, 20.05):
            self.data["inspections"][0]["value"] = value
            self.assertEqual(self.actions()[0], "review_quality_traceability")

    def test_invalid_tolerances_and_units(self):
        for change in ({"lower": 21}, {"unit": "inch"}, {"lower": True}, {"upper": float("nan")}):
            candidate = copy.deepcopy(self.data)
            candidate["work_order"]["tolerance"].update(change)
            with self.assertRaises(app.ValidationError):
                app.recommend(candidate)

    def test_quality_traceability(self):
        for change in ({"inspector_id": ""}, {"rationale": ""}, {"work_order_id": "other"},
                       {"decision": "fail"}, {"unit": "cm"}, {"timestamp": "2026-09-24T08:00:00"}):
            candidate = copy.deepcopy(self.data)
            candidate["inspections"][0].update(change)
            with self.assertRaises(app.ValidationError):
                app.recommend(candidate)
        evidence = app.recommend(self.data)["evidence"]
        self.assertEqual(evidence["quality_decisions"], self.data["inspections"])
        self.assertEqual(evidence["selected_inspection_id"], "SYN-QI-001")

    def test_invalid_sensor_csv(self):
        for row in ("SYN-MILL-044,2026-09-24T08:04:00Z,temperature,nan,degC",
                    "SYN-MILL-044,2026-09-24T08:04:00Z,temperature,999,degC",
                    "SYN-MILL-044,2026-09-24T08:04:00Z,temperature,50,F",
                    "other,2026-09-24T08:04:00Z,temperature,50,degC",
                    "SYN-MILL-044,2026-09-24T08:00:00Z,temperature,50,degC",
                    "SYN-MILL-044,missing", 'SYN-MILL-044,"unterminated'):
            candidate = copy.deepcopy(self.data)
            candidate["telemetry_csv"] += row + "\n"
            with self.assertRaises(app.ValidationError):
                app.recommend(candidate)

    def test_empty_and_incomplete_telemetry(self):
        self.data["telemetry_csv"] = "asset_id,timestamp,metric,value,unit\n"
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)
        self.data["telemetry_csv"] += "SYN-MILL-044,2026-09-24T08:00:00Z,temperature,50,degC\n"
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_journey_dependency_validation(self):
        with self.assertRaises(app.ValidationError):
            app.validate_journey(["request_human_release_review", "review_quality_traceability"],
                                 {"quality_passed", "no_safety_alert"})
        result = app.recommend(self.data)["journey"]
        self.assertTrue(result["validated"])
        self.assertEqual(result["steps"][1]["depends_on_step"], 1)
        self.assertTrue(set(result["steps"][1]["requires"]) <= set(result["steps"][0]["planned_effects"]))

    def test_invalid_schema(self):
        for candidate in ([], {}, dict(self.data, extra=1), dict(self.data, synthetic_fixture=False)):
            with self.assertRaises(app.ValidationError):
                app.recommend(candidate)

    def test_oversized_numeric_input(self):
        self.data["work_order"]["tolerance"]["upper"] = 10 ** 400
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_malformed_csv_header_cli(self):
        self.data["telemetry_csv"] = '"unterminated'
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data=json.dumps(self.data))), contextlib.redirect_stdout(output):
            self.assertEqual(app.main(["fixture.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_success(self):
        root = Path(__file__).parent
        proc = subprocess.run([sys.executable, "-B", str(root / "implementation.py"),
                               str(root / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_usage(self):
        root = Path(__file__).parent
        for args in ([], [str(root / "nonexistent.json")]):
            proc = subprocess.run([sys.executable, "-B", str(root / "implementation.py"), *args],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for raw in ("{", '{"x":1,"x":2}', "[]", json.dumps(dict(self.data, synthetic_fixture=False))):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(output):
                self.assertEqual(app.main(["fixture.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
