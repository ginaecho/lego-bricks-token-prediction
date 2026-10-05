import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.data = app.synthetic_fixture()

    def change_reading(self, **changes):
        rows = list(csv.DictReader(io.StringIO(self.data["telemetry_csv"])))
        rows[0].update(changes)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=app.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        self.data["telemetry_csv"] = stream.getvalue()

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.onboard(self.data)

    def test_normal_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        output = app.onboard(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["onboarding"]["next_step"], "safety")
        self.assertEqual(output["onboarding"]["steps"][1]["state"], "locked")
        self.assertEqual(output["onboarding"]["steps"][0]["guidance"], "worked_example")
        self.assertEqual(output["onboarding"]["blockers"], [])
        self.assertEqual(before, self.data)
        self.assertEqual(app.validate(output), output["telemetry_readings"])

    def test_deterministic_randomized_fixture(self):
        self.assertEqual(app.synthetic_fixture(67), self.data)
        self.assertNotEqual(app.synthetic_fixture(68)["telemetry_csv"], self.data["telemetry_csv"])
        self.assertEqual(app.onboard(self.data), app.onboard(self.data))

    def test_experience_and_preference(self):
        detailed = app.onboard(self.data)["onboarding"]["steps"][0]
        self.data["profile"].update(experience="expert", preference="concise")
        concise = app.onboard(self.data)["onboarding"]["steps"][0]
        self.assertEqual(concise["guidance"], "verification_checklist")
        self.assertLess(len(concise["explanation"]), len(detailed["explanation"]))
        self.assertIn("Example:", detailed["learning_task"])
        self.assertIn("Verify", concise["learning_task"])
        self.assertEqual(concise["state"], "available")

    def test_prerequisites_cannot_be_skipped(self):
        self.data["profile"]["completed_steps"] = ["inspection"]
        self.invalid()
        self.data["profile"]["completed_steps"] = ["safety", "safety"]
        self.invalid()

    def test_completed_prerequisites(self):
        self.data["profile"]["completed_steps"] = list(app.STEPS[:-1])
        output = app.onboard(self.data)["onboarding"]
        self.assertEqual(output["next_step"], "release")
        self.assertEqual(output["release_readiness"], "ready_for_human_review")
        self.assertFalse(output["operational_authorization"])
        self.data["profile"]["completed_steps"] = list(app.STEPS)
        self.assertTrue(app.onboard(self.data)["onboarding"]["complete"])

    def test_safety_threshold_cannot_be_relaxed_by_tolerance(self):
        self.change_reading(value="85", tolerance_max="100")
        self.data["profile"]["completed_steps"] = list(app.STEPS)
        output = app.onboard(self.data)
        self.assertTrue(output["telemetry_readings"][0]["within_tolerance"])
        self.assertEqual(output["safety_alerts"][0]["route"], "human_safety_supervisor")
        self.assertFalse(output["safety_alerts"][0]["automatic_clearance"])
        self.assertEqual(output["onboarding"]["steps"][-1]["state"], "blocked")
        self.assertFalse(output["onboarding"]["complete"])

    def test_vibration_safety(self):
        self.change_reading(metric="vibration", value="12", unit="mm/s",
                            tolerance_min="0", tolerance_max="15")
        self.assertEqual(len(app.onboard(self.data)["safety_alerts"]), 1)

    def test_noncritical_tolerance_breach_blocks(self):
        self.change_reading(value="76")
        output = app.onboard(self.data)
        self.assertEqual(output["safety_alerts"], [])
        self.assertIn("telemetry_outside_tolerance", output["onboarding"]["blockers"])

    def test_units_and_bounds(self):
        for changes in ({"unit": "F"}, {"value": "nan"}, {"value": "inf"},
                        {"value": "151"}, {"tolerance_min": "80", "tolerance_max": "70"}):
            with self.subTest(changes=changes):
                self.data = app.synthetic_fixture()
                self.change_reading(**changes)
                self.invalid()

    def test_traceability_required(self):
        for field in ("inspector", "rationale", "work_order_id", "evidence_reading_id", "timestamp"):
            with self.subTest(field=field):
                self.data = app.synthetic_fixture()
                del self.data["quality_inspections"][0][field]
                self.invalid()

    def test_quality_decision_trace(self):
        decision = app.onboard(self.data)["quality_decisions"][0]
        self.assertEqual(decision["work_order_id"], self.data["work_order"]["work_order_id"])
        self.assertEqual(decision["rule"], "inclusive_dimension_tolerance_v1")
        self.assertEqual(decision["measurement"]["unit"], "mm")
        self.assertEqual(decision["evidence_reading_id"], "SYN-R-005")

    def test_failed_and_contradictory_inspection(self):
        record = self.data["quality_inspections"][0]
        record["value"] = 20.06
        self.invalid()
        record["decision"] = "fail"
        self.assertIn("failed_quality_inspection_requires_human",
                      app.onboard(self.data)["onboarding"]["blockers"])

    def test_inclusive_inspection_boundaries(self):
        for value in (19.95, 20.05):
            self.data["quality_inspections"][0]["value"] = value
            self.assertEqual(app.onboard(self.data)["quality_decisions"][0]["decision"], "pass")

    def test_missing_inspection_is_valid_but_blocks_release(self):
        self.data["quality_inspections"] = []
        self.assertIn("quality_inspection_required", app.onboard(self.data)["onboarding"]["blockers"])

    def test_csv_schema_and_empty_rows(self):
        for value in ("incorrect,header\nx,y\n", ",".join(app.CSV_FIELDS) + "\n",
                      self.data["telemetry_csv"] + "bad,row\n"):
            with self.subTest(value=value):
                self.data["telemetry_csv"] = value
                self.invalid()

    def test_time_and_entity_relationships(self):
        for changes in ({"timestamp": "2026-09-24T06:00:00Z"},
                        {"timestamp": "2026-09-24T08:00:00"},
                        {"asset_id": "SYN-OTHER"}, {"reading_id": "SYN-R-005"}):
            self.data = app.synthetic_fixture()
            self.change_reading(**changes)
            self.invalid()
        self.data = app.synthetic_fixture()
        self.data["quality_inspections"][0]["timestamp"] = "2026-09-24T08:04:00Z"
        self.invalid()

    def test_malformed_shared_schema(self):
        for value in (None, [], 1, {"schema_version": "2.0"}):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.onboard(value)
        self.data["work_order"]["quantity"] = True
        self.invalid()
        self.data = app.synthetic_fixture()
        self.data["quality_inspections"][0]["value"] = 10 ** 1000
        self.invalid()

    def test_maintenance_and_synthetic_labels(self):
        self.data["maintenance_logs"][0]["synthetic"] = False
        self.invalid()
        self.data = app.synthetic_fixture()
        self.data["synthetic"] = False
        self.invalid()

    def test_cli_success(self):
        directory = Path(__file__).resolve().parent
        result = subprocess.run([sys.executable, "-B", str(directory / "implementation.py"),
                                 str(directory / "example_input.json")],
                                capture_output=True, text=True, cwd=directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_file_errors_and_usage(self):
        directory = Path(__file__).resolve().parent
        for args in ([], [str(directory / "nonexistent.json")]):
            result = subprocess.run([sys.executable, "-B", str(directory / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=directory)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_schema_without_extra_files(self):
        for raw in ('{', '{"schema_version": "2.0"}', '{"x": NaN}', '{"x": 1, "x": 2}'):
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
