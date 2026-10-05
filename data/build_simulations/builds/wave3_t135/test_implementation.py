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


ROOT = Path(__file__).resolve().parent


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_deterministic(self):
        output = app.run(self.data)
        self.assertEqual(output, app.run(copy.deepcopy(self.data)))
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["onboarding"]["next_module"], "seasonal_basics")
        self.assertEqual(len(output["entities"]["meter_readings"]), 2)

    def test_experience_adaptation(self):
        self.data["profile"]["experience"] = "expert"
        output = app.run(self.data)["onboarding"]
        self.assertNotIn("seasonal_basics", [s["module"] for s in output["steps"]])
        self.assertEqual(output["next_module"], "meter_readings")

    def test_preference_explanations(self):
        detailed = app.run(self.data)["onboarding"]["steps"]
        self.data["profile"]["preference"] = "concise"
        concise = app.run(self.data)["onboarding"]["steps"]
        self.assertTrue(all(len(a["explanation"]) > len(b["explanation"])
                            for a, b in zip(detailed, concise)))
        self.assertEqual([s["status"] for s in detailed], [s["status"] for s in concise])

    def test_prerequisite_gating(self):
        self.data["profile"]["completed"] = []
        steps = app.run(self.data)["onboarding"]["steps"]
        self.assertEqual([s["module"] for s in steps if s["status"] == "available"],
                         ["protection"])
        self.data["profile"]["completed"] = ["outage_triage"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_completed_onboarding(self):
        self.data["profile"]["completed"] = [r[0] for r in app.modules(self.data["profile"])]
        result = app.run(self.data)["onboarding"]
        self.assertTrue(result["ready"])
        self.assertIsNone(result["next_module"])

    def test_protected_identifiers_and_provenance(self):
        critical_id = self.data["grid_assets"]["assets"][0]["asset_id"]
        self.data["grid_assets"]["assets"][0]["emissions"]["source"] = "Synthetic " + critical_id
        output = app.run(self.data)
        rendered = json.dumps(output)
        for asset in self.data["grid_assets"]["assets"]:
            self.assertNotIn(asset["asset_id"], rendered)
        self.assertNotIn("SYN-METER-001", rendered)
        self.assertNotIn("SYN-OUTAGE-001", rendered)
        assets = output["entities"]["grid_assets"]
        self.assertEqual(assets[0]["neighbors"][0], assets[1]["asset_ref"])
        self.assertIn("[protected identifier]", assets[0]["emissions"]["source"])

    def test_safety_priorities_and_boundary(self):
        for life, service, customers, expected in [
                (True, True, 0, "P1"), (False, True, 0, "P2"),
                (False, False, 100, "P3"), (False, False, 99, "P4")]:
            with self.subTest(priority=expected):
                report = self.data["outage_reports"][0]
                report.update(life_threat=life, critical_service=service,
                              customers_affected=customers, priority=expected)
                self.assertEqual(app.run(self.data)["entities"]["outage_reports"][0]["priority"],
                                 expected)
                report["priority"] = "P1" if expected != "P1" else "P4"
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_emissions_require_units_source_and_finite_value(self):
        for field, value in [("unit", "tons"), ("source", ""), ("value", float("nan")),
                             ("value", -1), ("value", True)]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                data["grid_assets"]["assets"][0]["emissions"][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_topology_constraints(self):
        for neighbors in [["UNKNOWN"], ["SYN-SUBSTATION-001"], []]:
            with self.subTest(neighbors=neighbors):
                data = copy.deepcopy(self.data)
                data["grid_assets"]["assets"][0]["neighbors"] = neighbors
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_csv_constraints(self):
        original = self.data["meter_readings"]["csv"]
        for invalid in [original.replace("1.4", "NaN"),
                        original.replace("08:15", "08:30"),
                        original.replace("winter", "summer"),
                        original.replace("SYN-DEVICE-001", "UNKNOWN"),
                        '"unterminated header',
                        original.splitlines()[0] + "\n",
                        original.replace("1.4,winter", "1.4,winter,extra")]:
            with self.subTest(csv=invalid):
                self.data["meter_readings"]["csv"] = invalid
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_empty_outages_allowed(self):
        self.data["outage_reports"] = []
        self.assertEqual(app.run(self.data)["entities"]["outage_reports"], [])

    def test_identifiers_explicitly_synthetic(self):
        self.data["grid_assets"]["assets"][0]["asset_id"] = "REAL-CRITICAL"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_schema(self):
        for data in [None, [], {}, dict(self.data, synthetic=False),
                     dict(self.data, schema_version=True), dict(self.data, extra=1)]:
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in [[], [str(ROOT / "does-not-exist.json")]]:
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_sensitive_error(self):
        invalid = copy.deepcopy(self.data)
        invalid["grid_assets"]["assets"][0]["neighbors"] = ["SECRET-CRITICAL-ASSET"]
        for contents in ["not json", '{"synthetic":true,"synthetic":false}',
                         json.dumps(invalid), '{"schema_version":null}', "[]"]:
            with self.subTest(contents=contents[:25]):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=contents)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")
                self.assertNotIn("SECRET-CRITICAL-ASSET", output.getvalue())


if __name__ == "__main__":
    unittest.main()
