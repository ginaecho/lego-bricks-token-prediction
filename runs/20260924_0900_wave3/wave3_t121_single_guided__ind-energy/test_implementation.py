import contextlib
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


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_complete_and_deterministic(self):
        original = copy.deepcopy(self.data)
        result = app.run(self.data)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["progress"]["percent"], 100)
        self.assertEqual(len(result["entities"]["meter_readings"]), 2)
        self.assertEqual(result, app.run(self.data))
        self.assertEqual(original, self.data)

    def test_partial_and_resume(self):
        self.data["actions"] = list(app.STEPS[:2])
        result = app.run(self.data)
        self.assertEqual(result["progress"]["next_step"], "import_meter_readings")
        self.data["completed_steps"] = result["progress"]["completed_steps"]
        self.data["actions"] = list(app.STEPS[2:])
        self.assertEqual(app.run(self.data)["status"], "complete")

    def test_skip_prerequisite(self):
        self.data["actions"] = ["register_grid"]
        self.invalid()

    def test_duplicate_action(self):
        self.data["actions"] = ["protect_identifiers", "protect_identifiers"]
        self.invalid()

    def test_forged_progress(self):
        self.data["completed_steps"] = ["register_grid"]
        self.invalid()

    def test_unacknowledged_protection(self):
        self.data["protection"]["acknowledged"] = False
        self.data["actions"] = []
        result = app.run(self.data)
        self.assertFalse(result["progress"]["next_step_ready"])
        self.data["actions"] = ["protect_identifiers"]
        self.invalid()

    def test_critical_identifier_not_disclosed(self):
        output = json.dumps(app.run(self.data))
        for asset in self.data["grid_assets"]:
            self.assertNotIn(asset["asset_id"], output)
        self.data["protection"]["critical_identifier_policy"] = "plaintext"
        self.invalid()

    def test_outage_safety_precedence(self):
        self.data["outage_reports"][0]["priority"] = "high"
        self.invalid()
        self.data["outage_reports"][0]["danger_to_life"] = False
        self.assertEqual(app.run(self.data)["status"], "complete")
        self.data["outage_reports"][0]["essential_service"] = False
        self.data["outage_reports"][0]["priority"] = "normal"
        self.assertEqual(app.run(self.data)["status"], "complete")

    def test_declared_safety_cannot_weaken_policy(self):
        self.data["safety_rules"]["danger_to_life"] = "normal"
        self.invalid()

    def test_emissions_units_source_and_finite_value(self):
        original = copy.deepcopy(self.data)
        for field, value in (("unit", "kg"), ("source", ""), ("value", float("nan")),
                             ("value", -1), ("value", True)):
            with self.subTest(field=field, value=value):
                self.data = copy.deepcopy(original)
                self.data["telemetry"][0]["emissions"][field] = value
                self.invalid()

    def test_csv_invalid_header_and_numeric(self):
        original = self.data["meter_interval_csv"]
        for csv in (original.replace("energy_kwh", "energy"),
                    original.replace("2.4", "nan"), original.replace("2.4", "-1"),
                    original.replace("18:00:00Z", "18:01:00Z"),
                    original + original.splitlines()[1] + "\n"):
            with self.subTest(csv=csv):
                self.data["meter_interval_csv"] = csv
                self.invalid()

    def test_topology_and_unknown_reference(self):
        self.data["grid_assets"][1]["parent_id"] = "SUB-SYN-999"
        self.invalid()
        self.data["grid_assets"][1]["parent_id"] = "DEV-SYN-101"
        self.invalid()

    def test_empty_initial_state(self):
        self.data.update(grid_assets=[], telemetry=[], outage_reports=[], actions=[],
                         meter_interval_csv=",".join(app.CSV_FIELDS) + "\n")
        result = app.run(self.data)
        self.assertEqual(result["progress"]["completed_count"], 0)
        self.assertTrue(result["progress"]["next_step_ready"])
        self.data["actions"] = ["protect_identifiers", "register_grid"]
        self.invalid()

    def test_empty_outages_valid_review(self):
        self.data["outage_reports"] = []
        self.assertEqual(app.run(self.data)["status"], "complete")

    def test_unknown_fields_and_non_synthetic(self):
        self.data["unexpected"] = "x"
        self.invalid()
        del self.data["unexpected"]
        self.data["synthetic"] = False
        self.invalid()

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "complete")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["nonexistent-input.json"]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")]
                                    + args, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_without_creating_files(self):
        for raw in (b"{", b'{"a":1,"a":2}', b'{"x":NaN}', b"\xff",
                    b"[]", b" " * (app.MAX_BYTES + 1)):
            with self.subTest(raw=raw[:30]):
                stream = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                    with contextlib.redirect_stdout(stream):
                        code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_validation_error_protects_identifiers(self):
        self.data["outage_reports"][0]["priority"] = "normal"
        stream = io.StringIO()
        with patch.object(Path, "open",
                          return_value=io.BytesIO(json.dumps(self.data).encode())):
            with contextlib.redirect_stdout(stream):
                self.assertEqual(app.main(["fixture.json"]), 2)
        self.assertNotIn("SUB-SYN-001", stream.getvalue())
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
