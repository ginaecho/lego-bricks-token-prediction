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


ROOT = Path(__file__).parent


class InsightsTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self, data=None):
        with self.assertRaises((app.ValidationError, TypeError, ValueError)):
            app.run(self.data if data is None else data)

    def test_normal_deterministic_and_not_mutated(self):
        original = copy.deepcopy(self.data)
        result = app.run(self.data)
        self.assertEqual(result, app.run(self.data))
        self.assertEqual(self.data, original)
        self.assertEqual(len(result["meter_readings"]), 3)
        self.assertEqual(result["meter_readings"][0]["consumption"]["unit"], "kWh")
        self.assertEqual([r["priority"] for r in result["outage_reports"]], [1, 2, 3])

    def test_transparent_sentiment_and_negation(self):
        result = app.sentiment("Not good, great! Never bad; angry.")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["label"], "neutral")
        self.assertEqual([m["weight"] for m in result["matches"]], [-1, 2, 1, -2])
        self.assertEqual(app.sentiment("meter 101")["matches"], [])
        self.assertEqual(app.sentiment("poor")["label"], "negative")

    def test_empty_collections(self):
        for field in ("grid_assets", "outage_reports", "scada_telemetry", "emissions"):
            self.data[field] = []
        self.data["meter_interval_csv"] = "timestamp,device_id,consumption_kwh,season\n"
        self.assertEqual(app.run(self.data)["outage_reports"], [])

    def test_safety_overrides_positive_sentiment_and_customer_count(self):
        safety = self.data["outage_reports"][0]
        safety["description"] = "Great reliable good thanks"
        safety["outage_active"] = False
        result = app.run(self.data)
        self.assertEqual(result["outage_reports"][0]["priority"], 1)
        self.assertEqual(result["outage_reports"][0]["sentiment"]["label"], "positive")
        for flag in ("fire", "medical_dependency"):
            safety["safety_flags"] = [flag]
            self.assertEqual(app.run(self.data)["outage_reports"][0]["priority"], 1)

    def test_safety_rule_tampering(self):
        self.data["safety_rules"]["fire"] = 3
        self.invalid()

    def test_unknown_or_duplicate_safety_flag(self):
        for flags in (["none"], ["fire", "fire"], "fire", [True]):
            with self.subTest(flags=flags):
                self.data["outage_reports"][0]["safety_flags"] = flags
                self.invalid()

    def test_asset_protection_in_all_output(self):
        identifier = self.data["grid_assets"][0]["asset_id"]
        self.data["emissions"][0]["source"] = "Synthetic " + identifier
        output = json.dumps(app.run(self.data))
        for asset in self.data["grid_assets"]:
            self.assertNotIn(asset["asset_id"], output)
        for report in self.data["outage_reports"]:
            self.assertNotIn(report["report_id"], output)
            self.assertNotIn(report["description"], output)
        self.assertIn("[protected-asset]", output)

    def test_emissions_require_units_source_and_finite_amount(self):
        original = copy.deepcopy(self.data)
        for field, value in (("unit", "tons"), ("source", ""), ("value", -1),
                             ("value", float("nan")), ("value", True)):
            with self.subTest(field=field, value=value):
                self.data = copy.deepcopy(original)
                self.data["emissions"][0][field] = value
                self.invalid()
        self.data = original
        del self.data["emissions"][0]["source"]
        self.invalid()

    def test_topology_unknown_cycle_and_duplicate(self):
        original = copy.deepcopy(self.data)
        for parent in ("unknown", "SYN-METER-101"):
            self.data = copy.deepcopy(original)
            self.data["grid_assets"][0]["upstream_id"] = parent
            self.invalid()
        self.data = original
        self.data["grid_assets"].append(copy.deepcopy(self.data["grid_assets"][0]))
        self.invalid()

    def test_csv_invalid_intervals_numbers_and_columns(self):
        original = self.data["meter_interval_csv"]
        variants = [original.replace(",1.2,", ",nan,"),
                    original.replace(",1.2,", ",-1,"),
                    original.replace("06:00:00Z", "06:01:00Z"),
                    original.replace("06:00:00Z", "06:00:00"),
                    original.replace("consumption_kwh", "watts"),
                    original + original.splitlines()[1] + "\n",
                    original.replace(",1.2,winter", ",1.2,winter,extra")]
        for value in variants:
            with self.subTest(value=value):
                self.data["meter_interval_csv"] = value
                self.invalid()

    def test_invalid_telemetry_and_references(self):
        self.data["scada_telemetry"][0]["voltage_v"] = float("inf")
        self.invalid()
        self.data["scada_telemetry"][0]["voltage_v"] = 1
        self.data["outage_reports"][0]["asset_id"] = "unknown"
        self.invalid()

    def test_invalid_envelope_and_types(self):
        for payload in (None, [], {}, {"synthetic": False}):
            with self.assertRaises(app.ValidationError):
                app.run(payload)
        self.data["synthetic"] = False
        self.invalid()
        self.data["synthetic"] = True
        self.data["outage_reports"][0]["customers_affected"] = True
        self.invalid()

    def test_output_validation_enforces_safety(self):
        output = app.run(self.data)
        output["outage_reports"][0]["priority"] = 3
        with self.assertRaises(app.ValidationError):
            app.validate(output, "output")

    def test_cli_success_and_file_errors(self):
        for args, expected in ((["example_input.json"], 0), (["missing.json"], 2), ([], 2)):
            proc = subprocess.run([sys.executable, "-B", "implementation.py", *args],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, expected, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout)["status"], "ok" if expected == 0 else "error")
            self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_malformed_json_no_scratch_files(self):
        for raw in ('{', '{"x":NaN}', '{"x":1,"x":2}', '[]', '{"schema_version":true}'):
            with self.subTest(raw=raw):
                capture = io.StringIO()
                with patch("pathlib.Path.open", mock_open(read_data=raw)):
                    with contextlib.redirect_stdout(capture):
                        result = app.main(["synthetic.json"])
                self.assertEqual(result, 2)
                self.assertEqual(json.loads(capture.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
