import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_synthetic_entities(self):
        output = impl.triage(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["tickets"][0]["priority"], "P1")
        self.assertEqual(output["tickets"][1]["category"], "meter")
        self.assertEqual(output["meter_readings"][0]["consumption"],
                         {"value": 3.9, "unit": "kWh"})
        self.assertEqual(output["emissions"][0]["unit"], "kgCO2e")
        self.assertEqual(output["grid_assets"][0]["connected_to"], ["asset-002"])

    def test_deterministic_without_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(impl.triage(self.data), impl.triage(self.data))
        self.assertEqual(self.data, original)

    def test_all_declared_safety_rules(self):
        for flag, expected in impl.SAFETY_FLOORS.items():
            with self.subTest(flag=flag):
                safety = self.data["tickets"][0]["outage_report"]["safety"]
                for key in safety:
                    safety[key] = key == flag
                result = impl.triage(self.data)["tickets"][0]
                self.assertEqual(result["priority"], expected)
                self.assertEqual(result["safety_rules_applied"], [flag])

    def test_safety_cannot_be_weakened(self):
        self.data["config"]["safety_rules"]["downed_line"] = "P4"
        with self.assertRaises(impl.ValidationError):
            impl.triage(self.data)

    def test_stricter_declared_safety(self):
        self.data["config"]["safety_rules"]["critical_service"] = "P1"
        safety = self.data["tickets"][0]["outage_report"]["safety"]
        safety.update(downed_line=False, critical_service=True)
        self.assertEqual(impl.triage(self.data)["tickets"][0]["priority"], "P1")

    def test_identifier_protection_in_configurable_strings(self):
        critical = self.data["assets"][0]["asset_id"]
        self.data["config"]["routes"]["outage"]["accountable_owner"] = critical.lower()
        self.data["emissions"][0]["source"] = "Synthetic source " + critical
        encoded = json.dumps(impl.triage(self.data))
        self.assertNotIn(critical.lower(), encoded.lower())
        self.assertIn("[protected-asset]", encoded)

    def test_emissions_require_units_source_and_finite_value(self):
        for field, value in (("unit", "kg"), ("source", ""), ("value", float("nan")),
                             ("value", True), ("value", -1)):
            with self.subTest(field=field, value=value):
                payload = copy.deepcopy(self.data)
                payload["emissions"][0][field] = value
                with self.assertRaises(impl.ValidationError):
                    impl.triage(payload)
        del self.data["emissions"][0]["source"]
        with self.assertRaises(impl.ValidationError):
            impl.triage(self.data)

    def test_unknown_asset_and_topology(self):
        self.data["assets"][0]["connected_to"] = ["FICT-MISSING"]
        with self.assertRaises(impl.ValidationError):
            impl.triage(self.data)

    def test_csv_invalid_numeric_duplicate_header_and_timezone(self):
        original = self.data["meter_readings_csv"]
        invalid = [
            original.replace(",1.8", ",nan"),
            original.replace(",1.8", ",-2"),
            original.replace(",1.8", ",bad"),
            original.replace("consumption_kwh", "watts"),
            original + original.splitlines()[1] + "\n",
            original.replace("18:00:00Z", "18:00:00"),
            original.replace(",30,", ",20,"),
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.data["meter_readings_csv"] = raw
                with self.assertRaises(impl.ValidationError):
                    impl.triage(self.data)

    def test_missing_and_nonboolean_safety(self):
        safety = self.data["tickets"][0]["outage_report"]["safety"]
        safety["downed_line"] = "false"
        with self.assertRaises(impl.ValidationError):
            impl.triage(self.data)
        del safety["downed_line"]
        with self.assertRaises(impl.ValidationError):
            impl.triage(self.data)

    def test_configurable_keywords_routing_and_priority(self):
        self.data["tickets"][1]["text"] = "Synthetic invoice query"
        self.data["config"]["keywords"]["meter"] = ["invoice"]
        self.data["config"]["category_priority"]["meter"] = "P2"
        self.data["config"]["routes"]["meter"]["accountable_owner"] = "Synthetic invoice lead"
        ticket = impl.triage(self.data)["tickets"][1]
        self.assertEqual(ticket["category"], "meter")
        self.assertEqual(ticket["priority"], "P2")
        self.assertEqual(ticket["routing"]["accountable_owner"], "Synthetic invoice lead")

    def test_grid_telemetry_route(self):
        self.data["tickets"][0]["outage_report"] = None
        self.data["tickets"][0]["text"] = "Synthetic voltage investigation"
        ticket = impl.triage(self.data)["tickets"][0]
        self.assertEqual(ticket["category"], "grid")
        self.assertEqual(ticket["reason"], "telemetry_alert")

    def test_telemetry_cannot_mask_unassessed_outage(self):
        self.data["tickets"][0]["outage_report"] = None
        ticket = impl.triage(self.data)["tickets"][0]
        self.assertEqual(ticket["category"], "outage")
        self.assertEqual(ticket["priority"], "P1")

    def test_unknown_safety_outage_is_urgent(self):
        self.data["tickets"][1]["text"] = "Synthetic outage"
        ticket = impl.triage(self.data)["tickets"][1]
        self.assertEqual(ticket["priority"], "P1")
        self.assertEqual(ticket["safety_rules_applied"], ["unassessed_outage"])

    def test_empty_tickets_and_meter_intervals(self):
        self.data["tickets"] = []
        self.data["meter_readings_csv"] = self.data["meter_readings_csv"].splitlines()[0]
        output = impl.triage(self.data)
        self.assertEqual(output["tickets"], [])
        self.assertEqual(output["meter_readings"][0]["consumption"]["value"], 0)

    def test_general_fallback_and_word_boundaries(self):
        self.data["tickets"][1]["text"] = "Synthetic parameter discussion"
        self.assertEqual(impl.triage(self.data)["tickets"][1]["category"], "general")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_usage_and_file_error(self):
        for args in ([], [str(ROOT / "missing.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                     *args], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_validation_without_extra_files(self):
        for raw in ('{"assets":', '{"a":1,"a":2}', '[]',
                    json.dumps({**self.data, "synthetic": False})):
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    code = impl.main([str(ROOT / "example_input.json")])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
