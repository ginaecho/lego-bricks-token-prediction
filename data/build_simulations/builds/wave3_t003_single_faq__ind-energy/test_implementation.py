import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_grounded_answer_and_entities(self):
        result = app.answer_request(self.data)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], result["citations"][0]["excerpt"])
        self.assertEqual(len(result["entities"]["meter_readings"]), 3)
        self.assertEqual(result["entities"]["meter_readings"][0]["consumption_kwh"], 2.4)
        self.assertEqual(result["entities"]["telemetry"][0]["emissions"]["unit"], "gCO2e/kWh")

    def test_abstain_unknown_question(self):
        self.data["question"] = "What is tomorrow's electricity tariff?"
        result = app.answer_request(self.data)
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])

    def test_empty_knowledge_and_stopword_question(self):
        for question in ("How do I?", "outage"):
            self.data["question"] = question
            self.data["knowledge_base"] = []
            self.assertEqual(app.answer_request(self.data)["status"], "abstained")

    def test_protect_critical_identifiers_and_preserve_input(self):
        before = copy.deepcopy(self.data)
        result = app.answer_request(self.data)
        self.assertNotIn("SYN-SUB-WINTER", json.dumps(result))
        self.assertIn("[PROTECTED-ASSET]", result["answer"])
        self.assertEqual(self.data, before)

    def test_all_declared_safety_priorities(self):
        for hazard, priority in app.SAFETY_RULES.items():
            self.data["outage_reports"][0].update(hazard=hazard, priority=priority)
            app.answer_request(self.data)
            self.data["outage_reports"][0]["priority"] = "incorrect"
            with self.assertRaises(app.ValidationError):
                app.answer_request(self.data)

    def test_cannot_weaken_safety_rules(self):
        self.data["safety_rules"]["life_safety"] = "normal"
        with self.assertRaises(app.ValidationError):
            app.answer_request(self.data)

    def test_emissions_units_source_and_finiteness(self):
        for field, value in (("unit", "tons"), ("source", ""), ("value", float("nan")),
                             ("value", -1), ("value", True)):
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                data["telemetry"][0]["emissions"][field] = value
                with self.assertRaises(app.ValidationError):
                    app.answer_request(data)

    def test_csv_negative_duplicate_time_and_header(self):
        for old, new in ((",2.4", ",-1"), ("18:30:00Z,SYN-METER", "18:00:00Z,SYN-METER"),
                         ("consumption_kwh", "energy"), (",2.4", ",NaN")):
            data = copy.deepcopy(self.data)
            data["smart_meter_interval_csv"] = data["smart_meter_interval_csv"].replace(old, new)
            with self.assertRaises(app.ValidationError):
                app.answer_request(data)

    def test_unknown_topology_reference(self):
        self.data["grid_assets"][0]["connected_to"] = ["SYN-NOT-EXIST"]
        with self.assertRaises(app.ValidationError):
            app.answer_request(self.data)

    def test_schema_and_synthetic_required(self):
        for change in ({"synthetic": False}, {"schema_version": "2.0"}, {"unexpected": 1}):
            data = {**self.data, **change}
            with self.assertRaises(app.ValidationError):
                app.answer_request(data)

    def test_injected_callable_grounded_and_redacted(self):
        def injected(context):
            self.assertNotIn("SYN-SUB-WINTER", json.dumps(context))
            article = context["evidence"][0]
            return {"answer": article["text"], "evidence_ids": [article["id"]]}
        self.assertEqual(app.answer_request(self.data, injected)["status"], "answered")

    def test_injected_hallucination_rejected(self):
        def injected(context):
            return {"answer": "Power will return at noon.",
                    "evidence_ids": [context["evidence"][0]["id"]]}
        with self.assertRaises(app.ValidationError):
            app.answer_request(self.data, injected)

    def test_callable_mutation_cannot_inject_evidence(self):
        def injected(context):
            article = context["evidence"][0]
            article["text"] = "Invented restoration time."
            return {"answer": article["text"], "evidence_ids": [article["id"]]}
        with self.assertRaises(app.ValidationError):
            app.answer_request(self.data, injected)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "answered")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "missing.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                     *args], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_json_schema_and_duplicate_keys(self):
        import io
        for payload in ("{", "[]", '{"x":1,"x":2}', '{"synthetic":false}'):
            with patch("builtins.open", mock_open(read_data=payload)), \
                    patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(app.main(["input.json"]), 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
