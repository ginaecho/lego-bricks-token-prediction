import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
import contextlib
import io

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normalization_and_side_by_side(self):
        result = app.run_pipeline(self.data)["comparison"]
        attrs = result["products"][0]["attributes"]
        self.assertEqual(attrs["ram_gb"], 16)
        self.assertEqual(attrs["storage_gb"], 1024)
        self.assertEqual(attrs["battery_hours"], 10)
        self.assertEqual(attrs["weight_kg"], 1.5)
        self.assertEqual(result["side_by_side"][0]["values"], {"atlas": 800, "beacon": 1200})

    def test_ranking_and_weight_change(self):
        self.data["preferences"] = {"weights": {"ram_gb": 1}, "required_features": []}
        result = app.run_pipeline(self.data)["comparison"]
        self.assertEqual(result["ranking"], ["beacon", "atlas"])
        self.assertEqual(result["recommended_product_id"], "beacon")

    def test_comparison_propagates_to_routing(self):
        result = app.run_pipeline(self.data)
        ticket = result["triage"][0]
        self.assertEqual(ticket["category"], "product-fit")
        self.assertIn("comparison:over_budget", ticket["reasons"])
        self.assertEqual(ticket["comparison_context"]["missing_features"], ["usb-c"])
        self.assertEqual(ticket["comparison_context"]["rank"], 2)
        self.assertEqual(ticket["owner"], "synthetic-product-lead")

    def test_keyword_category_and_priority(self):
        ticket = app.run_pipeline(self.data)["triage"][1]
        self.assertEqual((ticket["category"], ticket["priority"]), ("billing", "urgent"))
        self.assertEqual(ticket["team"], "billing-support")

    def test_fallback_uses_recommendation(self):
        ticket = app.run_pipeline(self.data)["triage"][2]
        self.assertEqual(ticket["category"], "general")
        self.assertEqual(ticket["comparison_context"]["product_id"], "atlas")

    def test_no_eligible_product(self):
        self.data["preferences"]["max_price_usd"] = 0
        result = app.run_pipeline(self.data)
        self.assertIsNone(result["comparison"]["recommended_product_id"])
        self.assertIsNone(result["triage"][2]["comparison_context"]["product_id"])

    def test_missing_attribute_penalty(self):
        del self.data["products"][0]["attributes"]["ram_gb"]
        p = app.run_pipeline(self.data)["comparison"]["products"][0]
        self.assertEqual(p["score_contributions"]["ram_gb"], 0)
        self.assertIn("missing_attributes", p["signals"])

    def test_single_product_and_empty_tickets(self):
        self.data["products"] = self.data["products"][:1]
        self.data["tickets"] = []
        result = app.run_pipeline(self.data)
        self.assertAlmostEqual(result["comparison"]["products"][0]["score"], 1)
        self.assertEqual(result["triage"], [])

    def test_ties_use_id_not_input_order(self):
        p = copy.deepcopy(self.data["products"][0])
        p["id"] = "aardvark"
        self.data["products"].append(p)
        self.assertEqual(app.run_pipeline(self.data)["comparison"]["ranking"][0], "aardvark")

    def test_invalid_values(self):
        for value in (True, -2, float("nan"), float("inf"), "8 EUR", {}, None):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["products"][0]["attributes"]["price_usd"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_structure_and_references(self):
        cases = []
        data = copy.deepcopy(self.data)
        data["tickets"][0]["product_id"] = "unknown"
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["products"][1]["id"] = "atlas"
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["preferences"]["weights"] = {"price_usd": 0}
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["routing"]["rules"][0]["owner"] = " "
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["surprise"] = 1
        cases.append(data)
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_handoff_rejected(self):
        result = app.compare(self.data)
        result["products"][0]["signals"].append("over_budget")
        with self.assertRaises(app.ValidationError):
            app.triage(self.data, result)

    def test_rule_order_and_whole_word_matching(self):
        self.data["tickets"] = [{"id": "x", "text": "payment compatibility urgently", "product_id": "beacon"}]
        result = app.run_pipeline(self.data)["triage"][0]
        self.assertEqual(result["category"], "billing")
        self.assertEqual(result["priority"], "high")

    def test_deterministic_and_non_mutating(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(original, self.data)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ((), ("does-not-exist.json",), ("example_input.json", "extra")):
            process = self.cli(*args)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_malformed_json_duplicate_keys_and_invalid_schema(self):
        for content in ("{", '{"x":1,"x":2}', "[]"):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                self.assertEqual(app.main(["mocked.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
