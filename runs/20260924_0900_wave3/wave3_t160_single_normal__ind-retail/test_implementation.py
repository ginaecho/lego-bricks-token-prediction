import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self):
        with self.assertRaises(impl.ValidationError):
            impl.run(self.data)

    def test_normal_exact_citation(self):
        result = impl.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["findings"][0]["sku"], "FABLE-PACK")
        for finding in result["findings"]:
            cite = finding["citation"]
            source = next(s for s in result["sources"] if s["id"] == cite["source_id"])
            self.assertEqual(finding["quote"], source["text"][cite["start"]:cite["end"]])
            self.assertEqual(source["input_pointer"], "/catalog/0")

    def test_price_stock_and_basket(self):
        result = impl.run(self.data)
        self.assertEqual(result["basket"]["total"], "79.90")
        self.assertEqual(result["findings"][0]["price"], "39.95")
        self.assertEqual(result["findings"][0]["stock"], 8)

    def test_no_match(self):
        self.data["research"]["query"] = "telescopes"
        result = impl.run(self.data)
        self.assertTrue(result["no_matches"])
        self.assertEqual(result["findings"], [])

    def test_empty_catalog(self):
        self.data["catalog"] = []
        self.data["basket"]["items"] = []
        self.data["clickstream"] = []
        result = impl.run(self.data)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["basket"]["total"], "0.00")
        self.assertIsNone(result["basket"]["currency"])

    def test_consent_required(self):
        self.data["research"]["personalize"] = True
        self.data["customer"]["consent"]["personalization"] = False
        self.invalid()

    def test_opt_out_overrides_consent(self):
        self.data["research"]["personalize"] = True
        self.data["customer"]["consent"]["ccpa_opt_out"] = True
        self.invalid()

    def test_personalization_tiebreak(self):
        self.data["catalog"][1]["name"] = "Another commuter backpack"
        self.data["research"]["query"] = "commuter backpack"
        self.data["clickstream"][0]["sku"] = "FABLE-TOTE"
        self.assertEqual(impl.run(self.data)["findings"][0]["sku"], "FABLE-PACK")
        self.data["research"]["personalize"] = True
        self.assertEqual(impl.run(self.data)["findings"][0]["sku"], "FABLE-TOTE")

    def test_nonpersonalized_ignores_activity(self):
        before = impl.run(self.data)
        self.data["clickstream"] = []
        self.data["customer"]["consent"]["personalization"] = False
        self.assertEqual(before, impl.run(self.data))

    def test_reviews_and_endorsements_rejected(self):
        for claim in ("Customer reviews praise this bag.", "Celebrity endorsed tote.", "Five star rating."):
            with self.subTest(claim=claim):
                self.data["catalog"][0]["description"] = claim
                self.invalid()

    def test_unstructured_price_stock_rejected(self):
        self.data["catalog"][0]["description"] = "Price is free with plenty of stock."
        self.invalid()

    def test_no_review_fields(self):
        self.data["catalog"][0]["reviews"] = ["Invented testimonial"]
        self.invalid()

    def test_out_of_stock_basket(self):
        self.data["basket"]["items"][0]["sku"] = "FABLE-TOTE"
        self.invalid()

    def test_duplicate_sku(self):
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        self.invalid()

    def test_bad_types(self):
        for key, value in (("price", 39.95), ("stock", True), ("price", "-1.00")):
            with self.subTest(key=key, value=value):
                original = self.data["catalog"][0][key]
                self.data["catalog"][0][key] = value
                self.invalid()
                self.data["catalog"][0][key] = original

    def test_event_constraints(self):
        for key, value in (("sku", "UNKNOWN"), ("timestamp", "yesterday"),
                           ("customer_id", "someone-else")):
            original = self.data["clickstream"][0][key]
            self.data["clickstream"][0][key] = value
            self.invalid()
            self.data["clickstream"][0][key] = original

    def test_deterministic(self):
        self.assertEqual(impl.run(self.data), impl.run(copy.deepcopy(self.data)))

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "nonexistent.json")]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_malformed_and_invalid_json(self):
        for payload in ("{", "[]", '{"schema_version":"1.0","schema_version":"2.0"}'):
            with patch("builtins.open", mock_open(read_data=payload)), redirect_stdout(io.StringIO()) as out:
                self.assertEqual(impl.main(["input.json"]), 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
