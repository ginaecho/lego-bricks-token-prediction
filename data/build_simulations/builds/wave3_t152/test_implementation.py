import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

from implementation import Validator, ValidationError, answer_request, main


ROOT = Path(__file__).resolve().parent


class RetailFAQTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_grounded_returns(self):
        result = answer_request(self.data)
        self.assertEqual(result["answer"], "The return window is 30 days.")
        self.assertEqual(result["citations"], ["kb:synthetic-returns-default"])

    def test_sku_specific_policy(self):
        self.data["request"]["sku"] = "SYN-TOTE-02"
        self.assertEqual(answer_request(self.data)["answer"], "The return window is 14 days.")

    def test_shipping(self):
        self.data["request"]["question"] = "How long does shipping take?"
        self.assertEqual(answer_request(self.data)["answer"], "The shipping time is 4 days.")

    def test_price_matches_exact_minor_units(self):
        self.data["request"]["question"] = "price"
        result = answer_request(self.data)
        self.assertEqual(result["answer"], "SYN-MUG-01 costs USD 12.99.")
        self.assertEqual(result["product"]["price_minor"], 1299)

    def test_zero_stock(self):
        self.data["request"].update(question="stock", sku="SYN-TOTE-02")
        result = answer_request(self.data)
        self.assertEqual(result["product"]["stock"], 0)
        self.assertEqual(result["answer"], "SYN-TOTE-02 has 0 units in stock.")

    def test_consent_required(self):
        self.data["request"].update(question="price", sku=None, personalize=True)
        result = answer_request(self.data)
        self.assertEqual(result["reason"], "consent_required")
        self.assertIsNone(result["product"])
        self.assertFalse(result["customer"]["personalized"])

    def test_consented_persona(self):
        self.data["customer"]["consent_personalization"] = True
        self.data["request"].update(question="price", sku=None, personalize=True)
        result = answer_request(self.data)
        self.assertEqual(result["product"]["sku"], "SYN-TOTE-02")
        self.assertTrue(result["customer"]["personalized"])

    def test_clickstream_latest_by_timestamp(self):
        self.data["customer"].update(consent_personalization=True, preferred_sku=None)
        self.data["request"].update(question="price", sku=None, personalize=True)
        self.data["clickstream"][0].update(timestamp="2026-09-24T13:00:00+02:00", sku="SYN-TOTE-02")
        self.assertEqual(answer_request(self.data)["product"]["sku"], "SYN-TOTE-02")

    def test_no_implicit_personalization(self):
        self.data["request"].update(question="price", sku=None)
        self.assertEqual(answer_request(self.data)["reason"], "missing_product")

    def test_reviews_endorsements_and_mixed_questions_abstain(self):
        for question in ("Show customer reviews", "Who endorses it?", "price and celebrity endorsements",
                         "What is the return window for damaged food?", "What is the warranty?"):
            with self.subTest(question=question):
                self.data["request"]["question"] = question
                result = answer_request(self.data)
                self.assertEqual(result["reason"], "unsupported_question")
                self.assertIsNone(result["answer"])
                self.assertEqual(result["citations"], [])

    def test_missing_knowledge(self):
        self.data["knowledge_base"] = []
        self.assertEqual(answer_request(self.data)["reason"], "missing_knowledge")

    def test_empty_catalog_general_policy(self):
        self.data["catalog"]["products"] = []
        self.data["customer"]["preferred_sku"] = None
        self.data["basket"]["items"] = []
        self.data["clickstream"] = []
        self.data["knowledge_base"] = [self.data["knowledge_base"][0]]
        self.data["request"]["sku"] = None
        self.assertEqual(answer_request(self.data)["status"], "answered")

    def test_bad_catalog_values(self):
        for field, value in (("price_minor", -1), ("price_minor", 12.99),
                             ("price_minor", True), ("stock", -2), ("stock", "8")):
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                data["catalog"]["products"][0][field] = value
                with self.assertRaises(ValidationError):
                    answer_request(data)

    def test_invalid_references_consent_timestamp_and_shape(self):
        variants = []
        for section, key, value in (("request", "sku", "SYN-UNKNOWN"),
                                    ("customer", "consent_personalization", "yes"),
                                    ("request", "question", ""),
                                    ("request", "sku", [])):
            data = copy.deepcopy(self.data)
            data[section][key] = value
            variants.append(data)
        data = copy.deepcopy(self.data)
        data["clickstream"][0]["timestamp"] = "2026-09-24"
        variants.extend([data, {}, [], None])
        for data in variants:
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    answer_request(data)

    def test_duplicate_knowledge_and_sku_rejected(self):
        for key in ("knowledge_base", "products"):
            data = copy.deepcopy(self.data)
            target = data["knowledge_base"] if key == "knowledge_base" else data["catalog"]["products"]
            target.append(copy.deepcopy(target[0]))
            with self.assertRaises(ValidationError):
                answer_request(data)

    def test_output_tampering_rejected(self):
        self.data["request"]["question"] = "price"
        result = answer_request(self.data)
        products = Validator.validate_input(self.data)
        for key, value in (("price_minor", 100), ("stock", 99)):
            changed = copy.deepcopy(result)
            changed["product"][key] = value
            with self.assertRaises(ValidationError):
                Validator.validate_output(changed, self.data, products)
        result["answer"] = "Everyone loves it; endorsed by a celebrity."
        with self.assertRaises(ValidationError):
            Validator.validate_output(result, self.data, products)

    def test_input_not_mutated_and_deterministic(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(answer_request(self.data), answer_request(self.data))
        self.assertEqual(self.data, before)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "answered")
        self.assertEqual(run.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "nonexistent-input.json")], ["a", "b"]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_malformed_json_and_invalid_schema(self):
        for text in ('{', '{"a":1,"a":2}', '{"x":NaN}', '{}', 'null'):
            with self.subTest(text=text), patch("builtins.open", mock_open(read_data=text)):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
