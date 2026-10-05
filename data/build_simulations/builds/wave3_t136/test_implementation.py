import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_synonym_search_and_catalog_fidelity(self):
        output = app.search_products(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["results"][0]["sku"], "SYN-SHOE-101")
        self.assertEqual(output["results"][0]["matched_terms"], ["run", "shoe"])
        for result in output["results"]:
            source = next(p for p in self.data["catalog"] if p["sku"] == result["sku"])
            for field in ("price", "currency", "stock", "name"):
                self.assertEqual(result[field], source[field])
        self.assertEqual(output["basket"][0]["line_total"], "77.00")

    def test_stock_filter(self):
        self.data["search"]["include_out_of_stock"] = True
        self.assertEqual(app.search_products(self.data)["total_matches"], 2)

    def test_no_matches(self):
        self.data["search"]["query"] = "underwater telescope"
        self.assertEqual(app.search_products(self.data)["results"], [])

    def test_empty_catalog(self):
        self.data.update(catalog=[], basket=[], clickstream=[])
        self.assertEqual(app.search_products(self.data)["results"], [])

    def test_consent_required_before_embedding(self):
        for regulation in ("gdpr", "ccpa"):
            data = copy.deepcopy(self.data)
            data["customer"]["consent"][regulation] = False
            calls = []
            with self.assertRaisesRegex(app.ValidationError, "consent required"):
                app.search_products(data, lambda texts: calls.append(texts))
            self.assertEqual(calls, [])

    def test_no_personalization_without_opt_in(self):
        self.data["search"]["personalize"] = False
        baseline = app.search_products(self.data)
        self.data["customer"]["consent"] = {"gdpr": False, "ccpa": False}
        self.data["customer"]["preferences"] = ["unrelated"]
        self.data["clickstream"] = []
        self.data["basket"] = []
        output = app.search_products(self.data)
        self.assertFalse(output["personalized"])
        self.assertEqual(output["results"], baseline["results"])

    def test_rejects_reviews_and_endorsements(self):
        for field in ("reviews", "endorsements", "rating"):
            data = copy.deepcopy(self.data)
            data["catalog"][0][field] = "Invented claim"
            with self.assertRaises(app.ValidationError):
                app.search_products(data)

    def test_invalid_price_stock_and_quantity(self):
        for field, value in (("price", 1.25), ("price", "-2.00"),
                             ("stock", True), ("stock", -1)):
            data = copy.deepcopy(self.data)
            data["catalog"][0][field] = value
            with self.assertRaises(app.ValidationError):
                app.search_products(data)
        self.data["basket"][0]["quantity"] = 8
        with self.assertRaises(app.ValidationError):
            app.search_products(self.data)

    def test_invalid_schema_queries_and_limit(self):
        for search in ({"query": ""}, {"query": "!!!"}, {"query": "shoe", "limit": True},
                       {"query": "shoe", "limit": 0}, {"query": "shoe", "personalize": "yes"}):
            with self.subTest(search=search):
                self.data["search"] = search
                with self.assertRaises(app.ValidationError):
                    app.search_products(self.data)
        with self.assertRaises(app.ValidationError):
            app.search_products([])

    def test_referential_integrity(self):
        for section, field, value in (("basket", "sku", "missing"),
                                      ("clickstream", "customer_id", "other"),
                                      ("clickstream", "sku", "missing"),
                                      ("clickstream", "event", "endorsement")):
            data = copy.deepcopy(self.data)
            data[section][0][field] = value
            with self.assertRaises(app.ValidationError):
                app.search_products(data)
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.search_products(self.data)

    def test_embedding_fixture(self):
        self.data["search"].update(query="carry supplies", personalize=False)
        calls = []
        def fixture(texts):
            calls.append(texts)
            return [[1, 0]] + [[1, 0] if "Backpack" in t else [0, 1] for t in texts[1:]]
        output = app.search_products(self.data, fixture)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)
        self.assertEqual([r["sku"] for r in output["results"]], ["SYN-BAG-201"])
        self.assertEqual(output["results"][0]["score"], 2.0)

    def test_invalid_embedding_outputs(self):
        for vectors in ([], [[1]] * 3, [[0, 0]] * 4, [[float("nan")]] * 4,
                        [[float("inf")]] * 4, [[True]] * 4,
                        [[1], [1, 2], [1], [1]], "invalid"):
            with self.subTest(vectors=vectors):
                with self.assertRaises(app.ValidationError):
                    app.search_products(self.data, lambda texts: vectors)
        def fails(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.search_products(self.data, fails)

    def test_deterministic_ties_and_limit(self):
        product = copy.deepcopy(self.data["catalog"][0])
        product["sku"] = "SYN-SHOE-000"
        self.data["catalog"].append(product)
        self.data["search"].update(personalize=False, limit=1)
        first = app.search_products(self.data)
        self.data["catalog"].reverse()
        self.assertEqual(first, app.search_products(self.data))
        self.assertEqual(first["total_matches"], 2)
        self.assertEqual(first["results"][0]["sku"], "SYN-SHOE-000")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_usage_and_missing_file(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        for content in ("{", "[]", '{"synthetic":true,"synthetic":true}', '{"x":NaN}',
                        json.dumps({**self.data, "synthetic": False})):
            with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(io.StringIO()) as out:
                code = app.main(["fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
