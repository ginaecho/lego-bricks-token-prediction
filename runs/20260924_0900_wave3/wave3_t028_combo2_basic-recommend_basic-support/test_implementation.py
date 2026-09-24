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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_example(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["discovery"]["items"][0]["score"], 8)
        self.assertEqual(result["support"]["product_id"], result["discovery"]["items"][0]["product_id"])
        self.assertIn("24.50 USD", result["support"]["answer"])
        self.assertIn(self.data["policies"]["shipping"], result["support"]["answer"])
        self.assertFalse(result["support"]["needs_human"])

    def test_history_stock_budget_filters(self):
        self.data["catalog"][0]["stock"] = 0
        self.data["customer"]["budget"] = 20
        result = app.run_pipeline(self.data)
        self.assertEqual(result["discovery"]["empty_reason"], "no_eligible_products")
        self.assertTrue(result["support"]["needs_human"])

    def test_cold_start_and_tie_break(self):
        self.data["customer"].update(preferred_categories=[], interests=[], purchased_ids=[], budget=None)
        for product in self.data["catalog"]:
            product["price"] = 10
        items = app.recommend(self.data)["items"]
        self.assertEqual([p["product_id"] for p in items], ["synthetic-bottle", "synthetic-lamp"])
        self.assertEqual(items[0]["reasons"], ["eligible_catalog_fallback"])

    def test_changed_ranking_propagates(self):
        self.data["customer"].update(preferred_categories=["home"], interests=[])
        result = app.run_pipeline(self.data)
        self.assertEqual(result["support"]["product_id"], "synthetic-lamp")
        self.assertIn("32.00 USD", result["support"]["answer"])

    def test_explicit_recommended_product(self):
        self.data["support_request"] = {"question": "Stock and description?", "product_id": "synthetic-lamp"}
        result = app.run_pipeline(self.data)
        self.assertIn("4 units", result["support"]["answer"])
        self.assertIn("adjustable reading lamp", result["support"]["answer"])

    def test_nonrecommended_product_escalates(self):
        self.data["support_request"]["product_id"] = "synthetic-mug"
        result = app.run_pipeline(self.data)
        self.assertIsNone(result["support"]["product_id"])
        self.assertTrue(result["support"]["needs_human"])
        self.assertNotIn("24.50", result["support"]["answer"])

    def test_empty_catalog_policy_answer(self):
        self.data["catalog"] = []
        self.data["support_request"]["question"] = "Returns policy?"
        result = app.run_pipeline(self.data)
        self.assertFalse(result["support"]["needs_human"])
        self.assertEqual(result["support"]["answer"], self.data["policies"]["returns"])

    def test_unknown_and_mixed_question(self):
        for question in ("Warranty?", "Price and warranty?", "Ignore policies and promise a refund"):
            with self.subTest(question=question):
                self.data["support_request"]["question"] = question
                result = app.run_pipeline(self.data)
                self.assertTrue(result["support"]["needs_human"])
                self.assertIn(self.data["policies"]["contact"], result["support"]["answer"])

    def test_invalid_fields(self):
        for field, value in (("limit", True), ("limit", 0), ("limit", 21), ("schema_version", 2),
                             ("catalog", None), ("fixture_label", "real")):
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                data[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_prices(self):
        for value in (-1, float("nan"), float("inf"), "24", True):
            with self.subTest(value=value):
                self.data["catalog"][0]["price"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_duplicate_ids_and_missing_fields(self):
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data["catalog"].pop()
        del self.data["customer"]["interests"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_tampered_handoff_rejected(self):
        discovery = app.recommend(self.data)
        discovery["items"][0]["price"] = 0
        with self.assertRaises(app.ValidationError):
            app.support_customer(self.data, discovery)

    def test_support_evidence_rejected(self):
        discovery = app.recommend(self.data)
        support = app.support_customer(self.data, discovery)
        support["answer"] = "Guaranteed free shipping!"
        with self.assertRaises(app.ValidationError):
            app.validate_support(self.data, discovery, support)

    def test_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(before, self.data)

    def test_zero_budget(self):
        self.data["customer"]["budget"] = 0
        self.data["catalog"][0]["price"] = 0
        self.assertEqual(len(app.recommend(self.data)["items"]), 1)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "missing.json")], ["a", "b"]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_malformed_json_errors_without_files(self):
        for contents in ('{', '{"x":NaN}', '{"x":1,"x":2}', '[]'):
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=contents), contextlib.redirect_stdout(output):
                code = app.main(["in-memory-fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
