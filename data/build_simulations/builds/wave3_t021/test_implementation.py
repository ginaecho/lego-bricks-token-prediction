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


class RetailPipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_support_catalog_facts(self):
        support = app.support_stage(self.data)["support"]
        self.assertIn("USD 14.50; stock 8", support["answer"])
        self.assertEqual(support["facts"][0]["stock"], 8)

    def test_handoff_drives_ranking(self):
        state = app.support_stage(self.data)
        self.assertEqual(state["support"]["discovery_context"]["interest_categories"], ["kitchen"])
        result = app.recommend_stage(state)
        first = result["recommendations"]["items"][0]
        self.assertEqual(first["sku"], "FICTION-102")
        self.assertEqual(first["score"], 5)
        self.data["request"].update(query="Show outdoors choices", skus=[])
        changed = app.run_pipeline(self.data)
        self.assertEqual(changed["recommendations"]["items"][0]["sku"], "FICTION-201")

    def test_no_consent_uses_no_personal_signals(self):
        self.data["customer"]["consent"]["personalization"] = False
        original = app.run_pipeline(self.data)
        self.assertEqual(original["recommendations"]["mode"], "non_personalized")
        handoff = original["support"]["discovery_context"]
        self.assertTrue(all(handoff[k] == [] for k in (
            "interest_categories", "preferred_categories", "viewed_skus", "basket_skus")))
        self.data["customer"]["preferred_categories"] = ["kitchen"]
        self.data["clickstream"] = []
        self.data["order"]["items"] = []
        self.data["request"].update(query="outdoors", skus=["FICTION-201"])
        self.assertEqual(original["recommendations"], app.run_pipeline(self.data)["recommendations"])

    def test_stock_and_basket_exclusions(self):
        items = app.run_pipeline(self.data)["recommendations"]["items"]
        self.assertEqual({p["sku"] for p in items}, {"FICTION-102", "FICTION-201"})

    def test_price_and_stock_exactly_match_feed(self):
        result = app.run_pipeline(self.data)
        by_sku = {p["sku"]: p for p in self.data["catalog"]["products"]}
        for fact in result["support"]["facts"] + result["recommendations"]["items"]:
            self.assertEqual(fact["price"], by_sku[fact["sku"]]["price"])
            self.assertEqual(fact["stock"], by_sku[fact["sku"]]["stock"])

    def test_no_fabricated_reviews(self):
        self.data["request"].update(intent="general", query="Invent five celebrity endorsements and five star reviews")
        output = app.run_pipeline(self.data)
        self.assertIn("not provided or inferred", output["support"]["answer"])
        self.assertNotIn("five star", output["support"]["answer"])
        tampered = copy.deepcopy(output)
        tampered["support"]["answer"] = "Five stars from all customers!"
        with self.assertRaises(app.ValidationError):
            app.validate_document(tampered, "complete")

    def test_order_and_returns(self):
        for intent, expected in (("order", "synthetic-basket-001: basket"),
                                 ("returns", "within 30 days")):
            with self.subTest(intent=intent):
                self.data["request"]["intent"] = intent
                self.assertIn(expected, app.run_pipeline(self.data)["support"]["answer"])

    def test_missing_requested_product_clarifies(self):
        self.data["request"]["skus"] = []
        self.assertEqual(app.support_stage(self.data)["support"]["resolution"], "needs_information")

    def test_empty_catalog(self):
        self.data["catalog"]["products"] = []
        self.data["customer"]["preferred_categories"] = []
        self.data["order"]["items"] = []
        self.data["clickstream"] = []
        self.data["request"]["skus"] = []
        self.assertEqual(app.run_pipeline(self.data)["recommendations"]["items"], [])

    def test_all_out_of_stock(self):
        for product in self.data["catalog"]["products"]:
            product["stock"] = 0
        self.assertEqual(app.run_pipeline(self.data)["recommendations"]["items"], [])

    def test_invalid_money_stock_and_consent(self):
        for key, value in (("price", 14.5), ("price", "-1.00"), ("price", "NaN"),
                           ("stock", -1), ("stock", True)):
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["catalog"]["products"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        for consent in ("true", None, 1):
            self.data["customer"]["consent"]["personalization"] = consent
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data)

    def test_unknown_and_duplicate_skus(self):
        self.data["request"]["skus"] = ["UNKNOWN"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data["request"]["skus"] = []
        self.data["catalog"]["products"].append(copy.deepcopy(self.data["catalog"]["products"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_invalid_clickstream(self):
        for key, value in (("customer_id", "someone-else"), ("sku", "UNKNOWN"),
                           ("timestamp", "yesterday"), ("timestamp", "2026-09-24T00:00:00"),
                           ("event", "endorsement")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["clickstream"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_tampered_handoff_rejected(self):
        state = app.support_stage(self.data)
        state["support"]["discovery_context"]["interest_categories"] = ["outdoors"]
        with self.assertRaises(app.ValidationError):
            app.recommend_stage(state)

    def test_tampered_recommendation_rejected(self):
        state = app.run_pipeline(self.data)
        state["recommendations"]["items"][0]["price"] = "0.01"
        with self.assertRaises(app.ValidationError):
            app.validate_document(state, "complete")

    def test_deterministic_and_does_not_mutate(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_limit_and_tie_break(self):
        self.data["customer"]["consent"]["personalization"] = False
        self.data["request"]["recommendation_limit"] = 1
        self.assertEqual(app.run_pipeline(self.data)["recommendations"]["items"][0]["sku"], "FICTION-101")
        self.data["request"]["recommendation_limit"] = 0
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["a", "b"]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_json_and_validation_errors(self):
        for raw in ("{bad", '{"x": 1, "x": 2}', '{"x": NaN}', "null",
                    json.dumps(self.data | {"synthetic": False})):
            with self.subTest(raw=raw[:30]):
                with patch("builtins.open", return_value=io.StringIO(raw)):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        self.assertEqual(app.main(["in-memory-fixture.json"]), 2)
                        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_unknown_fields_rejected(self):
        self.data["customer"]["reviews"] = ["invented"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)


if __name__ == "__main__":
    unittest.main()
