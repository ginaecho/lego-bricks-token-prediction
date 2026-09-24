import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def test_complete_pipeline(self):
        output = self.run_data()
        self.assertEqual(output["status"], "ok")
        self.assertTrue(output["documents"]["valid"])
        self.assertEqual(output["documents"]["total"], "61.00")

    def test_new_customer_next_step(self):
        self.data["customer"]["completed_steps"] = []
        self.data["customer"]["email"] = ""
        result = self.run_data()["onboarding"]
        self.assertEqual(result["next_step"], "profile")
        self.assertIn("Alex Example", result["greeting"])

    def test_remaining_onboarding_steps(self):
        for completed, step in [
            (["profile"], "preferences"),
            (["profile", "preferences"], "verification"),
            (["profile", "preferences", "verification"], "discover"),
        ]:
            with self.subTest(step=step):
                self.data["customer"]["completed_steps"] = completed
                self.assertEqual(self.run_data()["onboarding"]["next_step"], step)

    def test_personalization_and_filtering(self):
        items = self.run_data()["discovery"]["items"]
        self.assertEqual([item["product_id"] for item in items], ["KIT-1", "BOOK-1"])
        self.assertEqual(items[0]["matched_interests"], ["creative", "learning"])
        self.assertEqual(items[0]["score"], 2)

    def test_fallback_and_tie_breaking(self):
        self.data["customer"]["completed_steps"] = ["profile"]
        self.data["customer"]["interests"] = []
        duplicate = copy.deepcopy(self.data["catalog"][1])
        duplicate["id"] = "AAA"
        self.data["catalog"].append(duplicate)
        items = self.run_data()["discovery"]["items"]
        self.assertEqual([p["product_id"] for p in items], ["AAA", "BOOK-1", "KIT-1"])
        self.assertEqual(items[0]["score"], 0)

    def test_cross_stage_propagation(self):
        self.data["customer"]["id"] = "synthetic-other"
        self.data["customer"]["budget"] = "11"
        result = self.run_data()
        self.assertEqual(result["discovery"]["items"][0]["product_id"], "BOOK-1")
        for stage in ("onboarding", "discovery", "documents"):
            self.assertEqual(result[stage]["customer_id"], "synthetic-other")
        self.assertEqual(result["documents"]["recommended_product_ids"], ["BOOK-1"])
        self.assertEqual(result["documents"]["onboarding_step"], "verification")
        self.assertIn("not_recommended", result["documents"]["rows"][0]["checks"])
        self.assertIn("over_budget", result["documents"]["checks"])

    def test_limit_propagates(self):
        self.data["limit"] = 1
        result = self.run_data()
        self.assertEqual(len(result["discovery"]["items"]), 1)
        self.assertEqual(result["documents"]["recommended_product_ids"], ["KIT-1"])
        self.assertFalse(result["documents"]["rows"][1]["recommended"])

    def test_document_reshaping_and_exact_money(self):
        self.data["document"]["content"] = ' Quantity ,UNIT_PRICE, SKU \r\n3,0.10,"KIT-1"\r\n\r\n'
        row = self.run_data()["documents"]["rows"][0]
        self.assertEqual(row["quantity"], 3)
        self.assertEqual(row["line_total"], "0.30")
        self.assertIn("price_mismatch", row["checks"])

    def test_document_business_checks(self):
        self.data["document"]["content"] = "sku,quantity,unit_price\nUNKNOWN,1,1.00\nART-1,1,20.00\n"
        docs = self.run_data()["documents"]
        self.assertIn("unknown_product", docs["rows"][0]["checks"])
        self.assertIn("out_of_stock", docs["rows"][1]["checks"])
        self.assertFalse(docs["valid"])

    def test_empty_catalog_and_document(self):
        self.data["catalog"] = []
        self.data["document"]["content"] = "sku,quantity,unit_price\n"
        result = self.run_data()
        self.assertEqual(result["discovery"]["items"], [])
        self.assertEqual(result["documents"]["total"], "0.00")
        self.assertEqual(result["documents"]["checks"], ["empty_document"])

    def test_zero_budget(self):
        self.data["customer"]["budget"] = "0"
        self.assertEqual(self.run_data()["discovery"]["items"], [])

    def test_invalid_input_types(self):
        for key, value in [("limit", True), ("schema_version", True), ("catalog", {}), ("customer", []), ("fixture", "real")]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_money(self):
        for value in ["NaN", "Infinity", "-1", "1.234", "1e3", 2, True, ""]:
            with self.subTest(value=value):
                self.data["customer"]["budget"] = value
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_invalid_catalog(self):
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_invalid_customer_state(self):
        for completed in [["verification"], ["profile", "profile"], ["unknown"]]:
            with self.subTest(completed=completed):
                self.data["customer"]["completed_steps"] = completed
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_invalid_csv(self):
        for content in [
            "sku,sku,unit_price\nA,1,2\n",
            "sku,quantity,unit_price\nA,0,1\n",
            "sku,quantity,unit_price\nA,1.5,1\n",
            "sku,quantity,unit_price\nA,1\n",
            "sku,quantity,unit_price\nA,1,NaN\n",
            'sku,quantity,unit_price\n"A,1,2\n',
            "sku,quantity,unit_price\nA,10001,1\n",
        ]:
            with self.subTest(content=content):
                self.data["document"]["content"] = content
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_reject_tampered_handoffs(self):
        onboarding = app.onboard(self.data)
        onboarding["budget"] = "200.00"
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data, onboarding)
        onboarding = app.onboard(self.data)
        discovery = app.recommend(self.data, onboarding)
        discovery["customer_id"] = "wrong"
        with self.assertRaises(app.ValidationError):
            app.automate_documents(self.data, onboarding, discovery)

    def test_document_output_validation(self):
        output = self.run_data()
        output["documents"]["total"] = "0.00"
        with self.assertRaises(app.ValidationError):
            app.validate("documents", output["documents"], output["discovery"])

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(self.data, original)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_usage_and_file_errors(self):
        for args in [[], [str(ROOT / "absent.json")], ["one", "two"]]:
            with self.subTest(args=args):
                proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args], capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for content in ['{', '{"a":1,"a":2}', '{}', 'null', '[]']:
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                    result = app.main(["input.json"])
                self.assertEqual(result, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
