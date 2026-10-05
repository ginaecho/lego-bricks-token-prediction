import copy
import csv
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_ranking_and_explanation(self):
        result = app.compare(self.data)
        self.assertEqual(result["ranking"][0]["product_id"], "SYN-LOW-RATE")
        self.assertAlmostEqual(result["ranking"][0]["score"], 70)
        for row in result["comparison"]["rows"]:
            self.assertAlmostEqual(row["score"], sum(
                c["contribution"] for c in row["explanation"]["components"]))
        self.assertEqual(result, app.compare(copy.deepcopy(self.data)))

    def test_apr_and_money_normalization(self):
        product = self.data["products"][0]
        expected = app.compare(self.data)
        product["apr"] = {"value": 0.05, "unit": "fraction"}
        product["fee"] = {"value": 150, "currency": "EUR"}
        self.assertEqual(expected, app.compare(self.data))

    def test_preference_changes_ranking(self):
        self.data["preferences"] = {"apr": 0, "fee": 1, "payment": 0}
        self.assertEqual(app.compare(self.data)["ranking"][0]["product_id"], "SYN-NO-FEE")

    def test_equal_products_stable_tie(self):
        first = self.data["products"][0]
        self.data["products"] = [dict(first, id="B"), dict(first, id="A")]
        result = app.compare(self.data)
        self.assertEqual([x["product_id"] for x in result["ranking"]], ["A", "B"])
        self.assertTrue(all(x["score"] == 100 for x in result["ranking"]))

    def test_zero_apr_and_empty_history(self):
        self.data["products"][0]["apr"]["value"] = 0
        self.data["ledger"]["data"] = []
        result = app.compare(self.data)
        self.assertAlmostEqual(result["comparison"]["rows"][0]["payment"], 12000 / 36)
        self.assertEqual(result["transaction_summary"]["count"], 0)

    def test_kyc_and_aml_blocks_with_reasons(self):
        for field, value, reason in (
            ("kyc_status", "pending", "KYC_NOT_VERIFIED"),
            ("aml_flags", ["SYNTHETIC_SANCTIONS_MATCH"], "AML_REVIEW_REQUIRED"),
        ):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["customer"][field] = value
                result = app.compare(data)
                self.assertEqual(result["ranking"], [])
                self.assertTrue(result["screening"]["review_required"])
                for row in result["comparison"]["rows"]:
                    self.assertIsNone(row["score"])
                    self.assertIn(reason, row["explanation"]["reasons"])

    def test_masking_every_output(self):
        self.data["products"][0]["name"] = "Synthetic 4111-1111-1111-1111"
        result = app.compare(self.data)
        serialized = json.dumps(result)
        self.assertNotIn("4111", serialized)
        self.assertIn("************1111", serialized)
        self.assertNotIn("FAKE-ACCOUNT", serialized)

    def test_affordability_and_range_exclusion(self):
        self.data["products"][0]["max_amount"]["value"] = 2000
        result = app.compare(self.data)
        self.assertEqual(len(result["ranking"]), 1)
        self.assertIn("AMOUNT_OUT_OF_RANGE",
                      result["comparison"]["rows"][0]["exclusion_reasons"])
        self.data["customer"]["monthly_income"] = 300
        self.assertEqual(app.compare(self.data)["ranking"], [])

    def test_invalid_values(self):
        cases = [
            ("synthetic", False),
            ("products", []),
            ("preferences", {"apr": 0, "fee": 0, "payment": 0}),
            ("preferences", {"apr": True, "fee": 1, "payment": 1}),
            ("preferences", {"apr": float("nan"), "fee": 1, "payment": 1}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.compare(data)
        for value in (0, -1, True, 1.5, 601):
            with self.subTest(term=value):
                self.data["loan_application"]["term_months"] = value
                with self.assertRaises(app.ValidationError):
                    app.compare(self.data)

    def test_currency_duplicates_and_date(self):
        for mutation in ("currency", "date", "duplicate", "pan"):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                rows = data["ledger"]["data"]
                if mutation == "currency":
                    rows[0]["currency"] = "USD"
                elif mutation == "date":
                    rows[0]["date"] = "2026-02-30"
                elif mutation == "pan":
                    rows[0]["card_number"] = "123"
                else:
                    rows.append(copy.deepcopy(rows[0]))
                with self.assertRaises(app.ValidationError):
                    app.compare(data)

    def test_seeded_random_history_across_formats(self):
        rng = random.Random(157)
        rows = [{
            "id": "SYN-RANDOM-" + str(i), "date": "2026-08-%02d" % (i + 1),
            "amount": round(rng.uniform(20, 300), 2),
            "currency": "EUR", "direction": "debit",
        } for i in range(20)]
        self.data["ledger"] = {"format": "json", "data": rows}
        expected = app.compare(self.data)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        self.data["ledger"] = {"format": "csv", "data": stream.getvalue()}
        self.assertEqual(expected, app.compare(self.data))
        payments = [{
            "EndToEndId": row["id"], "ReqdExctnDt": row["date"],
            "InstdAmt": {"value": row["amount"], "Ccy": row["currency"]},
            "Direction": row["direction"], "DbtrAcct": "ZZ00FAKE0000157",
        } for row in rows]
        self.data["ledger"] = {"format": "iso20022", "data": {
            "Document": {"CstmrCdtTrfInitn": {"PmtInf": payments}}}}
        self.assertEqual(expected, app.compare(self.data))

    def test_invalid_csv_and_iso(self):
        for ledger in (
            {"format": "csv", "data": "id,amount\nx,20"},
            {"format": "csv", "data": "id,date,amount,currency,direction\nx,2026-08-01,nan,EUR,debit"},
            {"format": "iso20022", "data": {"Document": {}}},
            {"format": "iso20022", "data": {"Document": None}},
        ):
            self.data["ledger"] = ledger
            with self.assertRaises(app.ValidationError):
                app.compare(self.data)

    def test_invalid_optional_card_values(self):
        for value in (False, 0, None, ""):
            self.data["ledger"]["data"][0]["card_number"] = value
            with self.assertRaises(app.ValidationError):
                app.compare(self.data)

    def test_cli_success(self):
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_errors(self):
        for args in ([], ["missing-input.json"], ["example_input.json", "extra"]):
            proc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_without_files(self):
        for content in ("{bad", '{"synthetic":false}', '{"schema_version":"1.0","synthetic":true,"customer":null}'):
            with patch("builtins.open", return_value=io.StringIO(content)):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    self.assertEqual(app.main(["synthetic.json"]), 2)
                    self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
