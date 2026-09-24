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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def result(self):
        return app.run_pipeline(self.raw)["data"]

    def test_pending_customer_has_personalized_next_step(self):
        row = self.result()["onboarding"][0]
        self.assertEqual(row["next_step"]["code"], "complete_kyc")
        self.assertIn("Synthetic Avery", row["next_step"]["message"])
        self.assertEqual(row["readiness_score"]["value"], 0)

    def test_loan_documents_next_step(self):
        self.assertEqual(self.result()["onboarding"][1]["next_step"]["code"], "complete_loan_documents")

    def test_customer_and_transaction_flags_block(self):
        self.raw["customers"][2]["screening_flags"] = []
        self.assertEqual(self.result()["onboarding"][2]["status"], "blocked")
        self.raw["ledger"]["data"][2]["screening_flags"] = []
        self.raw["customers"][2]["screening_flags"] = ["aml_review"]
        self.assertEqual(self.result()["onboarding"][2]["readiness_score"]["value"], 0)

    def test_insights_themes_actions_and_explainability(self):
        result = self.result()
        self.assertEqual({x["theme"] for x in result["insights"]}, {"verification", "loans", "payments"})
        for item in result["insights"]:
            explanation = item["priority_score"]
            self.assertEqual(explanation["value"], min(100, sum(explanation["components"].values())))
            self.assertTrue(explanation["formula"])
            self.assertTrue(explanation["limitations"])
            self.assertTrue(item["action"])

    def test_handoff_propagates_blocked_next_step(self):
        payments = next(x for x in self.result()["insights"] if x["theme"] == "payments")
        self.assertEqual(payments["onboarding_context"][0]["next_step_code"], "specialist_review")
        self.assertEqual(payments["priority_score"]["components"]["blocked_customers"], 20)

    def test_validated_handoff_rejects_tampering(self):
        onboarded = app.onboard(app.normalize(self.raw))
        onboarded["onboarding"][0]["readiness_score"]["value"] = 99
        with self.assertRaises(app.ValidationError):
            app.insights(onboarded)

    def test_stage_order_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.insights(app.normalize(self.raw))

    def test_cards_masked_in_fields_and_free_text(self):
        self.raw["feedback"][0]["text"] += " Card 4111-1111-1111-1111."
        self.raw["customers"][0]["name"] = "Synthetic 4111111111111111"
        result = self.result()
        serialized = json.dumps(result)
        self.assertNotIn("4111111111111111", serialized)
        self.assertNotIn("4111-1111-1111-1111", serialized)
        self.assertNotIn('"card_number"', serialized)
        self.assertEqual(result["transactions"][0]["masked_card"], "****1111")

    def test_csv_matches_json(self):
        expected = self.result()
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=sorted(app.TX_FIELDS))
        writer.writeheader()
        for item in self.raw["ledger"]["data"]:
            row = dict(item)
            row["screening_flags"] = "|".join(row["screening_flags"])
            writer.writerow(row)
        self.raw["ledger"] = {"format": "csv", "data": stream.getvalue()}
        self.assertEqual(self.result(), expected)

    def test_iso20022_matches_json(self):
        expected = self.result()
        payments = []
        for item in self.raw["ledger"]["data"]:
            payments.append({
                "PmtId": {"EndToEndId": item["id"]}, "Dbtr": {"Id": item["customer_id"]},
                "Amt": {"InstdAmt": {"value": item["amount"], "Ccy": item["currency"]}},
                "ReqdExctnDt": item["booking_date"], "RmtInf": {"Ustrd": item["description"]},
                "ScreeningFlags": item["screening_flags"], "CardNumber": item["card_number"],
            })
        self.raw["ledger"] = {"format": "iso20022", "data": {
            "Document": {"CstmrCdtTrfInitn": {"PmtInf": [{"CdtTrfTxInf": payments}]}}}}
        self.assertEqual(self.result(), expected)

    def test_empty_population(self):
        self.raw.update(customers=[], loans=[], feedback=[], ledger={"format": "json", "data": []})
        result = self.result()
        self.assertEqual(result["onboarding"], [])
        self.assertEqual(result["insights"], [])

    def test_no_history_and_borrowing_variants(self):
        self.raw["ledger"]["data"] = []
        self.raw["feedback"] = []
        self.raw["customers"][0]["kyc_status"] = "verified"
        self.assertEqual(self.result()["onboarding"][0]["next_step"]["code"], "review_first_payment")
        self.raw["loans"] = []
        self.assertEqual(self.result()["onboarding"][1]["next_step"]["code"], "start_loan_application")

    def test_rejected_kyc_requires_review(self):
        self.raw["customers"][0]["kyc_status"] = "rejected"
        self.assertEqual(self.result()["onboarding"][0]["status"], "blocked")

    def test_input_unchanged_and_deterministic(self):
        original = copy.deepcopy(self.raw)
        self.assertEqual(self.result(), self.result())
        self.assertEqual(self.raw, original)

    def test_randomized_synthetic_histories(self):
        rng = random.Random(3702)
        self.raw["feedback"] = []
        template = self.raw["ledger"]["data"][0]
        self.raw["ledger"]["data"] = [
            dict(template, id=f"R{i:03}", customer_id=f"C00{rng.randint(1, 3)}",
                 amount=f"{rng.randint(1, 9999)}.{rng.randint(0, 99):02}")
            for i in range(30)]
        result = self.result()
        self.assertEqual(len(result["transactions"]), 30)
        self.assertEqual(result, self.result())

    def test_duplicate_and_foreign_references_rejected(self):
        variants = []
        duplicate = copy.deepcopy(self.raw)
        duplicate["customers"].append(copy.deepcopy(duplicate["customers"][0]))
        variants.append(duplicate)
        unknown = copy.deepcopy(self.raw)
        unknown["ledger"]["data"][0]["customer_id"] = "Unknown"
        variants.append(unknown)
        ownership = copy.deepcopy(self.raw)
        ownership["feedback"][1]["customer_id"] = "C001"
        variants.append(ownership)
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)

    def test_invalid_industry_constraints(self):
        changes = [
            ("customers", "kyc_status", "approved"),
            ("customers", "screening_flags", ["ignored"]),
            ("loans", "amount", "NaN"),
            ("loans", "amount", "0"),
            ("loans", "documents_complete", "false"),
        ]
        for collection, field, value in changes:
            raw = copy.deepcopy(self.raw)
            raw[collection][0][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)
        self.raw["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            self.result()

    def test_realistic_account_rejected(self):
        self.raw["customers"][0]["account"]["iban"] = "DE89370400440532013000"
        with self.assertRaises(app.ValidationError):
            self.result()

    def test_bad_dates_amounts_and_card_lengths(self):
        for field, value in [("booking_date", "2026-02-30"), ("amount", "1.001"),
                             ("amount", 1.25), ("card_number", "1234")]:
            raw = copy.deepcopy(self.raw)
            raw["ledger"]["data"][0][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)

    def test_invalid_csv_and_iso(self):
        for ledger in [
            {"format": "csv", "data": "id,id\nx,y"},
            {"format": "iso20022", "data": {"Document": {}}},
            {"format": "yaml", "data": []},
        ]:
            self.raw["ledger"] = ledger
            with self.subTest(ledger=ledger), self.assertRaises(app.ValidationError):
                self.result()

    def test_score_cap_and_neutral_other_theme(self):
        self.raw["feedback"] = [
            dict(self.raw["feedback"][0], id=f"F{i:03}", text="Synthetic fee concern")
            for i in range(20)]
        self.assertEqual(self.result()["insights"][0]["priority_score"]["value"], 100)
        self.raw["feedback"] = [dict(self.raw["feedback"][0], text="Synthetic greeting", sentiment="neutral")]
        self.assertEqual(self.result()["insights"][0]["theme"], "other")
        self.assertEqual(self.result()["insights"][0]["priority_score"]["value"], 5)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in [[], [str(ROOT / "nonexistent.json")], ["a", "b"]]:
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_malformed_json_and_schema_without_files(self):
        for content in ['{', '{"schema_version":1,"schema_version":1}', 'NaN',
                        '[]', json.dumps(dict(self.raw, synthetic=False))]:
            with self.subTest(content=content), patch("builtins.open", return_value=io.StringIO(content)), \
                    patch("sys.stdout", new_callable=io.StringIO) as output:
                code = app.main(["fake.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
