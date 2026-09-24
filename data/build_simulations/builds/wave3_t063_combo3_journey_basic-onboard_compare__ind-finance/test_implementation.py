import copy
import csv
import io
import json
import pathlib
import subprocess
import sys
import unittest
from decimal import Decimal
from unittest.mock import mock_open, patch

import implementation as app


HERE = pathlib.Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_fixture(self):
        return app.run_pipeline(self.fixture)

    def test_integrated_success(self):
        result = self.run_fixture()
        self.assertEqual(result["stage"], "comparison")
        self.assertEqual(result["comparison"]["status"], "compared")
        self.assertEqual(len(result["journey"]["steps"]), 2)
        self.assertTrue(result["onboarding"]["comparison_ready"])

    def test_two_step_prerequisites(self):
        result = app.journey_stage(self.fixture)
        available = {"profile_available", "application_validated"}
        for step in result["journey"]["steps"]:
            self.assertTrue(set(step["requires"]) <= available)
            available.update(step["provides"])
        self.assertEqual(result["journey"]["next_action"], "prepare_comparison")

    def test_personalization_and_cross_stage_propagation(self):
        self.fixture["customer"]["name"] = "Synthetic Morgan"
        self.fixture["loan_application"]["amount"] = "7000"
        self.fixture["customer"]["preferences"] = {"cost": 9, "term": 1, "affordability": 0}
        result = self.run_fixture()
        journey, onboard, comparison = (result[k] for k in ("journey", "onboarding", "comparison"))
        self.assertEqual(onboard["consumed_action"], journey["next_action"])
        self.assertIn("Synthetic Morgan", onboard["personalized_message"])
        self.assertEqual(onboard["comparison_request"]["amount"], "7000")
        self.assertEqual(comparison["application_id"], journey["application_id"])
        score = comparison["ranking"][0]["score_explanation"]["components"]
        self.assertEqual(score["cost"]["raw_weight"], "9")
        self.assertEqual(score["cost"]["inputs"]["amount"], "7000")

    def test_pending_kyc_blocks_without_claiming_completion(self):
        self.fixture["customer"]["kyc"]["status"] = "pending"
        result = self.run_fixture()
        self.assertEqual(result["journey"]["next_action"], "clear_profile")
        self.assertEqual(result["onboarding"]["completed_actions"], [])
        self.assertEqual(result["comparison"]["status"], "blocked")
        self.assertEqual(result["comparison"]["ranking"], [])
        self.assertIn("kyc_pending", result["comparison"]["blockers"])

    def test_aml_and_kyc_screening_flags(self):
        self.fixture["customer"]["aml_flags"] = ["synthetic_review_flag"]
        self.fixture["customer"]["kyc"]["screening_flags"] = ["synthetic_identity_review"]
        result = self.run_fixture()
        self.assertEqual(result["comparison"]["blockers"],
                         ["kyc_screening_review", "aml_screening_review"])
        self.assertFalse(result["onboarding"]["comparison_ready"])

    def test_rejected_kyc_cannot_proceed(self):
        self.fixture["customer"]["kyc"]["status"] = "rejected"
        self.assertEqual(self.run_fixture()["comparison"]["status"], "blocked")

    def test_card_masking_every_handoff(self):
        for stage in (app.journey_stage,
                      lambda p: app.onboarding_stage(app.journey_stage(p)),
                      app.run_pipeline):
            output = stage(self.fixture)
            serialized = json.dumps(output)
            self.assertNotIn("9999888877776666", serialized)
            self.assertNotIn("card_number", serialized)
            self.assertIn("************6666", serialized)

    def test_normalization_of_equivalent_units(self):
        self.fixture["products"][1] = copy.deepcopy(self.fixture["products"][0])
        self.fixture["products"][1].update(id="SYNTH-P02", name="Synthetic equivalent")
        self.fixture["products"][1]["apr"] = {"value": 600, "unit": "basis_points"}
        self.fixture["products"][1]["monthly_fee"] = {"value": 200, "unit": "cents_per_month"}
        self.fixture["products"][1]["term"] = {"value": 2, "unit": "years"}
        result = self.run_fixture()["comparison"]
        first, second = result["side_by_side"][:2]
        for field in ("apr_percent", "monthly_fee", "term_months", "estimated_monthly_payment"):
            self.assertEqual(first[field], second[field])
        scores = {r["product_id"]: r["score"] for r in result["ranking"]}
        self.assertEqual(scores["SYNTH-P01"], scores["SYNTH-P02"])
        tied = [r["product_id"] for r in result["ranking"]
                if r["product_id"] in ("SYNTH-P01", "SYNTH-P02")]
        self.assertEqual(tied, ["SYNTH-P01", "SYNTH-P02"])

    def test_every_score_is_explained_and_reconciles(self):
        for ranked in self.run_fixture()["comparison"]["ranking"]:
            parts = ranked["score_explanation"]["components"]
            total = sum(Decimal(str(p["weighted_contribution"])) for p in parts.values())
            self.assertEqual(total, Decimal(str(ranked["score"])))
            for part in parts.values():
                self.assertTrue(part["formula"])
                self.assertTrue(part["inputs"])
                self.assertGreaterEqual(part["value"], 0)
                self.assertLessEqual(part["value"], 100)

    def test_score_formula_recomputed_independently(self):
        result = self.run_fixture()["comparison"]
        row = next(r for r in result["side_by_side"] if r["product_id"] == "SYNTH-P01")
        self.assertEqual(row["estimated_monthly_payment"], 282.00)
        score = next(r for r in result["ranking"] if r["product_id"] == "SYNTH-P01")
        self.assertAlmostEqual(score["score_explanation"]["components"]["cost"]["value"],
                               100 / 7.4, places=6)

    def test_json_csv_ledger_equivalence(self):
        baseline = self.run_fixture()
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=[
            "id", "date", "amount", "currency", "direction", "account", "card_number"])
        writer.writeheader()
        writer.writerows(self.fixture["transaction_ledger"]["data"])
        self.fixture["transaction_ledger"] = {"format": "csv", "data": stream.getvalue()}
        self.assertEqual(baseline, self.run_fixture())

    def test_iso_payment_ingested_and_summarized(self):
        context = self.run_fixture()["context"]
        iso_rows = [r for r in context["transactions"] if r["source"] == "iso20022_style"]
        self.assertEqual(len(iso_rows), 1)
        self.assertEqual(iso_rows[0]["amount"], "125.50")
        self.assertEqual(context["ledger_summary"]["count"],
                         len(self.fixture["transaction_ledger"]["data"]) + 1)
        debits = sum(Decimal(r["amount"]) for r in context["transactions"]
                     if r["direction"] == "debit")
        self.assertEqual(Decimal(context["ledger_summary"]["debit_total"]), debits)

    def test_empty_ledgers_and_products(self):
        self.fixture["transaction_ledger"]["data"] = []
        self.fixture["payment_payload"]["Document"]["CstmrCdtTrfInitn"]["CdtTrfTxInf"] = []
        self.fixture["products"] = []
        result = self.run_fixture()
        self.assertEqual(result["context"]["ledger_summary"]["count"], 0)
        self.assertEqual(result["comparison"]["status"], "no_eligible_products")
        self.assertIsNone(result["comparison"]["recommended_product_id"])

    def test_ineligible_amount(self):
        for product in self.fixture["products"]:
            product["maximum_amount"] = "2000"
        result = self.run_fixture()["comparison"]
        self.assertEqual(result["ranking"], [])
        self.assertEqual(len(result["excluded"]), len(self.fixture["products"]))

    def test_budget_is_not_inferred_from_ledger(self):
        self.fixture["loan_application"]["monthly_budget"] = "1"
        result = self.run_fixture()["comparison"]
        self.assertTrue(all(not r["within_monthly_budget"] for r in result["side_by_side"]))
        self.assertTrue(all(r["score_explanation"]["components"]["affordability"]["value"] < 1
                            for r in result["ranking"]))

    def test_reject_invalid_numeric_values(self):
        for invalid in (True, "NaN", "Infinity", -1, 0, "1.234", None, []):
            with self.subTest(value=invalid):
                data = copy.deepcopy(self.fixture)
                data["loan_application"]["amount"] = invalid
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_zero_preferences_rejected(self):
        self.fixture["customer"]["preferences"] = {"cost": 0, "affordability": 0, "term": 0}
        with self.assertRaises(app.ValidationError):
            self.run_fixture()

    def test_duplicate_transaction_across_sources(self):
        transfer = self.fixture["payment_payload"]["Document"]["CstmrCdtTrfInitn"]["CdtTrfTxInf"][0]
        transfer["PmtId"] = self.fixture["transaction_ledger"]["data"][0]["id"]
        with self.assertRaises(app.ValidationError):
            self.run_fixture()

    def test_reject_bad_dates_currency_accounts_and_units(self):
        mutations = [
            lambda p: p["transaction_ledger"]["data"][0].update(date="2026-02-30"),
            lambda p: p["transaction_ledger"]["data"][0].update(currency="GBP"),
            lambda p: p["customer"].update(account="DE89370400440532013000"),
            lambda p: p["products"][0]["apr"].update(unit="unknown"),
            lambda p: p["loan_application"].update(customer_id="SYNTH-OTHER"),
            lambda p: p["customer"].update(name="9999 8888 7777 6666"),
            lambda p: p["transaction_ledger"]["data"][0].update(card_number="bad-card"),
            lambda p: p.update(synthetic=False),
        ]
        for mutation in mutations:
            data = copy.deepcopy(self.fixture)
            mutation(data)
            with self.subTest(mutation=mutation), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_tampered_journey_rejected(self):
        previous = app.journey_stage(self.fixture)
        previous["journey"]["steps"][1]["requires"] = ["unavailable"]
        with self.assertRaises(app.ValidationError):
            app.onboarding_stage(previous)

    def test_tampered_onboarding_gate_rejected(self):
        self.fixture["customer"]["aml_flags"] = ["synthetic_hold"]
        previous = app.onboarding_stage(app.journey_stage(self.fixture))
        previous["onboarding"]["comparison_ready"] = True
        with self.assertRaises(app.ValidationError):
            app.comparison_stage(previous)

    def test_tampered_comparison_request_rejected(self):
        previous = app.onboarding_stage(app.journey_stage(self.fixture))
        previous["onboarding"]["comparison_request"]["amount"] = "10"
        with self.assertRaises(app.ValidationError):
            app.comparison_stage(previous)

    def test_stages_do_not_mutate_inputs(self):
        original = copy.deepcopy(self.fixture)
        journey = app.journey_stage(self.fixture)
        saved = copy.deepcopy(journey)
        app.onboarding_stage(journey)
        self.assertEqual(journey, saved)
        self.assertEqual(self.fixture, original)
        self.assertEqual(self.run_fixture(), self.run_fixture())

    def test_invalid_csv_and_duplicate_json_fields(self):
        self.fixture["transaction_ledger"] = {"format": "csv", "data": "id,id\none,two\n"}
        with self.assertRaises(app.ValidationError):
            self.run_fixture()
        with self.assertRaises(app.ValidationError):
            app.parse_json('{"synthetic": true, "synthetic": false}')
        with self.assertRaises(app.ValidationError):
            app.parse_json('{"value": NaN}')

    def test_cli_success_single_json_object(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")],
                                 cwd=HERE, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.strip().splitlines()), 1)
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["does-not-exist.json"], ["a", "b"]):
            process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                     cwd=HERE, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_input_without_writing_extra_files(self):
        for contents in ('{"card": "9999888877776666"', '{"schema_version": 1}', '[]'):
            stdout = io.StringIO()
            with patch("builtins.open", mock_open(read_data=contents)), patch("sys.stdout", stdout):
                code = app.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "error")
            self.assertNotIn("9999888877776666", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
