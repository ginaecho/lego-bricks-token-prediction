import copy
import csv
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_grounded_faq_and_handoff(self):
        result = app.run(self.data)
        self.assertEqual(result["faq"]["answer"], self.data["knowledge_base"][0]["answer"])
        self.assertEqual(result["faq"], result["documents"]["support_response"])
        self.assertEqual(result["transactions"], result["documents"]["rows"])

    def test_abstention_propagates(self):
        self.data["question"] = "Astronomy telescope?"
        result = app.run(self.data)
        self.assertEqual(result["faq"]["status"], "abstained")
        self.assertIsNone(result["documents"]["support_response"]["answer"])

    def test_empty_knowledge_base(self):
        self.data["knowledge_base"] = []
        self.assertEqual(app.run(self.data)["faq"]["status"], "abstained")

    def test_injected_grounded_callable(self):
        result = app.run(self.data, lambda question, evidence: evidence["answer"])
        self.assertEqual(result["faq"]["status"], "answered")

    def test_injected_hallucination_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.data, lambda question, evidence: "Guaranteed approval!")

    def test_mutated_handoff_rejected(self):
        result = app.answer_stage(app.normalize(self.data))
        result["faq"]["evidence_ids"] = ["unknown"]
        with self.assertRaises(app.ValidationError):
            app.document_stage(result)

    def test_masks_cards_everywhere(self):
        self.data["customer"]["name"] = "Synthetic 4111 1111 1111 1111"
        serialized = json.dumps(app.run(self.data))
        self.assertNotIn("4111111111111111", serialized)
        self.assertNotIn("4111 1111 1111 1111", serialized)
        self.assertIn("****1111", serialized)
        self.assertIn("****4444", serialized)

    def test_screening_and_explanation(self):
        docs = app.run(self.data)["documents"]
        self.assertTrue(docs["screening"]["requires_manual_review"])
        self.assertEqual(docs["loan_risk"]["score"], 95)
        self.assertEqual(docs["transaction_risks"][1]["score"], 90)
        for score in [docs["loan_risk"]] + docs["transaction_risks"]:
            self.assertEqual(score["score"], min(100, sum(f["points"] for f in score["factors"])))

    def test_clean_profile(self):
        self.data["customer"].update(kyc_status="verified", aml_flags=[])
        self.data["loan_application"]["amount"] = "1000"
        docs = app.run(self.data)["documents"]
        self.assertFalse(docs["screening"]["requires_manual_review"])
        self.assertEqual(docs["loan_risk"]["score"], 0)
        self.assertTrue(docs["loan_risk"]["factors"])

    def test_unknown_profile_fields_rejected(self):
        self.data["customer"]["raw_card"] = 4111111111111111
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_large_amount_not_mistaken_for_pan_and_score_cap(self):
        self.data["loan_application"]["amount"] = "1000000000000"
        self.data["customer"]["aml_flags"] = ["sanctions_match", "pep", "unusual_activity"]
        result = app.run(self.data)
        self.assertEqual(result["loan_application"]["amount"], "1000000000000.00")
        self.assertEqual(result["documents"]["loan_risk"]["score"], 100)

    def test_currency_totals_and_csv(self):
        docs = app.run(self.data)["documents"]
        self.assertEqual(docs["totals_by_currency"], {"EUR": "13708.65", "USD": "63.47"})
        self.assertEqual(list(csv.DictReader(io.StringIO(docs["ledger_csv"]))), docs["rows"])

    def test_csv_input(self):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=app.FIELDS)
        writer.writeheader()
        writer.writerows(self.data["transaction_source"]["data"])
        expected = app.run(self.data)["transactions"]
        self.data["transaction_source"] = {"format": "csv", "data": stream.getvalue()}
        self.assertEqual(app.run(self.data)["transactions"], expected)

    def test_json_text_input(self):
        self.data["transaction_source"]["data"] = json.dumps(self.data["transaction_source"]["data"])
        self.assertEqual(len(app.run(self.data)["transactions"]), 3)

    def test_iso20022_input(self):
        self.data["transaction_source"] = {"format": "iso20022", "data": f"""
        <Document xmlns="{app.NS}"><CstmrCdtTrfInitn><PmtInf>
        <ReqdExctnDt><Dt>2026-09-10</Dt></ReqdExctnDt><CdtTrfTxInf>
        <PmtId><EndToEndId>SYN-ISO-1</EndToEndId></PmtId>
        <Amt><InstdAmt Ccy="EUR">12.50</InstdAmt></Amt>
        <CdtrAcct><Id><IBAN>FAKE-IBAN-ZZ00-ALPHA</IBAN></Id></CdtrAcct>
        </CdtTrfTxInf></PmtInf></CstmrCdtTrfInitn></Document>"""}
        result = app.run(self.data)
        self.assertEqual(result["documents"]["totals_by_currency"], {"EUR": "12.50"})

    def test_xml_entities_rejected(self):
        self.data["transaction_source"] = {"format": "iso20022", "data": '<!DOCTYPE x><x/>'}
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_empty_ledger(self):
        self.data["transaction_source"]["data"] = []
        docs = app.run(self.data)["documents"]
        self.assertEqual(docs["totals_by_currency"], {})
        self.assertEqual(docs["transaction_risks"], [])

    def test_invalid_amounts(self):
        for value in ("NaN", "Infinity", "-1", "0", "1.001", True, None):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                data = copy.deepcopy(self.data)
                data["transaction_source"]["data"][0]["amount"] = value
                app.run(data)

    def test_invalid_profiles(self):
        for field, value in (("kyc_status", "approved"), ("aml_flags", ["unknown"])):
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                data = copy.deepcopy(self.data)
                data["customer"][field] = value
                app.run(data)

    def test_invalid_schema_and_duplicates(self):
        for mutate in (
            lambda d: d.update(synthetic=False),
            lambda d: d["loan_application"].update(annual_income=0),
            lambda d: d["transaction_source"]["data"][0].update(date="2026-02-30"),
            lambda d: d["transaction_source"]["data"].append(copy.deepcopy(d["transaction_source"]["data"][0])),
            lambda d: d["transaction_source"]["data"][0].update(card_number="123"),
        ):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                data = copy.deepcopy(self.data)
                mutate(data)
                app.run(data)

    def test_formula_csv_escape(self):
        self.data["transaction_source"]["data"][0]["account"] = "=FAKE()"
        docs = app.run(self.data)["documents"]
        self.assertIn("'=FAKE()", docs["ledger_csv"])

    def test_randomized_synthetic_history_is_repeatable(self):
        rng = random.Random(23)
        self.data["transaction_source"]["data"] = [
            {"transaction_id": f"SYN-R{i}", "date": "2026-09-01",
             "amount": f"{rng.randrange(1, 2000000) / 100:.2f}", "currency": "EUR",
             "account": f"FAKE-ACCOUNT-{i}", "card_number": ""}
            for i in range(50)]
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(app.run(self.data)["documents"]["checks"]["transaction_count"], 50)

    def test_input_not_mutated(self):
        before = copy.deepcopy(self.data)
        app.run(self.data)
        self.assertEqual(self.data, before)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, text=True, capture_output=True, check=False)

    def test_cli_success(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_error(self):
        process = self.cli("nonexistent-input.json")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_input_and_usage(self):
        for args in ((), ("implementation.py",), ("build_manifest.json",)):
            process = self.cli(*args)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")


if __name__ == "__main__":
    unittest.main()
