import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, text=True, capture_output=True)

    def test_normal_exact_citations(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["research"]["findings"])
        sources = {s["id"]: s["text"] for s in self.data["sources"]}
        for finding in result["research"]["findings"]:
            citation = finding["citation"]
            self.assertEqual(sources[citation["source_id"]][citation["start"]:citation["end"]],
                             finding["finding"])
            self.assertEqual(citation["quote"], finding["finding"])

    def test_masking_no_raw_pan(self):
        result = app.run(self.data)
        self.assertNotIn("4111111111111111", json.dumps(result))
        self.assertEqual(result["entities"]["customers"][0]["card_number"], "************1111")

    def test_explainable_screening_and_relevance(self):
        result = app.run(self.data)
        for assessment in result["assessments"]:
            score = assessment["screening_score"]
            self.assertEqual(score["value"], min(100, sum(x["points"] for x in score["explanation"])))
        for finding in result["research"]["findings"]:
            score = finding["relevance_score"]
            self.assertEqual(score["value"], len(score["explanation"]["matched_terms"]))

    def test_kyc_aml_loan_flags(self):
        result = app.run(self.data)
        rows = {(r["entity_type"], r["entity_id"]): r for r in result["assessments"]}
        self.assertEqual(rows["customer", "C2"]["screening_score"]["value"], 100)
        self.assertTrue(any("AML threshold" in flag for flag in rows["transaction", "P1"]["flags"]))
        self.assertTrue(any("3 times" in flag for flag in rows["loan_application", "L1"]["flags"]))

    def test_csv_matches_json(self):
        baseline = app.run(self.data)
        self.data["transactions"] = {
            "format": "csv",
            "data": "id,customer_id,amount,currency,date\nT1,C1,231.54,USD,2026-01-01\n"
                    "T2,C2,860.75,EUR,2026-01-02\n"}
        self.assertEqual(app.run(self.data), baseline)

    def test_iso_payment_normalized(self):
        transaction = app.run(self.data)["entities"]["transactions"][-1]
        self.assertEqual(transaction, {"id": "P1", "customer_id": "C2", "amount": 15000,
                                      "currency": "USD", "date": "2026-01-03"})

    def test_no_match_and_empty_sources(self):
        self.data["query"] = "unfindablequasar"
        self.assertTrue(app.run(self.data)["research"]["no_matches"])
        self.data["sources"] = []
        self.assertEqual(app.run(self.data)["research"]["findings"], [])

    def test_reject_invalid_numbers(self):
        for invalid in (True, -1, 0, float("nan"), float("inf"), "10"):
            with self.subTest(invalid=invalid):
                data = copy.deepcopy(self.data)
                data["transactions"][0]["amount"] = invalid
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_invalid_references_and_duplicates(self):
        for change in ("reference", "duplicate"):
            data = copy.deepcopy(self.data)
            if change == "reference":
                data["loan_applications"][0]["customer_id"] = "missing"
            else:
                data["transactions"].append(copy.deepcopy(data["transactions"][0]))
            with self.assertRaises(app.ValidationError):
                app.run(data)

    def test_reject_unmasked_source_and_non_synthetic(self):
        self.data["sources"][0]["text"] = "Card 4111-1111-1111-1111."
        with self.assertRaises(app.ValidationError):
            app.run(self.data)
        self.setUp()
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_csv_and_date(self):
        for content in ('"unterminated header', "wrong,header\nx,y\n",
                        "id,customer_id,amount,currency,date\nT9,C1,nan,USD,2026-01-01\n",
                        "id,customer_id,amount,currency,date\nT9,C1,10,USD,2026-02-30\n",
                        "id,customer_id,amount,currency,date\nT9,C1,10,USD\n"):
            self.data["transactions"] = {"format": "csv", "data": content}
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_newline_passages_and_unicode_offsets(self):
        self.data["sources"] = [{"id": "S1", "title": "Synthetic",
                                 "text": "  Café KYC review\n  loan review"}]
        result = app.run(self.data)
        self.assertEqual(len(result["research"]["findings"]), 2)
        for finding in result["research"]["findings"]:
            citation = finding["citation"]
            self.assertEqual(self.data["sources"][0]["text"][citation["start"]:citation["end"]],
                             citation["quote"])

    def test_input_not_mutated(self):
        before = copy.deepcopy(self.data)
        app.run(self.data)
        self.assertEqual(self.data, before)

    def test_seeded_random_history_deterministic(self):
        rng = random.Random(51)
        self.data["transactions"] = [
            {"id": "R" + str(i), "customer_id": rng.choice(["C1", "C2"]),
             "amount": round(rng.uniform(1, 20000), 2), "currency": "USD",
             "date": "2026-01-01"} for i in range(40)]
        self.assertEqual(app.run(self.data), app.run(copy.deepcopy(self.data)))
        self.assertEqual(len(app.run(self.data)["entities"]["transactions"]), 41)

    def test_zero_income_and_limit(self):
        self.data["loan_applications"][0]["annual_income"] = 0
        self.data["max_findings"] = 1
        result = app.run(self.data)
        self.assertEqual(len(result["research"]["findings"]), 1)
        self.assertEqual(result["assessments"][-1]["screening_score"]["value"], 50)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_errors(self):
        # Existing deliverables provide malformed JSON and schema-invalid JSON
        # without creating any additional files.
        for args in ([], ["missing.json"], ["implementation.py"], ["build_manifest.json"]):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_bad_shapes_are_validation_errors(self):
        for field, value in (("customers", {}), ("sources", [None]),
                             ("payments", [{"CdtTrfTxInf": []}]),
                             ("max_findings", True), ("query", "")):
            data = copy.deepcopy(self.data)
            data[field] = value
            with self.assertRaises(app.ValidationError):
                app.run(data)


if __name__ == "__main__":
    unittest.main()
