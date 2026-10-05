import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).parent


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        output = app.review(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["results"]), 4)
        self.assertEqual(output, app.review(self.data))
        self.assertEqual(original, self.data)

    def test_traceable_expired_and_unverified_gaps(self):
        output = app.review(self.data)
        address = next(g for g in output["gaps"] if g["requirement"] == "address")
        self.assertEqual(address["document_ids"], ["address_demo"])
        self.assertEqual(address["reason"], "no_valid_evidence")
        check = output["results"][0]["checks"][1]
        self.assertEqual(check["evidence"][0]["rejection_reasons"], ["expired"])
        loan = output["results"][-1]
        self.assertEqual(loan["checks"][0]["evidence"][0]["rejection_reasons"], ["unverified"])

    def test_masking_no_raw_pan(self):
        output = json.dumps(app.review(self.data))
        for tx in self.data["transaction_ledger"]["data"]:
            self.assertNotIn(tx["card_number"], output)
        self.assertIn("************1234", output)

    def test_screening_inherited_and_scores_explained(self):
        self.data["customers"][0]["screening"]["sanctions"] = True
        for result in app.review(self.data)["results"]:
            self.assertIn("aml_sanctions", result["screening_flags"])
            self.assertIn("kyc_incomplete", result["screening_flags"])
            score = result["risk_score"]
            self.assertEqual(score["raw_total"], sum(c["points"] for c in score["contributions"]))
            self.assertEqual(score["value"], min(100, score["raw_total"]))
            self.assertTrue(all(c["basis"] for c in score["contributions"]))

    def test_csv_json_equivalence(self):
        expected = app.review(self.data)
        import csv
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=app.LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(self.data["transaction_ledger"]["data"])
        self.data["transaction_ledger"] = {"format": "csv", "data": stream.getvalue()}
        self.assertEqual(app.review(self.data), expected)

    def test_invalid_csv(self):
        for data in ("id,amount\nx,2\n", ",".join(app.LEDGER_FIELDS) + "\nx,c,1,USD,2026-01-01,0000000000000000,extra\n"):
            with self.subTest(data=data):
                self.data["transaction_ledger"] = {"format": "csv", "data": data}
                with self.assertRaises(app.ValidationError):
                    app.review(self.data)

    def test_invalid_amounts(self):
        for value in (-1, 0, True, "NaN", "Infinity", "0.001", [], "oops"):
            with self.subTest(value=value):
                self.data["transaction_ledger"]["data"][0]["amount"] = value
                with self.assertRaises(app.ValidationError):
                    app.review(self.data)

    def test_payment_mismatch(self):
        self.data["payments"][0]["PmtInf"]["InstdAmt"]["Value"] = "12.00"
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_unknown_reference_and_duplicates(self):
        self.data["loans"][0]["customer_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.review(self.data)
        self.data["loans"][0]["customer_id"] = "customer_demo"
        self.data["customers"].append(copy.deepcopy(self.data["customers"][0]))
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_synthetic_and_boolean_constraints(self):
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.review(self.data)
        self.data["synthetic"] = True
        self.data["customers"][0]["screening"]["pep"] = "false"
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_empty_fixture(self):
        for key in ("customers", "loans", "payments", "documents"):
            self.data[key] = []
        self.data["transaction_ledger"]["data"] = []
        output = app.review(self.data)
        self.assertEqual(output["results"], [])
        self.assertEqual(output["gaps"], [])

    def test_current_verified_evidence_and_zero_score(self):
        self.data["transaction_ledger"]["data"] = []
        self.data["loans"] = []
        self.data["payments"] = []
        customer = self.data["customers"][0]
        customer["kyc_status"] = "complete"
        customer["screening"]["pep"] = False
        self.data["documents"] = [
            {"id": f"doc_{kind}", "entity_type": "customer",
             "entity_id": customer["id"], "kind": kind, "verified": True,
             "issued_on": self.data["as_of"], "expires_on": self.data["as_of"]}
            for kind in app.REQUIREMENTS["customer"]
        ]
        output = app.review(self.data)
        self.assertEqual(output["gaps"], [])
        self.assertEqual(output["results"][0]["risk_score"]["value"], 0)
        self.assertEqual(output["results"][0]["disposition"], "requirements_met")

    def test_seeded_randomized_synthetic_history(self):
        rng = random.Random(134)
        base = self.data["transaction_ledger"]["data"][0]
        self.data["payments"] = []
        self.data["documents"] = []
        transactions = []
        for i in range(25):
            tx = dict(base, id=f"synthetic_tx_{i}",
                      amount=f"{rng.randrange(1, 2000000) / 100:.2f}",
                      date=f"2026-09-{rng.randrange(1, 25):02}")
            transactions.append(tx)
        self.data["transaction_ledger"]["data"] = transactions
        output = app.review(self.data)
        self.assertEqual(len(output["results"]), 27)
        self.assertEqual(output, app.review(self.data))

    def test_zero_income_and_threshold(self):
        self.data["loans"][0]["monthly_income"] = "0"
        output = app.review(self.data)
        self.assertIn("affordability_review", output["results"][-1]["screening_flags"])
        self.data["loans"][0]["monthly_income"] = "4500"
        output = app.review(self.data)
        self.assertNotIn("affordability_review", output["results"][-1]["screening_flags"])

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_usage_and_invalid_schema(self):
        for args in ([], [str(ROOT / "nonexistent.json")],
                     [str(ROOT / "build_manifest.json")],
                     [str(ROOT / "implementation.py")]):
            with self.subTest(args=args):
                proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                      capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_duplicate_json_keys(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"synthetic": true, "synthetic": false}',
                       object_pairs_hook=app.unique_object)


if __name__ == "__main__":
    unittest.main()
