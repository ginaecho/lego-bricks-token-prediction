import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

from implementation import ValidationError, mask, run


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_deterministic(self):
        result = run(self.data)
        self.assertEqual(result, run(copy.deepcopy(self.data)))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["research"]["findings"]), 3)
        self.assertEqual(result["payments"][0]["transaction_id"], "txn-1")

    def test_evidence_hash_offsets(self):
        result = run(self.data)
        source = self.data["research"]["sources"][0]
        for finding in result["research"]["findings"]:
            self.assertEqual(finding["source_sha256"], hashlib.sha256(source["content"].encode()).hexdigest())
            self.assertEqual(finding["quote"], source["content"][finding["start"]:finding["end"]])
            self.assertEqual(finding["source_url"], source["url"])

    def test_allowlist_exact_match(self):
        for address in ("http://research.example.invalid/synthetic-finance",
                        "https://research.example.invalid.evil.test/synthetic-finance",
                        "https://research.example.invalid/synthetic-finance?redirect=1",
                        "https://user:pass@research.example.invalid/synthetic-finance"):
            with self.subTest(address=address):
                data = copy.deepcopy(self.data)
                data["research"]["sources"][0]["url"] = address
                with self.assertRaises(ValidationError):
                    run(data)

    def test_masking_all_output_paths(self):
        self.data["research"]["sources"][0]["content"] += " KYC card 4111-1111-1111-1111."
        self.data["customers"][0]["name"] = "SYNTHETIC 4111111111111111"
        result = run(self.data)
        serialized = json.dumps(result)
        self.assertNotIn("4111111111111111", serialized)
        self.assertNotIn("4111-1111-1111-1111", serialized)
        self.assertEqual(result["transactions"][0]["card_number"], "************1111")
        digest = "a" * 48 + "4111111111111111"
        self.assertEqual(mask({"sha256": digest})["sha256"], digest)

    def test_screening_and_explainability(self):
        result = run(self.data)
        self.assertIn("KYC_REVIEW", result["customers"][0]["flags"])
        self.assertIn("AML_REVIEW", result["customers"][1]["flags"])
        self.assertIn("LARGE_TRANSACTION_REVIEW", result["transactions"][0]["flags"])
        for collection in ("customers", "transactions", "loan_applications"):
            for entity in result[collection]:
                s = entity["score"]
                self.assertEqual(s["value"], min(100, sum(c["points"] for c in s["explanation"])))
                self.assertTrue(all(c["reason"] for c in s["explanation"]))

    def test_csv_equivalence(self):
        expected = run(self.data)
        self.data["ledger"] = {"format": "csv", "data":
            "id,customer_id,amount,currency,card_number,aml_flag\n"
            "txn-1,cust-fake-1,12500,USD,4111111111111111,false\n"
            "txn-2,cust-fake-2,78.53,USD,5555555555554444,true\n"}
        self.assertEqual(expected, run(self.data))

    def test_invalid_csv(self):
        for csv in ("id,amount\nx,2", "id,customer_id,amount,currency,card_number,aml_flag\nx"):
            self.data["ledger"] = {"format": "csv", "data": csv}
            with self.assertRaises(ValidationError):
                run(self.data)

    def test_invalid_schema_and_financial_values(self):
        for field, value in (("amount", -1), ("amount", True), ("amount", float("nan")),
                             ("customer_id", "missing"), ("aml_flag", "false"),
                             ("card_number", "123"), ("currency", "usd")):
            data = copy.deepcopy(self.data)
            data["ledger"]["data"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                run(data)
        self.data["synthetic"] = False
        with self.assertRaises(ValidationError):
            run(self.data)

    def test_empty_optional_arrays_and_no_findings(self):
        self.data["ledger"]["data"] = []
        self.data["loan_applications"] = []
        self.data["payment_payloads"] = []
        self.data["research"]["keywords"] = ["unmatched keyword"]
        result = run(self.data)
        self.assertEqual(result["research"]["findings"], [])
        self.assertEqual(result["transactions"], [])

    def test_xml_mismatch_and_entity_rejected(self):
        original = self.data["payment_payloads"][0]
        for payload in (original.replace(">12500<", ">99<"),
                        original.replace("ZZ00000000000000000001", "ZZ00000000000000000002"),
                        "<!DOCTYPE Document [<!ENTITY x 'bad'>]><Document/>",
                        "<Document>"):
            self.data["payment_payloads"] = [payload]
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                run(self.data)

    def test_injected_callable_evidence_contract(self):
        result = run(self.data, lambda content, keywords:
                     [{"start": 0, "end": len(content), "quote": content}])
        self.assertEqual(len(result["research"]["findings"]), 1)
        for extractor in (lambda c, k: [{"start": 0, "end": 4, "quote": "fake"}],
                          lambda c, k: {"summary": "unsupported"},
                          lambda c, k: [{"start": -1, "end": 4, "quote": "bad"}]):
            with self.assertRaises(ValidationError):
                run(self.data, extractor)

    def test_seeded_random_synthetic_histories(self):
        rng = random.Random(142)
        self.data["payment_payloads"] = []
        self.data["ledger"]["data"] = [
            {"id": "random-fake-" + str(i), "customer_id": rng.choice(["cust-fake-1", "cust-fake-2"]),
             "amount": rng.randint(1, 2000000) / 100, "currency": "USD",
             "card_number": "4111111111111111", "aml_flag": bool(rng.randrange(2))}
            for i in range(50)]
        result = run(self.data)
        self.assertEqual(len(result["transactions"]), 50)
        self.assertTrue(all(0 <= t["score"]["value"] <= 100 for t in result["transactions"]))

    def test_duplicate_ids_and_bad_timestamp(self):
        self.data["customers"].append(copy.deepcopy(self.data["customers"][0]))
        with self.assertRaises(ValidationError):
            run(self.data)
        self.data["customers"].pop()
        self.data["research"]["sources"][0]["retrieved_at"] = "2026-09-24"
        with self.assertRaises(ValidationError):
            run(self.data)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_invalid_input(self):
        for args in ([], ["nonexistent-fixture.json"], ["build_manifest.json"]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main()
