import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from implementation import ValidationError, search


ROOT = Path(__file__).resolve().parent


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_example_and_all_formats(self):
        result = search(self.request)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["indexed"], 4)
        formats = {hit["record"]["data"].get("source_format") for hit in result["results"]}
        self.assertTrue({"json", "csv", "iso20022"} <= formats)

    def test_semantic_synonym(self):
        self.request["query"] = "remittance"
        hits = search(self.request)["results"]
        self.assertIn("transaction_iso", [hit["record"]["id"] for hit in hits])
        self.assertTrue(all("payment" in hit["explanation"]["canonical_term_contributions"] for hit in hits))

    def test_scores_are_reconstructible(self):
        for hit in search(self.request)["results"]:
            why = hit["explanation"]
            self.assertAlmostEqual(sum(why["canonical_term_contributions"].values()), why["lexical_score"])
            self.assertAlmostEqual(hit["score"], why["lexical_weight"] * why["lexical_score"]
                                   + why["embedding_weight"] * why["embedding_score"])
            self.assertIn("not creditworthiness", why["meaning"])

    def test_screening_default_and_opt_in(self):
        self.assertEqual(search(self.request)["summary"]["excluded_by_screening"], 1)
        self.request["options"]["include_flagged"] = True
        hits = search(self.request)["results"]
        loan = next(hit for hit in hits if hit["record"]["id"] == "loan_demo")
        self.assertTrue(loan["record"]["screening_required"])
        self.assertEqual(loan["explanation"]["screening"], "review_required")

    def test_aml_alone_and_kyc_alone(self):
        loan = self.request["records"][-1]
        loan["kyc_status"] = "verified"
        self.assertEqual(search(self.request)["summary"]["excluded_by_screening"], 1)
        loan["aml_flags"] = []
        loan["kyc_status"] = "rejected"
        self.assertEqual(search(self.request)["summary"]["excluded_by_screening"], 1)

    def test_card_masking_in_structured_and_free_text(self):
        self.request["query"] += " 4111111111111111"
        self.request["records"][0]["text"] += " card 4111-1111-1111-1111"
        rendered = json.dumps(search(self.request))
        self.assertNotIn("4111111111111111", rendered)
        self.assertNotIn("4111-1111-1111-1111", rendered)
        self.assertIn("****1111", rendered)

    def test_empty_and_no_match(self):
        self.request["query"] = "unmatchablexyz"
        self.assertEqual(search(self.request)["results"], [])
        self.request["records"] = []
        self.assertEqual(search(self.request)["summary"]["indexed"], 0)

    def test_tie_break_and_limit(self):
        row = self.request["records"][1]
        twin = copy.deepcopy(row)
        twin["id"] = "aaa"
        self.request["records"] = [row, twin]
        self.request["query"] = "mortgage"
        self.request["options"]["top_k"] = 1
        self.assertEqual(search(self.request)["results"][0]["record"]["id"], "aaa")

    def test_invalid_inputs(self):
        cases = [None, [], {}, {**self.request, "synthetic": False},
                 {**self.request, "query": " "}, {**self.request, "records": {}},
                 {**self.request, "options": {"top_k": True}},
                 {**self.request, "options": {"include_flagged": 1}},
                 {**self.request, "schema_version": True}]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValidationError):
                    search(case)

    def test_invalid_record_constraints(self):
        for field, value in [("entity", "unknown"), ("kyc_status", "unknown"),
                             ("aml_flags", "clear"), ("data", []), ("id", "bad id")]:
            request = copy.deepcopy(self.request)
            request["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                search(request)
        self.request["records"].append(copy.deepcopy(self.request["records"][0]))
        with self.assertRaises(ValidationError):
            search(self.request)

    def test_invalid_money_card_and_formats(self):
        for amount in (False, -1, 0, float("nan"), float("inf"), "10"):
            request = copy.deepcopy(self.request)
            request["records"][1]["data"]["payload"]["amount"] = amount
            with self.subTest(amount=amount), self.assertRaises(ValidationError):
                search(request)
        self.request["records"][0]["data"]["card_number"] = 4111111111111111
        with self.assertRaises(ValidationError):
            search(self.request)

    def test_malformed_format_payloads(self):
        for fmt, payload in [("csv", "amount,currency\n1,EUR\n2,EUR"),
                             ("csv", "amount,amount\n1,2"),
                             ("csv", "amount,currency,customer_id\nnan,EUR,c"),
                             ("iso20022", {}), ("xml", "<payment/>")]:
            request = copy.deepcopy(self.request)
            request["records"][1]["data"] = {"format": fmt, "payload": payload}
            with self.subTest(fmt=fmt, payload=payload), self.assertRaises(ValidationError):
                search(request)

    def test_embedding_injection_and_masking(self):
        received = []
        def embed(batch):
            received.extend(batch)
            return [[1.0, 0.0] for _ in batch]
        self.request["query"] = "unmatchablexyz"
        hits = search(self.request, embed=embed)["results"]
        self.assertEqual(len(hits), 4)
        self.assertTrue(all(math.isclose(hit["score"], 0.35) for hit in hits))
        self.assertNotIn("4111111111111111", json.dumps(received))

    def test_invalid_embedding_contract(self):
        bad = [None, [], [[0, 0]] * 5, [[1, float("nan")]] * 5,
               [[1, True]] * 5, [[1], [1, 2], [1], [1], [1]]]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                search(self.request, embed=lambda batch, value=value: value)
        def failing(batch):
            raise RuntimeError("private provider internals")
        with self.assertRaisesRegex(ValidationError, "^injected embedding callable failed$"):
            search(self.request, embed=failing)

    def test_seeded_random_transaction_history_and_no_mutation(self):
        rng = random.Random(83)
        template = self.request["records"][1]
        rows = []
        for index in range(25):
            row = copy.deepcopy(template)
            row["id"] = "synthetic_tx_" + str(index)
            row["text"] = "SYNTHETIC randomized payment history"
            row["data"]["payload"]["amount"] = round(rng.uniform(1, 5000), 2)
            rows.append(row)
        self.request["records"] = rows
        original = copy.deepcopy(self.request)
        self.assertEqual(search(self.request), search(self.request))
        self.assertEqual(self.request, original)
        self.assertEqual(search(self.request)["summary"]["matched"], 25)

    def test_cli_success(self):
        run = self.cli("example_input.json")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(len(run.stdout.splitlines()), 1)
        self.assertEqual(run.stderr, "")

    def test_cli_error_paths(self):
        for args in [(), ("does-not-exist.json",), ("test_implementation.py",),
                     ("build_manifest.json",), ("example_input.json", "extra")]:
            with self.subTest(args=args):
                run = self.cli(*args)
                self.assertEqual(run.returncode, 2)
                self.assertEqual(json.loads(run.stdout)["status"], "error")
                self.assertEqual(run.stderr, "")


if __name__ == "__main__":
    unittest.main()
