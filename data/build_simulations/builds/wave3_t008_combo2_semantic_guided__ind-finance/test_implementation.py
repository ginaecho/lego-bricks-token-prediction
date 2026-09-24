import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).parent


def fixture():
    with (ROOT / "example_input.json").open(encoding="utf-8") as handle:
        return json.load(handle)


class PipelineTests(unittest.TestCase):
    def test_integrated_example_and_explanation(self):
        output = app.run(fixture())
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["semantic"]["selected_product_id"], "SYN-PRODUCT-LOAN")
        self.assertEqual(output["guided"]["selected_product_id"], output["semantic"]["selected_product_id"])
        self.assertEqual(output["guided"]["transaction_ids"], output["semantic"]["transaction_ids"])
        self.assertEqual(output["guided"]["customer_id"], output["customer"]["id"])
        self.assertEqual(output["guided"]["loan_application_id"], output["loan_application"]["id"])
        for match in output["semantic"]["matches"]:
            self.assertEqual(match["score"], match["explanation"]["lexical_coverage"])

    def test_synonym_search_and_stable_ties(self):
        data = fixture()
        data["query"] = "borrowing"
        duplicate = copy.deepcopy(data["products"][0])
        duplicate["id"] = "SYN-AAA"
        data["products"].append(duplicate)
        matches = app.run(data)["semantic"]["matches"]
        self.assertEqual([item["product"]["id"] for item in matches],
                         ["SYN-AAA", "SYN-PRODUCT-LOAN"])
        self.assertEqual(matches[0]["explanation"]["matched_tokens"], ["loan"])

    def test_no_relevant_product(self):
        data = fixture()
        data["query"] = "spaceships"
        output = app.run(data)
        self.assertEqual(output["semantic"]["matches"], [])
        self.assertEqual(output["guided"]["status"], "not_started")
        self.assertEqual(output["guided"]["progress"]["total"], 0)

    def test_optional_injected_embeddings(self):
        data = fixture()
        data["query"] = "spaceships"
        seen = []

        def embed(value):
            seen.append(value)
            return [1.0, 0.0]

        output = app.run(data, embedder=embed)
        self.assertGreater(len(seen), 1)
        for match in output["semantic"]["matches"]:
            self.assertEqual(match["score"], 0.3)
            self.assertEqual(match["explanation"]["embedding_cosine"], 1.0)

    def test_invalid_embedding_contracts(self):
        for value in ([], [0, 0], [float("nan")], [float("inf")], ["1"], [True], [1e9]):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run(fixture(), embedder=lambda _: value)
        with self.assertRaises(app.ValidationError):
            app.run(fixture(), embedder=lambda value: [1] if value == fixture()["query"] else [1, 2])
        with self.assertRaises(app.ValidationError):
            app.run(fixture(), embedder=lambda _: 1 / 0)

    def test_prerequisite_progress_and_resume(self):
        data = fixture()
        first = app.run(data)["guided"]
        self.assertEqual(first["next_steps"], ["kyc_review"])
        data["completed_steps"] = ["kyc_review", "aml_review"]
        second = app.run(data)["guided"]
        self.assertEqual(second["progress"]["percent"], 50)
        self.assertEqual(second["next_steps"], ["transaction_review"])
        data["completed_steps"] = second["resume"]["completed_steps"] + ["transaction_review", "loan_application"]
        self.assertEqual(app.run(data)["guided"]["status"], "complete")

    def test_cannot_skip_prerequisites(self):
        for steps in (["aml_review"], ["loan_application"], ["unknown"], ["kyc_review", "kyc_review"]):
            data = fixture()
            data["completed_steps"] = steps
            with self.subTest(steps=steps), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_kyc_flags_and_unverified_profiles_block(self):
        for change in ({"kyc_status": "pending"}, {"kyc_status": "rejected"},
                       {"kyc_flags": ["identity_mismatch"]}):
            data = fixture()
            data["customer"].update(change)
            self.assertEqual(app.run(data)["guided"]["status"], "blocked")
            data["completed_steps"] = ["kyc_review"]
            with self.assertRaises(app.ValidationError):
                app.run(data)

    def test_aml_flags_block_and_cannot_be_bypassed(self):
        data = fixture()
        data["customer"]["aml_flags"] = ["synthetic_watchlist_hit"]
        data["completed_steps"] = ["kyc_review"]
        self.assertEqual(app.run(data)["guided"]["status"], "blocked")
        data["completed_steps"].append("aml_review")
        with self.assertRaises(app.ValidationError):
            app.run(data)

    def test_card_masking_and_rejection_in_free_text(self):
        data = fixture()
        output = app.run(data)
        card = data["ledger"]["data"][0]["card_number"]
        self.assertNotIn(card, json.dumps(output))
        self.assertEqual(output["transactions"][0]["card_number"], "************4242")
        data["query"] = "card " + card
        with self.assertRaises(app.ValidationError):
            app.run(data)

    def test_json_csv_and_iso_share_canonical_schema(self):
        data = fixture()
        expected = app.run(data)["transactions"]
        rows = data["ledger"]["data"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        data["ledger"] = {"format": "csv", "data": buffer.getvalue()}
        self.assertEqual(app.run(data)["transactions"], expected)
        data["ledger"] = {"format": "iso20022", "data": {"CstmrCdtTrfInitn": {"CdtTrfTxInf": [
            {"EndToEndId": row["id"], "CustomerId": row["customer_id"],
             "InstdAmt": {"Value": row["amount"], "Ccy": row["currency"]},
             "DbtrAcct": row["account"], "RmtInf": row["description"],
             "CardNumber": row["card_number"]}
            for row in rows
        ]}}}
        self.assertEqual(app.run(data)["transactions"], expected)

    def test_randomized_synthetic_histories_are_deterministic(self):
        def generated():
            data = fixture()
            rng = random.Random(42)
            sample = data["ledger"]["data"][0]
            data["ledger"]["data"] = [
                dict(sample, id=f"SYN-RANDOM-{i}", amount=f"{rng.randint(-100000, 100000) / 100:.2f}")
                for i in range(50)
            ]
            return data
        self.assertEqual(app.run(generated()), app.run(generated()))

    def test_empty_history_blocks_review(self):
        data = fixture()
        data["ledger"]["data"] = []
        data["completed_steps"] = ["kyc_review", "aml_review"]
        guided = app.run(data)["guided"]
        self.assertEqual(guided["status"], "blocked")
        self.assertEqual(guided["steps"][2]["evidence_satisfied"], False)

    def test_invalid_industry_values(self):
        mutations = [
            lambda d: d.update(synthetic=False),
            lambda d: d["loan_application"].update(amount=-1),
            lambda d: d["loan_application"].update(amount="NaN"),
            lambda d: d["loan_application"].update(amount="1.001"),
            lambda d: d["loan_application"].update(term_months=True),
            lambda d: d["loan_application"].update(customer_id="SYN-OTHER"),
            lambda d: d["ledger"]["data"][0].update(customer_id="SYN-OTHER"),
            lambda d: d["ledger"]["data"][0].update(account="DE89370400440532013000"),
            lambda d: d["ledger"]["data"][0].update(card_number="123"),
            lambda d: d["ledger"]["data"][0].update(currency="XYZ"),
            lambda d: d["ledger"]["data"].append(copy.deepcopy(d["ledger"]["data"][0])),
            lambda d: d.update(top_k=True),
            lambda d: d.update(query="!!!"),
        ]
        for index, mutate in enumerate(mutations):
            data = fixture()
            mutate(data)
            with self.subTest(index=index), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_invalid_shapes(self):
        for key in ("customer", "loan_application", "ledger", "products"):
            data = fixture()
            data[key] = None
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.run(data)
        for value in (None, [], "text", 10):
            with self.assertRaises(app.ValidationError):
                app.run(value)

    def test_malformed_csv_and_iso(self):
        for ledger in (
            {"format": "csv", "data": "id,amount\nSYN-X,1"},
            {"format": "csv", "data": "id,customer_id,amount,currency,account,description\nSYN-X,too,few"},
            {"format": "iso20022", "data": {}},
            {"format": "iso20022", "data": {"CstmrCdtTrfInitn": {"CdtTrfTxInf": [None]}}},
        ):
            data = fixture()
            data["ledger"] = ledger
            with self.subTest(ledger=ledger), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_handoff_tampering_is_rejected(self):
        state = app.validate_input(fixture())
        for field, value in (("customer_id", "SYN-OTHER"), ("selected_product_id", "SYN-OTHER"),
                             ("loan_application_id", "SYN-OTHER"), ("transaction_ids", []),
                             ("schema_version", "2.0"), ("query", "tampered")):
            handoff = app.semantic_search(state)
            handoff[field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.guided_setup(state, handoff)
        handoff = app.semantic_search(state)
        handoff["matches"][0]["score"] = 0.123
        with self.assertRaises(app.ValidationError):
            app.guided_setup(state, handoff)

    def test_payment_has_no_loan_step_and_currency_filter(self):
        data = fixture()
        data["query"] = "transfer"
        guided = app.run(data)["guided"]
        self.assertEqual(guided["selected_product_id"], "SYN-PRODUCT-PAYMENT")
        self.assertEqual(guided["progress"]["total"], 3)
        data["products"][1]["currency"] = "USD"
        self.assertEqual(app.run(data)["guided"]["status"], "not_started")

    def test_cli_success_is_one_json_object(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertNotIn("4242424242424242", result.stdout)

    def test_cli_file_error_and_usage(self):
        for args in ([], ["nonexistent-build-input.json"]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_validation_errors(self):
        for content in ("{broken", '{"synthetic": false}', "null", '{"schema_version": "1.0"}'):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                exit_code = app.main(["input.json"])
            self.assertEqual(exit_code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_extreme_amounts_and_malformed_card_types(self):
        for amount in ("1e9999999", "-1e9999999", "Infinity"):
            data = fixture()
            data["loan_application"]["amount"] = amount
            with self.subTest(amount=amount), self.assertRaises(app.ValidationError):
                app.run(data)
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=json.dumps(data))), contextlib.redirect_stdout(output):
                self.assertEqual(app.main(["input.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
        for card in (False, 0, [], {}):
            data = fixture()
            data["ledger"]["data"][0]["card_number"] = card
            with self.subTest(card=card), self.assertRaises(app.ValidationError):
                app.run(data)
        with self.assertRaises(app.ValidationError):
            app.run(fixture(), embedder=lambda _: [10 ** 400])


if __name__ == "__main__":
    unittest.main()
