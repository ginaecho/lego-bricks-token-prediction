import copy
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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_discovery_preferences_and_history(self):
        state = app.recommend(app.normalize_input(self.data))
        scores = {r["product_id"]: r["score"] for r in state["recommendations"]}
        self.assertEqual(scores["travel-1"], 35)
        self.assertEqual(scores["saving-1"], 30)
        self.assertNotIn("invest-1", scores)

    def test_alias_search(self):
        self.assertEqual([r["product_id"] for r in app.run(self.data)["results"]], ["saving-1"])

    def test_typo_search(self):
        self.data["query"] = "travle"
        self.assertEqual(app.run(self.data)["results"][0]["product_id"], "travel-1")

    def test_all_query_terms_required(self):
        self.data["query"] = "travel saving"
        self.assertEqual(app.run(self.data)["results"], [])

    def test_handoff_score_propagation(self):
        output = app.run(self.data)
        source = next(r for r in output["recommendations"] if r["product_id"] == "saving-1")
        self.assertEqual(output["results"][0]["explanation"][0]["points"], source["score"])
        self.assertEqual(output["results"][0]["score"], source["score"] + 40)

    def test_scores_explain_every_point(self):
        self.data["query"] = ""
        output = app.run(self.data)
        for row in output["recommendations"] + output["results"]:
            self.assertEqual(row["score"], sum(c["points"] for c in row["explanation"]))

    def test_empty_query_preserves_order(self):
        self.data["query"] = ""
        output = app.run(self.data)
        self.assertEqual([r["product_id"] for r in output["recommendations"]],
                         [r["product_id"] for r in output["results"]])

    def test_empty_catalogue_history_no_loan(self):
        self.data["products"] = []
        self.data["ledger"]["data"] = []
        self.data["loan_application"] = None
        self.assertEqual(app.run(self.data)["results"], [])

    def test_kyc_and_aml_gate(self):
        for kyc, flags in [("pending", []), ("rejected", []), ("verified", ["SYNTH-REVIEW"])]:
            with self.subTest(kyc=kyc):
                self.data["customer"].update(kyc_status=kyc, aml_flags=flags)
                output = app.run(self.data)
                self.assertTrue(output["screening"]["blocked"])
                self.assertEqual(output["recommendations"], [])
                self.assertEqual(output["results"], [])

    def test_card_masking(self):
        output = app.run(self.data)
        self.assertNotIn("0000000000001234", json.dumps(output))
        self.assertEqual(output["transactions"][0]["masked_card"], "************1234")

    def test_invalid_card(self):
        self.data["ledger"]["data"][0]["card_number"] = "short"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_loan_eligibility(self):
        self.data["query"] = "borrow"
        self.assertEqual(app.run(self.data)["results"][0]["product_id"], "loan-1")
        for key, value in [("amount", 10001), ("currency", "USD"), ("status", "declined")]:
            payload = copy.deepcopy(self.data)
            payload["loan_application"][key] = value
            self.assertEqual(app.run(payload)["results"], [])

    def test_csv_ledger(self):
        self.data["ledger"] = {"format": "csv", "data":
            "id,date,amount,currency,category,card_number\n"
            "SYNTH-X,2026-09-01,-10.5,EUR,travel,0000000000001234\n"}
        output = app.run(self.data)
        self.assertEqual(output["transactions"][0]["amount"], -10.5)
        self.assertNotIn("0000000000001234", json.dumps(output))

    def test_iso_style_ledger(self):
        self.data["ledger"] = {"format": "iso20022", "data": {"CstmrCdtTrfInitn": {"PmtInf": [
            {"EndToEndId": "SYNTH-PAY-1", "ReqdExctnDt": "2026-09-01",
             "InstdAmt": {"value": 10, "Ccy": "EUR"}, "CtgyPurp": "travel",
             "CardNumber": "0000000000001234"}]}}}
        output = app.run(self.data)
        self.assertEqual(output["transactions"][0]["id"], "SYNTH-PAY-1")
        self.assertEqual(output["transactions"][0]["masked_card"], "************1234")

    def test_invalid_inputs(self):
        for key, value in [("schema_version", "2"), ("synthetic", False), ("query", 22),
                           ("products", {}), ("loan_application", {})]:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.data)
                payload[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(payload)

    def test_invalid_transaction_values(self):
        for key, value in [("amount", float("nan")), ("amount", True),
                           ("date", "2026-02-30"), ("currency", "euro")]:
            with self.subTest(key=key, value=value):
                payload = copy.deepcopy(self.data)
                payload["ledger"]["data"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(payload)

    def test_duplicates_and_unknown_fields(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        with self.assertRaises(app.ValidationError):
            app.run(self.data)
        self.data["products"].pop()
        self.data["customer"]["cvv"] = "123"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_tampered_handoff_rejected(self):
        state = app.recommend(app.normalize_input(self.data))
        state["recommendations"][0]["score"] += 1
        with self.assertRaises(app.ValidationError):
            app.search(state)
        with self.assertRaises(app.ValidationError):
            app.search(app.normalize_input(self.data))

    def test_deterministic_randomized_synthetic_history(self):
        rng = random.Random(45)
        self.data["ledger"]["data"] = [
            {"id": "SYNTH-RANDOM-" + str(i), "date": "2026-09-01",
             "amount": round(rng.uniform(-500, 500), 2), "currency": "EUR",
             "category": rng.choice(["travel", "saving", "groceries"])} for i in range(30)]
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, before)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["stage"], "search")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "nonexistent.json")]]:
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for contents in ["{", "[]", '{"schema_version": "1"}']:
            with patch("builtins.open", return_value=io.StringIO(contents)), \
                 patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(app.main(["input.json"]), 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_malformed_csv(self):
        for data in ["bad,headers\n1,2\n",
                     "id,date,amount,currency,category,card_number\nx,2026-09-01,nope,EUR,travel,\n"]:
            self.data["ledger"] = {"format": "csv", "data": data}
            with self.assertRaises(app.ValidationError):
                app.run(self.data)


if __name__ == "__main__":
    unittest.main()
