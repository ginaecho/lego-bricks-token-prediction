import copy
import csv
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

from implementation import ValidationError, onboard, validate


ROOT = Path(__file__).resolve().parent


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_guided_explanation(self):
        result = onboard(self.data)
        self.assertEqual(result["status"], "in_progress")
        self.assertEqual(result["onboarding"]["next_step"], "screening")
        self.assertEqual(result["scores"][0]["value"], 20)
        self.assertTrue(all(s["explanation"] for s in result["scores"]))
        self.assertEqual(result["onboarding"]["mode"], "guided")

    def test_expert_concise_and_preference(self):
        self.data["customer"].update(experience="expert", preference="concise")
        self.assertEqual(onboard(self.data)["onboarding"]["mode"], "concise")
        self.data["customer"]["preference"] = "guided"
        self.assertEqual(onboard(self.data)["onboarding"]["mode"], "guided")

    def test_complete_all(self):
        self.data["completed_steps"] = ["profile", "screening", "ledger", "payment", "loan"]
        result = onboard(self.data)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["scores"][0]["value"], 100)
        self.assertIsNone(result["onboarding"]["next_step"])

    def test_empty_ledger_no_loan(self):
        self.data["ledger"]["data"] = []
        del self.data["loan_application"]
        result = onboard(self.data)
        self.assertEqual(len(result["onboarding"]["steps"]), 4)
        self.assertIsNone(result["loan_application"])

    def test_kyc_and_aml_gates(self):
        for changes in ({"kyc_status": "pending"}, {"kyc_status": "rejected"},
                        {"aml_flags": ["SYNTHETIC screening hit"]}):
            with self.subTest(changes=changes):
                data = copy.deepcopy(self.data)
                data["customer"].update(changes)
                result = onboard(data)
                self.assertEqual(result["status"], "review_required")
                self.assertEqual(result["onboarding"]["steps"][1]["state"], "blocked")
                data["completed_steps"].append("screening")
                with self.assertRaises(ValidationError):
                    onboard(data)

    def test_prerequisite_bypass(self):
        self.data["completed_steps"] = ["ledger"]
        with self.assertRaises(ValidationError):
            onboard(self.data)

    def test_recursive_masking(self):
        self.data["ledger"]["data"][0]["description"] = "Card 5555-5555-5555-4444"
        result = onboard(self.data)
        encoded = json.dumps(result)
        self.assertNotIn("4111", encoded)
        self.assertNotIn("5555", encoded)
        self.assertIn("************1111", encoded)
        self.assertIn("************4444", encoded)

    def test_json_csv_equivalence(self):
        expected = onboard(self.data)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(self.data["ledger"]["data"][0]))
        writer.writeheader()
        writer.writerows(self.data["ledger"]["data"])
        self.data["ledger"] = {"format": "csv", "data": buffer.getvalue()}
        self.assertEqual(expected, onboard(self.data))

    def test_invalid_inputs(self):
        for field, value in (("synthetic", False), ("schema_version", "2"),
                             ("customer", None), ("completed_steps", ["profile", "profile"])):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data[field] = value
                with self.assertRaises(ValidationError):
                    onboard(data)

    def test_invalid_financial_values(self):
        for amount in ("NaN", "Infinity", "0.001", True, {}, "1e9999"):
            with self.subTest(amount=amount):
                self.data["payment"]["InstdAmt"]["Value"] = amount
                with self.assertRaises(ValidationError):
                    onboard(self.data)

    def test_invalid_loan_and_payment(self):
        self.data["loan_application"]["term_months"] = True
        with self.assertRaises(ValidationError):
            onboard(self.data)
        self.data["loan_application"]["term_months"] = 12
        self.data["payment"]["DbtrAcct"]["IBAN"] = "ZZ00OTHERFAKE001"
        with self.assertRaises(ValidationError):
            onboard(self.data)

    def test_invalid_csv_and_duplicate_transactions(self):
        data = copy.deepcopy(self.data)
        data["ledger"]["data"].append(data["ledger"]["data"][0])
        with self.assertRaises(ValidationError):
            onboard(data)
        self.data["ledger"] = {"format": "csv", "data": "amount,amount\n1,2"}
        with self.assertRaises(ValidationError):
            onboard(self.data)

    def test_seeded_randomized_synthetic_history(self):
        rng = random.Random(145)
        self.data["ledger"]["data"] = [
            {"transaction_id": f"SYN-RANDOM-{i}", "date": f"2026-09-{i + 1:02}",
             "amount": f"{rng.randint(-50000, 50000) / 100:.2f}",
             "currency": "EUR", "description": "Seed-145 fabricated transaction"}
            for i in range(20)]
        before = copy.deepcopy(self.data)
        self.assertEqual(onboard(self.data), onboard(self.data))
        self.assertEqual(self.data, before)
        self.assertEqual(len(onboard(self.data)["transactions"]), 20)

    def test_injected_guidance_fixture(self):
        result = onboard(self.data, lambda step: "Synthetic guidance for " + step["id"])
        self.assertTrue(result["onboarding"]["steps"][0]["guidance"].startswith("Synthetic"))
        for invalid in (None, {}, "", "x" * 201):
            with self.assertRaises(ValidationError):
                onboard(self.data, lambda step: invalid)

    def test_output_score_validation(self):
        result = onboard(self.data)
        result["scores"][0]["explanation"] = ""
        with self.assertRaises(ValidationError):
            validate(result, output=True)

    def test_hook_isolation_masking_and_failure(self):
        def hook(step):
            step["prerequisites"].clear()
            return "Synthetic card 4111111111111111"
        result = onboard(self.data, hook)
        self.assertEqual(result["onboarding"]["steps"][1]["prerequisites"], ["profile"])
        self.assertNotIn("4111111111111111", json.dumps(result))

        def broken(step):
            raise RuntimeError("Sensitive details should not escape")
        with self.assertRaisesRegex(ValidationError, "^Injected guidance callable failed$"):
            onboard(self.data, broken)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["schema_version"], "1.0")
        self.assertEqual(process.stderr, "")

    def test_cli_errors(self):
        for args in ([], ["does-not-exist.json"], ["implementation.py"]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                     capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")


if __name__ == "__main__":
    unittest.main()
