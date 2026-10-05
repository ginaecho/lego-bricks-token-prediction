import copy
import datetime
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import implementation as app


BASE = Path(__file__).resolve().parent


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((BASE / "example_input.json").read_text(encoding="utf-8"))

    def reject(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def claim_value(self, name, value):
        lines = self.data["claim"]["content"].splitlines()
        self.data["claim"]["content"] = "\n".join(
            f"{name}: {value}" if line.startswith(name + ":") else line for line in lines)

    def test_complete_and_deterministic(self):
        before = copy.deepcopy(self.data)
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["percent_complete"], 100)
        self.assertEqual(result["claim_decision"]["amount"], 7743.52)
        self.assertEqual(result, app.run(self.data))
        self.assertEqual(self.data, before)

    def test_prerequisite_block(self):
        self.data["onboarding"]["requested_steps"] = ["review_claim"]
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["events"][0]["result"], "blocked")
        self.assertEqual(result["onboarding"]["next_step"], "verify_policy")
        self.assertIsNone(result["claim_decision"])

    def test_missing_entity_blocks(self):
        self.data["claim"] = None
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["percent_complete"], 50)
        self.assertFalse(result["onboarding"]["next_step_ready"])

    def test_resume_and_idempotence(self):
        self.data["onboarding"]["completed_steps"] = ["verify_policy", "review_underwriting"]
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["events"][0]["result"], "already_complete")
        self.assertEqual(result["onboarding"]["percent_complete"], 100)
        self.data["onboarding"]["completed_steps"] = list(app.STEPS)
        self.assertEqual(app.run(self.data)["onboarding"]["percent_complete"], 100)

    def test_false_progress_rejected(self):
        self.data["onboarding"]["completed_steps"] = ["review_claim"]
        self.reject()
        self.data["onboarding"]["completed_steps"] = ["verify_policy"]
        self.data["policy"] = None
        self.reject()

    def test_policyholder_minimization(self):
        self.data["policy"]["policyholder"]["email"] = "synthetic@example.invalid"
        self.reject()

    def test_hidden_factor_rejected(self):
        self.data["underwriting_submission"]["content"]["ACORD"]["Submission"]["factors"][0]["disclosed"] = False
        self.reject()

    def test_protected_factor_rejected(self):
        self.data["underwriting_submission"]["content"]["ACORD"]["Submission"]["factors"][0]["name"] = "ethnicity"
        self.reject()

    def test_claim_fairness(self):
        expected = app.run(self.data)["claim_decision"]
        self.data["policy"]["policyholder"]["synthetic_id"] = "SYN-HOLDER-OTHER"
        self.data["underwriting_submission"]["content"]["ACORD"]["Submission"]["factors"][1]["value"] = "none"
        self.assertEqual(expected, app.run(self.data)["claim_decision"])

    def test_decision_reasons_and_boundaries(self):
        for amount, expected in [(0, 0), (500, 0), (501, 1), (100000, 50000)]:
            with self.subTest(amount=amount):
                self.claim_value("loss_amount", amount)
                decision = app.run(self.data)["claim_decision"]
                self.assertEqual(decision["amount"], expected)
                self.assertTrue(decision["reasons"])
        self.claim_value("loss_date", "2025-12-31")
        self.assertIn("outside", app.run(self.data)["claim_decision"]["reasons"][0])
        self.claim_value("loss_date", "2026-12-31")
        self.assertEqual(app.run(self.data)["claim_decision"]["decision"], "payable")
        self.claim_value("loss_type", "theft")
        self.assertIn("not included", app.run(self.data)["claim_decision"]["reasons"][0])

    def test_invalid_money_and_dates(self):
        for value in ["nan", "inf", "-1", "not money"]:
            with self.subTest(value=value):
                self.claim_value("loss_amount", value)
                self.reject()
        self.claim_value("loss_amount", 1000)
        self.claim_value("loss_date", "2026-02-30")
        self.reject()

    def test_mismatched_policy(self):
        self.claim_value("policy_id", "SYN-NOT-THE-POLICY")
        self.reject()

    def test_oversized_amount(self):
        self.data["policy"]["limit"] = 10 ** 400
        self.reject()

    def test_duplicate_claim_field(self):
        self.data["claim"]["content"] += "\nloss_amount: 1"
        self.reject()

    def test_xml_matches_json(self):
        expected = app.run(self.data)
        self.data["underwriting_submission"] = {
            "format": "acord-xml",
            "content": '<ACORD><Submission><submission_id>SYN-SUB-100</submission_id>'
                       '<policy_id>SYN-POL-100</policy_id><Factors>'
                       '<Factor name="property_type" value="house" disclosed="true"/>'
                       '<Factor name="protection" value="alarm" disclosed="true"/>'
                       '</Factors></Submission></ACORD>'}
        self.assertEqual(app.run(self.data), expected)

    def test_xml_rejects_entities_unknown_fields_and_malformed(self):
        for xml in ['<!DOCTYPE ACORD><ACORD/>', "<ACORD>",
                    '<ACORD><Submission><Factors/><Factors/></Submission></ACORD>',
                    "<ACORD><Submission><email>private</email></Submission></ACORD>"]:
            self.data["underwriting_submission"] = {"format": "acord-xml", "content": xml}
            self.reject()

    def test_invalid_envelopes(self):
        for value in [None, [], {}, {"schema_version": True}]:
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run(value)
        self.data["synthetic"] = False
        self.reject()

    def test_seeded_synthetic_loss_fixtures(self):
        rng = random.Random(150)
        for _ in range(25):
            amount = round(rng.uniform(0, 100000), 2)
            day = datetime.date(2026, 1, 1) + datetime.timedelta(days=rng.randrange(365))
            self.claim_value("loss_amount", amount)
            self.claim_value("loss_date", day.isoformat())
            decision = app.run(self.data)["claim_decision"]
            self.assertEqual(decision["amount"], round(min(max(amount - 500, 0), 50000), 2))

    def test_json_duplicates_and_nonfinite(self):
        for raw in ['{"x": 1, "x": 2}', '{"x": NaN}']:
            with self.assertRaises(app.ValidationError):
                app.decode_json(raw)

    def cli(self, *args):
        result = subprocess.run([sys.executable, "-B", str(BASE / "implementation.py"), *args],
                                cwd=BASE, capture_output=True, text=True)
        self.assertEqual(result.stderr, "")
        return result.returncode, json.loads(result.stdout)

    def test_cli_success(self):
        code, output = self.cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(output["status"], "ok")

    def test_cli_file_usage_and_invalid_json_errors(self):
        for args in [(), ("absent.json",), ("implementation.py",), ("example_input.json", "extra")]:
            with self.subTest(args=args):
                code, output = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(output["status"], "error")


if __name__ == "__main__":
    unittest.main()
