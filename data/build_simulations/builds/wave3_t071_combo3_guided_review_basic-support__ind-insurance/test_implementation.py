import copy
import datetime as dt
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.raw)

    def test_integrated_pipeline(self):
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["guided"]["progress"], {"completed": 4, "total": 4})
        for stage in ("guided", "review", "support"):
            self.assertEqual(result[stage]["case_id"], result["case"]["claim"]["claim_id"])
        self.assertIn("loss_evidence", result["support"]["answer"])
        self.assertIn("review.checks.loss_evidence", result["support"]["citations"])

    def test_incomplete_onboarding_blocks_downstream(self):
        self.raw["onboarding"]["completed_steps"] = [app.STEPS[0]]
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["review"])
        self.assertIsNone(result["support"])
        self.assertEqual(result["guided"]["steps"][1]["state"], "ready")
        self.assertEqual(result["guided"]["steps"][2]["state"], "blocked")

    def test_out_of_order_prerequisite(self):
        self.raw["onboarding"]["completed_steps"] = [app.STEPS[1]]
        self.invalid()

    def test_duplicate_steps(self):
        self.raw["onboarding"]["completed_steps"].append(app.STEPS[-1])
        self.invalid()

    def test_missing_evidence_has_traceable_reason(self):
        gap = app.run_pipeline(self.raw)["review"]["gaps"][0]
        self.assertEqual(gap["requirement"], "loss_evidence")
        self.assertEqual(gap["entity_id"], "CLM-SYN-071")
        self.assertEqual(gap["sources"], [])
        self.assertTrue(gap["reason"])

    def test_complete_evidence_not_certification(self):
        self.raw["evidence"].append({"evidence_id": "EV-3", "kind": "loss_evidence",
                                    "entity_id": "CLM-SYN-071", "text": "SYNTHETIC damage report."})
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["review"]["gaps"], [])
        self.assertEqual(result["review"]["decision"]["action"], "human_review")
        self.assertFalse(result["review"]["decision"]["final_claim_decision"])
        self.assertIn("no compliance certification", result["disclaimer"])

    def test_wrong_entity_does_not_satisfy_evidence(self):
        self.raw["evidence"][0]["entity_id"] = "OTHER-POLICY"
        result = app.run_pipeline(self.raw)
        self.assertIn("policy_schedule", [g["requirement"] for g in result["review"]["gaps"]])

    def test_disclosure_document_must_name_factors(self):
        self.raw["evidence"][1]["text"] = "SYNTHETIC generic document"
        self.assertIn("underwriting_disclosure",
                      [g["requirement"] for g in app.run_pipeline(self.raw)["review"]["gaps"]])

    def test_hidden_underwriting_factor_rejected(self):
        self.raw["submission"]["content"]["ACORD"]["factors"][0]["disclosed"] = False
        self.invalid()

    def test_protected_factor_rejected(self):
        self.raw["submission"]["content"]["ACORD"]["factors"][0]["name"] = "ethnicity"
        self.invalid()

    def test_minimization_rejects_policyholder_name(self):
        self.raw["policy"]["full_name"] = "Invented Example"
        self.invalid()

    def test_minimization_rejects_contact_in_free_text(self):
        self.raw["support"]["question"] = "Contact invented@example.invalid"
        self.invalid()

    def test_nonsynthetic_rejected(self):
        self.raw["synthetic"] = False
        self.invalid()

    def test_mismatched_policy(self):
        self.raw["claim_form"] = self.raw["claim_form"].replace("POL-SYN-071", "POL-OTHER")
        self.invalid()

    def test_nonfinite_negative_and_boolean_numbers(self):
        for bad in (float("nan"), float("inf"), -1, True, "25000"):
            with self.subTest(value=bad):
                self.raw["policy"]["coverage_limit"] = bad
                self.invalid()

    def test_invalid_and_future_loss_dates(self):
        for bad in ("2026-02-30", "2027-01-01", "20260517"):
            with self.subTest(value=bad):
                original = self.raw["claim_form"]
                self.raw["claim_form"] = original.replace("2026-05-17", bad)
                self.invalid()
                self.raw["claim_form"] = original

    def test_loss_outside_period_and_limit_get_reasons_not_denial(self):
        self.raw["claim_form"] = self.raw["claim_form"].replace("2026-05-17", "2025-05-17").replace("7384.62", "50000")
        decision = app.run_pipeline(self.raw)["review"]["decision"]
        self.assertTrue(any("outside" in r for r in decision["reasons"]))
        self.assertTrue(any("exceeds" in r for r in decision["reasons"]))
        self.assertFalse(decision["final_claim_decision"])

    def test_acord_xml_equivalence(self):
        expected = app.run_pipeline(self.raw)
        self.raw["submission"] = {"format": "acord_xml", "content":
            '<ACORD><SubmissionId>SUB-SYN-071</SubmissionId><PolicyId>POL-SYN-071</PolicyId>'
            '<Factors><Factor name="vehicle_age" value="4" disclosed="true"/>'
            '<Factor name="use_type" value="private" disclosed="true"/></Factors></ACORD>'}
        self.assertEqual(app.run_pipeline(self.raw), expected)

    def test_bad_xml_and_entities(self):
        for xml in ("<ACORD>", '<!DOCTYPE ACORD [<!ENTITY a "x">]><ACORD/>',
                    "<ACORD><Unknown/></ACORD>"):
            self.raw["submission"] = {"format": "acord_xml", "content": xml}
            self.invalid()

    def test_duplicate_claim_fields(self):
        self.raw["claim_form"] += "\nloss_date: 2026-05-17"
        self.invalid()

    def test_duplicate_evidence_id(self):
        self.raw["evidence"].append(copy.deepcopy(self.raw["evidence"][0]))
        self.invalid()

    def test_offline_response_does_not_promise_ticket(self):
        reply = app.run_pipeline(self.raw)["support"]
        self.assertIn("offline", reply["availability"])
        self.assertIn("no ticket has been created", reply["availability"])

    def test_general_support_topics_and_unknown(self):
        for question, topic, fragment in (
            ("What is the coverage limit?", "policy", "25000"),
            ("Explain underwriting factors", "underwriting", "vehicle_age=4"),
            ("Explain GDPR privacy", "privacy", "not certified"),
            ("Tell me tomorrow's weather", "unknown", "does not answer")):
            with self.subTest(topic=topic):
                self.raw["support"]["question"] = question
                reply = app.run_pipeline(self.raw)["support"]
                self.assertEqual(reply["topic"], topic)
                self.assertIn(fragment, reply["answer"])

    def test_handoff_precondition(self):
        self.raw["onboarding"]["completed_steps"] = []
        setup = app.guided(app.validate_case(self.raw))
        with self.assertRaises(app.ValidationError):
            app.review(setup)
        with self.assertRaises(app.ValidationError):
            app.support(setup)

    def test_no_input_mutation_and_deterministic(self):
        original = copy.deepcopy(self.raw)
        self.assertEqual(app.run_pipeline(self.raw), app.run_pipeline(self.raw))
        self.assertEqual(self.raw, original)

    def test_seeded_random_synthetic_properties_losses_dates(self):
        rng = random.Random(71)
        for _ in range(12):
            value = round(rng.uniform(1, 40000), 2)
            date = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(240))
            fixture = copy.deepcopy(self.raw)
            fixture["policy"]["asset"] = {"type": "property", "reference":
                f"SYNTHETIC {rng.randrange(1, 999)} Imaginary Lane, Nowhere"}
            fixture["claim_form"] = fixture["claim_form"].replace("7384.62", str(value)).replace("2026-05-17", date.isoformat())
            self.assertEqual(app.run_pipeline(fixture)["case"]["claim"]["loss_amount"], value)

    def test_cli_success_one_json_object(self):
        proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                               str(HERE / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(HERE / "does-not-exist.json")]):
            proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_bad_json_and_validation_without_scratch_files(self):
        for content in ('{"broken":', '{"synthetic":true,"synthetic":false}', '[]',
                        json.dumps({**self.raw, "synthetic": False})):
            with patch("builtins.open", return_value=io.StringIO(content)), redirect_stdout(io.StringIO()) as out:
                code = app.main(["virtual-input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
