import copy
import datetime as dt
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class JourneyTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_two_step_handoff(self):
        result = app.recommend(self.data)
        self.assertEqual(result["selected_goal"], "claim")
        self.assertEqual(result["journey"][0]["action"], "collect_claim_evidence")
        self.assertEqual(result["journey"][0]["projected_state_after"],
                         result["journey"][1]["state_before"])
        self.assertEqual([x["action"] for x in result["recommendations"]],
                         ["collect_claim_evidence"])
        self.assertEqual(result["claim_handling"]["decision"], "needs_evidence")

    def test_ready_claim_and_no_mutation(self):
        self.data["claim"] = app.decode_claim(self.data["claim"])
        self.data["claim"]["evidence_ready"] = True
        before = copy.deepcopy(self.data)
        result = app.recommend(self.data)
        self.assertEqual(result["journey"][1]["action"], "prepare_claim_submission")
        self.assertEqual(result["claim_handling"]["decision"], "ready_for_preparation")
        self.assertEqual(self.data, before)

    def test_personalized_explicit_goals(self):
        for goal in ("underwriting", "policy"):
            with self.subTest(goal=goal):
                self.data["goal"] = goal
                result = app.recommend(self.data)
                self.assertEqual(result["selected_goal"], goal)
                self.assertEqual(len(result["journey"]), 2)

    def test_xml_submission_equivalence(self):
        expected = app.recommend(self.data)
        self.data["underwriting_submission"] = (
            "<ACORDSubmission><SubmissionID>UW-SYN-107</SubmissionID>"
            "<PolicyID>POL-SYN-107</PolicyID><VIN>SYNTHET1C00000107</VIN>"
            '<Factor name="annual_mileage" value="9000" disclosed="true"/>'
            '<Factor name="vehicle_age" value="4" disclosed="true"/>'
            "</ACORDSubmission>")
        self.assertEqual(app.recommend(self.data), expected)

    def test_minimization(self):
        result = json.dumps(app.recommend(self.data))
        self.assertNotIn("invented-Robin-Finch", result)
        self.assertNotIn(self.data["underwriting_submission"]["vin"], result)
        for field in ("email", "name", "date_of_birth", "property_address"):
            data = copy.deepcopy(self.data)
            data["policy"][field] = "synthetic sensitive field"
            with self.assertRaises(app.ValidationError):
                app.recommend(data)

    def test_hidden_sensitive_duplicate_factors(self):
        for factor in (
            {"name": "vehicle_age", "value": 4, "disclosed": False},
            {"name": "ethnicity", "value": 1, "disclosed": True},
            {"name": "vehicle_age", "value": 4, "disclosed": True},
        ):
            data = copy.deepcopy(self.data)
            data["underwriting_submission"]["factors"].append(factor)
            with self.assertRaises(app.ValidationError):
                app.recommend(data)

    def test_fair_referral_and_reasons(self):
        self.data["claim"] = app.decode_claim(self.data["claim"])
        self.data["claim"]["loss_date"] = "2025-12-31"
        result = app.recommend(self.data)
        self.assertEqual(result["claim_handling"]["decision"], "human_review")
        self.assertTrue(result["claim_handling"]["reasons"])
        self.assertIn("not automatic denial", result["claim_handling"]["reasons"][1])
        changed = copy.deepcopy(self.data)
        changed["policy"]["policyholder_ref"] = "invented-Alex-River"
        changed["claim"]["loss_amount"] = 99999
        self.assertEqual(app.recommend(changed)["claim_handling"], result["claim_handling"])

    def test_invalid_numbers_dates_and_references(self):
        for key, value in (("loss_amount", float("nan")), ("loss_amount", True),
                           ("loss_amount", -1), ("loss_amount", float("inf")),
                           ("loss_date", "2026-02-30"), ("loss_date", "2027-01-01"),
                           ("policy_id", "wrong"), ("evidence_ready", "yes")):
            data = copy.deepcopy(self.data)
            data["claim"] = app.decode_claim(data["claim"])
            data["claim"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.recommend(data)

    def test_injected_planner_validation(self):
        good = app.recommend(self.data, lambda ctx: [
            "collect_claim_evidence", "check_claim_readiness"])
        self.assertTrue(good["journey_validated"])
        for plan in (["prepare_claim_submission", "check_claim_readiness"],
                     ["review_policy", "review_renewal_options"],
                     ["collect_claim_evidence"] * 2, [], ["unknown", "unknown"]):
            with self.assertRaises(app.ValidationError):
                app.recommend(self.data, lambda ctx: plan)
        def failing(ctx):
            raise RuntimeError("provider-like error")
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data, failing)

    def test_malformed_formats_and_schema(self):
        for payload in (None, [], {}, {"schema_version": True}):
            with self.assertRaises(app.ValidationError):
                app.recommend(payload)
        for text in ("<broken", '<!DOCTYPE x><ACORDSubmission/>',
                     "<ACORDSubmission><Email>x</Email></ACORDSubmission>"):
            data = copy.deepcopy(self.data)
            data["underwriting_submission"] = text
            with self.assertRaises(app.ValidationError):
                app.recommend(data)
        self.data["claim"] += "\nClaimID: duplicate"
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_seeded_synthetic_losses_and_boundaries(self):
        rng = random.Random(107)
        self.data["claim"] = app.decode_claim(self.data["claim"])
        for _ in range(20):
            self.data["claim"]["loss_amount"] = round(rng.uniform(10, 50000), 2)
            self.data["claim"]["loss_date"] = (
                dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(267))).isoformat()
            self.assertEqual(app.recommend(self.data), app.recommend(self.data))
        self.data["claim"]["loss_date"] = self.data["as_of"]
        self.data["claim"]["loss_amount"] = 0.01
        self.assertEqual(app.recommend(self.data)["status"], "ok")

    def test_cli_success_and_errors(self):
        for args, code in ((["example_input.json"], 0), (["missing.json"], 2),
                           (["implementation.py"], 2), ([], 2)):
            with self.subTest(args=args):
                run = subprocess.run([sys.executable, "-B", "implementation.py", *args],
                                     cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(run.returncode, code, run.stderr)
                output = json.loads(run.stdout)
                self.assertEqual(output["status"], "ok" if code == 0 else "error")
                self.assertEqual(run.stderr, "")

    def test_duplicate_json_keys(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"synthetic":true,"synthetic":false}', object_pairs_hook=app.unique_object)


if __name__ == "__main__":
    unittest.main()
