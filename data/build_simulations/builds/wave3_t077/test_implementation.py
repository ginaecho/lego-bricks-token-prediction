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

    def test_complete_pipeline(self):
        result = app.run(self.raw)
        self.assertEqual(result["stage"], "documents")
        self.assertEqual(len(result["documents"]), 2)
        self.assertIs(app.validate(result), result)

    def test_insight_themes_and_evidence(self):
        result = app.customer_insights(app.normalize(self.raw))
        themes = {t["theme"]: t for t in result["insights"][0]["themes"]}
        self.assertEqual(themes["water"]["evidence"], ["SYN-F01"])
        self.assertEqual(themes["speed"]["count"], 1)
        self.assertIn("Review", themes["clarity"]["action"])

    def test_synonym_search(self):
        result = app.run(self.raw)
        self.assertIn("motor", result["search"][1]["intent_terms"])
        self.assertEqual(result["search"][1]["matches"][0]["policy_id"], "SYN-P02")
        self.assertIn("theft", result["search"][1]["matches"][0]["matched_terms"])

    def test_cross_stage_feedback_propagates(self):
        self.raw["queries"][0]["text"] = "unmatched gibberish"
        result = app.run(self.raw)
        self.assertIn("water", result["search"][0]["insight_themes"])
        self.assertIn("water", result["documents"][0]["customer_themes"])
        self.assertIn("SYN-P01", result["documents"][0]["suggested_policy_ids"])
        self.raw["feedback"] = []
        without = app.run(self.raw)
        self.assertEqual(without["documents"][0]["suggested_policy_ids"], [])

    def test_actual_policy_not_suggestion_controls_claim(self):
        self.raw["queries"][0]["text"] = "motor vehicle theft collision"
        self.raw["feedback"] = []
        result = app.run(self.raw)
        self.assertEqual(result["documents"][0]["suggested_policy_ids"][0], "SYN-P02")
        self.assertEqual(result["documents"][0]["policy_id"], "SYN-P01")
        self.assertEqual(result["documents"][0]["assessment"], "preliminary_match")

    def test_reasoned_human_review(self):
        doc = app.run(self.raw)["documents"][1]
        self.assertEqual(doc["assessment"], "human_review")
        self.assertTrue(doc["human_decision_required"])
        self.assertIn("exceeds", doc["reasons"][0])

    def test_no_automated_denial_for_unlisted_peril(self):
        self.raw["claim_forms"][0] = self.raw["claim_forms"][0].replace("Peril: water", "Peril: earthquake")
        doc = app.run(self.raw)["documents"][0]
        self.assertEqual(doc["assessment"], "human_review")
        self.assertIn("wording and evidence", doc["reasons"][0])

    def test_date_outside_policy(self):
        self.raw["policies"][0]["end_date"] = "2025-01-01"
        doc = app.run(self.raw)["documents"][0]
        self.assertTrue(any("outside" in r for r in doc["reasons"]))

    def test_disclosed_factors_reach_documents(self):
        doc = app.run(self.raw)["documents"][0]
        self.assertEqual(doc["disclosed_underwriting_factors"], self.raw["submissions"][0]["data"]["factors"])

    def test_hidden_factors_rejected(self):
        self.raw["submissions"][0]["data"]["factors"][0]["disclosed"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_protected_factor_rejected(self):
        self.raw["submissions"][0]["data"]["factors"][0]["name"] = "ethnicity"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_data_minimization(self):
        self.raw["feedback"][0]["text"] += " Contact imaginary.person@example.invalid"
        rendered = json.dumps(app.run(self.raw))
        for forbidden in ("Imaginary Harbor", "SYNTHETIC-VIN", "example.invalid", "fixture_address", "fixture_vin"):
            self.assertNotIn(forbidden, rendered)
        self.raw["policies"][0]["email"] = "imaginary.person@example.invalid"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_xml_json_equivalence(self):
        xml_value = app.parse_submission(self.raw["submissions"][1])
        self.assertEqual(app.parse_submission({"format": "json", "data": xml_value}), xml_value)

    def test_xml_entities_rejected(self):
        self.raw["submissions"][1]["data"] = '<!DOCTYPE a [<!ENTITY a "x">]><ACORDSubmission/>'
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_malformed_xml_rejected(self):
        self.raw["submissions"][1]["data"] = "<ACORDSubmission>"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_duplicate_claim_field_rejected(self):
        self.raw["claim_forms"][0] += "\nPeril: fire"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_nonfinite_negative_and_boolean_amounts_rejected(self):
        for amount in ("NaN", "Infinity", "-1"):
            raw = copy.deepcopy(self.raw)
            lines = raw["claim_forms"][0].splitlines()
            raw["claim_forms"][0] = "\n".join("Loss-Amount: " + amount if l.startswith("Loss-Amount:") else l for l in lines)
            with self.subTest(amount=amount), self.assertRaises(app.ValidationError):
                app.run(raw)
        normalized = app.normalize(self.raw)
        normalized["claims"][0]["loss_amount"] = True
        with self.assertRaises(app.ValidationError):
            app.validate(normalized)

    def test_invalid_calendar_date(self):
        self.raw["policies"][0]["start_date"] = "2025-02-30"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_unknown_reference(self):
        self.raw["feedback"][0]["claim_id"] = "SYN-NONEXISTENT"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_duplicate_policy(self):
        self.raw["policies"].append(copy.deepcopy(self.raw["policies"][0]))
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_missing_submission(self):
        self.raw["submissions"] = []
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_missing_query(self):
        self.raw["queries"].pop()
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_stage_order_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.smart_search(app.normalize(self.raw))

    def test_tampered_handoffs_rejected(self):
        insights = app.customer_insights(app.normalize(self.raw))
        insights["insights"][0]["themes"][0]["count"] = 100
        with self.assertRaises(app.ValidationError):
            app.smart_search(insights)
        search = app.smart_search(app.customer_insights(app.normalize(self.raw)))
        search["search"][0]["matches"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.automate_documents(search)
        documents = app.run(self.raw)
        documents["documents"][0]["reasons"] = []
        with self.assertRaises(app.ValidationError):
            app.validate(documents)

    def test_empty_input_collections(self):
        for key in ("policies", "claim_forms", "submissions", "feedback", "queries"):
            self.raw[key] = []
        self.assertEqual(app.run(self.raw)["documents"], [])

    def test_empty_query_rejected(self):
        self.raw["queries"][0]["text"] = " "
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_determinism_and_no_mutation(self):
        original = copy.deepcopy(self.raw)
        self.assertEqual(app.run(self.raw), app.run(self.raw))
        self.assertEqual(self.raw, original)

    def test_seeded_random_synthetic_claims(self):
        rng = random.Random(77)
        for _ in range(12):
            day = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(365))
            amount = round(rng.uniform(0, 35000), 2)
            self.raw["claim_forms"][0] = (
                f"Claim-ID: SYN-C01\nPolicy-ID: SYN-P01\nLoss-Date: {day}\n"
                f"Loss-Amount: {amount}\nPeril: water"
            )
            doc = app.run(self.raw)["documents"][0]
            self.assertEqual(doc["assessment"], "human_review" if amount > 25000 else "preliminary_match")

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                               str(HERE / "example_input.json")], capture_output=True, text=True, cwd=HERE)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in ([], ["missing-input.json"]):
            proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                  capture_output=True, text=True, cwd=HERE)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_invalid_json_and_input_without_scratch_files(self):
        for payload in ("{not-json", '{"synthetic": false}', 'null'):
            output = io.StringIO()
            with patch("builtins.open", return_value=io.StringIO(payload)), redirect_stdout(output):
                code = app.main(["invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
