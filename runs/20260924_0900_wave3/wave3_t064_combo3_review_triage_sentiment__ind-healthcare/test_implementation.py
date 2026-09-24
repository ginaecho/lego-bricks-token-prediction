import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def result(self):
        return app.run(self.data)["records"][0]

    def test_review_traces_diagnosis_and_lab(self):
        review = self.result()["stages"]["review"]
        self.assertEqual(review["checks"][0]["evidence"][0]["sources"], ["patient.diagnoses[0]"])
        self.assertEqual(review["checks"][1]["evidence"][0]["sources"], ["patient.labs[0]"])
        self.assertEqual(review["gap_ids"], ["referral"])

    def test_all_requirement_terms_required(self):
        self.data["settings"]["requirements"][0]["evidence_terms"].append("unavailable")
        self.assertEqual(self.result()["stages"]["review"]["checks"][0]["missing_terms"], ["unavailable"])

    def test_missing_evidence_is_not_approval(self):
        result = self.result()
        self.assertEqual(result["stages"]["triage"]["decision"], "human_administrative_review_required")
        self.assertEqual(result["payload"]["prior_authorization"]["status"], "draft")

    def test_gap_handoff_and_owner(self):
        stages = self.result()["stages"]
        self.assertEqual(stages["review"]["gap_ids"], stages["triage"]["gap_ids"])
        self.assertEqual(stages["triage"]["gap_ids"], stages["sentiment"]["gap_ids"])
        self.assertEqual(stages["sentiment"]["owner"], "authorization-review-team")
        self.assertEqual(stages["sentiment"]["ticket_id"], stages["triage"]["ticket_id"])

    def test_first_matching_routing_rule(self):
        rules = self.data["settings"]["routing"]["rules"]
        rules.insert(0, {"category": "first", "owner": "first-team", "keywords": ["authorization"]})
        self.assertEqual(self.result()["stages"]["triage"]["owner"], "first-team")

    def test_fallback_routing(self):
        self.data["settings"]["routing"]["rules"] = []
        self.assertEqual(self.result()["stages"]["triage"]["owner"], "intake-review-team")

    def test_urgent_priority(self):
        self.assertEqual(self.result()["stages"]["triage"]["priority"], "urgent")

    def test_gap_priority_without_urgency(self):
        self.data["settings"]["routing"]["urgent_keywords"] = []
        self.assertEqual(self.result()["stages"]["triage"]["priority"], "high")

    def test_complete_evidence_normal_priority(self):
        self.data["settings"]["routing"]["urgent_keywords"] = []
        self.data["records"][0]["prior_authorization"]["evidence"].append("referral attached")
        self.assertEqual(self.result()["stages"]["triage"]["priority"], "normal")

    def test_transparent_sentiment(self):
        result = self.result()["stages"]["sentiment"]
        self.assertEqual(result["score"], -0.5)
        self.assertEqual(result["label"], "negative")
        self.assertEqual(result["priority_score"], 87)
        self.assertEqual(sum(x["weight"] for x in result["contributions"]), -2)

    def test_negation_and_word_boundaries(self):
        self.data["records"][0]["clinical_note"]["text"] = "Not helpful. Never poor. Unclearly."
        result = self.result()["stages"]["sentiment"]
        self.assertEqual(result["score"], 0)
        self.assertEqual(len(result["contributions"]), 2)

    def test_neutral_no_matches(self):
        self.data["records"][0]["clinical_note"]["text"] = "Synthetic administrative paperwork."
        self.assertEqual(self.result()["stages"]["sentiment"]["score"], 0)

    def test_severity_dominates_positive_sentiment(self):
        self.data["records"][0]["clinical_note"]["text"] = "Expedited helpful clear excellent."
        result = self.result()["stages"]["sentiment"]
        self.assertEqual(result["score"], 1)
        self.assertGreaterEqual(result["priority_score"], 80)

    def test_no_clinical_interpretation(self):
        baseline = self.result()["stages"]["triage"]["priority"]
        self.data["records"][0]["patient"]["labs"][0]["value"] = 999
        result = self.result()
        self.assertEqual(result["stages"]["triage"]["priority"], baseline)
        self.assertTrue(result["safety"]["human_review_required"])
        self.assertFalse(result["safety"]["autonomous_clinical_decision"])

    def test_structured_identifiers_rejected(self):
        for field, value in [("name", "Synthetic Person"), ("birthDate", "1980-01-01"),
                             ("identifier", "real-mrn"), ("address", "road")]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["records"][0]["patient"][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_free_text_identifier_patterns_rejected(self):
        for text in ["MRN 12345", "DOB 1980-01-01", "person@example.invalid",
                     "123-45-6789", "555-123-4567", "patient name: Example"]:
            with self.subTest(text=text):
                self.data["records"][0]["clinical_note"]["text"] = text
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_synthetic_label_required(self):
        self.data["records"][0]["patient"]["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_non_synthetic_patient_id_rejected(self):
        self.data["records"][0]["patient"]["id"] = "mrn-123"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_audit_all_changes_and_deterministic_output(self):
        result = self.result()
        self.assertEqual([event["action"] for event in result["audit"]],
                         ["ingest", "review", "triage", "sentiment"])
        for before, after in zip(result["audit"], result["audit"][1:]):
            self.assertEqual(before["after_sha256"], after["before_sha256"])
        self.assertEqual(self.result(), result)

    def test_tampering_rejected_at_handoff(self):
        output = app.advance(app.ingest(self.data), "review")
        output["records"][0]["stages"]["review"]["gap_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.advance(output, "triage")

    def test_removed_audit_rejected(self):
        output = app.advance(app.ingest(self.data), "review")
        output["records"][0]["audit"].pop()
        with self.assertRaises(app.ValidationError):
            app.advance(output, "triage")

    def test_removed_human_review_flag_rejected(self):
        output = app.ingest(self.data)
        output["records"][0]["safety"]["human_review_required"] = False
        with self.assertRaises(app.ValidationError):
            app.advance(output, "review")

    def test_stage_cannot_be_skipped(self):
        with self.assertRaises(app.ValidationError):
            app.advance(app.ingest(self.data), "sentiment")

    def test_input_not_mutated(self):
        original = copy.deepcopy(self.data)
        app.run(self.data)
        self.assertEqual(self.data, original)

    def test_multiple_records_isolated(self):
        second = copy.deepcopy(self.data["records"][0])
        second["id"] = "record-two"
        second["clinical_note"]["text"] = "Synthetic helpful staff."
        self.data["records"].append(second)
        results = app.run(self.data)["records"]
        self.assertNotEqual(results[0]["stages"]["sentiment"]["score"],
                            results[1]["stages"]["sentiment"]["score"])
        self.assertNotEqual(results[0]["stages"]["triage"]["ticket_id"],
                            results[1]["stages"]["triage"]["ticket_id"])

    def test_duplicate_records_and_empty_input(self):
        self.data["records"] *= 2
        with self.assertRaises(app.ValidationError):
            app.run(self.data)
        self.data["records"] = []
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_values(self):
        for value in [float("nan"), float("inf"), True, "7.2"]:
            with self.subTest(value=value):
                self.data["records"][0]["patient"]["labs"][0]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_configuration_missing_owner_rejected(self):
        del self.data["settings"]["routing"]["rules"][0]["owner"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_empty_requirement_terms_rejected(self):
        self.data["settings"]["requirements"][0]["evidence_terms"] = []
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["phase"], "sentiment")
        self.assertEqual(process.stderr, "")

    def test_cli_file_error_and_usage(self):
        for args in [[], ["absent-input.json"]]:
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                     capture_output=True, text=True, cwd=ROOT, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_and_invalid_json_no_reflection(self):
        for raw in ['{', '{"synthetic":true,"synthetic":false}', 'NaN', '[]',
                    '{"private":"MRN secret"}']:
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(output):
                result = app.main(["virtual-input.json"])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
            self.assertNotIn("secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
