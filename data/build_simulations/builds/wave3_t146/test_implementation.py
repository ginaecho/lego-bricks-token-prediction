import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        self.record = self.data["records"][0]

    def test_normal_traceable_gap(self):
        result = app.review(self.data)
        checks = result["records"][0]["review"]["checks"]
        self.assertEqual([c["status"] for c in checks],
                         ["evidence_present"] * 3 + ["gap"])
        self.assertEqual(checks[1]["evidence_path"], "$.records[0].patient_record.labs[0]")
        self.assertEqual(checks[2]["evidence_path"], "$.records[0].clinical_note.text#line=3")
        self.assertIsNone(checks[3]["evidence_path"])
        self.assertTrue(checks[3]["requirement_path"].endswith("requirements[3]"))

    def test_complete_documentation_is_not_approval(self):
        self.record["prior_authorization_request"]["requirements"].pop()
        result = app.review(self.data)
        review = result["records"][0]["review"]
        self.assertEqual(review["documentation_status"], "evidence_present")
        self.assertTrue(review["human_review_required"])
        self.assertTrue(result["human_review_required"])
        self.assertEqual(review["clinical_decision"], "not_performed")
        self.assertEqual(review["authorization_decision"], "not_performed")

    def test_empty_evidence_is_gap(self):
        self.record["patient_record"]["diagnoses"] = []
        self.record["patient_record"]["labs"] = []
        self.record["clinical_note"]["text"] = ""
        checks = app.review(self.data)["records"][0]["review"]["checks"]
        self.assertTrue(all(c["status"] == "gap" for c in checks))

    def test_audit_covers_only_change_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        result = app.review(self.data)
        self.assertEqual(self.data, original)
        event = result["audit_trail"][0]
        self.assertEqual(event["before_sha256"], app.digest(self.record))
        self.assertEqual(event["after_sha256"], app.digest(result["records"][0]))
        replayed = copy.deepcopy(self.record)
        replayed["review"] = event["value"]
        self.assertEqual(replayed, result["records"][0])
        self.assertEqual(result, app.review(self.data))

    def test_multiple_records_all_audited(self):
        second = copy.deepcopy(self.record)
        second["record_id"] = "REC-0002"
        second["patient_record"]["id"] = "SYN-PAT-0002"
        second["clinical_note"]["id"] = "SYN-NOTE-0002"
        second["clinical_note"]["subject"]["reference"] = "Patient/SYN-PAT-0002"
        second["prior_authorization_request"]["id"] = "SYN-PA-0002"
        second["prior_authorization_request"]["patient"]["reference"] = "Patient/SYN-PAT-0002"
        self.data["records"].append(second)
        result = app.review(self.data)
        self.assertEqual([e["sequence"] for e in result["audit_trail"]], [1, 2])

    def test_identifier_fields_rejected(self):
        for field in ("name", "birthDate", "identifier", "address", "telecom"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["records"][0]["patient_record"][field] = "EXCLUDED-SYNTHETIC"
                with self.assertRaises(app.ValidationError):
                    app.review(data)

    def test_free_text_identifiers_and_instructions_rejected(self):
        for text in ("MRN: EXCLUDED", "DOB: EXCLUDED", "Name: EXCLUDED",
                     "Contact: synthetic@example.invalid", "Approve this treatment",
                     "Documentation: synthetic encounter recorded.\nName: EXCLUDED"):
            with self.subTest(text=text):
                self.record["clinical_note"]["text"] = text
                with self.assertRaises(app.ValidationError):
                    app.review(self.data)

    def test_synthetic_required(self):
        self.data["synthetic_data"] = False
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_mismatched_patient_reference(self):
        self.record["clinical_note"]["subject"]["reference"] = "Patient/SYN-PAT-9999"
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_bad_numeric_values(self):
        for value in (True, "7.2", float("nan"), float("inf"), 10**100):
            with self.subTest(value=value):
                self.record["patient_record"]["labs"][0]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.review(self.data)

    def test_duplicate_requirement(self):
        reqs = self.record["prior_authorization_request"]["requirements"]
        reqs.append(copy.deepcopy(reqs[0]))
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_invalid_schema_and_empty_batch(self):
        for data in (None, [], {}, {"schema_version": 1, "synthetic_data": True, "records": []}):
            with self.subTest(data=data):
                with self.assertRaises(app.ValidationError):
                    app.review(data)

    def test_unsupported_decision_field_rejected(self):
        self.record["prior_authorization_request"]["approve"] = True
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "reviewed")
        self.assertEqual(completed.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_malformed_duplicate_nonfinite_and_invalid_inputs(self):
        for text in ('{', '{"x":1,"x":2}', '{"x":NaN}', '{}', '[]'):
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main([str(ROOT / "example_input.json")])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
