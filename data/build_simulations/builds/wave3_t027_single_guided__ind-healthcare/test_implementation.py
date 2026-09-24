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


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_complete_setup_keeps_human_decisions_pending(self):
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["progress_percent"], 100)
        self.assertTrue(result["onboarding"]["setup_complete"])
        self.assertTrue(result["onboarding"]["human_review_required"])
        self.assertFalse(result["onboarding"]["autonomous_clinical_decisions"])
        self.assertEqual(result["active_records"]["prior_authorization_request"]["status"], "pending-human-review")

    def test_empty_actions(self):
        self.data["actions"] = []
        result = app.run(self.data)
        self.assertEqual(result["active_records"], {})
        self.assertEqual(result["audit_trail"], [])
        self.assertEqual(result["onboarding"]["progress_percent"], 0)
        self.assertEqual(result["onboarding"]["next_step"], "check_prerequisites")

    def test_partial_progress_and_replay(self):
        self.data["actions"] = list(app.STEPS[:2])
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["progress_percent"], 40)
        self.assertEqual(result["onboarding"]["next_step"], "attach_clinical_note")
        self.data["actions"] += list(app.STEPS[2:])
        self.assertTrue(app.run(self.data)["onboarding"]["setup_complete"])

    def test_out_of_order(self):
        self.data["actions"] = ["attach_clinical_note"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_prerequisite_failures(self):
        for key in ("reviewer_available", "deidentification_training_complete"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["prerequisites"][key] = False
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_idempotent_duplicate_actions(self):
        expected = app.run(self.data)
        self.data["actions"] = [step for step in app.STEPS for _ in range(2)]
        result = app.run(self.data)
        self.assertEqual(result["audit_trail"], expected["audit_trail"])
        self.assertEqual(result["onboarding"], expected["onboarding"])

    def test_audit_reconstructs_every_record_change(self):
        result = app.run(self.data)
        state = {}
        self.assertEqual(len(result["audit_trail"]), 4)
        for seq, event in enumerate(result["audit_trail"], 1):
            self.assertEqual(event["sequence"], seq)
            record_id = event["record_id"]
            self.assertEqual(state.get(record_id), event["before"])
            state[record_id] = event["after"]
        self.assertEqual(state, {r["id"]: r for r in result["active_records"].values()})
        self.assertEqual(result["audit_trail"][-1]["before"]["status"], "draft")

    def test_deterministic_without_mutating_input(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, original)

    def test_direct_identifiers_rejected(self):
        for field in ("name", "birthDate", "identifier", "mrn", "address", "telecom"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["records"]["patient_record"][field] = "PROHIBITED-FIXTURE"
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_uncontrolled_notes_rejected(self):
        for text in ("Patient MRN: 12345", "SYNTHETIC: Start treatment immediately.", ""):
            self.data["records"]["clinical_note"]["text"] = text
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_review_flag_and_autonomous_decisions_rejected(self):
        for key in ("clinical_note", "prior_authorization_request"):
            data = copy.deepcopy(self.data)
            data["records"][key]["human_review_required"] = False
            with self.assertRaises(app.ValidationError):
                app.run(data)
        self.data["records"]["prior_authorization_request"]["status"] = "approved"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_reference_mismatch(self):
        self.data["records"]["clinical_note"]["subject"] = "Patient/syn-patient-002"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_lab_values(self):
        for value in (True, "95", -1, 10001, 10 ** 400, float("nan"), float("inf")):
            self.data["records"]["patient_record"]["lab_values"][0]["value"] = value
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_wrong_schema_and_actions(self):
        for key, value in (("schema_version", True), ("synthetic_fixture", False), ("actions", ["approve"]), ("actions", "register_patient")):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.assertRaises(app.ValidationError):
                app.run(data)

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(completed.stderr, "")

    def test_cli_file_error_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_malformed_json_and_validation_errors(self):
        invalid = copy.deepcopy(self.data)
        invalid["actions"] = ["request_human_review"]
        for content in ("{", "null", '{"x":1,"x":2}', '{"x":NaN}', json.dumps(invalid)):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(app.main(["synthetic-input.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
