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


ROOT = Path(__file__).parent


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_guided_blocked(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["onboarding"]["state"], "blocked")
        self.assertEqual([x["id"] for x in result["onboarding"]["steps"][:2]],
                         list(app.PREREQUISITES))
        self.assertTrue(all(x["state"] == "blocked" for x in result["onboarding"]["steps"][2:]))
        self.assertTrue(all("guidance" in x and x["explanation"] for x in result["onboarding"]["steps"]))

    def test_ready_is_not_clinical_approval(self):
        self.data["onboarding"]["completed"] = list(app.PREREQUISITES)
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["state"], "ready_for_human_review")
        for record in result["records"].values():
            self.assertTrue(record["workflow"]["human_review_required"])
            self.assertEqual(record["workflow"]["clinical_decision"], "not_performed")
        self.assertEqual(result["records"]["prior_authorization_request"]["status"], "draft")

    def test_experience_and_preferences(self):
        self.data["onboarding"].update(experience="experienced", detail="concise", format="text")
        steps = app.run(self.data)["onboarding"]["steps"]
        self.assertTrue(all("guidance" not in x and x["presentation"] == "text" for x in steps))
        self.data["onboarding"]["detail"] = "guided"
        self.assertTrue(all("guidance" in x for x in app.run(self.data)["onboarding"]["steps"]))

    def test_partial_prerequisites(self):
        self.data["onboarding"]["completed"] = ["synthetic-data"]
        result = app.run(self.data)
        self.assertEqual(result["onboarding"]["missing_prerequisites"], ["human-review"])
        self.assertEqual(result["onboarding"]["steps"][0]["state"], "completed")

    def test_every_change_audited_and_input_preserved(self):
        before = copy.deepcopy(self.data)
        result = app.run(self.data)
        self.assertEqual(self.data, before)
        self.assertEqual(len(result["audit_trail"]), 3)
        for event in result["audit_trail"]:
            key = event["path"].split("/")[2]
            changed = result["records"][key]
            self.assertEqual(event["after"], changed["workflow"])
            self.assertIsNone(event["before"])
            self.assertEqual({k: v for k, v in changed.items() if k != "workflow"},
                             before["records"][key])
        self.assertEqual(result, app.run(self.data))

    def test_identifiers_rejected(self):
        for field in ("name", "birthDate", "identifier", "MRN", "address", "telecom"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["records"]["patient_record"][field] = "FORBIDDEN_SYNTHETIC_MARKER"
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_unrestricted_note_rejected(self):
        self.data["records"]["clinical_note"]["text"] += " Extra identifying prose"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_autonomous_decision_rejected(self):
        self.data["records"]["prior_authorization_request"]["status"] = "approved"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_reference_mismatch(self):
        self.data["records"]["clinical_note"]["subject"]["reference"] = "Patient/anon-999"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_empty_clinical_data(self):
        patient = self.data["records"]["patient_record"]
        patient.update(diagnoses=[], labs=[])
        self.data["records"]["clinical_note"]["text"] = app.canonical_note(patient)
        self.assertEqual(app.run(self.data)["status"], "ok")

    def test_invalid_lab_values(self):
        for value in (True, float("nan"), float("inf"), "5.2", 1000001):
            with self.subTest(value=value):
                self.data["records"]["patient_record"]["labs"][0]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_invalid_schema_and_preferences(self):
        for change in ({"schema_version": "2"}, {"synthetic": False}, {"unexpected": 1}):
            data = copy.deepcopy(self.data)
            data.update(change)
            with self.assertRaises(app.ValidationError):
                app.run(data)
        for completion in (["unknown"], ["human-review", "human-review"], "human-review", [1]):
            self.data["onboarding"]["completed"] = completion
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "missing.json")], ["a", "b"]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_malformed_duplicate_and_invalid_json_cli(self):
        for text in ('{', '{"synthetic":true,"synthetic":false}', 'null', '[]',
                     json.dumps(dict(self.data, synthetic=False))):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=text)), contextlib.redirect_stdout(output):
                code = app.main(["unused.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
