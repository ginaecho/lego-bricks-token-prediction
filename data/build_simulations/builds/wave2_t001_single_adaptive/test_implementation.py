"""Tests use synthetic fixtures only; no network, providers, or temporary files."""

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


def fixture():
    return {
        "schema_version": 1,
        "synthetic": True,
        "profile": {"experience": "beginner", "preference": "guided"},
        "goal": "first_sale",
        "completed_steps": [],
    }


class OnboardingTests(unittest.TestCase):
    def test_normal_beginner(self):
        result = app.onboard(fixture())
        self.assertEqual(result["status"], "ok")
        onboarding = result["onboarding"]
        self.assertEqual(onboarding["next_step"], "account")
        self.assertEqual(onboarding["ready_steps"], ["account"])
        self.assertEqual(len(onboarding["steps"]), 6)
        self.assertEqual(len(onboarding["steps"][0]["instructions"]), 3)
        self.assertTrue(all(step["explanation"] for step in onboarding["steps"]))
        self.assertEqual(onboarding["steps"][-1]["unmet_prerequisites"], ["payout", "listing"])

    def test_experience_and_preference_adaptation(self):
        lengths = {}
        for experience in app.EXPERIENCES:
            for preference in app.PREFERENCES:
                value = fixture()
                value["profile"] = {"experience": experience, "preference": preference}
                result = app.onboard(value)["onboarding"]
                lengths[experience, preference] = len(result["steps"][0]["instructions"])
                self.assertEqual(result["next_step"], "account")
                self.assertEqual(len(result["steps"]), 6)
                self.assertEqual(result["presentation"]["mode"],
                                 "walkthrough" if preference == "guided" else "checklist")
        self.assertGreater(lengths["beginner", "concise"], lengths["advanced", "concise"])
        self.assertGreater(lengths["advanced", "guided"], lengths["advanced", "concise"])

    def test_resume_and_parallel_ready_steps(self):
        value = fixture()
        value["completed_steps"] = ["seller_profile", "account"]
        result = app.onboard(value)["onboarding"]
        self.assertEqual(result["ready_steps"], ["payout", "catalog"])
        self.assertEqual(result["progress"], {"completed": 2, "total": 6})
        self.assertEqual(result["completed_steps"], ["account", "seller_profile"])

    def test_goal_specific_prerequisites(self):
        value = fixture()
        value["goal"] = "manage_catalog"
        result = app.onboard(value)["onboarding"]
        self.assertEqual([step["id"] for step in result["steps"]],
                         ["account", "seller_profile", "catalog"])

    def test_fully_complete(self):
        value = fixture()
        value["completed_steps"] = list(app.STEP_MAP)
        result = app.onboard(value)["onboarding"]
        self.assertTrue(result["complete"])
        self.assertIsNone(result["next_step"])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["ready_steps"], [])
        self.assertEqual(result["progress"], {"completed": 6, "total": 6})

    def test_other_goal_history_is_not_counted(self):
        value = fixture()
        value["goal"] = "manage_catalog"
        value["completed_steps"] = ["account", "seller_profile", "payout"]
        result = app.onboard(value)["onboarding"]
        self.assertEqual(result["progress"], {"completed": 2, "total": 3})
        self.assertEqual(result["next_step"], "catalog")

    def test_invalid_inputs(self):
        invalid = [None, [], "input", {}, {"schema_version": 1}]
        for field, replacement in [
            ("schema_version", True), ("schema_version", 2), ("synthetic", False),
            ("profile", None), ("profile", {"experience": "expert", "preference": "guided"}),
            ("profile", {"experience": "beginner", "preference": []}),
            ("goal", "unknown"), ("completed_steps", "account"),
            ("completed_steps", ["unknown"]), ("completed_steps", [[]]),
            ("completed_steps", ["account", "account"]),
            ("completed_steps", ["publish"]), ("unexpected", 1),
        ]:
            value = fixture()
            value[field] = replacement
            invalid.append(value)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.onboard(value)

    def test_optional_history_and_no_mutation(self):
        value = fixture()
        del value["completed_steps"]
        before = copy.deepcopy(value)
        self.assertEqual(app.onboard(value), app.onboard(value))
        self.assertEqual(value, before)

    def test_every_step_is_topologically_ordered(self):
        steps = app.onboard(fixture())["onboarding"]["steps"]
        seen = set()
        for step in steps:
            self.assertTrue(set(step["prerequisites"]) <= seen)
            seen.add(step["id"])

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, "")
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(len(run.stdout.splitlines()), 1)

    def test_cli_usage_and_file_errors(self):
        for args, expected in [([], "usage_error"), (["a", "b"], "usage_error"),
                               ([str(ROOT / "missing-fixture.json")], "file_error"),
                               ([str(ROOT)], "file_error")]:
            with self.subTest(args=args):
                run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(run.returncode, 2)
                result = json.loads(run.stdout)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error"]["code"], expected)
                self.assertEqual(run.stderr, "")

    def test_cli_json_and_validation_errors(self):
        for text, code in [
            ("{", "invalid_json"), ('{"a":1,"a":2}', "invalid_json"),
            ('{"a":NaN}', "invalid_json"), ("null", "validation_error"),
            (json.dumps({**fixture(), "goal": "bad"}), "validation_error"),
        ]:
            output = io.StringIO()
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(app.main(["synthetic-fixture.json"]), 2)
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error"]["code"], code)

    def test_non_utf8_file_error(self):
        output = io.StringIO()
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        with patch.object(Path, "read_text", side_effect=error), contextlib.redirect_stdout(output):
            self.assertEqual(app.main(["synthetic-fixture.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "file_error")


if __name__ == "__main__":
    unittest.main()
