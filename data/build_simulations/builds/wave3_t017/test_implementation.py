"""All customer data used here is synthetic; tests never write files."""

import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_personalized_next_step(self):
        result = app.onboard(self.payload)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["next_step"]["id"], "complete_profile")
        self.assertIn("Morgan Example", result["next_step"]["instructions"])
        self.assertEqual(result["progress"], {"completed": 1, "total": 3, "percent": 33})

    def test_new_customer_starts_with_verification(self):
        self.payload["completed_steps"] = []
        self.assertEqual(app.onboard(self.payload)["next_step"]["id"], "verify_email")

    def test_each_goal_routes_to_its_own_action(self):
        for goal, sequence in app.GOALS.items():
            with self.subTest(goal=goal):
                self.payload["goal"] = goal
                self.payload["completed_steps"] = list(sequence[:2])
                self.assertEqual(app.onboard(self.payload)["next_step"]["id"], sequence[2])

    def test_completed_onboarding_has_no_next_step(self):
        self.payload["completed_steps"] = list(reversed(app.GOALS["launch_project"]))
        result = app.onboard(self.payload)
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["next_step"])
        self.assertEqual(result["progress"]["percent"], 100)
        self.assertIn("workspace dashboard", result["message"])

    def test_experience_changes_guidance(self):
        beginner = app.onboard(self.payload)
        self.payload["customer"]["experience"] = "experienced"
        expert = app.onboard(self.payload)
        self.assertNotEqual(beginner["next_step"]["guidance"], expert["next_step"]["guidance"])

    def test_deterministic_and_nonmutating(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(app.onboard(self.payload), app.onboard(self.payload))
        self.assertEqual(self.payload, original)

    def test_invalid_top_level_and_fields(self):
        invalids = [None, [], {}, {**self.payload, "unexpected": True}]
        for key, value in [
            ("schema_version", True), ("schema_version", 2),
            ("synthetic_fixture", "true"), ("goal", []),
            ("goal", "unknown"), ("customer", None),
            ("completed_steps", "verify_email"), ("completed_steps", [2]),
            ("completed_steps", ["verify_email", "verify_email"]),
            ("completed_steps", ["create_project"]),
            ("completed_steps", ["take_tour"]),
        ]:
            invalids.append({**self.payload, key: value})
        for payload in invalids:
            with self.subTest(payload=payload), self.assertRaises(app.ValidationError):
                app.onboard(payload)

    def test_invalid_customer_fields(self):
        for key, value in [("id", ""), ("name", " "), ("name", "x" * 121), ("experience", {})]:
            payload = copy.deepcopy(self.payload)
            payload["customer"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.onboard(payload)

    def test_trimmed_name(self):
        self.payload["customer"]["name"] = "  Synthetic Person  "
        self.assertTrue(app.onboard(self.payload)["message"].startswith("Synthetic Person,"))

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), app.onboard(self.payload))
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in [(), ("missing-synthetic-input.json",), ("example_input.json", "extra"),
                     ("test_implementation.py",), (".",)]:
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_invalid_schema_and_json(self):
        for raw in ['{}', '{"x": 1, "x": 2}', '{"x": NaN}', '[', 'null']:
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_error(self):
        with patch.object(Path, "read_text", side_effect=UnicodeError("synthetic invalid encoding")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
