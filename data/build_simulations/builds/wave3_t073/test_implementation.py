"""Synthetic fixtures only; tests create no files."""

import copy
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
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, text=True, capture_output=True, check=False)

    def test_personalized_next_step(self):
        result = app.onboard(self.payload)
        self.assertEqual(result["next_step"]["id"], "draft")
        self.assertIn("Morgan Example", result["message"])
        self.assertIn(self.payload["customer"]["goal"], result["message"])
        self.assertIn("one step at a time", result["next_step"]["guidance"])
        self.assertEqual(result["progress"], {"completed": 1, "total": 3, "percent": 33.33})

    def test_defaults_and_experience(self):
        del self.payload["steps"]
        self.payload["customer"]["experience"] = "experienced"
        result = app.onboard(self.payload)
        self.assertEqual(result["next_step"]["id"], "setup")
        self.assertIn("existing workflow", result["next_step"]["guidance"])

    def test_complete(self):
        for step in self.payload["steps"]:
            step["completed"] = True
        result = app.onboard(self.payload)
        self.assertEqual(result["onboarding_status"], "complete")
        self.assertIsNone(result["next_step"])
        self.assertEqual(result["progress"]["percent"], 100)

    def test_dependency_order_and_tie_break(self):
        steps = self.payload["steps"]
        self.payload["steps"] = [steps[2], steps[1], steps[0]]
        self.assertEqual(app.onboard(self.payload)["next_step"]["id"], "draft")
        self.payload["steps"][0]["requires"] = ["profile"]
        self.assertEqual(app.onboard(self.payload)["next_step"]["id"], "publish")

    def test_deterministic_without_mutation(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(app.onboard(self.payload), app.onboard(self.payload))
        self.assertEqual(self.payload, original)

    def test_invalid_schema(self):
        for value in (None, [], {}, {"schema_version": True, "customer": {}}):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.onboard(value)
        for field, value in (("name", " "), ("goal", 1), ("experience", "expert")):
            payload = copy.deepcopy(self.payload)
            payload["customer"][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.onboard(payload)

    def test_invalid_steps(self):
        for transform in (
            lambda p: p.update(steps=[]),
            lambda p: p.update(unrecognized=True),
            lambda p: p["steps"][1].update(id="profile"),
            lambda p: p["steps"][1].update(completed="false"),
            lambda p: p["steps"][1].update(requires=["missing"]),
            lambda p: p["steps"][1].update(requires=["profile", "profile"]),
            lambda p: p["steps"][1].update(requires=["publish"]),
            lambda p: p["steps"][2].update(completed=True),
        ):
            payload = copy.deepcopy(self.payload)
            transform(payload)
            with self.subTest(payload=payload), self.assertRaises(app.ValidationError):
                app.onboard(payload)

    def test_cli_success(self):
        run = self.cli("example_input.json")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), app.onboard(self.payload))
        self.assertEqual(run.stderr, "")
        self.assertEqual(len(run.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in ((), ("nonexistent-synthetic-input.json",), ("a", "b")):
            with self.subTest(args=args):
                run = self.cli(*args)
                self.assertEqual(run.returncode, 2)
                self.assertEqual(json.loads(run.stdout)["status"], "error")
                self.assertEqual(run.stderr, "")

    def test_malformed_json_and_validation_cli_boundary(self):
        from contextlib import redirect_stdout
        from io import StringIO
        for content in ('{', '{"schema_version":1,"schema_version":1}',
                        '{"schema_version":NaN}', '{}'):
            output = StringIO()
            with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
