import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import contextlib
import io

import implementation as app


ROOT = Path(__file__).resolve().parent


class GuidedSetupTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_progress_and_determinism(self):
        original = copy.deepcopy(self.data)
        result = app.run(self.data)
        self.assertEqual(result["progress"], {
            "completed": 2, "total": 3, "percent": 66.67, "finished": False})
        self.assertEqual(result["next_steps"], ["review"])
        self.assertEqual(result, app.run(self.data))
        self.assertEqual(original, self.data)

    def test_initial_guidance(self):
        self.data["actions"] = []
        result = app.run(self.data)
        self.assertEqual(result["next_steps"], ["profile"])
        self.assertEqual(result["steps"][2]["missing_prerequisites"], ["profile", "inventory"])
        self.assertEqual(result["progress"]["percent"], 0)

    def test_complete_and_resume(self):
        checkpoint = app.run(self.data)["state"]["completed"]
        self.data["initial_completed"] = checkpoint
        self.data["actions"] = [{"step_id": "review", "answers": {"confirmed": True}}]
        result = app.run(self.data)
        self.assertTrue(result["progress"]["finished"])
        self.assertEqual(result["progress"]["percent"], 100)
        self.assertEqual(result["next_steps"], [])

    def test_order_is_enforced(self):
        self.data["actions"].reverse()
        with self.assertRaisesRegex(app.ValidationError, "unmet prerequisites"):
            app.run(self.data)

    def test_invalid_answers(self):
        for answers in ({}, {"item_count": True}, {"item_count": "3"},
                        {"item_count": 3, "extra": 1}):
            with self.subTest(answers=answers):
                self.data["actions"][1]["answers"] = answers
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_invalid_schemas(self):
        for value in (None, [], {}, {"schema_version": True, "fixture_label": "x", "steps": []}):
            with self.subTest(value=value):
                with self.assertRaises(app.ValidationError):
                    app.run(value)

    def test_graph_errors(self):
        for prereqs in (["unknown"], ["profile"], ["inventory"], ["inventory", "inventory"]):
            with self.subTest(prereqs=prereqs):
                data = copy.deepcopy(self.data)
                data["steps"][0]["prerequisites"] = prereqs
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_duplicate_completion_and_invalid_checkpoint(self):
        self.data["actions"].append(copy.deepcopy(self.data["actions"][0]))
        with self.assertRaisesRegex(app.ValidationError, "already completed"):
            app.run(self.data)
        self.data["actions"] = []
        self.data["initial_completed"] = ["review"]
        with self.assertRaisesRegex(app.ValidationError, "unmet prerequisite"):
            app.run(self.data)

    def test_optional_fields_and_independent_steps(self):
        self.data["steps"][1]["prerequisites"] = []
        self.data["steps"][1]["fields"]["item_count"]["required"] = False
        self.data["actions"] = [{"step_id": "inventory", "answers": {}}]
        self.assertEqual(app.run(self.data)["state"]["completed"], ["inventory"])

    def test_choices_and_blank_strings(self):
        for name, value in (("display_name", " "), ("market", "invalid")):
            with self.subTest(name=name):
                data = copy.deepcopy(self.data)
                data["actions"][0]["answers"][name] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "nonexistent_input.json")]):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                    capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_validation_errors(self):
        for content in ('{', '{"x":1,"x":2}', 'NaN', 'null', '{"schema_version":1}'):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=content):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic_fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
