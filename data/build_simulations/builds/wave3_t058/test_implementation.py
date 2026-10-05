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


class GuidedSetupTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_progress(self):
        result = app.run(self.request)
        self.assertEqual(result["progress"]["percent_complete"], 33.33)
        self.assertEqual(result["progress"]["in_progress"], 1)
        self.assertEqual(result["next_steps"], ["shipping"])
        self.assertEqual(result["steps"][2]["blocked_by"], ["shipping"])

    def test_full_completion(self):
        self.request["actions"].extend([
            {"step_id": "shipping", "operation": "complete", "answers": {"region": "Synthetic North"}},
            {"step_id": "launch", "operation": "start"},
            {"step_id": "launch", "operation": "complete"},
        ])
        result = app.run(self.request)
        self.assertTrue(result["progress"]["finished"])
        self.assertEqual(result["progress"]["percent_complete"], 100)
        self.assertEqual(result["next_steps"], [])
        self.assertEqual(len(result["events"]), 6)

    def test_empty_plan(self):
        self.request.update(steps=[], actions=[])
        result = app.run(self.request)
        self.assertTrue(result["progress"]["finished"])
        self.assertEqual(result["progress"]["total"], 0)

    def test_initial_plan_and_forward_references(self):
        self.request["actions"] = []
        self.request["steps"].reverse()
        result = app.run(self.request)
        self.assertEqual(result["next_steps"], ["profile"])
        self.assertEqual(result["progress"]["percent_complete"], 0)

    def test_prerequisites_prevent_start(self):
        self.request["actions"] = [{"step_id": "shipping", "operation": "start"}]
        with self.assertRaisesRegex(app.ValidationError, "prerequisites incomplete"):
            app.run(self.request)

    def test_transition_guards(self):
        for actions in [
            [{"step_id": "launch", "operation": "complete"}],
            [{"step_id": "profile", "operation": "complete", "answers": {"display_name": "X"}}],
            [self.request["actions"][0], self.request["actions"][0]],
            self.request["actions"][:2] + [self.request["actions"][1]],
        ]:
            with self.subTest(actions=actions), self.assertRaises(app.ValidationError):
                app.run(dict(self.request, actions=actions))

    def test_invalid_answer_payloads(self):
        for answers in [{}, {"display_name": " "}, {"display_name": 1},
                        {"display_name": "x", "extra": "y"}, None]:
            request = copy.deepcopy(self.request)
            request["actions"][1]["answers"] = answers
            with self.subTest(answers=answers), self.assertRaises(app.ValidationError):
                app.run(request)

    def test_invalid_graphs(self):
        for prerequisites in [["missing"], ["profile"], ["shipping"], ["launch", "launch"]]:
            request = copy.deepcopy(self.request)
            request["steps"][0]["prerequisites"] = prerequisites
            with self.subTest(prerequisites=prerequisites), self.assertRaises(app.ValidationError):
                app.run(request)

    def test_duplicate_ids(self):
        self.request["steps"].append(copy.deepcopy(self.request["steps"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate step"):
            app.run(self.request)

    def test_shared_schema_rejects_invalid_inputs(self):
        variants = [None, [], {}, dict(self.request, schema_version=True),
                    dict(self.request, unknown=1), dict(self.request, steps={}),
                    dict(self.request, actions=[{"step_id": [], "operation": "start"}]),
                    dict(self.request, actions=[{"step_id": "missing", "operation": "start"}]),
                    dict(self.request, actions=[{"step_id": "profile", "operation": "skip"}])]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run(value)

    def test_determinism_and_input_immutability(self):
        before = copy.deepcopy(self.request)
        first = app.run(self.request)
        self.assertEqual(first, app.run(self.request))
        self.assertEqual(before, self.request)
        first["steps"][0]["answers"]["display_name"] = "Changed"
        self.assertEqual(before, self.request)

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), app.run(self.request))
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "nonexistent-input.json")], ["one", "two"]]:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, cwd=ROOT, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_schema_without_writing_files(self):
        for raw in ["{", "[]", '{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}']:
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=raw), contextlib.redirect_stdout(output):
                code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_unicode_file_error(self):
        output = io.StringIO()
        with patch.object(Path, "read_text", side_effect=UnicodeError("synthetic invalid UTF-8")):
            with contextlib.redirect_stdout(output):
                code = app.main(["synthetic.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
