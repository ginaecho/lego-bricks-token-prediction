"""All fixtures are fictional; no provider, network, or real user data."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

from implementation import ValidationError, evaluate, main


def fixture():
    return {
        "steps": [
            {"id": "fictional-b", "action": "pretend_b", "prerequisites": ["fictional-a"],
             "required_fields": ["fictional_name"], "completion_evidence": ["receipt"]},
            {"id": "fictional-a", "action": "pretend_a", "completion_evidence": ["receipt"]},
            {"id": "fictional-c", "action": "pretend_c"},
        ],
        "progress": {"fields": {}, "completed": {}},
    }


def claim():
    return {"evidence": {"receipt": "FICTIONAL-NOT-REAL"}}


class PlannerTests(unittest.TestCase):
    def test_eligibility_blockers_and_order(self):
        result = evaluate(fixture())
        self.assertEqual(result["topological_plan"], ["fictional-a", "fictional-b", "fictional-c"])
        self.assertEqual(result["eligible_next_steps"], ["fictional-a", "fictional-c"])
        self.assertEqual(result["blocked_reasons"]["fictional-b"], [
            {"code": "missing_prerequisites", "ids": ["fictional-a"]},
            {"code": "missing_required_fields", "fields": ["fictional_name"]},
        ])

    def test_batch_completion_and_input_immutability(self):
        data = fixture()
        data["updates"] = {"fields": {"fictional_name": "Imaginary Quokka"},
                           "complete": {"fictional-b": claim(), "fictional-a": claim()}}
        original = copy.deepcopy(data)
        result = evaluate(data)
        self.assertEqual(result["topological_plan"], ["fictional-c"])
        self.assertEqual(sorted(result["progress"]["completed"]), ["fictional-a", "fictional-b"])
        self.assertEqual(data, original)
        self.assertEqual(evaluate({"steps": data["steps"], "progress": result["progress"]}), result)

    def test_all_completed(self):
        data = fixture()
        data["updates"] = {"fields": {"fictional_name": "Imaginary Quokka"},
                           "complete": {key: claim() for key in
                                        ["fictional-a", "fictional-b", "fictional-c"]}}
        result = evaluate(data)
        self.assertEqual(result["topological_plan"], [])
        self.assertEqual(result["blocked_reasons"], {})
        self.assertEqual(result["eligible_next_steps"], [])

    def test_order_independent(self):
        data = fixture()
        expected = evaluate(data)
        data["steps"].reverse()
        self.assertEqual(evaluate(data), expected)

    def test_unknown_prerequisite(self):
        data = fixture()
        data["steps"][0]["prerequisites"] = ["fictional-unknown"]
        with self.assertRaisesRegex(ValidationError, "unknown prerequisite"):
            evaluate(data)

    def test_cycles_and_self_loop(self):
        for deps in (["fictional-b"], ["fictional-a"]):
            with self.subTest(deps=deps):
                data = fixture()
                data["steps"][1]["prerequisites"] = deps
                with self.assertRaisesRegex(ValidationError, "cycle detected"):
                    evaluate(data)

    def test_duplicate_ids(self):
        data = fixture()
        data["steps"].append(copy.deepcopy(data["steps"][0]))
        with self.assertRaisesRegex(ValidationError, "duplicate ID"):
            evaluate(data)

    def test_unknown_completion_ids(self):
        for location in ("progress", "updates"):
            with self.subTest(location=location):
                data = fixture()
                data[location] = {"completed" if location == "progress" else "complete":
                                  {"fictional-unknown": claim()}}
                with self.assertRaisesRegex(ValidationError, "unknown step IDs"):
                    evaluate(data)

    def test_missing_prerequisite_completion(self):
        data = fixture()
        data["updates"] = {"fields": {"fictional_name": "Imaginary Quokka"},
                           "complete": {"fictional-b": claim()}}
        with self.assertRaisesRegex(ValidationError, "missing completed prerequisites"):
            evaluate(data)

    def test_missing_required_field_completion(self):
        data = fixture()
        data["updates"] = {"complete": {"fictional-a": claim(), "fictional-b": claim()}}
        with self.assertRaisesRegex(ValidationError, "missing required fields"):
            evaluate(data)

    def test_missing_evidence_completion(self):
        data = fixture()
        data["updates"] = {"complete": {"fictional-a": {}}}
        with self.assertRaisesRegex(ValidationError, "missing completion evidence"):
            evaluate(data)

    def test_empty_values_not_completion_evidence(self):
        for value in (None, "", "  ", [], {}):
            with self.subTest(value=value):
                data = fixture()
                data["updates"] = {"complete": {"fictional-a": {"evidence": {"receipt": value}}}}
                with self.assertRaisesRegex(ValidationError, "missing completion evidence"):
                    evaluate(data)

    def test_false_and_zero_are_present_values(self):
        for value in (False, 0):
            with self.subTest(value=value):
                data = fixture()
                data["updates"] = {"complete": {"fictional-a": {"evidence": {"receipt": value}}}}
                self.assertIn("fictional-a", evaluate(data)["progress"]["completed"])

    def test_cannot_invalidate_existing_completion(self):
        data = fixture()
        data["progress"] = {"fields": {"fictional_name": "Imaginary Quokka"},
                            "completed": {"fictional-a": claim(), "fictional-b": claim()}}
        data["updates"] = {"fields": {"fictional_name": None}}
        original = copy.deepcopy(data)
        with self.assertRaisesRegex(ValidationError, "missing required fields"):
            evaluate(data)
        self.assertEqual(data, original)

    def test_invalid_original_progress_cannot_be_rescued_by_updates(self):
        data = fixture()
        data["progress"]["completed"] = {"fictional-a": {}}
        data["updates"] = {"complete": {"fictional-a": claim()}}
        with self.assertRaisesRegex(ValidationError, "progress.completed"):
            evaluate(data)

    def test_callback_wording_and_mutation_isolation(self):
        data = fixture()
        original = copy.deepcopy(data)

        def callback(steps):
            steps[0]["action"] = "tampered"
            steps[1]["prerequisites"].clear()
            return {"fictional-a": {"title": "Imaginary greeting", "description": "Pretend only"}}

        result = evaluate(data, callback)
        self.assertEqual(result["steps"][0]["title"], "Imaginary greeting")
        self.assertEqual(result["steps"][0]["action"], "pretend_a")
        self.assertEqual(result["steps"][1]["prerequisites"], ["fictional-a"])
        self.assertEqual(result["topological_plan"], evaluate(data)["topological_plan"])
        self.assertEqual(data, original)

    def test_reject_invalid_callback_outputs(self):
        outputs = [None, [], {"fictional-unknown": {"title": "Imaginary"}},
                   {"fictional-a": {"action": "tampered"}},
                   {"fictional-a": {"prerequisites": []}},
                   {"fictional-a": {"title": ""}}, {"fictional-a": "Imaginary"}]
        for output in outputs:
            with self.subTest(output=output):
                with self.assertRaises(ValidationError):
                    evaluate(fixture(), lambda steps: output)

    def test_callback_exception_is_validation_error(self):
        def callback(steps):
            raise RuntimeError("fictional failure")
        with self.assertRaisesRegex(ValidationError, "wording callback failed"):
            evaluate(fixture(), callback)

    def test_invalid_shapes(self):
        for data in (None, [], {}, {"steps": {}}, {"steps": [None]},
                     {"steps": [{"id": "fictional-a", "action": " "}]},
                     {"steps": [], "progress": []},
                     {"steps": [], "updates": {"complete": []}},
                     {"steps": [], "typo": True}):
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    evaluate(data)

    def test_bad_list_fields(self):
        for field in ("prerequisites", "required_fields", "completion_evidence"):
            for value in (None, "fictional-a", [3], ["fictional-a", "fictional-a"]):
                with self.subTest(field=field, value=value):
                    data = fixture()
                    data["steps"][0][field] = value
                    with self.assertRaises(ValidationError):
                        evaluate(data)

    def test_empty_steps(self):
        result = evaluate({"steps": []})
        self.assertEqual(result["topological_plan"], [])
        self.assertEqual(result["progress"], {"fields": {}, "completed": {}})

    def test_malformed_claim(self):
        for value in (True, [], {"evidence": []}, {"unexpected": True}):
            with self.subTest(value=value):
                data = fixture()
                data["updates"] = {"complete": {"fictional-a": value}}
                with self.assertRaises(ValidationError):
                    evaluate(data)

    def test_cli_json_input_errors(self):
        for raw in ('{', '{"steps":[],"steps":[]}', '{"steps":[],"progress":{"fields":{"x":NaN}}}'):
            with self.subTest(raw=raw), patch("builtins.open", mock_open(read_data=raw)):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["fictional-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["error"]["code"], "validation_error")

    def test_cli_usage_and_file_errors(self):
        for args in ([], ["a", "b"], ["fictional-file-that-does-not-exist.json"]):
            with self.subTest(args=args):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(args), 2)
                self.assertIn("error", json.loads(output.getvalue()))

    def test_exact_example_cli(self):
        directory = Path(__file__).resolve().parent
        run = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                             cwd=directory, capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, "")
        result = json.loads(run.stdout)
        self.assertEqual(result["eligible_next_steps"], ["fictional-orientation"])
        self.assertEqual(result["topological_plan"], ["fictional-orientation", "fictional-launch"])


if __name__ == "__main__":
    unittest.main()
