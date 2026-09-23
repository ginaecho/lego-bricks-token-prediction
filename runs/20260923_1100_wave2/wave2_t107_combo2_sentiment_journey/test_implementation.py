"""Synthetic fixtures only; no network, provider, or third-party dependencies."""

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


def fixture(text="broken and slow", severity=5, completed=None):
    return {
        "schema_version": 1, "synthetic": True,
        "feedback": [{"id": "synthetic-1", "text": text, "severity": severity}],
        "completed_actions": [] if completed is None else completed,
    }


class PipelineTests(unittest.TestCase):
    def test_transparent_negative_score(self):
        issue = app.sentiment_stage(fixture())["issues"][0]
        self.assertEqual(issue["sentiment"], -1)
        self.assertEqual(issue["priority"], 520)
        self.assertEqual([entry["weight"] for entry in issue["evidence"]], [-2, -1])

    def test_negation_and_punctuation(self):
        self.assertEqual(app.score("not good")[0], -1)
        self.assertEqual(app.score("not bad")[0], 1)
        self.assertEqual(app.score("not. good")[0], 1)
        self.assertEqual(app.score("goodbye")[0], 0)

    def test_neutral_and_mixed(self):
        self.assertEqual(app.score("ordinary package")[0], 0)
        self.assertEqual(app.score("good bad")[0], 0)
        self.assertEqual(app.score("great slow")[0], 0.333333)

    def test_severity_dominates_sentiment(self):
        data = fixture("excellent", 5)
        data["feedback"].append({"id": "synthetic-2", "text": "awful", "severity": 4})
        issues = app.sentiment_stage(data)["issues"]
        self.assertEqual(issues[0]["issue_id"], "synthetic-1")

    def test_deterministic_tie_break(self):
        data = fixture()
        data["feedback"].insert(0, {"id": "synthetic-z", "text": "bad", "severity": 5})
        self.assertEqual(app.sentiment_stage(data)["issues"][0]["issue_id"], "synthetic-1")

    def test_handoff_high_severity(self):
        result = app.run_pipeline(fixture(completed=["acknowledge"]))
        journey = result["journey"]
        self.assertEqual(journey["source_priority"], result["sentiment"]["issues"][0]["priority"])
        self.assertEqual([step["action"] for step in journey["steps"]], ["triage", "investigate"])
        self.assertEqual(journey["steps"][1]["requires"], ["triage"])
        self.assertTrue(all(step["issue_id"] == "synthetic-1" for step in journey["steps"]))
        self.assertEqual(journey["next_actions"], ["triage"])

    def test_low_severity_skips_triage(self):
        result = app.run_pipeline(fixture(severity=2))
        self.assertEqual([step["action"] for step in result["journey"]["steps"]],
                         ["acknowledge", "investigate"])

    def test_tampered_handoff_rejected(self):
        data = fixture()
        insights = app.sentiment_stage(data)
        insights["issues"][0]["severity"] = 1
        with self.assertRaises(app.ValidationError):
            app.journey_stage(insights, data)

    def test_tampered_journey_rejected(self):
        data = fixture()
        result = app.run_pipeline(data)
        result["journey"]["steps"][1]["requires"] = []
        with self.assertRaises(app.ValidationError):
            app.validate("output", result, data)

    def test_boolean_not_accepted_as_stage_integer(self):
        data = fixture()
        insights = app.sentiment_stage(data)
        insights["schema_version"] = True
        with self.assertRaises(app.ValidationError):
            app.journey_stage(insights, data)
        result = app.run_pipeline(data)
        result["journey"]["steps"][0]["step"] = True
        with self.assertRaises(app.ValidationError):
            app.validate("output", result, data)

    def test_empty_feedback(self):
        data = fixture()
        data["feedback"] = []
        result = app.run_pipeline(data)
        self.assertEqual(result["journey"]["reason"], "no_issues")
        self.assertEqual(result["journey"]["steps"], [])

    def test_insufficient_remaining_steps(self):
        for completed in (list(app.ACTIONS[:-1]), list(app.ACTIONS)):
            with self.subTest(completed=completed):
                journey = app.run_pipeline(fixture(completed=completed))["journey"]
                self.assertEqual(journey["status"], "unavailable")
                self.assertEqual(journey["steps"], [])
        self.assertEqual(app.run_pipeline(fixture(completed=list(app.ACTIONS[:-1])))
                         ["journey"]["next_actions"], ["monitor"])

    def test_invalid_inputs(self):
        invalid = [None, [], {"schema_version": 1}]
        for field, value in (("severity", True), ("severity", 0), ("severity", 6),
                             ("severity", 2.5), ("text", " "), ("id", "")):
            data = fixture()
            data["feedback"][0][field] = value
            invalid.append(data)
        for field, value in (("synthetic", False), ("schema_version", True),
                             ("feedback", {}), ("completed_actions", ["resolve"]),
                             ("completed_actions", ["unknown"]),
                             ("completed_actions", ["acknowledge", "acknowledge"])):
            data = fixture()
            data[field] = value
            invalid.append(data)
        duplicate = fixture()
        duplicate["feedback"].append(copy.deepcopy(duplicate["feedback"][0]))
        invalid.append(duplicate)
        extra = fixture()
        extra["unexpected"] = 1
        invalid.append(extra)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "missing-input.json")], ["one", "two"], [str(ROOT)]):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    capture_output=True, text=True, cwd=ROOT, check=False,
                )
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_malformed_json_and_validation(self):
        for content in ('{', '{"a": 1, "a": 2}', '{"a": NaN}', 'null',
                        json.dumps(fixture(severity=0))):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=content), contextlib.redirect_stdout(output):
                    exit_code = app.main(["synthetic-invalid.json"])
                self.assertEqual(exit_code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_no_input_mutation(self):
        data = fixture()
        original = copy.deepcopy(data)
        self.assertEqual(app.run_pipeline(data), app.run_pipeline(data))
        self.assertEqual(data, original)


if __name__ == "__main__":
    unittest.main()
