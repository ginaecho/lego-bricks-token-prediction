import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
import contextlib
import io

import implementation as impl


ROOT = Path(__file__).resolve().parent


def fixture(issues=None):
    return {"schema_version": 1, "synthetic": True, "issues": issues or []}


def issue(identifier="SYN-A", text="good", severity="low"):
    return {"id": identifier, "text": text, "severity": severity}


class SentimentTests(unittest.TestCase):
    def test_normal_and_transparent_evidence(self):
        result = impl.analyze(fixture([issue(text="very good but broken")]))
        sentiment = result["issues"][0]["sentiment"]
        self.assertEqual(sentiment["raw_score"], 0)
        self.assertEqual(sentiment["label"], "neutral")
        self.assertEqual([e["contribution"] for e in sentiment["evidence"]], [2, -2])

    def test_negation_and_punctuation(self):
        for text, expected in [("not very good", -2), ("not. good", 1),
                               ("not not good", 1), ("isn’t good", -1),
                               ("not one two three good", 1)]:
            with self.subTest(text=text):
                self.assertEqual(impl.score_text(text)["raw_score"], expected)

    def test_severity_dominates_sentiment(self):
        result = impl.analyze(fixture([
            issue("SYN-low", "terrible " * 100, "low"),
            issue("SYN-critical", "excellent", "critical"),
        ]))
        self.assertEqual(result["issues"][0]["id"], "SYN-critical")
        self.assertEqual(result["issues"][1]["priority"]["score"], 199)

    def test_ties_and_negative_urgency(self):
        result = impl.analyze(fixture([
            issue("b", "good"), issue("a", "good"), issue("c", "bad")]))
        self.assertEqual([row["id"] for row in result["issues"]], ["c", "a", "b"])
        self.assertEqual([row["priority"]["rank"] for row in result["issues"]], [1, 2, 3])

    def test_empty_unknown_and_case(self):
        self.assertEqual(impl.analyze(fixture())["summary"]["total"], 0)
        for text in ["", "   ", "ordinary words", "你好"]:
            self.assertEqual(impl.score_text(text)["label"], "neutral")
        self.assertEqual(impl.score_text("GREAT!")["raw_score"], 2)

    def test_invalid_inputs(self):
        cases = [None, [], {}, fixture([issue(), issue()]),
                 fixture([issue(severity="urgent")]), fixture([issue(text=1)]),
                 fixture([issue(identifier=" ")]), fixture([issue(identifier=" a")]),
                 fixture([issue(text="x" * 10001)]), fixture([issue(severity=[])]),
                 {**fixture(), "schema_version": True},
                 {**fixture(), "synthetic": False}, {**fixture(), "extra": 1},
                 {**fixture(), "issues": {}}, fixture([{}]),
                 fixture([issue(str(i)) for i in range(1001)])]
        for payload in cases:
            with self.subTest(payload_type=type(payload).__name__):
                with self.assertRaises(impl.ValidationError):
                    impl.analyze(payload)

    def test_deterministic_and_nonmutating(self):
        payload = fixture([issue()])
        original = copy.deepcopy(payload)
        self.assertEqual(impl.analyze(payload), impl.analyze(payload))
        self.assertEqual(payload, original)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["total"], 4)
        self.assertEqual(result["issues"][0]["id"], "SYN-002")
        self.assertEqual(run.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in [[], [str(ROOT / "absent.json")],
                     [str(ROOT / "example_input.json"), "extra"]]:
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for text in ["{", '{"x": NaN}', '{"x":1,"x":2}', "[]",
                     '{"schema_version":1,"synthetic":true,"issues":[{}]}']:
            with patch("builtins.open", mock_open(read_data=text)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = impl.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
