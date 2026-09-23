"""Synthetic fixtures only; no networks or external dependencies."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from implementation import ValidationError, analyze, sentiment


ROOT = Path(__file__).resolve().parent


def fixture():
    return {
        "schema_version": 1, "dataset_label": "synthetic unit-test fixture",
        "issues": [
            {"id": "a", "text": "Excellent helpful service", "severity": "low", "affected_customers": 2},
            {"id": "b", "text": "Broken and unusable", "severity": "high", "affected_customers": 7},
        ],
    }


class SentimentTests(unittest.TestCase):
    def test_normal_analysis_and_transparency(self):
        result = analyze(fixture())
        self.assertEqual(result["status"], "ok")
        self.assertEqual([item["id"] for item in result["issues"]], ["b", "a"])
        first = result["issues"][0]
        self.assertEqual(first["sentiment"]["label"], "negative")
        self.assertEqual(first["sentiment"]["raw_score"], -5)
        self.assertEqual(first["priority"]["score"], 247)
        self.assertEqual(first["priority"]["rank"], 1)
        self.assertEqual(sum(first["priority"]["components"].values()), 247)

    def test_negation_and_punctuation(self):
        self.assertEqual(sentiment("not good")["score"], -1)
        self.assertEqual(sentiment("not bad")["score"], 1)
        self.assertEqual(sentiment("not. good")["score"], 1)
        self.assertEqual(sentiment("not never good")["score"], 1)
        self.assertEqual(sentiment("isn’t good")["score"], -1)
        self.assertEqual(sentiment("not one two three good")["score"], 1)

    def test_unknown_balanced_and_repeated_terms(self):
        self.assertEqual(sentiment("Package arrived Tuesday")["matched_terms"], 0)
        self.assertEqual(sentiment("Package arrived Tuesday")["label"], "neutral")
        self.assertEqual(sentiment("good bad")["score"], 0)
        self.assertEqual(sentiment("GOOD good")["raw_score"], 2)
        self.assertEqual(sentiment("!!!")["score"], 0)

    def test_severity_always_dominates(self):
        data = fixture()
        data["issues"][0].update(severity="critical", affected_customers=1)
        data["issues"][1].update(affected_customers=1000000)
        result = analyze(data)
        self.assertEqual(result["issues"][0]["id"], "a")
        self.assertEqual(result["issues"][1]["priority"]["score"], 260)

    def test_ties_determinism_and_nonmutation(self):
        data = fixture()
        data["issues"][1] = {**data["issues"][0], "id": "B"}
        before = copy.deepcopy(data)
        first = analyze(data)
        self.assertEqual(first["issues"][0]["id"], "B")
        self.assertEqual(data, before)
        data["issues"].reverse()
        self.assertEqual(first, analyze(data))

    def test_empty_collection(self):
        data = fixture()
        data["issues"] = []
        result = analyze(data)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["summary"]["issue_count"], 0)

    def test_invalid_top_level(self):
        for value in (None, [], {}, {"schema_version": 1}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                analyze(value)
        for field, value in (("schema_version", True), ("schema_version", 2),
                             ("dataset_label", " "), ("issues", {}), ("extra", 1)):
            data = fixture()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                analyze(data)

    def test_invalid_issue_fields(self):
        for field, value in (("id", ""), ("id", " a"), ("text", " "),
                             ("text", 5), ("severity", "urgent"), ("severity", []),
                             ("affected_customers", True), ("affected_customers", 0),
                             ("affected_customers", -1), ("affected_customers", 1.5),
                             ("affected_customers", 1000001), ("extra", "x")):
            data = fixture()
            data["issues"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                analyze(data)
        data = fixture()
        data["issues"][1]["id"] = "a"
        with self.assertRaises(ValidationError):
            analyze(data)

    def test_length_limits(self):
        for field, value in (("id", "a" * 101), ("text", "a" * 10001)):
            data = fixture()
            data["issues"][0][field] = value
            with self.assertRaises(ValidationError):
                analyze(data)
        data = fixture()
        data["issues"] *= 501
        with self.assertRaises(ValidationError):
            analyze(data)

    def run_cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        return result.returncode, json.loads(result.stdout)

    def test_cli_success(self):
        code, result = self.run_cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ok")
        self.assertIn("synthetic", result["dataset_label"].lower())

    def test_cli_file_and_usage_errors(self):
        for args in ((), ("does-not-exist.json",), (str(ROOT),), ("a", "b")):
            with self.subTest(args=args):
                code, result = self.run_cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_malformed_and_invalid_json(self):
        # Existing files exercise syntax and schema errors without scratch files.
        for name in ("implementation.py", "build_manifest.json"):
            code, result = self.run_cli(name)
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "error")

    def test_json_duplicate_and_nonfinite_rejected(self):
        from implementation import reject_constant, unique_object
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValidationError):
                json.loads(value, parse_constant=reject_constant, object_pairs_hook=unique_object)


if __name__ == "__main__":
    unittest.main()
