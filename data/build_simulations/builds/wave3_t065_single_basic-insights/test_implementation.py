import contextlib
import copy
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from implementation import ValidationError, analyze, main


ROOT = Path(__file__).resolve().parent


class InsightsTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_themes_and_priority(self):
        result = analyze(self.request)
        self.assertEqual(result["summary"]["feedback_count"], 5)
        self.assertEqual(result["summary"]["sentiment_counts"],
                         {"negative": 3, "neutral": 1, "positive": 1})
        theme = result["themes"][0]
        self.assertEqual(theme["name"], "performance")
        self.assertEqual(theme["priority_score"], 7)
        self.assertEqual(theme["share_of_feedback"], 0.6)
        self.assertEqual([x["feedback_id"] for x in theme["evidence"]],
                         ["demo-001", "demo-004"])

    def test_empty(self):
        self.request["feedback"] = []
        result = analyze(self.request)
        self.assertEqual(result["themes"], [])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["summary"]["feedback_count"], 0)

    def test_custom_phrase_boundary_and_multiple_themes(self):
        self.request["theme_rules"] = [
            {"name": "delivery", "keywords": ["late delivery"], "action": "Check shipment times."},
            {"name": "speed", "keywords": ["slow"], "action": "Measure speed."}]
        self.request["feedback"] = [
            {"id": "a", "source": "synthetic", "text": "LATE   delivery, slow!"},
            {"id": "b", "source": "synthetic", "text": "A slowdown."}]
        result = analyze(self.request)
        self.assertEqual(result["assignments"][0]["themes"], ["delivery", "speed"])
        self.assertEqual(result["assignments"][1]["themes"], ["uncategorized"])

    def test_sentiment_fallback_and_rating_override(self):
        self.request["feedback"] = [
            {"id": "a", "source": "synthetic", "text": "great helpful"},
            {"id": "b", "source": "synthetic", "text": "great helpful", "rating": 1},
            {"id": "c", "source": "synthetic", "text": "great bad"}]
        self.assertEqual([a["sentiment"] for a in analyze(self.request)["assignments"]],
                         ["positive", "negative", "neutral"])

    def test_deterministic_no_mutation(self):
        original = copy.deepcopy(self.request)
        result = analyze(self.request)
        self.assertEqual(original, self.request)
        self.request["feedback"].reverse()
        self.assertEqual(result, analyze(self.request))

    def test_duplicate_ids(self):
        self.request["feedback"].append(self.request["feedback"][0])
        with self.assertRaises(ValidationError):
            analyze(self.request)

    def test_invalid_fields(self):
        for key, value in [("schema_version", True), ("synthetic", "yes"),
                           ("max_evidence", 0), ("feedback", None),
                           ("theme_rules", []), ("dataset_label", " ")]:
            with self.subTest(key=key):
                request = copy.deepcopy(self.request)
                request[key] = value
                with self.assertRaises(ValidationError):
                    analyze(request)
        for rating in (True, 0, 6, 1.5, None):
            with self.subTest(rating=rating):
                self.request["feedback"][0]["rating"] = rating
                with self.assertRaises(ValidationError):
                    analyze(self.request)

    def test_invalid_rule_names_and_unknown_fields(self):
        rule = {"name": "One", "keywords": ["x"], "action": "Inspect."}
        for rules in ([rule, dict(rule, name=" one ")],
                      [dict(rule, name="uncategorized")],
                      [dict(rule, keywords=["!!!"])]):
            self.request["theme_rules"] = rules
            with self.assertRaises(ValidationError):
                analyze(self.request)
        with self.assertRaises(ValidationError):
            analyze(dict(self.request, extra=True))

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), analyze(self.request))
        self.assertEqual(proc.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "nonexistent.json")], ["one", "two"]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_malformed_json_and_validation(self):
        for body in ("{", "[]", '{"schema_version": NaN}', '{"x":1,"x":2}',
                     json.dumps(dict(self.request, max_evidence=-1))):
            with self.subTest(body=body), patch.object(Path, "read_text", return_value=body):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["synthetic-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_error(self):
        with patch.object(Path, "read_text",
                          side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["synthetic-in-memory.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
