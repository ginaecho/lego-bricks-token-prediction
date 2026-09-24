"""Synthetic fixtures only; tests never create additional files."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture(*messages):
    return {"schema_version": "1.0", "data_label": "SYNTHETIC test fixture",
            "feedback": [{"id": f"F{i:03d}", "text": message}
                         for i, message in enumerate(messages)]}


class InsightTests(unittest.TestCase):
    def test_example(self):
        result = app.analyze(json.loads((ROOT / "example_input.json").read_text("utf-8")))
        self.assertEqual(result["summary"]["total_feedback"], 6)
        self.assertEqual(result["summary"]["sentiment_counts"],
                         {"negative": 2, "neutral": 2, "positive": 2})
        self.assertEqual(result["summary"]["theme_count"], 7)
        self.assertEqual(result["assignments"][0]["theme_ids"], ["delivery", "quality"])
        self.assertTrue(all(theme["recommended_action"] for theme in result["themes"]))

    def test_empty(self):
        result = app.analyze(fixture())
        self.assertEqual(result["themes"], [])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["summary"]["total_feedback"], 0)

    def test_whole_words_and_fallback(self):
        result = app.analyze(fixture("Supporting an event.", "SUPPORT is GREAT!"))
        self.assertEqual(result["assignments"][0]["theme_ids"], ["other"])
        self.assertEqual(result["assignments"][1]["theme_ids"], ["support"])
        self.assertEqual(result["assignments"][1]["sentiment"], "positive")

    def test_custom_dictionary(self):
        value = fixture("Please offer DARK MODE.")
        value["themes"] = [{"id": "display", "name": "Display",
                            "keywords": ["dark mode"], "action": "Validate dark mode demand."}]
        result = app.analyze(value)
        self.assertEqual(result["themes"][0]["id"], "display")
        self.assertEqual(result["themes"][0]["priority_score"], 1)

    def test_sentiment_negation_and_mixed(self):
        result = app.analyze(fixture("Not good.", "Not bad.", "Great but slow."))
        self.assertEqual([row["sentiment"] for row in result["assignments"]],
                         ["negative", "positive", "neutral"])

    def test_determinism_and_no_input_mutation(self):
        value = fixture("Support is slow.", "Quality is great.", "Shipping is late.")
        before = copy.deepcopy(value)
        first = app.analyze(value)
        self.assertEqual(value, before)
        value["feedback"].reverse()
        self.assertEqual(first, app.analyze(value))
        self.assertEqual(first["themes"][0]["priority_score"], 3)

    def test_evidence_bound_and_repeated_text(self):
        result = app.analyze(fixture(*(["Support is slow."] * 5)))
        theme = result["themes"][0]
        self.assertEqual(theme["feedback_count"], 5)
        self.assertEqual(theme["priority_score"], 15)
        self.assertEqual(len(theme["evidence"]), 3)
        self.assertEqual(theme["share_of_feedback"], 1)

    def test_invalid_inputs(self):
        invalid = [
            None, [], {}, {"schema_version": "2.0", "feedback": []},
            {"schema_version": "1.0", "feedback": {}},
            fixture(" "), fixture(42),
            {**fixture(), "unknown": True},
            {**fixture(), "themes": []},
            {**fixture(), "data_label": None},
        ]
        duplicate = fixture("One", "Two")
        duplicate["feedback"][1]["id"] = "F000"
        invalid.append(duplicate)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.analyze(value)

    def test_invalid_theme_definitions(self):
        base = {"id": "custom", "name": "Custom", "keywords": ["term"], "action": "Review."}
        for change in ({"id": "other"}, {"keywords": ["!!!"]},
                       {"keywords": "term"}, {"action": ""}):
            with self.subTest(change=change), self.assertRaises(app.ValidationError):
                app.analyze({**fixture(), "themes": [{**base, **change}]})
        with self.assertRaises(app.ValidationError):
            app.analyze({**fixture(), "themes": [base, base]})

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        process = self.run_cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ((), (str(ROOT / "does-not-exist.json"),), ("one", "two")):
            with self.subTest(args=args):
                process = self.run_cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_invalid_schema(self):
        process = self.run_cli(str(ROOT / "build_manifest.json"))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_malformed_and_nonstandard_file_content(self):
        for content in (b"{", b"\xff", b'{"schema_version":"1.0","feedback":[],"feedback":[]}',
                        b'{"schema_version":"1.0","feedback":NaN}',
                        b" " * (app.MAX_FILE_BYTES + 1)):
            with self.subTest(content=content[:80]):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(content)), \
                        contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
