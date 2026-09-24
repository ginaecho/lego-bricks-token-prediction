"""All records in these tests are synthetic."""

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


def fixture(*texts):
    return {"schema_version": 1, "dataset_label": "SYNTHETIC test fixture",
            "feedback": [{"id": "syn-%d" % i, "text": text} for i, text in enumerate(texts)]}


class FeedbackTests(unittest.TestCase):
    def cli(self, *args):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.stderr, "")
        return process.returncode, json.loads(process.stdout)

    def test_example_deduplication_and_counts(self):
        data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        result = app.analyze(data)
        self.assertEqual(result["summary"],
                         {"input_count": 6, "unique_count": 5, "duplicate_count": 1})
        delivery = next(t for t in result["themes"] if t["theme_id"] == "delivery")
        self.assertEqual(delivery["unique_feedback_count"], 2)
        self.assertEqual(delivery["source_count"], 3)

    def test_normalized_duplicates(self):
        result = app.analyze(fixture("  ＤＥＬＩＶＥＲＹ late! ", "delivery LATE."))
        self.assertEqual(result["summary"]["unique_count"], 1)
        self.assertEqual(result["themes"][0]["source_count"], 2)

    def test_multiple_themes_and_exact_excerpts(self):
        data = fixture("😀 " + "x " * 30 + "broken product and helpful support")
        result = app.analyze(data)
        self.assertEqual({t["theme_id"] for t in result["themes"]}, {"product_quality", "support"})
        for theme in result["themes"]:
            for excerpt in theme["excerpts"]:
                self.assertEqual(excerpt["text"],
                                 data["feedback"][0]["text"][excerpt["start"]:excerpt["end"]])

    def test_unicode_compatibility_token_boundaries(self):
        result = app.analyze(fixture("delivery℀", "deliverya/c"))
        self.assertEqual(result["summary"]["unique_count"], 1)
        self.assertEqual([t["theme_id"] for t in result["themes"]], ["other"])
        result = app.analyze(fixture("℀delivery", "a/cdelivery"))
        self.assertEqual(result["summary"]["unique_count"], 1)
        self.assertEqual([t["theme_id"] for t in result["themes"]], ["other"])

    def test_whole_word_match_and_other(self):
        result = app.analyze(fixture("The template supports colorful artwork."))
        self.assertEqual([t["theme_id"] for t in result["themes"]], ["other"])

    def test_empty(self):
        result = app.analyze(fixture())
        self.assertEqual(result["groups"], [])
        self.assertEqual(result["themes"], [])
        self.assertEqual(result["summary"]["input_count"], 0)

    def test_deterministic_and_no_input_mutation(self):
        data = fixture("support helpful", "quality durable")
        before = copy.deepcopy(data)
        self.assertEqual(app.analyze(data), app.analyze(data))
        self.assertEqual(data, before)

    def test_invalid_shapes_and_values(self):
        invalid = [None, [], {}, fixture("!!!"), fixture(" "), fixture("x" * 10001)]
        for field, value in (("schema_version", True), ("schema_version", 2),
                             ("dataset_label", ""), ("feedback", {})):
            data = fixture("fine")
            data[field] = value
            invalid.append(data)
        for value in invalid:
            with self.subTest(value=str(value)[:80]), self.assertRaises(app.ValidationError):
                app.analyze(value)

    def test_unknown_fields_and_repeated_ids(self):
        data = fixture("late", "broken")
        data["feedback"][1]["id"] = data["feedback"][0]["id"]
        with self.assertRaises(app.ValidationError):
            app.analyze(data)
        data = fixture("late")
        data["feedback"][0]["rating"] = 2
        with self.assertRaises(app.ValidationError):
            app.analyze(data)

    def test_record_limit(self):
        with self.assertRaises(app.ValidationError):
            app.analyze(fixture(*(["fine"] * 1001)))
        self.assertEqual(app.analyze(fixture("a" * 10000))["summary"]["input_count"], 1)

    def test_evidence_tampering_rejected(self):
        result = app.analyze(fixture("late delivery"))
        result["themes"][0]["excerpts"][0]["text"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.validate(result, "output")

    def test_cli_success(self):
        code, result = self.cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ok")

    def test_cli_missing_file_and_usage(self):
        for args in ((), ("nonexistent-input.json",), ("one", "two")):
            with self.subTest(args=args):
                code, result = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_invalid_schema(self):
        code, result = self.cli("build_manifest.json")
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "error")

    def test_cli_malformed_duplicate_keys_and_nonfinite(self):
        for raw in ('{', '{"a": 1, "a": 2}', '{"value": NaN}', '[]'):
            output = io.StringIO()
            with patch.object(app.Path, "read_text", return_value=raw), contextlib.redirect_stdout(output):
                self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_file_decoding_error(self):
        output = io.StringIO()
        with patch.object(app.Path, "read_text",
                          side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
