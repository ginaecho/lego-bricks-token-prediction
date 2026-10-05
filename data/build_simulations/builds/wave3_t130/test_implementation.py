import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_all_entities(self):
        result = app.analyze(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["results"][0]["severity"], "critical")
        self.assertEqual(result["results"][0]["sentiment"]["score"], -3)

    def test_neutral_and_repeated_words(self):
        self.data["records"][0]["text"] = "helpful helpful delayed delayed"
        sentiment = app.analyze(self.data)["results"][0]["sentiment"]
        self.assertEqual(sentiment["score"], 0)
        self.assertEqual(sentiment["label"], "neutral")
        self.assertEqual(sentiment["positive_matches"], ["helpful", "helpful"])

    def test_severity_dominates_large_negative_count(self):
        self.data["records"][0]["text"] = "clear"
        self.data["records"][0]["reported_severity"] = "critical"
        self.data["records"][1]["text"] = "delay " * 500
        self.assertEqual(app.analyze(self.data)["results"][0]["case_number"], "SYN-0001")

    def test_deterministic_tie_break(self):
        for record in self.data["records"]:
            record["text"] = "Meeting notice."
            record["reported_severity"] = "routine"
        first = app.analyze(self.data)
        self.data["records"].reverse()
        self.assertEqual(first, app.analyze(self.data))

    def test_empty_blank_and_oversize(self):
        for text in ("", " \n", "a" * 5001):
            with self.subTest(text_length=len(text)):
                self.data["records"][0]["text"] = text
                with self.assertRaises(app.ValidationError):
                    app.analyze(self.data)
        self.data["records"] = []
        with self.assertRaises(app.ValidationError):
            app.analyze(self.data)

    def test_invalid_schema_types_and_unknown_fields(self):
        for key, value in (("schema_version", True), ("synthetic", False),
                           ("records", {}), ("extra", "ignored?")):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.analyze(data)

    def test_invalid_records(self):
        for key, value in (("entity", "unknown"), ("reported_severity", "low"),
                           ("text", 42), ("case_number", "real-1"),
                           ("persona", "Jane Doe"), ("address", "12 Main Street")):
            data = copy.deepcopy(self.data)
            data["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.analyze(data)

    def test_duplicate_cases(self):
        self.data["records"].append(copy.deepcopy(self.data["records"][0]))
        with self.assertRaises(app.ValidationError):
            app.analyze(self.data)

    def test_personal_identifiers_blocked(self):
        for text in ("Email a@example.invalid", "Call 202-555-0123",
                     "Number 123-45-6789", "Lives at 12 Main Street",
                     "My name is Example", "Date of birth: yesterday"):
            self.data["records"][0]["text"] = text
            with self.subTest(text=text), self.assertRaises(app.ValidationError):
                app.analyze(self.data)

    def test_output_minimizes_personal_data_and_explains(self):
        result = app.analyze(self.data)
        rendered = json.dumps(result)
        self.assertNotIn("Synthetic resident", rendered)
        self.assertNotIn("NOT DELIVERABLE", rendered)
        self.assertNotIn(self.data["records"][0]["text"], rendered)
        self.assertIn("severity_rule", result["results"][0]["explanation"])
        self.assertIn("No benefit decision", result["results"][0]["next_step"])

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_duplicate_and_invalid_input(self):
        for text in ("{", '{"synthetic":true,"synthetic":true}', "null",
                     '{"schema_version":1,"synthetic":false,"records":[]}'):
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=text), \
                    contextlib.redirect_stdout(output):
                code = app.main([str(ROOT / "example_input.json")])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_unicode_read_error_is_json(self):
        output = io.StringIO()
        with patch.object(Path, "read_text",
                          side_effect=UnicodeDecodeError("utf8", b"\xff", 0, 1, "invalid")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(app.main([str(ROOT / "example_input.json")]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
