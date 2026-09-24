"""All data in this suite is synthetic. Tests create no files."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

from implementation import ValidationError, Validator, extract, main


ROOT = Path(__file__).resolve().parent


def fixture(document="count: 7", kind="integer", required=True):
    return {"document": document, "schema": [
        {"name": "count", "type": kind, "required": required}
    ]}


class ExtractionTests(unittest.TestCase):
    def test_synthetic_example_all_types_and_missing(self):
        data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        result = extract(data)
        self.assertEqual([f["value"] for f in result["fields"]],
                         ["SYN-034", "Ada Example", 12, 125.5, False, "2026-09-24", None, None])
        self.assertEqual(result["missing_fields"], [{"name": "purchase_order", "reason": "missing"}])

    def test_unicode_crlf_alias_and_exact_span(self):
        data = fixture("SYNTHETIC café\r\n  Unit   Count =  42 \r\n")
        data["schema"][0]["aliases"] = ["unit count"]
        result = extract(data)
        field = result["fields"][0]
        self.assertEqual(field["value"], 42)
        source = field["sources"][0]
        self.assertEqual(source["start"], data["document"].index("42"))
        self.assertEqual(data["document"][source["start"]:source["end"]], "42")

    def test_blank_and_optional_missing(self):
        for text in ["", "count:  \n"]:
            with self.subTest(text=text):
                result = extract(fixture(text, required=False))
                self.assertEqual(result["fields"][0]["state"], "missing")
                self.assertEqual(result["missing_fields"], [])

    def test_duplicate_is_ambiguous_even_when_equal(self):
        result = extract(fixture("count: 7\nCOUNT = 7"))
        self.assertEqual(result["fields"][0]["state"], "ambiguous")
        self.assertEqual(len(result["fields"][0]["sources"]), 2)
        self.assertEqual(result["missing_fields"], [{"name": "count", "reason": "ambiguous"}])

    def test_invalid_values_do_not_fabricate(self):
        for kind, value in [("integer", "3.5"), ("integer", "9" * 101),
                            ("number", "1e999"), ("number", "NaN"),
                            ("boolean", "yes"), ("date", "2026-02-30")]:
            with self.subTest(kind=kind, value=value):
                result = extract(fixture("count: " + value, kind))
                field = result["fields"][0]
                self.assertEqual(field["state"], "invalid")
                self.assertIsNone(field["value"])
                self.assertEqual(field["sources"][0]["text"], value)
                self.assertEqual(result["missing_fields"][0]["reason"], "invalid")

    def test_string_preserves_inner_delimiters(self):
        self.assertEqual(extract(fixture("count: https://synthetic.invalid/a=b", "string"))
                         ["fields"][0]["value"], "https://synthetic.invalid/a=b")

    def test_unknown_labels_and_prose_ignored(self):
        result = extract(fixture("A count: 7\ncounting: 8\ncount has value 9"))
        self.assertEqual(result["fields"][0]["state"], "missing")

    def test_input_validation(self):
        cases = [None, [], {}, {"document": 1, "schema": []},
                 {"document": "", "schema": []}, {**fixture(), "extra": 1}]
        for prop, value in [("required", 1), ("type", "money"), ("type", []),
                            ("name", "bad name"), ("aliases", "count"),
                            ("aliases", ["COUNT"]), ("aliases", ["bad:label"])]:
            data = fixture()
            data["schema"][0][prop] = value
            cases.append(data)
        data = fixture()
        data["schema"].append({"name": "other", "type": "string", "required": False,
                               "aliases": ["count"]})
        cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                extract(data)

    def test_result_validation_rejects_corrupt_span(self):
        data = fixture()
        result = extract(data)
        result["fields"][0]["sources"][0]["end"] = 100
        with self.assertRaises(ValidationError):
            Validator.result(data, result)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_usage_and_missing_file(self):
        for args in [[], [str(ROOT / "nonexistent.json")]]:
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_schema_without_writes(self):
        for text in ["{", '{"document":"","document":"","schema":[]}',
                     '{"document":NaN,"schema":[]}', '{"document":1,"schema":[]}']:
            with self.subTest(text=text):
                stdout = io.StringIO()
                with patch("builtins.open", mock_open(read_data=text)), contextlib.redirect_stdout(stdout):
                    code = main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_valid_edge_values(self):
        for kind, text, expected in [("integer", "-0", 0), ("number", ".25", .25),
                                     ("number", "1e2", 100.0), ("boolean", "TRUE", True),
                                     ("date", "2024-02-29", "2024-02-29")]:
            with self.subTest(kind=kind):
                self.assertEqual(extract(fixture("count=" + text, kind))
                                 ["fields"][0]["value"], expected)


if __name__ == "__main__":
    unittest.main()
