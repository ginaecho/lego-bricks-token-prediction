"""Synthetic fixtures only; no network, external dependencies, or temp files."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


def fixture(document="Amount: 12.50", kind="number", required=True):
    return {"document": document, "schema": [
        {"name": "amount", "labels": ["Amount"], "type": kind, "required": required}
    ]}


class ExtractionTests(unittest.TestCase):
    def test_example_and_spans(self):
        payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        result = impl.extract(payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["fields"]["quantity"]["value"], 7)
        self.assertEqual(result["fields"]["total"]["value"], 84.5)
        for field in result["fields"].values():
            if field is not None:
                span = field["span"]
                self.assertEqual(payload["document"][span["start"]:span["end"]],
                                 field["source_text"])

    def test_unicode_crlf_and_whitespace(self):
        document = "Synthetic 🧱\r\n\tAMOUNT =  café 🧱  \r\n"
        result = impl.extract(fixture(document, "string"))
        field = result["fields"]["amount"]
        self.assertEqual(field["value"], "café 🧱")
        self.assertEqual(document[field["span"]["start"]:field["span"]["end"]], "café 🧱")

    def test_required_missing_empty_document(self):
        result = impl.extract(fixture(""))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_fields"], [{"field": "amount", "required": True}])
        self.assertIsNone(result["fields"]["amount"])

    def test_optional_missing_is_ok(self):
        result = impl.extract(fixture("Synthetic fixture", required=False))
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["missing_fields"][0]["required"])

    def test_first_match_wins_including_blank(self):
        self.assertEqual(impl.extract(fixture("Amount: 2\nAmount: 3"))["fields"]["amount"]["value"], 2)
        result = impl.extract(fixture("Amount: \nAmount: 3"))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["missing_fields"]), 1)

    def test_invalid_typed_values_have_evidence(self):
        for kind, value in [("integer", "2.5"), ("number", "NaN"),
                            ("number", "9" * 400), ("date", "2026-02-30"),
                            ("date", "20260923")]:
            with self.subTest(kind=kind, value=value):
                result = impl.extract(fixture("Amount: " + value, kind))
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["invalid_fields"][0]["source_text"], value)
                self.assertEqual(result["missing_fields"], [])

    def test_valid_types_and_literal_aliases(self):
        for kind, source, expected in [("integer", "-12", -12),
                                       ("date", "2024-02-29", "2024-02-29"),
                                       ("string", "alpha: beta", "alpha: beta")]:
            payload = fixture("Total (USD).: " + source, kind)
            payload["schema"][0]["labels"].append("Total (USD).")
            self.assertEqual(impl.extract(payload)["fields"]["amount"]["value"], expected)

    def test_invalid_schema(self):
        candidates = [None, [], {}, {"document": 12, "schema": []}]
        for key, value in [("type", []), ("required", 1), ("labels", [""]),
                           ("labels", ["Amount", "amount"]), ("name", "bad name")]:
            payload = fixture()
            payload["schema"][0][key] = value
            candidates.append(payload)
        duplicate = fixture()
        duplicate["schema"].append(copy.deepcopy(duplicate["schema"][0]))
        candidates.append(duplicate)
        for payload in candidates:
            with self.subTest(payload=payload):
                with self.assertRaises(impl.ValidationError):
                    impl.extract(payload)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_usage_and_missing_file(self):
        for arguments in [[], [str(ROOT / "nonexistent-synthetic-fixture.json")]]:
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_invalid_json_and_file_errors(self):
        for text in ['{', '{"document":"","document":"","schema":[]}',
                     '{"document":NaN,"schema":[]}', '{"document":null,"schema":[]}']:
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=text), redirect_stdout(output):
                code = impl.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
        with patch.object(Path, "read_text", side_effect=PermissionError("synthetic denial")):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(impl.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
