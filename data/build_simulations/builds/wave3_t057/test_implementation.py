"""All examples are synthetic; tests create no files."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, automate


ROOT = Path(__file__).resolve().parent


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "schema_version": 1,
            "fixture_label": "SYNTHETIC unit fixture",
            "documents": [{"id": "synthetic-1", "format": "records",
                           "content": [{"name": " A ", "count": "2"}]}],
            "fields": [{"source": "name", "target": "label", "type": "string",
                        "required": True},
                       {"source": "count", "target": "quantity", "type": "integer",
                        "min": 0, "max": 10}],
            "unique": ["label"]}

    def test_normal_reshape(self):
        result = automate(self.payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["records"][0]["data"], {"label": "A", "quantity": 2})
        self.assertEqual(result["summary"]["valid_records"], 1)

    def test_example_three_formats(self):
        result = automate(json.loads((ROOT / "example_input.json").read_text("utf-8")))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["records"], 4)
        self.assertEqual(result["records"][2]["data"]["total"], 19.0)

    def test_empty_documents(self):
        self.payload["documents"] = []
        self.assertEqual(automate(self.payload)["summary"],
                         {"documents": 0, "records": 0, "valid_records": 0, "issues": 0})

    def test_missing_default_and_explicit_null(self):
        self.payload["fields"][1]["default"] = "3"
        self.payload["documents"][0]["content"] = [{"name": "A"}, {"name": "", "count": None}]
        result = automate(self.payload)
        self.assertEqual(result["records"][0]["data"]["quantity"], 3)
        self.assertIsNone(result["records"][1]["data"]["quantity"])
        self.assertEqual([i["code"] for i in result["issues"]], ["required"])

    def test_type_and_range_failures(self):
        self.payload["documents"][0]["content"] = [
            {"name": "A", "count": True}, {"name": "B", "count": -1},
            {"name": "C", "count": "1.5"}, {"name": "D", "count": 11}]
        result = automate(self.payload)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual([i["code"] for i in result["issues"]], ["type", "min", "type", "max"])

    def test_cross_document_uniqueness(self):
        self.payload["documents"].append({"id": "synthetic-2", "format": "key_value",
                                          "content": "name: A\ncount: 4"})
        result = automate(self.payload)
        self.assertEqual(result["issues"][0]["code"], "unique")
        self.assertEqual(result["issues"][0]["document_id"], "synthetic-2")

    def test_patterns_and_choices(self):
        self.payload["fields"][0].update(pattern="B+", choices=["B", "C"])
        self.assertEqual([i["code"] for i in automate(self.payload)["issues"]],
                         ["pattern", "choices"])

    def test_invalid_schema(self):
        for mutate in (
            lambda p: p.update(schema_version=True),
            lambda p: p.update(unique=["unknown"]),
            lambda p: p["fields"].append(copy.deepcopy(p["fields"][0])),
            lambda p: p["fields"][0].update(pattern="["),
            lambda p: p["fields"][1].update(min=20),
            lambda p: p.update(unexpected=1),
            lambda p: p["documents"].append(copy.deepcopy(p["documents"][0])),
        ):
            with self.subTest(mutate=mutate):
                payload = copy.deepcopy(self.payload)
                mutate(payload)
                with self.assertRaises(ValidationError):
                    automate(payload)

    def test_malformed_extraction(self):
        for fmt, content in (
            ("csv", "a,a\n1,2"), ("csv", "a,b\n1"),
            ("csv", 'a\n"unterminated'), ("key_value", "a: 1\na: 2"),
            ("key_value", "not a field"),
        ):
            with self.subTest(fmt=fmt, content=content):
                self.payload["documents"][0].update(format=fmt, content=content)
                with self.assertRaises(ValidationError):
                    automate(self.payload)

    def test_quoted_csv(self):
        self.payload["documents"][0].update(format="csv", content='name,count\n"A, B",2\n')
        self.assertEqual(automate(self.payload)["records"][0]["data"]["label"], "A, B")

    def test_nonfinite_and_boolean_conversion(self):
        self.payload["fields"] = [{"source": "n", "target": "n", "type": "number"},
                                  {"source": "b", "target": "b", "type": "boolean"}]
        self.payload["unique"] = []
        self.payload["documents"][0]["content"] = [{"n": "NaN", "b": "TRUE"},
                                                  {"n": "Infinity", "b": "yes"}]
        result = automate(self.payload)
        self.assertTrue(result["records"][0]["data"]["b"])
        self.assertEqual(result["summary"]["issues"], 3)

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(automate(self.payload), automate(self.payload))
        self.assertEqual(original, self.payload)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        completed = self.cli("example_input.json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")

    def test_cli_errors(self):
        for args in ((), ("absent-synthetic-file.json",), ("implementation.py",),
                     ("example_input.json", "extra")):
            with self.subTest(args=args):
                completed = self.cli(*args)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(json.loads(completed.stdout)["status"], "error")
                self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
