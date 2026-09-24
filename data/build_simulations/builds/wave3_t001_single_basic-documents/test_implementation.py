"""All fixtures are synthetic; tests never create files or contact services."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class DocumentAutomationTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def json_rows(self, rows):
        self.payload["documents"] = [
            {"id": "synthetic-test", "format": "json", "content": json.dumps(rows)}]

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_example_and_determinism(self):
        before = copy.deepcopy(self.payload)
        result = app.automate(self.payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["rows_emitted"], 3)
        self.assertEqual(result["records"][0]["data"],
                         {"customer": "synthetic alpha", "units": 3,
                          "amount": 42.5, "approved": True})
        self.assertEqual(result, app.automate(self.payload))
        self.assertEqual(before, self.payload)

    def test_empty_documents(self):
        self.payload["documents"] = []
        self.assertEqual(app.automate(self.payload)["status"], "ok")
        self.assertEqual(app.automate(self.payload)["records"], [])

    def test_missing_optional_and_required(self):
        self.payload["fields"][0]["required"] = False
        self.json_rows([{"Units": 1, "Amount": 0, "Approved": False}, {}])
        result = app.automate(self.payload)
        self.assertIsNone(result["records"][0]["data"]["customer"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["summary"]["rows_rejected"], 1)
        self.assertEqual(len(result["issues"]), 3)

    def test_invalid_types_and_bounds(self):
        self.json_rows([{"Customer": "Synthetic", "Units": True,
                         "Amount": "-1", "Approved": "yes"}])
        result = app.automate(self.payload)
        self.assertEqual(len(result["issues"]), 3)
        self.assertEqual(result["records"], [])

    def test_nonfinite_rejected(self):
        self.json_rows([{"Customer": "Synthetic", "Units": 1,
                         "Amount": "NaN", "Approved": True}])
        self.assertEqual(app.automate(self.payload)["status"], "error")
        self.payload["documents"][0]["content"] = '[{"Amount":NaN}]'
        self.assertEqual(app.automate(self.payload)["status"], "error")

    def test_csv_quoting(self):
        self.payload["documents"] = [{"id": "synthetic", "format": "csv",
                                     "content": 'Customer,Units,Amount,Approved\n"Synthetic, A",1,2,true\n'}]
        result = app.automate(self.payload)
        self.assertEqual(result["records"][0]["data"]["customer"], "synthetic, a")

    def test_bad_documents(self):
        for fmt, content in [("csv", "x,x\n1,2\n"), ("csv", "x,y\n1\n"),
                             ("csv", ""), ("json", "{}"), ("json", "[1]"),
                             ("json", '[{"x":1,"x":2}]'), ("json", "broken")]:
            with self.subTest(fmt=fmt, content=content):
                self.payload["documents"] = [{"id": "synthetic", "format": fmt,
                                             "content": content}]
                result = app.automate(self.payload)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["issues"][0]["document_id"], "synthetic")

    def test_bad_schema(self):
        for payload in [None, [], {}, {"schema_version": True, "documents": [], "fields": []}]:
            with self.subTest(payload=payload):
                self.assertEqual(app.automate(payload)["status"], "error")
        self.payload["fields"][0]["pattern"] = "["
        self.assertEqual(app.automate(self.payload)["status"], "error")

    def test_duplicate_ids_targets_and_unknown_keys(self):
        cases = []
        p = copy.deepcopy(self.payload)
        p["documents"].append(p["documents"][0])
        cases.append(p)
        p = copy.deepcopy(self.payload)
        p["fields"].append(p["fields"][0])
        cases.append(p)
        p = copy.deepcopy(self.payload)
        p["surprise"] = True
        cases.append(p)
        for payload in cases:
            self.assertEqual(app.automate(payload)["status"], "error")

    def test_pattern_choices_and_upper_transform(self):
        self.payload["fields"] = [{"source": "name", "target": "name", "type": "string",
                                  "transforms": ["strip", "upper"], "pattern": "SYNTHETIC-[AB]",
                                  "choices": ["SYNTHETIC-A"]}]
        self.json_rows([{"name": " synthetic-a "}, {"name": "synthetic-b"}, {"name": "wrong"}])
        result = app.automate(self.payload)
        self.assertEqual(result["summary"]["rows_emitted"], 1)
        self.assertEqual(result["summary"]["rows_rejected"], 2)

    def test_cli_success(self):
        proc = self.cli("example_input.json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_errors(self):
        for args in [(), ("nonexistent-synthetic.json",), ("implementation.py",),
                     ("build_manifest.json",), ("a", "b")]:
            with self.subTest(args=args):
                proc = self.cli(*args)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main()
