import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

from implementation import ValidationError, main, search


ROOT = Path(__file__).resolve().parent


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def ids(self, data=None):
        return [item["product"]["id"] for item in search(data or self.data)["results"]]

    def test_normal_exact_beats_synonym(self):
        self.assertEqual(self.ids(), ["syn-002", "syn-001"])
        result = search(self.data)
        self.assertEqual(result["results"][1]["matches"][1]["kind"], "synonym")

    def test_typo_and_plural(self):
        self.data.update(query="wireles headphones", filters={})
        result = search(self.data)
        self.assertEqual(self.ids(), ["syn-004"])
        self.assertIn("typo", [match["kind"] for match in result["results"][0]["matches"]])

    def test_unicode_normalization(self):
        self.data["query"] = "CAFE table"
        self.assertEqual(self.ids(), ["syn-005"])

    def test_blank_browses_with_limit_and_total(self):
        self.data.update(query="   ", limit=1)
        result = search(self.data)
        self.assertEqual(result["total_matches"], 3)
        self.assertEqual(self.ids(), ["syn-001"])

    def test_empty_catalog(self):
        self.data["products"] = []
        self.assertEqual(search(self.data)["results"], [])

    def test_no_match_and_all_terms_required(self):
        self.data["query"] = "comfy submarine"
        self.assertEqual(self.ids(), [])

    def test_inclusive_filters(self):
        self.data.update(query="", filters={"category": " furniture ", "min_price": 399,
                                          "max_price": 399, "in_stock": True})
        self.assertEqual(self.ids(), ["syn-001"])

    def test_false_stock_filter(self):
        self.data.update(query="", filters={"in_stock": False})
        self.assertEqual(self.ids(), ["syn-004"])

    def test_deterministic_and_not_mutating(self):
        before = copy.deepcopy(self.data)
        first = search(self.data)
        self.assertEqual(before, self.data)
        self.data["products"].reverse()
        self.assertEqual(first, search(self.data))

    def test_invalid_top_level(self):
        cases = [None, [], {}, dict(self.data, query=1), dict(self.data, limit=True),
                 dict(self.data, limit=0), dict(self.data, schema_version=True),
                 dict(self.data, unknown=1), dict(self.data, products={}),
                 dict(self.data, filters={"min_price": 5, "max_price": 1}),
                 dict(self.data, filters={"in_stock": 1})]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    search(data)

    def test_invalid_products(self):
        for key, value in [("price", float("nan")), ("price", -1), ("price", True),
                           ("price", float("inf")), ("tags", "bad"), ("tags", [3]),
                           ("in_stock", "yes"), ("name", "")]:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["products"][0][key] = value
                with self.assertRaises(ValidationError):
                    search(data)
        self.data["products"].append(self.data["products"][0])
        with self.assertRaises(ValidationError):
            search(self.data)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "missing.json")], ["one", "two"]]:
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_malformed_and_invalid_json_without_extra_files(self):
        for content in ["{", '{"schema_version":1,"schema_version":1}',
                        '{"query":NaN}', "[]", '{"schema_version":1}']:
            stdout = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)):
                with contextlib.redirect_stdout(stdout):
                    code = main(["synthetic-fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
