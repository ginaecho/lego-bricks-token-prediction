"""Tests use only synthetic in-memory fixtures and the shipped example."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_synonyms(self):
        response = app.search(self.payload)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["normalized_tokens"], ["running", "shoe"])
        self.assertEqual(response["total_matches"], 1)
        self.assertEqual(response["results"][0]["product"]["id"], "synthetic-shoe-1")

    def test_typo_and_prefix(self):
        for query, kind in (("runnign shoes", "typo"), ("run shoes", "prefix")):
            with self.subTest(query=query):
                self.payload["query"] = query
                result = app.search(self.payload)
                self.assertEqual(result["total_matches"], 1)
                self.assertIn(kind, [m["kind"] for m in result["results"][0]["matches"]])

    def test_unicode_and_case(self):
        self.payload["query"] = "RÚNNING SHÓES"
        self.assertEqual(app.search(self.payload)["total_matches"], 1)

    def test_all_terms_required(self):
        self.payload["query"] = "running laptop"
        self.assertEqual(app.search(self.payload)["total_matches"], 0)

    def test_browse_filters_and_sort(self):
        self.payload.update(query="", sort="price_asc", limit=1)
        result = app.search(self.payload)
        self.assertEqual(result["mode"], "browse")
        self.assertEqual(result["total_matches"], 2)
        self.assertEqual(result["results"][0]["product"]["price"], 49)
        self.payload.update(sort="price_desc")
        self.assertEqual(app.search(self.payload)["results"][0]["product"]["price"], 69.5)
        self.payload["filters"] = {"category": " electronics ", "in_stock": False, "min_price": 499}
        self.assertEqual(app.search(self.payload)["total_matches"], 1)

    def test_no_match_and_empty_catalog(self):
        for query in ("volcano", "!!!", "please find me"):
            self.payload["query"] = query
            self.assertEqual(app.search(self.payload)["results"], [])
        self.payload.update(query="", products=[])
        self.assertEqual(app.search(self.payload)["total_matches"], 0)

    def test_duplicate_query_terms_and_no_mutation(self):
        self.payload["query"] = "running running sneakers shoes"
        original = copy.deepcopy(self.payload)
        result = app.search(self.payload)
        self.assertEqual(len(result["normalized_tokens"]), 2)
        self.assertEqual(self.payload, original)

    def test_stable_ties(self):
        duplicate = copy.deepcopy(self.payload["products"][0])
        duplicate["id"] = "a"
        self.payload["products"].append(duplicate)
        first = app.search(self.payload)
        self.payload["products"].reverse()
        self.assertEqual(first, app.search(self.payload))
        self.assertEqual(first["results"][0]["product"]["id"], "a")

    def test_invalid_schema(self):
        invalid = [
            None, [], {}, {"query": 3, "products": []},
            {**self.payload, "limit": True},
            {**self.payload, "limit": 0},
            {**self.payload, "sort": []},
            {**self.payload, "filters": {"min_price": 10, "max_price": 1}},
            {**self.payload, "filters": {"in_stock": 1}},
            {**self.payload, "unknown": True},
            {**self.payload, "products": self.payload["products"] * 2},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(app.ValidationError):
                    app.search(payload)

    def test_invalid_product_fields(self):
        for field, value in (("price", float("nan")), ("price", float("inf")),
                             ("price", -1), ("price", True), ("price", 10**400),
                             ("tags", "running"), ("in_stock", "yes"), ("name", "")):
            with self.subTest(field=field, value=value):
                payload = copy.deepcopy(self.payload)
                payload["products"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.search(payload)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), app.search(self.payload))
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_schema(self):
        for raw in ('{', '{"query":"","query":"x","products":[]}',
                    '{"query":"","products":[],"limit":NaN}', 'null',
                    '{"query":"","products":[],"limit":false}'):
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=raw), redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
