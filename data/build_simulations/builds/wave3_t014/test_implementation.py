"""Tests use only clearly synthetic fixtures; no network or provider calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import Schema, ValidationError, extract, run_pipeline, search


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_synonyms_and_filters(self):
        result = run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([p["product_id"] for p in result["search"]["results"]], ["syn-1"])
        self.assertEqual(result["search"]["intent"]["terms"], ["run", "shoe"])
        self.assertEqual(result["search"]["intent"]["max_price"], 90)

    def test_source_spans_unicode_crlf_and_whitespace(self):
        self.data["document"] = "Synthetic ☀\r\n Looking for:  café sneakers  \r\nBudget: 80\r\n"
        result = extract(self.data)
        for item in result["fields"].values():
            if item["span"]:
                span = item["span"]
                self.assertEqual(self.data["document"][span["start"]:span["end"]], item["source_text"])
        self.assertEqual(result["fields"]["query"]["value"], "café sneakers")

    def test_missing_optional_fields_reported(self):
        self.assertEqual(extract(self.data)["missing_fields"], ["color"])
        self.assertEqual(run_pipeline(self.data)["status"], "ok")

    def test_missing_required_prevents_search(self):
        self.data["document"] = "Budget: 90"
        result = run_pipeline(self.data)
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["extraction"]["missing_required"], ["query"])
        self.assertEqual(result["search"]["results"], [])

    def test_optional_query_still_must_supply_terms(self):
        self.data["fields"][0]["required"] = False
        self.data["document"] = "Looking for: the and please"
        self.assertEqual(run_pipeline(self.data)["search"]["reason"], "missing_search_terms")

    def test_budget_handoff_changes_results(self):
        self.data["document"] = self.data["document"].replace("90", "150")
        self.assertEqual(len(run_pipeline(self.data)["search"]["results"]), 2)

    def test_false_stock_filter_is_not_ignored(self):
        self.data["document"] = "Looking for: shoes\nAvailable: no"
        self.assertEqual([p["product_id"] for p in run_pipeline(self.data)["search"]["results"]],
                         ["syn-3"])

    def test_typo_and_general_catalog(self):
        self.data["document"] = "Looking for: wireles headphones"
        # Use an original catalog term to verify typo tolerance, independently of synonyms.
        self.data["catalog"][3]["name"] = "Wireless headphones"
        result = run_pipeline(self.data)["search"]["results"][0]
        self.assertEqual(result["product_id"], "syn-4")
        self.assertTrue(all(match["score"] > 0 for match in result["matches"]))
        self.data["document"] = "Looking for: runn"
        self.assertEqual(run_pipeline(self.data)["search"]["results"][0]["product_id"], "syn-1")

    def test_no_match_and_empty_catalog(self):
        self.data["document"] = "Looking for: xylophone"
        self.assertEqual(run_pipeline(self.data)["search"]["results"], [])
        self.data["catalog"] = []
        self.assertEqual(run_pipeline(self.data)["search"]["total_matches"], 0)

    def test_duplicate_document_label_rejected(self):
        self.data["document"] += "query: something else"
        with self.assertRaises(ValidationError):
            run_pipeline(self.data)

    def test_invalid_values_and_configuration(self):
        for document in ("Budget: bananas", "Available: perhaps", "Budget: -2\nquery: shoes"):
            with self.subTest(document=document):
                self.data["document"] = document
                with self.assertRaises(ValidationError):
                    run_pipeline(self.data)
        for value in (True, 0, 101, "5"):
            with self.subTest(limit=value):
                self.data["search"]["limit"] = value
                with self.assertRaises(ValidationError):
                    run_pipeline(self.data)

    def test_schema_rejects_bad_types_and_ambiguous_aliases(self):
        for mutate in (
            lambda d: d.update(catalog=None),
            lambda d: d["catalog"][0].update(price=True),
            lambda d: d["catalog"][0].update(price=float("nan")),
            lambda d: d["catalog"][0].update(price=10 ** 1000),
            lambda d: d["fields"][1].update(aliases=["Looking for"]),
            lambda d: d["search"].update(query_field="budget"),
            lambda d: d["catalog"].append(copy.deepcopy(d["catalog"][0])),
        ):
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(ValidationError):
                Schema.validate_input(data)

    def test_tampered_handoff_rejected(self):
        handoff = extract(self.data)
        handoff["fields"]["budget"]["value"] = 200
        with self.assertRaises(ValidationError):
            search(handoff, self.data)

    def test_deterministic_ties_and_limit(self):
        self.data["document"] = "query: shoes"
        product = copy.deepcopy(self.data["catalog"][0])
        product["id"] = "syn-0"
        self.data["catalog"].append(product)
        self.data["search"]["limit"] = 2
        first = run_pipeline(self.data)
        self.assertEqual(first, run_pipeline(self.data))
        self.assertEqual([p["product_id"] for p in first["search"]["results"]], ["syn-3", "syn-0"])
        self.assertEqual(first["search"]["total_matches"], 4)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, text=True, capture_output=True, check=False)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_file_usage_and_json_errors(self):
        for args in ((), ("does-not-exist.json",), ("implementation.py",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
