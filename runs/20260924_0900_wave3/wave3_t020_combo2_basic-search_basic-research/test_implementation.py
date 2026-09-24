import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, research, run_pipeline, search, validate


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_synonyms_and_deterministic_ranking(self):
        result = search(self.request)
        self.assertEqual([r["product"]["id"] for r in result["results"]], ["p1", "p2"])
        self.assertEqual(result["results"][0]["score"], 1)
        self.assertEqual(result, search(self.request))

    def test_typo_tolerance(self):
        self.request["query"] = "headphnes"
        rows = search(self.request)["results"]
        self.assertTrue(rows)
        self.assertEqual(rows[0]["matches"][0]["method"], "typo")

    def test_filters(self):
        self.request["filters"] = {"category": "audio", "max_price": 90, "in_stock": True}
        self.assertEqual([r["product"]["id"] for r in search(self.request)["results"]], ["p1"])
        self.request["filters"]["in_stock"] = False
        self.assertEqual(search(self.request)["results"], [])

    def test_research_citations_are_verbatim(self):
        result = run_pipeline(self.request)
        evidence = result["research"]["evidence"]
        sources = {s["id"]: s for s in self.request["sources"]}
        self.assertEqual(len(evidence), 3)
        for row in evidence:
            self.assertEqual(row["quote"], sources[row["source_id"]]["text"])
        self.assertEqual(result["research"]["recommendation"]["product_id"], "p1")

    def test_limit_propagates_and_excludes_other_sources(self):
        self.request["limit"] = 1
        result = run_pipeline(self.request)
        self.assertEqual({e["product_id"] for e in result["research"]["evidence"]}, {"p1"})
        self.assertEqual([r["product_id"] for r in result["research"]["comparison"]], ["p1"])

    def test_tampered_handoff_rejected(self):
        result = search(self.request)
        result["results"][0]["product"]["price"] = 1
        with self.assertRaises(ValidationError):
            research(self.request, result)

    def test_tampered_citation_rejected(self):
        result = run_pipeline(self.request)
        result["research"]["evidence"][0]["quote"] = "invented claim"
        with self.assertRaises(ValidationError):
            validate("output", result, self.request)

    def test_no_matches_and_stopwords(self):
        for query in ("xyzzzzzz", "the and for", "!!!"):
            self.request["query"] = query
            result = run_pipeline(self.request)
            self.assertEqual(result["search"]["results"], [])
            self.assertIsNone(result["research"]["recommendation"])

    def test_empty_catalog(self):
        self.request["products"] = []
        self.request["sources"] = []
        self.assertEqual(run_pipeline(self.request)["research"]["comparison"], [])

    def test_missing_and_irrelevant_evidence(self):
        self.request["question"] = "repairability"
        self.assertIsNone(run_pipeline(self.request)["research"]["recommendation"])
        self.request["sources"] = []
        self.assertEqual(run_pipeline(self.request)["research"]["evidence"], [])

    def test_invalid_input_types(self):
        for field, value in (("query", ""), ("limit", True), ("limit", 0),
                             ("schema_version", True), ("products", {}), ("sources", None)):
            request = copy.deepcopy(self.request)
            request[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                run_pipeline(request)

    def test_invalid_prices_and_currency(self):
        for value in (-1, True, float("nan"), float("inf")):
            request = copy.deepcopy(self.request)
            request["products"][0]["price"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                run_pipeline(request)
        self.request["products"][0]["currency"] = "EUR"
        with self.assertRaises(ValidationError):
            run_pipeline(self.request)

    def test_duplicate_and_orphan_ids(self):
        request = copy.deepcopy(self.request)
        request["products"].append(copy.deepcopy(request["products"][0]))
        with self.assertRaises(ValidationError):
            run_pipeline(request)
        request = copy.deepcopy(self.request)
        request["sources"][0]["product_id"] = "absent"
        with self.assertRaises(ValidationError):
            run_pipeline(request)
        self.request["sources"].append(copy.deepcopy(self.request["sources"][0]))
        with self.assertRaises(ValidationError):
            run_pipeline(self.request)

    def test_unknown_fields_rejected(self):
        self.request["surprise"] = True
        with self.assertRaises(ValidationError):
            run_pipeline(self.request)

    def test_large_integer_price_is_filtered_without_overflow(self):
        self.request["products"][0]["price"] = 10 ** 400
        result = run_pipeline(self.request)
        self.assertEqual([r["product"]["id"] for r in result["search"]["results"]], ["p2"])

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_errors(self):
        for args in ((), ("absent.json",), ("implementation.py",), ("build_manifest.json",),
                     ("example_input.json", "extra")):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
