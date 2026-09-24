import contextlib
import copy
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_ranking_and_index(self):
        result = app.search(self.request)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([m["id"] for m in result["matches"]], ["SYN-001", "SYN-003"])
        self.assertEqual(result["matches"][0]["matched_terms"],
                         ["backpack", "hiking", "waterproof"])
        self.assertEqual(result["index"]["product_count"], 3)
        self.assertGreater(result["index"]["term_count"], 0)

    def test_casefold_limit_and_repeat_query_terms(self):
        original = app.search(self.request)
        self.request.update(query="WATERPROOF HIKING backpack hiking", limit=1)
        result = app.search(self.request)
        self.assertEqual(result["matches"], original["matches"][:1])
        self.assertEqual(result["total_matches"], 2)

    def test_empty_catalog_and_no_match(self):
        self.request["query"] = "astronaut"
        self.assertEqual(app.search(self.request)["matches"], [])
        self.request["products"] = []
        result = app.search(self.request)
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["index"], {"product_count": 0, "term_count": 0})

    def test_stable_ties_and_input_not_mutated(self):
        self.request["products"] = [{"id": name, "title": "hiking"} for name in ["b", "a"]]
        before = copy.deepcopy(self.request)
        result = app.search(self.request)
        self.assertEqual([m["id"] for m in result["matches"]], ["a", "b"])
        self.assertEqual(before, self.request)
        self.request["products"].reverse()
        self.assertEqual(app.search(self.request), result)

    def test_invalid_root_and_options(self):
        for payload in [None, [], {}, {**self.request, "query": " "},
                        {**self.request, "query": "!!!"},
                        {**self.request, "limit": True},
                        {**self.request, "limit": 0},
                        {**self.request, "semantic_weight": float("nan")},
                        {**self.request, "semantic_weight": 2},
                        {**self.request, "extra": 1}]:
            with self.subTest(payload=payload):
                self.assertEqual(app.search(payload)["status"], "error")

    def test_invalid_products(self):
        for products in [[{}], [False], [{"id": "a", "title": "x", "tags": "tag"}],
                         [{"id": "a", "title": "x", "description": None}],
                         [{"id": "a", "title": "x"}, {"id": " a ", "title": "y"}]]:
            with self.subTest(products=products):
                self.request["products"] = products
                self.assertEqual(app.search(self.request)["status"], "error")

    def test_injected_synonym_embeddings(self):
        self.request.update(query="rucksack", semantic_weight=0.8)
        calls = []

        def synthetic_embedder(texts):
            calls.append(texts)
            return [[1, 0] if "rucksack" in t or "backpack" in t else [0, 1]
                    for t in texts]

        result = app.search(self.request, synthetic_embedder)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([m["id"] for m in result["matches"]], ["SYN-001"])
        self.assertEqual(result["matches"][0]["score"], 0.8)
        self.assertEqual(result["matches"][0]["lexical_score"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)

    def test_embedding_validation(self):
        self.request["semantic_weight"] = 1
        invalid = [None, [], [[1, 0]] * 3, [[0, 0]] * 4,
                   [[1, 0], [1], [1, 0], [1, 0]],
                   [[float("inf"), 0]] * 4, [[float("nan"), 0]] * 4,
                   [[True, 0]] * 4, [["bad", 0]] * 4, [[10 ** 1000, 0]] * 4]
        for vectors in invalid:
            with self.subTest(vectors=vectors):
                result = app.search(self.request, lambda texts: vectors)
                self.assertEqual(result["status"], "error")
        self.assertEqual(app.search(self.request)["status"], "error")

    def test_embedding_failure_and_disabled_interface(self):
        def broken(texts):
            raise RuntimeError("synthetic failure")
        self.assertEqual(app.search(self.request, broken)["status"], "ok")
        self.request["semantic_weight"] = 0.5
        self.assertEqual(app.search(self.request, broken)["status"], "error")

    def test_cosine_large_values_and_negative_similarity(self):
        self.request["semantic_weight"] = 1
        result = app.search(self.request, lambda texts: [
            [1e308, 1e308], [1e308, 1e308], [-1, -1], [1, -1]])
        self.assertEqual(result["status"], "ok")
        self.assertEqual([m["id"] for m in result["matches"]], ["SYN-001"])
        self.assertEqual(result["matches"][0]["score"], 1.0)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), app.search(self.request))
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_usage_and_json_errors(self):
        for arguments in [[], ["does-not-exist.json"], ["implementation.py"],
                          ["example_input.json", "extra"]]:
            with self.subTest(arguments=arguments):
                process = subprocess.run(
                    [sys.executable, "-B", "implementation.py", *arguments],
                    cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_rejects_nonstandard_and_duplicate_json(self):
        for data in ['{"query":"a","query":"b"}', '{"semantic_weight":NaN}',
                     '{"semantic_weight":Infinity}', "[]"]:
            with self.subTest(data=data), patch.object(
                    Path, "open", return_value=io.StringIO(data)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-memory-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_empty_catalog_does_not_invoke_embedder(self):
        self.request.update(products=[], semantic_weight=1)
        def unused(texts):
            self.fail("empty corpus should not call embeddings")
        self.assertEqual(app.search(self.request, unused)["matches"], [])


if __name__ == "__main__":
    unittest.main()
