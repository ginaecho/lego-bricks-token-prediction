import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
from contextlib import redirect_stdout
from io import StringIO

import implementation as impl


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_full_pipeline(self):
        output = impl.run_pipeline(self.data)
        self.assertEqual(output["review"]["evidence_found"], 1)
        self.assertEqual(output["review"]["gaps"], 1)
        self.assertIn("not certification", output["review"]["notice"])

    def test_index_includes_documents(self):
        self.data["query"] = "recyclable"
        self.assertEqual(impl.semantic_search(self.data)["hits"][0]["score"], 1)

    def test_ranking_ties_and_limit(self):
        self.data["products"][1]["description"] = "solar lantern"
        self.data["products"].reverse()
        self.data["limit"] = 1
        self.assertEqual(impl.semantic_search(self.data)["hits"][0]["product_id"],
                         "synthetic-lantern")

    def test_cross_stage_excludes_unretrieved_evidence(self):
        check = impl.run_pipeline(self.data)["review"]["checks"][1]
        self.assertEqual(check["status"], "gap")
        self.assertNotIn("synthetic-mug-note",
                         [d["document_id"] for d in check["examined_documents"]])

    def test_trace_propagation(self):
        output = impl.run_pipeline(self.data)
        evidence = output["review"]["checks"][0]["evidence"][0]
        hit = output["search"]["hits"][0]
        self.assertEqual(evidence["search_score"], hit["score"])
        self.assertEqual(evidence["search_rank"], hit["rank"])
        self.assertIn(evidence["document_id"], hit["document_ids"])

    def test_empty_catalog(self):
        self.data["products"] = []
        output = impl.run_pipeline(self.data)
        self.assertEqual(output["search"]["hits"], [])
        self.assertEqual(output["review"]["gaps"], 2)

    def test_no_match_even_with_zero_threshold(self):
        self.data.update(query="unobtainium", min_score=0)
        self.assertEqual(impl.run_pipeline(self.data)["search"]["hits"], [])

    def test_terms_not_combined_across_documents(self):
        self.data["requirements"][0]["terms"] = ["battery", "recyclable"]
        check = impl.run_pipeline(self.data)["review"]["checks"][0]
        self.assertEqual(check["status"], "gap")
        self.assertTrue(check["partial_evidence"][0]["missing_terms"])

    def test_phrase_and_word_boundaries(self):
        self.assertTrue(impl.contains_phrase("RECHARGEABLE, battery.", "rechargeable battery"))
        self.assertFalse(impl.contains_phrase("waterproofing", "waterproof"))

    def test_injected_embedding_fixture(self):
        seen = []
        def embedding(texts):
            seen.extend(texts)
            return [[1, 0], [0, 1], [1, 0]]
        output = impl.run_pipeline(self.data, embedding)
        self.assertEqual(len(seen), 3)
        self.assertEqual(output["search"]["method"], "injected_embedding")
        self.assertEqual(output["search"]["hits"][0]["product_id"], "synthetic-mug")
        self.assertEqual(output["review"]["checks"][1]["status"], "evidence_found")
        self.assertEqual(output["review"]["checks"][0]["status"], "gap")

    def test_invalid_embeddings(self):
        for vectors in ([], [[1], [1, 0], [1]], [[0], [1], [1]],
                        [[float("nan")], [1], [1]], [[True], [1], [1]]):
            with self.subTest(vectors=vectors), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(self.data, lambda texts: vectors)

    def test_large_embeddings_and_callback_error(self):
        self.assertEqual(len(impl.semantic_search(
            self.data, lambda texts: [[1e308, 1e308]] * 3)["hits"]), 2)
        def broken(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(impl.ValidationError, "callable failed"):
            impl.run_pipeline(self.data, broken)

    def test_invalid_inputs(self):
        variants = [
            {"query": "!!!"}, {"limit": True}, {"min_score": float("inf")},
            {"requirements": []}, {"products": {}}, {"synthetic": False},
            {"schema_version": True}, {"unknown": 1}
        ]
        for change in variants:
            with self.subTest(change=change), self.assertRaises(impl.ValidationError):
                impl.run_pipeline({**self.data, **change})

    def test_duplicate_identifiers_and_terms(self):
        for category in ("product", "document", "requirement", "term"):
            data = copy.deepcopy(self.data)
            if category == "product":
                data["products"].append(copy.deepcopy(data["products"][0]))
            elif category == "document":
                data["products"][1]["documents"][0]["id"] = "synthetic-manual"
            elif category == "requirement":
                data["requirements"].append(copy.deepcopy(data["requirements"][0]))
            else:
                data["requirements"][0]["terms"] = ["Battery", "battery"]
            with self.subTest(category=category), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(data)

    def test_tampered_handoff(self):
        for change in ({"product_id": "unknown"}, {"document_ids": ["synthetic-mug-note"]},
                       {"score": float("nan")}, {"rank": 2}):
            search = impl.semantic_search(self.data)
            search["hits"][0].update(change)
            with self.subTest(change=change), self.assertRaises(impl.ValidationError):
                impl.review_documents(self.data, search)

    def test_input_not_mutated_and_deterministic(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(impl.run_pipeline(self.data), impl.run_pipeline(self.data))
        self.assertEqual(before, self.data)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], ["does-not-exist.json"]):
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_malformed_and_invalid_json(self):
        for value in ("{", '{"x":1,"x":2}', '{"x":NaN}', "[]"):
            output = StringIO()
            with patch("pathlib.Path.open", mock_open(read_data=value)), redirect_stdout(output):
                status = impl.main(["fixture.json"])
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
