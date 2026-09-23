import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_grounded_faq_and_ranked_search(self):
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["faq"]["answer"], self.payload["knowledge_base"][0]["answer"])
        self.assertEqual(output["faq"]["citations"], ["synthetic-faq-boots"])
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "synthetic-ridge")
        self.assertTrue(output["semantic"]["results"][0]["faq_linked"])

    def test_abstains_without_search_results(self):
        self.payload["request"]["query"] = "quantum spaceship"
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["faq"]["status"], "abstained")
        self.assertIsNone(output["faq"]["answer"])
        self.assertEqual(output["faq"]["citations"], [])
        self.assertEqual(output["semantic"]["results"], [])

    def test_abstention_still_searches_original_query(self):
        self.payload["request"]["query"] = "city shoes"
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["faq"]["status"], "abstained")
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "synthetic-city")
        self.assertFalse(output["semantic"]["results"][0]["faq_linked"])

    def test_grounded_keywords_and_product_ids_propagate(self):
        self.payload["request"]["query"] = "recommendation"
        self.payload["knowledge_base"][0]["question"] = "recommendation"
        output = app.run_pipeline(self.payload)
        self.assertIn("waterproof", output["semantic"]["expanded_terms"])
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "synthetic-ridge")
        self.payload["knowledge_base"] = []
        self.assertEqual(app.run_pipeline(self.payload)["semantic"]["results"], [])

    def test_rejects_tampered_handoff(self):
        faq = app.answer_faq(self.payload)
        for mutate in (
            lambda f: f.update(answer="invented answer"),
            lambda f: f["handoff"].update(product_ids=["synthetic-city"]),
            lambda f: f["handoff"].update(grounded_terms=["invented"]),
            lambda f: f["handoff"].update(query="different query"),
        ):
            broken = copy.deepcopy(faq)
            mutate(broken)
            with self.assertRaises(app.ValidationError):
                app.search_products(self.payload, broken)

    def test_synonym_normalization_and_top_k(self):
        self.payload["knowledge_base"] = []
        self.payload["request"].update(query="trail boots", top_k=1)
        result = app.run_pipeline(self.payload)["semantic"]
        self.assertEqual(result["expanded_terms"], ["boot", "hike"])
        self.assertEqual(len(result["results"]), 1)

    def test_deterministic_ties_and_no_mutation(self):
        self.payload["products"].append({
            **self.payload["products"][0], "id": "aaa-tie"
        })
        self.payload["knowledge_base"] = []
        before = copy.deepcopy(self.payload)
        first = app.run_pipeline(self.payload)
        self.assertEqual(first, app.run_pipeline(self.payload))
        self.assertEqual(first["semantic"]["results"][0]["product_id"], "aaa-tie")
        self.assertEqual(self.payload, before)

    def test_empty_catalog_and_knowledge_base(self):
        self.payload["products"] = []
        self.payload["knowledge_base"] = []
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["faq"]["confidence"], 0)
        self.assertEqual(output["semantic"]["results"], [])

    def test_invalid_inputs(self):
        for key, value in (("query", ""), ("query", "the and"), ("query", "!!!"),
                           ("top_k", True), ("top_k", 0), ("top_k", 101),
                           ("faq_min_score", float("nan")),
                           ("search_min_score", -1), ("unknown", 1)):
            payload = copy.deepcopy(self.payload)
            payload["request"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(payload)
        for payload in ([], {}, {"schema_version": 1}):
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(payload)

    def test_bad_references_duplicates_and_synthetic_label(self):
        for mutate in (
            lambda p: p["products"].append(copy.deepcopy(p["products"][0])),
            lambda p: p["knowledge_base"][0].update(product_ids=["missing"]),
            lambda p: p.update(synthetic_fixture=False),
            lambda p: p["knowledge_base"][0].update(answer=" "),
        ):
            payload = copy.deepcopy(self.payload)
            mutate(payload)
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(payload)

    def test_threshold_requires_positive_evidence(self):
        self.payload["request"].update(query="quantum", faq_min_score=0, search_min_score=0)
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["faq"]["status"], "abstained")
        self.assertEqual(output["semantic"]["results"], [])

    def test_optional_embedding_fixture(self):
        self.payload["request"]["query"] = "unrelated concept"
        captured = []

        def embed(texts):
            captured.extend(texts)
            return [[1, 0], [0, 1], [1, 0], [-1, 0]]

        output = app.run_pipeline(self.payload, embed)
        self.assertEqual(len(captured), 4)
        self.assertTrue(output["semantic"]["embedding_used"])
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "synthetic-city")
        self.assertEqual(output["semantic"]["results"][0]["score"], 0.3)

    def test_embedder_receives_faq_enrichment(self):
        self.payload["request"]["query"] = "recommendation"
        self.payload["knowledge_base"][0]["question"] = "recommendation"
        captured = []

        def embed(texts):
            captured.extend(texts)
            return [[1, 0] for _ in texts]

        app.run_pipeline(self.payload, embed)
        self.assertIn("waterproof", captured[0])
        self.assertIn("recommendation", captured[0])

    def test_invalid_embedding_outputs(self):
        for vectors in (None, [], [[1]] * 3, [[0, 0]] * 4, [[True]] * 4,
                        [[float("inf")]] * 4, [[1], [1, 2], [1], [1]]):
            with self.subTest(vectors=vectors), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.payload, lambda texts: vectors)
        with self.assertRaisesRegex(app.ValidationError, "embedder failed"):
            app.run_pipeline(self.payload, lambda texts: 1 / 0)

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["a", "b"]):
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, cwd=ROOT, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for content in ("{broken", '{"x": 1, "x": 2}', '{"x": NaN}', "[]",
                        json.dumps({**self.payload, "synthetic_fixture": False})):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                code = app.main(["fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
