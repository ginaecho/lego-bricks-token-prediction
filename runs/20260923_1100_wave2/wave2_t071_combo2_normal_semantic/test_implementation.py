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


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def test_research_exact_citations(self):
        result = app.research_stage(self.request)
        sources = {s["source_id"]: s for s in self.request["sources"]}
        self.assertEqual(len(result["findings"]), 3)
        for finding in result["findings"]:
            c = finding["citation"]
            self.assertEqual(c["quote"], sources[c["source_id"]]["text"][c["start"]:c["end"]])
            self.assertEqual(finding["text"], c["quote"])
        self.assertEqual(result["findings"][0]["citation"]["start"], 2)
        self.assertEqual(result["findings"][0]["score"], 1)

    def test_semantic_ranking(self):
        output = app.run_pipeline(self.request)
        self.assertEqual(output["semantic"]["indexed_products"], 2)
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "pack-a")
        self.assertEqual(output["semantic"]["ranking_mode"], "lexical")

    def test_provenance_propagates(self):
        output = app.run_pipeline(self.request)
        for result in output["semantic"]["results"]:
            findings = [f for f in output["research"]["findings"]
                        if f["product_id"] == result["product_id"]]
            self.assertEqual(result["finding_ids"], [f["finding_id"] for f in findings])
            self.assertEqual(result["citations"], [f["citation"] for f in findings])

    def test_research_limit_gates_index(self):
        self.request["research_limit"] = 1
        output = app.run_pipeline(self.request)
        self.assertEqual(output["semantic"]["indexed_products"], 1)
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "pack-a")

    def test_search_limit_does_not_truncate_research(self):
        self.request["search_limit"] = 1
        output = app.run_pipeline(self.request)
        self.assertEqual(len(output["semantic"]["results"]), 1)
        self.assertEqual(len(output["research"]["findings"]), 3)

    def test_no_matches(self):
        self.request["query"] = "telescope"
        output = app.run_pipeline(self.request)
        self.assertEqual(output["research"]["findings"], [])
        self.assertEqual(output["semantic"]["results"], [])

    def test_empty_catalog(self):
        self.request["products"] = []
        self.request["sources"] = []
        self.assertEqual(app.run_pipeline(self.request)["semantic"]["indexed_products"], 0)

    def test_punctuation_query(self):
        self.request["query"] = "?!"
        self.assertEqual(app.run_pipeline(self.request)["semantic"]["results"], [])

    def test_unicode_offsets(self):
        self.request["query"] = "café"
        self.request["sources"][0]["text"] = " \nCafé café! 😀 Next."
        output = app.run_pipeline(self.request)
        citation = output["research"]["findings"][0]["citation"]
        self.assertEqual(citation["quote"], "Café café!")
        self.assertEqual(citation["start"], 2)

    def test_invalid_inputs(self):
        for key, bad_value in (("schema_version", True), ("query", " "),
                               ("products", {}), ("sources", None),
                               ("research_limit", 0), ("search_limit", True),
                               ("embedding_weight", float("nan")),
                               ("synthetic", False)):
            with self.subTest(key=key):
                invalid = copy.deepcopy(self.request)
                invalid[key] = bad_value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(invalid)

    def test_duplicate_and_broken_references(self):
        duplicate = copy.deepcopy(self.request)
        duplicate["sources"].append(duplicate["sources"][0])
        broken = copy.deepcopy(self.request)
        broken["sources"][0]["product_id"] = "missing"
        duplicate_product = copy.deepcopy(self.request)
        duplicate_product["products"].append(duplicate_product["products"][0])
        for request in (duplicate, broken, duplicate_product):
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(request)

    def test_tampered_handoff_rejected(self):
        for key, value in (("quote", "invented"), ("start", -1),
                           ("title", "invented"), ("source_id", "missing")):
            research = app.research_stage(self.request)
            research["findings"][0]["citation"][key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.semantic_stage(self.request, research)

    def test_tampered_final_provenance_rejected(self):
        output = app.run_pipeline(self.request)
        output["semantic"]["results"][0]["finding_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.validate("output", output, self.request)

    def test_injected_embeddings_change_ranking(self):
        calls = []
        def fixture(texts):
            calls.append(texts)
            return [[1, 0], [0, 1], [1, 0]]
        self.request["embedding_weight"] = 1
        output = app.run_pipeline(self.request, fixture)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], self.request["query"])
        self.assertNotIn("ceramic", " ".join(calls[0]))
        self.assertIn("not waterproof", calls[0][2])
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "pack-b")
        self.assertEqual(output["semantic"]["ranking_mode"], "hybrid")

    def test_invalid_embeddings(self):
        for response in ([], [[1], [1]], [[1], [1, 2], [1]],
                         [[1], [float("inf")], [1]], [[True], [1], [1]],
                         [[], [], []], "bad"):
            with self.subTest(response=response), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.request, lambda texts: response)

    def test_embedder_exception_is_validation_error(self):
        def broken(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "embedding callable failed"):
            app.run_pipeline(self.request, broken)

    def test_zero_vectors_and_ties(self):
        self.request["embedding_weight"] = 1
        output = app.run_pipeline(self.request, lambda texts: [[0, 0] for _ in texts])
        self.assertEqual([r["product_id"] for r in output["semantic"]["results"]],
                         ["pack-a", "pack-b"])
        self.assertTrue(all(r["score"] == 0 for r in output["semantic"]["results"]))

    def test_empty_research_does_not_invoke_embedder(self):
        self.request["query"] = "telescope"
        def forbidden(texts):
            self.fail("embedding should not run for an empty index")
        app.run_pipeline(self.request, forbidden)

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.request)
        first = app.run_pipeline(self.request)
        self.request["sources"].reverse()
        self.assertEqual(first, app.run_pipeline(self.request))
        self.request["sources"].reverse()
        self.assertEqual(self.request, original)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                              cwd=HERE, text=True, capture_output=True, check=False)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_file_error(self):
        result = self.cli("missing-fixture.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_usage_error(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_json_and_validation_errors(self):
        for document in ("{", "[]", '{"schema_version": NaN}',
                         json.dumps({**self.request, "query": ""})):
            with self.subTest(document=document):
                stdout = io.StringIO()
                with patch("builtins.open", mock_open(read_data=document)), redirect_stdout(stdout):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
