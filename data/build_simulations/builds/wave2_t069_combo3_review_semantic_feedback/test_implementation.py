import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_review_supported_partial_gap(self):
        review = impl.review_stage(self.data)["review"]
        self.assertEqual([check["status"] for check in review["checks"]], ["supported", "partial", "gap"])
        self.assertEqual(review["checks"][1]["missing_terms"], ["return label"])
        self.assertIn("not certification", review["notice"])

    def test_evidence_is_exact_source_excerpt(self):
        result = impl.review_stage(self.data)
        docs = {doc["id"]: doc["text"] for doc in self.data["documents"]}
        for check in result["review"]["checks"]:
            for evidence in check["evidence"]:
                self.assertEqual(evidence["excerpt"], docs[evidence["document_id"]][evidence["start"]:evidence["end"]])

    def test_phrase_matching_word_boundaries(self):
        self.assertIsNone(impl.find_phrase("batteries and batteryish", "battery"))
        self.assertEqual(impl.find_phrase("A RETURN,\n label.", "return label"), (2, 16))

    def test_semantic_index_and_ranking(self):
        result = impl.semantic_stage(impl.review_stage(self.data))
        self.assertEqual(len(result["semantic"]["index"]), 3)
        self.assertEqual(len(result["semantic"]["hits"]), 2)
        self.assertEqual(result["semantic"]["hits"][0]["score"], 1)
        self.assertEqual(result["semantic"]["hits"][0]["evidence_id"], "r0d0t0")
        self.assertEqual({hit["document_id"] for hit in result["semantic"]["hits"]}, {"doc-battery"})

    def test_query_limit(self):
        self.data["query"]["limit"] = 1
        result = impl.run_pipeline(self.data)
        self.assertEqual(len(result["semantic"]["hits"]), 1)
        self.assertTrue(all(len(group["evidence_ids"]) == 1 for group in result["feedback"]["groups"]))

    def test_dedup_and_theme_counts(self):
        result = impl.run_pipeline(self.data)["feedback"]
        self.assertEqual(result["duplicate_count"], 1)
        self.assertEqual(result["groups"][0]["source_ids"], ["fb-1", "fb-2"])
        self.assertEqual(result["themes"][0]["unique_feedback_count"], 1)
        self.assertEqual(result["excluded_source_ids"], ["fb-4"])
        self.assertEqual(len(result["groups"][0]["supporting_excerpts"]), 2)

    def test_feedback_provenance(self):
        result = impl.run_pipeline(self.data)
        sources = {item["id"]: item for item in self.data["feedback"]}
        hits = {hit["evidence_id"] for hit in result["semantic"]["hits"]}
        for group in result["feedback"]["groups"]:
            self.assertTrue(set(group["evidence_ids"]) <= hits)
            for quote in group["supporting_excerpts"]:
                source = sources[quote["feedback_id"]]
                self.assertEqual(source["document_id"], quote["document_id"])
                self.assertEqual(source["text"][quote["start"]:quote["end"]], quote["excerpt"])

    def test_query_propagates_to_feedback(self):
        self.data["query"]["text"] = "recycled paper"
        result = impl.run_pipeline(self.data)
        self.assertEqual([group["id"] for group in result["feedback"]["groups"]], ["fb-4"])
        self.assertEqual(result["feedback"]["themes"][0]["id"], "other")

    def test_review_gap_removes_search_and_feedback(self):
        self.data["requirements"] = [{"id": "missing", "description": "Synthetic absent term", "terms": ["nonexistent"]}]
        result = impl.run_pipeline(self.data)
        self.assertEqual(result["semantic"]["index"], [])
        self.assertEqual(result["feedback"]["groups"], [])
        self.assertEqual(len(result["feedback"]["excluded_source_ids"]), 4)

    def test_no_query_matches(self):
        self.data["query"]["text"] = "unfindable"
        result = impl.run_pipeline(self.data)
        self.assertEqual(result["semantic"]["hits"], [])
        self.assertEqual(result["feedback"]["themes"], [])

    def test_empty_collections(self):
        for name in ("documents", "requirements", "feedback", "themes"):
            self.data[name] = []
        result = impl.run_pipeline(self.data)
        self.assertEqual(result["review"]["checks"], [])
        self.assertEqual(result["feedback"]["groups"], [])

    def test_input_not_mutated_and_deterministic(self):
        before = copy.deepcopy(self.data)
        one = impl.run_pipeline(self.data)
        self.assertEqual(self.data, before)
        self.assertEqual(one, impl.run_pipeline(self.data))

    def test_invalid_inputs(self):
        for change in (
            lambda d: d.update(synthetic=False),
            lambda d: d.update(schema_version=True),
            lambda d: d["query"].update(limit=True),
            lambda d: d["query"].update(text="???"),
            lambda d: d["feedback"][0].update(document_id="unknown"),
            lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
            lambda d: d["requirements"][0].update(terms=["battery", "BATTERY!"]),
            lambda d: d["themes"][0].update(id="other"),
            lambda d: d.update(extra=1),
            lambda d: d.update(documents={}),
        ):
            data = copy.deepcopy(self.data)
            change(data)
            with self.subTest(data=data), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(data)

    def test_review_handoff_rejects_altered_evidence(self):
        review = impl.review_stage(self.data)
        review["review"]["checks"][0]["evidence"][0]["excerpt"] = "fabricated"
        with self.assertRaises(impl.ValidationError):
            impl.semantic_stage(review)

    def test_semantic_handoff_rejects_altered_links(self):
        result = impl.semantic_stage(impl.review_stage(self.data))
        result["semantic"]["hits"][0]["document_id"] = "doc-packaging"
        with self.assertRaises(impl.ValidationError):
            impl.feedback_stage(result)

    def test_final_validation_rejects_altered_feedback(self):
        result = impl.run_pipeline(self.data)
        result["feedback"]["groups"][0]["supporting_excerpts"][0]["excerpt"] = "fabricated"
        with self.assertRaises(impl.ValidationError):
            impl.validate(result, "feedback")

    def test_injected_embedding_fixture(self):
        self.data["query"]["text"] = "power"
        calls = []

        def fixture(texts):
            calls.append(texts)
            return [[1.0, 0.0]] + [[1.0, 0.0] if "battery" in value.lower() else [0.0, 1.0] for value in texts[1:]]

        result = impl.run_pipeline(self.data, fixture)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["semantic"]["mode"], "hybrid")
        self.assertEqual(len(result["semantic"]["hits"]), 2)
        self.assertEqual(result["semantic"]["hits"][0]["score"], 0.5)
        self.assertEqual(len(result["feedback"]["groups"]), 2)

    def test_invalid_embedding_outputs(self):
        fixtures = [
            lambda texts: [],
            lambda texts: [[0, 0] for _ in texts],
            lambda texts: [[float("nan")] for _ in texts],
            lambda texts: [[True] for _ in texts],
            lambda texts: [[1], [1, 2], [1], [1]],
            lambda texts: "not vectors",
        ]
        for fixture in fixtures:
            with self.subTest(fixture=fixture), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(self.data, fixture)

    def test_embedding_exception_is_validation_error(self):
        def fixture(texts):
            raise RuntimeError("synthetic failure")
        with self.assertRaisesRegex(impl.ValidationError, "embedding callable failed"):
            impl.run_pipeline(self.data, fixture)

    def test_large_embedding_values_are_stable(self):
        result = impl.run_pipeline(self.data, lambda texts: [[1e308, 1e308] for _ in texts])
        self.assertEqual(result["status"], "ok")

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False, cwd=ROOT,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["stage"], "feedback")
        self.assertEqual(completed.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                capture_output=True, text=True, check=False, cwd=ROOT,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for contents in ('{', '{"schema_version": 1, "schema_version": 1}', '{"x": NaN}', '{}'):
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=contents), redirect_stdout(output):
                code = impl.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_error(self):
        output = io.StringIO()
        with patch.object(Path, "read_text", side_effect=UnicodeError("synthetic encoding failure")), redirect_stdout(output):
            code = impl.main(["synthetic-invalid.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_late_theme_terms_are_in_supporting_excerpts(self):
        self.data["feedback"] = [{
            "id": "late", "document_id": "doc-battery",
            "text": "Synthetic neutral words. " * 30 + "replacement" + " neutral" * 40 + " broken",
        }]
        group = impl.run_pipeline(self.data)["feedback"]["groups"][0]
        excerpts = {quote["theme_id"]: quote for quote in group["supporting_excerpts"]}
        self.assertIn("replacement", excerpts["maintenance"]["excerpt"])
        self.assertIn("broken", excerpts["durability"]["excerpt"])
        self.assertGreater(excerpts["maintenance"]["start"], 240)

    def test_duplicates_across_selected_documents_preserve_links(self):
        self.data["query"]["text"] = "synthetic"
        self.data["feedback"][3]["text"] = self.data["feedback"][0]["text"]
        group = impl.run_pipeline(self.data)["feedback"]["groups"][0]
        self.assertEqual(group["source_ids"], ["fb-1", "fb-2", "fb-4"])
        self.assertEqual(group["document_ids"], ["doc-battery", "doc-packaging"])
        self.assertEqual(len(group["evidence_ids"]), 3)

    def test_empty_index_does_not_invoke_embedder(self):
        self.data["requirements"] = []

        def fixture(texts):
            self.fail("Empty evidence must not invoke the embedder")

        result = impl.run_pipeline(self.data, fixture)
        self.assertEqual(result["semantic"]["hits"], [])

    def test_wrong_stage_is_rejected(self):
        with self.assertRaises(impl.ValidationError):
            impl.feedback_stage(impl.review_stage(self.data))


if __name__ == "__main__":
    unittest.main()
