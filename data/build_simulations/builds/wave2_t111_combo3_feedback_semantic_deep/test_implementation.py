"""All data, annotations, and injected vectors in this suite are synthetic."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


def fixture():
    return json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))


class PipelineTests(unittest.TestCase):
    def test_deduplication_and_exact_excerpts(self):
        output = impl.run_pipeline(fixture())["feedback_analysis"]
        self.assertEqual((output["raw_count"], output["unique_count"],
                          output["duplicate_count"]), (4, 3, 1))
        group = output["groups"][0]
        self.assertEqual(group["source_ids"], ["f1", "f2"])
        self.assertEqual(group["excerpts"][1]["text"], fixture()["feedback"][1]["text"])

    def test_themes_count_unique_groups_not_duplicates(self):
        analysis = impl.run_pipeline(fixture())["feedback_analysis"]
        quality = next(theme for theme in analysis["themes"] if theme["theme_id"] == "quality")
        self.assertEqual(quality["unique_feedback_count"], 2)
        self.assertEqual(quality["raw_feedback_count"], 3)
        self.assertEqual(len(quality["supporting_excerpts"]), 3)

    def test_index_ranking_and_exclusion(self):
        search = impl.run_pipeline(fixture())["semantic_search"]
        self.assertEqual(search["index"]["durable"][0]["document_id"], "d1")
        self.assertEqual(search["results"][0]["document_id"], "d1")
        self.assertNotIn("d4", [row["document_id"] for row in search["results"]])
        self.assertEqual([r["rank"] for r in search["results"]], [1, 2, 3])

    def test_feedback_changes_search_and_research(self):
        payload = fixture()
        before = impl.run_pipeline(payload)
        payload["feedback"] = [{"id": "new", "text": "Galaxies emit radiation"}]
        after = impl.run_pipeline(payload)
        self.assertNotEqual(before["semantic_search"]["effective_query"],
                            after["semantic_search"]["effective_query"])
        self.assertIn("d4", [r["document_id"] for r in after["semantic_search"]["results"]])
        self.assertIn("d4", [e["document_id"] for e in after["deep_research"]["evidence"]])

    def test_research_disagreement_and_provenance(self):
        payload = fixture()
        output = impl.run_pipeline(payload)
        research = output["deep_research"]
        self.assertEqual(len(research["disagreements"]), 2)
        self.assertTrue(all(d["cross_document"] for d in research["disagreements"]))
        documents = {d["id"]: d for d in payload["documents"]}
        ranked = {r["document_id"]: r for r in output["semantic_search"]["results"]}
        for evidence in research["evidence"]:
            self.assertIn(evidence["excerpt"], documents[evidence["document_id"]]["text"])
            self.assertEqual(evidence["search_rank"], ranked[evidence["document_id"]]["rank"])
            self.assertEqual(evidence["feedback_theme_ids"],
                             ranked[evidence["document_id"]]["feedback_theme_ids"])
        self.assertIn("explicit_uncertainty",
                      {q["code"] for q in research["unresolved_questions"]})

    def test_limit_propagates_and_reports_incomplete_coverage(self):
        payload = fixture()
        payload["limit"] = 1
        output = impl.run_pipeline(payload)
        self.assertEqual(len(output["semantic_search"]["results"]), 1)
        self.assertEqual(output["deep_research"]["summary"]["retrieved_document_count"], 1)
        self.assertEqual(output["deep_research"]["disagreements"], [])
        self.assertIn("retrieval_truncated",
                      {q["code"] for q in output["deep_research"]["unresolved_questions"]})

    def test_no_matches_reports_no_evidence(self):
        payload = fixture()
        payload["query"] = "quasar"
        payload["feedback"] = [{"id": "f", "text": "Nebula"}]
        output = impl.run_pipeline(payload)
        self.assertEqual(output["semantic_search"]["results"], [])
        self.assertEqual(output["deep_research"]["evidence"], [])
        self.assertEqual(output["deep_research"]["unresolved_questions"][0]["code"], "no_evidence")

    def test_unstructured_document_is_not_invented_claim(self):
        payload = fixture()
        payload["documents"] = [{"id": "u", "title": "Quality", "text": "Quality is sturdy."}]
        research = impl.run_pipeline(payload)["deep_research"]
        self.assertEqual(research["evidence"][0]["position"], "observation")
        self.assertIsNone(research["evidence"][0]["claim_number"])
        self.assertEqual(research["findings"][0]["assessment"], "descriptive")

    def test_deterministic_order_for_score_ties(self):
        payload = fixture()
        payload["documents"] = [{"id": key, "title": "Quality", "text": "Durable quality."}
                                for key in ("z", "a")]
        output = impl.run_pipeline(payload)
        self.assertEqual([r["document_id"] for r in output["semantic_search"]["results"]], ["a", "z"])
        self.assertEqual(output, impl.run_pipeline(payload))

    def test_optional_embedding_and_feedback_query(self):
        calls = []
        def embedding(value):
            calls.append(value)
            return [1.0, 0.0]
        output = impl.run_pipeline(fixture(), embedding=embedding)
        self.assertEqual(len(calls), 5)
        self.assertIn("Feedback:", calls[0])
        self.assertIn("shipping", calls[0])
        self.assertEqual(output["semantic_search"]["method"], "lexical+injected-cosine")
        self.assertIn("d4", [r["document_id"] for r in output["semantic_search"]["results"]])
        impl.validate_state(output, 3)

    def test_bad_embedding_vectors(self):
        for vector in ([], [0, 0], [float("nan")], [float("inf")], ["1"], [True], None):
            with self.subTest(vector=vector), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(fixture(), embedding=lambda value: vector)

    def test_embedding_dimension_and_provider_failure(self):
        values = iter([[1, 0], [1]])
        with self.assertRaisesRegex(impl.ValidationError, "dimension"):
            impl.run_pipeline(fixture(), embedding=lambda value: next(values))
        def broken(value):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(impl.ValidationError, "callable failed"):
            impl.run_pipeline(fixture(), embedding=broken)

    def test_embedding_large_finite_values_and_negative_cosine(self):
        values = iter([[1e308, 1e308]] + [[-1e308, -1e308]] * 4)
        output = impl.run_pipeline(fixture(), embedding=lambda value: next(values))
        self.assertEqual(output["semantic_search"]["semantic_scores"]["d1"], -1.0)
        self.assertNotIn("d4", [r["document_id"] for r in output["semantic_search"]["results"]])

    def test_stage_order_and_tampered_feedback(self):
        initial = impl.initialize(fixture())
        with self.assertRaises(impl.ValidationError):
            impl.research_stage(initial)
        feedback = impl.feedback_stage(initial)
        feedback["feedback_analysis"]["groups"][0]["source_ids"] = ["missing"]
        with self.assertRaises(impl.ValidationError):
            impl.semantic_stage(feedback)

    def test_tampered_search_and_research_rejected(self):
        search = impl.semantic_stage(impl.feedback_stage(impl.initialize(fixture())))
        search["semantic_search"]["results"][0]["document_id"] = "missing"
        with self.assertRaises(impl.ValidationError):
            impl.research_stage(search)
        output = impl.run_pipeline(fixture())
        output["deep_research"]["evidence"][0]["excerpt"] = "fabricated"
        with self.assertRaises(impl.ValidationError):
            impl.validate_state(output, 3)

    def test_invalid_input_shapes_and_limits(self):
        for name, value in (("feedback", []), ("documents", []), ("query", "!!!"),
                            ("synthetic", False), ("schema_version", "2"),
                            ("limit", True), ("limit", 0), ("limit", 21),
                            ("documents", "bad"), ("query", 3)):
            payload = fixture()
            payload[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(payload)
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline([])

    def test_duplicate_ids_and_false_citations_rejected(self):
        for collection in ("feedback", "documents"):
            payload = fixture()
            payload[collection][1]["id"] = payload[collection][0]["id"]
            with self.assertRaises(impl.ValidationError):
                impl.run_pipeline(payload)
        payload = fixture()
        payload["documents"][0]["claims"][0]["excerpt"] = "not in document"
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(payload)

    def test_unknown_fields_and_invalid_positions(self):
        payload = fixture()
        payload["extra"] = "not allowed"
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(payload)
        payload = fixture()
        payload["documents"][0]["claims"][0]["position"] = "truth"
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(payload)

    def test_input_is_not_mutated(self):
        payload = fixture()
        original = copy.deepcopy(payload)
        output = impl.run_pipeline(payload)
        self.assertEqual(payload, original)
        output["request"]["feedback"][0]["text"] = "changed"
        self.assertEqual(payload, original)

    def test_internal_conflict_not_mislabeled_cross_document(self):
        payload = fixture()
        payload["documents"] = [
            {"id": "one", "title": "Quality", "text": "Quality varies.",
             "claims": [{"topic": "quality", "position": position,
                         "excerpt": "Quality varies."} for position in ("support", "oppose")]},
            {"id": "two", "title": "Quality", "text": "Quality unknown.",
             "claims": [{"topic": "quality", "position": "uncertain",
                         "excerpt": "Quality unknown."}]},
        ]
        output = impl.run_pipeline(payload)
        self.assertFalse(output["deep_research"]["disagreements"][0]["cross_document"])

    def test_shared_validation_rejects_boolean_for_numeric_count(self):
        state = impl.feedback_stage(impl.initialize(fixture()))
        state["feedback_analysis"]["duplicate_count"] = True
        with self.assertRaises(impl.ValidationError):
            impl.semantic_stage(state)

    def test_unicode_normalization_and_other_theme(self):
        payload = fixture()
        payload["feedback"] = [{"id": "u1", "text": "CAFÉ pleasant!"},
                               {"id": "u2", "text": "café pleasant."}]
        analysis = impl.run_pipeline(payload)["feedback_analysis"]
        self.assertEqual(analysis["unique_count"], 1)
        self.assertEqual(analysis["themes"][0]["theme_id"], "other")

    def test_cli_success_actual_process(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_missing_file_and_usage_actual_process(self):
        for args in ([], [str(ROOT / "does_not_exist.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stderr, "")
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema_without_scratch_files(self):
        for contents in ('{', '[]', '{"query": NaN}', '{"x": 1, "x": 2}'):
            with patch("builtins.open", mock_open(read_data=contents)), \
                    patch("sys.stdout", new_callable=io.StringIO) as stdout:
                code = impl.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
