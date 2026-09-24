"""Synthetic fixtures only; tests do not create files or call external services."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    return json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))


def items(result, stage):
    return result["stages"][stage]["items"]


class PipelineTests(unittest.TestCase):
    def test_example_integrated(self):
        result = app.run_pipeline(fixture())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), ["insights", "semantic", "review", "guided"])
        self.assertEqual([r["status"] for r in items(result, "review")], ["supported", "gap"])
        self.assertEqual([s["status"] for s in items(result, "guided")],
                         ["completed", "ready", "blocked"])

    def test_themes_deduplicate_feedback_occurrences(self):
        raw = fixture()
        raw["feedback"][0]["text"] += " Delivery delivery."
        theme = next(t for t in items(app.run_pipeline(raw), "insights") if t["keyword"] == "delivery")
        self.assertEqual(theme["count"], 2)
        self.assertEqual(theme["feedback_ids"], ["f1", "f2"])
        self.assertEqual(theme["sentiment"], "mixed")

    def test_determinism_and_no_input_mutation(self):
        raw = fixture()
        before = copy.deepcopy(raw)
        self.assertEqual(app.run_pipeline(raw), app.run_pipeline(raw))
        self.assertEqual(raw, before)

    def test_theme_limit(self):
        raw = fixture()
        raw["search"]["max_themes"] = 1
        result = app.run_pipeline(raw)
        self.assertEqual(len(items(result, "insights")), 1)
        self.assertEqual(len(items(result, "semantic")), 1)
        self.assertGreater(result["stages"]["insights"]["summary"]["omitted_keyword_count"], 0)

    def test_empty_collections(self):
        raw = {"schema_version": 1, "synthetic": True, "feedback": [], "documents": [],
               "requirements": [], "steps": []}
        result = app.run_pipeline(raw)
        self.assertEqual(result["stages"]["guided"]["summary"]["percent_complete"], 100)
        self.assertTrue(all(not s["items"] for s in result["stages"].values()))

    def test_stopwords_and_punctuation_produce_no_themes(self):
        raw = fixture()
        raw["feedback"] = [{"id": "empty", "text": "The and !!! good"}]
        result = app.run_pipeline(raw)
        self.assertEqual(items(result, "insights"), [])
        self.assertEqual(items(result, "review")[0]["gap"]["reason"], "no_retrieved_documents")
        self.assertEqual(items(result, "guided")[1]["status"], "blocked")

    def test_search_index_and_provenance(self):
        result = app.run_pipeline(fixture())
        self.assertEqual(result["stages"]["semantic"]["summary"]["index"]["tracking"], ["delivery-guide"])
        themes = {t["id"]: t for t in items(result, "insights")}
        for query in items(result, "semantic"):
            self.assertEqual(query["feedback_ids"], themes[query["theme_id"]]["feedback_ids"])
            self.assertEqual(query["hits"], sorted(query["hits"], key=lambda h: (-h["score"], h["document_id"])))

    def test_search_ties_top_k(self):
        raw = fixture()
        raw["documents"] = [{"id": key, "title": "delivery", "text": "delivery tracking"}
                            for key in ("z", "a", "b")]
        raw["search"]["top_k"] = 2
        result = app.run_pipeline(raw)
        query = next(q for q in items(result, "semantic") if q["query"] == "delivery")
        self.assertEqual([h["document_id"] for h in query["hits"]], ["a", "b"])

    def test_zero_scores_excluded_even_at_zero_threshold(self):
        raw = fixture()
        raw["documents"] = [{"id": "x", "title": "zebra", "text": "zebra"}]
        raw["search"]["min_score"] = 0
        self.assertTrue(all(not q["hits"] for q in items(app.run_pipeline(raw), "semantic")))

    def test_empty_documents(self):
        raw = fixture()
        raw["documents"] = []
        self.assertTrue(all(r["status"] == "gap" for r in items(app.run_pipeline(raw), "review")))

    def test_injected_embedding_enables_synonym_retrieval(self):
        raw = fixture()
        raw["feedback"] = [{"id": "f", "text": "shipping"}]
        raw["search"]["embedding_weight"] = 1
        calls = []

        def embed(texts):
            calls.append(texts)
            return [[1.0, 0.0] if ("shipping" in text or "Delivery" in text)
                    else [0.0, 1.0] for text in texts]

        result = app.run_pipeline(raw, embed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(items(result, "semantic")[0]["hits"][0]["document_id"], "delivery-guide")
        self.assertEqual(items(result, "review")[0]["status"], "supported")
        self.assertTrue(result["stages"]["semantic"]["summary"]["embedding_used"])

    def test_invalid_embeddings(self):
        invalid = [[], "bad", [[1]], [[0, 0]] * 7, [[True, 1]] * 7,
                   [[float("nan"), 1]] * 7, [[float("inf"), 1]] * 7,
                   [[1, 2]] * 6 + [[1]], [[10**400, 1]] * 7]
        for value in invalid:
            with self.subTest(value=repr(value)[:50]), self.assertRaises(app.ValidationError):
                app.run_pipeline(fixture(), lambda texts: value)

    def test_embedding_exception_is_validation_error(self):
        def failing(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "injected callable failed"):
            app.run_pipeline(fixture(), failing)

    def test_no_embedding_call_without_queries(self):
        raw = fixture()
        raw["feedback"] = []
        app.run_pipeline(raw, lambda _: self.fail("should not call embedder"))

    def test_evidence_offsets_and_search_trace(self):
        raw = fixture()
        result = app.run_pipeline(raw)
        evidence = items(result, "review")[0]["evidence"][0]
        doc = next(d for d in raw["documents"] if d["id"] == evidence["document_id"])
        self.assertEqual(evidence["quote"], doc["text"][evidence["start"]:evidence["end"]])
        queries = {q["id"]: q for q in items(result, "semantic")}
        self.assertTrue(evidence["query_ids"])
        for key in evidence["query_ids"]:
            self.assertIn(evidence["document_id"], [h["document_id"] for h in queries[key]["hits"]])

    def test_gap_trace_missing_terms(self):
        gap = items(app.run_pipeline(fixture()), "review")[1]["gap"]
        candidate = next(c for c in gap["candidate_checks"] if c["document_id"] == "refund-guide")
        self.assertEqual(candidate["missing_terms"], ["deadline"])
        self.assertEqual(gap["needed_count"], 1)

    def test_duplicate_search_hits_count_as_one_document(self):
        raw = fixture()
        raw["requirements"][0]["min_evidence"] = 2
        reviewed = items(app.run_pipeline(raw), "review")[0]
        self.assertEqual(len(reviewed["evidence"]), 1)
        self.assertEqual(reviewed["status"], "gap")
        self.assertEqual(reviewed["gap"]["needed_count"], 1)

    def test_review_does_not_use_unretrieved_documents(self):
        raw = fixture()
        raw["feedback"] = [{"id": "f", "text": "furniture"}]
        result = app.run_pipeline(raw)
        self.assertEqual(items(result, "review")[0]["evidence"], [])
        self.assertEqual(items(result, "guided")[1]["status"], "blocked")

    def test_title_is_not_body_evidence(self):
        raw = fixture()
        raw["documents"][0]["title"] = "Delivery tracking"
        raw["documents"][0]["text"] = "Shipment notifications."
        self.assertEqual(items(app.run_pipeline(raw), "review")[0]["status"], "gap")

    def test_exact_words_not_substrings(self):
        raw = fixture()
        raw["requirements"][0]["required_terms"] = ["track"]
        self.assertEqual(items(app.run_pipeline(raw), "review")[0]["status"], "gap")

    def test_casefold_unicode(self):
        raw = fixture()
        raw["feedback"] = [{"id": "f", "text": "CAFÉ Straße"}]
        raw["documents"] = [{"id": "d", "title": "café", "text": "Café STRASSE"}]
        raw["requirements"][0]["required_terms"] = ["café", "straße"]
        self.assertEqual(items(app.run_pipeline(raw), "review")[0]["status"], "supported")

    def test_review_explicitly_disclaims_negation_and_certification(self):
        raw = fixture()
        raw["documents"][0]["text"] = "Delivery tracking is not available."
        result = app.run_pipeline(raw)
        notice = result["stages"]["review"]["summary"]["notice"]
        self.assertIn("no certification", notice)
        self.assertIn("Negation requires human review", notice)

    def test_gap_repair_propagates_to_guided(self):
        raw = fixture()
        raw["documents"][1]["text"] += " Refund deadline is 30 days."
        raw["completed_step_ids"].append("delivery")
        result = app.run_pipeline(raw)
        self.assertEqual(items(result, "review")[1]["status"], "supported")
        self.assertEqual(items(result, "guided")[2]["status"], "ready")
        self.assertEqual(result["stages"]["guided"]["summary"]["percent_complete"], 66.67)

    def test_invalid_completion_rejected(self):
        for complete in (["delivery"], ["register", "delivery", "refund"]):
            raw = fixture()
            raw["completed_step_ids"] = complete
            with self.subTest(complete=complete), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)

    def test_ready_prerequisite_does_not_count_as_completed(self):
        raw = fixture()
        raw["completed_step_ids"] = []
        self.assertEqual([s["status"] for s in items(app.run_pipeline(raw), "guided")],
                         ["ready", "blocked", "blocked"])

    def test_topological_order_independent_of_input(self):
        raw = fixture()
        raw["steps"].reverse()
        self.assertEqual([s["id"] for s in items(app.run_pipeline(raw), "guided")],
                         ["register", "delivery", "refund"])

    def test_cycles_and_unknown_references(self):
        raw = fixture()
        raw["steps"][0]["prerequisites"] = ["refund"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            app.run_pipeline(raw)
        raw = fixture()
        raw["steps"][0]["requirement_ids"] = ["unknown"]
        with self.assertRaisesRegex(app.ValidationError, "unknown reference"):
            app.run_pipeline(raw)

    def test_duplicate_ids_and_references(self):
        raw = fixture()
        raw["feedback"].append(raw["feedback"][0])
        with self.assertRaisesRegex(app.ValidationError, "duplicate id"):
            app.run_pipeline(raw)
        raw = fixture()
        raw["completed_step_ids"] = ["register", "register"]
        with self.assertRaisesRegex(app.ValidationError, "duplicate references"):
            app.run_pipeline(raw)

    def test_invalid_input_shapes_and_bounds(self):
        variants = [
            ("schema_version", True), ("schema_version", 2), ("synthetic", "true"),
            ("feedback", {}), ("documents", None), ("steps", "steps"),
            ("search", {"top_k": True}), ("search", {"top_k": 0}),
            ("search", {"min_score": float("nan")}), ("search", {"embedding_weight": 2}),
            ("search", {"max_themes": 31}), ("search", {"unexpected": 1}),
        ]
        for key, value in variants:
            raw = fixture()
            raw[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)

    def test_invalid_terms_and_text(self):
        for terms in ([], ["delivery tracking"], ["tracking", "TRACKING"], [42]):
            raw = fixture()
            raw["requirements"][0]["required_terms"] = terms
            with self.subTest(terms=terms), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)
        raw = fixture()
        raw["feedback"][0]["text"] = " "
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(raw)

    def test_tampered_insight_boundary_rejected(self):
        raw = fixture()
        previous = app.insights(raw)
        previous["items"][0]["feedback_ids"] = ["unknown"]
        with self.assertRaises(app.ValidationError):
            app.semantic(raw, previous)

    def test_tampered_search_boundary_rejected(self):
        raw = fixture()
        a = app.insights(raw)
        b = app.semantic(raw, a)
        b["items"][0]["hits"][0]["text"] = "fabricated"
        with self.assertRaisesRegex(app.ValidationError, "provenance"):
            app.review(raw, b, a)

    def test_tampered_review_boundary_rejected(self):
        raw = fixture()
        result = app.run_pipeline(raw)
        c = result["stages"]["review"]
        c["items"][0]["evidence"][0]["quote"] = "fabricated"
        with self.assertRaisesRegex(app.ValidationError, "quote"):
            app.guided(raw, c, result["stages"]["semantic"])


class CLITests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=20)

    def test_cli_success_single_json(self):
        process = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file(self):
        process = self.cli(str(ROOT / "nonexistent.json"))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_wrong_arguments(self):
        for args in ((), ("a", "b")):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_malformed_and_invalid_inputs_without_files(self):
        for content in (b"{", b"\xff", b"[]", b"null", b'{"x": NaN}',
                        b'{"schema_version":1,"schema_version":1}',
                        b"x" * (app.MAX_INPUT_BYTES + 1)):
            with self.subTest(content=content[:30]):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_schema_error(self):
        raw = fixture()
        raw["search"]["top_k"] = 0
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data=json.dumps(raw).encode())), contextlib.redirect_stdout(output):
            code = app.main(["synthetic-fixture.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
