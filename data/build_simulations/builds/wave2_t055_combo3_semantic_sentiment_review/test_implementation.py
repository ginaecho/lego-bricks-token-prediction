"""Offline fixtures only; tests never access other builds or create scratch files."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


def fixture():
    return json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))


class PipelineTests(unittest.TestCase):
    def test_example_and_shared_envelope(self):
        source = fixture()
        result = app.run_pipeline(source)
        self.assertEqual(set(result), set(source))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"], source["data"])
        self.assertEqual(set(result["results"]), {"search", "sentiment", "review"})
        app.validate_envelope(result, "ok")

    def test_deterministic_and_no_input_mutation(self):
        source = fixture()
        original = copy.deepcopy(source)
        self.assertEqual(app.run_pipeline(source), app.run_pipeline(source))
        self.assertEqual(source, original)

    def test_synonym_retrieval_and_explanation(self):
        result = app.semantic_stage(fixture())["results"]["search"]
        self.assertEqual(result["method"], "concept_tfidf")
        self.assertEqual(result["hits"][0]["product_id"], "fan-1")
        self.assertIn("quiet", result["hits"][0]["matched_concepts"])
        self.assertIn("wireless", result["hits"][0]["matched_concepts"])
        self.assertIsNone(result["hits"][0]["embedding_score"])
        self.assertEqual(result["index_stats"]["product_count"], 3)

    def test_top_k_propagates_to_sentiment_and_review(self):
        source = fixture()
        source["data"]["options"]["top_k"] = 1
        result = app.run_pipeline(source)["results"]
        self.assertEqual([h["product_id"] for h in result["search"]["hits"]], ["fan-1"])
        self.assertEqual({s["feedback_id"] for s in result["sentiment"]["items"]},
                         {"fb-1", "fb-2"})
        self.assertEqual([r["requirement_id"] for r in result["review"]["checks"]],
                         ["req-1"])
        self.assertEqual(result["review"]["unreviewed_requirement_ids"],
                         ["req-2", "req-3"])

    def test_no_match_does_not_review_unrelated_products(self):
        source = fixture()
        source["data"]["query"] = "astronomical telescope"
        source["data"]["options"]["min_relevance"] = 0
        result = app.run_pipeline(source)["results"]
        self.assertEqual(result["search"]["hits"], [])
        self.assertEqual(result["sentiment"]["items"], [])
        self.assertEqual(result["review"]["checks"], [])
        self.assertEqual(result["review"]["gaps"], [])
        self.assertEqual(len(result["review"]["unreviewed_requirement_ids"]), 3)

    def test_empty_collections_are_valid(self):
        source = fixture()
        for key in ("products", "feedback", "requirements", "documents"):
            source["data"][key] = []
        result = app.run_pipeline(source)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["results"]["search"]["index_stats"]["product_count"], 0)

    def test_relevance_threshold(self):
        source = fixture()
        source["data"]["options"]["min_relevance"] = 1
        self.assertEqual(app.run_pipeline(source)["results"]["search"]["hits"], [])

    def test_ties_break_by_identifier_not_input_order(self):
        source = fixture()
        source["data"]["products"] = [
            {"id": "z", "name": "Fan", "description": "quiet fan"},
            {"id": "a", "name": "Fan", "description": "quiet fan"},
        ]
        for key in ("feedback", "requirements", "documents"):
            source["data"][key] = []
        source["data"]["query"] = "fan"
        first = app.run_pipeline(source)["results"]["search"]["hits"]
        source["data"]["products"].reverse()
        second = app.run_pipeline(source)["results"]["search"]["hits"]
        self.assertEqual(first, second)
        self.assertEqual([h["product_id"] for h in first], ["a", "z"])

    def test_local_embedding_callable_and_fusion(self):
        calls = []

        def local_embedder(texts):
            calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

        source = fixture()
        source["data"]["query"] = "unlistedconcept"
        source["data"]["options"]["top_k"] = 3
        result = app.run_pipeline(source, local_embedder)["results"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)
        self.assertEqual(calls[0][0], "unlistedconcept")
        self.assertEqual(result["search"]["method"], "hybrid_injected")
        self.assertEqual(len(result["search"]["hits"]), 3)
        self.assertTrue(all(hit["score"] == 0.35 for hit in result["search"]["hits"]))
        self.assertEqual(len(result["sentiment"]["items"]), 4)
        self.assertEqual(len(result["review"]["checks"]), 3)

    def test_negative_cosine_is_clipped(self):
        source = fixture()
        source["data"]["query"] = "unlistedconcept"
        result = app.run_pipeline(source, lambda texts: [[1, 0]] +
                                  [[-1, 0] for _ in texts[1:]])
        self.assertEqual(result["results"]["search"]["hits"], [])

    def test_embedding_contract_rejects_bad_vectors(self):
        bad_results = [
            None, [], [[1, 0]] * 3, [[1, 0], [1], [1, 0], [1, 0]],
            [[0, 0]] * 4, [[True, 0]] * 4, [[float("nan"), 1]] * 4,
            [[float("inf"), 1]] * 4, [["1", 0]] * 4,
            [[1e101, 0]] * 4, [[]] * 4,
        ]
        for bad in bad_results:
            with self.subTest(bad=repr(bad)), self.assertRaises(app.ValidationError):
                app.run_pipeline(fixture(), lambda texts, value=bad: value)

    def test_embedder_failure_is_wrapped(self):
        def broken(texts):
            raise RuntimeError("synthetic failure")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(fixture(), broken)
        with self.assertRaisesRegex(app.ValidationError, "must be callable"):
            app.run_pipeline(fixture(), 123)

    def test_large_and_tiny_vectors_are_stable(self):
        for magnitude in (1e99, 1e-300):
            with self.subTest(magnitude=magnitude):
                result = app.run_pipeline(
                    fixture(), lambda texts: [[magnitude, magnitude] for _ in texts])
                self.assertEqual(result["results"]["search"]["hits"][0]["embedding_score"],
                                 1.0)

    def test_sentiment_transparent_positive_negative_neutral(self):
        for value, expected in (("great", "positive"), ("broken", "negative"),
                                ("fan arrived Tuesday", "neutral"),
                                ("great awful", "neutral")):
            with self.subTest(value=value):
                score, label, cues = app.sentiment_score(value)
                self.assertEqual(label, expected)
                denominator = max(1, sum(abs(c["contribution"]) for c in cues))
                self.assertEqual(score, round(sum(c["contribution"] for c in cues)
                                              / denominator, 6))

    def test_negation_scope_and_word_boundaries(self):
        for value, expected in (("not good", "negative"), ("not broken", "positive"),
                                ("not not good", "positive"),
                                ("not. good", "positive"), ("not but good", "positive"),
                                ("not one two three good", "positive"),
                                ("goodness", "neutral")):
            with self.subTest(value=value):
                self.assertEqual(app.sentiment_score(value)[1], expected)
        self.assertEqual(app.sentiment_score("not good")[2][0]["multiplier"], -1)

    def test_severity_dominates_even_positive_feedback(self):
        source = fixture()
        source["data"]["feedback"][0]["text"] = "excellent"
        source["data"]["feedback"][1]["text"] = "awful"
        result = app.run_pipeline(source)["results"]["sentiment"]["items"]
        self.assertEqual(result[0]["feedback_id"], "fb-1")
        self.assertEqual(result[0]["sentiment_label"], "positive")
        self.assertEqual(result[0]["priority_score"], 100)
        self.assertEqual(result[0]["priority"], "critical")
        self.assertEqual(result[-1]["priority_score"], 30)

    def test_search_metadata_propagation(self):
        result = app.run_pipeline(fixture())["results"]
        hits = {h["product_id"]: h for h in result["search"]["hits"]}
        for item in result["sentiment"]["items"]:
            hit = hits[item["product_id"]]
            self.assertEqual(item["search_rank"], hit["rank"])
            self.assertEqual(item["relevance"], hit["score"])
            self.assertIn(item["feedback_id"], hit["feedback_ids"])

    def test_review_gap_evidence_and_priority_propagation(self):
        source = fixture()
        result = app.run_pipeline(source)["results"]
        review = result["review"]
        self.assertEqual(len(review["gaps"]), 1)
        gap = review["gaps"][0]
        self.assertEqual(gap["requirement_id"], "req-1")
        self.assertEqual(gap["missing_terms"], ["recall procedure"])
        self.assertEqual(gap["feedback_ids"], ["fb-1", "fb-2"])
        self.assertEqual(gap["priority_score"],
                         result["sentiment"]["items"][0]["priority_score"])
        self.assertEqual(gap["priority"], "critical")
        self.assertEqual(review["checks"][1]["status"], "evidence_found")
        self.assertIn("not certification", review["disclaimer"])
        documents = {d["id"]: d for d in source["data"]["documents"]}
        for check in review["checks"]:
            for evidence in check["evidence"]:
                document = documents[evidence["document_id"]]
                self.assertEqual(document["product_id"], check["product_id"])
                self.assertEqual(document["text"][evidence["start"]:evidence["end"]],
                                 evidence["quote"])

    def test_feedback_severity_change_changes_review_priority(self):
        source = fixture()
        first = app.run_pipeline(source)["results"]["review"]["gaps"][0]
        source["data"]["feedback"][0]["severity"] = "low"
        second = app.run_pipeline(source)["results"]["review"]["gaps"][0]
        self.assertGreater(first["priority_score"], second["priority_score"])
        self.assertEqual(second["priority"], "low")

    def test_no_feedback_still_reviews_selected_requirements(self):
        source = fixture()
        source["data"]["feedback"] = []
        result = app.run_pipeline(source)["results"]
        self.assertEqual(result["sentiment"]["items"], [])
        self.assertEqual(len(result["review"]["checks"]), 2)
        for check in result["review"]["checks"]:
            self.assertEqual(check["priority_score"], 0)
            self.assertEqual(check["feedback_ids"], [])

    def test_missing_documents_and_no_requirements(self):
        source = fixture()
        source["data"]["documents"] = []
        review = app.run_pipeline(source)["results"]["review"]
        self.assertEqual(len(review["gaps"]), 2)
        self.assertTrue(all(check["evidence"] == [] for check in review["checks"]))
        source["data"]["requirements"] = []
        self.assertEqual(app.run_pipeline(source)["results"]["review"]["checks"], [])

    def test_phrase_boundaries_case_unicode_and_multiple_documents(self):
        source = fixture()
        source["data"]["requirements"][0]["required_terms"] = ["safe", "Café guide"]
        source["data"]["documents"][0]["text"] = "Unsafe device. CAFÉ guide."
        review = app.run_pipeline(source)["results"]["review"]
        self.assertEqual(review["checks"][0]["missing_terms"], ["safe"])
        self.assertEqual(review["checks"][0]["evidence"][0]["quote"], "CAFÉ guide")
        source["data"]["documents"].append({
            "id": "doc-extra", "product_id": "fan-1",
            "title": "Synthetic supplement", "text": "Safe use."})
        review = app.run_pipeline(source)["results"]["review"]
        self.assertEqual(review["checks"][0]["status"], "evidence_found")
        self.assertEqual(len(review["checks"][0]["evidence"]), 2)

    def test_textual_mentions_never_claim_certification(self):
        source = fixture()
        source["data"]["documents"][0]["text"] = (
            "There is no battery safety section and no recall procedure.")
        check = app.run_pipeline(source)["results"]["review"]["checks"][0]
        self.assertEqual(check["status"], "evidence_found")
        self.assertNotIn("certified", check)
        self.assertIn("not that claims are true", app.DISCLAIMER)

    def test_each_stage_rejects_wrong_handoff(self):
        source = fixture()
        for function in (app.sentiment_stage, app.review_stage):
            with self.subTest(stage=function.__name__), self.assertRaises(app.ValidationError):
                function(source)
        with self.assertRaises(app.ValidationError):
            app.semantic_stage(app.semantic_stage(source))

    def test_tampered_search_and_sentiment_are_rejected(self):
        source = app.semantic_stage(fixture())
        source["results"]["search"]["hits"][0]["feedback_ids"] = ["fb-4"]
        with self.assertRaises(app.ValidationError):
            app.sentiment_stage(source)
        source = app.sentiment_stage(app.semantic_stage(fixture()))
        source["results"]["sentiment"]["items"][0]["priority_score"] = 0
        with self.assertRaises(app.ValidationError):
            app.review_stage(source)

    def test_tampered_review_evidence_is_rejected(self):
        source = app.run_pipeline(fixture())
        source["results"]["review"]["checks"][0]["evidence"][0]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.validate_envelope(source)

    def test_hybrid_handoff_validation(self):
        source = app.semantic_stage(fixture(), lambda texts: [[1, 0] for _ in texts])
        source["results"]["search"]["hits"][0]["embedding_score"] = 0
        with self.assertRaises(app.ValidationError):
            app.sentiment_stage(source)

    def test_invalid_schema_and_types(self):
        mutations = [
            lambda x: x.update(schema_version=True),
            lambda x: x.update(schema_version=2),
            lambda x: x.update(fixture_label="real data"),
            lambda x: x.update(status=[]),
            lambda x: x.update(extra="not allowed"),
            lambda x: x["data"].update(query="   "),
            lambda x: x["data"].update(query="?!"),
            lambda x: x["data"].update(products={}),
            lambda x: x["data"]["options"].update(top_k=True),
            lambda x: x["data"]["options"].update(top_k=0),
            lambda x: x["data"]["options"].update(top_k=101),
            lambda x: x["data"]["options"].update(min_relevance=float("nan")),
            lambda x: x["data"]["options"].update(min_relevance=1.1),
            lambda x: x["data"]["options"].update(min_relevance=True),
            lambda x: x["data"]["feedback"][0].update(severity="urgent"),
            lambda x: x["data"]["feedback"][0].update(severity=[]),
            lambda x: x["data"]["documents"][0].update(product_id="missing"),
            lambda x: x["data"]["requirements"][0].update(required_terms=[]),
            lambda x: x["data"]["requirements"][0].update(required_terms=["Safe", "safe"]),
            lambda x: x["data"]["requirements"][0].update(required_terms=["!!!"]),
            lambda x: x["data"]["products"][0].update(id="bad id"),
            lambda x: x["data"]["products"].append(copy.deepcopy(x["data"]["products"][0])),
        ]
        for i, mutate in enumerate(mutations):
            source = fixture()
            mutate(source)
            with self.subTest(case=i), self.assertRaises(app.ValidationError):
                app.run_pipeline(source)
        for source in (None, [], True, 17, "input"):
            with self.subTest(root=source), self.assertRaises(app.ValidationError):
                app.run_pipeline(source)

    def test_resource_bounds(self):
        source = fixture()
        source["data"]["feedback"][0]["text"] = "x" * 20001
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(source)
        source = fixture()
        source["data"]["products"] *= 334
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(source)


class CLITests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"), *args],
            cwd=HERE, capture_output=True, text=True, encoding="utf-8", timeout=20)

    def assert_cli_error(self, process):
        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_example(self):
        process = self.command("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        result = json.loads(process.stdout)
        self.assertEqual(result["status"], "ok")
        app.validate_envelope(result, "ok")

    def test_cli_missing_file(self):
        self.assert_cli_error(self.command("synthetic_nonexistent_input.json"))

    def test_cli_invalid_file_contents(self):
        self.assert_cli_error(self.command("implementation.py"))

    def test_cli_invalid_arguments(self):
        self.assert_cli_error(self.command())
        self.assert_cli_error(self.command("example_input.json", "extra"))

    def test_cli_directory_is_file_error(self):
        self.assert_cli_error(self.command("."))

    def test_json_parse_and_schema_errors_are_json(self):
        payloads = [
            b'{"status": "input", "status": "ok"}', b'{"value":NaN}',
            b'{"value":Infinity}', b"\xff", b"null", b"[]", b"{}",
            b'{"data":', b"x" * 2_000_001,
        ]
        for payload in payloads:
            with self.subTest(payload=payload[:80]):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(payload)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_utf8_bom_is_accepted(self):
        payload = b"\xef\xbb\xbf" + json.dumps(fixture()).encode("utf-8")
        output = io.StringIO()
        with patch.object(Path, "open", return_value=io.BytesIO(payload)):
            with contextlib.redirect_stdout(output):
                code = app.main(["synthetic.json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "ok")


if __name__ == "__main__":
    unittest.main()
