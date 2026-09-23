"""Tests use only explicitly synthetic fixture data and standard-library mocks."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def stages(self):
        return app.run_pipeline(self.data)["stages"]

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def finding(self, deep, theme, product="p-trail"):
        return next(f for f in deep["findings"]
                    if f["theme"] == theme and f["product_id"] == product)

    def test_integrated_order_and_handoff_ids(self):
        stages = self.stages()
        self.assertEqual(list(stages), ["interests", "faq", "feedback", "deep"])
        self.assertEqual(stages["faq"]["input_ids"],
                         [r["product_id"] for r in stages["interests"]["recommendations"]])
        self.assertEqual(stages["feedback"]["input_ids"],
                         [a["question_id"] for a in stages["faq"]["answers"]])
        self.assertEqual(stages["deep"]["input_ids"],
                         [i["feedback_id"] for i in stages["feedback"]["items"]])

    def test_preference_ranking_and_grounded_explanation(self):
        result = self.stages()["interests"]
        self.assertEqual([(r["product_id"], r["score"]) for r in result["recommendations"]],
                         [("p-trail", 6), ("p-desk", 4)])
        top = result["recommendations"][0]
        self.assertIn("battery (3)", top["explanation"])
        self.assertEqual(top["evidence"][0]["excerpt"], self.data["catalog"][0]["description"])

    def test_exclusions_are_hard_constraints(self):
        result = self.stages()["interests"]
        self.assertEqual(result["excluded_product_ids"], ["p-blocked", "p-glass"])
        self.assertNotIn("p-book", [r["product_id"] for r in result["recommendations"]])

    def test_normalized_topics_match(self):
        self.data["interests"]["preferences"][0]["topic"] = "  BATTERY  "
        self.assertEqual(self.stages()["interests"]["recommendations"][0]["score"], 6)

    def test_ties_are_broken_by_product_id(self):
        self.data["catalog"][1]["topics"].append("outdoors")
        self.assertEqual(self.stages()["faq"]["input_ids"], ["p-desk", "p-trail"])

    def test_limit_propagates_through_every_stage(self):
        self.data["interests"]["limit"] = 1
        stages = self.stages()
        self.assertEqual(stages["faq"]["input_ids"], ["p-trail"])
        self.assertIn("q-desk-return", stages["faq"]["skipped_question_ids"])
        self.assertIn("f4", stages["feedback"]["skipped_feedback_ids"])
        self.assertNotIn("research-desk", stages["deep"]["documents_used"])

    def test_all_excluded_propagates_empty_pipeline(self):
        self.data["interests"]["excluded_product_ids"] = [p["id"] for p in self.data["catalog"]]
        stages = self.stages()
        self.assertEqual(stages["interests"]["recommendations"], [])
        self.assertEqual(stages["faq"]["answers"], [])
        self.assertEqual(stages["feedback"]["items"], [])
        self.assertEqual(stages["deep"]["findings"], [])

    def test_empty_preferences_and_empty_collection_inputs(self):
        self.data["interests"]["preferences"] = []
        self.assertEqual(self.stages()["deep"]["findings"], [])
        self.data["catalog"] = []
        self.data["interests"]["excluded_product_ids"] = []
        self.data["faq"] = {"questions": [], "documents": []}
        self.data["feedback"] = []
        self.data["research"]["documents"] = []
        self.assertEqual(self.stages()["interests"]["input_ids"], [])

    def test_faq_answers_are_verbatim_citations(self):
        answers = self.stages()["faq"]["answers"]
        answer = next(a for a in answers if a["question_id"] == "q-trail-runtime")
        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["answer"], self.data["faq"]["documents"][0]["text"])
        self.assertEqual(answer["citations"][0]["source_id"], "kb-trail")

    def test_faq_explicit_abstention(self):
        answer = next(a for a in self.stages()["faq"]["answers"]
                      if a["question_id"] == "q-trail-water")
        self.assertEqual(answer["status"], "abstained")
        self.assertIsNone(answer["answer"])
        self.assertEqual(answer["citations"], [])
        self.assertIn("No product-scoped source", answer["reason"])

    def test_faq_rejects_partial_or_product_mismatched_retrieval(self):
        self.data["faq"]["documents"] = [self.data["faq"]["documents"][2]]
        self.assertTrue(all(a["status"] == "abstained" for a in self.stages()["faq"]["answers"]))
        self.data["faq"]["documents"] = [
            {"id": "partial", "product_ids": ["p-trail"], "text": "Synthetic battery information."}
        ]
        self.assertEqual(self.stages()["faq"]["answers"][0]["status"], "abstained")

    def test_faq_stopword_only_question_abstains(self):
        self.data["faq"]["questions"][0]["text"] = "What is it?"
        self.assertEqual(self.stages()["faq"]["answers"][0]["status"], "abstained")

    def test_faq_retrieval_tie_and_two_document_bound(self):
        original = self.data["faq"]["documents"][0]
        self.data["faq"]["documents"] += [
            dict(original, id="kb-a"), dict(original, id="kb-z"),
        ]
        answer = self.stages()["faq"]["answers"][0]
        self.assertEqual([c["source_id"] for c in answer["citations"]], ["kb-a", "kb-trail"])
        self.assertEqual(answer["answer"], "\n".join(c["excerpt"] for c in answer["citations"]))

    def test_feedback_normalized_dedup_and_lineage(self):
        feedback = self.stages()["feedback"]
        self.assertEqual([i["feedback_id"] for i in feedback["items"]], ["f1", "f3", "f4"])
        self.assertEqual(feedback["items"][0]["duplicate_ids"], ["f2"])
        battery = next(t for t in feedback["themes"] if t["theme"] == "battery")
        self.assertEqual(battery["count"], 1)
        self.assertEqual(battery["excerpts"][0]["duplicate_ids"], ["f2"])
        self.assertEqual(battery["excerpts"][0]["excerpt"], self.data["feedback"][0]["text"])

    def test_identical_feedback_in_different_question_context_is_distinct(self):
        self.data["feedback"].append({
            "id": "f6", "question_id": "q-trail-water", "text": self.data["feedback"][0]["text"],
        })
        result = self.stages()["feedback"]
        self.assertEqual(len(result["items"]), 4)
        self.assertEqual(next(t for t in result["themes"] if t["theme"] == "battery")["count"], 2)

    def test_feedback_multi_theme_and_other(self):
        feedback = self.stages()["feedback"]
        self.assertEqual(feedback["items"][0]["themes"], ["battery", "usability"])
        self.data["feedback"][0]["text"] = "Purple shade looks lovely."
        self.assertEqual(self.stages()["feedback"]["items"][0]["themes"], ["other"])

    def test_feedback_from_excluded_product_never_reaches_research(self):
        stages = self.stages()
        self.assertEqual(stages["feedback"]["skipped_feedback_ids"], ["f5"])
        self.assertNotIn("research-blocked", stages["deep"]["documents_used"])
        self.assertFalse(any(f["product_id"] == "p-blocked" for f in stages["deep"]["findings"]))

    def test_multidocument_disagreement_has_both_verbatim_sources(self):
        deep = self.stages()["deep"]
        battery = self.finding(deep, "battery")
        claim = battery["claims"][0]
        self.assertEqual(claim["assessment"], "mixed")
        self.assertEqual({e["source_id"] for e in claim["evidence"]},
                         {"research-bench", "research-field"})
        for evidence in claim["evidence"]:
            doc = next(d for d in self.data["research"]["documents"]
                       if d["id"] == evidence["source_id"])
            self.assertIn(evidence["excerpt"], doc["text"])
        self.assertEqual(len(deep["disagreements"]), 1)
        self.assertIn("no consensus", battery["unresolved"][0])

    def test_deep_dedup_provenance_is_traceable(self):
        battery = self.finding(self.stages()["deep"], "battery")
        self.assertEqual(battery["feedback_ids"], ["f1"])
        self.assertEqual(battery["feedback_excerpts"][0]["duplicate_ids"], ["f2"])

    def test_deep_unsupported_and_uncertain_themes(self):
        deep = self.stages()["deep"]
        support = self.finding(deep, "support")
        self.assertEqual(support["claims"], [])
        self.assertIn("No research evidence", support["unresolved"][0])
        durability = self.finding(deep, "durability")
        self.assertEqual(durability["claims"][0]["assessment"], "uncertain")
        self.assertTrue(any("Only uncertain" in u for u in durability["unresolved"]))
        self.assertTrue(any("Only one document" in u for u in durability["unresolved"]))

    def test_deep_support_and_opposition_do_not_imply_certainty(self):
        deep = self.stages()["deep"]
        self.assertEqual(self.finding(deep, "usability")["claims"][0]["assessment"], "opposed")
        support = self.finding(deep, "support", "p-desk")
        self.assertEqual(support["claims"][0]["assessment"], "supported")
        self.assertIn("not verified facts", support["summary"])
        self.assertTrue(any("corroboration missing" in u for u in support["unresolved"]))

    def test_deep_does_not_conflate_products(self):
        self.data["research"]["documents"][1]["product_ids"] = ["p-desk"]
        deep = self.stages()["deep"]
        self.assertEqual(self.finding(deep, "battery")["claims"][0]["assessment"], "supported")
        self.assertEqual(deep["disagreements"], [])

    def test_no_feedback_still_preserves_faq_abstentions(self):
        self.data["feedback"] = []
        deep = self.stages()["deep"]
        self.assertEqual(deep["findings"], [])
        self.assertEqual(deep["unresolved_questions"][0]["question_id"], "q-trail-water")

    def test_empty_knowledge_bases_abstain_and_record_gaps(self):
        self.data["faq"]["documents"] = []
        self.data["research"]["documents"] = []
        stages = self.stages()
        self.assertEqual(len(stages["deep"]["unresolved_questions"]), 3)
        self.assertTrue(all(not f["claims"] for f in stages["deep"]["findings"]))
        self.assertTrue(all(f["unresolved"] for f in stages["deep"]["findings"]))

    def test_determinism_and_input_not_mutated(self):
        before = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        second = app.run_pipeline(self.data)
        self.assertEqual(first, second)
        self.assertEqual(self.data, before)

    def test_shared_schema_rejects_unknown_missing_and_wrong_type_fields(self):
        original = copy.deepcopy(self.data)
        for mutate in (
            lambda d: d.update(extra=True),
            lambda d: d.pop("feedback"),
            lambda d: d.update(catalog={}),
            lambda d: d.update(schema_version="2.0"),
            lambda d: d.update(synthetic=1),
        ):
            with self.subTest(mutate=mutate):
                self.data = copy.deepcopy(original)
                mutate(self.data)
                self.invalid()

    def test_invalid_weights_and_limits(self):
        for weight in (True, 0, -1, 101, 10 ** 400, float("nan"), float("inf"), "3"):
            with self.subTest(weight=weight):
                self.data["interests"]["preferences"][0]["weight"] = weight
                self.invalid()
        self.data["interests"]["preferences"][0]["weight"] = 3
        for limit in (False, 0, -1, 21, 1.5):
            with self.subTest(limit=limit):
                self.data["interests"]["limit"] = limit
                self.invalid()

    def test_duplicate_ids_and_normalized_preferences(self):
        self.data["feedback"].append(copy.deepcopy(self.data["feedback"][0]))
        self.invalid()
        self.data["feedback"].pop()
        self.data["interests"]["preferences"].append({"topic": " BATTERY ", "weight": 2})
        self.invalid()

    def test_unknown_references_are_rejected(self):
        original = copy.deepcopy(self.data)
        for location, key in (
            (("faq", "questions", 0), "product_id"),
            (("feedback", 0), "question_id"),
        ):
            with self.subTest(location=location):
                self.data = copy.deepcopy(original)
                obj = self.data
                for part in location:
                    obj = obj[part]
                obj[key] = "missing"
                self.invalid()
        self.data = original
        self.data["research"]["documents"][0]["product_ids"] = ["missing"]
        self.invalid()

    def test_research_excerpts_and_claim_identity_are_validated(self):
        claim = self.data["research"]["documents"][0]["claims"][0]
        claim["excerpt"] = "Fabricated source excerpt."
        self.invalid()
        claim["excerpt"] = self.data["research"]["documents"][0]["text"].split(" Synthetic controls")[0]
        self.data["research"]["documents"][1]["claims"][0]["statement"] = "A different proposition."
        self.invalid()

    def test_blank_punctuation_and_non_synthetic_data_rejected(self):
        self.data["fixture_label"] = "real customer data"
        self.invalid()
        self.data["fixture_label"] = "Synthetic data"
        self.data["feedback"][0]["text"] = "!!!"
        self.invalid()
        self.data["feedback"][0]["text"] = " "
        self.invalid()

    def test_invalid_stage_is_rejected_before_next_stage_runs(self):
        bad = app.recommend(self.data)
        bad["recommendations"][0]["score"] = 99
        with patch.object(app, "recommend", return_value=bad), \
                patch.object(app, "answer_faq") as downstream:
            self.invalid()
            downstream.assert_not_called()

    def test_shared_validator_rejects_fabricated_answers_and_evidence(self):
        stages = self.stages()
        answer = stages["faq"]["answers"][0]
        answer["answer"] = "Unsupported generated claim."
        with self.assertRaises(app.ValidationError):
            app.validate_stage("faq", stages["faq"], self.data, stages["interests"])
        stages = self.stages()
        stages["deep"]["findings"][0]["claims"][0]["evidence"][0]["excerpt"] = "Invented quote."
        with self.assertRaises(app.ValidationError):
            app.validate_stage("deep", stages["deep"], self.data, stages["feedback"])

    def test_shared_validator_rejects_lost_handoff_ids(self):
        stages = self.stages()
        stages["feedback"]["input_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.validate_stage("feedback", stages["feedback"], self.data, stages["faq"])


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )

    def assert_json_error(self, result):
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_success_is_exactly_one_json_object(self):
        result = self.invoke("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        self.assert_json_error(self.invoke("nonexistent-synthetic-input.json"))

    def test_cli_directory_path_is_file_error(self):
        self.assert_json_error(self.invoke(str(ROOT)))

    def test_cli_missing_and_extra_arguments(self):
        self.assert_json_error(self.invoke())
        self.assert_json_error(self.invoke("example_input.json", "extra"))

    def test_main_malformed_duplicate_nonfinite_and_wrong_schema_json(self):
        cases = [
            b"{",
            b'{"synthetic": true, "synthetic": false}',
            b'{"weight": NaN}',
            b'{"weight": Infinity}',
            b'{"weight": ' + b"9" * 5000 + b"}",
            b"null",
            b"[]",
            b"\xff",
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                stdout = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(payload)), redirect_stdout(stdout):
                    code = app.main(["synthetic-mocked-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_main_oversized_input_is_json_error(self):
        stdout = io.StringIO()
        with patch.object(Path, "open", return_value=io.BytesIO(b" " * 1_000_001)), \
                redirect_stdout(stdout):
            code = app.main(["synthetic-mocked-input.json"])
        self.assertEqual(code, 2)
        self.assertIn("exceeds", json.loads(stdout.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
