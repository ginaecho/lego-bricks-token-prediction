"""Synthetic fixtures; no network, external dependency, or temporary directory."""

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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)["stages"]

    def empty_state(self):
        return {"schema_version": "1.0", "status": "ok",
                "input": copy.deepcopy(self.data), "stages": {}}

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                              cwd=HERE, text=True, capture_output=True, check=False)

    def invoke_text(self, content):
        output = io.StringIO()
        with patch("builtins.open", return_value=io.StringIO(content)):
            with contextlib.redirect_stdout(output):
                code = app.main(["synthetic-input.json"])
        return code, json.loads(output.getvalue())

    def test_complete_pipeline_and_shared_validation(self):
        result = app.run_pipeline(self.data)
        app.validate(result, "state")
        self.assertEqual(list(result["stages"]), list(app.STAGES))
        self.assertEqual(result["stages"]["faq"]["status"], "answered")

    def test_transparent_sentiment_evidence(self):
        issue = self.run_data()["sentiment"]["issues"][0]
        self.assertEqual(issue["score"], -5)
        self.assertEqual(issue["label"], "negative")
        self.assertEqual(sum(e["contribution"] for e in issue["evidence"]), -5)
        for evidence in issue["evidence"]:
            self.assertEqual(issue["text"][evidence["start"]:evidence["end"]].lower(),
                             evidence["term"])

    def test_negation_and_punctuation_scope(self):
        self.data["feedback"][0]["text"] = "not very good. great; not bad"
        issue = self.run_data()["sentiment"]["issues"][0]
        self.assertEqual(issue["score"], 2)
        self.assertEqual([e["negated"] for e in issue["evidence"]], [True, False, True])

    def test_double_negation(self):
        self.data["feedback"][0]["text"] = "not never good"
        self.assertEqual(self.run_data()["sentiment"]["issues"][0]["score"], 1)

    def test_severity_dominates_sentiment(self):
        self.data["feedback"][0]["text"] = "great"
        self.data["feedback"][1]["text"] = "terrible " * 100
        self.assertEqual(self.run_data()["sentiment"]["issues"][0]["feedback_id"],
                         "checkout-issue")

    def test_priority_tie_break_is_identifier(self):
        self.data["feedback"] = [
            {"id": name, "text": "bad", "severity": "normal"} for name in ("z", "a")]
        self.assertEqual([i["feedback_id"] for i in self.run_data()["sentiment"]["issues"]],
                         ["a", "z"])

    def test_research_extracts_exact_source_offsets(self):
        finding = self.run_data()["normal"]["findings"][0]
        cite = finding["citation"]
        original = self.data["research_passages"][0]
        self.assertEqual(cite["source"], original["source"])
        self.assertEqual(original["text"][cite["start"]:cite["end"]], finding["text"])
        self.assertEqual(finding["text"], cite["quote"])

    def test_unicode_offsets_and_second_sentence_selection(self):
        self.data["research_passages"][0]["text"] = "  Café note.  Checkout payment is blocked!"
        finding = self.run_data()["normal"]["findings"][0]
        self.assertEqual(finding["text"], "Checkout payment is blocked!")
        self.assertEqual(finding["citation"]["start"], 14)

    def test_retrieval_limit_and_deterministic_ties(self):
        self.data["retrieval_limit"] = 1
        self.data["research_passages"].append(
            {"id": "aaa", "source": "synthetic:tie",
             "text": self.data["research_passages"][0]["text"]})
        findings = self.run_data()["normal"]["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["citation"]["document_id"], "aaa")

    def test_newline_delimited_evidence_without_punctuation(self):
        self.data["research_passages"][0]["text"] = "Checkout payment blocked\nUnrelated note"
        self.data["knowledge_base"][0]["text"] = "Check payment details\nUnrelated note"
        stages = self.run_data()
        self.assertEqual(stages["normal"]["findings"][0]["text"], "Checkout payment blocked")
        self.assertEqual(stages["faq"]["answers"][0]["answer"], "Check payment details")

    def test_two_step_prerequisites_and_grounding(self):
        stages = self.run_data()
        journey = stages["journey"]
        self.assertTrue(journey["validated_two_step"])
        self.assertEqual([s["action_id"] for s in journey["steps"]],
                         ["check-payment", "retry-checkout"])
        self.assertEqual(journey["steps"][1]["satisfied_prerequisites"], ["check-payment"])
        self.assertEqual(journey["steps"][0]["citations"][0],
                         stages["normal"]["findings"][0]["citation"])
        self.assertNotIn("filter-catalog", journey["next_action_ids"])

    def test_completed_action_is_not_recommended(self):
        self.data["completed_action_ids"] = ["check-payment"]
        journey = self.run_data()["journey"]
        self.assertEqual(journey["next_action_ids"], ["retry-checkout"])
        self.assertFalse(journey["validated_two_step"])
        self.assertEqual(journey["steps"], [])

    def test_completed_prerequisite_unlocks_two_remaining_steps(self):
        self.data["actions"].append({
            "id": "verify-order", "title": "Verify order", "prerequisites": ["retry-checkout"],
            "passage_ids": ["payment-guide"], "faq_question": "Verify order?",
        })
        self.data["completed_action_ids"] = ["check-payment"]
        journey = self.run_data()["journey"]
        self.assertEqual([s["action_id"] for s in journey["steps"]],
                         ["retry-checkout", "verify-order"])

    def test_ungrounded_prerequisite_blocks_journey(self):
        self.data["actions"][0]["passage_ids"] = ["catalog-guide"]
        journey = self.run_data()["journey"]
        self.assertFalse(journey["validated_two_step"])
        self.assertEqual(journey["next_action_ids"], [])

    def test_faq_is_extractive_and_action_scoped(self):
        stages = self.run_data()
        faq = stages["faq"]
        self.assertEqual(faq["journey_action_ids"], ["check-payment", "retry-checkout"])
        for answer in faq["answers"]:
            cite = answer["citations"][0]
            doc = next(d for d in self.data["knowledge_base"] if d["id"] == cite["document_id"])
            self.assertIn(answer["action_id"], doc["action_ids"])
            self.assertEqual(answer["answer"], doc["text"][cite["start"]:cite["end"]])

    def test_partial_faq_abstention(self):
        self.data["knowledge_base"] = self.data["knowledge_base"][:1]
        faq = self.run_data()["faq"]
        self.assertEqual(faq["status"], "partial")
        self.assertEqual(faq["answers"][1]["status"], "abstained")
        self.assertIsNone(faq["answers"][1]["answer"])
        self.assertEqual(faq["answers"][1]["citations"], [])

    def test_faq_abstains_with_irrelevant_action_scoped_text(self):
        for document in self.data["knowledge_base"]:
            document["text"] = "Zebras wander."
        faq = self.run_data()["faq"]
        self.assertEqual(faq["status"], "abstained")
        self.assertTrue(all(a["answer"] is None for a in faq["answers"]))

    def test_faq_cannot_borrow_other_actions_evidence(self):
        self.data["knowledge_base"][0]["action_ids"] = ["filter-catalog"]
        faq = self.run_data()["faq"]
        self.assertEqual(faq["answers"][0]["status"], "abstained")

    def test_cross_stage_focus_propagation(self):
        stages = self.run_data()
        for name in ("normal", "journey", "faq"):
            self.assertEqual(stages[name]["focus"]["feedback_id"], "checkout-issue")
            self.assertEqual(stages[name]["focus"]["severity"], "critical")
        self.data["feedback"][0]["severity"] = "low"
        changed = self.run_data()
        self.assertEqual(changed["normal"]["focus"]["feedback_id"], "catalog-praise")
        self.assertEqual(changed["normal"]["findings"][0]["citation"]["document_id"],
                         "catalog-guide")
        self.assertEqual(changed["journey"]["next_action_ids"], ["filter-catalog"])
        self.assertEqual(changed["faq"]["status"], "abstained")

    def test_empty_feedback_and_empty_collections(self):
        for key in ("feedback", "actions", "research_passages", "knowledge_base"):
            self.data[key] = []
        stages = self.run_data()
        self.assertEqual(stages["sentiment"]["issues"], [])
        self.assertEqual(stages["normal"]["status"], "no_findings")
        self.assertIsNone(stages["faq"]["focus"])
        self.assertEqual(stages["faq"]["reason"], "no_valid_journey")

    def test_no_research_match_propagates_abstention(self):
        self.data["feedback"][0]["text"] = "xyzzy"
        stages = self.run_data()
        self.assertEqual(stages["normal"]["findings"], [])
        self.assertEqual(stages["journey"]["reason"], "no_research_evidence")
        self.assertEqual(stages["faq"]["answers"], [])

    def test_unknown_and_missing_fields_rejected(self):
        for change in ("unknown", "missing"):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                if change == "unknown":
                    data["extra"] = 1
                else:
                    del data["feedback"]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_types_and_values(self):
        cases = [("retrieval_limit", True), ("retrieval_limit", 0),
                 ("retrieval_limit", 6), ("synthetic", False), ("schema_version", "2"),
                 ("feedback", {}), ("fixture_label", " ")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_record_fields(self):
        for key, value in (("id", "bad id"), ("text", ""), ("severity", []),
                           ("severity", "urgent")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["feedback"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_ids_and_references(self):
        self.data["feedback"].append(copy.deepcopy(self.data["feedback"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data["feedback"].pop()
        self.data["actions"][1]["prerequisites"] *= 2
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_unknown_references_rejected(self):
        for collection, key in (("actions", "prerequisites"), ("actions", "passage_ids"),
                                ("knowledge_base", "action_ids")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[collection][0][key] = ["missing"]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        self.data["completed_action_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_prerequisite_cycle_rejected(self):
        self.data["actions"][0]["prerequisites"] = ["retry-checkout"]
        with self.assertRaisesRegex(app.ValidationError, "cyclic"):
            app.run_pipeline(self.data)

    def test_stage_order_and_tampered_handoff_rejected(self):
        state = self.empty_state()
        with self.assertRaises(app.ValidationError):
            app.run_stage(state, "journey")
        state = app.run_stage(state, "sentiment")
        state["stages"]["sentiment"]["issues"][0]["score"] = 1000
        with self.assertRaises(app.ValidationError):
            app.run_stage(state, "normal")

    def test_tampered_citation_and_boolean_score_rejected(self):
        state = app.run_pipeline(self.data)
        state["stages"]["normal"]["findings"][0]["citation"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.validate(state, "state")
        state = app.run_pipeline(self.data)
        state["stages"]["sentiment"]["issues"][0]["priority_rank"] = True
        with self.assertRaises(app.ValidationError):
            app.validate(state, "state")

    def test_tampered_journey_and_faq_rejected(self):
        for stage in ("journey", "faq"):
            with self.subTest(stage=stage):
                state = app.run_pipeline(self.data)
                if stage == "journey":
                    state["stages"][stage]["steps"].reverse()
                else:
                    state["stages"][stage]["answers"][0]["answer"] = "Unsupported answer"
                with self.assertRaises(app.ValidationError):
                    app.validate(state, "state")

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(self.data, original)
        first["input"]["feedback"][0]["text"] = "changed"
        self.assertEqual(self.data, original)

    def test_cli_success_single_json_object(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_file_error_and_argument_errors(self):
        for args in ((), ("not-present.json",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stderr, "")
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_schema_error_real_subprocess(self):
        process = self.cli("build_manifest.json")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_malformed_duplicate_and_nonfinite_json(self):
        for value in ("{", '{"schema_version": "1.0", "schema_version": "1.0"}',
                      '{"x": NaN}', "null", "[]"):
            with self.subTest(value=value):
                code, result = self.invoke_text(value)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_encoding_error(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("invalid encoding")):
            with contextlib.redirect_stdout(output):
                self.assertEqual(app.main(["fixture.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
