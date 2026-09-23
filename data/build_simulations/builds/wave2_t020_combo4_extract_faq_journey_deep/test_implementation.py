"""Synthetic fixtures only; test execution creates no files or network traffic."""

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

    def pipeline(self):
        return app.run_pipeline(self.request)

    def answers(self, result):
        return {a["id"]: a for a in result["faq"]["answers"]}

    def intake(self, old, new):
        self.request["documents"][0]["text"] = self.request["documents"][0]["text"].replace(old, new)

    def invoke_main(self, raw, argv=None):
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data=raw)), redirect_stdout(output):
            code = app.main(["fixture.json"] if argv is None else argv)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        return code, json.loads(lines[0])

    def test_complete_pipeline_order(self):
        result = self.pipeline()
        self.assertEqual(list(result), ["schema_version", "synthetic", "status",
                                        "extract", "faq", "journey", "deep"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["journey"]["status"], "ready")

    def test_typed_extraction_and_exact_spans(self):
        result = app.extract(self.request)["extract"]
        self.assertEqual(result["values"], {"name": "Ada Example", "plan": "starter",
                                          "seats": 4, "consent": True, "region": None})
        documents = {d["id"]: d["text"] for d in self.request["documents"]}
        for span in result["spans"].values():
            if span is not None:
                self.assertEqual(span["text"], documents[span["document_id"]][span["start"]:span["end"]])

    def test_missing_required_and_optional_fields(self):
        self.intake("Name: Ada Example\n", "")
        result = self.pipeline()["extract"]
        self.assertEqual(result["missing_fields"], ["name", "region"])
        self.assertEqual(result["required_missing"], ["name"])

    def test_unicode_offsets_and_trimmed_values(self):
        self.intake("Name: Ada Example", "Name:   Zoë 🧱  ")
        result = app.extract(self.request)["extract"]
        self.assertEqual(result["values"]["name"], "Zoë 🧱")
        span = result["spans"]["name"]
        self.assertEqual(self.request["documents"][0]["text"][span["start"]:span["end"]], "Zoë 🧱")

    def test_invalid_typed_value_becomes_missing(self):
        self.intake("Seats: 4", "Seats: four")
        self.intake("Consent: yes", "Consent: perhaps")
        result = self.pipeline()["extract"]
        self.assertIsNone(result["values"]["seats"])
        self.assertIsNone(result["spans"]["seats"])
        self.assertEqual(result["required_missing"], ["seats", "consent"])

    def test_first_valid_match_in_document_order(self):
        self.intake("Seats: 4", "Seats: invalid\nSeats: 7\nSeats: 9")
        self.assertEqual(app.extract(self.request)["extract"]["values"]["seats"], 7)

    def test_false_and_zero_are_present_not_missing(self):
        self.intake("Seats: 4", "Seats: 0")
        self.intake("Consent: yes", "Consent: no")
        output = self.pipeline()
        self.assertEqual(output["extract"]["values"]["seats"], 0)
        self.assertIs(output["extract"]["values"]["consent"], False)
        self.assertEqual(output["extract"]["required_missing"], [])
        self.assertEqual(output["journey"]["status"], "ready")

    def test_faq_returns_grounded_complete_quote(self):
        answers = self.answers(self.pipeline())
        self.assertEqual(answers["trial"]["answer"], "Starter accounts can activate a guided trial.")
        self.assertEqual(answers["export"]["answer"], "Starter accounts can export CSV reports.")
        self.assertEqual(answers["trial"]["answer"], answers["trial"]["source"]["text"])
        self.assertEqual(answers["trial"]["score"], 1.0)

    def test_faq_missing_field_abstention(self):
        answer = self.answers(self.pipeline())["residency"]
        self.assertEqual(answer["status"], "abstained")
        self.assertEqual(answer["reason"], "missing_required_fields")
        self.assertEqual(answer["missing_fields"], ["region"])
        self.assertIsNone(answer["source"])

    def test_faq_unrelated_question_abstains(self):
        self.request["questions"][0]["text"] = "Lunar telescope warranty"
        answer = self.answers(self.pipeline())["trial"]
        self.assertEqual(answer["reason"], "no_relevant_evidence")
        self.assertIsNone(answer["answer"])

    def test_faq_empty_token_query_abstains(self):
        self.request["questions"][0]["text"] = "What is the?"
        self.assertEqual(self.answers(self.pipeline())["trial"]["status"], "abstained")

    def test_extraction_changes_kb_eligibility_and_downstream_stages(self):
        self.intake("Plan: starter", "Plan: enterprise")
        result = self.pipeline()
        self.assertEqual(self.answers(result)["trial"]["status"], "abstained")
        self.assertEqual(result["journey"]["steps"], [])
        self.assertEqual(result["deep"]["topics"], [])
        self.assertEqual(result["deep"]["unresolved_questions"][0]["reasons"],
                         ["no_journey_research_topics"])

    def test_missing_plan_propagates_to_faq_and_journey(self):
        self.intake("Plan: starter\n", "")
        result = self.pipeline()
        self.assertIn("plan", result["extract"]["required_missing"])
        self.assertEqual(self.answers(result)["trial"]["missing_fields"], ["plan"])
        self.assertEqual(result["journey"]["status"], "blocked")

    def test_retrieval_tie_breaks_by_id(self):
        duplicate = copy.deepcopy(self.request["knowledge_base"][0])
        duplicate["id"] = "aaa"
        self.request["knowledge_base"].append(duplicate)
        self.assertEqual(self.answers(self.pipeline())["trial"]["kb_id"], "aaa")

    def test_two_step_journey_obeys_prerequisites_not_global_priority(self):
        journey = self.pipeline()["journey"]
        self.assertEqual(journey["next_actions"], ["activate"])
        self.assertEqual([s["action_id"] for s in journey["steps"]], ["activate", "export"])
        self.assertEqual(journey["steps"][1]["prerequisites"], ["activate"])
        self.assertEqual(journey["blocked"][0]["unmet_prerequisites"], ["activate"])

    def test_completed_action_is_not_recommended(self):
        self.request["completed_actions"] = ["activate"]
        journey = self.pipeline()["journey"]
        self.assertEqual(journey["status"], "partial")
        self.assertEqual(journey["next_actions"], ["export"])
        self.assertEqual(len(journey["steps"]), 1)

    def test_missing_faq_evidence_prevents_dependent_second_step(self):
        self.request["knowledge_base"].pop(1)
        result = self.pipeline()
        self.assertEqual(result["journey"]["status"], "partial")
        self.assertEqual(result["deep"]["topics"], ["guided-onboarding"])

    def test_completed_all_eligible_actions_reports_blocked(self):
        self.request["completed_actions"] = ["activate", "export"]
        journey = self.pipeline()["journey"]
        self.assertEqual(journey["steps"], [])
        self.assertEqual(journey["next_actions"], [])

    def test_research_topics_come_only_from_selected_journey(self):
        deep = self.pipeline()["deep"]
        self.assertEqual(deep["topics"], ["guided-onboarding", "export-reliability"])
        self.assertNotIn("residency", deep["topics"])

    def test_research_multidocument_disagreement_preserves_evidence(self):
        deep = self.pipeline()["deep"]
        summary = deep["synthesis"][0]
        self.assertEqual(summary["assessment"], "disputed")
        self.assertEqual(summary["document_ids"], ["synthetic-study-a", "synthetic-study-b"])
        self.assertEqual(len(summary["evidence"]), 2)
        self.assertEqual(deep["disagreements"][0]["supporting_claims"], ["pilot-a"])
        self.assertEqual(deep["disagreements"][0]["opposing_claims"], ["pilot-b"])
        self.assertEqual(deep["unresolved_questions"][0]["reasons"], ["conflicting_evidence"])
        self.assertTrue(all(e["source"]["text"].endswith(".") for e in summary["evidence"]))

    def test_research_reports_absent_evidence(self):
        deep = self.pipeline()["deep"]
        self.assertEqual(deep["synthesis"][1]["assessment"], "insufficient_evidence")
        self.assertEqual(deep["unresolved_questions"][1]["reasons"], ["no_evidence"])

    def test_single_document_does_not_imply_corroboration(self):
        self.request["claims"].pop()
        deep = self.pipeline()["deep"]
        self.assertEqual(deep["synthesis"][0]["assessment"], "supported")
        self.assertEqual(deep["unresolved_questions"][0]["reasons"], ["single_document_only"])
        self.assertEqual(deep["disagreements"], [])

    def test_agreeing_documents_are_not_disagreement(self):
        self.request["claims"][1]["stance"] = "supports"
        document = self.request["documents"][3]
        document["text"] = "Synthetic pilot B supports guided onboarding for faster activation."
        self.request["claims"][1]["source"]["end"] = len(document["text"])
        deep = self.pipeline()["deep"]
        self.assertEqual(deep["synthesis"][0]["assessment"], "supported")
        self.assertEqual(deep["disagreements"], [])
        self.assertEqual([q["topic"] for q in deep["unresolved_questions"]], ["export-reliability"])

    def test_empty_collections_are_valid_and_abstain(self):
        self.request.update(documents=[], questions=[], knowledge_base=[], actions=[],
                            claims=[], research_questions={})
        self.request["extraction"] = {"document_ids": [], "fields": []}
        result = self.pipeline()
        self.assertEqual(result["extract"]["values"], {})
        self.assertEqual(result["faq"]["answers"], [])
        self.assertEqual(result["journey"]["status"], "blocked")

    def test_pipeline_does_not_mutate_input_or_previous_state(self):
        original = copy.deepcopy(self.request)
        previous = app.extract(self.request)
        snapshot = copy.deepcopy(previous)
        result = app.faq(self.request, previous)
        result["extract"]["values"]["name"] = "changed"
        self.assertEqual(previous, snapshot)
        self.assertEqual(self.request, original)

    def test_deterministic_repeated_pipeline(self):
        self.assertEqual(json.dumps(self.pipeline()), json.dumps(self.pipeline()))

    def test_rejects_forged_extraction_span(self):
        state = app.extract(self.request)
        state["extract"]["spans"]["name"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.faq(self.request, state)

    def test_rejects_forged_missing_extraction(self):
        state = app.extract(self.request)
        state["extract"]["values"]["name"] = None
        state["extract"]["spans"]["name"] = None
        state["extract"]["missing_fields"] = ["name", "region"]
        state["extract"]["required_missing"] = ["name"]
        with self.assertRaises(app.ValidationError):
            app.faq(self.request, state)

    def test_rejects_forged_faq_answer(self):
        state = app.faq(self.request, app.extract(self.request))
        state["faq"]["answers"][0]["answer"] = "An ungrounded promise"
        with self.assertRaises(app.ValidationError):
            app.journey(self.request, state)

    def test_rejects_reordered_journey(self):
        state = app.journey(self.request, app.faq(self.request, app.extract(self.request)))
        state["journey"]["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.deep(self.request, state)

    def test_rejects_forged_research_evidence(self):
        state = self.pipeline()
        state["deep"]["synthesis"][0]["evidence"][0]["source"]["text"] = "Invented finding"
        with self.assertRaises(app.ValidationError):
            app.Validator(self.request).state(state, "deep")

    def test_handoff_rejects_boolean_step_number(self):
        state = self.pipeline()
        state["journey"]["steps"][0]["step"] = True
        with self.assertRaises(app.ValidationError):
            app.Validator(self.request).state(state, "deep")

    def test_invalid_regex_and_capture_counts(self):
        for pattern in ("[", "no capture", "(one)(two)"):
            with self.subTest(pattern=pattern):
                self.request["extraction"]["fields"][0]["pattern"] = pattern
                with self.assertRaises(app.ValidationError):
                    self.pipeline()

    def test_invalid_source_boundaries_and_unknown_document(self):
        for source in ({"document_id": "absent", "start": 0, "end": 3},
                       {"document_id": "synthetic-kb", "start": -1, "end": 3},
                       {"document_id": "synthetic-kb", "start": 0, "end": 10000},
                       {"document_id": "synthetic-kb", "start": 3, "end": 3},
                       {"document_id": "synthetic-kb", "start": False, "end": 3}):
            with self.subTest(source=source):
                self.request["knowledge_base"][0]["source"] = source
                with self.assertRaises(app.ValidationError):
                    self.pipeline()

    def test_duplicate_ids_rejected(self):
        self.request["documents"].append(copy.deepcopy(self.request["documents"][0]))
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_cycles_and_unknown_prerequisites_rejected(self):
        for prerequisite in ("export", "absent", "activate"):
            with self.subTest(prerequisite=prerequisite):
                self.request["actions"][0]["prerequisites"] = [prerequisite]
                with self.assertRaises(app.ValidationError):
                    self.pipeline()

    def test_completed_actions_require_valid_prerequisites(self):
        self.request["completed_actions"] = ["export"]
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_unknown_cross_stage_references_rejected(self):
        changes = [
            ("questions", "required_fields"),
            ("actions", "requires_fields"),
            ("actions", "requires_answers"),
            ("actions", "research_topics"),
        ]
        for collection, field in changes:
            with self.subTest(collection=collection, field=field):
                candidate = copy.deepcopy(self.request)
                candidate[collection][0][field] = ["absent"]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(candidate)

    def test_input_version_unknown_keys_and_synthetic_marker_rejected(self):
        for update in ({"schema_version": True}, {"schema_version": 2},
                       {"synthetic": False}, {"extra": 1}):
            with self.subTest(update=update):
                candidate = dict(self.request, **update)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(candidate)

    def test_boolean_does_not_satisfy_integer_requirement(self):
        self.request["knowledge_base"][0]["requires"] = {"seats": True}
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_cli_actual_success_one_json_no_stderr(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout), self.pipeline())

    def test_cli_actual_missing_file(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py", "nonexistent-fixture.json"],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_actual_usage_error(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py"],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_malformed_json_duplicate_keys_nonfinite_and_wrong_shapes(self):
        for raw in ("{", '{"schema_version":1,"schema_version":1}', '{"x":NaN}',
                    "null", "[]", "1", '{"documents":[]}'):
            with self.subTest(raw=raw):
                code, result = self.invoke_main(raw)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_nested_bad_types_return_json_errors(self):
        for field, value in (("documents", None), ("actions", [1]), ("claims", {}),
                             ("research_questions", []), ("questions", "not-an-array")):
            with self.subTest(field=field):
                candidate = dict(self.request, **{field: value})
                code, result = self.invoke_main(json.dumps(candidate))
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_permission_and_decoding_errors_return_json(self):
        for error in (PermissionError("denied"), UnicodeError("invalid UTF-8")):
            with self.subTest(error=error):
                output = io.StringIO()
                with patch("builtins.open", side_effect=error), redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
