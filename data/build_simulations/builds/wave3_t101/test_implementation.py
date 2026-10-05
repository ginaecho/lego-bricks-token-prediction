"""All fixtures are fictitious. Tests create no files and call no providers."""

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


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def output(self):
        return app.run_pipeline(self.data)

    def test_01_complete_pipeline_order(self):
        output = self.output()
        self.assertEqual(list(output["stages"]), list(app.PHASES))
        self.assertEqual(output["status"], "ok")
        self.assertIs(app.validate_envelope(output, "recommend"), output)

    def test_02_onboarding_missing_documents(self):
        output = app.advance(self.data, "onboarding")["stages"]["onboarding"]
        self.assertEqual(output["missing_documents"], ["insurance-summary", "referral"])
        self.assertIn("Prepare insurance-summary", output["next_step"])

    def test_03_returning_caregiver_onboarding(self):
        self.data["customer"]["experience"] = "returning"
        self.data["records"]["patient"]["age_band"] = "child"
        self.assertIn("Continue", self.output()["stages"]["onboarding"]["next_step"])
        self.assertIn("caregiver", self.output()["stages"]["onboarding"]["next_step"])

    def test_04_complete_document_checklist(self):
        claim = self.data["records"]["prior_authorization_request"]
        claim["provided_documents"] = list(claim["required_documents"])
        onboarding = self.output()["stages"]["onboarding"]
        self.assertEqual(onboarding["missing_documents"], [])
        self.assertIn("human navigator", onboarding["next_step"])

    def test_05_support_is_grounded(self):
        support = self.output()["stages"]["support"]
        self.assertEqual((support["source_id"], support["answer"]), app.KB["authorization"])
        self.assertEqual(support["availability"], "offline-knowledge")
        self.assertFalse(support["escalated"])

    def test_06_onboarding_support_handoff(self):
        self.data["customer"]["goal"] = "portal"
        self.data["request"].update(question="", search_query="")
        stages = self.output()["stages"]
        self.assertEqual(stages["support"]["intent"], "portal")
        self.assertEqual(stages["support"]["onboarding_next_step"],
                         stages["onboarding"]["next_step"])

    def test_07_clinical_question_requires_human(self):
        self.data["request"]["question"] = "Should I increase insulin dose?"
        stages = self.output()["stages"]
        self.assertTrue(stages["support"]["escalated"])
        self.assertEqual(stages["support"]["intent"], "clinical")
        self.assertIn("cannot diagnose", stages["support"]["answer"])
        self.assertEqual(stages["search"]["query_terms"], ["records"])
        self.assertEqual(stages["recommend"]["clinical_decision"], "not-made")

    def test_08_clinical_query_cannot_bypass_support(self):
        self.data["request"]["search_query"] = "prescribe treatment"
        self.assertEqual(self.output()["stages"]["support"]["intent"], "clinical")

    def test_09_unknown_question_escalates(self):
        self.data["request"].update(question="unicorn mystery", search_query="unicorn")
        stages = self.output()["stages"]
        self.assertEqual(stages["support"]["intent"], "unknown")
        self.assertTrue(stages["support"]["escalated"])
        self.assertTrue(stages["search"]["no_results"])
        self.assertEqual(stages["recommend"]["recommendations"], [])
        self.assertIn("human care navigator", stages["recommend"]["next_step"])

    def test_10_search_synonym_and_typo(self):
        stages = self.output()["stages"]
        self.assertEqual(stages["support"]["search_terms"], ["authorization"])
        self.assertEqual(stages["search"]["query_terms"], ["authorization"])
        self.assertTrue(stages["search"]["results"])

    def test_11_support_search_handoff(self):
        stages = self.output()["stages"]
        self.assertEqual(stages["search"]["support_source_id"], stages["support"]["source_id"])
        self.assertEqual(stages["search"]["query_terms"], stages["support"]["search_terms"])

    def test_12_search_recommendation_subset_and_personalization(self):
        stages = self.output()["stages"]
        ids = [item["resource_id"] for item in stages["search"]["results"]]
        recommendations = stages["recommend"]["recommendations"]
        self.assertEqual(stages["recommend"]["search_result_ids"], ids)
        self.assertTrue(all(item["resource_id"] in ids for item in recommendations))
        self.assertEqual(recommendations[0]["resource_id"], "resource-auth-audio")
        self.assertIn("Matches preferred audio format.", recommendations[0]["reasons"])

    def test_13_recommendation_limit(self):
        self.data["request"]["limit"] = 1
        self.assertEqual(len(self.output()["stages"]["recommend"]["recommendations"]), 1)

    def test_14_language_and_accessibility_filter(self):
        self.data["customer"].update(language="es", goal="portal", preferred_format="video")
        self.data["request"].update(question="portal help", search_query="login")
        results = self.output()["stages"]["search"]["results"]
        self.assertEqual([item["resource_id"] for item in results], ["resource-portal-text"])

    def test_15_no_language_match(self):
        self.data["customer"].update(language="es", goal="billing")
        self.data["request"].update(question="billing", search_query="invoice")
        self.assertTrue(self.output()["stages"]["search"]["no_results"])

    def test_16_deterministic_and_does_not_mutate_input(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.output(), self.output())
        self.assertEqual(self.data, original)

    def test_17_audit_replay_covers_all_record_changes(self):
        output = self.output()
        replay = copy.deepcopy(self.data["records"])
        for index, event in enumerate(output["audit"], 1):
            claim = replay["prior_authorization_request"]
            self.assertEqual(index, event["sequence"])
            self.assertEqual(event["before"], claim[event["field"]])
            self.assertEqual(event["record_reference"], "Claim/" + claim["id"])
            claim[event["field"]] = event["after"]
            self.assertTrue(event["human_review_required"])
        self.assertEqual(replay, output["records"])
        self.assertEqual(len(output["audit"]), 5)
        self.assertEqual(output["records"]["prior_authorization_request"]["status"], "draft")

    def test_18_no_clinical_record_mutation(self):
        output = self.output()
        for kind in ("patient", "clinical_note"):
            self.assertEqual(output["records"][kind], self.data["records"][kind])

    def test_19_direct_patient_identifiers_rejected(self):
        for field, value in (("name", "Synthetic Example"),
                             ("birthDate", "1900-01-01"),
                             ("identifier", [{"value": "EXAMPLE-MRN"}]),
                             ("address", "Example street")):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["records"]["patient"][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_20_untemplated_note_rejected(self):
        self.data["records"]["clinical_note"]["text"] += " Patient name: Example."
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_21_identifying_or_unbounded_request_rejected(self):
        for text in ("example@example.invalid", "MRN 12345", "1900-01-01",
                     "Call 555-0100", "A" * 501, "Alexandra"):
            with self.subTest(text=text):
                self.data["request"]["question"] = text
                with self.assertRaises(app.ValidationError):
                    self.output()

    def test_22_non_synthetic_records_rejected(self):
        self.data["records"]["patient"]["id"] = "patient-123"
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_23_human_review_flags_required(self):
        for record in self.data["records"]:
            with self.subTest(record=record):
                data = copy.deepcopy(self.data)
                data["records"][record]["human_review_required"] = False
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        self.data["human_review_required"] = False
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_24_reference_mismatch_rejected(self):
        self.data["records"]["clinical_note"]["subject"]["reference"] = "Patient/synthetic-patient-002"
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_25_autonomous_approval_rejected(self):
        self.data["records"]["prior_authorization_request"]["status"] = "approved"
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_26_invalid_lab_values_rejected(self):
        for value in (True, "7.2", float("nan"), float("inf"), -1, 10001):
            with self.subTest(value=value):
                self.data["records"]["patient"]["labs"][0]["value"] = value
                with self.assertRaises(app.ValidationError):
                    self.output()

    def test_27_empty_clinical_context_allowed(self):
        patient = self.data["records"]["patient"]
        patient.update(diagnoses=[], labs=[])
        self.data["records"]["clinical_note"]["text"] = app.canonical_note(patient)
        self.assertEqual(self.output()["status"], "ok")

    def test_28_invalid_limits_rejected(self):
        for limit in (0, 6, True, "3", 1.5):
            with self.subTest(limit=limit):
                self.data["request"]["limit"] = limit
                with self.assertRaises(app.ValidationError):
                    self.output()

    def test_29_skip_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.advance(self.data, "search")
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.output())

    def test_30_tampered_support_handoff_rejected(self):
        state = app.advance(app.advance(self.data, "onboarding"), "support")
        state["stages"]["support"]["answer"] = "Your authorization is approved."
        with self.assertRaises(app.ValidationError):
            app.advance(state, "search")

    def test_31_tampered_search_handoff_rejected(self):
        state = self.data
        for phase in app.PHASES[:3]:
            state = app.advance(state, phase)
        state["stages"]["search"]["results"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.advance(state, "recommend")

    def test_32_tampered_audit_rejected(self):
        for mutation in ("delete", "before", "record"):
            with self.subTest(mutation=mutation):
                output = self.output()
                if mutation == "delete":
                    output["audit"].pop()
                elif mutation == "before":
                    output["audit"][0]["before"] = "approved"
                else:
                    output["records"]["prior_authorization_request"]["workflow_status"] = "approved"
                with self.assertRaises(app.ValidationError):
                    app.validate_envelope(output)

    def test_33_injected_fixture_responder(self):
        calls = []

        def responder(evidence):
            calls.append(copy.deepcopy(evidence))
            return {"answer": evidence["answer"], "source_id": evidence["source_id"]}

        self.assertEqual(app.run_pipeline(self.data, responder), self.output())
        self.assertEqual(len(calls), 1)
        self.assertNotIn("records", calls[0])

    def test_34_injected_hallucination_or_failure_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data, lambda evidence: {"answer": "Take insulin",
                                                         "source_id": evidence["source_id"]})

        def broken(evidence):
            raise RuntimeError("fixture failure")

        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data, broken)

    def test_35_injected_evidence_mutation_cannot_change_grounding(self):
        def responder(evidence):
            evidence["answer"] = "Autonomous approval"
            return {"answer": evidence["answer"], "source_id": evidence["source_id"]}

        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data, responder)

    def test_36_invalid_root_and_unknown_fields(self):
        for data in (None, [], {}, dict(self.data, unexpected=True)):
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_37_duplicate_documents_rejected(self):
        self.data["records"]["prior_authorization_request"]["provided_documents"] *= 2
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_38_intermediate_states_survive_json_roundtrip(self):
        state = self.data
        for phase in app.PHASES:
            state = app.advance(state, phase)
            state = json.loads(json.dumps(state, sort_keys=True))
            app.validate_envelope(state, phase)


class CliTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                              cwd=str(HERE), capture_output=True, text=True, check=False)

    def assert_error(self, process):
        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_39_cli_success(self):
        process = self.invoke("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_40_cli_missing_file(self):
        self.assert_error(self.invoke("nonexistent-input.json"))

    def test_41_cli_invalid_json(self):
        self.assert_error(self.invoke("implementation.py"))

    def test_42_cli_usage(self):
        self.assert_error(self.invoke())
        self.assert_error(self.invoke("example_input.json", "extra"))

    def test_43_cli_directory_error(self):
        self.assert_error(self.invoke("."))

    def test_44_cli_duplicate_keys_nonfinite_and_invalid_schema(self):
        for text in ('{"schema_version":1,"schema_version":1}', '{"number":NaN}',
                     '{"number":Infinity}', '[]', '{}', 'null', '{"secret":"EXAMPLE"}'):
            with self.subTest(text=text):
                stdout = io.StringIO()
                with patch("builtins.open", mock_open(read_data=text)):
                    with contextlib.redirect_stdout(stdout):
                        self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")
                self.assertNotIn("EXAMPLE", stdout.getvalue())

    def test_45_cli_oversized_file(self):
        stdout = io.StringIO()
        with patch("builtins.open", mock_open(read_data=" " * 131073)):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(app.main(["fixture.json"]), 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
