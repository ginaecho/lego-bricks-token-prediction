"""Synthetic fixtures only; no network, providers, temporary files, or dependencies."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    return json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))


def through(stage, data=None):
    result = app.intake(fixture() if data is None else data)
    for next_stage in app.STAGES[1:app.STAGES.index(stage) + 1]:
        result = app.advance(result, next_stage)
    return result


class PipelineTests(unittest.TestCase):
    def test_complete_pipeline_and_determinism(self):
        result = app.run(fixture())
        self.assertEqual(result, app.run(fixture()))
        self.assertEqual(result["stage"], "sentiment")
        self.assertEqual(list(result["payload"]), ["deep", "extract", "support", "sentiment"])
        app.validate(result, "sentiment")

    def test_synthesis_agreement_and_disagreement(self):
        deep = through("deep")["payload"]["deep"]
        self.assertEqual([d["field"] for d in deep["disagreements"]], ["authorization_status"])
        lab = next(f for f in deep["findings"] if f["field"] == "hba1c")
        self.assertEqual(lab["values"], [8.2])
        self.assertEqual(len(lab["evidence_ids"]), 2)
        self.assertEqual(len(deep["sources_reviewed"]), 4)

    def test_source_spans_are_exact(self):
        record = through("deep")
        documents = {d["id"]: d["text"] for d in record["documents"]}
        for e in record["payload"]["deep"]["evidence"]:
            span = e["span"]
            self.assertEqual(documents[e["source_id"]][span["start"]:span["end"]], e["quote"])

    def test_research_records_unresolved_questions(self):
        questions = through("deep")["payload"]["deep"]["unresolved_questions"]
        self.assertTrue(any("conflicting authorization_status" in q for q in questions))
        self.assertTrue(any("missing decision_reason" in q for q in questions))

    def test_extraction_types_missing_and_conflicts(self):
        extracted = through("extract")["payload"]["extract"]
        self.assertEqual(extracted["fields"]["hba1c"]["value"], 8.2)
        self.assertEqual(extracted["fields"]["authorization_status"]["status"], "conflict")
        self.assertIsNone(extracted["fields"]["authorization_status"]["value"])
        self.assertEqual(extracted["missing_required"], ["decision_reason"])

    def test_invalid_numeric_evidence_is_not_silently_discarded(self):
        data = fixture()
        data["documents"][-1]["text"] = data["documents"][-1]["text"].replace("HbA1c: 8.2", "HbA1c: unknown")
        record = app.run(data)
        self.assertEqual(record["payload"]["extract"]["fields"]["hba1c"]["status"], "invalid")
        self.assertIn("hba1c:invalid", record["payload"]["sentiment"]["issues"])

    def test_boolean_extraction(self):
        data = fixture()
        data["field_schema"].append({"name": "consent", "type": "boolean", "required": True,
                                     "sources": [{"label": "Consent"}]})
        data["documents"][-1]["text"] += "\nConsent: false"
        extracted = app.run(data)["payload"]["extract"]["fields"]["consent"]
        self.assertIs(extracted["value"], False)

    def test_optional_missing_field(self):
        data = fixture()
        data["field_schema"][-1]["required"] = False
        result = app.run(data)
        self.assertEqual(result["payload"]["extract"]["missing_required"], [])
        self.assertNotIn("decision_reason:missing", result["payload"]["support"]["issues"])

    def test_offline_grounded_support(self):
        record = app.run(fixture())
        support = record["payload"]["support"]
        self.assertIn("team is offline", support["answer"])
        self.assertIn("does not contact them or create a ticket", support["answer"])
        self.assertIn("hba1c: 8.2", support["answer"])
        self.assertEqual(support["action_taken"], "none")
        evidence_ids = {e["id"] for e in record["payload"]["deep"]["evidence"]}
        for citation in support["citations"]:
            self.assertTrue(set(citation["evidence_ids"]) <= evidence_ids)
        self.assertNotIn("authorization_status", [c["field"] for c in support["citations"]])

    def test_online_response_does_not_claim_offline(self):
        data = fixture()
        data["request"]["team_online"] = True
        self.assertNotIn("team is offline", app.run(data)["payload"]["support"]["answer"])

    def test_clinical_decision_boundary(self):
        data = fixture()
        data["request"]["message"] = "Should I change my medication dose?"
        support = app.run(data)["payload"]["support"]
        self.assertTrue(support["clinical_boundary_applied"])
        self.assertIn("cannot diagnose, recommend treatment, change medication", support["answer"])
        self.assertEqual(support["action_taken"], "none")

    def test_sentiment_scores_customer_not_assistant(self):
        record = app.run(fixture())
        sentiment = record["payload"]["sentiment"]
        self.assertEqual(sentiment["scored_text"], record["payload"]["support"]["customer_message"])
        self.assertEqual(sentiment["score"], -3)
        self.assertEqual(sentiment["label"], "negative")
        self.assertEqual(sum(m["weight"] for m in sentiment["matches"]), sentiment["unclamped_score"])
        for match in sentiment["matches"]:
            self.assertEqual(sentiment["scored_text"][match["start"]:match["end"]], match["token"])

    def test_severity_overrides_positive_sentiment(self):
        data = fixture()
        data["request"]["message"] = "Thanks, you are great and helpful, but I have chest pain."
        record = app.run(data)
        sentiment = record["payload"]["sentiment"]
        self.assertGreater(sentiment["score"], 0)
        self.assertEqual(sentiment["priority"], "P1")
        self.assertIn("contact local emergency services", record["payload"]["support"]["answer"])
        self.assertTrue(sentiment["human_review_required"])

    def test_deadline_prioritization(self):
        result = app.run(fixture())["payload"]["sentiment"]
        self.assertEqual(result["priority"], "P2")
        self.assertEqual(result["deadline_signals"], ["deadline"])

    def test_negation(self):
        data = fixture()
        data["request"]["message"] = "I am not happy."
        result = app.run(data)["payload"]["sentiment"]
        self.assertEqual(result["score"], -2)
        self.assertTrue(result["matches"][0]["negated"])

    def test_negation_does_not_cross_sentence(self):
        data = fixture()
        data["request"]["message"] = "No. Great!"
        self.assertEqual(app.run(data)["payload"]["sentiment"]["score"], 2)

    def test_neutral_and_routine(self):
        data = fixture()
        data["documents"][-1]["text"] = data["documents"][-1]["text"].replace(
            "Authorization status: cancelled", "Authorization status: active")
        data["documents"][-1]["text"] += "\nDecision reason: Synthetic administrative review"
        data["request"]["message"] = "Please show the supplied records."
        data["request"]["team_online"] = True
        result = app.run(data)["payload"]["sentiment"]
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["priority"], "P3")

    def test_sentiment_clamps_and_retains_raw_score(self):
        data = fixture()
        data["request"]["message"] = "angry " * 10
        result = app.run(data)["payload"]["sentiment"]
        self.assertEqual(result["score"], -10)
        self.assertEqual(result["unclamped_score"], -30)

    def test_patient_deidentification_and_known_token_redaction(self):
        data = fixture()
        # Deliberate fictitious direct-identifier placeholders, never real identities.
        data["patient"]["name"] = [{"text": "DEMO_PERSON_ALPHA"}]
        data["patient"]["identifier"] = [{"value": "DEMO_IDENTIFIER_ALPHA"}]
        data["patient"]["birthDate"] = "DEMO_BIRTH_DATE"
        data["patient"]["telecom"] = [{"value": "DEMO_CONTACT_ALPHA"}]
        data["patient"]["address"] = [{"text": "DEMO_ADDRESS_ALPHA"}]
        data["documents"][-1]["text"] += "\nPatient name: DEMO_PERSON_ALPHA"
        data["request"]["message"] = "DEMO_IDENTIFIER_ALPHA asks about synthetic-patient-001."
        result = app.run(data)
        encoded = json.dumps(result)
        for secret in ("DEMO_PERSON_ALPHA", "DEMO_IDENTIFIER_ALPHA", "DEMO_BIRTH_DATE",
                       "DEMO_CONTACT_ALPHA", "DEMO_ADDRESS_ALPHA", "synthetic-patient-001"):
            self.assertNotIn(secret, encoded)
        self.assertIn("[REDACTED]", encoded)
        self.assertFalse(app.IDENTIFIER_KEYS & result["patient"].keys())

    def test_residual_identifier_rejected(self):
        for identifier in ("demo@example.invalid", "123-45-6789", "1900-01-01", "MRN: DEMO_UNLISTED"):
            data = fixture()
            data["request"]["message"] = "Synthetic test: " + identifier
            with self.subTest(identifier=identifier), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_no_real_data_mode(self):
        data = fixture()
        data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(data)

    def test_unknown_patient_reference(self):
        data = fixture()
        data["documents"][0]["resource"]["subject"]["reference"] = "Patient/synthetic-patient-other"
        with self.assertRaises(app.ValidationError):
            app.run(data)

    def test_resource_identifier_extension_rejected(self):
        data = fixture()
        data["documents"][0]["resource"]["identifier"] = "DEMO_IDENTIFIER"
        with self.assertRaises(app.ValidationError):
            app.run(data)

    def test_audit_covers_every_change(self):
        result = app.run(fixture())
        audit = result["audit"]
        self.assertEqual(len(audit), 11)
        self.assertEqual([e["event_id"] for e in audit], list(range(1, 12)))
        self.assertEqual([e["stage"] for e in audit[-4:]], ["deep", "extract", "support", "sentiment"])
        self.assertTrue(all(e["changed_fields"] and e["human_review_required"] for e in audit))

    def test_input_and_previous_stage_not_mutated(self):
        data = fixture()
        original = copy.deepcopy(data)
        first = app.intake(data)
        old = copy.deepcopy(first)
        app.advance(first, "deep")
        self.assertEqual(first, old)
        self.assertEqual(data, original)

    def test_tampered_provenance_is_rejected_at_handoff(self):
        result = through("deep")
        result["payload"]["deep"]["evidence"][0]["span"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.advance(result, "extract")

    def test_tampered_extracted_value_is_rejected(self):
        result = through("extract")
        result["payload"]["extract"]["fields"]["hba1c"]["value"] = 999
        with self.assertRaises(app.ValidationError):
            app.advance(result, "support")

    def test_tampered_support_message_is_rejected(self):
        result = through("support")
        result["payload"]["support"]["customer_message"] = "great"
        with self.assertRaises(app.ValidationError):
            app.advance(result, "sentiment")

    def test_policy_and_audit_tampering_rejected(self):
        for what in ("policy", "audit"):
            result = through("extract")
            if what == "policy":
                result["policy"]["human_review_required"] = False
            else:
                result["audit"].pop()
            with self.subTest(what=what), self.assertRaises(app.ValidationError):
                app.advance(result, "support")

    def test_stage_skip_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.advance(through("intake"), "support")

    def test_invalid_schemas(self):
        for modification in ("duplicate", "type", "selector", "required"):
            data = fixture()
            if modification == "duplicate":
                data["field_schema"].append(copy.deepcopy(data["field_schema"][0]))
            elif modification == "type":
                data["field_schema"][0]["type"] = "arbitrary"
            elif modification == "required":
                data["field_schema"][0]["required"] = "true"
            else:
                data["field_schema"][0]["sources"] = [{"resource_type": "Patient", "path": "name"}]
            with self.subTest(modification=modification), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_empty_and_duplicate_sources(self):
        for sources in ([], [fixture()["documents"][0]] * 2):
            data = fixture()
            data["documents"] = sources
            with self.assertRaises(app.ValidationError):
                app.run(data)

    def test_no_evidence_is_explicit(self):
        data = fixture()
        data["documents"] = [{"id": "synthetic-source-empty", "format": "clinical_note",
                              "text": "SYNTHETIC: No structured evidence available."}]
        result = app.run(data)
        self.assertEqual(result["payload"]["deep"]["evidence"], [])
        self.assertEqual(len(result["payload"]["extract"]["missing_required"]), 5)
        self.assertEqual(result["payload"]["support"]["citations"], [])

    def test_duplicate_selectors_do_not_duplicate_evidence(self):
        data = fixture()
        data["field_schema"][0]["sources"] *= 2
        deep = app.run(data)["payload"]["deep"]
        diagnosis = next(f for f in deep["findings"] if f["field"] == "diagnosis")
        self.assertEqual(len(diagnosis["evidence_ids"]), 2)

    def test_repeated_note_facts_and_unicode_spans(self):
        data = fixture()
        data["documents"][-1]["text"] = "SYNTHETIC café\nHbA1c: 8.2\nHbA1c: 9.1"
        record = app.run(data)
        self.assertEqual(record["payload"]["extract"]["fields"]["hba1c"]["status"], "conflict")
        for e in record["payload"]["deep"]["evidence"]:
            if e["source_id"] == "source-004":
                note = record["documents"][-1]["text"]
                self.assertEqual(note[e["span"]["start"]:e["span"]["end"]], e["quote"])

    def test_nonfinite_numeric_values(self):
        for bad in (float("nan"), float("inf")):
            data = fixture()
            data["documents"][1]["resource"]["valueQuantity"]["value"] = bad
            with self.assertRaises(app.ValidationError):
                app.run(data)

    def test_nonfinite_note_value_is_invalid(self):
        data = fixture()
        data["documents"][-1]["text"] += "\nHbA1c: NaN"
        self.assertEqual(app.run(data)["payload"]["extract"]["fields"]["hba1c"]["status"], "invalid")

    def test_unknown_top_level_field(self):
        data = fixture()
        data["freeform_identifiers"] = "DEMO"
        with self.assertRaises(app.ValidationError):
            app.run(data)

    def test_cli_success_single_json(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["stage"], "sentiment")

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "does-not-exist.json")],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_no_arguments(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema_without_file_writes(self):
        for contents in ("{bad", '{"schema_version":"1.0","schema_version":"2.0"}', "NaN", "[]",
                         '{"synthetic":false}', "1e9999"):
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=contents), contextlib.redirect_stdout(output):
                code = app.main([str(ROOT / "example_input.json")])
            with self.subTest(contents=contents):
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_input_does_not_echo_secrets(self):
        data = fixture()
        data["request"]["message"] = "demo@example.invalid"
        output = io.StringIO()
        with patch.object(Path, "read_text", return_value=json.dumps(data)), contextlib.redirect_stdout(output):
            code = app.main([str(ROOT / "example_input.json")])
        self.assertEqual(code, 2)
        self.assertNotIn("demo@example.invalid", output.getvalue())


if __name__ == "__main__":
    unittest.main()
