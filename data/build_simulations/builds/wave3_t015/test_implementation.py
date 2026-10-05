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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_discovery_personalization(self):
        result = app.recommend(self.data)
        self.assertEqual(result["discovery"][0]["product_id"], "syn-product-education")
        self.assertEqual(result["discovery"][0]["score"], 5)
        self.assertEqual(result["discovery"][1]["score"], 2)

    def test_preference_changes_rank(self):
        self.data["preferences"]["interests"] = ["paperwork"]
        self.assertEqual(app.recommend(self.data)["discovery"][0]["product_id"], "syn-product-auth")

    def test_exact_source_citations(self):
        result = app.run_pipeline(self.data)
        self.assertTrue(result["research"])
        sources = app.source_map(result)
        for finding in result["research"]:
            citation = finding["citation"]
            self.assertEqual(sources[citation["source_id"]][citation["start"]:citation["end"]],
                             finding["finding"])
            self.assertEqual(finding["finding"], citation["quote"])

    def test_cross_stage_propagation(self):
        self.data["preferences"]["limit"] = 1
        result = app.run_pipeline(self.data)
        self.assertTrue(result["research"])
        for finding in result["research"]:
            self.assertEqual(finding["recommendation_reference"], result["discovery"][0]["id"])
            self.assertEqual(finding["patient_reference"], self.data["patient_record"]["id"])

    def test_tampered_handoff_rejected(self):
        result = app.recommend(self.data)
        result["discovery"][0]["query_terms"] = ["unrelated"]
        with self.assertRaises(app.ValidationError):
            app.research(result)

    def test_human_review_and_no_clinical_decisions(self):
        result = app.run_pipeline(self.data)
        for record in result["discovery"] + result["research"]:
            self.assertIs(record["human_review_required"], True)
            self.assertIs(record["autonomous_clinical_decision"], False)
        self.assertEqual(result["prior_authorization_requests"][0]["status"], "draft")

    def test_audit_and_input_immutability(self):
        before = copy.deepcopy(self.data)
        result = app.run_pipeline(self.data)
        self.assertEqual(self.data, before)
        records = result["discovery"] + result["research"]
        self.assertEqual(len(records), len(result["audit_trail"]))
        for event, record in zip(result["audit_trail"], records):
            self.assertEqual(event["after_sha256"], app.fingerprint(record))
            self.assertEqual(event["record_id"], record["id"])
        result["audit_trail"].pop()
        with self.assertRaises(app.ValidationError):
            app.validate(result, "final")

    def test_direct_identifiers_rejected(self):
        for field, value in (("name", "Fictitious Person"), ("birthDate", "2000-01-01"),
                             ("identifier", [{"value": "fake"}])):
            data = copy.deepcopy(self.data)
            data["patient_record"][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_note_identifier_rejected(self):
        self.data["clinical_notes"][0]["text"] = "MRN: fictitious-123"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_synthetic_review_and_auth_guards(self):
        for collection, field, value in (
            ("patient_record", "synthetic", False),
            ("clinical_notes", "human_review_required", False),
            ("prior_authorization_requests", "status", "approved"),
        ):
            data = copy.deepcopy(self.data)
            record = data[collection] if collection == "patient_record" else data[collection][0]
            record[field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_invalid_references_and_duplicates(self):
        self.data["clinical_notes"][0]["subject"] = "Patient/syn-other"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.setUp()
        self.data["catalog"][0]["id"] = self.data["patient_record"]["id"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_invalid_limits_and_labs(self):
        for value in (True, 0, 11, "2"):
            data = copy.deepcopy(self.data)
            data["preferences"]["limit"] = value
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)
        for value in (True, -1, float("nan"), float("inf")):
            data = copy.deepcopy(self.data)
            data["patient_record"]["labs"][0]["value"] = value
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_empty_catalog(self):
        self.data["catalog"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["discovery"], [])
        self.assertEqual(result["research"], [])
        self.assertEqual(result["audit_trail"], [])

    def test_no_evidence_is_not_fabricated(self):
        self.data["clinical_notes"] = []
        self.data["prior_authorization_requests"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(result["discovery"])
        self.assertEqual(result["research"], [])

    def test_tie_break_and_determinism(self):
        self.data["preferences"]["interests"] = []
        self.data["patient_record"]["conditions"] = []
        self.data["prior_authorization_requests"] = []
        first = app.run_pipeline(self.data)
        self.data["catalog"].reverse()
        second = app.run_pipeline(self.data)
        self.assertEqual(first["discovery"], second["discovery"])
        self.assertEqual(first["research"], second["research"])
        self.assertEqual(first["audit_trail"], second["audit_trail"])

    def test_unicode_citation_and_tampering(self):
        self.data["clinical_notes"][0]["text"] = "  SYNTHETIC — diabetes education requested.\n"
        result = app.run_pipeline(self.data)
        citation = result["research"][0]["citation"]
        self.assertEqual(citation["start"], 2)
        result["research"][0]["citation"]["start"] = 0
        with self.assertRaises(app.ValidationError):
            app.validate(result, "final")

    def test_clinical_product_blocked(self):
        self.data["catalog"][0]["category"] = "treatment"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_decimal_lab_passage_is_preserved(self):
        self.data["catalog"][0]["tags"] = ["a1c"]
        self.data["preferences"]["interests"] = ["a1c"]
        result = app.run_pipeline(self.data)
        self.assertIn("7.2 percent", result["research"][0]["finding"])

    def test_output_review_requires_boolean(self):
        result = app.recommend(self.data)
        result["discovery"][0]["human_review_required"] = 1
        result["audit_trail"] = app.expected_audit(result)
        with self.assertRaises(app.ValidationError):
            app.research(result)

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        app.validate(payload["data"], "final")
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "missing.json")], [str(ROOT)]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_and_invalid_schema(self):
        for raw in ("{", "[]", '{"synthetic":true,"synthetic":true}', '{"value":NaN}',
                    json.dumps({"synthetic": False})):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(output):
                code = app.main(["virtual-input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
