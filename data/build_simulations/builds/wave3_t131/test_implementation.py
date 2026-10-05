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


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def rejected(self, data=None):
        with self.assertRaises(app.ValidationError):
            app.run(self.data if data is None else data)

    def test_multi_document_conflict_and_uncertainty(self):
        result = app.run(self.data)
        first, second = result["synthesis"]["questions"]
        self.assertEqual(first["finding"], "conflicting_evidence")
        self.assertEqual(first["source_count"], 2)
        self.assertTrue(first["disagreement"])
        self.assertEqual(second["finding"], "uncertain_evidence")
        self.assertEqual(len(second["unresolved_questions"]), 2)

    def test_deterministic_and_immutable_with_complete_audit(self):
        original = copy.deepcopy(self.data)
        first = app.run(self.data)
        self.assertEqual(first, app.run(self.data))
        self.assertEqual(self.data, original)
        self.assertEqual(first["records"], original["records"])
        self.assertEqual(len(first["audit_trail"]), len(first["records"]) + 1)
        self.assertEqual([e["sequence"] for e in first["audit_trail"]], [1, 2, 3, 4])
        first["records"][0]["resource"]["labs"][0]["value"] = 8
        self.assertEqual(self.data, original)
        with self.assertRaises(app.ValidationError):
            app.validate_document(first, output=True)

    def test_missing_evidence_is_unresolved(self):
        self.data["research_questions"].append("Is the supporting document available?")
        question = app.run(self.data)["synthesis"]["questions"][-1]
        self.assertEqual(question["finding"], "no_evidence")
        self.assertEqual(question["source_count"], 0)
        self.assertTrue(question["unresolved_questions"])

    def test_agreement_is_not_a_decision(self):
        evidence = self.data["records"][2]["evidence"][0]
        evidence["stance"] = "support"
        evidence["quote"] = "The requested lab documentation is attached."
        self.data["records"][2]["resource"]["request_text"] = (
            "SYNTHETIC FIXTURE: " + evidence["quote"]
        )
        question = app.run(self.data)["synthesis"]["questions"][0]
        self.assertEqual(question["finding"], "supporting_evidence_recorded")
        self.assertFalse(question["disagreement"])
        result = app.run(self.data)
        self.assertIsNone(result["synthesis"]["clinical_decision"])
        self.assertFalse(result["autonomous_clinical_decisions"])
        self.assertTrue(result["human_review_required"])

    def test_single_source_not_overcounted(self):
        self.data["records"].pop()
        question = app.run(self.data)["synthesis"]["questions"][0]
        self.assertEqual(question["source_count"], 1)
        self.assertTrue(question["unresolved_questions"])

    def test_direct_identifier_fields_are_rejected(self):
        for field in ("name", "birthDate", "identifier", "address", "telecom"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["records"][0]["resource"][field] = "prohibited synthetic placeholder"
                self.rejected(data)

    def test_free_text_identifier_patterns_rejected(self):
        for value in ("DOB: 2000-01-01", "MRN: 12345678", "person@example.invalid",
                      "555-123-4567", "SSN 000-00-0000", "Patient name: Fiction"):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["records"][1]["resource"]["text"] += " " + value
                self.rejected(data)

    def test_human_review_and_synthetic_flags_required(self):
        for field in ("human_review_required", "synthetic"):
            data = copy.deepcopy(self.data)
            data[field] = False
            self.rejected(data)
        self.data["records"][0]["resource"]["synthetic"] = False
        self.rejected()

    def test_authorization_cannot_be_finalized(self):
        self.data["records"][2]["resource"]["status"] = "approved"
        self.rejected()

    def test_fabricated_citation_and_unknown_question_rejected(self):
        self.data["records"][1]["evidence"][0]["quote"] = "Not in the clinical note"
        self.rejected()
        self.setUp()
        self.data["records"][1]["evidence"][0]["question"] = "Unknown research question?"
        self.rejected()

    def test_duplicate_records_and_evidence_rejected(self):
        self.data["records"].append(copy.deepcopy(self.data["records"][0]))
        self.rejected()
        self.setUp()
        evidence = self.data["records"][1]["evidence"]
        evidence.append(copy.deepcopy(evidence[0]))
        self.rejected()

    def test_cross_patient_reference_rejected(self):
        self.data["records"][1]["resource"]["subject"]["reference"] = "Patient/synthetic-patient-002"
        self.rejected()

    def test_numeric_and_collection_boundaries(self):
        for value in (True, -1, float("nan"), float("inf"), "7.2", 10 ** 1000):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["records"][0]["resource"]["labs"][0]["value"] = value
                self.rejected(data)
        for field in ("records", "research_questions"):
            data = copy.deepcopy(self.data)
            data[field] = []
            self.rejected(data)

    def test_patient_only_input(self):
        self.data["records"] = self.data["records"][:1]
        result = app.run(self.data)
        self.assertTrue(all(q["finding"] == "no_evidence"
                            for q in result["synthesis"]["questions"]))

    def test_output_audit_and_decision_tampering_rejected(self):
        for field, value in (("audit_trail", []), ("autonomous_clinical_decisions", True)):
            result = app.run(self.data)
            result[field] = value
            with self.assertRaises(app.ValidationError):
                app.validate_document(result, output=True)
        result = app.run(self.data)
        result["synthesis"]["clinical_decision"] = "approve"
        with self.assertRaises(app.ValidationError):
            app.validate_document(result, output=True)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "nonexistent.json")]):
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_bad_json_encoding_and_schema_without_disk_writes(self):
        cases = [b"{", b"\xff", b'{"synthetic":true,"synthetic":true}',
                 b'{"n":NaN}', b"[]", b"x" * (app.MAX_BYTES + 1)]
        invalid = copy.deepcopy(self.data)
        invalid["records"][0]["resource"]["labs"][0]["unit"] = {}
        cases.append(json.dumps(invalid).encode())
        for raw in cases:
            with self.subTest(raw_size=len(raw)):
                stdout = io.StringIO()
                with patch("pathlib.Path.open", mock_open(read_data=raw)):
                    with contextlib.redirect_stdout(stdout):
                        code = app.main(["input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
