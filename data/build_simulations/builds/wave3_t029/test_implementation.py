import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_documents_extract_all_entities(self):
        result = app.document_stage(self.data)
        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(result["records"][1]["facts"][1]["value"], 7.2)
        self.assertIsNone(result["research"])

    def test_cross_stage_propagation(self):
        docs = app.document_stage(self.data)
        result = app.research_stage(docs, self.data["research"])
        for before, after in zip(docs["records"], result["records"]):
            self.assertEqual(before["facts"], after["facts"])
            self.assertEqual(before["patient_ref"], after["patient_ref"])
            self.assertEqual(after["audit"][1]["before"], app.record_state(before))
        self.assertEqual(len(result["records"][2]["evidence"]), 2)
        self.assertEqual(docs["records"][2]["evidence"], [])

    def test_research_conflicts_and_gaps(self):
        result = app.run_pipeline(self.data)
        findings = {f["code"]: f["assessment"] for f in result["research"]["findings"]}
        self.assertEqual(findings, {"E11.9": "support_only", "LAB-A1C": "conflicting",
                                    "4548-4": "no_evidence"})

    def test_no_autonomous_clinical_decision(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["clinical_decision"], "not_performed")
        self.assertTrue(result["human_review_required"])
        self.assertTrue(all(r["human_review_required"] for r in result["records"]))
        self.assertEqual(result["research"]["disposition"], "human_review_only")

    def test_identifiers_excluded_including_audit(self):
        output = json.dumps(app.run_pipeline(self.data))
        self.assertNotIn("SYN-P1", output)
        self.assertNotIn("SYN-N1", output)
        self.assertNotIn("FICTITIOUS teaching", output)

    def test_phi_fields_rejected(self):
        for key in ("name", "birthDate", "identifier", "address", "telecom"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["resources"][0][key] = "REDACTED-SYNTHETIC"
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_unstructured_note_rejected(self):
        self.data["resources"][1]["text"] = "Patient name: SYNTHETIC PERSON"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_non_synthetic_rejected(self):
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_broken_reference_rejected(self):
        self.data["resources"][1]["subject"]["reference"] = "Patient/SYN-P2"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_duplicate_ids_rejected(self):
        self.data["resources"].append(copy.deepcopy(self.data["resources"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_note_deduplication_and_whitespace(self):
        self.assertEqual(app.parse_note("  Diagnosis: E11.9\n\nDiagnosis: E11.9 "),
                         [app.fact("diagnosis", "E11.9")])

    def test_invalid_lab_values(self):
        for value in ("nan", "inf", "-inf", "abc"):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.parse_note("Lab: 4548-4 " + value + " %")

    def test_no_sources_is_explicit_gap(self):
        self.data["research"]["sources"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(all(f["assessment"] == "no_evidence"
                            for f in result["research"]["findings"]))

    def test_focus_does_not_invent_document_facts(self):
        self.data["research"]["focus_codes"] = ["UNKNOWN"]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["findings"][0]["assessment"], "not_in_documents")
        self.assertTrue(all(not r["evidence"] for r in result["records"]))

    def test_quote_grounding(self):
        self.data["research"]["sources"][0]["claims"][0]["quote"] = "Invented E11.9 evidence"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_source_text_not_copied(self):
        self.data["research"]["sources"][0]["text"] += " SYN-P1"
        self.data["research"]["question"] += " SYN-P1"
        self.assertNotIn("SYN-P1", json.dumps(app.run_pipeline(self.data)))

    def test_mutation_requires_audit(self):
        docs = app.document_stage(self.data)
        docs["records"][1]["facts"][1]["value"] = 9.0
        with self.assertRaises(app.ValidationError):
            app.research_stage(docs, self.data["research"])

    def test_review_flag_cannot_be_disabled(self):
        docs = app.document_stage(self.data)
        docs["human_review_required"] = False
        with self.assertRaises(app.ValidationError):
            app.research_stage(docs, self.data["research"])

    def test_deterministic_and_non_mutating(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(before, self.data)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_schema_file(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = app.main([str(ROOT / "build_manifest.json")])
        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_duplicate_json_keys_and_nonfinite(self):
        for payload in ('{"synthetic":true,"synthetic":false}', '{"value":NaN}'):
            with self.assertRaises(app.ValidationError):
                json.loads(payload, parse_constant=app.reject_constant,
                           object_pairs_hook=app.unique_object)


if __name__ == "__main__":
    unittest.main()
