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

    def test_integrated_example(self):
        output = app.run_pipeline(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["extraction"]), 3)
        self.assertEqual(output["extraction"][0]["fields"][0]["value"], "SYNTH-SERVICE-0001")

    def test_deduplication_and_alias_propagation(self):
        output = app.run_pipeline(self.data)
        self.assertEqual(output["feedback"]["duplicate_count"], 1)
        self.assertEqual(output["extraction"][0]["original_ids"], ["doc-1", "doc-2"])

    def test_theme_evidence_and_propagation(self):
        output = app.run_pipeline(self.data)
        sources = {s["id"]: s for s in output["feedback"]["sources"]}
        self.assertIn("Service delay", output["extraction"][0]["themes"])
        for theme in output["feedback"]["themes"]:
            for excerpt in theme["excerpts"]:
                self.assertEqual(excerpt["text"], sources[excerpt["source_id"]]["text"][
                    excerpt["start"]:excerpt["end"]])

    def test_extracted_spans_are_exact(self):
        output = app.run_pipeline(self.data)
        sources = {s["id"]: s["text"] for s in output["feedback"]["sources"]}
        for record in output["extraction"]:
            for field in record["fields"]:
                if field["span"]:
                    span = field["span"]
                    self.assertEqual(field["value"], sources[span["source_id"]][span["start"]:span["end"]])

    def test_pii_removed_everywhere(self):
        output = json.dumps(app.run_pipeline(self.data))
        for val in self.data["citizen_pii"][0].values():
            self.assertNotIn(val, output)
        self.assertIn("[REDACTED]", output)
        self.assertIn("protected_personal_details", output)

    def test_undeclared_pattern_pii(self):
        self.data["documents"][0]["content"]["Feedback"] = "Call 202-555-0199 or person@example.invalid, ID 123-45-6789."
        output = json.dumps(app.run_pipeline(self.data))
        for secret in ("202-555-0199", "person@example.invalid", "123-45-6789"):
            self.assertNotIn(secret, output)

    def test_missing_required_fields(self):
        benefit = app.run_pipeline(self.data)["extraction"][1]
        self.assertEqual(benefit["missing_fields"], ["application_date"])
        self.assertTrue(benefit["review_required"])
        self.assertFalse(app.run_pipeline(self.data)["extraction"][0]["review_required"])

    def test_ambiguous_policy_field(self):
        self.data["documents"][-1]["content"] += "\nPolicy title: Another invented rule"
        record = app.run_pipeline(self.data)["extraction"][-1]
        self.assertEqual(record["fields"][0]["reason"], "ambiguous")
        self.assertIsNone(record["fields"][0]["value"])

    def test_empty_field(self):
        self.data["documents"][0]["content"]["Service"] = ""
        record = app.run_pipeline(self.data)["extraction"][0]
        self.assertEqual(record["fields"][1]["reason"], "empty")
        self.assertTrue(record["review_required"])

    def test_invalid_inputs(self):
        bad_values = [None, [], {}, {"schema_version": "bogus"}]
        for key, val in [("synthetic", False), ("documents", []), ("fields", []),
                         ("citizen_pii", "not a list")]:
            item = copy.deepcopy(self.data)
            item[key] = val
            bad_values.append(item)
        for item in bad_values:
            with self.subTest(item=item), self.assertRaises(app.ValidationError):
                app.run_pipeline(item)

    def test_invalid_document_and_field_schemas(self):
        for mutate in (
            lambda d: d["documents"][1].update(id="doc-1"),
            lambda d: d["documents"][0].update(format="policy_text"),
            lambda d: d["fields"][0].update(required="yes"),
            lambda d: d["fields"][0].update(label=".*"),
            lambda d: d["documents"][0]["content"].update(Service="x\nPolicy title: injected"),
        ):
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_corrupted_handoff_is_rejected(self):
        feedback = app.analyze_feedback(self.data)
        feedback["feedback"]["themes"][0]["excerpts"][0]["text"] = "invented unsupported claim"
        with self.assertRaises(app.ValidationError):
            app.extract_fields(feedback)

    def test_extraction_output_validation(self):
        output = app.run_pipeline(self.data)
        output["extraction"][0]["fields"][0]["value"] = "unsupported value"
        with self.assertRaises(app.ValidationError):
            app.validate(output, "extraction")

    def test_determinism_and_input_immutability(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(original, self.data)

    def test_no_keyword_feedback(self):
        self.data["documents"][0]["content"]["Feedback"] = "Thank you."
        output = app.run_pipeline(self.data)
        self.assertIn("Other feedback", output["extraction"][0]["themes"])

    def test_plain_language_and_no_decision(self):
        output = app.run_pipeline(self.data)
        self.assertIn("No benefit or service decision is made.", output["notice"])
        self.assertIn("A person must review", output["notice"])
        self.assertNotIn("eligible", json.dumps(output).lower())

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], ["nonexistent-input.json"]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for content in ("{", '{"private":"must not echo"}', "null"):
            output = io.StringIO()
            with patch.object(Path, "open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                code = app.main(["in-memory.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
            self.assertNotIn("must not echo", output.getvalue())


if __name__ == "__main__":
    unittest.main()
