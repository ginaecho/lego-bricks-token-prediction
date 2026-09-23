"""All fixtures are synthetic; tests perform no network or temporary-file I/O."""

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

    def test_extraction_exact_trimmed_source_spans(self):
        extraction = app.extract(self.data)
        field = extraction["records"][0]["fields"]["feedback"]
        self.assertEqual(field["value"], "Delivery was slow, but support was helpful!")
        for record, document in zip(extraction["records"], self.data["documents"]):
            for item in record["fields"].values():
                if item:
                    source = item["source"]
                    self.assertEqual(source["document_id"], document["id"])
                    self.assertEqual(document["text"][source["start"]:source["end"]], item["value"])

    def test_missing_required_and_optional_fields(self):
        records = app.extract(self.data)["records"]
        self.assertTrue(records[1]["complete"])
        self.assertEqual(records[1]["missing_fields"], [{"field": "order", "required": False}])
        self.assertFalse(records[3]["complete"])
        self.assertIn({"field": "feedback", "required": True}, records[3]["missing_fields"])
        self.assertIsNone(records[3]["fields"]["feedback"])

    def test_dedup_and_cross_stage_propagation(self):
        result = app.run_pipeline(self.data)["feedback"]
        self.assertEqual(result["counts"], {
            "input_documents": 4, "eligible_documents": 3, "unique_feedback": 2,
            "duplicate_documents": 1, "skipped_documents": 1,
        })
        self.assertEqual(result["groups"][0]["document_ids"], ["synthetic-001", "synthetic-002"])
        self.assertEqual(result["skipped_documents"], [
            {"document_id": "synthetic-004", "reason": "missing_required_fields"}])
        self.assertEqual(result["unclassified_group_ids"], ["feedback-2"])

    def test_theme_support_is_traceable_and_counts_unique_groups(self):
        output = app.run_pipeline(self.data)
        documents = {item["id"]: item["text"] for item in self.data["documents"]}
        shipping = output["feedback"]["themes"][0]
        self.assertEqual(shipping["unique_feedback_count"], 1)
        self.assertEqual(len(shipping["supporting_excerpts"]), 4)
        for theme in output["feedback"]["themes"]:
            for support in theme["supporting_excerpts"]:
                span = support["source"]
                self.assertEqual(documents[support["document_id"]][span["start"]:span["end"]],
                                 support["excerpt"])
        self.assertEqual(output["feedback"]["themes"][2]["unique_feedback_count"], 0)

    def test_incomplete_record_does_not_leak_feedback(self):
        self.data["documents"] = [{"id": "synthetic-a", "text": "Feedback: delivery is slow"}]
        result = app.run_pipeline(self.data)["feedback"]
        self.assertEqual(result["groups"], [])
        self.assertEqual(result["themes"][0]["supporting_excerpts"], [])

    def test_empty_documents(self):
        self.data["documents"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["extraction"]["records"], [])
        self.assertEqual(result["feedback"]["counts"]["unique_feedback"], 0)

    def test_blank_and_punctuation_feedback(self):
        self.data["documents"] = [
            {"id": "synthetic-a", "text": "Customer: X\nFeedback:    "},
            {"id": "synthetic-b", "text": "Customer: Y\nFeedback: !!!"},
        ]
        result = app.run_pipeline(self.data)["feedback"]
        self.assertEqual([item["reason"] for item in result["skipped_documents"]],
                         ["missing_required_fields", "no_feedback_words"])

    def test_optional_feedback_missing(self):
        self.data["schema"]["fields"][1]["required"] = False
        result = app.run_pipeline(self.data)["feedback"]
        self.assertEqual(result["skipped_documents"][0]["reason"], "missing_feedback")

    def test_first_match_and_unmatched_optional_capture(self):
        self.data["documents"] = [
            {"id": "synthetic-a", "text": "Customer: X\nFeedback: first\nFeedback: second"}]
        self.assertEqual(app.extract(self.data)["records"][0]["fields"]["feedback"]["value"], "first")
        self.data["schema"]["fields"][1]["pattern"] = r"(?P<value>NO_MATCH)?Customer"
        self.assertIsNone(app.extract(self.data)["records"][0]["fields"]["feedback"])

    def test_unicode_spans_and_word_boundaries(self):
        self.data["documents"] = [
            {"id": "synthetic-a", "text": "Customer: Zoë 🧱\nFeedback: slowpoke DELIVERY café"}]
        self.data["feedback_config"]["themes"][0]["keywords"] = ["slow", "delivery", "café"]
        output = app.run_pipeline(self.data)
        supports = output["feedback"]["themes"][0]["supporting_excerpts"]
        self.assertEqual([item["excerpt"] for item in supports], ["DELIVERY", "café"])
        for item in supports:
            self.assertEqual(
                self.data["documents"][0]["text"][item["source"]["start"]:item["source"]["end"]],
                item["excerpt"])

    def test_rejects_tampered_handoff_and_output(self):
        extraction = app.extract(self.data)
        extraction["records"][0]["fields"]["feedback"]["source"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.analyze_feedback(self.data, extraction)
        output = app.run_pipeline(self.data)
        output["feedback"]["themes"][0]["unique_feedback_count"] = 999
        with self.assertRaises(app.ValidationError):
            app.validate("output", output, input_data=self.data)

    def test_invalid_inputs(self):
        cases = []
        for key, value in [("schema_version", True), ("synthetic", "yes"), ("documents", None)]:
            data = copy.deepcopy(self.data)
            data[key] = value
            cases.append(data)
        data = copy.deepcopy(self.data)
        data["documents"].append(copy.deepcopy(data["documents"][0]))
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["feedback_config"]["text_field"] = "unknown"
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["schema"]["fields"][0]["pattern"] = "("
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["schema"]["fields"][0]["pattern"] = "(no_named_group)"
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["extra"] = "not accepted"
        cases.append(data)
        cases.extend([None, [], {"schema_version": 1}])
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_repeat_runs_deterministic_and_input_unmodified(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_oversized_regex_repeat_is_validation_error(self):
        self.data["schema"]["fields"][0]["pattern"] = r"(?P<value>x{999999999999999999999})"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data=json.dumps(self.data))), \
                contextlib.redirect_stdout(output):
            self.assertEqual(app.main(["synthetic-input.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_success_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout), app.run_pipeline(self.data))
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file_and_argument(self):
        for arguments in [[], [str(ROOT / "nonexistent-synthetic.json")]]:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_schema_encoding_and_duplicate_keys(self):
        for raw in ['{', '{}', '{"schema_version":1,"schema_version":1}']:
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(output):
                code = app.main(["synthetic-input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("invalid encoding")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(app.main(["synthetic-input.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
