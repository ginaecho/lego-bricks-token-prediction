"""All fixtures are synthetic. Tests never call networks or external providers."""

import copy
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

    def test_integrated_answer_and_source_spans(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["faq"]["status"], "answered")
        self.assertEqual(result["faq"]["citations"], ["synthetic-returns"])
        self.assertIn("Ada Example, for order 104", result["faq"]["answer"])
        for item in result["extraction"]["fields"].values():
            if item["source"]:
                span = item["source"]
                self.assertEqual(self.data["document"][span["start"]:span["end"]], span["text"])
        self.assertEqual(result["extraction"]["missing_fields"], ["email"])

    def test_changed_field_propagates(self):
        self.data["document"] = self.data["document"].replace("104", "205")
        result = app.run_pipeline(self.data)
        self.assertEqual(result["extraction"]["fields"]["order"]["value"], 205)
        self.assertIn("order 205", result["faq"]["answer"])
        self.assertNotIn("104", result["faq"]["answer"])

    def test_missing_required_abstains(self):
        self.data["document"] = self.data["document"].replace("Order: 104\n", "")
        result = app.run_pipeline(self.data)
        self.assertFalse(result["extraction"]["complete"])
        self.assertIn("order", result["extraction"]["missing_fields"])
        self.assertEqual(result["faq"]["reason"], "required_extraction_unavailable")
        self.assertIsNone(result["faq"]["answer"])

    def test_invalid_integer_abstains(self):
        self.data["document"] = self.data["document"].replace("Order: 104", "Order: one")
        result = app.run_pipeline(self.data)
        self.assertEqual(result["extraction"]["errors"], [{"field": "order", "reason": "invalid_integer"}])
        self.assertEqual(result["faq"]["status"], "abstained")

    def test_duplicate_label_not_arbitrarily_selected(self):
        self.data["document"] += "Order: 999\n"
        result = app.run_pipeline(self.data)
        self.assertEqual(result["extraction"]["errors"][0]["reason"], "duplicate_label")
        self.assertIsNone(result["extraction"]["fields"]["order"]["value"])

    def test_crlf_unicode_whitespace_spans(self):
        self.data["document"] = "Customer:  Zoë 測試  \r\nOrder: +00104 \r\nQuestion: return unopened bricks\r\n"
        result = app.run_pipeline(self.data)
        self.assertEqual(result["extraction"]["fields"]["customer"]["value"], "Zoë 測試")
        self.assertEqual(result["extraction"]["fields"]["order"]["value"], 104)
        self.assertIn("Zoë 測試", result["faq"]["answer"])

    def test_empty_document_reports_all_missing(self):
        self.data["document"] = ""
        result = app.run_pipeline(self.data)
        self.assertEqual(len(result["extraction"]["missing_fields"]), 4)
        self.assertEqual(result["faq"]["status"], "abstained")

    def test_blank_value_does_not_consume_next_line(self):
        self.data["document"] = self.data["document"].replace("Customer: Ada Example", "Customer:   ")
        result = app.run_pipeline(self.data)
        self.assertEqual(result["extraction"]["fields"]["customer"]["state"], "missing")
        self.assertEqual(result["extraction"]["fields"]["order"]["value"], 104)

    def test_no_evidence_and_empty_kb(self):
        for kb in ([], [{"id": "synthetic-x", "question": "Account password recovery",
                         "answer": "Synthetic account instructions.", "required_fields": []}]):
            with self.subTest(kb=kb):
                self.data["knowledge_base"] = kb
                self.assertEqual(app.run_pipeline(self.data)["faq"]["reason"], "insufficient_evidence")

    def test_tied_retrieval_abstains(self):
        duplicate = copy.deepcopy(self.data["knowledge_base"][0])
        duplicate["id"] = "synthetic-conflict"
        duplicate["answer"] = "A conflicting synthetic policy."
        self.data["knowledge_base"].append(duplicate)
        result = app.run_pipeline(self.data)
        self.assertEqual(result["faq"]["reason"], "ambiguous_evidence")
        self.assertEqual(result["faq"]["citations"], [])

    def test_optional_field_needed_by_answer(self):
        self.data["knowledge_base"][0]["answer"] = "Send to {email}."
        result = app.run_pipeline(self.data)
        self.assertTrue(result["extraction"]["complete"])
        self.assertEqual(result["faq"]["reason"], "answer_fields_unavailable")

    def test_optional_question_missing(self):
        self.data["schema"]["fields"][2]["required"] = False
        self.data["document"] = "Customer: Ada Example\nOrder: 104\n"
        self.assertEqual(app.run_pipeline(self.data)["faq"]["reason"], "question_unavailable")

    def test_handoff_rejects_corrupted_extraction(self):
        extraction = app.extract(self.data)
        extraction["fields"]["order"]["value"] = 999
        with self.assertRaises(app.ValidationError):
            app.answer_faq(self.data, extraction)

    def test_answer_validator_rejects_invention(self):
        result = app.run_pipeline(self.data)
        result["faq"]["answer"] = "Invented refund promise."
        with self.assertRaises(app.ValidationError):
            app.validate_faq(self.data, result["extraction"], result["faq"])

    def test_schema_rejections(self):
        cases = [
            lambda d: d.update(version=True),
            lambda d: d.update(min_overlap=False),
            lambda d: d.update(document=[]),
            lambda d: d.update(extra="unexpected"),
            lambda d: d["schema"].update(question_field="unknown"),
            lambda d: d["schema"]["fields"].append(copy.deepcopy(d["schema"]["fields"][0])),
            lambda d: d["schema"]["fields"][0].update(required="yes"),
            lambda d: d["knowledge_base"][0].update(answer="{customer.__class__}"),
            lambda d: d["knowledge_base"][0].update(answer="{unknown}"),
            lambda d: d["knowledge_base"][0].update(answer="{customer!r}"),
            lambda d: d["knowledge_base"][0].update(answer="{"),
        ]
        for change in cases:
            data = copy.deepcopy(self.data)
            change(data)
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_deterministic(self):
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(copy.deepcopy(self.data)))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8")

    def test_cli_success(self):
        process = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["faq"]["status"], "answered")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ((), (str(ROOT / "nonexistent.json"),), ("a", "b"),
                     (str(ROOT / "implementation.py"),), (str(ROOT / "build_manifest.json"),)):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"version": 1, "version": 1}', object_pairs_hook=app.unique_object)


if __name__ == "__main__":
    unittest.main()
