import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_review_support_and_traceable_gap(self):
        review = app.review_documents(self.data)
        supported, gap = review["findings"]
        self.assertEqual(supported["status"], "supported")
        self.assertEqual(supported["citations"][0]["locator"], "section 1")
        self.assertEqual(gap["missing_phrases"], ["5 business days"])
        self.assertIn("not certification", review["notice"])

    def test_grounded_answer_exact_quote(self):
        result = app.run_pipeline(self.data)
        answer = result["faq"]["answers"][0]
        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["answer"], self.data["evidence"][0]["text"])
        self.assertEqual(answer["citations"][0]["evidence_id"], "policy-1")

    def test_gap_propagates_to_abstention(self):
        answer = app.run_pipeline(self.data)["faq"]["answers"][1]
        self.assertEqual(answer["reason"], "review_gap")
        self.assertEqual(answer["gap_requirement_ids"], ["refunds"])
        self.assertIsNone(answer["answer"])
        self.assertEqual(answer["citations"], [])

    def test_no_relevant_evidence_abstains(self):
        answer = app.run_pipeline(self.data)["faq"]["answers"][2]
        self.assertEqual(answer["reason"], "no_relevant_evidence")

    def test_upstream_evidence_fix_changes_downstream_answer(self):
        self.data["evidence"][1]["text"] = "Synthetic refund processing takes 5 business days."
        result = app.run_pipeline(self.data)
        self.assertEqual(result["review"]["findings"][1]["status"], "supported")
        self.assertEqual(result["faq"]["answers"][1]["status"], "answered")

    def test_tampered_handoff_rejected(self):
        for mutation in ("status", "quote", "source"):
            with self.subTest(mutation=mutation):
                review = app.review_documents(self.data)
                if mutation == "status":
                    review["findings"][1]["status"] = "supported"
                else:
                    review["findings"][0]["citations"][0][mutation] = "fabricated"
                with self.assertRaises(app.ValidationError):
                    app.answer_questions(self.data, review)

    def test_tampered_final_answer_rejected(self):
        output = app.run_pipeline(self.data)
        output["faq"]["answers"][0]["answer"] = "Unsupported promise."
        with self.assertRaises(app.ValidationError):
            app.validate_output(self.data, output)

    def test_missing_evidence_yields_gap(self):
        self.data["requirements"][0]["evidence_ids"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["review"]["findings"][0]["missing_phrases"], ["returns", "30 days"])
        self.assertEqual(result["faq"]["answers"][0]["reason"], "review_gap")

    def test_unlinked_evidence_cannot_fill_gap(self):
        self.data["evidence"][0]["text"] += " Refund processing takes 5 business days."
        self.assertEqual(app.run_pipeline(self.data)["faq"]["answers"][1]["status"], "abstained")

    def test_case_insensitive_and_word_boundaries(self):
        self.data["evidence"][0]["text"] = "RETURNS within 30 DAYS."
        self.assertEqual(app.review_documents(self.data)["findings"][0]["status"], "supported")
        self.data["evidence"][0]["text"] = "Nonreturns within 130 days."
        self.assertEqual(app.review_documents(self.data)["findings"][0]["missing_phrases"],
                         ["returns", "30 days"])

    def test_empty_collections(self):
        data = {"schema_version": 1, "synthetic": True, "requirements": [],
                "evidence": [], "questions": []}
        output = app.run_pipeline(data)
        self.assertEqual(output["review"]["findings"], [])
        self.assertEqual(output["faq"]["answers"], [])

    def test_invalid_input_variants(self):
        variants = []
        for key, value in (("schema_version", True), ("synthetic", False),
                           ("requirements", {}), ("questions", None)):
            data = copy.deepcopy(self.data)
            data[key] = value
            variants.append(data)
        data = copy.deepcopy(self.data)
        data["requirements"][0]["evidence_ids"] = ["unknown"]
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["requirements"][0]["required_phrases"] = []
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["evidence"].append(copy.deepcopy(data["evidence"][0]))
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["questions"][0]["requirement_ids"] = ["unknown"]
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["questions"][0]["requirement_ids"] = []
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["unexpected"] = 1
        variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_multirequirement_question_abstains_on_any_gap(self):
        self.data["questions"][0]["requirement_ids"].append("refunds")
        answer = app.run_pipeline(self.data)["faq"]["answers"][0]
        self.assertEqual(answer["reason"], "review_gap")
        self.assertEqual(answer["citations"], [])

    def test_deterministic_without_input_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in ([], ["missing-input.json"], ["example_input.json", "extra"]):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", "implementation.py", *args],
                                        cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_malformed_duplicate_nonfinite_and_schema_errors(self):
        for content in ("{bad", '{"x":1,"x":2}', '{"x":NaN}', '[]',
                        '{"schema_version":1}'):
            with self.subTest(content=content), patch.object(Path, "read_text", return_value=content):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(buffer.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
