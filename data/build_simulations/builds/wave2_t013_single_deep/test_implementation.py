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


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def finding(self, result, claim_id):
        return next(item for item in result["findings"] if item["claim_id"] == claim_id)

    def test_multidocument_synthesis(self):
        result = app.synthesize(self.data)
        self.assertEqual(result["summary"], {"documents": 3, "claims": 3,
                                            "disagreements": 1, "open_questions": 3})
        self.assertEqual(self.finding(result, "sales")["status"], "contested")
        self.assertEqual(self.finding(result, "returns")["status"], "supported")
        self.assertEqual(self.finding(result, "retention")["status"], "insufficient")
        self.assertEqual(result["disagreements"][0]["supporting_documents"], ["pilot-a"])
        self.assertEqual(result["disagreements"][0]["opposing_documents"], ["pilot-b"])
        self.assertEqual(self.finding(result, "returns")["contexts"][0]["quality_weight_totals"]["support"], 5)

    def test_duplicate_citations_do_not_inflate_weights(self):
        claim = self.data["claims"][0]
        claim["evidence"] += [copy.deepcopy(claim["evidence"][0])] * 3
        result = self.finding(app.synthesize(self.data), "sales")
        self.assertEqual(len(result["citations"]), 3)
        urban = next(item for item in result["contexts"] if item["context"] == "urban")
        self.assertEqual(urban["source_counts"]["support"], 1)
        self.assertEqual(urban["quality_weight_totals"]["support"], 2)

    def test_multiple_quotes_one_source_count_once(self):
        self.data["claims"][0]["evidence"].append(
            {"document_id": "pilot-a", "quote": "The pilot lasted two weeks.",
             "stance": "support", "quality": "high", "context": "urban"})
        finding = self.finding(app.synthesize(self.data), "sales")
        urban = next(item for item in finding["contexts"] if item["context"] == "urban")
        self.assertEqual(urban["source_counts"]["support"], 1)
        self.assertEqual(urban["quality_weight_totals"]["support"], 3)

    def test_context_difference_is_not_direct_disagreement(self):
        self.data["claims"][0]["evidence"].pop(1)
        result = app.synthesize(self.data)
        self.assertEqual(self.finding(result, "sales")["status"], "context_dependent")
        self.assertEqual(result["disagreements"], [])

    def test_no_documents_or_evidence_is_valid_gap(self):
        self.data["documents"] = []
        for claim in self.data["claims"]:
            claim["evidence"] = []
        result = app.synthesize(self.data)
        self.assertTrue(all(item["status"] == "insufficient" for item in result["findings"]))
        self.assertEqual(len(result["unresolved_questions"]), 4)

    def test_opposition_only(self):
        self.data["claims"][0]["evidence"].pop(0)
        self.assertEqual(self.finding(app.synthesize(self.data), "sales")["status"], "opposed")

    def test_same_document_internal_disagreement(self):
        opposing = self.data["claims"][0]["evidence"][1]
        opposing["document_id"] = "pilot-a"
        self.data["documents"][0]["text"] += " " + opposing["quote"]
        result = app.synthesize(self.data)["disagreements"][0]
        self.assertEqual(result["supporting_documents"], result["opposing_documents"])

    def test_reordering_is_deterministic_and_input_is_unchanged(self):
        original = copy.deepcopy(self.data)
        expected = app.synthesize(self.data)
        self.assertEqual(self.data, original)
        self.data["documents"].reverse()
        self.data["claims"].reverse()
        for claim in self.data["claims"]:
            claim["evidence"].reverse()
        self.assertEqual(app.synthesize(self.data), expected)

    def test_invalid_inputs(self):
        cases = [
            lambda d: d.update(synthetic=False),
            lambda d: d.update(schema_version=True),
            lambda d: d.update(extra=1),
            lambda d: d.update(research_question=" "),
            lambda d: d.update(documents={}),
            lambda d: d.update(claims=[]),
            lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
            lambda d: d["claims"].append(copy.deepcopy(d["claims"][0])),
            lambda d: d["claims"][0]["evidence"][0].update(document_id="missing"),
            lambda d: d["claims"][0]["evidence"][0].update(quote="fabricated quote"),
            lambda d: d["claims"][0]["evidence"][0].update(stance=[]),
            lambda d: d["claims"][0]["evidence"][0].update(quality=3),
            lambda d: d["questions"][0].update(claim_ids=["missing"]),
        ]
        for mutate in cases:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.synthesize(data)

    def test_shared_output_validation(self):
        result = app.synthesize(self.data)
        result["summary"]["claims"] = 99
        with self.assertRaises(app.ValidationError):
            app.validate(result, "output")

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), app.synthesize(self.data))
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "does-not-exist.json")], ["a", "b"]]:
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                     capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_invalid_schema(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "build_manifest.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_parser_and_file_errors_emit_json(self):
        payloads = [b"{", b"\xff", b'{"synthetic":true,"synthetic":false}',
                    b'{"x":NaN}', b"x" * (app.MAX_FILE_BYTES + 1)]
        for payload in payloads:
            stream = io.StringIO()
            with self.subTest(payload=payload[:50]), patch.object(Path, "open", return_value=io.BytesIO(payload)):
                with redirect_stdout(stream):
                    code = app.main(["synthetic-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_explicit_questions_stay_open(self):
        self.data["questions"][0]["claim_ids"] = ["returns"]
        result = app.synthesize(self.data)
        question = next(item for item in result["unresolved_questions"] if item["id"] == "requested:confounders")
        self.assertEqual(question["claim_ids"], ["returns"])

    def test_maximum_statement_can_generate_gap_question(self):
        self.data["claims"][0]["statement"] = "x" * 10000
        result = app.synthesize(self.data)
        question = next(item for item in result["unresolved_questions"] if item["id"] == "generated:sales")
        self.assertTrue(question["question"].endswith("x" * 10000))


if __name__ == "__main__":
    unittest.main()
