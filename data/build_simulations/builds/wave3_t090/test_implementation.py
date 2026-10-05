"""Synthetic fixtures only; tests create no files and call no providers."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text("utf-8"))

    def test_normal_coverage_and_traceable_gap(self):
        output = app.review(self.payload)
        self.assertEqual(output["summary"], {"requirements": 2, "supported": 1, "gaps": 1})
        gap = output["findings"][1]
        self.assertEqual(gap["supporting_document_ids"], ["DOC-01"])
        self.assertEqual(gap["gap"]["additional_sources_needed"], 1)
        self.assertEqual(gap["evidence"][1]["missing_terms"], ["deletion"])
        self.assertIn("not a determination", output["disclaimer"])

    def test_quotes_are_exact_source_spans(self):
        output = app.review(self.payload)
        docs = {doc["id"]: doc["content"] for doc in self.payload["documents"]}
        for finding in output["findings"]:
            for source in finding["evidence"]:
                for match in source["matches"]:
                    self.assertEqual(
                        docs[source["document_id"]][match["start"]:match["end"]],
                        match["quote"])
        self.assertEqual(output["findings"][0]["evidence"][0]["matches"][0]["quote"],
                         "Owner")

    def test_no_documents_produces_gaps(self):
        self.payload["documents"] = []
        output = app.review(self.payload)
        self.assertEqual(output["summary"]["gaps"], 2)
        self.assertEqual(output["findings"][0]["evidence"], [])

    def test_empty_document(self):
        self.payload["documents"][0]["content"] = ""
        self.assertEqual(app.review(self.payload)["summary"]["supported"], 0)

    def test_terms_must_cooccur_in_one_document(self):
        self.payload["documents"][0]["content"] = "Owner"
        self.payload["documents"][1]["content"] = "annual review"
        finding = app.review(self.payload)["findings"][0]
        self.assertEqual(finding["status"], "gap")
        self.assertEqual(finding["supporting_document_ids"], [])

    def test_literals_not_regex_and_unicode_offsets(self):
        self.payload["requirements"] = [{
            "id": "R", "text": "Synthetic literal check", "evidence_terms": ["[a+b]"]
        }]
        self.payload["documents"][0]["content"] = "😀 prefix [A+B] [a+b]"
        match = app.review(self.payload)["findings"][0]["evidence"][0]["matches"][0]
        self.assertEqual((match["start"], match["end"], match["quote"]),
                         (9, 14, "[A+B]"))

    def test_deterministic_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.payload)
        self.assertEqual(app.review(self.payload), app.review(self.payload))
        self.assertEqual(self.payload, before)

    def test_invalid_shapes_and_limits(self):
        variants = [
            ("schema_version", "2.0"), ("requirements", []),
            ("documents", {}), ("dataset", {"label": "x", "synthetic": 1}),
        ]
        for key, value in variants:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload[key] = value
                with self.assertRaises(app.ValidationError):
                    app.review(payload)
        for value in [True, 0, 101, 1.5, "2"]:
            with self.subTest(min_sources=value):
                self.payload["requirements"][0]["min_sources"] = value
                with self.assertRaises(app.ValidationError):
                    app.review(self.payload)

    def test_duplicate_ids_terms_and_unknown_fields(self):
        for kind in ("ids", "terms", "unknown"):
            with self.subTest(kind=kind):
                payload = copy.deepcopy(self.payload)
                if kind == "ids":
                    payload["documents"][1]["id"] = payload["documents"][0]["id"]
                elif kind == "terms":
                    payload["requirements"][0]["evidence_terms"] = ["owner", "OWNER"]
                else:
                    payload["requirements"][0]["certified"] = True
                with self.assertRaises(app.ValidationError):
                    app.review(payload)

    def test_bad_json_encoding_duplicates_and_oversize(self):
        for data in (b"{", b"\xff", b'{"a":1,"a":2}', b'{"x":NaN}',
                     b" " * (app.MAX_INPUT_BYTES + 1)):
            with self.subTest(prefix=data[:20]):
                with patch.object(Path, "open", return_value=io.BytesIO(data)):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_missing_file_usage_and_invalid_schema(self):
        for arguments in ([], ["missing-synthetic-input.json"], ["build_manifest.json"],
                          ["example_input.json", "extra"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py")] + arguments,
                    cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stderr, "")
                self.assertEqual(json.loads(result.stdout)["status"], "error")


if __name__ == "__main__":
    unittest.main()
