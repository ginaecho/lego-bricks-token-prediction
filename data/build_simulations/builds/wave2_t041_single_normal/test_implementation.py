import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
from contextlib import redirect_stdout
from io import StringIO

from implementation import ValidationError, main, research, validate


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "query": "solar storage",
            "sources": [
                {"id": "synthetic-a", "title": "Synthetic lab notes",
                 "text": "  Solar panels work.  Solar storage lasts overnight.\nWind differs."},
                {"id": "synthetic-b", "title": "Synthetic backup notes",
                 "text": "Storage costs vary."},
            ],
        }

    def test_rank_and_exact_citations(self):
        result = research(self.payload)
        self.assertEqual(result["findings"][0]["text"], "Solar storage lasts overnight.")
        self.assertEqual(result["findings"][0]["score"], 1)
        self.assertEqual(result["summary"]["matching_passages"], 3)
        by_id = {s["id"]: s for s in self.payload["sources"]}
        for finding in result["findings"]:
            c = finding["citation"]
            self.assertEqual(by_id[c["source_id"]]["text"][c["start"]:c["end"]],
                             finding["text"])

    def test_limit_and_stable_ties(self):
        self.payload["query"] = "storage"
        self.payload["max_findings"] = 1
        result = research(self.payload)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["citation"]["source_id"], "synthetic-a")
        self.assertEqual(result, research(copy.deepcopy(self.payload)))

    def test_no_match_and_empty_sources(self):
        for query, sources in [("unicorn", self.payload["sources"]),
                               ("!!!", self.payload["sources"]), ("solar", [])]:
            result = research({"query": query, "sources": sources})
            self.assertEqual(result["findings"], [])
            self.assertTrue(result["summary"]["no_matches"])

    def test_unicode_and_whitespace_offsets(self):
        text = "\r\n  Café ☀ solar!\r\n\tOther sentence.  "
        result = research({"query": "CAFÉ", "sources": [
            {"id": "s", "title": "Synthetic Unicode", "text": text}]})
        finding = result["findings"][0]
        self.assertEqual(finding["text"], "Café ☀ solar!")
        self.assertEqual(finding["citation"]["start"], 4)
        self.assertEqual(text[4:finding["citation"]["end"]], finding["text"])

    def test_blank_source_and_whole_word_matching(self):
        result = research({"query": "solar", "sources": [
            {"id": "blank", "title": "Synthetic blank", "text": " \n "},
            {"id": "word", "title": "Synthetic word", "text": "Solarium."}]})
        self.assertEqual(result["findings"], [])

    def test_invalid_requests(self):
        cases = [None, [], {}, {"query": "", "sources": []},
                 {"query": "x", "sources": {}},
                 {**self.payload, "max_findings": True},
                 {**self.payload, "max_findings": 0},
                 {**self.payload, "max_findings": 51},
                 {**self.payload, "extra": 1},
                 {**self.payload, "sources": [self.payload["sources"][0]] * 2},
                 {"query": "x", "sources": [{"id": "s", "title": "t", "text": 1}]}]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                research(payload)

    def test_evidence_validator_rejects_inexact_quote(self):
        finding = research(self.payload)["findings"][0]
        finding["text"] += " fabricated"
        with self.assertRaises(ValidationError):
            validate((finding, self.payload["sources"][0]), "evidence")

    def test_cli_example(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")], capture_output=True, text=True,
            cwd=ROOT, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "missing-input.json")]]:
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, cwd=ROOT, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_malformed_json_and_schema(self):
        for text in ['{', '{"query":"x","query":"y","sources":[]}',
                     '{"query":"x","sources":[],"max_findings":NaN}', '[]']:
            with self.subTest(text=text):
                output = StringIO()
                with patch("pathlib.Path.open", mock_open(read_data=text)):
                    with redirect_stdout(output):
                        status = main(["synthetic-invalid.json"])
                self.assertEqual(status, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_input_not_mutated(self):
        original = copy.deepcopy(self.payload)
        research(self.payload)
        self.assertEqual(self.payload, original)


if __name__ == "__main__":
    unittest.main()
