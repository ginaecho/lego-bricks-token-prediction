import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_rank_and_exact_citations(self):
        result = impl.research(self.request)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["findings"]), 3)
        self.assertEqual(result["findings"][0]["score"], 1.0)
        self.assertEqual(result["retrieval"]["passage_count"], 5)
        self.assertEqual(result["retrieval"]["matched_passage_count"], 3)
        sources = {source["id"]: source for source in self.request["sources"]}
        for finding in result["findings"]:
            citation = finding["citation"]
            self.assertEqual(sources[citation["source_id"]]["text"][citation["start"]:citation["end"]],
                             citation["quote"])
            self.assertEqual(finding["text"], citation["quote"])

    def test_tie_order_determinism_and_no_mutation(self):
        original = copy.deepcopy(self.request)
        result = impl.research(self.request)
        self.assertEqual(result, impl.research(self.request))
        self.assertEqual(self.request, original)
        self.assertEqual([f["citation"]["source_id"] for f in result["findings"]],
                         ["synthetic-report-a", "synthetic-report-a", "synthetic-report-b"])

    def test_unicode_crlf_and_trimmed_offsets(self):
        self.request["query"] = "CAFÉ"
        text = "\r\n  ☀ Café solar. \t\r\n \t\r\nCafé harbor."
        self.request["sources"] = [{"id": "s", "title": "Synthetic Unicode", "text": text}]
        result = impl.research(self.request)
        self.assertEqual(len(result["findings"]), 2)
        self.assertEqual(result["findings"][0]["citation"]["start"], 4)
        for finding in result["findings"]:
            cite = finding["citation"]
            self.assertEqual(text[cite["start"]:cite["end"]], cite["quote"])

    def test_no_matches_and_empty_sources_text(self):
        self.request["query"] = "unfindable"
        result = impl.research(self.request)
        self.assertEqual(result["findings"], [])
        self.assertTrue(result["retrieval"]["no_matches"])
        self.request["sources"][0]["text"] = " \r\n\t"
        self.request["sources"][1]["text"] = ""
        self.assertEqual(impl.research(self.request)["retrieval"]["passage_count"], 0)

    def test_top_k_default_and_repeated_query_terms(self):
        self.request["query"] = "solar SOLAR solar"
        self.request["top_k"] = 1
        self.assertEqual(len(impl.research(self.request)["findings"]), 1)
        self.assertEqual(impl.research(self.request)["findings"][0]["score"], 1.0)
        del self.request["top_k"]
        self.assertEqual(len(impl.research(self.request)["findings"]), 2)

    def test_invalid_shared_schema(self):
        for key, value in (("top_k", True), ("top_k", 0), ("top_k", 51),
                           ("top_k", 1.5), ("query", "!!!"), ("query", None),
                           ("sources", []), ("sources", [None]),
                           ("schema_version", "v2"), ("dataset_label", "")):
            with self.subTest(key=key, value=value):
                request = copy.deepcopy(self.request)
                request[key] = value
                with self.assertRaises(impl.ValidationError):
                    impl.research(request)
        for request in ([], {}, dict(self.request, unexpected=1)):
            with self.assertRaises(impl.ValidationError):
                impl.research(request)

    def test_invalid_source_duplicate_and_bounds(self):
        duplicate = copy.deepcopy(self.request)
        duplicate["sources"].append(copy.deepcopy(duplicate["sources"][0]))
        with self.assertRaises(impl.ValidationError):
            impl.research(duplicate)
        for key, value in (("id", ""), ("text", 12), ("text", "x" * 100001)):
            request = copy.deepcopy(self.request)
            request["sources"][0][key] = value
            with self.assertRaises(impl.ValidationError):
                impl.research(request)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        result = self.run_cli(str(ROOT / "example_input.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), impl.research(self.request))
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_errors(self):
        for args in ((), ("nonexistent-input.json",), ("a", "b"), ("implementation.py",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_loader_rejects_duplicate_nonstandard_encoding_and_oversize(self):
        fixtures = [b'{"query":"a","query":"b"}', b'{"top_k":NaN}', b'\xff',
                    b'{' , b"x" * (impl.MAX_INPUT_BYTES + 1)]
        for raw in fixtures:
            with self.subTest(raw_length=len(raw)):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                    with contextlib.redirect_stdout(output):
                        code = impl.main(["synthetic-memory-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
