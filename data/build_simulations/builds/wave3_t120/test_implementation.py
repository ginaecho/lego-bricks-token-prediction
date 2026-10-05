import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_provenance_and_redirect(self):
        result = app.research(self.request)
        self.assertEqual("ok", result["status"])
        self.assertEqual(2, len(result["findings"]))
        source = result["sources"][0]
        self.assertEqual(["https://research.example/latest",
                          "https://research.example/report"], source["redirect_chain"])
        self.assertEqual(hashlib.sha256(self.request["fixtures"][1]["body"].encode()).hexdigest(),
                         source["content_sha256"])
        self.assertNotIn("secretly", source["text"])
        for finding in result["findings"]:
            cited = next(s for s in result["sources"] if s["source_id"] == finding["source_id"])
            self.assertEqual(finding["quote"], cited["text"][finding["start"]:finding["end"]])

    def test_deterministic_and_input_not_mutated(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(app.research(self.request), app.research(self.request))
        self.assertEqual(original, self.request)

    def test_no_match_and_empty_content(self):
        self.request["query"] = "unobtainium"
        self.assertEqual([], app.research(self.request)["findings"])
        self.request["fixtures"][1]["body"] = ""
        result = app.research(self.request)
        self.assertEqual("", result["sources"][0]["text"])
        self.assertTrue(result["warnings"])

    def test_exact_allowlist_and_bad_url_forms(self):
        for url in ("http://research.example/a", "https://research.example.evil/a",
                    "https://sub.research.example/a", "https://user@research.example/a",
                    "https://research.example:444/a", "https://research.example/a#x",
                    "https://research.example\\@evil.example/a",
                    "https://research.example/\npath"):
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                request = copy.deepcopy(self.request)
                request["urls"] = [url]
                app.research(request)

    def test_canonical_duplicates(self):
        self.request["urls"].append("https://RESEARCH.example:443/latest")
        self.assertEqual(2, len(app.research(self.request)["sources"]))

    def test_redirect_target_blocked(self):
        self.request["fixtures"][0]["redirect_to"] = "https://evil.example/"
        with self.assertRaisesRegex(app.ValidationError, "not allowlisted"):
            app.research(self.request)

    def test_redirect_cycle(self):
        self.request["fixtures"][0]["redirect_to"] = "https://research.example/latest"
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            app.research(self.request)

    def test_redirect_limit(self):
        self.request["urls"] = ["https://research.example/0"]
        self.request["fixtures"] = [
            dict(url=f"https://research.example/{i}",
                 retrieved_at="2026-09-24T09:00:00Z",
                 redirect_to=f"https://research.example/{i + 1}")
            for i in range(6)]
        with self.assertRaisesRegex(app.ValidationError, "limit"):
            app.research(self.request)

    def test_missing_fixture(self):
        self.request["urls"] = ["https://research.example/missing"]
        with self.assertRaisesRegex(app.ValidationError, "No synthetic fixture"):
            app.research(self.request)

    def test_invalid_schema(self):
        for key, value in (("synthetic", False), ("schema_version", True),
                           ("query", " "), ("urls", []), ("max_findings", True),
                           ("max_findings", 0), ("allowed_hosts", ["*.example"]),
                           ("unknown", 1), ("fixtures", {})):
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                request = copy.deepcopy(self.request)
                request[key] = value
                app.research(request)

    def test_invalid_fixture_fields(self):
        for key, value in (("retrieved_at", "2026-09-24"),
                           ("content_type", "application/pdf"), ("body", 3)):
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                request = copy.deepcopy(self.request)
                request["fixtures"][1][key] = value
                app.research(request)

    def test_duplicate_fixture_rejected(self):
        self.request["fixtures"].append(copy.deepcopy(self.request["fixtures"][1]))
        with self.assertRaisesRegex(app.ValidationError, "Duplicate"):
            app.research(self.request)

    def test_truncation(self):
        self.request["max_findings"] = 1
        result = app.research(self.request)
        self.assertEqual(1, len(result["findings"]))
        self.assertIn("truncated", result["warnings"][0])

    def test_unpunctuated_lines_preserve_findings(self):
        self.request["fixtures"][1]["body"] = "<p>Recycled bricks</p><p>Other material</p>"
        result = app.research(self.request)
        self.assertEqual("Recycled bricks", result["findings"][0]["quote"])

    def test_html_entities_and_hidden_content(self):
        text = app.extract_text("<style>bricks</style><p>Bricks &amp; tiles.</p>"
                                "<template><div>secret</div></template>"
                                "<p>Recycled <b>bricks</b>.</p>", "text/html")
        self.assertEqual("Bricks & tiles.\nRecycled bricks.", text)

    def test_output_validation_catches_tampered_provenance(self):
        result = app.research(self.request)
        result["findings"][0]["url"] = "https://other.example/"
        with self.assertRaises(app.ValidationError):
            app.Validator.response(result)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(0, process.returncode, process.stderr)
        self.assertEqual("ok", json.loads(process.stdout)["status"])
        self.assertEqual("", process.stderr)
        self.assertEqual(1, len(process.stdout.splitlines()))

    def test_cli_file_error_and_usage(self):
        for args in ([], [str(ROOT / "missing-input.json")], [str(ROOT)]):
            with self.subTest(args=args):
                process = subprocess.run([sys.executable, "-B",
                                          str(ROOT / "implementation.py"), *args],
                                         capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(2, process.returncode)
                self.assertEqual("error", json.loads(process.stdout)["status"])
                self.assertEqual("", process.stderr)

    def test_cli_malformed_json_and_encoding_in_memory(self):
        for raw in (b"{", b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff", b"[]" ,
                    b"x" * 2_000_001):
            with self.subTest(prefix=raw[:20]):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-input.json"])
                self.assertEqual(2, code)
                self.assertEqual("error", json.loads(output.getvalue())["status"])


if __name__ == "__main__":
    unittest.main()
