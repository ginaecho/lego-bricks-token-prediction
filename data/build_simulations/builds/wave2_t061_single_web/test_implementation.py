import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, canonical_url, research


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_normal_provenance(self):
        result = research(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["findings"]), 1)
        finding = result["findings"][0]
        source = result["sources"][finding["source_index"]]
        self.assertEqual(source["text"][finding["start"]:finding["end"]], finding["quote"])
        self.assertEqual(finding["sha256"], hashlib.sha256(
            self.data["fixtures"][0]["body"].encode()).hexdigest())
        self.assertEqual(finding["matched_terms"], ["bricks", "recycled"])
        self.assertNotIn("hidden claim", source["text"])

    def test_deterministic_deduplication(self):
        self.data["urls"].append("https://RESEARCH.EXAMPLE:443/materials#other")
        result = research(self.data)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result, research(copy.deepcopy(self.data)))

    def test_missing_fixture_is_explicit(self):
        self.data["fixtures"] = []
        result = research(self.data)
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["warnings"][0]["code"], "fixture_missing")

    def test_empty_body_and_no_matches(self):
        for body in ("", "Nothing relevant here.", "<style>bricks</style>"):
            with self.subTest(body=body):
                self.data["fixtures"][0]["body"] = body
                self.assertEqual(research(self.data)["findings"], [])

    def test_disallowed_urls(self):
        for url in ("http://research.example/a", "https://evil.example/a",
                    "https://research.example.evil.example/a", "https://sub.research.example/a",
                    "https://user@research.example/a", "https://research.example:444/a",
                    "https://research.example\\@evil.example/a", "https://research.example:/a"):
            with self.subTest(url=url):
                self.data["urls"] = [url]
                with self.assertRaises(ValidationError):
                    research(self.data)

    def test_invalid_schema(self):
        for key, value in (("schema_version", True), ("query", ""), ("query", "!!!"),
                           ("urls", []), ("urls", "not a list"),
                           ("allowlisted_hosts", ["*.example"]), ("fixtures", {})):
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(ValidationError):
                    research(data)
        with self.assertRaises(ValidationError):
            research([])

    def test_fixture_validation(self):
        for key, value in (("synthetic", False), ("body", None),
                           ("content_type", "application/json"), ("title", "")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["fixtures"][0][key] = value
                with self.assertRaises(ValidationError):
                    research(data)
        self.data["fixtures"] *= 2
        with self.assertRaises(ValidationError):
            research(self.data)

    def test_plain_text_unicode(self):
        self.data["query"] = "café"
        self.data["fixtures"][0].update(content_type="text/plain", body="A café exists! Other.")
        finding = research(self.data)["findings"][0]
        self.assertEqual(finding["quote"], "A café exists!")
        self.assertEqual(finding["start"], 0)

    def test_canonicalization(self):
        self.assertEqual(canonical_url("https://RESEARCH.EXAMPLE:443#x"), "https://research.example/")
        self.assertNotEqual(canonical_url("https://research.example/?a=1"),
                            canonical_url("https://research.example/?a=2"))

    def test_cli_success(self):
        proc = self.cli("example_input.json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), research(self.data))
        self.assertEqual(proc.stderr, "")

    def test_cli_errors(self):
        for args in ((), ("missing-input.json",), ("implementation.py",), ("build_manifest.json",)):
            with self.subTest(args=args):
                proc = self.cli(*args)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main()
