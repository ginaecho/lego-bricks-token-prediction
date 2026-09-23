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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        self.url = self.data["urls"][0]

    def single(self, text):
        data = copy.deepcopy(self.data)
        data["urls"] = [self.url]
        data["retrieval_fixtures"] = {self.url: {
            "final_url": self.url, "content": text, "retrieved_at": "2026-09-23T09:00:00Z",
        }}
        return data

    def test_integrated_example(self):
        output = app.run_pipeline(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["insights"]), 6)
        self.assertEqual(output["priority_order"], ["F0004", "F0005", "F0002", "F0003"])

    def test_exact_provenance_and_lines(self):
        output = app.run_pipeline(self.single("  good  \n\nbad"))
        self.assertEqual(output["research"]["findings"][0]["text"], "  good  ")
        self.assertEqual(output["insights"][1]["line_number"], 3)
        for finding, insight in zip(output["research"]["findings"], output["insights"]):
            for key, value in finding.items():
                self.assertEqual(insight[key], value)

    def test_positive_negative_neutral(self):
        output = app.run_pipeline(self.single("love good\nterrible bad\nparcel today"))
        self.assertEqual([i["sentiment_score"] for i in output["insights"]], [3, -3, 0])
        self.assertEqual([i["sentiment"] for i in output["insights"]],
                         ["positive", "negative", "neutral"])

    def test_negation_and_punctuation(self):
        output = app.run_pipeline(self.single("not good\nnot bad\nnot. good"))
        self.assertEqual([i["sentiment_score"] for i in output["insights"]], [-1, 1, 1])
        self.assertTrue(output["insights"][0]["sentiment_evidence"][0]["negated"])

    def test_severity_overrides_positive_sentiment(self):
        output = app.run_pipeline(self.single("great great unsafe\nbad bad bad bad"))
        self.assertEqual(output["insights"][0]["sentiment"], "positive")
        self.assertEqual(output["insights"][0]["severity"], "critical")
        self.assertEqual(output["priority_order"][0], "F0001")

    def test_severity_gap_cannot_be_overwhelmed(self):
        output = app.run_pipeline(self.single("bad " * 300 + "\nslow"))
        self.assertEqual(output["insights"][0]["priority_score"], 99)
        self.assertEqual(output["priority_order"], ["F0002", "F0001"])

    def test_stable_priority_ties(self):
        output = app.run_pipeline(self.single("bad\nbad"))
        self.assertEqual(output["priority_order"], ["F0001", "F0002"])

    def test_word_boundaries(self):
        output = app.run_pipeline(self.single("goodness crashworthy"))
        self.assertEqual(output["insights"][0]["sentiment_score"], 0)
        self.assertEqual(output["insights"][0]["severity"], "low")

    def test_allowlist_rejections(self):
        for url in ("http://reviews.example.test/a", "https://reviews.example.test.evil.test/a",
                    "https://sub.reviews.example.test/a", "https://u@reviews.example.test/a",
                    "https://reviews.example.test:444/a", "https://reviews.example.test/a#b",
                    "https://reviews.example.test\\@evil.test/a"):
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.Schema.url(url, self.data["allowlisted_hosts"])

    def test_redirect_provenance(self):
        response = {"final_url": "https://support.example.test/new", "content": "bad",
                    "retrieved_at": "2026-09-23T09:00:00Z"}
        output = app.run_pipeline(self.single("good"), lambda url: response)
        self.assertEqual(output["insights"][0]["source_url"], response["final_url"])
        self.assertEqual(output["research"]["sources"][0]["requested_url"], self.url)
        self.assertEqual(output["insights"][0]["sentiment"], "negative")

    def test_injected_retrieval_validation(self):
        for response in (None, {}, {"final_url": "https://evil.test/a", "content": "good",
                                   "retrieved_at": "2026-09-23T09:00:00Z"}):
            with self.subTest(response=response), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.single("good"), lambda url: response)

    def test_injected_retrieval_failure(self):
        def broken(url):
            raise RuntimeError("fixture error")
        with self.assertRaisesRegex(app.ValidationError, "retrieval failed"):
            app.run_pipeline(self.data, broken)

    def test_cross_stage_tampering_rejected(self):
        for field, value in (("text", "replacement"), ("source_url", "https://evil.test/a"),
                             ("content_sha256", "0" * 64), ("line_number", 10)):
            stage = app.research(self.data)
            stage["findings"][0][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.sentiment(stage)

    def test_missing_or_extra_findings_rejected(self):
        stage = app.research(self.data)
        stage["findings"].pop()
        with self.assertRaises(app.ValidationError):
            app.sentiment(stage)

    def test_invalid_input_variants(self):
        variants = [None, [], {}, {**self.data, "synthetic": False},
                    {**self.data, "schema_version": "2"}, {**self.data, "urls": []},
                    {**self.data, "extra": True}, {**self.data, "retrieval_fixtures": {}},
                    {**self.data, "urls": [self.url, self.url]}]
        for data in variants:
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_invalid_content_and_timestamp(self):
        for content in ("", " \n ", "x" * 20001, "\ud800", "a\n" * 201):
            with self.subTest(content=repr(content[:20])), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.single(content))
        self.data["retrieval_fixtures"][self.url]["retrieved_at"] = "2026-09-23T09:00:00"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_deterministic_and_does_not_mutate(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_output_validation(self):
        output = app.run_pipeline(self.data)
        output["insights"][0]["sentiment_score"] = 900
        with self.assertRaises(app.ValidationError):
            app.Schema.output(output)

    def test_json_roundtrip_validation(self):
        output = json.loads(json.dumps(app.run_pipeline(self.data)))
        self.assertEqual(app.Schema.output(output), output)

    def test_method_mutation_is_isolated(self):
        output = app.run_pipeline(self.data)
        output["method"]["lexicon"]["good"] = 900
        with self.assertRaises(app.ValidationError):
            app.Schema.output(output)
        self.assertEqual(app.run_pipeline(self.single("good"))["insights"][0]["sentiment_score"], 1)

    def test_unicode_offsets_and_uppercase(self):
        output = app.run_pipeline(self.single("\u0130 NOT GOOD UNSAFE"))
        insight = output["insights"][0]
        self.assertEqual(insight["sentiment_score"], -4)
        self.assertEqual(insight["sentiment_evidence"][0]["offset"], 6)
        self.assertEqual(insight["severity_evidence"][0]["offset"], 11)

    def cli(self, *args):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                   cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        return completed.returncode, json.loads(completed.stdout)

    def test_cli_success(self):
        code, output = self.cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(output["status"], "ok")

    def test_cli_missing_file(self):
        code, output = self.cli("nonexistent-input.json")
        self.assertEqual((code, output["status"]), (2, "error"))

    def test_cli_usage(self):
        code, output = self.cli()
        self.assertEqual((code, output["status"]), (2, "error"))

    def test_cli_bad_json_and_encoding_without_scratch_files(self):
        for raw in (b"{broken", b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff", b"null"):
            with self.subTest(raw=raw), patch.object(Path, "open", return_value=io.BytesIO(raw)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
