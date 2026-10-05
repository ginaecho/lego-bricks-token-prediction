"""Synthetic, local fixtures only; tests create no files."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_grounded_offline_answer(self):
        result = app.support(self.request)
        self.assertEqual(result["disposition"], "answered")
        self.assertEqual(result["answer"], self.request["knowledge_base"][0]["answer"])
        self.assertEqual(result["citations"][0]["id"], "synthetic-password-reset")
        self.assertIn("offline", result["notice"])
        self.assertEqual(result["handoff"], "not_needed")

    def test_online_and_normalization(self):
        self.request.update(team_online=True, message="RESET PASSWORD!!!")
        result = app.support(self.request)
        self.assertEqual(result["disposition"], "answered")
        self.assertIn("online", result["notice"])

    def test_unknown_topic_does_not_invent(self):
        self.request["message"] = "Satellite telemetry stopped working"
        result = app.support(self.request)
        self.assertEqual(result["disposition"], "needs_clarification")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["handoff"], "consent_required")
        self.assertIn("No ticket", result["notice"])

    def test_empty_knowledge_base(self):
        self.request["knowledge_base"] = []
        self.assertEqual(app.support(self.request)["disposition"], "needs_clarification")

    def test_stopwords_and_unicode_are_safe(self):
        for message in ("How do I?", "???", "你好，请帮助"):
            with self.subTest(message=message):
                self.request["message"] = message
                self.assertEqual(app.support(self.request)["disposition"], "needs_clarification")

    def test_ambiguous_answers_are_not_selected(self):
        duplicate = copy.deepcopy(self.request["knowledge_base"][0])
        duplicate.update(id="synthetic-other-password", answer="A different service policy.")
        self.request["knowledge_base"].append(duplicate)
        result = app.support(self.request)
        self.assertEqual(result["disposition"], "needs_clarification")
        self.assertIn("More than one", result["answer"])

    def test_human_request_overrides_matching_article(self):
        self.request.update(message="Talk to a human agent about password reset",
                            consent_contact=True)
        result = app.support(self.request)
        self.assertEqual(result["disposition"], "human_requested")
        self.assertEqual(result["handoff"], "suggested")
        self.assertIn("No ticket", result["notice"])
        self.assertEqual(result["citations"], [])

    def test_invalid_input_fields_and_types(self):
        for field, value in (
            ("message", ""), ("message", " " * 4), ("message", 7),
            ("message", "x" * 4001), ("request_id", None),
            ("team_online", 1), ("consent_contact", "yes"),
            ("knowledge_base", {}), ("fixture_label", ""),
            ("unknown_field", True),
        ):
            with self.subTest(field=field, value=str(value)[:30]):
                request = copy.deepcopy(self.request)
                request[field] = value
                with self.assertRaises(app.ValidationError):
                    app.support(request)
        for invalid in ([], None, {}, "request"):
            with self.assertRaises(app.ValidationError):
                app.support(invalid)

    def test_invalid_articles_and_duplicate_ids(self):
        for field, value in (("keywords", "password"), ("keywords", [4]),
                             ("answer", ""), ("title", None), ("id", " id ")):
            request = copy.deepcopy(self.request)
            request["knowledge_base"][0][field] = value
            with self.assertRaises(app.ValidationError):
                app.support(request)
        self.request["knowledge_base"].append(copy.deepcopy(self.request["knowledge_base"][0]))
        with self.assertRaises(app.ValidationError):
            app.support(self.request)

    def test_output_grounding_validation(self):
        result = app.support(self.request)
        result["answer"] = "An invented guarantee."
        with self.assertRaises(app.ValidationError):
            app.validate(result, "output", self.request)

    def test_deterministic_and_does_not_mutate(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(app.support(self.request), app.support(self.request))
        self.assertEqual(original, self.request)

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_file_json_and_usage_errors(self):
        for args in ([], ["--not-a-file"], ["implementation.py"],
                     ["example_input.json", "extra"]):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_malformed_json_and_schema_without_files(self):
        fixtures = [b'{"x":1,"x":2}', b'{"x":NaN}', b'{}', b'\xff', b'[', b'[' * 2000,
                    b' ' * (app.MAX_FILE_BYTES + 1)]
        for raw in fixtures:
            with self.subTest(size=len(raw)):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                    with redirect_stdout(output):
                        code = app.main(["synthetic-memory-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
