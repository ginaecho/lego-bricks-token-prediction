"""All fixtures are synthetic; tests never contact providers or create files."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, support


ROOT = Path(__file__).parent


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_offline_grounded_answer(self):
        result = support(self.payload)
        self.assertEqual(result["resolution"], "answered")
        self.assertEqual(result["citations"][0]["id"], "synthetic-returns")
        self.assertIn(self.payload["knowledge_base"][0]["body"], result["answer"])
        self.assertFalse(result["handoff"]["needed"])

    def test_online_unknown_question(self):
        self.payload.update(question="Quantum wormhole warranty?", team_online=True)
        result = support(self.payload)
        self.assertEqual(result["handoff"]["reason"], "no_grounded_match")
        self.assertEqual(result["citations"], [])
        self.assertIn("marked online", result["answer"])
        self.assertEqual(result["handoff"]["delivery"], "not_sent")

    def test_empty_knowledge_offline(self):
        self.payload["knowledge_base"] = []
        result = support(self.payload)
        self.assertEqual(result["resolution"], "escalated")
        self.assertIn("offline", result["answer"])
        self.assertIn("No ticket", result["answer"])

    def test_human_request_overrides_match(self):
        self.payload["question"] = "Human agent for return purchase please"
        result = support(self.payload, lambda _: self.fail("Selector must not run"))
        self.assertEqual(result["handoff"]["reason"], "human_requested")

    def test_stopwords_and_unicode_abstain(self):
        for question in ("???", "How do I?", "你好"):
            with self.subTest(question=question):
                self.payload["question"] = question
                self.assertEqual(support(self.payload)["resolution"], "escalated")

    def test_invalid_request(self):
        for replacement in (
            {"question": ""}, {"team_online": 1}, {"knowledge_base": {}},
            {"unexpected": True}, {"question": "x" * 2001},
        ):
            with self.subTest(replacement=replacement):
                request = copy.deepcopy(self.payload)
                request.update(replacement)
                with self.assertRaises(ValidationError):
                    support(request)
        for request in (None, [], {}, "question"):
            with self.assertRaises(ValidationError):
                support(request)

    def test_duplicate_and_malformed_articles(self):
        self.payload["knowledge_base"].append(copy.deepcopy(self.payload["knowledge_base"][0]))
        with self.assertRaises(ValidationError):
            support(self.payload)
        self.payload["knowledge_base"] = [{"id": "bad", "title": "Bad", "body": "Text",
                                          "keywords": "not-a-list"}]
        with self.assertRaises(ValidationError):
            support(self.payload)

    def test_injected_selector_fixture(self):
        def selector(context):
            self.assertEqual(context["candidates"][0]["id"], "synthetic-returns")
            return {"article_ids": ["synthetic-returns"]}
        self.assertEqual(support(self.payload, selector), support(self.payload))

    def test_selector_invalid_output_and_abstention(self):
        for value in (None, {"article_ids": ["invented"]},
                      {"article_ids": ["synthetic-password"]},
                      {"article_ids": ["synthetic-returns"] * 2},
                      {"article_ids": [], "answer": "Unverified claim"}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                support(self.payload, lambda context: value)
        result = support(self.payload, lambda context: {"article_ids": []})
        self.assertEqual(result["handoff"]["reason"], "selector_abstained")

    def test_selector_exception(self):
        def broken(context):
            raise RuntimeError("Fixture failure")
        with self.assertRaises(ValidationError):
            support(self.payload, broken)

    def test_determinism_and_no_mutation(self):
        before = copy.deepcopy(self.payload)
        first = support(self.payload)
        self.payload["knowledge_base"].reverse()
        self.assertEqual(first, support(self.payload))
        self.payload["knowledge_base"].reverse()
        self.assertEqual(before, self.payload)

    def run_cli(self, *args):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        return result.returncode, json.loads(result.stdout)

    def test_cli_success(self):
        code, result = self.run_cli(str(ROOT / "example_input.json"))
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ok")

    def test_cli_missing_file_arguments_and_invalid_json(self):
        for args in ((), ("missing-synthetic-file.json",),
                     (str(ROOT / "implementation.py"),), ("one", "two"),
                     (str(ROOT / "build_manifest.json"),)):
            with self.subTest(args=args):
                code, result = self.run_cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_strict_json_parser(self):
        from implementation import strict_object, reject_constant
        for text in ('{"question":"a","question":"b"}', '{"question": NaN}'):
            with self.assertRaises(ValidationError):
                json.loads(text, object_pairs_hook=strict_object, parse_constant=reject_constant)


if __name__ == "__main__":
    unittest.main()
