"""All customer-support fixtures below are synthetic."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    return {
        "question": "What is the synthetic return policy?",
        "knowledge_base": [
            {"id": "returns", "title": "Synthetic returns FAQ",
             "content": "Synthetic return policy: unopened items may be returned within 30 days."},
            {"id": "shipping", "title": "Synthetic shipping FAQ",
             "content": "Standard shipping takes five business days."},
        ],
    }


class SupportTests(unittest.TestCase):
    def test_grounded_answer(self):
        request = fixture()
        result = app.answer_question(request)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], ["returns"])
        self.assertEqual(result["answer"], request["knowledge_base"][0]["content"])
        self.assertEqual(result["evidence"][0]["score"], 1.0)

    def test_no_match_abstains(self):
        request = fixture()
        request["question"] = "Warranty exclusions?"
        result = app.answer_question(request)
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])

    def test_empty_knowledge_base_and_stopwords(self):
        for request in (
            {"question": "Returns?", "knowledge_base": []},
            {"question": "What is the?", "knowledge_base": fixture()["knowledge_base"]},
            {"question": "!!!", "knowledge_base": fixture()["knowledge_base"]},
        ):
            self.assertEqual(app.answer_question(request)["status"], "abstained")

    def test_invalid_requests(self):
        invalid = [None, [], {}, {"question": "", "knowledge_base": []}]
        for key, value in (
            ("question", 5), ("knowledge_base", {}), ("min_score", True),
            ("min_score", 0), ("min_score", float("nan")), ("min_score", 1.1),
            ("max_results", False), ("max_results", 0), ("unknown", "value"),
        ):
            request = fixture()
            request[key] = value
            invalid.append(request)
        for request in invalid:
            with self.subTest(request=request):
                self.assertEqual(app.answer_question(request)["status"], "error")

    def test_invalid_articles(self):
        for articles in (
            [{"id": "a", "title": "a"}],
            [{"id": "a", "title": "a", "content": " "}],
            fixture()["knowledge_base"][:1] * 2,
        ):
            request = fixture()
            request["knowledge_base"] = articles
            self.assertEqual(app.answer_question(request)["status"], "error")

    def test_threshold_and_title_only(self):
        request = {"question": "return refund", "min_score": 0.75,
                   "knowledge_base": [{"id": "a", "title": "return refund",
                                       "content": "return unopened packages"}]}
        self.assertEqual(app.answer_question(request)["status"], "abstained")
        request["min_score"] = 0.5
        self.assertEqual(app.answer_question(request)["status"], "answered")
        request["question"] = "refund"
        self.assertEqual(app.answer_question(request)["status"], "abstained")

    def test_deterministic_ties_limit_and_no_mutation(self):
        request = {"question": "RETURN", "max_results": 1, "knowledge_base": [
            {"id": "b", "title": "B", "content": "Return B"},
            {"id": "a", "title": "A", "content": "Return A"},
        ]}
        original = copy.deepcopy(request)
        self.assertEqual(app.answer_question(request)["citations"], ["a"])
        self.assertEqual(request, original)

    def test_injected_grounded_callable(self):
        def answerer(question, evidence):
            return {"answer": evidence[0]["quote"], "citations": [evidence[0]["id"]]}
        self.assertEqual(app.answer_question(fixture(), answerer)["status"], "answered")

    def test_injected_unsupported_answer_and_citation(self):
        for candidate in (
            {"answer": "Refunds are guaranteed.", "citations": ["returns"]},
            {"answer": "Anything", "citations": ["missing"]},
            {"answer": "", "citations": []}, None,
        ):
            self.assertEqual(
                app.answer_question(fixture(), lambda q, e: candidate)["status"], "error"
            )

    def test_injected_mutation_and_exception(self):
        def mutate(question, evidence):
            evidence[0]["quote"] = "invented"
            return {"answer": "invented", "citations": [evidence[0]["id"]]}

        def fail(question, evidence):
            raise RuntimeError("synthetic failure")

        for answerer in (mutate, fail):
            self.assertEqual(app.answer_question(fixture(), answerer)["status"], "error")

    def test_abstention_does_not_call_answerer(self):
        def forbidden(question, evidence):
            self.fail("answerer called without supporting evidence")
        request = {"question": "unknown", "knowledge_base": []}
        self.assertEqual(app.answer_question(request, forbidden)["status"], "abstained")

    def test_cli_example(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "answered")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_encoding(self):
        for text in ('{', '{"question":"x","question":"y"}', '{"min_score":NaN}', '[]'):
            output = StringIO()
            with patch.object(Path, "read_text", return_value=text), redirect_stdout(output):
                self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
        output = StringIO()
        with patch.object(Path, "read_text", side_effect=UnicodeError), redirect_stdout(output):
            self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
