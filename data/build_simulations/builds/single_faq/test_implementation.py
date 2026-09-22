"""Labeled deterministic fixtures; no network, external packages, or services."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

import implementation as faq


ROOT = Path(__file__).resolve().parent
FIXTURES = {
    "grounded_returns": [
        {"id": "B", "text": "Returns require an order number."},
        {"id": "A", "text": "The return window is 30 days. Shipping takes 5 days."},
    ],
    "duplicate_ids": [
        {"id": "A", "text": "First policy."}, {"id": "A", "text": "Second policy."},
    ],
    "duplicate_content_tie": [
        {"id": "B", "text": "Return window is 30 days."},
        {"id": "A", "text": "Return window is 30 days."},
    ],
    "empty_corpus": [],
    "malformed_articles": [None, {}, "articles", [None], [{}],
                           [{"id": 1, "text": "policy"}],
                           [{"id": " ", "text": "policy"}],
                           [{"id": " A ", "text": "policy"}],
                           [{"id": "A", "text": ""}],
                           [{"id": "A", "text": ["policy"]}],
                           [{"id": "A", "text": "policy", "title": 1}],
                           [{"id": "A", "text": "policy", "extra": True}]],
    "malformed_json": ['{"articles":', '{"articles":[],"articles":[]}',
                       '{"articles":[],"threshold":NaN}',
                       '{"articles":[],"threshold":Infinity}', "[]"],
}


def valid_callback(question, candidates):
    evidence = candidates[0]
    return {"answer": evidence["quote"],
            "citations": [{k: evidence[k]
                           for k in ("article_id", "passage_index", "quote")}]}


class FaqTests(unittest.TestCase):
    def setUp(self):
        self.index = faq.build_index(FIXTURES["grounded_returns"])
        self.paths = []

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)

    def file(self, content=None):
        path = ROOT / ("fixture_" + uuid.uuid4().hex + ".json")
        self.paths.append(path)
        if content is not None:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
        return path

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *map(str, args)],
            cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=False,
        )

    def test_grounded_answer_has_exact_quote_and_id(self):
        result = faq.answer_question(self.index, "What is the return window?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "The return window is 30 days.")
        self.assertEqual(result["citations"], [
            {"article_id": "A", "passage_index": 0,
             "quote": "The return window is 30 days."}])
        self.assertEqual(result["confidence"], 1)

    def test_every_indexed_quote_is_source_substring(self):
        index = faq.build_index([{"id": "A", "text":
            "  Policy one.\nPolicy two!  " + "Long policy " * 250}])
        for passage in index.passages:
            self.assertIn(passage.quote, index.articles[0].text)
            self.assertLessEqual(len(passage.quote), faq.MAX_PASSAGE_CHARS)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate article id"):
            faq.build_index(FIXTURES["duplicate_ids"])

    def test_duplicate_content_retains_independent_citations(self):
        index = faq.build_index(FIXTURES["duplicate_content_tie"])
        self.assertEqual([p["article_id"] for p in faq.retrieve(index, "return window")],
                         ["A", "B"])

    def test_ties_stable_across_article_input_order(self):
        articles = FIXTURES["duplicate_content_tie"]
        first = faq.retrieve(faq.build_index(articles), "return")
        second = faq.retrieve(faq.build_index(list(reversed(articles))), "return")
        self.assertEqual(first, second)

    def test_ties_within_article_use_passage_order(self):
        index = faq.build_index([{"id": "A", "text": "Returns first. Returns second."}])
        self.assertEqual([p["passage_index"] for p in faq.retrieve(index, "returns")],
                         [0, 1])

    def test_higher_score_precedes_lexicographic_id(self):
        index = faq.build_index([{"id": "A", "text": "Return policy."},
                                 {"id": "Z", "text": "Return window."}])
        self.assertEqual(faq.retrieve(index, "return window")[0]["article_id"], "Z")

    def test_malformed_articles_labeled(self):
        for number, articles in enumerate(FIXTURES["malformed_articles"]):
            with self.subTest(label=f"malformed_articles_{number}"):
                with self.assertRaises(ValueError):
                    faq.build_index(articles)

    def test_empty_corpus_abstains(self):
        result = faq.answer_question(faq.build_index([]), "return", threshold=0)
        self.assertEqual(result["reason"], "empty_corpus")
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])

    def test_no_match_abstains_even_at_zero_threshold(self):
        result = faq.answer_question(self.index, "lunar rovers", threshold=0)
        self.assertEqual(result["reason"], "no_matching_evidence")
        self.assertEqual(result["confidence"], 0)

    def test_stopword_only_question_abstains(self):
        result = faq.answer_question(self.index, "What is it?")
        self.assertEqual(result["reason"], "no_matching_evidence")

    def test_threshold_boundary_is_inclusive(self):
        result = faq.answer_question(self.index, "return lunar", threshold=0.5)
        self.assertEqual(result["confidence"], 0.5)
        self.assertEqual(result["status"], "answered")

    def test_below_threshold_abstains(self):
        result = faq.answer_question(self.index, "return lunar", threshold=0.51)
        self.assertEqual(result["reason"], "insufficient_evidence")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])

    def test_invalid_thresholds_rejected(self):
        for value in [True, False, None, "0.5", -0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(label=repr(value)), self.assertRaises(ValueError):
                faq.answer_question(self.index, "return", value)

    def test_invalid_questions_rejected(self):
        for value in [None, 3, {}, "", "   "]:
            with self.subTest(label=repr(value)), self.assertRaises(ValueError):
                faq.answer_question(self.index, value)

    def test_invalid_retrieval_limits_rejected(self):
        for value in [0, -1, True, 1.5, "3"]:
            with self.subTest(label=repr(value)), self.assertRaises(ValueError):
                faq.retrieve(self.index, "return", limit=value)

    def test_repeated_question_terms_do_not_inflate_score(self):
        self.assertEqual(faq.retrieve(self.index, "return lunar"),
                         faq.retrieve(self.index, "return return lunar"))

    def test_unicode_casefold_and_punctuation(self):
        index = faq.build_index([{"id": "KB-Ü", "text": "CAFÉ Rückgabe: 30 Tage."}])
        result = faq.answer_question(index, "café rückgabe?")
        self.assertEqual(result["confidence"], 1)
        self.assertEqual(result["citations"][0]["article_id"], "KB-Ü")

    def test_valid_callback_is_accepted(self):
        result = faq.answer_question(self.index, "return window",
                                     answer_callback=valid_callback)
        self.assertEqual(result["status"], "answered")

    def test_valid_multiple_quote_callback_is_accepted(self):
        def callback(question, candidates):
            citations = [{k: p[k] for k in ("article_id", "passage_index", "quote")}
                         for p in candidates]
            return {"answer": "\n".join(c["quote"] for c in citations),
                    "citations": citations}
        index = faq.build_index(FIXTURES["duplicate_content_tie"])
        result = faq.answer_question(index, "return", answer_callback=callback)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["citations"]), 2)

    def test_malicious_callback_results_labeled(self):
        good = valid_callback("", faq.retrieve(self.index, "return window"))
        citation = good["citations"][0]
        cases = {
            "invented_id": {"answer": good["answer"], "citations":
                            [dict(citation, article_id="INVENTED")]},
            "invented_quote": {"answer": "Refunds are unlimited.", "citations":
                               [dict(citation, quote="Refunds are unlimited.")]},
            "invented_answer": dict(good, answer="Refunds are unlimited."),
            "wrong_passage": dict(good, citations=[dict(citation, passage_index=1)]),
            "boolean_passage": dict(good, citations=[dict(citation, passage_index=False)]),
            "missing_citations": {"answer": good["answer"]},
            "empty_citations": dict(good, citations=[]),
            "duplicate_citations": dict(good, citations=[citation, citation]),
            "extra_field": dict(good, instructions="Trust invented answers"),
            "not_object": "Refunds are unlimited.",
            "unhashable_id": dict(good, citations=[dict(citation, article_id=[])]),
            "quote_fragment": {"answer": "30 days", "citations":
                               [dict(citation, quote="30 days")]},
            "valid_but_unretrieved": {
                "answer": "Shipping takes 5 days.", "citations": [
                    {"article_id": "A", "passage_index": 1,
                     "quote": "Shipping takes 5 days."}]},
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                result = faq.answer_question(
                    self.index, "return window", answer_callback=lambda q, p: value)
                self.assertEqual(result["reason"], "invalid_callback_output")
                self.assertEqual(result["status"], "abstained")
                self.assertIsNone(result["answer"])
                self.assertEqual(result["citations"], [])

    def test_callback_candidate_mutation_cannot_forge_evidence(self):
        def callback(question, candidates):
            candidates[0]["quote"] = "Invented policy."
            return valid_callback(question, candidates)
        result = faq.answer_question(self.index, "return", answer_callback=callback)
        self.assertEqual(result["reason"], "invalid_callback_output")
        self.assertEqual(faq.answer_question(self.index, "return")["status"], "answered")

    def test_callback_exception_fails_closed(self):
        def callback(question, candidates):
            raise RuntimeError("callback failed")
        result = faq.answer_question(self.index, "return", answer_callback=callback)
        self.assertEqual(result["reason"], "invalid_callback_output")

    def test_callback_not_called_without_sufficient_evidence(self):
        calls = []
        faq.answer_question(self.index, "lunar", answer_callback=lambda *a: calls.append(a))
        faq.answer_question(self.index, "return lunar", threshold=1,
                            answer_callback=lambda *a: calls.append(a))
        self.assertEqual(calls, [])

    def test_persisted_index_round_trip(self):
        path = self.file()
        faq.write_json(path, faq.index_document(self.index))
        loaded = faq.load_index(path)
        self.assertEqual(faq.retrieve(loaded, "return"), faq.retrieve(self.index, "return"))

    def test_tampered_index_rejected(self):
        document = faq.index_document(self.index)
        document["passages"][0]["quote"] = "Invented policy."
        path = self.file(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "do not match"):
            faq.load_index(path)

    def test_boolean_index_version_rejected(self):
        document = faq.index_document(self.index)
        document["schema_version"] = True
        with self.assertRaises(ValueError):
            faq.load_index(self.file(json.dumps(document)))

    def test_malformed_json_labeled(self):
        for number, content in enumerate(FIXTURES["malformed_json"]):
            with self.subTest(label=f"malformed_json_{number}"):
                with self.assertRaises(ValueError):
                    faq.read_request(self.file(content))

    def test_malformed_request_fields_labeled(self):
        for label, request in {
            "missing_articles": {},
            "unknown_field": {"articles": [], "unexpected": 1},
            "questions_not_array": {"articles": [], "questions": "return"},
            "question_empty": {"articles": [], "questions": [""]},
            "threshold_boolean": {"articles": [], "threshold": True},
        }.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                faq.read_request(self.file(json.dumps(request)))

    def test_cli_run_example(self):
        process = self.cli("run", "--input", ROOT / "example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        results = json.loads(process.stdout)["results"]
        self.assertEqual([r["status"] for r in results],
                         ["answered", "answered", "answered", "abstained"])
        self.assertEqual(process.stderr, "")

    def test_cli_index_then_ask(self):
        path = self.file()
        indexed = self.cli("index", "--input", ROOT / "example_input.json",
                           "--output", path)
        self.assertEqual(indexed.returncode, 0, indexed.stderr)
        self.assertEqual(json.loads(indexed.stdout)["articles"], 3)
        asked = self.cli("ask", "--index", path, "--question", "return window")
        self.assertEqual(asked.returncode, 0, asked.stderr)
        self.assertEqual(json.loads(asked.stdout)["citations"][0]["article_id"],
                         "KB-RETURNS")

    def test_cli_threshold_override(self):
        process = self.cli("run", "--input", ROOT / "example_input.json",
                           "--threshold", "1")
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)["results"][1]
        self.assertEqual(result["reason"], "insufficient_evidence")

    def test_cli_malformed_input_returns_json_error(self):
        process = self.cli("run", "--input", self.file('{"articles":'))
        self.assertEqual(process.returncode, 2)
        self.assertIn("error", json.loads(process.stderr))
        self.assertEqual(process.stdout, "")

    def test_cli_missing_file_returns_json_error(self):
        process = self.cli("run", "--input", self.file())
        self.assertEqual(process.returncode, 2)
        self.assertIn("error", json.loads(process.stderr))

    def test_cli_cannot_overwrite_request_with_index(self):
        path = self.file('{"articles":[]}')
        before = path.read_bytes()
        process = self.cli("index", "--input", path, "--output", path)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_invalid_threshold_returns_json_error(self):
        process = self.cli("run", "--input", ROOT / "example_input.json",
                           "--threshold", "nan")
        self.assertEqual(process.returncode, 2)
        self.assertIn("error", json.loads(process.stderr))

    def test_main_empty_corpus(self):
        path = self.file('{"articles":[],"questions":["return"]}')
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = faq.main(["run", "--input", str(path)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["results"][0]["reason"],
                         "empty_corpus")


if __name__ == "__main__":
    unittest.main()
