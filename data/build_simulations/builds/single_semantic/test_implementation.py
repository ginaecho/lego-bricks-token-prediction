import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from implementation import TextIndex, main, run_request


LABELED_DOCUMENTS = [
    {"id": "billing", "text": "invoice payment billing"},
    {"id": "delivery", "text": "parcel shipment delivery"},
    {"id": "refund", "text": "refund return reimbursement"},
]
LEXICAL_LABELS = [
    ("payment invoice", "billing"),
    ("shipment", "delivery"),
    ("return reimbursement", "refund"),
]
EMBEDDING_LABELS = [
    ("Where is my package?", "delivery", [0, 1, 0]),
    ("Give my money back", "refund", [0, 0, 1]),
    ("Settle the charge", "billing", [1, 0, 0]),
]
DOCUMENT_VECTORS = {"billing": [1, 0, 0], "delivery": [0, 1, 0], "refund": [0, 0, 1]}


class LexicalTests(unittest.TestCase):
    def test_labeled_retrieval(self):
        index = TextIndex(LABELED_DOCUMENTS)
        for query, label in LEXICAL_LABELS:
            with self.subTest(query=query):
                response = index.search(query)
                self.assertEqual(response["results"][0]["id"], label)
                self.assertEqual(response["mode"], "lexical_tfidf")
                self.assertIn("not neural", response["interpretation"])
                self.assertIsNotNone(response["fallback_reason"])

    def test_hand_computed_tfidf_cosine(self):
        index = TextIndex([
            {"id": "a", "text": "cat cat dog"},
            {"id": "b", "text": "dog"},
        ])
        score = index.search("cat")["results"][0]["score"]
        cat_weight = 2 * (math.log(3 / 2) + 1)
        self.assertAlmostEqual(score, cat_weight / math.sqrt(cat_weight ** 2 + 1))

    def test_stable_ties_and_input_order(self):
        documents = [{"id": key, "text": "same word"} for key in ["z", "b", "a"]]
        for ordered in [documents, list(reversed(documents))]:
            self.assertEqual(
                [r["id"] for r in TextIndex(ordered).search("word", top_k=2)["results"]],
                ["a", "b"],
            )

    def test_threshold_and_large_top_k(self):
        index = TextIndex([{"id": "a", "text": "cat dog"}, {"id": "b", "text": "cat"}])
        self.assertEqual(len(index.search("cat", top_k=100)["results"]), 2)
        self.assertEqual([r["id"] for r in index.search("cat", min_score=1)["results"]], ["b"])
        self.assertEqual(index.search("dog", min_score=1)["reason"], "no_match_above_threshold")

    def test_zero_vector_and_no_arbitrary_results(self):
        index = TextIndex(LABELED_DOCUMENTS)
        for query in ["astronomy", "!!!"]:
            response = index.search(query)
            self.assertEqual(response["reason"], "zero_query_vector")
            self.assertEqual(response["results"], [])

    def test_exact_evidence_offsets_escaping_and_token_boundaries(self):
        text = "<script>Cat</script> cat scatter CAT & café"
        result = TextIndex([{"id": "a", "text": text}]).search("cat café")["results"][0]
        self.assertEqual([span["text"] for span in result["evidence"]], ["Cat", "cat", "CAT", "café"])
        for span in result["evidence"]:
            self.assertEqual(text[span["start"]:span["end"]], span["text"])
        self.assertIn("&lt;script&gt;<mark>Cat</mark>", result["highlighted_text"])
        self.assertIn(" scatter ", result["highlighted_text"])
        self.assertIn("&amp;", result["highlighted_text"])

    def test_explicit_lexical_mode(self):
        self.assertIsNone(TextIndex(LABELED_DOCUMENTS).search("invoice", mode="lexical")["fallback_reason"])

    def test_input_is_copied(self):
        documents = [{"id": "a", "text": "cat"}]
        index = TextIndex(documents)
        documents[0]["text"] = "dog"
        self.assertEqual(index.search("cat")["results"][0]["text"], "cat")

    def test_invalid_documents(self):
        bad = [None, [], {}, "text", [None], [{"id": "a"}],
               [{"id": "", "text": "cat"}], [{"id": 1, "text": "cat"}],
               [{"id": "a", "text": value} for value in ["cat", "dog"]]]
        bad += [[{"id": "a", "text": value}] for value in ["", " ", "...", None, 1]]
        bad += [[{"id": "a", "text": "cat", "extra": True}]]
        for documents in bad:
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                TextIndex(documents)

    def test_invalid_queries_and_options(self):
        index = TextIndex(LABELED_DOCUMENTS)
        for query in ["", " ", None, 1]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                index.search(query)
        options = [{"top_k": value} for value in [0, -1, True, 1.5, "2"]]
        options += [{"min_score": value} for value in [-0.1, 1.1, True, "0", math.nan, math.inf]]
        options += [{"mode": value} for value in ["semantic", None, [], {}]]
        options += [{"mode": "embedding"}]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                index.search("invoice", **kwargs)


class EmbeddingTests(unittest.TestCase):
    def test_labeled_injected_callback_and_batch_contract(self):
        calls = []
        query_vectors = {query: vector for query, _, vector in EMBEDDING_LABELS}

        def callback(batch):
            calls.append(batch)
            if len(calls) == 1:
                return DOCUMENT_VECTORS
            return {batch[0]["id"]: query_vectors[batch[0]["text"]]}

        index = TextIndex(LABELED_DOCUMENTS, callback)
        self.assertEqual([d["id"] for d in calls[0]], ["billing", "delivery", "refund"])
        for query, label, _ in EMBEDDING_LABELS:
            result = index.search(query)
            self.assertEqual(result["mode"], "embedding_cosine")
            self.assertEqual(result["results"][0]["id"], label)
            self.assertEqual(result["results"][0]["evidence"], [])
            self.assertEqual(result["results"][0]["score"], 1.0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(index.metadata["embedding_dimensions"], 3)
        self.assertEqual(index.search("invoice", mode="lexical")["mode"], "lexical_tfidf")
        self.assertEqual(len(calls), 4)

    def test_nonunit_cosine_and_stable_tie(self):
        request = {
            "documents": [{"id": "b", "text": "one"}, {"id": "a", "text": "two"}],
            "query": "other",
            "embeddings": {"documents": {"a": [3, 4], "b": [6, 8]}, "query": [1, 0]},
        }
        result = run_request(request)["search"]["results"]
        self.assertEqual([r["id"] for r in result], ["a", "b"])
        self.assertAlmostEqual(result[0]["score"], 0.6)

    def test_zero_negative_orthogonal_and_threshold(self):
        for document_vector, query_vector, expected_reason in [
            ([0, 0], [1, 0], "no_match_above_threshold"),
            ([1, 0], [0, 0], "zero_query_vector"),
            ([-1, 0], [1, 0], "no_match_above_threshold"),
            ([0, 1], [1, 0], "no_match_above_threshold"),
        ]:
            with self.subTest(document=document_vector, query=query_vector):
                result = run_request({
                    "documents": [{"id": "a", "text": "text"}], "query": "query",
                    "embeddings": {"documents": {"a": document_vector}, "query": query_vector},
                })["search"]
                self.assertEqual(result["reason"], expected_reason)
                self.assertEqual(result["results"], [])

    def test_extreme_finite_coordinates(self):
        for scale in [1e308, 1e-308]:
            result = run_request({
                "documents": [{"id": "a", "text": "text"}], "query": "query",
                "embeddings": {"documents": {"a": [scale] * 4}, "query": [scale] * 4},
            })["search"]["results"]
            self.assertAlmostEqual(result[0]["score"], 1.0)

    def test_invalid_document_embeddings(self):
        bad = [None, [], {}, {"wrong": [1]}, {"a": [1], "extra": [1]},
               {"a": []}, {"a": "vector"}, {"a": [True]}, {"a": ["1"]},
               {"a": [math.nan]}, {"a": [math.inf]}, {"a": [10 ** 1000]}]
        for vectors in bad:
            with self.subTest(vectors=str(vectors)[:80]), self.assertRaises(ValueError):
                TextIndex([{"id": "a", "text": "word"}], lambda batch: vectors)
        with self.assertRaises(ValueError):
            TextIndex(LABELED_DOCUMENTS, "not callable")
        with self.assertRaises(ValueError):
            TextIndex(LABELED_DOCUMENTS, lambda batch: {
                "billing": [1], "delivery": [1, 0], "refund": [1],
            })

    def test_invalid_query_embeddings(self):
        for query_mapping in [{}, {"wrong": [1, 0]}, {"__query__": []},
                              {"__query__": [1]}, {"__query__": [1, math.nan]},
                              {"__query__": [1, 0], "extra": [1, 0]}]:
            calls = iter([{"a": [1, 0]}, query_mapping])
            index = TextIndex([{"id": "a", "text": "word"}], lambda batch: next(calls))
            with self.subTest(mapping=query_mapping), self.assertRaises(ValueError):
                index.search("query")

    def test_document_id_can_equal_query_id(self):
        result = run_request({
            "documents": [{"id": "__query__", "text": "word"}], "query": "other",
            "embeddings": {"documents": {"__query__": [1, 0]}, "query": [1, 0]},
        })
        self.assertEqual(result["search"]["results"][0]["id"], "__query__")


class CliTests(unittest.TestCase):
    def test_real_example_cli(self):
        root = Path(__file__).resolve().parent
        completed = subprocess.run(
            [sys.executable, "-B", str(root / "implementation.py"), str(root / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["search"]["results"][0]["id"], "billing")

    def test_cli_embedding_json(self):
        request = {"documents": LABELED_DOCUMENTS, "query": "Where is my package?",
                   "embeddings": {"documents": DOCUMENT_VECTORS, "query": [0, 1, 0]}}
        with patch.object(Path, "read_text", return_value=json.dumps(request)):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["input.json"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["search"]["results"][0]["id"], "delivery")

    def test_cli_invalid_json_and_requests(self):
        payloads = ["{", "[]", "{}", '{"documents":[],"documents":[]}',
                    '{"query":NaN}', '{"query":Infinity}', '{"unknown":1}']
        for payload in payloads:
            with self.subTest(payload=payload), patch.object(Path, "read_text", return_value=payload):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(["input.json"]), 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("error", json.loads(stderr.getvalue()))

    def test_cli_missing_file(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("missing")):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["missing.json"]), 2)

    def test_invalid_embedding_payload(self):
        for embeddings in [None, {}, {"documents": {}}, {"documents": {}, "query": [], "extra": 1}]:
            with self.subTest(embeddings=embeddings), self.assertRaises(ValueError):
                run_request({"documents": LABELED_DOCUMENTS, "query": "word", "embeddings": embeddings})


if __name__ == "__main__":
    unittest.main()
