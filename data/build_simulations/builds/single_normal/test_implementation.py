import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import MAX_TOTAL_CHARACTERS, ValidationError, research


LABELED_FIXTURES = [
    {
        "label": "relevant_passage_beats_distractor",
        "question": "solar battery capacity",
        "documents": [
            {"id": "distractor", "text": "The garden contains red flowers."},
            {"id": "battery", "text": "Solar battery capacity is 12 kWh. Delivery takes two days."},
        ],
        "expected_source": "battery",
        "expected_quote": "Solar battery capacity is 12 kWh.",
    },
    {
        "label": "unicode_offsets",
        "question": "café capacity",
        "documents": [{"id": "unicode", "text": "☀ Welcome!\n  Café capacity is 25 guests."}],
        "expected_source": "unicode",
        "expected_quote": "Café capacity is 25 guests.",
    },
    {
        "label": "query_coverage_ranking",
        "question": "battery warranty",
        "documents": [
            {"id": "weak", "text": "Battery battery battery battery."},
            {"id": "strong", "text": "Battery warranty lasts 10 years."},
        ],
        "expected_source": "strong",
        "expected_quote": "Battery warranty lasts 10 years.",
    },
]


def request(question="battery warranty", texts=None):
    return {
        "question": question,
        "documents": [
            {"id": str(i), "text": text}
            for i, text in enumerate(texts or ["Battery warranty lasts 10 years."])
        ],
    }


class ResearchTests(unittest.TestCase):
    def assert_grounded(self, result, original):
        sources = {d["id"]: d["text"] for d in original["documents"]}
        for finding in result["findings"]:
            self.assertEqual(finding["text"], finding["citations"][0]["quote"])
            for citation in finding["citations"]:
                self.assertEqual(
                    citation["quote"],
                    sources[citation["source_id"]][citation["start"]:citation["end"]],
                )

    def test_labeled_retrieval_fixtures(self):
        for fixture in LABELED_FIXTURES:
            with self.subTest(label=fixture["label"]):
                original = {key: fixture[key] for key in ("question", "documents")}
                result = research(original)
                first = result["findings"][0]
                self.assertEqual(first["text"], fixture["expected_quote"])
                self.assertEqual(first["citations"][0]["source_id"], fixture["expected_source"])
                self.assert_grounded(result, original)

    def test_no_match(self):
        result = research(request("ocean salinity"))
        self.assertEqual(result["status"], "no_matching_evidence")
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["missing_evidence"]["terms_absent_from_documents"],
                         ["ocean", "salinity"])

    def test_empty_collection_and_stopword_question(self):
        self.assertEqual(research({"question": "battery", "documents": []})["findings"], [])
        result = research(request("what is the"))
        self.assertEqual(result["missing_evidence"]["reason"], "no_content_query_terms")

    def test_partial_evidence(self):
        result = research(request("battery warranty recycling"))
        self.assertEqual(result["missing_evidence"]["terms_absent_from_documents"], ["recycling"])

    def test_duplicate_ids_rejected(self):
        data = request(texts=["Battery one.", "Battery two."])
        data["documents"][1]["id"] = "0"
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            research(data)

    def test_malformed_requests(self):
        invalid = [
            None, [], {}, {"question": "ok", "documents": "bad"},
            {"question": 42, "documents": []},
            {"question": " ", "documents": []},
            {"question": "ok", "documents": [None]},
            {"question": "ok", "documents": [{"id": [], "text": "x"}]},
            {"question": "ok", "documents": [{"id": "", "text": "x"}]},
            {"question": "ok", "documents": [{"id": "x", "text": 42}]},
            {"question": "ok", "documents": [{"id": "x", "text": " "}]},
            {"question": "ok", "documents": [{"id": "x"}]},
            {"question": "ok", "documents": [], "extra": True},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                research(data)

    def test_bounds(self):
        for data in [
            request("a" * 2001),
            request(" ".join("term" + str(i) for i in range(65))),
            request(texts=["a" * (MAX_TOTAL_CHARACTERS + 1)]),
            request(texts=["text"] * 65),
        ]:
            with self.assertRaises(ValidationError):
                research(data)
        for limit in (True, 0, 13, "6"):
            with self.assertRaises(ValidationError):
                research(request(), max_findings=limit)

    def test_duplicate_quotes_merge_provenance(self):
        data = request(texts=["Battery warranty lasts 10 years."] * 2)
        result = research(data)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(len(result["findings"][0]["citations"]), 2)
        self.assertEqual(result["statistics"]["duplicate_passages_merged"], 1)
        self.assert_grounded(result, data)

    def test_long_overlapping_spans_are_suppressed(self):
        text = "battery " * 170 + "warranty"
        data = request(texts=[text])
        result = research(data)
        self.assertGreater(result["statistics"]["overlapping_passages_suppressed"], 0)
        self.assert_grounded(result, data)
        spans = [f["citations"][0] for f in result["findings"]]
        for i, left in enumerate(spans):
            self.assertLessEqual(left["end"] - left["start"], 600)
            for right in spans[i + 1:]:
                self.assertTrue(left["end"] <= right["start"] or right["end"] <= left["start"])

    def test_labeled_conflicting_documents(self):
        cases = [
            ("numeric_difference", ["Battery warranty lasts 10 years.",
                                    "Battery warranty lasts 2 years."]),
            ("polarity_difference", ["The battery supports rapid charging.",
                                     "The battery does not support rapid charging."]),
        ]
        for label, texts in cases:
            with self.subTest(label=label):
                data = request("battery warranty charging", texts)
                result = research(data)
                self.assertEqual(len(result["findings"]), 2)
                self.assertEqual(result["conflicts"][0]["reason"], label)
                self.assert_grounded(result, data)

    def test_nonconflicting_documents(self):
        result = research(request(texts=["Battery warranty lasts 10 years.",
                                         "Battery recycling protects rivers."]))
        self.assertEqual(result["conflicts"], [])

    def test_output_limit_and_uncovered_terms(self):
        result = research(request("battery recycling", ["Battery lasts.", "Recycling helps."]),
                          max_findings=1)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["missing_evidence"]["terms_not_in_selected_findings"], ["recycling"])

    def test_deterministic_and_does_not_mutate_input(self):
        data = request()
        original = copy.deepcopy(data)
        self.assertEqual(research(data), research(data))
        self.assertEqual(data, original)

    def test_valid_callback(self):
        def callback(result):
            first = result["findings"][0]
            return {"findings": [{"text": first["text"], "citations": first["citations"]}]}
        result = research(request(), synthesis_callback=callback)
        self.assertEqual(result["synthesized_findings"][0]["text"], result["findings"][0]["text"])

    def test_callback_rejects_ungrounded_outputs(self):
        def altered(field, value):
            def callback(result):
                first = result["findings"][0]
                citation = first["citations"][0]
                citation[field] = value
                return {"findings": [{"text": first["text"], "citations": [citation]}]}
            return callback
        for callback in [
            altered("source_id", "invented"), altered("quote", "Invented claim."),
            altered("start", -1), altered("end", 999), altered("start", True),
            lambda _: {"findings": [{"text": "Unsupported conclusion.", "citations": []}]},
            lambda _: {"other": []},
            lambda result: {"findings": [{"text": "Unsupported conclusion.",
                                         "citations": result["findings"][0]["citations"]}]},
        ]:
            with self.subTest(callback=callback), self.assertRaises(ValidationError):
                research(request(), synthesis_callback=callback)

    def test_callback_rejects_unretrieved_quote(self):
        data = request(texts=["Battery warranty lasts 10 years. Cats sleep."])
        text = data["documents"][0]["text"]
        start = text.index("Cats")
        def callback(_):
            return {"findings": [{"text": "Cats sleep.", "citations": [{
                "source_id": "0", "start": start, "end": len(text), "quote": "Cats sleep."
            }]}]}
        with self.assertRaisesRegex(ValidationError, "outside retrieved"):
            research(data, synthesis_callback=callback)

    def test_cli_example_and_invalid_json(self):
        root = Path(__file__).parent
        completed = subprocess.run(
            [sys.executable, str(root / "implementation.py"), str(root / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "evidence_found")
        self.assertTrue(result["conflicts"])
        invalid = subprocess.run([sys.executable, str(root / "implementation.py")],
                                 input="{", capture_output=True, text=True, check=False)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("error", json.loads(invalid.stderr))


if __name__ == "__main__":
    unittest.main()
