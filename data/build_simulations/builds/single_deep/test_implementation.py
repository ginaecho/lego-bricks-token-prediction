"""All source documents and claims in these tests are explicitly SYNTHETIC."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from implementation import ValidationError, analyze, load_json, normalize


ROOT = Path(__file__).resolve().parent


def synthetic_document(identifier, claims):
    """Create synthetic (claim id, subject, value) assertions, not measurements."""
    assertions = [
        {"id": cid, "subject": subject, "value": value,
         "evidence": f"{subject}: {value}."}
        for cid, subject, value in claims
    ]
    return {
        "id": identifier,
        "title": "EXPLICITLY SYNTHETIC test fixture",
        "text": "SYNTHETIC TEST DATA. " + " ".join(item["evidence"] for item in assertions),
        "claims": assertions,
    }


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def valid_response(self):
        return {"statements": copy.deepcopy(analyze(self.data)["synthesis"]["statements"])}

    def run_cli(self, *arguments, stdin=None):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
            input=stdin, capture_output=True, text=True, encoding="utf-8",
            cwd=ROOT, timeout=15,
        )

    def test_example_detects_disagreement_and_agreement(self):
        result = analyze(self.data)
        groups = {item["subject"]: item for item in result["grouped_claims"]}
        self.assertEqual(groups["battery life"]["status"], "disagreement")
        self.assertEqual(groups["water resistance"]["status"], "agreement")
        self.assertEqual([item["value"] for item in groups["battery life"]["alternatives"]],
                         ["10 hours", "8 hours"])
        self.assertEqual(len(result["evidence"]), 5)
        self.assertNotIn("mass", groups)

    def test_majority_does_not_discard_conflict(self):
        result = analyze(self.data)
        self.assertIn("battery-b", {record["claim_id"] for record in result["evidence"]})
        self.assertIn("8 hours [synthetic-lab-b/battery-b]", result["synthesis"]["text"])
        self.assertIn("Sources disagree (unresolved)", result["synthesis"]["text"])

    def test_provenance_offsets_round_trip(self):
        result = analyze(self.data)
        documents = {document["id"]: document for document in self.data["documents"]}
        for record in result["evidence"]:
            span = record["source_span"]
            text = documents[record["document_id"]]["text"]
            self.assertEqual(text[span["start"]:span["end"]], record["quote"])
            self.assertEqual(record["document_title"], documents[record["document_id"]]["title"])

    def test_retrieval_expands_subject_to_unmatched_conflicts(self):
        self.data["question"] = "IP67"
        self.data["documents"][1]["text"] = self.data["documents"][1]["text"].replace("IP67", "IP68")
        claim = self.data["documents"][1]["claims"][1]
        claim["value"], claim["evidence"] = "IP68", "Water resistance: IP68."
        result = analyze(self.data)
        self.assertEqual(result["retrieval"]["matched_claim_count"], 1)
        self.assertEqual(result["retrieval"]["selected_claim_count"], 2)
        expanded = next(item for item in result["evidence"] if item["claim_id"] == "water-b")
        self.assertEqual(expanded["retrieval"]["score"], 0)
        self.assertEqual(expanded["retrieval"]["reason"], "same_subject_expansion")
        self.assertEqual(result["grouped_claims"][0]["status"], "disagreement")

    def test_numeric_query_is_not_dropped(self):
        self.data["question"] = "200"
        result = analyze(self.data)
        self.assertEqual([item["claim_id"] for item in result["evidence"]], ["mass-a"])
        self.assertEqual(result["retrieval"]["question_terms"], ["200"])

    def test_single_source_requires_corroboration(self):
        self.data["question"] = "Mass?"
        result = analyze(self.data)
        self.assertEqual(result["grouped_claims"][0]["status"], "single_source")
        self.assertEqual(result["unresolved_questions"][0]["kind"], "needs_corroboration")

    def test_same_document_repeated_assertions_do_not_create_agreement(self):
        doc = synthetic_document("synthetic", [("one", "Mass", "200 grams")])
        doc["text"] += " A second assertion says Mass: 200 grams."
        doc["claims"].append({
            "id": "two", "subject": "Mass", "value": "200 grams",
            "evidence": "A second assertion says Mass: 200 grams.",
        })
        result = analyze({"question": "Mass?", "documents": [doc]})
        self.assertEqual(result["grouped_claims"][0]["status"], "single_source")
        self.assertEqual(len(result["evidence"]), 2)

    def test_internal_document_conflict_is_preserved(self):
        doc = synthetic_document("synthetic", [
            ("one", "Mass", "200 grams"), ("two", "Mass", "300 grams"),
        ])
        result = analyze({"question": "Mass?", "documents": [doc]})
        self.assertEqual(result["grouped_claims"][0]["status"], "disagreement")

    def test_case_whitespace_and_unicode_normalization(self):
        self.assertEqual(normalize(" ＩＰ６７  \n"), "ip67")
        docs = [
            synthetic_document("synthetic-a", [("a", "Ｍａｓｓ", "２００ GRAMS")]),
            synthetic_document("synthetic-b", [("b", "mass", "200  grams")]),
        ]
        result = analyze({"question": "Mass", "documents": docs})
        self.assertEqual(result["grouped_claims"][0]["status"], "agreement")
        self.assertEqual(result["grouped_claims"][0]["alternatives"][0]["value"], "200 grams")

    def test_normalization_preserves_signs_decimals_and_negation(self):
        docs = [
            synthetic_document("synthetic-a", [("a", "Offset", "-1.0")]),
            synthetic_document("synthetic-b", [("b", "Offset", "1.0")]),
            synthetic_document("synthetic-c", [("c", "Offset", "not 1.0")]),
        ]
        result = analyze({"question": "Offset?", "documents": docs})
        self.assertEqual(len(result["grouped_claims"][0]["alternatives"]), 3)

    def test_deterministic_across_repeated_calls(self):
        self.assertEqual(analyze(self.data), analyze(self.data))

    def test_deterministic_under_document_and_claim_reordering(self):
        baseline = analyze(self.data)
        self.data["documents"].reverse()
        for document in self.data["documents"]:
            document["claims"].reverse()
        self.assertEqual(analyze(self.data), baseline)

    def test_input_is_not_mutated(self):
        original = copy.deepcopy(self.data)
        analyze(self.data)
        self.assertEqual(self.data, original)

    def test_empty_documents_are_valid_unanswered_input(self):
        result = analyze({"question": "Battery life?", "documents": []})
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["synthesis"]["statements"], [])
        self.assertEqual(result["synthesis_plan"][0]["action"], "request_evidence")
        self.assertEqual(result["unresolved_questions"][0]["kind"], "no_relevant_evidence")

    def test_empty_document_text_without_claims_is_valid(self):
        doc = synthetic_document("synthetic-empty", [])
        doc["text"] = ""
        result = analyze({"question": "Battery life?", "documents": [doc]})
        self.assertEqual(result["retrieval"]["candidate_claim_count"], 0)

    def test_no_relevance_produces_no_answer(self):
        self.data["question"] = "Orbital inclination?"
        result = analyze(self.data)
        self.assertEqual(result["evidence"], [])
        self.assertIn("remains unanswered", result["synthesis"]["text"])
        self.assertEqual(result["retrieval"]["candidate_claim_count"], 6)

    def test_stopword_only_question_does_not_match(self):
        self.data["question"] = "What is it?"
        result = analyze(self.data)
        self.assertEqual(result["retrieval"]["question_terms"], [])
        self.assertEqual(result["evidence"], [])

    def test_plan_and_unresolved_questions_are_evidence_linked(self):
        result = analyze(self.data)
        actions = {step["action"] for step in result["synthesis_plan"]}
        self.assertEqual(actions, {"report_disagreement", "report_agreement"})
        conflict = next(item for item in result["unresolved_questions"]
                        if item["kind"] == "conflicting_values")
        self.assertEqual(conflict["claim_ids"], ["battery-a", "battery-b", "battery-c"])
        self.assertEqual(result["unresolved_questions"][-1]["kind"], "answer_coverage_unverified")

    def test_duplicate_document_ids_rejected(self):
        self.data["documents"][1]["id"] = self.data["documents"][0]["id"]
        with self.assertRaisesRegex(ValidationError, "duplicate document"):
            analyze(self.data)

    def test_duplicate_claim_ids_across_documents_rejected(self):
        self.data["documents"][1]["claims"][0]["id"] = "battery-a"
        with self.assertRaisesRegex(ValidationError, "duplicate claim"):
            analyze(self.data)

    def test_duplicate_source_assertions_rejected(self):
        claim = copy.deepcopy(self.data["documents"][0]["claims"][0])
        claim["id"] = "duplicate-with-new-id"
        claim["subject"] = "BATTERY  LIFE"
        self.data["documents"][0]["claims"].append(claim)
        with self.assertRaisesRegex(ValidationError, "duplicate source"):
            analyze(self.data)

    def test_malformed_top_level_data_rejected(self):
        for value in (None, [], {}, {"question": "x"}, {"question": 3, "documents": []},
                      {"question": " ", "documents": []}, {"question": "x", "documents": {}},
                      {"question": "x", "documents": [], "extra": True}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                analyze(value)

    def test_malformed_documents_and_claims_rejected(self):
        mutations = [
            lambda data: data["documents"].__setitem__(0, None),
            lambda data: data["documents"][0].__setitem__("claims", {}),
            lambda data: data["documents"][0].__setitem__("text", 3),
            lambda data: data["documents"][0].__setitem__("title", ""),
            lambda data: data["documents"][0].__setitem__("id", " bad id "),
            lambda data: data["documents"][0]["claims"].__setitem__(0, []),
            lambda data: data["documents"][0]["claims"][0].__setitem__("subject", None),
            lambda data: data["documents"][0]["claims"][0].__setitem__("value", True),
            lambda data: data["documents"][0]["claims"][0].__setitem__("evidence", ""),
            lambda data: data["documents"][0]["claims"][0].__setitem__("id", {}),
            lambda data: data["documents"][0]["claims"][0].__setitem__("extra", "x"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                data = copy.deepcopy(self.data)
                mutate(data)
                analyze(data)

    def test_invalid_unicode_rejected(self):
        self.data["question"] = "\ud800"
        with self.assertRaisesRegex(ValidationError, "valid Unicode"):
            analyze(self.data)

    def test_unanchored_excerpt_rejected(self):
        self.data["documents"][0]["claims"][0]["evidence"] = "Invented evidence."
        with self.assertRaisesRegex(ValidationError, "exact source excerpt"):
            analyze(self.data)

    def test_unsupported_subject_and_value_rejected(self):
        for field, value in (("subject", "Screen size"), ("value", "99 hours")):
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "lexical"):
                data = copy.deepcopy(self.data)
                data["documents"][0]["claims"][0][field] = value
                analyze(data)

    def test_value_substring_is_not_support(self):
        self.data["documents"][0]["claims"][0]["value"] = "0 hours"
        with self.assertRaisesRegex(ValidationError, "lexical"):
            analyze(self.data)

    def test_duplicate_json_keys_rejected_including_nested(self):
        for text in ('{"question":"a","question":"b","documents":[]}',
                     '{"x":{"id":"a","id":"b"}}'):
            with self.subTest(text=text), self.assertRaisesRegex(ValidationError, "duplicate JSON"):
                load_json(text)

    def test_invalid_json_and_nonstandard_numbers_rejected(self):
        for text in ("{", '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', "",
                     '{"x":' + "9" * 5000 + "}"):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                load_json(text)

    def test_valid_injected_callback_is_called_and_canonicalized(self):
        response = self.valid_response()
        response["statements"].reverse()
        for statement in response["statements"]:
            statement["subject"] = statement["subject"].upper()
            statement["claim_ids"].reverse()
            statement["citations"].reverse()
        calls = []

        def callback(request):
            calls.append(request)
            return response

        result = analyze(self.data, callback)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["question"], self.data["question"])
        self.assertEqual(calls[0]["evidence"], result["evidence"])
        self.assertEqual(result["synthesis"]["mode"], "validated_callback")
        self.assertEqual(result["synthesis"]["statements"], self.valid_response()["statements"])
        self.assertEqual(result["synthesis"]["text"], analyze(self.data)["synthesis"]["text"])

    def test_callback_cannot_mutate_analysis(self):
        response = self.valid_response()

        def callback(request):
            request["evidence"].clear()
            request["grouped_claims"].clear()
            request["synthesis_plan"].clear()
            return response

        result = analyze(self.data, callback)
        self.assertEqual(len(result["evidence"]), 5)
        self.assertEqual(len(result["grouped_claims"]), 2)
        self.assertEqual(len(result["synthesis_plan"]), 2)

    def test_callback_skipped_without_relevant_evidence(self):
        self.data["question"] = "Orbital inclination?"

        def callback(_):
            self.fail("Callback must not run without evidence")

        self.assertEqual(analyze(self.data, callback)["synthesis"]["mode"], "deterministic")

    def test_unknown_callback_claim_rejected(self):
        response = self.valid_response()
        response["statements"][0]["claim_ids"] = ["invented"]
        with self.assertRaisesRegex(ValidationError, "unknown or unretrieved"):
            analyze(self.data, lambda _: response)

    def test_known_but_unretrieved_callback_claim_rejected(self):
        response = self.valid_response()
        response["statements"].append({
            "subject": "mass", "value": "200 grams", "claim_ids": ["mass-a"],
            "citations": [{"document_id": "synthetic-lab-a", "claim_id": "mass-a"}],
        })
        with self.assertRaisesRegex(ValidationError, "unknown or unretrieved"):
            analyze(self.data, lambda _: response)

    def test_callback_cannot_change_known_assertion(self):
        for field, value in (("subject", "Mass"), ("value", "99 hours")):
            response = self.valid_response()
            response["statements"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "unsupported"):
                analyze(self.data, lambda _: response)

    def test_callback_cannot_cite_wrong_document(self):
        response = self.valid_response()
        response["statements"][0]["citations"][0]["document_id"] = "synthetic-lab-b"
        with self.assertRaisesRegex(ValidationError, "source"):
            analyze(self.data, lambda _: response)

    def test_callback_cannot_cite_wrong_claim(self):
        response = self.valid_response()
        response["statements"][0]["citations"][0]["claim_id"] = "water-a"
        with self.assertRaisesRegex(ValidationError, "source"):
            analyze(self.data, lambda _: response)

    def test_callback_cannot_omit_conflicting_evidence(self):
        response = self.valid_response()
        response["statements"] = [item for item in response["statements"] if item["value"] != "8 hours"]
        with self.assertRaisesRegex(ValidationError, "preserve every"):
            analyze(self.data, lambda _: response)

    def test_callback_requires_every_claim_citation(self):
        response = self.valid_response()
        response["statements"][0]["citations"].pop()
        with self.assertRaisesRegex(ValidationError, "own citation"):
            analyze(self.data, lambda _: response)

    def test_callback_duplicate_citation_rejected(self):
        response = self.valid_response()
        response["statements"][0]["citations"].append(response["statements"][0]["citations"][0])
        with self.assertRaisesRegex(ValidationError, "duplicate synthesis citation"):
            analyze(self.data, lambda _: response)

    def test_callback_duplicate_claim_rejected(self):
        response = self.valid_response()
        response["statements"][0]["claim_ids"].append(response["statements"][0]["claim_ids"][0])
        with self.assertRaisesRegex(ValidationError, "duplicate synthesis claim"):
            analyze(self.data, lambda _: response)

    def test_callback_duplicate_statement_rejected(self):
        response = self.valid_response()
        response["statements"].append(copy.deepcopy(response["statements"][0]))
        with self.assertRaisesRegex(ValidationError, "duplicate synthesis subject"):
            analyze(self.data, lambda _: response)

    def test_callback_cannot_add_unvalidated_free_prose(self):
        response = self.valid_response()
        response["statements"][0]["text"] = "An unsupported fabricated conclusion."
        with self.assertRaisesRegex(ValidationError, "exactly"):
            analyze(self.data, lambda _: response)

    def test_callback_malformed_responses_rejected(self):
        responses = [None, [], {}, {"statements": None}, {"statements": [None]}]
        for field, value in (("claim_ids", []), ("claim_ids", {}), ("claim_ids", [[]]),
                             ("citations", None), ("citations", [{}]), ("subject", None)):
            response = self.valid_response()
            response["statements"][0][field] = value
            responses.append(response)
        for index, response in enumerate(responses):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                analyze(self.data, lambda _: response)

    def test_noncallable_callback_rejected(self):
        with self.assertRaisesRegex(ValidationError, "callable"):
            analyze(self.data, "not a function")

    def test_callback_failure_is_explicit(self):
        def callback(_):
            raise RuntimeError("synthetic failure")

        with self.assertRaisesRegex(ValidationError, "callback failed"):
            analyze(self.data, callback)

    def test_cli_example_file_succeeds(self):
        process = self.run_cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), analyze(self.data))

    def test_cli_stdin_succeeds(self):
        process = self.run_cli("-", stdin=json.dumps(self.data))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), analyze(self.data))

    def test_cli_output_is_byte_deterministic(self):
        first, second = self.run_cli("example_input.json"), self.run_cli("example_input.json")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout.encode(), second.stdout.encode())

    def test_cli_invalid_input_returns_json_error(self):
        for text in ('{"question":"x","question":"y","documents":[]}', "{", '{"question":""}',
                     '{"question":' + "9" * 5000 + ',"documents":[]}'):
            with self.subTest(text=text):
                process = self.run_cli("-", stdin=text)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stdout, "")
                self.assertIn("error", json.loads(process.stderr))

    def test_cli_missing_file_returns_json_error(self):
        process = self.run_cli("nonexistent-synthetic-input.json")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, "")
        self.assertIn("error", json.loads(process.stderr))


if __name__ == "__main__":
    unittest.main()
