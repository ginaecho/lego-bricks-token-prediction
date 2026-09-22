"""All test scenarios, documents, and requirements in this file are FICTIONAL."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


def fictional_data():
    return {
        "label": "FICTIONAL test scenario",
        "requirements": [{
            "id": "FIC-1", "description": "Fictional documented controls", "priority": 2,
            "evidence_rules": {"all_of": ["alpha", "beta"], "contradicts": ["waived"]},
        }],
        "documents": [{"id": "FIC-DOC", "text": "🙂 alpha beta"}],
    }


class FictionalEvidenceTests(unittest.TestCase):
    def test_supported_exact_unicode_spans(self):
        data = fictional_data()
        report = impl.analyze(data)
        finding = report["findings"][0]
        self.assertEqual(finding["status"], "supported")
        self.assertEqual(finding["citations"][0]["start"], 2)
        for citation in finding["citations"]:
            self.assertEqual(
                data["documents"][0]["text"][citation["start"]:citation["end"]],
                citation["quote"],
            )
        self.assertEqual(report["gaps"], [])
        self.assertIn("not proof of failure", report["scope"])
        self.assertIn("not legal advice or compliance certification", report["scope"])

    def test_no_sources_is_missing_not_failure(self):
        data = fictional_data()
        data["documents"] = []
        report = impl.analyze(data)
        self.assertEqual(report["findings"][0]["status"], "missing")
        self.assertEqual(report["findings"][0]["citations"], [])
        self.assertEqual(report["gaps"][0]["unmatched_all_of"], ["alpha", "beta"])
        self.assertIn("not proof of failure", report["gaps"][0]["action"])

    def test_partial(self):
        data = fictional_data()
        data["documents"][0]["text"] = "alpha"
        report = impl.analyze(data)
        self.assertEqual(report["findings"][0]["status"], "partial")
        self.assertEqual(report["gaps"][0]["unmatched_all_of"], ["beta"])

    def test_contradiction_across_documents(self):
        data = fictional_data()
        data["documents"].append({"id": "FIC-OTHER", "text": "waived"})
        report = impl.analyze(data)
        finding = report["findings"][0]
        self.assertEqual(finding["status"], "conflicting")
        self.assertEqual(finding["citations"][-1]["document_id"], "FIC-OTHER")
        self.assertEqual(report["gaps"][0]["citations"], finding["citations"])

    def test_contradiction_without_positive_evidence(self):
        data = fictional_data()
        data["documents"][0]["text"] = "waived"
        self.assertEqual(impl.analyze(data)["findings"][0]["status"], "conflicting")

    def test_all_and_any_are_conjunctive(self):
        data = fictional_data()
        data["requirements"][0]["evidence_rules"]["any_of"] = ["gamma", "delta"]
        report = impl.analyze(data)
        self.assertEqual(report["findings"][0]["status"], "partial")
        self.assertEqual(report["gaps"][0]["unmatched_any_of_options"], ["gamma", "delta"])
        data["documents"][0]["text"] += " delta"
        self.assertEqual(impl.analyze(data)["findings"][0]["status"], "supported")

    def test_any_only_and_case_sensitive(self):
        data = fictional_data()
        data["requirements"][0]["evidence_rules"] = {"any_of": ["ALPHA", "beta"]}
        self.assertEqual(impl.analyze(data)["findings"][0]["status"], "supported")
        data["requirements"][0]["evidence_rules"] = {"any_of": ["ALPHA"]}
        self.assertEqual(impl.analyze(data)["findings"][0]["status"], "missing")

    def test_duplicate_ids(self):
        for kind in ("requirements", "documents"):
            with self.subTest(kind=kind):
                data = fictional_data()
                data[kind].append(copy.deepcopy(data[kind][0]))
                with self.assertRaises(impl.ValidationError):
                    impl.analyze(data)

    def test_invalid_rules(self):
        for rules in (
            {}, [], {"all_of": []}, {"contradicts": ["waived"]},
            {"all_of": "alpha"}, {"all_of": [""]}, {"all_of": [None]},
            {"all_of": ["alpha", "alpha"]}, {"regex": ["alpha"]},
            {"all_of": ["alpha"], "contradicts": ["alpha"]},
            {"all_of": ["alpha"], "any_of": ["alpha"]},
        ):
            with self.subTest(rules=rules):
                data = fictional_data()
                data["requirements"][0]["evidence_rules"] = rules
                with self.assertRaises(impl.ValidationError):
                    impl.analyze(data)

    def test_invalid_shapes_and_priority(self):
        for value in (None, [], {}, {"requirements": [], "documents": None}):
            with self.subTest(value=value), self.assertRaises(impl.ValidationError):
                impl.analyze(value)
        for priority in (True, 0, 6, "1", 1.5):
            data = fictional_data()
            data["requirements"][0]["priority"] = priority
            with self.subTest(priority=priority), self.assertRaises(impl.ValidationError):
                impl.analyze(data)

    def test_overlapping_occurrences(self):
        data = fictional_data()
        data["requirements"][0]["evidence_rules"] = {"all_of": ["aa"]}
        data["documents"][0]["text"] = "aaa"
        citations = impl.analyze(data)["findings"][0]["citations"]
        self.assertEqual([c["start"] for c in citations], [0, 1])

    def test_empty_requirements_do_not_certify(self):
        data = fictional_data()
        data["requirements"] = []
        report = impl.analyze(data)
        self.assertEqual(report["summary"]["requirements_reviewed"], 0)
        self.assertIn("no assessment", report["summary"]["note"])

    def test_example_prioritization_and_traceability(self):
        data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        report = impl.analyze(data)
        self.assertEqual(report["summary"]["counts"], dict.fromkeys(impl.STATUSES, 1))
        self.assertEqual(report["summary"]["prioritized_gap_ids"],
                         ["gap:FIC-ACCESS", "gap:FIC-REVIEW", "gap:FIC-TRAINING"])
        by_id = {f["requirement_id"]: f for f in report["findings"]}
        for gap in report["gaps"]:
            self.assertEqual(gap["finding_status"], by_id[gap["requirement_id"]]["status"])
            self.assertEqual(gap["citations"], by_id[gap["requirement_id"]]["citations"])
        self.assertEqual(report, impl.analyze(data))

    def test_input_and_review_bundle_isolation(self):
        data = fictional_data()
        original = copy.deepcopy(data)

        def reviewer(bundle):
            bundle["documents"][0]["text"] = "changed"
            bundle["requirements"].clear()
            return bundle["findings"]

        report = impl.analyze(data, reviewer)
        self.assertEqual(data, original)
        self.assertTrue(report["summary"]["injected_reviewer_validated"])
        self.assertEqual(report["findings"][0]["status"], "supported")

    def test_reviewer_can_reorder_citations(self):
        def reviewer(bundle):
            bundle["findings"][0]["citations"].reverse()
            return bundle["findings"]
        self.assertTrue(impl.analyze(fictional_data(), reviewer)["summary"]
                        ["injected_reviewer_validated"])

    def test_bad_callback_shapes(self):
        for value in (None, {}, True, "approved", [], [{}]):
            with self.subTest(value=value), self.assertRaises(impl.ValidationError):
                impl.analyze(fictional_data(), lambda bundle: value)

    def test_bad_callback_ids_status_and_citations(self):
        def mutate_id(f):
            f[0]["requirement_id"] = "UNKNOWN"

        def duplicate(f):
            f.append(copy.deepcopy(f[0]))

        mutations = [
            mutate_id, duplicate,
            lambda f: f[0].update(status="certified"),
            lambda f: f[0].update(status="missing"),
            lambda f: f[0].update(citations=[]),
            lambda f: f[0]["citations"].append(copy.deepcopy(f[0]["citations"][0])),
            lambda f: f[0]["citations"][0].update(start=True),
            lambda f: f[0]["citations"][0].update(start=-1),
            lambda f: f[0]["citations"][0].update(end=9999),
            lambda f: f[0]["citations"][0].update(quote="fabricated"),
            lambda f: f[0]["citations"][0].update(document_id="UNKNOWN"),
            lambda f: f[0]["citations"][0].update(rule="contradicts"),
            lambda f: f[0]["citations"][0].update(term="fabricated"),
        ]
        for index, mutation in enumerate(mutations):
            def reviewer(bundle):
                mutation(bundle["findings"])
                return bundle["findings"]
            with self.subTest(index=index), self.assertRaises(impl.ValidationError):
                impl.analyze(fictional_data(), reviewer)

    def test_callback_exception_and_noncallable(self):
        def broken(bundle):
            raise RuntimeError("fictional callback failure")
        for callback in (broken, 42):
            with self.subTest(callback=callback), self.assertRaises(impl.ValidationError):
                impl.analyze(fictional_data(), callback)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", "implementation.py", "example_input.json"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["summary"]["requirements_reviewed"], 4)

    def test_cli_usage_and_missing_file(self):
        for args in ([], ["fictional-nonexistent-input.json"]):
            with redirect_stderr(io.StringIO()) as error, redirect_stdout(io.StringIO()) as out:
                self.assertEqual(impl.main(args), 2)
            self.assertIn("error", json.loads(error.getvalue()))
            self.assertEqual(out.getvalue(), "")

    def test_cli_invalid_json_and_duplicate_keys(self):
        for text in ("{", '{"requirements":[],"requirements":[],"documents":[]}',
                     '{"requirements":[],"documents":[],"label":NaN}'):
            with patch("builtins.open", mock_open(read_data=text)):
                with redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(impl.main(["fictional-input.json"]), 2)
            self.assertIn("error", json.loads(error.getvalue()))


if __name__ == "__main__":
    unittest.main()
