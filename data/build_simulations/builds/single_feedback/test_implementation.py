"""All feedback in this test suite is synthetic."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, analyze, normalize


ROOT = Path(__file__).resolve().parent


class FeedbackTests(unittest.TestCase):
    def payload(self, text="fast"):
        return {"records": [{"id": "synthetic-a", "text": text}],
                "themes": [{"id": "speed", "keywords": ["fast", "slow"]}]}

    def proposal(self, **changes):
        value = {"record_id": "synthetic-a", "theme_id": "speed", "start": 0, "end": 4}
        value.update(changes)
        return {"assignments": [value], "syntheses": []}

    def cli(self, raw):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), "-"],
            input=raw, text=True, capture_output=True, check=False)

    def test_normalized_deduplication_preserves_originals(self):
        value = self.payload("  ＦＡＳＴ  response ")
        value["records"].append({"id": "synthetic-b", "text": "fast response"})
        result = analyze(value)
        theme = result["themes"][0]
        self.assertEqual(result["counts"]["duplicate_records"], 1)
        self.assertEqual(theme["distinct_feedback_count"], 1)
        self.assertEqual(theme["supporting_record_count"], 2)
        self.assertEqual(theme["evidence"][0]["excerpt"], "  ＦＡＳＴ  response ")
        self.assertEqual(normalize("Straße\tFAST"), "strasse fast")

    def test_conflicting_views_are_preserved(self):
        value = self.payload()
        value["records"].append({"id": "synthetic-b", "text": "slow"})
        result = analyze(value)["themes"][0]
        self.assertEqual(result["distinct_feedback_count"], 2)
        self.assertEqual([e["excerpt"] for e in result["evidence"]], ["fast", "slow"])

    def test_empty_feedback(self):
        for records in ([], [{"id": "synthetic-a", "text": " \n\t"}]):
            result = analyze({"records": records})
            self.assertEqual(result["status"], "empty_feedback")
            self.assertEqual(result["counts"]["distinct_feedback"], 0)
            self.assertEqual(result["counts"]["empty_records"], len(records))

    def test_unmatched_and_unconfigured_themes(self):
        value = self.payload("purple icon")
        result = analyze(value)
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["themes"][0]["status"], "no_match")
        self.assertEqual(result["unmatched"][0]["evidence"][0]["excerpt"], "purple icon")
        value["themes"] = []
        self.assertEqual(analyze(value)["status"], "no_themes")

    def test_whole_phrase_matching(self):
        value = self.payload("breakfast is FAST\tRESPONSE")
        value["themes"] = [{"id": "phrase", "keywords": ["fast response"]},
                           {"id": "partial", "keywords": ["break"]}]
        themes = {t["id"]: t for t in analyze(value)["themes"]}
        self.assertEqual(themes["phrase"]["distinct_feedback_count"], 1)
        self.assertEqual(themes["partial"]["distinct_feedback_count"], 0)

    def test_multitheme_counts_are_not_exclusive(self):
        result = analyze({"records": [{"id": "synthetic-a", "text": "fast and easy"}]})
        self.assertEqual(sum(t["distinct_feedback_count"] for t in result["themes"]), 2)
        self.assertEqual(result["counts"]["matched_feedback"], 1)

    def test_input_permutations_are_deterministic(self):
        value = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        baseline = analyze(value)
        value["records"].reverse()
        value["themes"].reverse()
        for theme in value["themes"]:
            theme["keywords"].reverse()
        self.assertEqual(analyze(value), baseline)

    def test_strict_record_validation(self):
        invalid = [None, [], {}, {"records": None}, {"records": [None]},
                   {"records": [{"id": "a", "text": 1}]},
                   {"records": [{"id": " ", "text": "x"}]},
                   {"records": [{"id": "a", "text": "x", "extra": 1}]},
                   {"records": [{"id": "a", "text": "\ud800"}]},
                   {"records": [], "extra": 1}]
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaises(ValidationError):
                analyze(value)

    def test_duplicate_record_ids_rejected(self):
        value = self.payload()
        value["records"] *= 2
        with self.assertRaises(ValidationError):
            analyze(value)

    def test_strict_theme_validation(self):
        invalid = [None, [None], [{"id": "x", "keywords": []}],
                   [{"id": "x", "keywords": [" "]}],
                   [{"id": "x", "keywords": [1]}],
                   [{"id": "x", "keywords": ["a"]}] * 2]
        for themes in invalid:
            value = self.payload()
            value["themes"] = themes
            with self.subTest(themes=themes), self.assertRaises(ValidationError):
                analyze(value)

    def test_limits(self):
        value = self.payload("x" * 100001)
        with self.assertRaises(ValidationError):
            analyze(value)
        value = {"records": [{"id": f"synthetic-{i}", "text": "x" * 100000}
                             for i in range(11)]}
        with self.assertRaises(ValidationError):
            analyze(value)

    def test_callback_assignment_and_extractive_synthesis(self):
        value = self.payload("nice UI")
        proposal = self.proposal(start=0, end=7)
        proposal["syntheses"] = [{
            "theme_id": "speed", "text": "nice UI",
            "supports": [{"record_id": "synthetic-a", "start": 0, "end": 7}],
        }]
        result = analyze(value, lambda _: proposal)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["themes"][0]["syntheses"][0]["text"], "nice UI")
        self.assertEqual(result["themes"][0]["evidence"][0]["excerpt"], "nice UI")

    def test_callback_unknown_ids_and_themes(self):
        for changes in ({"record_id": "missing"}, {"theme_id": "unknown"},
                        {"record_id": []}, {"theme_id": []}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                analyze(self.payload(), lambda _: self.proposal(**changes))

    def test_callback_invalid_offsets(self):
        for changes in ({"start": -1}, {"end": 5}, {"end": 0},
                        {"start": True}, {"end": 4.0}, {"start": "0"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                analyze(self.payload(), lambda _: self.proposal(**changes))

    def test_callback_empty_source_rejected(self):
        with self.assertRaises(ValidationError):
            analyze(self.payload("    "), lambda _: self.proposal())

    def test_callback_cannot_mutate_input(self):
        value = self.payload()
        before = copy.deepcopy(value)
        def callback(context):
            context["records"][0]["text"] = "changed"
            context["themes"][0]["keywords"].append("changed")
            return {"assignments": [], "syntheses": []}
        self.assertEqual(analyze(value, callback), analyze(before))
        self.assertEqual(value, before)

    def test_callback_malformed_and_failure(self):
        for proposal in (None, [], {}, {"assignments": [], "syntheses": [], "extra": 1},
                         {"assignments": None, "syntheses": []}):
            with self.subTest(proposal=proposal), self.assertRaises(ValidationError):
                analyze(self.payload(), lambda _: proposal)
        def failing(_):
            raise RuntimeError("synthetic failure")
        with self.assertRaisesRegex(ValidationError, "callback failed"):
            analyze(self.payload(), failing)
        with self.assertRaises(ValidationError):
            analyze(self.payload(), 3)

    def test_fabricated_synthesis_rejected(self):
        proposal = {"assignments": [], "syntheses": [{
            "theme_id": "speed", "text": "Customers saved money.",
            "supports": [{"record_id": "synthetic-a", "start": 0, "end": 4}],
        }]}
        with self.assertRaisesRegex(ValidationError, "exactly"):
            analyze(self.payload(), lambda _: proposal)

    def test_synthesis_needs_assigned_sources(self):
        proposal = {"assignments": [], "syntheses": [{
            "theme_id": "speed", "text": "nice",
            "supports": [{"record_id": "synthetic-a", "start": 0, "end": 4}],
        }]}
        with self.assertRaisesRegex(ValidationError, "assignment"):
            analyze(self.payload("nice"), lambda _: proposal)
        proposal["syntheses"][0]["supports"] = []
        with self.assertRaises(ValidationError):
            analyze(self.payload(), lambda _: proposal)

    def test_callback_assignment_deduplication(self):
        proposal = self.proposal()
        proposal["assignments"] *= 2
        result = analyze(self.payload(), lambda _: proposal)
        self.assertEqual(len(result["themes"][0]["evidence"]), 1)

    def test_every_evidence_span_is_exact(self):
        value = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        records = {r["id"]: r["text"] for r in value["records"]}
        result = analyze(value)
        evidence = [e for t in result["themes"] for e in t["evidence"]]
        evidence += [e for group in result["unmatched"] for e in group["evidence"]]
        for source in evidence:
            self.assertEqual(source["excerpt"],
                             records[source["record_id"]][source["start"]:source["end"]])
        self.assertEqual(result["counts"], {
            "input_records": 7, "empty_records": 1, "distinct_feedback": 5,
            "duplicate_records": 1, "matched_feedback": 4, "unmatched_feedback": 1})

    def test_cli_success_and_repeatability(self):
        first = self.cli(json.dumps(self.payload()))
        second = self.cli(json.dumps(self.payload()))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(json.loads(first.stdout), analyze(self.payload()))
        self.assertEqual(first.stderr, "")

    def test_cli_rejects_nonstandard_and_invalid_json(self):
        for raw in ('{"records":[],"records":[]}', '{"records":NaN}',
                    '{"records":Infinity}', '{"records":', '[]'):
            with self.subTest(raw=raw):
                result = self.cli(raw)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error", json.loads(result.stderr))

    def test_cli_file_and_missing_file(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json"), "--pretty"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["counts"]["input_records"], 7)
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "does-not-exist.json")],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("error", json.loads(result.stderr))


if __name__ == "__main__":
    unittest.main()
