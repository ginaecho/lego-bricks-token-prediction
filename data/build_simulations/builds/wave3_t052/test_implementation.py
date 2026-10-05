"""Tests use only clearly labeled synthetic fixtures and never write files."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_insights_groups_and_sentiment(self):
        result = app.insights(self.raw)["data"]
        self.assertEqual(result["feedback"][0]["themes"], ["onboarding", "performance"])
        self.assertEqual(result["feedback"][0]["sentiment"], "negative")
        theme = next(t for t in result["themes"] if t["name"] == "onboarding")
        self.assertEqual(theme["count"], 2)
        self.assertEqual(theme["sentiment_counts"]["positive"], 1)
        self.assertEqual(theme["priority"], "review")

    def test_unmatched_theme(self):
        self.raw["feedback"][0]["text"] = "A purple widget."
        self.assertEqual(app.insights(self.raw)["data"]["feedback"][0]["themes"], ["other"])

    def test_keyword_is_not_substring(self):
        self.raw["feedback"][0]["text"] = "Breakfast is ready."
        row = app.insights(self.raw)["data"]["feedback"][0]
        self.assertEqual(row["sentiment"], "neutral")
        self.assertEqual(row["themes"], ["other"])

    def test_documents_extract_and_check(self):
        records = app.documents(app.insights(self.raw))["data"]["records"]
        self.assertEqual(records[0]["fields"]["next_step"], "Simplify setup guidance.")
        self.assertEqual(records[0]["check_status"], "valid")
        self.assertEqual(records[2]["issues"], ["missing:owner"])

    def test_duplicate_field_is_reviewed(self):
        self.raw["documents"][0]["text"] += "\nOWNER: Another team"
        row = app.documents(app.insights(self.raw))["data"]["records"][0]
        self.assertEqual(row["fields"]["owner"], "Experience team")
        self.assertIn("duplicate:owner", row["issues"])

    def test_empty_required_field(self):
        self.raw["documents"][0]["text"] = "Owner: \nFinding: setup"
        self.assertIn("missing:owner", app.run(self.raw)["data"]["records"][0]["issues"])

    def test_research_evidence_and_abstention(self):
        answers = app.run(self.raw)["data"]["answers"]
        self.assertEqual(answers[0]["evidence_status"], "evidence_found")
        self.assertEqual([e["document_id"] for e in answers[0]["evidence"]], ["d1"])
        self.assertEqual(answers[2]["evidence_status"], "insufficient_evidence")

    def test_cross_stage_propagation(self):
        result = app.run(self.raw)["data"]
        record = result["records"][0]
        evidence = result["answers"][0]["evidence"][0]
        self.assertEqual(record["themes"], ["onboarding", "performance"])
        self.assertEqual(evidence["themes"], record["themes"])
        self.assertEqual(evidence["feedback_ids"], ["f1", "f2"])
        self.assertEqual(evidence["insight_negative_count"], 1)
        self.assertTrue(result["answers"][0]["suggested_actions"])
        self.assertEqual(evidence["excerpt"], self.raw["documents"][0]["text"])

    def test_input_change_propagates(self):
        self.raw["feedback"][0]["text"] = "Setup is easy."
        evidence = app.run(self.raw)["data"]["answers"][0]["evidence"][0]
        self.assertEqual(evidence["themes"], ["onboarding"])
        self.assertEqual(evidence["insight_negative_count"], 0)

    def test_empty_collections(self):
        for name in ("feedback", "documents", "questions"):
            self.raw[name] = []
        result = app.run(self.raw)["data"]
        self.assertEqual(result["themes"], [])
        self.assertEqual(result["records"], [])
        self.assertEqual(result["answers"], [])

    def test_stopword_question_abstains(self):
        self.raw["questions"][0]["text"] = "What should we do?"
        self.assertEqual(app.run(self.raw)["data"]["answers"][0]["evidence"], [])

    def test_unknown_reference(self):
        self.raw["documents"][0]["feedback_ids"] = ["absent"]
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_duplicate_ids(self):
        self.raw["feedback"].append(copy.deepcopy(self.raw["feedback"][0]))
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_invalid_types_and_unknown_fields(self):
        for name, value in (("feedback", None), ("schema_version", 1), ("extra", True),
                            ("required_fields", [False]), ("theme_rules", {"other": ["x"]})):
            with self.subTest(name=name):
                raw = copy.deepcopy(self.raw)
                raw[name] = value
                with self.assertRaises(app.ValidationError):
                    app.run(raw)

    def test_insight_handoff_tampering(self):
        previous = app.insights(self.raw)
        previous["data"]["feedback"][0]["sentiment"] = "positive"
        with self.assertRaises(app.ValidationError):
            app.documents(previous)

    def test_document_handoff_tampering(self):
        previous = app.documents(app.insights(self.raw))
        previous["data"]["records"][2]["check_status"] = "valid"
        with self.assertRaises(app.ValidationError):
            app.research(previous)

    def test_wrong_stage(self):
        with self.assertRaises(app.ValidationError):
            app.research(app.insights(self.raw))

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.raw)
        self.assertEqual(app.run(self.raw), app.run(self.raw))
        self.assertEqual(self.raw, original)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["stage"], "research")
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], [str(ROOT / "nonexistent.json")], ["a", "b"]):
            with self.subTest(args=args):
                process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                          *args], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_malformed_and_invalid_json(self):
        for content in ("{", "[]", '{"x": NaN}', '{"x":1,"x":2}', '{"schema_version":"9"}'):
            with self.subTest(content=content):
                stream = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(stream):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_unicode(self):
        self.raw["feedback"][0]["text"] = "Setup is confusing — café 用户."
        self.assertEqual(app.run(self.raw)["status"], "ok")


if __name__ == "__main__":
    unittest.main()
