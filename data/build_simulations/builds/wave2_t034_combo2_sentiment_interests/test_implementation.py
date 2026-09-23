import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_transparent_sentiment(self):
        result = app.run_pipeline(self.source)
        first = result["insights"]["issues"][0]
        self.assertEqual(first["feedback_id"], "synthetic-feedback-01")
        self.assertEqual(first["sentiment_score"], -5)
        self.assertEqual(first["sentiment"], "negative")
        self.assertEqual(sum(e["contribution"] for e in first["evidence"]), -5)
        self.assertEqual(first["priority_score"], 425)

    def test_negation_and_punctuation(self):
        for message, score in [("not good", -1), ("not bad", 1),
                               ("not. good", 1), ("GREAT!", 2),
                               ("good bad", 0), ("ordinary", 0)]:
            with self.subTest(message=message):
                feedback = {"id": "f", "text": message, "severity": "low", "tags": []}
                self.assertEqual(app.analyze_feedback(feedback)["sentiment_score"], score)

    def test_severity_dominates_sentiment(self):
        self.source["feedback"] = [
            {"id": "f-low", "text": "unsafe " * 100, "severity": "low", "tags": []},
            {"id": "f-high", "text": "good", "severity": "high", "tags": []},
        ]
        issues = app.sentiment_stage(self.source)["insights"]["issues"]
        self.assertEqual([i["feedback_id"] for i in issues], ["f-high", "f-low"])
        self.assertEqual(issues[1]["priority_score"], 145)

    def test_cross_stage_changes_ranking(self):
        result = app.run_pipeline(self.source)
        recommendations = result["discovery"]["recommendations"]
        self.assertEqual(recommendations[0]["item_id"], "synthetic-item-book")
        outdoors = recommendations[1]
        self.assertEqual(outdoors["score"], 38)
        self.assertEqual(outdoors["explanation"]["feedback_concerns"][0]["feedback_id"],
                         "synthetic-feedback-01")
        self.source["feedback"][0]["text"] = "great"
        changed = app.run_pipeline(self.source)["discovery"]["recommendations"]
        self.assertEqual(changed[0]["item_id"], "synthetic-item-outdoors")
        self.assertEqual(changed[0]["score"], 50)

    def test_exclusions_override_high_preferences(self):
        discovery = app.run_pipeline(self.source)["discovery"]
        recommended = {item["item_id"] for item in discovery["recommendations"]}
        self.assertNotIn("synthetic-item-blocked", recommended)
        self.assertNotIn("synthetic-item-leather", recommended)
        self.assertEqual(len(discovery["exclusions"]), 2)
        self.assertTrue(discovery["exclusions"][0]["excluded_by_id"])
        self.assertEqual(discovery["exclusions"][1]["excluded_tags"], ["leather"])

    def test_empty_inputs_and_unknown_tags(self):
        self.source["feedback"] = []
        self.source["customer"]["interests"] = {"unknown": 3}
        result = app.run_pipeline(self.source)
        self.assertEqual(result["insights"], {"issues": [], "concerns": []})
        self.assertEqual(result["discovery"]["recommendations"], [])
        self.source["catalog"] = []
        self.assertEqual(app.run_pipeline(self.source)["discovery"]["exclusions"], [])

    def test_all_items_excluded(self):
        self.source["customer"]["excluded_item_ids"] = [
            item["id"] for item in self.source["catalog"]
        ]
        result = app.run_pipeline(self.source)["discovery"]
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(len(result["exclusions"]), len(self.source["catalog"]))

    def test_ties_are_id_ordered_and_top_k_enforced(self):
        self.source["feedback"] = []
        self.source["catalog"] = [
            {"id": "z", "title": "Synthetic Z", "tags": ["books"]},
            {"id": "a", "title": "Synthetic A", "tags": ["books"]},
        ]
        self.source["top_k"] = 1
        items = app.run_pipeline(self.source)["discovery"]["recommendations"]
        self.assertEqual([item["item_id"] for item in items], ["a"])

    def test_grounded_explanations_reconcile(self):
        result = app.run_pipeline(self.source)
        feedback_ids = {feedback["id"] for feedback in self.source["feedback"]}
        for item in result["discovery"]["recommendations"]:
            explanation = item["explanation"]
            self.assertEqual(item["score"], explanation["preference_score"] -
                             explanation["concern_penalty"])
            for match in explanation["preference_matches"]:
                self.assertEqual(match["weight"],
                                 self.source["customer"]["interests"][match["tag"]])
            for concern in explanation["feedback_concerns"]:
                self.assertIn(concern["feedback_id"], feedback_ids)

    def test_rejects_tampered_handoff(self):
        intermediate = app.sentiment_stage(self.source)
        intermediate["insights"]["concerns"][0]["penalty"] = 0
        with self.assertRaises(app.ValidationError):
            app.interests_stage(intermediate)

    def test_rejects_tampered_final_output(self):
        result = app.run_pipeline(self.source)
        result["discovery"]["recommendations"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.validate(result, "complete")

    def test_invalid_input_variants(self):
        variants = [
            ("top_k", True), ("top_k", 0), ("top_k", 21),
            ("schema_version", 2), ("synthetic", False),
            ("feedback", None), ("catalog", "not-an-array"),
        ]
        for field, value in variants:
            with self.subTest(field=field, value=value):
                source = copy.deepcopy(self.source)
                source[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(source)
        for source in [None, [], {}, {"extra": 1}]:
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(source)

    def test_nested_validation_and_duplicates(self):
        variants = []
        source = copy.deepcopy(self.source)
        source["feedback"].append(source["feedback"][0])
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["catalog"][0]["tags"] = ["books", "books"]
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["customer"]["interests"]["books"] = True
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["feedback"][0]["severity"] = "urgent"
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["customer"]["excluded_tags"] = ["Books"]
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["feedback"][0]["text"] = " "
        variants.append(source)
        for source in variants:
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(source)

    def test_deterministic_and_non_mutating(self):
        original = copy.deepcopy(self.source)
        first = app.run_pipeline(self.source)
        self.assertEqual(first, app.run_pipeline(self.source))
        self.assertEqual(self.source, original)
        first["input"]["customer"]["interests"]["books"] = 1
        self.assertEqual(self.source, original)

    def cli(self, *args):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        return completed.returncode, json.loads(completed.stdout)

    def test_cli_success(self):
        code, output = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(code, 0)
        self.assertEqual(output, app.run_pipeline(self.source))

    def test_cli_file_error_and_arguments(self):
        for args in [(), ("missing-synthetic-file.json",), ("one", "two"), (str(ROOT),)]:
            with self.subTest(args=args):
                code, output = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(output["status"], "error")

    def test_cli_json_and_schema_errors(self):
        for content in ['{', '{"x":1,"x":2}', 'NaN', '{}', '[1]', '{"top_k":true}']:
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_error(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("synthetic invalid UTF-8")):
            with contextlib.redirect_stdout(output):
                code = app.main(["synthetic-fixture.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
