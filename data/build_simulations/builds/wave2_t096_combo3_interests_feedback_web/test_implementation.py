import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_pipeline(self):
        return app.run_pipeline(self.source)

    def test_preference_ranking_and_grounding(self):
        items = self.run_pipeline()["interests"]["recommendations"]
        self.assertEqual([(item["item_id"], item["score"]) for item in items],
                         [("bottle", 8), ("bag", 5)])
        self.assertEqual(items[0]["reasons"],
                         [{"tag": "cycling", "weight": 5}, {"tag": "outdoors", "weight": 3}])

    def test_exclusions_override_preferences(self):
        self.source["profile"]["limit"] = 100
        selected = self.run_pipeline()["feedback"]["selected_item_ids"]
        self.assertNotIn("leather-bag", selected)
        self.assertNotIn("excluded", selected)
        self.assertNotIn("mug", selected)

    def test_deterministic_tie_break(self):
        self.source["items"][0]["tags"] = ["cycling"]
        self.assertEqual(self.run_pipeline()["feedback"]["selected_item_ids"], ["bag", "bottle"])

    def test_deduplication_preserves_ids(self):
        feedback = self.run_pipeline()["feedback"]
        self.assertEqual(feedback["duplicate_count"], 1)
        quality = next(t for t in feedback["themes"] if t["theme"] == "quality")
        self.assertEqual(quality["count"], 2)
        self.assertEqual(quality["excerpts"][0]["feedback_ids"], ["f1", "f2"])
        self.assertEqual(quality["excerpts"][0]["text"], self.source["feedback"][0]["text"])

    def test_same_text_different_items_not_deduplicated(self):
        self.source["feedback"][2]["text"] = self.source["feedback"][0]["text"]
        self.assertEqual(self.run_pipeline()["feedback"]["duplicate_count"], 1)

    def test_cross_stage_exclusion_and_limit(self):
        self.source["profile"]["limit"] = 1
        result = self.run_pipeline()
        self.assertEqual(result["feedback"]["analyzed_ids"], ["f1", "f2"])
        self.assertEqual({q["theme"] for q in result["web"]["queries"]}, {"quality", "shipping"})
        for finding in result["web"]["findings"]:
            self.assertEqual(finding["feedback_ids"], ["f1", "f2"])

    def test_findings_preserve_page_provenance(self):
        result = self.run_pipeline()
        pages = {page["id"]: page for page in self.source["web"]["pages"]}
        self.assertTrue(result["web"]["findings"])
        for finding in result["web"]["findings"]:
            page = pages[finding["page_id"]]
            self.assertEqual(finding["url"], page["url"])
            self.assertEqual(finding["excerpt"], page["content"][finding["start"]:finding["end"]])
            self.assertTrue(set(finding["matched_terms"]) <= app.words(finding["excerpt"]))
            self.assertNotIn("f4", finding["feedback_ids"])
            self.assertNotIn("f5", finding["feedback_ids"])

    def test_web_limit_and_determinism(self):
        self.source["web"]["max_findings"] = 1
        first = self.run_pipeline()
        self.assertEqual(len(first["web"]["findings"]), 1)
        self.assertEqual(first, self.run_pipeline())

    def test_no_matching_pages(self):
        for page in self.source["web"]["pages"]:
            page["content"] = "Synthetic astronomy snapshot."
        self.assertEqual(self.run_pipeline()["web"]["findings"], [])

    def test_empty_preferences_propagate(self):
        self.source["profile"]["interests"] = {}
        result = self.run_pipeline()
        self.assertEqual(result["interests"]["recommendations"], [])
        self.assertEqual(result["feedback"]["themes"], [])
        self.assertEqual(result["web"], {"queries": [], "findings": []})

    def test_empty_feedback_and_pages(self):
        self.source["feedback"] = []
        self.source["web"]["pages"] = []
        result = self.run_pipeline()
        self.assertEqual(result["feedback"]["duplicate_count"], 0)
        self.assertEqual(result["web"]["findings"], [])

    def test_unclassified_feedback_is_not_invented_theme(self):
        self.source["feedback"] = [{"id": "x", "item_id": "bag", "text": "Lovely purple color."}]
        result = self.run_pipeline()
        self.assertEqual(result["feedback"]["themes"][0]["theme"], "other")
        self.assertEqual(result["web"]["queries"], [])

    def test_allowlist_enforced(self):
        for url in ["http://research.example/a", "https://evil.example/a",
                    "https://research.example.evil.example/a",
                    "https://sub.research.example/a", "https://user@research.example/a",
                    "https://research.example:444/a", "https://research.example/a#fragment",
                    "https://research.example\\@evil.example/a", "https://research.example/a b",
                    "https://research.example:bad/a"]:
            with self.subTest(url=url):
                source = copy.deepcopy(self.source)
                source["web"]["pages"][0]["url"] = url
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(source)

    def test_canonical_url_duplicate(self):
        self.source["web"]["pages"][1]["url"] = "https://RESEARCH.EXAMPLE:443/quality"
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_invalid_schema_types_and_fields(self):
        for value in [None, [], {}, {"schema_version": "2"}]:
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(value)
        for weight in [True, -1, 0, 1.5, "5"]:
            with self.subTest(weight=weight):
                self.source["profile"]["interests"]["cycling"] = weight
                with self.assertRaises(app.ValidationError):
                    self.run_pipeline()

    def test_duplicate_and_unknown_ids(self):
        for target in ["items", "feedback"]:
            source = copy.deepcopy(self.source)
            source[target].append(copy.deepcopy(source[target][0]))
            with self.subTest(target=target), self.assertRaises(app.ValidationError):
                app.run_pipeline(source)
        self.source["feedback"][0]["item_id"] = "missing"
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_input_not_mutated(self):
        before = copy.deepcopy(self.source)
        self.run_pipeline()
        self.assertEqual(self.source, before)

    def test_interests_handoff_rejects_tampering(self):
        previous = app.recommend(self.source)
        previous["recommendations"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.analyze_feedback(self.source, previous)

    def test_feedback_handoff_rejects_tampering(self):
        interests = app.recommend(self.source)
        previous = app.analyze_feedback(self.source, interests)
        previous["themes"][0]["excerpts"][0]["text"] = "Invented testimony."
        with self.assertRaises(app.ValidationError):
            app.research(self.source, previous, interests)

    def test_web_validator_rejects_ungrounded_excerpt(self):
        result = self.run_pipeline()
        result["web"]["findings"][0]["excerpt"] = "Invented finding."
        with self.assertRaises(app.ValidationError):
            app.validate("web", result["web"],
                         {"input": self.source, "interests": result["interests"],
                          "feedback": result["feedback"]})

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in [[], [str(ROOT / "missing.json")], ["one", "two"]]:
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, cwd=ROOT, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_malformed_utf8_json_duplicate_and_oversized(self):
        for raw in [b"{", b"\xff", b'{"x":1,"x":2}', b'{"x":NaN}',
                    b"[]" , b" " * (app.MAX_FILE_BYTES + 1)]:
            with self.subTest(prefix=repr(raw[:30])):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(raw)), redirect_stdout(output):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
