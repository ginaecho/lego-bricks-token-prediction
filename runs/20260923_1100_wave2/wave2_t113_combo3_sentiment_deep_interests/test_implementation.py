import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def result(self):
        return app.run_pipeline(self.data)

    def topic(self, name):
        return next(t for t in self.result()["research"]["topics"] if t["topic"] == name)

    def test_example_complete_and_valid(self):
        result = self.result()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(app.validate("result", result, self.data), result)
        self.assertIn("SYNTHETIC", result["fixture_label"])
        self.assertEqual([i["item_id"] for i in result["recommendations"]["ranked"]],
                         ["latch-review", "dispatch-guide"])

    def test_transparent_negative_scoring(self):
        issue = next(i for i in self.result()["sentiment"]["issues"] if i["id"] == "f-delivery")
        self.assertEqual(issue["score"], -4)
        self.assertEqual(issue["priority"], 64)
        self.assertEqual(issue["label"], "negative")
        self.assertEqual([m["contribution"] for m in issue["matches"]], [-1, -3])

    def test_negation_and_punctuation(self):
        self.data["feedback"][0]["text"] = "NOT bad. helpful"
        issue = next(i for i in self.result()["sentiment"]["issues"] if i["id"] == "f-delivery")
        self.assertEqual(issue["score"], 2)
        self.assertEqual([m["negated"] for m in issue["matches"]], [True, False])

    def test_double_negation(self):
        self.data["feedback"][0]["text"] = "not never good"
        issue = next(i for i in self.result()["sentiment"]["issues"] if i["id"] == "f-delivery")
        self.assertEqual(issue["score"], 1)

    def test_unknown_words_neutral(self):
        self.data["feedback"][0]["text"] = "Parcel arrived Tuesday."
        issue = next(i for i in self.result()["sentiment"]["issues"] if i["id"] == "f-delivery")
        self.assertEqual((issue["score"], issue["matches"], issue["label"]), (0, [], "neutral"))

    def test_severity_outweighs_sentiment_and_clamps(self):
        self.data["feedback"][0]["text"] = "terrible " * 30
        self.data["feedback"][1]["text"] = "excellent " * 30
        issues = self.result()["sentiment"]["issues"]
        self.assertEqual(issues[0]["id"], "f-safety")
        self.assertEqual(issues[0]["score"], 10)
        self.assertEqual(issues[1]["score"], -10)
        self.assertEqual(issues[1]["priority"], 70)

    def test_multi_document_disagreement(self):
        delivery = self.topic("delivery")
        self.assertEqual(delivery["document_ids"], ["doc-a", "doc-b"])
        self.assertEqual(delivery["findings"][0]["status"], "contested")
        self.assertEqual(delivery["disagreements"], ["Scheduled dispatch reduces delays"])
        self.assertTrue(any("disagreement" in q for q in delivery["unresolved_questions"]))
        self.assertEqual(delivery["issue_ids"], ["f-delivery", "f-delivery-2"])

    def test_uncertainty_is_not_consensus(self):
        safety = self.topic("safety")
        self.assertEqual(safety["findings"][0]["status"], "uncertain")
        self.assertTrue(safety["unresolved_questions"])
        self.assertEqual(safety["disagreements"], [])

    def test_uncovered_topic_and_item(self):
        support = self.topic("support")
        self.assertEqual(support["findings"], [])
        self.assertIn("What evidence", support["unresolved_questions"][0])
        self.assertNotIn("help-guide", [i["item_id"] for i in self.result()["recommendations"]["ranked"]])

    def test_single_source_corroboration_question(self):
        self.data["documents"].pop()
        delivery = self.topic("delivery")
        self.assertEqual(delivery["findings"][0]["status"], "supported")
        self.assertIn("independent document", delivery["unresolved_questions"][0])

    def test_agreement_with_multiple_sources(self):
        self.data["documents"][1]["evidence"][0]["stance"] = "support"
        delivery = self.topic("delivery")
        self.assertEqual(delivery["findings"][0]["status"], "supported")
        self.assertEqual(delivery["unresolved_questions"], [])

    def test_refuted_claim_caution(self):
        self.data["documents"][0]["evidence"][0]["stance"] = "refute"
        result = self.result()
        dispatch = next(i for i in result["recommendations"]["ranked"] if i["item_id"] == "dispatch-guide")
        self.assertTrue(any("Refuted claim" in c for c in dispatch["cautions"]))

    def test_grounded_explanations_and_cautions(self):
        result = self.result()
        dispatch = next(i for i in result["recommendations"]["ranked"] if i["item_id"] == "dispatch-guide")
        self.assertEqual(dispatch["document_ids"], ["doc-a", "doc-b"])
        self.assertEqual(dispatch["issue_ids"], ["f-delivery", "f-delivery-2"])
        self.assertEqual(dispatch["interest_score"], 300)
        self.assertEqual(dispatch["research_score"], 32)
        self.assertEqual(dispatch["score"], 332)
        self.assertIn("doc-b", dispatch["explanation"])
        self.assertIn("not proof of item effectiveness", dispatch["explanation"])
        self.assertTrue(dispatch["cautions"])

    def test_exclusions_override_preferences(self):
        recs = self.result()["recommendations"]
        self.assertEqual(recs["excluded_item_ids"], ["blocked-guide", "sponsored-review"])
        self.data["preferences"]["excluded_tags"] = ["safety", "operations"]
        self.assertEqual(self.result()["recommendations"]["ranked"], [])

    def test_preference_ranking_changes(self):
        self.data["preferences"]["interests"][0]["weight"] = 5
        self.assertEqual(self.result()["recommendations"]["ranked"][0]["item_id"], "dispatch-guide")

    def test_cross_stage_priority_propagation(self):
        before = self.result()
        self.data["feedback"][0]["severity"] = "critical"
        self.data["feedback"][0]["text"] = "terrible " * 10
        after = self.result()
        research = next(t for t in after["research"]["topics"] if t["topic"] == "delivery")
        self.assertEqual(research["priority"], 100)
        old = next(i for i in before["recommendations"]["ranked"] if i["item_id"] == "dispatch-guide")
        new = next(i for i in after["recommendations"]["ranked"] if i["item_id"] == "dispatch-guide")
        self.assertEqual(new["research_score"], 50)
        self.assertGreater(new["score"], old["score"])

    def test_cross_stage_document_propagation(self):
        before = self.result()
        self.data["documents"] = []
        after = self.result()
        self.assertEqual(before["sentiment"], after["sentiment"])
        self.assertEqual(after["recommendations"]["ranked"], [])
        self.assertTrue(all(t["unresolved_questions"] for t in after["research"]["topics"]))

    def test_handoff_rejects_forged_priority(self):
        sentiment = app.sentiment_stage(self.data)
        sentiment["issues"][0]["priority"] = 999
        with self.assertRaises(app.ValidationError):
            app.research_stage(self.data, sentiment)

    def test_handoff_rejects_forged_citation(self):
        research = self.result()["research"]
        research["topics"][0]["findings"][0]["evidence"][0]["document_id"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.interests_stage(self.data, research)

    def test_output_rejects_forged_recommendation(self):
        result = self.result()
        result["recommendations"]["ranked"][0]["item_id"] = "blocked-guide"
        with self.assertRaises(app.ValidationError):
            app.validate("result", result, self.data)

    def test_empty_collections(self):
        self.data["feedback"] = []
        self.data["documents"] = []
        self.data["catalog"] = []
        result = self.result()
        self.assertEqual(result["sentiment"]["issues"], [])
        self.assertEqual(result["research"]["topics"], [])
        self.assertEqual(result["recommendations"]["ranked"], [])

    def test_no_interests_and_zero_limit(self):
        self.data["preferences"]["interests"] = []
        self.assertEqual(self.result()["recommendations"]["ranked"], [])
        self.setUp()
        self.data["preferences"]["limit"] = 0
        self.assertEqual(self.result()["recommendations"]["ranked"], [])

    def test_ties_deterministic_and_input_unchanged(self):
        self.data["catalog"].append(dict(self.data["catalog"][0], id="a-dispatch"))
        original = copy.deepcopy(self.data)
        first = self.result()
        self.assertEqual(first, self.result())
        self.assertEqual(self.data, original)
        ids = [i["item_id"] for i in first["recommendations"]["ranked"]]
        self.assertLess(ids.index("a-dispatch"), ids.index("dispatch-guide"))
        self.data["catalog"].reverse()
        self.data["documents"].reverse()
        self.data["feedback"].reverse()
        self.assertEqual(first, self.result())

    def test_invalid_types_and_ranges(self):
        for key, value in [("limit", True), ("limit", -1), ("limit", 101), ("limit", 1.0)]:
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["preferences"][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        for weight in [0, 6, True, "3"]:
            with self.subTest(weight=weight):
                data = copy.deepcopy(self.data)
                data["preferences"]["interests"][0]["weight"] = weight
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicates_unknown_keys_and_invalid_version(self):
        changes = [
            lambda d: d["feedback"].append(copy.deepcopy(d["feedback"][0])),
            lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
            lambda d: d["preferences"]["interests"].append({"tag": "safety", "weight": 2}),
            lambda d: d.update(extra=True),
            lambda d: d.update(schema_version=2),
            lambda d: d.update(schema_version=True),
            lambda d: d["feedback"][0].update(severity="urgent"),
            lambda d: d["feedback"][0].update(text=" "),
            lambda d: d["catalog"][0]["tags"].append("operations"),
        ]
        for change in changes:
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_quote_is_rejected(self):
        self.data["documents"][0]["evidence"][0]["quote"] = "Invented source content"
        with self.assertRaisesRegex(app.ValidationError, "quote not found"):
            self.result()

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_single_json_object(self):
        process = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout), self.result())

    def test_cli_missing_file_and_usage(self):
        for args in [(), ("missing-input.json",), ("example_input.json", "extra")]:
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_schema_and_encoding(self):
        # Patch file reads rather than create extra deliverables or temporary files.
        invalid = ['{', '[]', '{"x":1,"x":2}', '{"value":NaN}', '{"value":Infinity}',
                   json.dumps(dict(self.data, schema_version=3))]
        for text in invalid:
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main([str(ROOT / "example_input.json")])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        with patch.object(Path, "read_text", side_effect=error):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(app.main([str(ROOT / "example_input.json")]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
