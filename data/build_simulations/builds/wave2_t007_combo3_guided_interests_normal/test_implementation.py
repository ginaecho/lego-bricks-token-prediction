import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    @property
    def prefs(self):
        return self.data["guided"]["responses"]["preferences"]

    def test_complete_progress(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["guided"]["progress"]["fraction"], 1)
        self.assertIsNone(result["guided"]["next_step"])

    def test_each_partial_progress_is_gated(self):
        for count in range(3):
            with self.subTest(count=count):
                self.data["guided"]["completed_steps"] = list(app.STEPS[:count])
                result = app.run_pipeline(self.data)
                self.assertEqual(result["status"], "needs_setup")
                self.assertEqual(result["guided"]["next_step"], app.STEPS[count])
                self.assertEqual(result["guided"]["progress"]["completed"], count)
                self.assertIsNone(result["interests"])
                self.assertIsNone(result["normal"])

    def test_fresh_onboarding(self):
        self.data["guided"] = {"completed_steps": [], "responses": {}}
        self.assertEqual(app.run_pipeline(self.data)["guided"]["next_step"], "consent")

    def test_prerequisite_order(self):
        for done in (["profile"], ["consent", "preferences"], ["consent", "consent"]):
            self.data["guided"]["completed_steps"] = done
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data)

    def test_consent_and_missing_response(self):
        self.data["guided"]["responses"]["consent"] = False
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        del self.data["guided"]["responses"]["consent"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_preference_rank_and_grounded_explanation(self):
        recommendations = app.run_pipeline(self.data)["interests"]["recommendations"]
        self.assertEqual([row["score"] for row in recommendations], [5, 3])
        self.assertEqual(recommendations[0]["item_id"], "garden-water")
        self.assertEqual(recommendations[0]["matched_interests"],
                         [{"topic": "gardening", "weight": 3}, {"topic": "water", "weight": 2}])
        self.assertEqual(recommendations[0]["explanation"],
                         "Matched declared interests: gardening (weight 3), water (weight 2)")

    def test_exclusions_propagate_to_research(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["interests"]["excluded"],
                         [{"item_id": "blocked-guide", "reasons": ["excluded_id"]},
                          {"item_id": "sponsored", "reasons": ["excluded_topic"]}])
        self.assertEqual(result["normal"]["eligible_item_ids"], ["garden-water", "garden-basics"])
        self.assertTrue(all(row["citation"]["item_id"] in {"garden-water", "garden-basics"}
                            for row in result["normal"]["findings"]))

    def test_exclusion_overrides_interest(self):
        self.prefs["exclude_topics"] += ["WATER", "gardening"]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["interests"]["recommendations"], [])
        self.assertTrue(result["normal"]["no_evidence"])

    def test_profile_and_limit_handoffs(self):
        self.prefs["max_recommendations"] = 1
        result = app.run_pipeline(self.data)
        self.assertEqual(result["interests"]["profile"], result["guided"]["profile"])
        self.assertEqual(result["interests"]["preferences"], self.prefs)
        self.assertEqual(result["normal"]["eligible_item_ids"], ["garden-water"])

    def test_changed_preference_changes_downstream(self):
        self.prefs["interests"] = [{"topic": "music", "weight": 9}]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["normal"]["eligible_item_ids"], ["unrelated"])
        self.assertEqual(result["normal"]["findings"][0]["citation"]["item_id"], "unrelated")

    def test_exact_citations_and_rank(self):
        result = app.run_pipeline(self.data)["normal"]
        self.assertEqual(result["findings"][0]["citation"]["passage_id"], "p2")
        for row in result["findings"]:
            citation = row["citation"]
            item = next(item for item in self.data["catalog"] if item["id"] == citation["item_id"])
            passage = next(p for p in item["passages"] if p["id"] == citation["passage_id"])
            self.assertEqual(citation["source"], passage["source"])
            self.assertEqual(row["text"], passage["text"][citation["start"]:citation["end"]])

    def test_unicode_exact_offsets(self):
        passage = self.data["catalog"][0]["passages"][0]
        passage["text"] = "  Café 🌱 garden soil.  "
        self.data["research"]["query"] = "CAFÉ"
        row = app.run_pipeline(self.data)["normal"]["findings"][0]
        self.assertEqual(row["text"], passage["text"])
        self.assertEqual(row["citation"]["end"], len(passage["text"]))

    def test_no_query_evidence(self):
        self.data["research"]["query"] = "astronomy"
        result = app.run_pipeline(self.data)["normal"]
        self.assertEqual(result["findings"], [])
        self.assertTrue(result["no_evidence"])

    def test_empty_catalog(self):
        self.data["catalog"] = []
        self.assertTrue(app.run_pipeline(self.data)["normal"]["no_evidence"])

    def test_case_insensitive_topics(self):
        self.prefs["interests"][0]["topic"] = "GARDENING"
        self.prefs["exclude_topics"] = ["ADVERTISING"]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["interests"]["recommendations"][0]["score"], 5)
        self.assertNotIn("sponsored", result["normal"]["eligible_item_ids"])

    def test_ties_and_catalog_order(self):
        self.prefs["interests"] = [{"topic": "gardening", "weight": 1}]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["normal"]["eligible_item_ids"], ["garden-basics", "garden-water"])
        self.data["catalog"].reverse()
        self.assertEqual(result, app.run_pipeline(self.data))

    def test_finding_limit(self):
        self.data["research"]["max_findings"] = 1
        self.assertEqual(len(app.run_pipeline(self.data)["normal"]["findings"]), 1)

    def test_invalid_weights(self):
        for weight in (True, 0, -1, 101, 10 ** 400, "3", float("nan"), float("inf"), None):
            with self.subTest(weight=weight):
                self.prefs["interests"][0]["weight"] = weight
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_invalid_shapes_and_limits(self):
        baseline = copy.deepcopy(self.data)
        for key, value in (("schema_version", True), ("catalog", {}), ("fixture_label", "real"),
                           ("research", {"query": "!!!", "max_findings": 1}),
                           ("research", {"query": "soil", "max_findings": True}),
                           ("research", {"query": "", "max_findings": 0})):
            with self.subTest(key=key, value=value):
                self.data = copy.deepcopy(baseline)
                self.data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_duplicate_identifiers_and_topics(self):
        baseline = copy.deepcopy(self.data)
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data = copy.deepcopy(baseline)
        self.data["catalog"][0]["passages"].append(copy.deepcopy(self.data["catalog"][0]["passages"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data = baseline
        self.prefs["interests"].append({"topic": "GARDENING", "weight": 1})
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_unknown_fields_and_non_synthetic_source(self):
        self.data["unexpected"] = 1
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        del self.data["unexpected"]
        self.data["catalog"][0]["passages"][0]["source"] = "https://example.invalid"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_tampered_handoffs(self):
        request = app.validate_request(self.data)
        guided = app.guided_setup(request)
        interests = app.interest_recommendations(request, guided)
        interests["recommendations"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.validate_handoff(request, "interests", interests, guided)
        interests = app.interest_recommendations(request, guided)
        normal = app.normal_research(request, interests)
        normal["findings"][0]["citation"]["end"] = 1
        with self.assertRaises(app.ValidationError):
            app.validate_handoff(request, "normal", normal, interests)
        guided["preferences"]["exclude_topics"] = []
        with self.assertRaises(app.ValidationError):
            app.validate_handoff(request, "guided", guided)

    def test_deterministic_without_input_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(original, self.data)
        first["guided"]["preferences"]["exclude_topics"].clear()
        self.assertEqual(original, self.data)

    def invoke(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        process = self.invoke("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout), app.run_pipeline(self.data))

    def test_cli_usage_missing_and_directory_errors(self):
        for args in ((), ("does-not-exist.json",), (".",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                process = self.invoke(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_duplicate_keys_and_schema(self):
        for content in ("{", '{"schema_version": 1, "schema_version": 1}', "[]",
                        json.dumps(dict(self.data, schema_version=2))):
            with self.subTest(content=content[:40]):
                with patch("builtins.open", return_value=io.StringIO(content)):
                    with patch("sys.stdout", new_callable=io.StringIO) as output:
                        self.assertEqual(app.main(["fixture.json"]), 2)
                        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_decoding_error(self):
        with patch("builtins.open", side_effect=UnicodeError("invalid fixture encoding")):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_excessive_json_nesting(self):
        content = "[" * 2000 + "0" + "]" * 2000
        with patch("builtins.open", return_value=io.StringIO(content)):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
