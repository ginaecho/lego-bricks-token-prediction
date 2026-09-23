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
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_success(self):
        result = app.run_pipeline(self.request)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stage"], "interests")
        self.assertEqual(result["recommendations"][0]["id"], "camera-trail")
        self.assertEqual(result["context"]["fixture_label"], self.request["fixture_label"])

    def test_input_is_not_mutated(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(app.run_pipeline(self.request), app.run_pipeline(self.request))
        self.assertEqual(original, self.request)

    def test_onboarding_prerequisite_order(self):
        self.request["profile"]["completed_lessons"] = []
        steps = app.adaptive_onboarding(self.request)["onboarding"]["steps"]
        self.assertEqual([s["lesson"] for s in steps], ["basics", "search", "compare"])
        self.assertEqual(steps[-1]["prerequisites"], ["search"])
        self.assertIn("preferences", steps[-1]["explanation"])

    def test_experience_adapts_explanations(self):
        self.request["profile"]["experience"] = "expert"
        result = app.adaptive_onboarding(self.request)
        self.assertIn("expert", result["onboarding"]["steps"][0]["explanation"])

    def test_preferences_adapt_plan(self):
        self.request["profile"]["preferences"] = {}
        steps = app.adaptive_onboarding(self.request)["onboarding"]["steps"]
        self.assertEqual([s["lesson"] for s in steps], ["search"])

    def test_planned_prerequisites_do_not_unlock(self):
        self.request["profile"]["completed_lessons"] = []
        result = app.run_pipeline(self.request)
        self.assertNotIn("camera-trail", [r["id"] for r in result["search"]])

    def test_blank_query_interest_handoff(self):
        self.request["query"] = ""
        result = app.run_pipeline(self.request)
        self.assertEqual(result["onboarding"]["search_query"], "photography hiking")
        self.assertIn("tent", [r["id"] for r in result["search"]])

    def test_synonym_search(self):
        self.request["query"] = "photos"
        result = app.run_pipeline(self.request)
        self.assertEqual({r["id"] for r in result["search"]}, {"camera-trail", "camera-studio"})

    def test_exclusions_and_experience_enforced(self):
        result = app.run_pipeline(self.request)
        for field in ("search", "recommendations"):
            ids = {r["id"] for r in result[field]}
            self.assertFalse(ids & {"camera-excluded", "camera-restricted", "camera-pro"})

    def test_case_insensitive_category_exclusion(self):
        self.request["profile"]["excluded_categories"] = ["OUTDOOR", "RESTRICTED"]
        result = app.run_pipeline(self.request)
        self.assertEqual([r["id"] for r in result["search"]], ["camera-studio"])

    def test_recommendations_only_from_search(self):
        self.request["options"]["limit"] = 1
        result = app.run_pipeline(self.request)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertTrue({r["id"] for r in result["recommendations"]} <=
                        {r["id"] for r in result["search"]})

    def test_preference_reranking_and_grounding(self):
        self.request["query"] = "camera"
        self.request["profile"]["interests"] = []
        self.request["profile"]["preferences"] = {"studio": 1}
        result = app.run_pipeline(self.request)
        self.assertEqual(result["recommendations"][0]["id"], "camera-studio")
        self.assertIn("Preferred category/tag concepts: studio",
                      result["recommendations"][0]["explanations"])
        self.assertNotIn("Preferred category/tag concepts: studio",
                         result["recommendations"][1]["explanations"])

    def test_empty_catalog(self):
        self.request["catalog"] = []
        result = app.run_pipeline(self.request)
        self.assertEqual(result["search"], [])
        self.assertEqual(result["recommendations"], [])

    def test_unmatched_query(self):
        self.request["query"] = "zzzzunknown"
        self.assertEqual(app.run_pipeline(self.request)["recommendations"], [])

    def test_blank_query_browses_without_fabricating_match(self):
        self.request["query"] = ""
        self.request["profile"]["interests"] = []
        self.request["profile"]["preferences"] = {}
        result = app.run_pipeline(self.request)
        self.assertTrue(result["search"])
        self.assertTrue(all(r["score"] == 0 for r in result["search"]))
        self.assertIn("Browse candidate", result["search"][0]["explanations"][0])

    def test_threshold(self):
        self.request["options"]["min_score"] = 1
        result = app.run_pipeline(self.request)
        self.assertEqual([r["id"] for r in result["search"]], ["camera-trail"])

    def test_deterministic_ties(self):
        self.request["query"] = "camera"
        rows = app.run_pipeline(self.request)["search"]
        self.assertEqual([r["id"] for r in rows], ["camera-studio", "camera-trail"])

    def test_injected_embedding_fixture_and_query_handoff(self):
        calls = []
        self.request["query"] = "unmatchedconcept"
        def embed(texts):
            calls.append(texts)
            return [[1, 0]] + [[1, 0] for _ in texts[1:]]
        result = app.run_pipeline(self.request, embed)
        self.assertEqual(calls[0][0], "unmatchedconcept")
        self.assertTrue(result["search"])
        self.assertTrue(all(r["score"] == 0.5 for r in result["search"]))
        self.assertIn("Injected embedding", result["search"][0]["explanations"][0])

    def test_invalid_embedding_fixtures(self):
        fixtures = [None, [], [[0, 0]] * 4, [[float("nan")]] * 4,
                    [[True]] * 4, [[1], [1, 0], [1], [1]]]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.request, lambda texts: fixture)

    def test_embedding_exception_wrapped(self):
        def broken(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(self.request, broken)

    def test_invalid_inputs(self):
        for field, value in [("schema_version", True), ("query", None),
                             ("catalog", {}), ("profile", [])]:
            with self.subTest(field=field):
                request = copy.deepcopy(self.request)
                request[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_invalid_preferences_and_limits(self):
        for value in [float("nan"), float("inf"), -1, True, "high"]:
            with self.subTest(value=value):
                self.request["profile"]["preferences"] = {"portable": value}
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.request)
        self.request["profile"]["preferences"] = {}
        self.request["options"]["limit"] = True
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.request)

    def test_duplicate_ids(self):
        self.request["catalog"].append(copy.deepcopy(self.request["catalog"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate product"):
            app.run_pipeline(self.request)

    def test_invalid_completed_prerequisites(self):
        self.request["profile"]["completed_lessons"] = ["compare"]
        with self.assertRaisesRegex(app.ValidationError, "prerequisite"):
            app.run_pipeline(self.request)

    def test_handoff_rejects_wrong_stage(self):
        with self.assertRaisesRegex(app.ValidationError, "pipeline stage"):
            app.recommend_interests(app.adaptive_onboarding(self.request))

    def test_handoff_rejects_ineligible_candidates(self):
        envelope = app.semantic_search(app.adaptive_onboarding(self.request))
        envelope["search"][0]["id"] = "camera-excluded"
        with self.assertRaisesRegex(app.ValidationError, "ineligible"):
            app.recommend_interests(envelope)

    def test_handoff_rejects_invalid_score(self):
        envelope = app.semantic_search(app.adaptive_onboarding(self.request))
        envelope["search"][0]["score"] = float("nan")
        with self.assertRaises(app.ValidationError):
            app.recommend_interests(envelope)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file(self):
        proc = subprocess.run([sys.executable, "-B", "implementation.py", "absent-fixture.json"],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")
        self.assertEqual(proc.stderr, "")

    def test_cli_usage_error(self):
        proc = subprocess.run([sys.executable, "-B", "implementation.py"],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_invalid_json_and_validation_errors(self):
        for content in ["{", '{"a":1,"a":2}', '{"score":NaN}', "{}", "[]"]:
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
