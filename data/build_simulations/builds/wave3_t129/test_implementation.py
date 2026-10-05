import copy
import contextlib
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
        self.fixture = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_complete_pipeline_and_schema(self):
        result = app.run_pipeline(self.fixture)
        self.assertEqual(result["stage"], "support")
        self.assertEqual(list(result["results"]), list(app.STAGES[1:]))
        self.assertIs(app.validate_state(result), result)

    def test_determinism_and_input_immutability(self):
        original = copy.deepcopy(self.fixture)
        self.assertEqual(app.run_pipeline(self.fixture), app.run_pipeline(self.fixture))
        self.assertEqual(self.fixture, original)

    def test_research_matches_and_quotes_policy(self):
        research = app.research(self.fixture)["results"]["research"]
        self.assertEqual([e["policy_id"] for e in research["evidence"]], ["policy-housing"])
        self.assertIn(research["evidence"][0]["excerpt"], self.fixture["context"]["policies"][0]["text"])

    def test_recommendation_score_is_explainable(self):
        result = app.recommend(app.research(self.fixture))["results"]["recommend"]
        item = result["items"][0]
        self.assertEqual(item["service_id"], "housing-help")
        self.assertEqual(item["score"], 3 * len(item["interest_matches"]) + len(item["question_matches"]))
        self.assertEqual(result["eligibility_decision"], "not_made")

    def test_insights_only_selected_services(self):
        result = app.run_pipeline(self.fixture)["results"]["insights"]
        self.assertEqual(result["feedback_count"], 2)
        self.assertEqual(result["selected_service_ids"], ["housing-help"])
        self.assertTrue(all("feedback-three" not in t["feedback_ids"] for t in result["themes"]))
        self.assertEqual({t["theme"] for t in result["themes"]},
                         {"clarity", "accessibility", "delays", "positive"})

    def test_support_citations_and_policy_steps(self):
        result = app.run_pipeline(self.fixture)["results"]["support"]
        self.assertEqual(result["citations"][0]["policy_id"], "policy-housing")
        self.assertEqual([s["number"] for s in result["steps"]], [1, 2, 3])
        policy = self.fixture["context"]["policies"][0]["text"]
        self.assertTrue(all(s["text"] in policy for s in result["steps"]))

    def test_insights_propagate_to_support(self):
        result = app.run_pipeline(self.fixture)["results"]
        self.assertIn(app.THEMES["clarity"][1], result["support"]["insight_actions"])
        self.fixture["context"]["feedback"] = []
        changed = app.run_pipeline(self.fixture)["results"]
        self.assertNotIn(app.THEMES["clarity"][1], changed["support"]["insight_actions"])

    def test_research_changes_all_downstream_stages(self):
        self.fixture["context"]["policies"][0]["services"][0]["topics"] = ["unrelated"]
        self.fixture["context"]["policies"][0]["services"][0]["name"] = "Unrelated service"
        self.fixture["context"]["policies"][0]["services"][0]["description"] = "Unrelated activities"
        result = app.run_pipeline(self.fixture)["results"]
        self.assertTrue(result["research"]["evidence"])
        self.assertEqual(result["recommend"]["items"], [])
        self.assertEqual(result["insights"]["feedback_count"], 0)
        self.assertTrue(result["support"]["handoff_required"])

    def test_pii_removed_from_all_output(self):
        context = self.fixture["context"]
        secrets = app.sensitive_values(context)
        context["feedback"][0]["text"] += " " + " ".join(secrets)
        context["support_question"] += " " + " ".join(secrets)
        encoded = json.dumps(app.run_pipeline(self.fixture))
        for secret in secrets:
            self.assertNotIn(secret, encoded)

    def test_free_text_contacts_redacted(self):
        self.fixture["context"]["feedback"][0]["text"] += " person@example.invalid 999-11-2222"
        output = json.dumps(app.run_pipeline(self.fixture))
        self.assertNotIn("person@example.invalid", output)
        self.assertNotIn("999-11-2222", output)

    def test_pii_does_not_change_ranking(self):
        before = app.run_pipeline(self.fixture)["results"]["recommend"]
        self.fixture["context"]["government_form"]["citizen"]["name"] = "Synthetic Citizen Birch"
        self.assertEqual(before, app.run_pipeline(self.fixture)["results"]["recommend"])

    def test_audit_has_ordered_handoffs(self):
        result = app.run_pipeline(self.fixture)
        self.assertEqual([x["consumed_stage"] for x in result["audit"]], list(app.STAGES[:-1]))
        self.assertTrue(all(x["rule"] and x["source_policy_ids"] for x in result["audit"]))

    def test_empty_sources_safe_fallback(self):
        self.fixture["context"]["policies"] = []
        self.fixture["context"]["feedback"] = []
        result = app.run_pipeline(self.fixture)["results"]
        self.assertTrue(result["research"]["unanswered"])
        self.assertEqual(result["recommend"]["items"], [])
        self.assertEqual(result["support"]["citations"], [])
        self.assertTrue(result["support"]["handoff_required"])

    def test_unknown_support_question_is_not_guessed(self):
        self.fixture["context"]["support_question"] = "Explain interstellar navigation."
        result = app.run_pipeline(self.fixture)["results"]["support"]
        self.assertTrue(result["handoff_required"])
        self.assertEqual(result["steps"], [])

    def test_eligibility_and_appeal_require_human(self):
        self.fixture["context"]["support_question"] = "Am I eligible for housing? My application was denied."
        result = app.run_pipeline(self.fixture)["results"]["support"]
        self.assertTrue(result["handoff_required"])
        self.assertEqual(result["eligibility_decision"], "not_made")

    def test_accessible_plain_language_output(self):
        result = app.run_pipeline(self.fixture)["results"]["support"]
        self.assertEqual(result["format"], "plain_language_numbered_steps")
        self.assertIn("Ask for an accessible format if you need one.", result["insight_actions"])
        self.assertTrue(all(len(step["text"].split()) <= 24 for step in result["steps"]))

    def test_non_synthetic_rejected(self):
        self.fixture["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.fixture)

    def test_unknown_pii_field_rejected(self):
        self.fixture["context"]["government_form"]["citizen"]["national_id"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.fixture)

    def test_bad_ratings_and_boolean_not_integer(self):
        for rating in (0, 6, True, "3", None):
            with self.subTest(rating=rating):
                self.fixture["context"]["feedback"][0]["rating"] = rating
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.fixture)

    def test_duplicate_policy_rejected(self):
        self.fixture["context"]["policies"].append(copy.deepcopy(self.fixture["context"]["policies"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.fixture)

    def test_dangling_feedback_reference_rejected(self):
        self.fixture["context"]["feedback"][0]["service_id"] = "unknown-service"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.fixture)

    def test_ungrounded_application_steps_rejected(self):
        self.fixture["context"]["policies"][0]["services"][0]["application_steps"] = ["Get automatic approval."]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.fixture)

    def test_cannot_skip_stage(self):
        with self.assertRaises(app.ValidationError):
            app.insights(app.research(self.fixture))

    def test_tampered_handoff_rejected(self):
        state = app.research(self.fixture)
        state["results"]["research"]["catalog"][0]["policy_id"] = "fabricated-policy"
        with self.assertRaises(app.ValidationError):
            app.recommend(state)

    def test_tampered_insight_rejected(self):
        state = app.insights(app.recommend(app.research(self.fixture)))
        state["results"]["insights"]["themes"][0]["action"] = "Promise approval."
        with self.assertRaises(app.ValidationError):
            app.support(state)

    def test_tampered_audit_rejected(self):
        state = app.research(self.fixture)
        state["audit"][0]["rule"] = "Hidden rule"
        with self.assertRaises(app.ValidationError):
            app.recommend(state)

    def test_feedback_no_keywords_uses_other(self):
        self.fixture["context"]["feedback"] = [self.fixture["context"]["feedback"][0]]
        self.fixture["context"]["feedback"][0]["text"] = "Submitted yesterday."
        themes = app.run_pipeline(self.fixture)["results"]["insights"]["themes"]
        self.assertEqual(themes[0]["theme"], "other")

    def test_tie_break_and_result_limit(self):
        policy = self.fixture["context"]["policies"][0]
        extra = copy.deepcopy(policy["services"][0])
        extra["service_id"] = "a-housing"
        policy["services"].append(extra)
        self.fixture["context"]["government_form"]["preferences"]["max_results"] = 1
        items = app.run_pipeline(self.fixture)["results"]["recommend"]["items"]
        self.assertEqual([x["service_id"] for x in items], ["a-housing"])

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "does-not-exist.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_bad_json_without_extra_files(self):
        for source in ('{"broken"', '{"a":1,"a":2}', '{"number":NaN}', '[]'):
            with self.subTest(source=source):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=source), contextlib.redirect_stdout(output):
                    code = app.main([str(ROOT / "example_input.json")])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_validation_error_never_echoes_pii(self):
        self.fixture["context"]["government_form"]["citizen"]["unexpected"] = "secret@example.invalid"
        output = io.StringIO()
        with patch.object(Path, "read_text", return_value=json.dumps(self.fixture)), contextlib.redirect_stdout(output):
            code = app.main([str(ROOT / "example_input.json")])
        self.assertEqual(code, 2)
        self.assertNotIn("secret", output.getvalue())

    def test_cli_usage_error(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
