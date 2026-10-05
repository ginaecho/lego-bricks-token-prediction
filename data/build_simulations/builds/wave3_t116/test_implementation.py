import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.form = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args, input=None):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            input=input, text=True, capture_output=True, cwd=ROOT, check=False,
        )

    def assert_invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.form)

    def test_complete_pipeline_and_stage_order(self):
        result = app.run_pipeline(self.form)
        self.assertEqual(result["stage"], "recommend")
        self.assertEqual(tuple(result["results"]), app.STAGES)
        self.assertEqual([x["stage"] for x in result["audit"]], list(app.STAGES))

    def test_onboarding_draft(self):
        result = app.onboard(self.form)["results"]["onboard"]
        self.assertIn("gather", result["next_step"])

    def test_onboarding_submitted(self):
        self.form["benefits_application"]["status"] = "submitted"
        self.assertIn("status", app.onboard(self.form)["results"]["onboard"]["next_step"])

    def test_onboarding_missing_information(self):
        self.form["benefits_application"]["status"] = "needs_information"
        self.assertIn("missing", app.onboard(self.form)["results"]["onboard"]["next_step"])

    def test_support_is_grounded_and_service_filtered(self):
        result = app.support(app.onboard(self.form))["results"]["support"]
        self.assertIn("proof of income", result["answer"])
        self.assertEqual(result["citations"][0]["policy_id"], "SYN-POL-HOUSING")
        self.assertNotIn("travel", result["answer"].lower())

    def test_support_unknown_question(self):
        self.form["service_request"]["question"] = "Volcanic telescope?"
        result = app.run_pipeline(self.form)["results"]
        self.assertEqual(result["support"]["citations"], [])
        self.assertEqual(result["support"]["unanswered_question"], "Volcanic telescope?")
        self.assertEqual(result["recommend"]["items"], [])
        self.assertIsNotNone(result["recommend"]["fallback"])

    def test_research_handoff_is_exact(self):
        result = app.run_pipeline(self.form)["results"]
        self.assertEqual(result["research"]["evidence"], result["support"]["citations"])
        self.assertEqual(result["research"]["next_step"], result["onboard"]["next_step"])
        self.assertTrue(result["research"]["limitations"])

    def test_recommendation_cites_research_and_requires_review(self):
        result = app.run_pipeline(self.form)["results"]
        item = result["recommend"]["items"][0]
        self.assertEqual(item["program_id"], "SYN-PROG-HOUSING")
        ids = {x["policy_id"] for x in result["research"]["evidence"]}
        self.assertTrue(set(item["source_ids"]) <= ids)
        self.assertTrue(result["recommend"]["review_required"])

    def test_pii_removed_everywhere_in_output(self):
        output = json.dumps(app.run_pipeline(self.form))
        for field in ("name", "email", "address"):
            self.assertNotIn(self.form["citizen"][field], output)
        self.assertNotIn('"citizen"', output)

    def test_consent_required(self):
        self.form["privacy"]["processing_consent"] = False
        self.assert_invalid()

    def test_name_in_free_text_rejected(self):
        self.form["service_request"]["question"] = "Help " + self.form["citizen"]["name"]
        self.assert_invalid()

    def test_contact_in_policy_rejected(self):
        self.form["policy_documents"][0]["text"] += " Contact someone@example.invalid."
        self.assert_invalid()

    def test_identifier_number_in_question_rejected(self):
        self.form["service_request"]["question"] = "Please check 123-45-6789."
        self.assert_invalid()

    def test_accessibility_preferences_preserved(self):
        result = app.run_pipeline(self.form)
        self.assertEqual(result["context"]["preferences"]["accessibility_needs"],
                         ["plain_text", "screen_reader"])
        self.assertIn("Plain-text", result["safeguards"]["accessibility"])

    def test_unsupported_language_rejected_not_silently_ignored(self):
        self.form["citizen"]["preferred_language"] = "fr"
        self.assert_invalid()

    def test_unknown_field_rejected(self):
        self.form["citizen"]["date_of_birth"] = "invented"
        self.assert_invalid()

    def test_non_synthetic_rejected(self):
        self.form["synthetic"] = False
        self.assert_invalid()

    def test_non_synthetic_case_number_rejected(self):
        self.form["service_request"]["case_number"] = "REAL-123"
        self.assert_invalid()

    def test_unknown_application_status_rejected(self):
        self.form["benefits_application"]["status"] = "approved"
        self.assert_invalid()

    def test_blank_question_rejected(self):
        self.form["service_request"]["question"] = " "
        self.assert_invalid()

    def test_empty_policies_rejected(self):
        self.form["policy_documents"] = []
        self.assert_invalid()

    def test_duplicate_policy_rejected(self):
        self.form["policy_documents"].append(copy.deepcopy(self.form["policy_documents"][0]))
        self.assert_invalid()

    def test_wrong_stage_handoff_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.research(app.onboard(self.form))

    def test_forged_citation_rejected(self):
        result = app.support(app.onboard(self.form))
        result["results"]["support"]["citations"][0]["quote"] = "All people qualify."
        with self.assertRaises(app.ValidationError):
            app.research(result)

    def test_forged_research_rejected(self):
        result = app.research(app.support(app.onboard(self.form)))
        result["results"]["research"]["evidence"] = []
        with self.assertRaises(app.ValidationError):
            app.recommend(result)

    def test_forged_recommendation_rejected(self):
        result = app.run_pipeline(self.form)
        result["results"]["recommend"]["review_required"] = False
        with self.assertRaises(app.ValidationError):
            app.validate(result, "recommend")

    def test_inputs_are_not_mutated(self):
        original = copy.deepcopy(self.form)
        first = app.onboard(self.form)
        saved = copy.deepcopy(first)
        app.support(first)
        self.assertEqual(saved, first)
        self.assertEqual(original, self.form)

    def test_deterministic_results(self):
        self.assertEqual(app.run_pipeline(self.form), app.run_pipeline(self.form))

    def test_long_source_sentence_is_not_truncated_into_claim(self):
        self.form["policy_documents"][0]["text"] = "Housing " + "details " * 50 + "."
        self.assertEqual(app.run_pipeline(self.form)["results"]["support"]["citations"], [])

    def test_policy_order_does_not_change_ranking(self):
        first = app.run_pipeline(self.form)["results"]
        self.form["policy_documents"].reverse()
        self.assertEqual(first, app.run_pipeline(self.form)["results"])

    def test_cli_example_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        result = self.cli("nonexistent-input.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertNotIn("nonexistent-input", result.stdout)

    def test_cli_invalid_json(self):
        result = self.cli("-", input="{bad")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_duplicate_fields(self):
        result = self.cli("-", input='{"synthetic":true,"synthetic":false}')
        self.assertEqual(result.returncode, 2)
        self.assertIn("Duplicate", json.loads(result.stdout)["message"])

    def test_cli_no_argument(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_oversized_input(self):
        result = self.cli("-", input=" " * (app.MAX_BYTES + 1))
        self.assertEqual(result.returncode, 2)
        self.assertIn("size", json.loads(result.stdout)["message"])

    def test_cli_wrong_json_shapes(self):
        for malformed in ([], None, True, {"schema_version": "1.0"}):
            with self.subTest(malformed=malformed):
                result = self.cli("-", input=json.dumps(malformed))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_error_does_not_echo_pii(self):
        self.form["service_request"]["question"] = self.form["citizen"]["email"]
        result = self.cli("-", input=json.dumps(self.form))
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(self.form["citizen"]["email"], result.stdout)
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
