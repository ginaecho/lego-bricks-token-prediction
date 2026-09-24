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

    def test_personalized_team_onboarding(self):
        stage = app.onboard(self.request)
        self.assertEqual(stage["next_step"]["id"], "invite_team")
        self.assertIn("Morgan", stage["next_step"]["instruction"])
        self.assertIn(self.request["customer"]["goal"], stage["next_step"]["instruction"])

    def test_individual_skips_team_invitation(self):
        self.request["customer"]["role"] = "individual"
        self.assertEqual(app.onboard(self.request)["next_step"]["id"], "start_project")

    def test_new_customer_first_step(self):
        self.request["customer"]["completed_steps"] = []
        self.assertEqual(app.onboard(self.request)["next_step"]["id"], "verify_email")

    def test_completed_onboarding_still_supports(self):
        self.request["customer"]["completed_steps"] = sorted(app.STEP_IDS)
        result = app.run_pipeline(self.request)
        self.assertEqual(result["onboarding"]["status"], "complete")
        self.assertIsNone(result["support"]["next_step"])
        self.assertEqual(result["support"]["status"], "answered")

    def test_grounded_offline_answer(self):
        support = app.run_pipeline(self.request)["support"]
        self.assertEqual(support["answer"], self.request["knowledge"][0]["body"])
        self.assertEqual(support["citations"], ["synthetic-kb-invite"])
        self.assertIsNone(support["handoff"])

    def test_cross_stage_propagation(self):
        result = app.run_pipeline(self.request)
        self.assertEqual(result["onboarding"]["next_step"], result["support"]["next_step"])
        self.assertEqual(result["support"]["customer_id"], self.request["customer"]["id"])
        self.assertEqual(result["support"]["onboarding_step_id"], "invite_team")

    def test_tie_prefers_current_onboarding_step(self):
        self.request["knowledge"][1]["keywords"].append("invite")
        self.request["knowledge"][1]["keywords"].append("teammate")
        self.request["knowledge"].reverse()
        self.assertEqual(app.run_pipeline(self.request)["support"]["citations"],
                         ["synthetic-kb-invite"])

    def test_unknown_question_does_not_invent_answer(self):
        self.request["support"]["question"] = "Refund?"
        support = app.run_pipeline(self.request)["support"]
        self.assertEqual(support["status"], "needs_human")
        self.assertEqual(support["citations"], [])
        self.assertEqual(support["handoff"]["route"], "offline_queue")
        self.assertIn("invite_team", support["handoff"]["summary"])
        self.assertIn(self.request["customer"]["goal"], support["handoff"]["summary"])

    def test_online_fallback(self):
        self.request["knowledge"] = []
        self.request["support"]["team_online"] = True
        self.assertEqual(app.run_pipeline(self.request)["support"]["handoff"]["route"], "live_team")

    def test_invalid_inputs(self):
        for key, value in (("schema_version", True), ("knowledge", {}),
                           ("customer", None), ("extra", 1)):
            with self.subTest(key=key):
                request = copy.deepcopy(self.request)
                request[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)
        for value in ("", "   ", None, 42):
            self.request["support"]["question"] = value
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.request)

    def test_invalid_steps_and_duplicates(self):
        for steps in (["unknown"], ["verify_email", "verify_email"]):
            self.request["customer"]["completed_steps"] = steps
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.request)

    def test_duplicate_knowledge_rejected(self):
        self.request["knowledge"].append(copy.deepcopy(self.request["knowledge"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.request)

    def test_foreign_or_stale_handoff_rejected(self):
        stage = app.onboard(self.request)
        for field, value in (("customer_id", "other"), ("goal", "other goal")):
            corrupted = copy.deepcopy(stage)
            corrupted[field] = value
            with self.assertRaises(app.ValidationError):
                app.answer_support(self.request, corrupted)

    def test_deterministic_and_no_input_mutation(self):
        before = copy.deepcopy(self.request)
        self.assertEqual(app.run_pipeline(self.request), app.run_pipeline(self.request))
        self.assertEqual(before, self.request)

    def test_malformed_output_handoff_rejected(self):
        result = app.run_pipeline(self.request)
        result["support"]["onboarding_step_id"] = "verify_email"
        with self.assertRaises(app.ValidationError):
            app.validate("output", result)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema_without_files(self):
        for content in ("{", '{"x": 1, "x": 2}', '{"schema_version": NaN}', "{}"):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
