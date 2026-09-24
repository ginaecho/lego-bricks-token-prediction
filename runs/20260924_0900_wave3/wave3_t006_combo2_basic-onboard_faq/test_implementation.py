import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
import contextlib
import io

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_personalized_next_step(self):
        handoff = app.onboard(self.data)["onboarding"]
        self.assertEqual(handoff["next_step"]["id"], "invite")
        self.assertIn("Alex", handoff["message"])

    def test_first_step(self):
        self.data["customer"]["completed_steps"] = []
        self.assertEqual(app.run(self.data)["support"]["citations"], ["synthetic-kb-create"])

    def test_cross_stage_propagation_and_grounding(self):
        output = app.run(self.data)
        self.assertEqual(output["support"]["question"], output["onboarding"]["support_question"])
        self.assertEqual(output["support"]["customer_id"], self.data["customer"]["id"])
        self.assertEqual(output["support"]["step_id"], output["onboarding"]["next_step"]["id"])
        self.assertEqual(output["support"]["answer"], self.data["knowledge_base"][1]["answer"])
        self.assertEqual(output["support"]["citations"], ["synthetic-kb-invite"])

    def test_explicit_question_abstains(self):
        self.data["customer"]["question"] = "Can you predict next year's weather?"
        output = app.run(self.data)["support"]
        self.assertEqual(output["status"], "abstained")
        self.assertEqual(output["citations"], [])
        self.assertIsNone(output["answer"])

    def test_empty_kb(self):
        self.data["knowledge_base"] = []
        self.assertEqual(app.run(self.data)["support"]["reason"], "insufficient_evidence")

    def test_completion(self):
        self.data["customer"]["completed_steps"] = ["create", "invite"]
        output = app.run(self.data)
        self.assertTrue(output["onboarding"]["completed"])
        self.assertIsNone(output["onboarding"]["next_step"])
        self.assertEqual(output["support"]["citations"], ["synthetic-kb-activity"])

    def test_ambiguous_answers_abstain(self):
        article = dict(self.data["knowledge_base"][1], id="synthetic-other", answer="Conflicting text.")
        self.data["knowledge_base"].append(article)
        self.assertEqual(app.run(self.data)["support"]["reason"], "ambiguous_matches")

    def test_scope_excludes_other_step(self):
        self.data["customer"]["question"] = "How do I create a workspace?"
        self.assertEqual(app.run(self.data)["support"]["status"], "abstained")

    def test_reject_tampered_handoff(self):
        state = app.onboard(self.data)
        state["onboarding"]["customer_id"] = "different"
        with self.assertRaises(app.ValidationError):
            app.faq(state)

    def test_reject_ungrounded_output(self):
        state = app.run(self.data)
        state["support"]["answer"] = "Invented instructions."
        with self.assertRaises(app.ValidationError):
            app.validate(state, "faq")

    def test_input_invalid_cases(self):
        cases = [
            ("schema_version", True),
            ("customer", None),
            ("knowledge_base", {}),
            ("fixture_label", "real"),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.data)
                payload[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(payload)
        for steps in (["unknown"], ["create", "create"], [None]):
            with self.subTest(steps=steps):
                self.data["customer"]["completed_steps"] = steps
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_bad_catalog(self):
        self.data["knowledge_base"][0]["step_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_deterministic_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, original)

    def test_general_purpose_workflow(self):
        self.data["customer"]["goal"] = "learn pottery"
        self.data["workflow"]["goal"] = "learn pottery"
        for article in self.data["knowledge_base"]:
            article["goal"] = "learn pottery"
        self.data["workflow"]["steps"][1]["instruction"] = "Choose your clay."
        self.assertIn("Choose your clay", app.run(self.data)["onboarding"]["message"])

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "missing.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_bad_json_and_invalid_input(self):
        for content in ("{", "null", '{"a":1,"a":2}', '{"schema_version":NaN}'):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["virtual-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
