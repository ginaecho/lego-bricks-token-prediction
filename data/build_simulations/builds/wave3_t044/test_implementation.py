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

    def test_new_customer_personalized_step(self):
        self.data["customer"]["completed_steps"] = []
        result = app.onboard(self.data)["onboarding"]
        self.assertEqual(result["next_step"]["id"], "discover_preferences")
        self.assertIn("Alex Example", result["next_step"]["instruction"])
        self.assertIn("learn", result["next_step"]["instruction"])

    def test_progress_and_completion(self):
        for count in range(4):
            with self.subTest(count=count):
                self.data["customer"]["completed_steps"] = list(app.STEPS[:count])
                result = app.run_pipeline(self.data)
                self.assertEqual(result["onboarding"]["completed"], count == 3)
                self.assertEqual(result["discovery"]["onboarding_step"],
                                 app.STEPS[count] if count < 3 else "complete")

    def test_ranking_and_filters(self):
        result = app.run_pipeline(self.data)["discovery"]
        self.assertEqual([item["product_id"] for item in result["items"]], ["kit-02", "kit-01"])
        self.assertEqual(result["items"][0]["score"], 7)
        self.assertEqual(result["excluded"], {"unavailable": 1, "over_budget": 1, "no_relevance": 0})

    def test_cross_stage_context_propagation(self):
        self.data["customer"].update(id="synthetic-other", goal="upgrade", interests=["robotics"],
                                     budget=150, experience="advanced")
        result = app.recommend(app.onboard(self.data))
        self.assertEqual(result["discovery"]["customer_id"], "synthetic-other")
        self.assertEqual(result["discovery"]["items"][0]["product_id"], "kit-03")
        self.assertEqual(result["onboarding"]["discovery_context"]["max_price"], 150)

    def test_tampered_handoff_rejected(self):
        state = app.onboard(self.data)
        state["onboarding"]["discovery_context"]["max_price"] = 1000
        with self.assertRaises(app.ValidationError):
            app.recommend(state)

    def test_invalid_output_rejected(self):
        state = app.run_pipeline(self.data)
        state["discovery"]["items"][0]["score"] = 1000
        with self.assertRaises(app.ValidationError):
            app.validate_state(state, "discovery")

    def test_empty_catalog(self):
        self.data["catalog"] = []
        result = app.run_pipeline(self.data)["discovery"]
        self.assertEqual(result["items"], [])
        self.assertIn("No suitable products", result["message"])

    def test_cold_start_goal_only(self):
        self.data["customer"]["interests"] = []
        result = app.run_pipeline(self.data)["discovery"]
        self.assertEqual(result["items"][0]["score"], 5)

    def test_no_relevant_products(self):
        self.data["customer"].update(goal="unrepresented", interests=[])
        result = app.run_pipeline(self.data)["discovery"]
        self.assertEqual(result["items"], [])
        self.assertEqual(result["excluded"]["no_relevance"], 2)

    def test_zero_budget_free_product(self):
        self.data["customer"]["budget"] = 0
        self.data["catalog"][0]["price"] = 0
        self.assertEqual(app.run_pipeline(self.data)["discovery"]["eligible_count"], 1)

    def test_tie_breaker_and_limit(self):
        self.data["catalog"][1]["price"] = 45
        self.data["limit"] = 1
        result = app.run_pipeline(self.data)["discovery"]
        self.assertEqual(result["items"][0]["product_id"], "kit-01")
        self.assertEqual(result["eligible_count"], 2)

    def test_deterministic_no_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.data["catalog"].reverse()
        second = app.run_pipeline(self.data)
        self.assertEqual(first["discovery"], second["discovery"])
        self.data["catalog"].reverse()
        self.assertEqual(self.data, original)

    def test_invalid_input_variants(self):
        cases = [
            ("budget", -1), ("budget", True), ("budget", float("inf")),
            ("name", " "), ("interests", "robotics"), ("interests", ["x", "x"]),
            ("experience", "expert"), ("completed_steps", ["choose_product"]),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["customer"][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        for data in (None, {}, dict(self.data, limit=True), dict(self.data, synthetic=False)):
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_duplicate_product_rejected(self):
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "missing.json")], [str(ROOT)]):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                        capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_bad_json_and_schema(self):
        for payload in ("{", "null", '{"x":NaN}', '{"x":1,"x":2}', '{"schema_version":1}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
