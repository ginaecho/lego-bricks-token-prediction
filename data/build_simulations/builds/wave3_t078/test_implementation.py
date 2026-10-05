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


ROOT = Path(__file__).parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normalization(self):
        product = app.normalize(self.raw)["catalog"][0]
        self.assertEqual(product["attributes"], {"price_usd": 40, "weight_g": 200, "battery_hours": 10})

    def test_comparison_and_side_by_side(self):
        state = app.compare(app.normalize(self.raw))
        self.assertEqual(state["comparison"]["ranking"][0]["product_id"], "demo-a")
        self.assertAlmostEqual(state["comparison"]["ranking"][0]["score"], .8)
        self.assertEqual(state["comparison"]["rows"], state["catalog"])

    def test_preference_propagation(self):
        self.raw["preferences"]["weights"] = dict(price_usd=0, weight_g=0, battery_hours=1)
        result = app.run(self.raw)
        self.assertEqual(result["comparison"]["ranking"][0]["product_id"], "demo-b")
        self.assertEqual(result["onboarding"]["preferred_product_id"], "demo-b")
        self.assertEqual(result["behavior"]["seed_product_id"], "demo-b")

    def test_personalized_onboarding(self):
        result = app.run(self.raw)
        step = result["onboarding"]["next_step"]
        self.assertEqual(step["id"], "learn")
        self.assertIn("Synthetic Alex", step["message"])
        self.assertIn("Synthetic Pocket Speaker", step["message"])
        self.assertEqual(result["behavior"]["onboarding_step"], step)

    def test_new_customer_profile(self):
        self.raw["customer"]["completed_steps"] = []
        self.assertEqual(app.run(self.raw)["onboarding"]["next_step"]["id"], "profile")

    def test_experienced_customer(self):
        self.raw["customer"]["experience"] = "experienced"
        self.assertEqual(app.run(self.raw)["onboarding"]["next_step"]["id"], "select")

    def test_onboarding_complete(self):
        self.raw["customer"]["completed_steps"] = list(app.STEPS)
        result = app.run(self.raw)["onboarding"]
        self.assertTrue(result["complete"])
        self.assertEqual(result["next_step"]["id"], "done")

    def test_explore_has_no_checkout(self):
        self.raw["customer"].update(goal="explore", completed_steps=["profile", "learn", "select"])
        self.assertTrue(app.run(self.raw)["onboarding"]["complete"])

    def test_cold_start(self):
        self.raw["events"] = []
        result = app.run(self.raw)
        self.assertTrue(result["behavior"]["cold_start"])
        self.assertEqual(result["behavior"]["ranking"][0]["product_id"], "demo-a")
        self.assertTrue(all(row["activity_score"] == 0 for row in result["behavior"]["ranking"]))

    def test_recency_decay(self):
        self.raw["events"] = [
            {"product_id": "demo-a", "kind": "browse", "at": self.raw["as_of"]},
            {"product_id": "demo-b", "kind": "browse", "at": "2026-08-25T10:00:00Z"}]
        scores = {r["product_id"]: r["activity_score"] for r in app.run(self.raw)["behavior"]["ranking"]}
        self.assertEqual(scores["demo-a"], 1)
        self.assertAlmostEqual(scores["demo-b"], .5)

    def test_purchase_weight(self):
        self.raw["events"] = [
            {"product_id": "demo-a", "kind": "browse", "at": self.raw["as_of"]},
            {"product_id": "demo-b", "kind": "purchase", "at": self.raw["as_of"]}]
        scores = {r["product_id"]: r["activity_score"] for r in app.run(self.raw)["behavior"]["ranking"]}
        self.assertAlmostEqual(scores["demo-a"], 1 / 3)
        self.assertEqual(scores["demo-b"], 1)

    def test_behavior_changes_order(self):
        self.raw["preferences"]["weights"] = dict(price_usd=0, weight_g=0, battery_hours=1)
        for p in self.raw["products"]:
            p["attributes"] = dict(price_usd=10, weight_g=10, battery_hours=10)
        self.raw["events"] = [{"product_id": "demo-c", "kind": "purchase", "at": self.raw["as_of"]}]
        result = app.run(self.raw)
        self.assertEqual(result["comparison"]["ranking"][0]["product_id"], "demo-a")
        self.assertEqual(result["behavior"]["ranking"][0]["product_id"], "demo-c")

    def test_single_product(self):
        self.raw["products"] = self.raw["products"][:1]
        self.raw["events"] = []
        self.assertEqual(app.run(self.raw)["comparison"]["ranking"][0]["score"], 1)

    def test_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.raw)
        result = app.run(self.raw)
        self.assertEqual(self.raw, before)
        self.assertEqual(app.run(self.raw), result)
        self.raw["events"].reverse()
        self.assertEqual(app.run(self.raw)["behavior"], result["behavior"])

    def test_invalid_numeric_inputs(self):
        for value in (-1, True, float("nan"), float("inf"), "10"):
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw)
                raw["products"][0]["attributes"]["price"] = value
                with self.assertRaises(app.ValidationError):
                    app.run(raw)

    def test_invalid_contracts(self):
        for key, value in (("products", []), ("half_life_days", 0), ("schema_version", 2),
                           ("synthetic", False), ("as_of", "2026-09-24"), ("events", {})):
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw)
                raw[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(raw)

    def test_invalid_events(self):
        for change in ({"product_id": "missing"}, {"kind": "click"}, {"at": "2027-01-01T00:00:00Z"}):
            with self.subTest(change=change):
                raw = copy.deepcopy(self.raw)
                raw["events"][0].update(change)
                with self.assertRaises(app.ValidationError):
                    app.run(raw)

    def test_alias_collision_and_duplicate_ids(self):
        self.raw["products"][0]["attributes"]["price_usd"] = 40
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)
        del self.raw["products"][0]["attributes"]["price_usd"]
        self.raw["products"][1]["id"] = "demo-a"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_boundary_rejects_tampering(self):
        state = app.compare(app.normalize(self.raw))
        state["comparison"]["ranking"][0]["product_id"] = "missing"
        with self.assertRaises(app.ValidationError):
            app.onboard(state)
        state = app.onboard(app.compare(app.normalize(self.raw)))
        state["onboarding"]["preferred_product_id"] = "missing"
        with self.assertRaises(app.ValidationError):
            app.personalize(state)

    def test_boundary_revalidates_context(self):
        for key, value in (("half_life_days", 0), ("events", [{"product_id": "missing", "kind": "browse",
                                                           "at": self.raw["as_of"]}])):
            with self.subTest(key=key):
                state = app.onboard(app.compare(app.normalize(self.raw)))
                state[key] = value
                with self.assertRaises(app.ValidationError):
                    app.personalize(state)

    def test_zero_weights_and_invalid_customer(self):
        self.raw["preferences"]["weights"] = dict.fromkeys(app.ATTRS, 0)
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)
        self.raw["preferences"]["weights"]["price_usd"] = 1
        self.raw["customer"]["completed_steps"] = ["profile", "profile"]
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "not-present.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for content in ("{", '{"x":NaN}', '{"x":1,"x":2}', '{}', '[]',
                        json.dumps({**self.raw, "half_life_days": 0})):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                    code = app.main(["virtual-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
