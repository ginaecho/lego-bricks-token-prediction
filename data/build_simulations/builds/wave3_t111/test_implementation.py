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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)["stages"]

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_integrated_pipeline(self):
        result = self.run_data()
        self.assertEqual(tuple(result), app.STAGES)
        self.assertEqual(result["triage"]["selected_product_id"], result["compare"]["selected_product_id"])
        self.assertEqual(result["onboard"]["ticket_id"], result["triage"]["ticket_id"])
        self.assertEqual(result["interests"]["next_step"], result["onboard"]["next_step"])

    def test_normalization(self):
        context = app.validate_input(self.data)
        self.assertEqual(context["normalized_products"][0]["attributes"]["lead_hours"], 48)
        self.assertEqual(context["normalized_products"][0]["attributes"]["vibration_reduction_pct"], 60)

    def test_preference_ranking(self):
        self.data["preferences"]["weights"] = {"price_usd": 1, "lead_hours": 0, "vibration_reduction_pct": 0}
        result = self.run_data()["compare"]
        self.assertEqual(result["selected_product_id"], "KIT-B")
        self.assertEqual(result["ranking"][0]["score"], 1)
        self.assertEqual(result["columns"]["lead_hours"], "h")

    def test_tie_break(self):
        self.data["products"][1]["attributes"] = copy.deepcopy(self.data["products"][0]["attributes"])
        self.assertEqual(self.run_data()["compare"]["selected_product_id"], "KIT-A")

    def test_configurable_routing(self):
        self.data["routing"]["procurement"]["owner"] = "Fictitious Buyer"
        self.data["routing"]["procurement"]["priority"] = "high"
        ticket = self.run_data()["triage"]
        self.assertEqual(ticket["accountable_owner"], "Fictitious Buyer")
        self.assertEqual(ticket["priority"], "high")

    def test_safety_escalation_and_hold(self):
        lines = self.data["telemetry_csv"].splitlines()
        parts = lines[1].split(",")
        parts[2] = "90"
        lines[1] = ",".join(parts)
        self.data["telemetry_csv"] = "\n".join(lines)
        stages = self.run_data()
        self.assertEqual(stages["triage"]["category"], "safety")
        self.assertTrue(stages["triage"]["requires_human"])
        self.assertEqual(stages["triage"]["priority"], "critical")
        self.assertTrue(stages["onboard"]["blocked"])
        self.assertEqual(stages["interests"]["recommendations"], [])

    def test_safety_threshold_inclusive(self):
        self.data["telemetry_csv"] = (
            "timestamp,asset_id,temperature,temperature_unit,vibration,vibration_unit\n"
            "2026-09-24T08:00:00Z,SYN-ASSET-111,85,C,2,mm/s\n")
        self.assertEqual(self.run_data()["triage"]["category"], "safety")

    def test_telemetry_tolerance_routing(self):
        self.data["telemetry_csv"] = (
            "timestamp,asset_id,temperature,temperature_unit,vibration,vibration_unit\n"
            "2026-09-24T08:00:00Z,SYN-ASSET-111,70,C,2,mm/s\n")
        ticket = self.run_data()["triage"]
        self.assertEqual(ticket["category"], "telemetry")
        self.assertEqual(ticket["evidence"][0]["kind"], "tolerance")
        self.assertEqual(ticket["accountable_owner"], self.data["routing"]["telemetry"]["owner"])

    def test_tolerance_boundary_allowed(self):
        self.data["telemetry_csv"] = (
            "timestamp,asset_id,temperature,temperature_unit,vibration,vibration_unit\n"
            "2026-09-24T08:00:00Z,SYN-ASSET-111,65,C,3,mm/s\n")
        self.assertEqual(self.run_data()["triage"]["category"], "procurement")

    def test_quality_trace_and_hold(self):
        item = self.data["quality_inspections"][0]
        item["decision"], item["measured"] = "reject", 20.5
        stages = self.run_data()
        self.assertEqual(stages["triage"]["category"], "quality")
        self.assertEqual(stages["triage"]["quality_decision_trace"][0], item)
        self.assertEqual(stages["onboard"]["next_step"], "await_quality_review")

    def test_onboarding_personalized(self):
        result = self.run_data()["onboard"]
        self.assertIn(self.data["customer"]["name"], result["instruction"])
        self.assertEqual(result["next_step"], "confirm_profile")
        self.assertFalse(result["work_authorized"])

    def test_onboarding_completed_steps(self):
        self.data["customer"]["completed_steps"] = ["confirm_profile"]
        self.assertEqual(self.run_data()["onboard"]["next_step"], "review_work_order")

    def test_experienced_customer(self):
        self.data["customer"]["experience"] = "experienced"
        self.assertEqual(self.run_data()["onboard"]["next_step"], "review_work_order")

    def test_grounded_interests(self):
        results = self.run_data()["interests"]["recommendations"]
        products = {p["id"]: p for p in self.data["products"]}
        for row in results:
            explanation = row["explanation"]
            self.assertEqual(explanation["matched_interests"], sorted(
                set(products[row["product_id"]]["tags"]) & set(self.data["customer"]["interests"])))
            self.assertEqual(explanation["onboarding_next_step"], "confirm_profile")

    def test_interest_changes_ranking(self):
        self.data["products"][1]["attributes"] = copy.deepcopy(self.data["products"][0]["attributes"])
        self.data["customer"]["interests"] = ["fast"]
        self.assertEqual(self.run_data()["interests"]["recommendations"][0]["product_id"], "KIT-B")

    def test_empty_interests_and_limit(self):
        self.data["customer"]["interests"] = []
        self.data["preferences"]["recommendation_limit"] = 1
        result = self.run_data()["interests"]["recommendations"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["explanation"]["matched_interests"], [])

    def test_exclusions_both_stages(self):
        self.data["preferences"]["excluded_product_ids"] = ["KIT-A"]
        stages = self.run_data()
        self.assertEqual(stages["compare"]["selected_product_id"], "KIT-B")
        self.assertEqual([r["product_id"] for r in stages["interests"]["recommendations"]], ["KIT-B"])

    def test_tag_exclusion(self):
        self.data["preferences"]["excluded_tags"] = ["fast"]
        self.assertEqual(len(self.run_data()["interests"]["recommendations"]), 1)

    def test_no_eligible_product(self):
        self.data["preferences"]["budget_usd"] = 0
        stages = self.run_data()
        self.assertIsNone(stages["compare"]["selected_product_id"])
        self.assertEqual(stages["onboard"]["next_step"], "resolve_product_preferences")
        self.assertEqual(stages["interests"]["recommendations"], [])

    def test_incompatible_products(self):
        for product in self.data["products"]:
            product["compatible_assets"] = ["OTHER-SYNTHETIC-ASSET"]
        self.assertEqual(self.run_data()["compare"]["ranking"], [])

    def test_invalid_units(self):
        self.data["telemetry_csv"] = self.data["telemetry_csv"].replace(",C,", ",F,")
        self.invalid()

    def test_invalid_tolerance(self):
        self.data["sensor_specs"]["temperature"]["tolerance"] = -1
        self.invalid()

    def test_invalid_traceability(self):
        self.data["quality_inspections"][0]["work_order_id"] = "OTHER"
        self.invalid()

    def test_missing_quality_reason(self):
        del self.data["quality_inspections"][0]["reason"]
        self.invalid()

    def test_accept_outside_tolerance(self):
        self.data["quality_inspections"][0]["measured"] = 25
        self.invalid()

    def test_invalid_safety_routing(self):
        self.data["routing"]["safety"]["human"] = False
        self.invalid()

    def test_malformed_csv(self):
        self.data["telemetry_csv"] += "bad,row\n"
        self.invalid()

    def test_nonmonotonic_telemetry(self):
        lines = self.data["telemetry_csv"].splitlines()
        self.data["telemetry_csv"] = "\n".join([lines[0], lines[2], lines[1]])
        self.invalid()

    def test_nonfinite_reading(self):
        lines = self.data["telemetry_csv"].splitlines()
        row = lines[1].split(",")
        row[2] = "NaN"
        self.data["telemetry_csv"] = "\n".join([lines[0], ",".join(row)])
        self.invalid()

    def test_duplicate_products(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        self.invalid()

    def test_invalid_weights(self):
        self.data["preferences"]["weights"] = {k: 0 for k in app.METRICS}
        self.invalid()

    def test_conflicting_normalization(self):
        self.data["products"][0]["attributes"]["lead_hours"] = {"value": 48, "unit": "h"}
        self.invalid()

    def test_tampered_handoff_rejected(self):
        comparison = app.advance(self.data, "compare")
        comparison["stages"]["compare"]["selected_product_id"] = "UNKNOWN"
        with self.assertRaises(app.ValidationError):
            app.advance(comparison, "triage")

    def test_wrong_stage_rejected(self):
        comparison = app.advance(self.data, "compare")
        with self.assertRaises(app.ValidationError):
            app.advance(comparison, "interests")

    def test_tampered_later_handoffs(self):
        comparison = app.advance(self.data, "compare")
        triage = app.advance(comparison, "triage")
        onboarding = app.advance(triage, "onboard")
        triage["stages"]["triage"]["accountable_owner"] = "unaccountable"
        with self.assertRaises(app.ValidationError):
            app.advance(triage, "onboard")
        onboarding["stages"]["onboard"]["interests"] = ["fabricated"]
        with self.assertRaises(app.ValidationError):
            app.advance(onboarding, "interests")

    def test_deterministic_nonmutating(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_file_error(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "nonexistent.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["status"], "error")

    def test_cli_validation_error(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "build_manifest.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["status"], "error")

    def test_cli_usage_error(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["status"], "error")

    def test_cli_malformed_json(self):
        output = io.StringIO()
        with patch("builtins.open", return_value=io.StringIO("{")), redirect_stdout(output):
            code = app.main(["malformed.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_fixture_reproducibility(self):
        import random
        rng = random.Random(self.data["fixture_metadata"]["random_seed"])
        readings = app.validate_input(self.data)["readings"]
        for reading in readings:
            self.assertEqual(reading["temperature"], round(rng.uniform(58, 62), 3))
            self.assertEqual(reading["vibration"], round(rng.uniform(1.7, 2.3), 3))


if __name__ == "__main__":
    unittest.main()
