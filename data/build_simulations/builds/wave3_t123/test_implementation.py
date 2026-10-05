import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_full_pipeline(self):
        result = app.run(self.data)
        self.assertEqual(result["stage"], "onboard")
        self.assertEqual(result["status"], "ok")

    def test_document_numeric_extraction(self):
        state = app.documents(self.data)
        self.assertIsInstance(state["readings"][0]["value"], float)
        self.assertEqual(len(state["readings"]), 12)

    def test_quality_traceability(self):
        state = app.documents(self.data)
        decision = state["quality_decisions"][0]
        self.assertEqual(decision["inspection_id"], "QI-101")
        self.assertEqual(decision["inspector"], "Inspector Fictitious-7")
        self.assertEqual(decision["outcome"], "fail")
        self.assertIn("tolerance", decision)

    def test_insight_maintenance_feedback(self):
        state = app.insights(app.documents(self.data))
        theme = next(t for t in state["themes"] if t["theme"] == "maintenance")
        self.assertIn("feedback:FB-1", theme["evidence"])

    def test_quality_to_theme_handoff(self):
        state = app.insights(app.documents(self.data))
        theme = next(t for t in state["themes"] if t["theme"] == "quality")
        self.assertIn("decision-QI-101", theme["evidence"])

    def test_search_synonyms(self):
        self.data["customer"]["query"] = "shaking"
        state = app.search(app.insights(app.documents(self.data)))
        self.assertEqual(state["search_results"][0]["product_id"], "P-BEARING")
        self.assertIn("vibration", state["search_results"][0]["query_matches"])

    def test_insights_to_search_handoff(self):
        state = app.search(app.insights(app.documents(self.data)))
        self.assertTrue(any(r["theme_ids"] for r in state["search_results"]))
        theme_ids = {t["id"] for t in state["themes"]}
        self.assertTrue(all(set(r["theme_ids"]) <= theme_ids for r in state["search_results"]))

    def test_search_to_onboarding_handoff(self):
        state = app.run(self.data)
        self.assertEqual(state["onboarding"]["recommended_product_id"], state["search_results"][0]["product_id"])

    def test_safety_escalation(self):
        state = app.run(self.data)
        self.assertTrue(state["alerts"])
        self.assertTrue(all(a["escalate_to"] == "human_safety_supervisor" for a in state["alerts"]))
        self.assertEqual(state["onboarding"]["next_step"]["owner"], "human_safety_supervisor")
        self.assertEqual(state["onboarding"]["operational_status"], "blocked_pending_human_review")

    def test_inspection_alone_can_escalate(self):
        self.data["sensor_csv"] = self.data["sensor_csv"].splitlines()[0] + "\n"
        self.data["inspections"][0]["value"] = 110
        state = app.run(self.data)
        self.assertEqual(state["alerts"][0]["source_type"], "inspection")

    def test_new_customer_training(self):
        state = app.run(self.data)
        self.assertTrue(any("orientation" in step["action"] for step in state["onboarding"]["steps"]))

    def test_experienced_customer(self):
        self.data["customer"]["experience"] = "experienced"
        state = app.run(self.data)
        self.assertFalse(any("orientation" in step["action"] for step in state["onboarding"]["steps"]))

    def test_empty_catalog(self):
        self.data["products"] = []
        state = app.run(self.data)
        self.assertIsNone(state["onboarding"]["recommended_product_id"])
        self.assertIn("clarify", state["onboarding"]["steps"][-1]["action"])

    def test_incompatible_products_excluded(self):
        for product in self.data["products"]:
            product["compatible_assets"] = ["OTHER-ASSET"]
        self.assertEqual(app.run(self.data)["search_results"], [])

    def test_unit_mismatch(self):
        self.data["sensor_csv"] = self.data["sensor_csv"].replace("degC", "degF")
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_tolerance(self):
        self.data["work_orders"][0]["specifications"]["temperature"]["min"] = 90
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_missing_inspector(self):
        del self.data["inspections"][0]["inspector"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_unknown_work_order(self):
        self.data["feedback"][0]["work_order_id"] = "MISSING"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_nonfinite_sensor(self):
        lines = self.data["sensor_csv"].splitlines()
        parts = lines[1].split(",")
        parts[4] = "NaN"
        lines[1] = ",".join(parts)
        self.data["sensor_csv"] = "\n".join(lines)
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_duplicate_reading(self):
        self.data["sensor_csv"] += self.data["sensor_csv"].splitlines()[1] + "\n"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_timestamp_timezone_required(self):
        self.data["inspections"][0]["timestamp"] = "2026-09-24T10:00:00"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_boundary_inclusive(self):
        self.data["inspections"][0]["value"] = 80
        state = app.documents(self.data)
        self.assertEqual(state["quality_decisions"][0]["outcome"], "pass")

    def test_tampered_stage_rejected(self):
        state = app.documents(self.data)
        state["quality_decisions"][0]["outcome"] = "pass"
        with self.assertRaises(app.ValidationError):
            app.insights(state)

    def test_tampered_safety_alert_rejected(self):
        state = app.documents(self.data)
        state["alerts"] = []
        with self.assertRaises(app.ValidationError):
            app.insights(state)

    def test_stage_order_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.search(app.documents(self.data))

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(original, self.data)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "nonexistent.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_json(self):
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data="{invalid")), redirect_stdout(output):
            code = app.main(["invalid.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_validation_error(self):
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data='{"schema_version":"wrong"}')), redirect_stdout(output):
            code = app.main(["invalid.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_usage_error(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(app.main([]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
