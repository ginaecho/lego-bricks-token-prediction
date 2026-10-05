import contextlib
import copy
import io
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self, mutate):
        mutate(self.raw)
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.raw)

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["stage"], "insights")
        self.assertEqual(result["status"], "ok")
        self.assertIs(app.validate(result, "insights"), result)

    def test_document_csv_extraction(self):
        result = app.documents(self.raw)
        self.assertEqual(len(result["telemetry"]), 8)
        self.assertIsInstance(result["telemetry"][0]["value"], float)
        self.assertNotIn("telemetry_csv", result)
        self.assertEqual(result["documents"]["record_counts"]["work_orders"], 2)

    def test_quality_decision_trace(self):
        result = app.documents(self.raw)
        decision = result["documents"]["quality_decisions"][0]
        self.assertEqual(decision["inspection_id"], "SYN-QI-001")
        self.assertEqual(decision["lot_id"], "SYN-LOT-A")
        self.assertEqual(decision["decision_by"], "Fictitious Inspector Ada")
        self.assertIn("tolerance", decision)
        self.assertEqual(decision["asset_id"], "SYN-PRESS-17")

    def test_customer_personalization(self):
        result = app.onboard(app.documents(self.raw))
        customers = result["onboarding"]["customers"]
        self.assertEqual(len(customers), 2)
        tasks = {task["task_id"]: task for task in result["onboarding"]["tasks"]}
        for customer in customers:
            next_step = tasks[customer["next_step_id"]]
            self.assertEqual(customer["customer_id"], next_step["customer_id"])
            self.assertIn(customer["customer_name"], next_step["instruction"])

    def test_safety_escalates_and_blocks(self):
        result = app.run_pipeline(self.raw)
        critical = [item for item in result["documents"]["alerts"]
                    if item["severity"] == "critical"]
        self.assertEqual(len(critical), 2)
        for alert in critical:
            task = next(task for task in result["onboarding"]["tasks"]
                        if task["source_alert_id"] == alert["alert_id"])
            self.assertEqual(task["status"], "awaiting_human")
            self.assertEqual(task["assignee_role"], "human_safety_lead")
            self.assertTrue(task["blocks_onboarding"])
        self.assertEqual(result["onboarding"]["automatic_machine_actions"], [])
        self.assertEqual(result["insights"]["summary"]["human_safety_escalations"], 2)

    def test_feedback_themes_and_task_propagation(self):
        result = app.run_pipeline(self.raw)
        themes = {theme["theme"]: theme for theme in result["insights"]["themes"]}
        self.assertIn("SYN-FB-001", themes["quality"]["feedback_ids"])
        self.assertIn("review:inspection:SYN-QI-001", themes["quality"]["task_ids"])
        self.assertIn("SYN-WO-001", themes["quality"]["work_order_ids"])
        self.assertTrue(themes["safety"]["human_review_required"])

    def test_immutable_and_deterministic(self):
        before = copy.deepcopy(self.raw)
        first = app.run_pipeline(self.raw)
        self.assertEqual(first, app.run_pipeline(self.raw))
        self.assertEqual(self.raw, before)
        doc = app.documents(self.raw)
        saved = copy.deepcopy(doc)
        app.onboard(doc)
        self.assertEqual(doc, saved)

    def test_seeded_plausible_telemetry(self):
        rng = random.Random(61)
        rows = [",".join(app.CSV_COLUMNS)]
        for minute in range(40):
            rows.append(
                f"SYN-R-{minute},2026-09-24T08:{minute:02d}:00Z,"
                f"SYN-WO-002,SYN-LATHE-24,temperature,{rng.uniform(45, 65):.3f},C"
            )
        self.raw["telemetry_csv"] = "\n".join(rows) + "\n"
        result = app.documents(self.raw)
        self.assertEqual(len(result["telemetry"]), 40)
        self.assertFalse(any(alert["source_type"] == "telemetry"
                             for alert in result["documents"]["alerts"]))

    def test_empty_optional_collections(self):
        self.raw["telemetry_csv"] = ",".join(app.CSV_COLUMNS) + "\n"
        for field in ("quality_inspections", "maintenance_logs", "feedback"):
            self.raw[field] = []
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["insights"]["summary"]["unique_feedback_count"], 0)
        self.assertEqual(result["documents"]["alerts"], [])
        self.assertTrue(all(customer["status"] == "ready_for_guided_setup"
                            for customer in result["onboarding"]["customers"]))
        self.assertEqual(result["insights"]["themes"][0]["theme"], "training")

    def test_no_work_orders(self):
        self.invalid(lambda raw: raw.update(work_orders=[]))

    def test_unknown_work_order(self):
        self.invalid(lambda raw: raw["feedback"][0].update(work_order_id="missing"))

    def test_bad_asset(self):
        self.invalid(lambda raw: raw.update(
            telemetry_csv=raw["telemetry_csv"].replace("SYN-PRESS-17", "OTHER-ASSET")))

    def test_unit_mismatch(self):
        self.invalid(lambda raw: raw.update(
            telemetry_csv=raw["telemetry_csv"].replace(",C\n", ",F\n")))

    def test_invalid_tolerance(self):
        self.invalid(lambda raw: raw["work_orders"][0]["tolerances"]["temperature"].update(min=101))

    def test_invalid_safety_threshold(self):
        self.invalid(lambda raw: raw["work_orders"][0]["tolerances"]["temperature"].update(
            safety_max=50))

    def test_nonfinite_and_boolean_measurements(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw)
                raw["quality_inspections"][0]["measurement"]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.documents(raw)

    def test_impossible_physical_value(self):
        self.invalid(lambda raw: raw.update(
            telemetry_csv=raw["telemetry_csv"].replace(",96.2,C", ",900,C")))

    def test_decision_actor_required(self):
        self.invalid(lambda raw: raw["quality_inspections"][0].update(decision_by=" "))

    def test_quality_lot_traceability(self):
        self.invalid(lambda raw: raw["quality_inspections"][0].update(lot_id="wrong"))

    def test_false_acceptance_rejected(self):
        self.invalid(lambda raw: raw["quality_inspections"][0].update(decision="accept"))

    def test_quality_rationale_required(self):
        self.invalid(lambda raw: raw["quality_inspections"][0].update(rationale=""))

    def test_timestamp_timezone_required(self):
        self.invalid(lambda raw: raw["quality_inspections"][0].update(
            decided_at="2026-09-24T09:00:00"))

    def test_duplicate_record_id(self):
        self.invalid(lambda raw: raw["feedback"].append(copy.deepcopy(raw["feedback"][0])))

    def test_malformed_csv_columns(self):
        self.invalid(lambda raw: raw.update(
            telemetry_csv=raw["telemetry_csv"] + "incomplete,row\n"))

    def test_duplicate_csv_header(self):
        self.invalid(lambda raw: raw.update(
            telemetry_csv=raw["telemetry_csv"].replace("reading_id,timestamp", "timestamp,timestamp")))

    def test_out_of_order_series(self):
        self.invalid(lambda raw: raw.update(
            telemetry_csv=raw["telemetry_csv"].replace("08:02:00Z", "07:02:00Z")))

    def test_tolerance_and_safety_boundaries(self):
        self.raw["telemetry_csv"] = (
            ",".join(app.CSV_COLUMNS) + "\n"
            "SYN-BOUND-1,2026-09-24T08:00:00Z,SYN-WO-001,SYN-PRESS-17,temperature,75,C\n"
            "SYN-BOUND-2,2026-09-24T08:01:00Z,SYN-WO-001,SYN-PRESS-17,temperature,90,C\n"
        )
        alerts = [alert for alert in app.documents(self.raw)["documents"]["alerts"]
                  if alert["source_type"] == "telemetry"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["severity"], "warning")

    def test_tampered_documents_handoff(self):
        state = app.documents(self.raw)
        state["documents"]["alerts"] = []
        with self.assertRaises(app.ValidationError):
            app.onboard(state)

    def test_tampered_onboarding_handoff(self):
        state = app.onboard(app.documents(self.raw))
        state["onboarding"]["tasks"][0]["assignee_role"] = "automatic_controller"
        with self.assertRaises(app.ValidationError):
            app.insights(state)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.insights(app.documents(self.raw))

    def test_unrecognized_feedback_theme(self):
        self.raw["feedback"][0]["text"] = "Our invoice contact changed."
        themes = app.run_pipeline(self.raw)["insights"]["themes"]
        other = next(theme for theme in themes if theme["theme"] == "other")
        self.assertIn("SYN-FB-001", other["feedback_ids"])

    def test_multiple_themes_not_duplicate_unique_count(self):
        self.raw["feedback"][0]["text"] = "Unsafe vibration and quality defects require training."
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["insights"]["summary"]["unique_feedback_count"], 3)
        matches = [theme for theme in result["insights"]["themes"]
                   if "SYN-FB-001" in theme["feedback_ids"]]
        self.assertEqual(len(matches), 4)

    def test_resolved_safety_log_does_not_create_open_alert(self):
        self.raw["maintenance_logs"][0]["status"] = "resolved"
        result = app.documents(self.raw)
        self.assertFalse(any(alert["source_type"] == "maintenance"
                             for alert in result["documents"]["alerts"]))

    def test_critical_inspection_escalates(self):
        self.raw["quality_inspections"][0]["measurement"]["value"] = 22
        result = app.run_pipeline(self.raw)
        task = next(task for task in result["onboarding"]["tasks"]
                    if task["source_alert_id"] == "inspection:SYN-QI-001")
        self.assertEqual(task["assignee_role"], "human_safety_lead")

    def test_cli_success(self):
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "nonexistent-input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")
        self.assertEqual(proc.stderr, "")

    def test_cli_usage_error(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(app.main([]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_bad_json_and_duplicate_keys(self):
        for value in ("{bad", '{"x":1,"x":2}', '{"x":NaN}', "[]"):
            with self.subTest(value=value):
                with patch("builtins.open", mock_open(read_data=value)):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(app.main(["synthetic-invalid.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_schema(self):
        self.raw["synthetic"] = False
        with patch("builtins.open", mock_open(read_data=json.dumps(self.raw))):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(app.main(["synthetic-invalid.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
