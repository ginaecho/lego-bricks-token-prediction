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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_determinism(self):
        self.assertEqual(app.pipeline(self.data), app.pipeline(copy.deepcopy(self.data)))

    def test_dedup_preserves_all_excerpts(self):
        result = app.feedback_stage(self.data)["feedback"]
        self.assertEqual(result["raw_count"], 4)
        self.assertEqual(result["unique_count"], 3)
        group = next(g for g in result["deduplicated"] if g["id"] == "F1")
        self.assertEqual([e["feedback_id"] for e in group["evidence"]], ["F1", "F2"])
        self.assertEqual(group["evidence"][0]["excerpt"], self.data["feedback"][0]["text"])

    def test_theme_extraction(self):
        self.assertEqual(app.themes_for("Temperature and vibration"), ["thermal", "vibration"])
        self.assertEqual(app.themes_for("Invented neutral report"), ["general"])

    def test_recency_decay(self):
        self.data["events"] = [
            {"id": "E1", "user_id": "operator-demo", "work_order_id": "WO-100",
             "type": "browse", "timestamp": "2026-08-25T10:00:00Z"}]
        rank = app.pipeline(self.data)["behavior"]["rankings"]
        self.assertEqual(next(r for r in rank if r["work_order_id"] == "WO-100")[
            "direct_recency_score"], 0.5)

    def test_purchase_weight(self):
        self.data["events"] = [self.data["events"][0]]
        event = self.data["events"][0]
        event["timestamp"] = self.data["as_of"]
        event["type"] = "purchase"
        rows = app.pipeline(self.data)["behavior"]["rankings"]
        self.assertEqual(next(r for r in rows if r["work_order_id"] == event["work_order_id"])[
            "direct_recency_score"], 3)

    def test_cold_start_and_tie_breaking(self):
        self.data["events"] = []
        self.data["feedback"] = []
        result = app.pipeline(self.data)["behavior"]
        self.assertTrue(result["cold_start"])
        self.assertEqual([r["work_order_id"] for r in result["rankings"]], ["WO-100", "WO-200"])

    def test_other_users_do_not_personalize(self):
        for event in self.data["events"]:
            event["user_id"] = "someone-else"
        self.assertTrue(app.pipeline(self.data)["behavior"]["cold_start"])

    def test_feedback_changes_downstream_theme_score(self):
        self.data["events"] = [{
            "id": "X", "user_id": "operator-demo", "work_order_id": "WO-100",
            "type": "browse", "timestamp": self.data["as_of"]}]
        before = app.pipeline(self.data)
        self.data["feedback"][-1]["text"] = "Temperature maintenance concern."
        after = app.pipeline(self.data)
        find = lambda out: next(r for r in out["behavior"]["rankings"] if r["work_order_id"] == "WO-200")
        self.assertGreater(find(after)["theme_affinity_score"], find(before)["theme_affinity_score"])
        self.assertIn("F4", find(after)["supporting_feedback_ids"])

    def test_tampered_handoff_rejected(self):
        envelope = app.feedback_stage(self.data)
        envelope["feedback"]["deduplicated"][0]["evidence"][0]["excerpt"] = "fabricated"
        with self.assertRaises(ValueError):
            app.behavior_stage(envelope)

    def test_safety_escalates_both_stages(self):
        output = app.pipeline(self.data)
        alerts = output["feedback"]["human_escalations"]
        self.assertTrue(any(a["source_type"] == "sensor" for a in alerts))
        self.assertTrue(any(a["source_type"] == "feedback" for a in alerts))
        self.assertEqual(alerts, output["behavior"]["human_escalations"])
        self.assertTrue(all(not r["automatic_execution_allowed"] for r in output["behavior"]["rankings"]))

    def test_duplicate_safety_flag_not_lost(self):
        self.data["feedback"][0]["safety_critical"] = False
        self.data["feedback"][1]["safety_critical"] = True
        group = next(g for g in app.feedback_stage(self.data)["feedback"]["deduplicated"] if g["id"] == "F1")
        self.assertTrue(group["safety_critical"])

    def test_quality_trace(self):
        trace = app.pipeline(self.data)["feedback"]["quality_decision_trace"][0]
        self.assertEqual(trace["inspector_id"], "SYN-INSPECTOR-01")
        self.assertEqual(trace["applied_tolerance"]["unit"], "C")
        self.assertTrue(trace["decision_reason"])

    def test_invalid_units(self):
        self.data["sensor_csv"] = self.data["sensor_csv"].replace(",C", ",F")
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_invalid_tolerance(self):
        self.data["work_orders"][0]["tolerances"]["temperature"]["min"] = 100
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_wrong_quality_decision(self):
        self.data["quality_inspections"][0]["decision"] = "fail"
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_missing_traceability(self):
        del self.data["quality_inspections"][0]["inspector_id"]
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_nonfinite_input(self):
        self.data["quality_inspections"][0]["value"] = float("nan")
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_invalid_reference(self):
        self.data["feedback"][0]["inspection_id"] = "absent"
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_future_event(self):
        self.data["events"][0]["timestamp"] = "2030-01-01T00:00:00Z"
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_duplicate_ids(self):
        self.data["events"].append(copy.deepcopy(self.data["events"][0]))
        with self.assertRaises(ValueError):
            app.pipeline(self.data)

    def test_empty_sensor_and_feedback_allowed(self):
        self.data["feedback"] = []
        self.data["sensor_csv"] = self.data["sensor_csv"].splitlines()[0] + "\n"
        self.assertEqual(app.pipeline(self.data)["feedback"]["unique_count"], 0)

    def test_closed_orders_not_ranked(self):
        for order in self.data["work_orders"]:
            order["status"] = "closed"
        self.assertEqual(app.pipeline(self.data)["behavior"]["rankings"], [])

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              capture_output=True, text=True, cwd=ROOT)

    def test_cli_success(self):
        proc = self.cli("example_input.json")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_file_error(self):
        proc = self.cli("does-not-exist.json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_validation_error(self):
        proc = self.cli("build_manifest.json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_usage_error(self):
        proc = self.cli()
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")


if __name__ == "__main__":
    unittest.main()
