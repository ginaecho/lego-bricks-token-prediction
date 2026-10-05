import copy
import csv
import io
import json
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def rewrite_sensor(self, key, value, index=0):
        rows = list(csv.DictReader(io.StringIO(self.data["sensor_csv"])))
        rows[index][key] = value
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=app.SENSOR_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        self.data["sensor_csv"] = stream.getvalue()

    def test_normal_ranking_and_grounded_trace(self):
        result = app.recommend(self.data)
        self.assertEqual([r["score"] for r in result["recommendations"]], [7, 2])
        first = result["recommendations"][0]
        self.assertIn("quality (+3)", first["explanation"])
        self.assertEqual(first["quality_decision_trace"], self.data["quality_inspections"][:1])
        self.assertEqual(first["sensor_evidence"][0]["reading_id"], "SYN-R-001")

    def test_deterministic_and_does_not_mutate(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.recommend(self.data), app.recommend(self.data))
        self.assertEqual(original, self.data)

    def test_each_exclusion_is_enforced(self):
        for key, value in (
            ("excluded_work_order_ids", "SYN-WO-001"),
            ("excluded_asset_ids", "SYN-MILL-001"),
            ("excluded_tags", "quality"),
        ):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["preferences"][key] = [value]
                result = app.recommend(data)
                self.assertNotIn("SYN-WO-001", [r["work_order_id"] for r in result["recommendations"]])

    def test_excluded_safety_alerts_still_escalate(self):
        result = app.recommend(self.data)
        self.assertEqual(len(result["escalations"]), 3)
        self.assertTrue(all(e["route"] == "human_safety_reviewer" for e in result["escalations"]))
        self.assertTrue(all(e["work_order_id"] == "SYN-WO-003" for e in result["escalations"]))

    def test_safety_markers_and_review_flag(self):
        self.data["preferences"]["excluded_asset_ids"] = []
        self.rewrite_sensor("safety_critical", "false", index=4)
        result = app.recommend(self.data)
        third = next(r for r in result["recommendations"] if r["work_order_id"] == "SYN-WO-003")
        self.assertTrue(third["requires_human_review"])
        self.assertEqual(len(result["escalations"]), 3)

    def test_empty_interests_preserve_alerts(self):
        self.data["preferences"]["interests"] = {}
        result = app.recommend(self.data)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(len(result["escalations"]), 3)

    def test_limit_and_tie_break(self):
        self.data["preferences"]["interests"] = {"quality": 2, "energy_efficiency": 2}
        self.data["preferences"]["limit"] = 1
        result = app.recommend(self.data)
        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual(result["recommendations"][0]["work_order_id"], "SYN-WO-001")

    def test_closed_orders_not_recommended(self):
        self.data["work_orders"][0]["status"] = "closed"
        self.assertEqual(len(app.recommend(self.data)["recommendations"]), 1)

    def test_invalid_units_and_tolerances(self):
        for key, value in (("unit", "F"), ("lower", "90"), ("value", "NaN"),
                           ("value", "501"), ("safety_critical", "yes")):
            with self.subTest(key=key, value=value):
                original = self.data["sensor_csv"]
                self.rewrite_sensor(key, value)
                with self.assertRaises(app.ValidationError):
                    app.recommend(self.data)
                self.data["sensor_csv"] = original

    def test_quality_trace_required(self):
        for key in ("inspector_id", "procedure_revision", "rationale", "timestamp"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["quality_inspections"][0][key] = ""
                with self.assertRaises(app.ValidationError):
                    app.recommend(data)
        self.data["quality_inspections"] = self.data["quality_inspections"][1:]
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_quality_decision_and_boundary(self):
        self.data["quality_inspections"][0]["value"] = 20.05
        app.recommend(self.data)
        self.data["quality_inspections"][0]["value"] = 20.051
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_foreign_keys_and_timestamp(self):
        for key, value in (("work_order_id", "UNKNOWN"), ("asset_id", "UNKNOWN"),
                           ("timestamp", "2026-09-24T08:00:00")):
            with self.subTest(key=key):
                original = self.data["sensor_csv"]
                self.rewrite_sensor(key, value)
                with self.assertRaises(app.ValidationError):
                    app.recommend(self.data)
                self.data["sensor_csv"] = original

    def test_duplicate_ids_and_csv_structure(self):
        self.data["work_orders"].append(copy.deepcopy(self.data["work_orders"][0]))
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)
        self.data["work_orders"].pop()
        self.data["sensor_csv"] += "short,row\n"
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_invalid_preferences_and_synthetic_marker(self):
        for weight in (True, 0, 11, 1.2, "4"):
            self.data["preferences"]["interests"]["quality"] = weight
            with self.assertRaises(app.ValidationError):
                app.recommend(self.data)
        self.data["preferences"]["interests"]["quality"] = 3
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_empty_dataset(self):
        self.data["work_orders"] = []
        self.data["quality_inspections"] = []
        self.data["sensor_csv"] = ",".join(app.SENSOR_COLUMNS) + "\n"
        result = app.recommend(self.data)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["escalations"], [])

    def test_seeded_plausible_randomized_telemetry(self):
        rng = random.Random(self.data["fixture_seed"])
        expected = [round(rng.uniform(lo, hi), 3) for lo, hi in
                    [(50, 60), (60, 65), (4, 5.5), (4, 7), (8, 10), (8, 10)]]
        rows = list(csv.DictReader(io.StringIO(self.data["sensor_csv"])))
        self.assertEqual([float(r["value"]) for r in rows], expected)
        app.recommend(self.data)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "nonexistent.json")]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")

    def test_cli_malformed_validation_and_duplicate_json(self):
        for content in ("{", "[]", '{"a": 1, "a": 2}',
                        json.dumps({**self.data, "schema_version": "unsupported"})):
            with self.subTest(content=content[:30]):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=content), redirect_stdout(output):
                    code = app.main(["invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
