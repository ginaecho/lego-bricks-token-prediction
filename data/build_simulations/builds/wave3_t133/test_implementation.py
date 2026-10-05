import copy
import csv
import io
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_personalization_and_trace(self):
        result = app.run(self.data)
        self.assertEqual(result["mode"], "personalized")
        self.assertEqual([r["work_order_id"] for r in result["rankings"]],
                         ["SYN-WO-103", "SYN-WO-101", "SYN-WO-102"])
        self.assertEqual(result["rankings"][1]["quality_trace"][0]["source_reading_ids"], ["SYN-R1"])
        self.assertEqual(result["rankings"][2]["score"], 0.5)

    def test_recency_decay(self):
        self.data["events"][0]["timestamp"] = "2026-09-10T12:00:00Z"
        scores = {r["work_order_id"]: r["score"] for r in app.run(self.data)["rankings"]}
        self.assertEqual(scores["SYN-WO-101"], 1.5)

    def test_cold_start_and_user_isolation(self):
        self.data["user_id"] = "new-synthetic-user"
        result = app.run(self.data)
        self.assertEqual(result["mode"], "cold_start")
        self.assertEqual([r["work_order_id"] for r in result["rankings"]],
                         ["SYN-WO-103", "SYN-WO-102", "SYN-WO-101"])
        self.assertTrue(all(r["score"] == 0 for r in result["rankings"]))

    def test_empty_catalog(self):
        self.data.update(work_orders=[], events=[], quality_inspections=[],
                         telemetry_csv=",".join(app.CSV_COLUMNS) + "\n")
        self.assertEqual(app.run(self.data)["rankings"], [])

    def test_safety_escalation_survives_limit(self):
        self.data["limit"] = 1
        result = app.run(self.data)
        self.assertEqual(result["rankings"][0]["work_order_id"], "SYN-WO-103")
        self.assertEqual(result["alerts"][0]["escalate_to"], "human")
        self.assertFalse(result["alerts"][0]["automatic_action_taken"])
        self.assertEqual(result["alerts"][0]["state"], "pending_human_review")

    def test_invalid_units_and_bounds(self):
        for key, value in [("unit", "F"), ("tolerance_min", 90), ("safety_max", 70),
                           ("tolerance_max", float("nan")), ("safety_min", True)]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["work_orders"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_traceability_required(self):
        for key, value in [("decision_by", ""), ("reason", ""), ("source_reading_ids", []),
                           ("source_reading_ids", ["SYN-R2"]), ("decision", "fail"),
                           ("timestamp", "2026-09-24T08:00:00Z")]:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["quality_inspections"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_invalid_csv(self):
        for old, new in [("74.31", "NaN"), ("74.31", "500"),
                         (",C\n", ",F\n"), ("reading_id,", "bad_header,")]:
            with self.subTest(new=new):
                data = copy.deepcopy(self.data)
                data["telemetry_csv"] = data["telemetry_csv"].replace(old, new)
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_invalid_events(self):
        for key, value in [("timestamp", "2027-01-01T00:00:00Z"),
                           ("timestamp", "2026-09-01T00:00:00"),
                           ("work_order_id", "unknown"), ("action", "click")]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["events"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_invalid_configuration(self):
        for key, value in [("half_life_days", 0), ("limit", True), ("synthetic", False),
                           ("schema_version", "2.0"), ("events", {})]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_duplicate_ids(self):
        self.data["events"].append(copy.deepcopy(self.data["events"][0]))
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_deterministic_nonmutating(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, before)

    def test_randomized_plausible_synthetic_telemetry(self):
        rng = random.Random(133)
        rows = list(csv.DictReader(io.StringIO(self.data["telemetry_csv"])))
        for _ in range(25):
            rows[0]["value"] = str(round(rng.uniform(65, 80), 3))
            rows[1]["value"] = str(round(rng.uniform(3.5, 5.5), 3))
            rows[2]["value"] = str(round(rng.uniform(7.5, 10), 3))
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=app.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            self.data["telemetry_csv"] = buffer.getvalue()
            result = app.run(self.data)
            self.assertEqual(len(result["alerts"]), 1)

    def test_same_asset_affinity(self):
        order = copy.deepcopy(self.data["work_orders"][0])
        order["id"] = "SYN-WO-104"
        self.data["work_orders"].append(order)
        rows = {r["work_order_id"]: r for r in app.run(self.data)["rankings"]}
        self.assertAlmostEqual(rows["SYN-WO-104"]["score"], rows["SYN-WO-101"]["score"] * 0.4, places=7)
        self.assertEqual(rows["SYN-WO-104"]["quality_status"], "uninspected")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_errors(self):
        for args in [[], [str(ROOT / "does-not-exist.json")], [str(ROOT / "implementation.py")]]:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_schema_error_without_extra_files(self):
        output = io.StringIO()
        with patch("builtins.open", return_value=io.StringIO('{"schema_version": "wrong"}')):
            with patch("sys.stdout", output):
                code = app.main(["example_input.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"a": 1, "a": 2}', object_pairs_hook=app.unique_object)


if __name__ == "__main__":
    unittest.main()
