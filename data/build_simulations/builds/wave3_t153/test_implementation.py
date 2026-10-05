import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def work_order(self, **changes):
        order = json.loads(self.payload["documents"]["work_order"])
        order.update(changes)
        self.payload["documents"]["work_order"] = json.dumps(order)

    def test_normal_and_deterministic(self):
        result = app.extract(self.payload)
        self.assertEqual(result, app.extract(copy.deepcopy(self.payload)))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["quality_trace"]["complete"])
        self.assertEqual(result["missing_fields"], [])
        self.assertFalse(result["quality_trace"]["release_authorized"])

    def test_source_spans_all_fields(self):
        result = app.extract(self.payload)
        for entity, rows in result["entities"].items():
            for row in rows:
                for field in row.values():
                    span = field["source_span"]
                    raw = self.payload["documents"][entity][span["start"]:span["end"]]
                    if entity == "work_order":
                        self.assertEqual(json.loads(raw), field["value"])
                    else:
                        self.assertEqual(raw, str(field["value"]))

    def test_missing_quality_trace(self):
        self.payload["documents"]["inspection"] = self.payload["documents"]["inspection"].replace(
            "inspector_id: SYN-INSPECTOR-3\n", "")
        result = app.extract(self.payload)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["quality_trace"]["complete"])
        self.assertIn({"entity": "inspection", "record": 0, "field": "inspector_id"},
                      result["missing_fields"])

    def test_no_sensor_rows(self):
        self.payload["documents"]["telemetry"] = "timestamp,asset_id\n"
        result = app.extract(self.payload)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["entities"]["telemetry"], [])

    def test_units_rejected(self):
        self.payload["documents"]["telemetry"] = self.payload["documents"]["telemetry"].replace(
            "degC", "degF")
        with self.assertRaisesRegex(app.ValidationError, "unit"):
            app.extract(self.payload)

    def test_tolerances_rejected(self):
        for tolerance in (-0.1, 26, True, "NaN"):
            with self.subTest(tolerance=tolerance):
                self.work_order(tolerance=tolerance)
                with self.assertRaises(app.ValidationError):
                    app.extract(self.payload)

    def test_quality_boundary_and_contradiction(self):
        inspection = self.payload["documents"]["inspection"]
        self.payload["documents"]["inspection"] = inspection.replace("25.06", "25.1")
        self.assertEqual(app.extract(self.payload)["quality_trace"]["expected_decision"], "pass")
        self.payload["documents"]["inspection"] = inspection.replace("25.06", "25.11")
        with self.assertRaisesRegex(app.ValidationError, "contradicts"):
            app.extract(self.payload)
        self.payload["documents"]["inspection"] = self.payload["documents"]["inspection"].replace(
            "decision: pass", "decision: fail")
        self.assertEqual(app.extract(self.payload)["quality_trace"]["decision"], "fail")

    def test_asset_traceability(self):
        self.work_order(asset_id="SYN-WRONG-ASSET")
        with self.assertRaisesRegex(app.ValidationError, "traceability"):
            app.extract(self.payload)

    def test_safety_escalates_and_never_claims_delivery(self):
        result = app.extract(self.payload)
        alert = result["alerts"][0]
        self.assertEqual(alert["action"], "escalate_to_human")
        self.assertTrue(alert["human_review_required"])
        self.assertEqual(alert["delivery_status"], "pending_external_delivery")
        self.assertEqual(alert["record"], 2)

    def test_seeded_physically_plausible_randomized_telemetry(self):
        rng = random.Random(153)
        rows = ["timestamp,asset_id,temperature,temperature_unit,vibration,vibration_unit"]
        for minute in range(20):
            rows.append(f"2026-09-24T08:{minute:02}:00Z,SYN-MILL-42,"
                        f"{rng.uniform(62, 73):.2f},degC,{rng.uniform(1.5, 3):.2f},mm/s")
        self.payload["documents"]["telemetry"] = "\n".join(rows)
        result = app.extract(self.payload)
        self.assertEqual(len(result["entities"]["telemetry"]), 20)
        self.assertEqual(result["alerts"], [])

    def test_quoted_csv_and_unicode_json_spans(self):
        self.work_order(maintenance_log='Invented: μ sensor, "checked".')
        self.payload["documents"]["telemetry"] = self.payload["documents"]["telemetry"].replace(
            "SYN-MILL-42", '"SYN-MILL-42"').replace("\n", "\r\n")
        result = app.extract(self.payload)
        field = result["entities"]["telemetry"][0]["asset_id"]
        span = field["source_span"]
        self.assertEqual(self.payload["documents"]["telemetry"][span["start"]:span["end"]],
                         '"SYN-MILL-42"')
        field = result["entities"]["work_order"][0]["maintenance_log"]
        span = field["source_span"]
        self.assertEqual(json.loads(self.payload["documents"]["work_order"][span["start"]:span["end"]]),
                         field["value"])

    def test_duplicate_and_malformed_fields(self):
        for doc, value in [
            ("work_order", '{"unit":"mm","unit":"inch"}'),
            ("work_order", "[]"),
            ("telemetry", "timestamp,timestamp\nx,y\n"),
            ("telemetry", 'timestamp\n"unterminated\n'),
            ("telemetry", 'timestamp\n2026-09-24T08:00:00Z,extra\n'),
            ("inspection", "decision: pass\ndecision: fail\n"),
        ]:
            payload = copy.deepcopy(self.payload)
            payload["documents"][doc] = value
            with self.subTest(document=doc, value=value), self.assertRaises(ValueError):
                app.extract(payload)

    def test_invalid_time_and_order(self):
        for replacement in ("2026-09-24T08:00:00", "not-a-date", "2026-09-24T07:00:00Z"):
            payload = copy.deepcopy(self.payload)
            payload["documents"]["telemetry"] = payload["documents"]["telemetry"].replace(
                "2026-09-24T08:01:00Z", replacement)
            with self.subTest(value=replacement), self.assertRaises(ValueError):
                app.extract(payload)

    def test_invalid_envelope(self):
        for payload in ([], {}, {"schema_version": True, "synthetic": True,
                                 "documents": self.payload["documents"]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                app.extract(payload)

    def test_cli_success_and_file_error(self):
        for args, expected in [(["example_input.json"], 0), (["absent.json"], 2), ([], 2)]:
            completed = subprocess.run(
                [sys.executable, "-B", "implementation.py", *args],
                cwd=ROOT, capture_output=True, text=True, check=False)
            with self.subTest(args=args):
                self.assertEqual(completed.returncode, expected)
                result = json.loads(completed.stdout)
                self.assertEqual(result["status"], "ok" if expected == 0 else "error")
                self.assertEqual(completed.stderr, "")

    def test_cli_invalid_json_and_semantic_error(self):
        for content in ('{"bad":', '{"schema_version":1,"schema_version":1}',
                        json.dumps({**self.payload, "synthetic": False})):
            with patch("builtins.open", return_value=io.StringIO(content)), \
                    patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
