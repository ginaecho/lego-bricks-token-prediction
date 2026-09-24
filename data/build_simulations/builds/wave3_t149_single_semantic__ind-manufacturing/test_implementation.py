import contextlib
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


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def sensor(self, field, value, position=0):
        rows = list(csv.DictReader(io.StringIO(self.payload["sensor_csv"])))
        rows[position][field] = value
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=app.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        self.payload["sensor_csv"] = output.getvalue()

    def rejected(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.payload)

    def cli(self, *args):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                   cwd=ROOT, capture_output=True, text=True, timeout=15)
        return completed, json.loads(completed.stdout)

    def test_example_all_entities_and_determinism(self):
        before = copy.deepcopy(self.payload)
        first = app.run(self.payload)
        self.assertEqual(first, app.run(self.payload))
        self.assertEqual(self.payload, before)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["index"]["document_count"], 7)
        self.assertEqual({item["entity_type"] for item in first["results"]},
                         {"work_order", "sensor_reading", "quality_inspection"})
        self.assertEqual(first["results"], sorted(first["results"], key=lambda d: (-d["score"], d["id"])))

    def test_synonyms_find_thermal_readings(self):
        self.payload["search"] = {"query": "hot", "top_k": 100}
        result = app.run(self.payload)
        self.assertIn("SYN-S-01", [item["id"] for item in result["results"]])
        self.assertNotIn("SYN-S-04", [item["id"] for item in result["results"]])

    def test_quality_traceability_preserved(self):
        self.payload["search"]["query"] = "quality inspection"
        result = app.run(self.payload)["results"][0]
        self.assertEqual(result["id"], "SYN-Q-01")
        trace = result["traceability"]
        for field in ("decided_by", "timestamp", "decision", "rationale", "measurements", "tolerances"):
            self.assertTrue(trace[field])
        self.assertEqual(result["work_order_id"], "SYN-WO-100")

    def test_critical_alert_not_hidden_by_search_or_top_k(self):
        self.payload["search"] = {"query": "unfindablexyz", "top_k": 1}
        result = app.run(self.payload)
        self.assertEqual(result["results"], [])
        critical = next(alert for alert in result["alerts"] if alert["severity"] == "critical")
        self.assertTrue(critical["requires_human"])
        self.assertEqual(critical["route"], "human_safety_operator")
        self.assertEqual(critical["state"], "pending_human_review")
        self.assertFalse(critical["automatic_action_taken"])

    def test_critical_even_when_within_tolerance(self):
        self.sensor("safety_critical", "true")
        alerts = app.run(self.payload)["alerts"]
        self.assertEqual(next(a for a in alerts if a["record_id"] == "SYN-S-01")["severity"], "critical")

    def test_seeded_physically_plausible_randomized_telemetry(self):
        # Clearly synthetic noise around a nominal 62 C spindle temperature.
        rng = random.Random(149)
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(app.CSV_FIELDS)
        for i in range(30):
            value = round(rng.uniform(60, 65), 3)
            writer.writerow([f"SYN-R-{i:02}", f"2026-09-24T08:{i:02}:00Z", "SYN-MILL-07",
                             "SYN-WO-100", "temperature", value, "C", "false"])
        self.payload["sensor_csv"] = output.getvalue()
        result = app.run(self.payload)
        self.assertEqual(result["index"]["document_count"], 33)
        self.assertEqual(result["alerts"], [])
        self.assertEqual(result, app.run(copy.deepcopy(self.payload)))

    def test_empty_corpus(self):
        self.payload["work_orders"] = []
        self.payload["quality_inspections"] = []
        self.payload["sensor_csv"] = ",".join(app.CSV_FIELDS) + "\n"
        result = app.run(self.payload)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["alerts"], [])
        self.assertEqual(result["index"]["document_count"], 0)

    def test_top_k(self):
        self.payload["search"] = {"query": "temperature", "top_k": 1}
        self.assertEqual(len(app.run(self.payload)["results"]), 1)

    def test_invalid_queries_and_limits(self):
        for value in ("", "   ", "!!!", None, 123):
            with self.subTest(query=value):
                self.payload["search"]["query"] = value
                self.rejected()
        self.payload["search"]["query"] = "pump"
        for value in (0, -1, 101, True, 1.5, "3"):
            with self.subTest(top_k=value):
                self.payload["search"]["top_k"] = value
                self.rejected()

    def test_missing_traceability_and_contradictory_pass(self):
        for field in ("decided_by", "timestamp", "rationale"):
            with self.subTest(field=field):
                original = self.payload["quality_inspections"][0][field]
                self.payload["quality_inspections"][0][field] = ""
                self.rejected()
                self.payload["quality_inspections"][0][field] = original
        self.payload["quality_inspections"][0]["decision"] = "pass"
        self.rejected()

    def test_invalid_unit_tolerance_and_physical_value(self):
        original = copy.deepcopy(self.payload)
        for field, value in (("unit", "F"), ("value", "nan"), ("value", "-100"),
                             ("safety_critical", "yes"), ("value", "inf")):
            with self.subTest(field=field, value=value):
                self.payload = copy.deepcopy(original)
                self.sensor(field, value)
                self.rejected()
        self.payload = original
        self.payload["work_orders"][0]["tolerances"]["temperature"]["min"] = 90
        self.rejected()

    def test_reference_identity_and_timestamp_validation(self):
        original = copy.deepcopy(self.payload)
        for field, value in (("asset_id", "SYN-WRONG"), ("work_order_id", "SYN-UNKNOWN"),
                             ("id", "SYN-WO-100"), ("timestamp", "2026-09-24T08:00:00")):
            with self.subTest(field=field):
                self.payload = copy.deepcopy(original)
                self.sensor(field, value)
                self.rejected()

    def test_malformed_csv(self):
        for content in ("", "wrong,header\n", ",".join(app.CSV_FIELDS) + "\nshort,row\n",
                        ",".join(app.CSV_FIELDS) + '\n"unterminated'):
            with self.subTest(content=content):
                self.payload["sensor_csv"] = content
                self.rejected()

    def test_schema_and_synthetic_label(self):
        self.payload["synthetic"] = False
        self.rejected()
        self.payload["synthetic"] = True
        self.payload["unexpected"] = "field"
        self.rejected()

    def test_injected_embedding_can_retrieve_no_keyword_match(self):
        self.payload["search"] = {"query": "unfindablexyz", "top_k": 1}
        calls = []

        def local_embed(texts):
            calls.append(texts)
            return [[1.0, 0.0]] + [[1.0, 0.0] if "SYN-PRESS-12" in t else [0.0, 1.0]
                                  for t in texts[1:]]

        result = app.run(self.payload, embed=local_embed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 8)
        self.assertEqual(result["results"][0]["asset_id"], "SYN-PRESS-12")
        self.assertAlmostEqual(result["results"][0]["score"], 0.3)
        self.assertEqual(result, app.run(self.payload, embed=local_embed))

    def test_invalid_embedding_vectors(self):
        bad_values = [
            [], [[1.0]] * 7, [[0.0, 0.0]] * 8, [[float("nan")]] * 8,
            [[True]] * 8, [[1.0]] + [[1.0, 2.0]] * 7, "not vectors",
        ]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(app.ValidationError):
                    app.run(self.payload, embed=lambda texts: value)

    def test_embedding_callable_failure_and_large_vectors(self):
        def broken(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "embedding callable failed"):
            app.run(self.payload, embed=broken)
        result = app.run(self.payload, embed=lambda texts: [[1e308, 1e308]] * len(texts))
        self.assertTrue(all(0 <= item["score"] <= 1 for item in result["results"]))

    def test_boundary_tolerance_is_inclusive(self):
        self.sensor("value", "75")
        result = app.run(self.payload)
        self.assertNotIn("SYN-S-01", [alert["record_id"] for alert in result["alerts"]])

    def test_cli_success(self):
        process, result = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(process.stderr, "")

    def test_cli_file_json_and_usage_errors(self):
        for arguments in ((), ("nonexistent-input.json",), ("implementation.py",),
                          ("example_input.json", "extra")):
            with self.subTest(arguments=arguments):
                process, result = self.cli(*arguments)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(result["status"], "error")
                self.assertEqual(len(process.stdout.splitlines()), 1)
                self.assertEqual(process.stderr, "")

    def test_cli_validation_and_strict_json_without_scratch_files(self):
        invalid = copy.deepcopy(self.payload)
        invalid["search"]["top_k"] = 0
        for content in (json.dumps(invalid), '{"x":NaN}', '{"x":1,"x":2}', "[]"):
            with self.subTest(content=content), patch.object(Path, "open", return_value=io.StringIO(content)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
