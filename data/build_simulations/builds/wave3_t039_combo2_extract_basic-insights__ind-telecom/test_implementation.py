import contextlib
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_pipeline(self):
        result = app.run(self.payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["insights"]["record_count"], 4)
        self.assertEqual(result["insights"]["subscriber_count"], 2)
        self.assertEqual(result["insights"]["excluded_count"], 1)

    def test_transcript_source_span(self):
        record = app.extract(self.payload)["records"][1]
        field = record["fields"]["adjustment_reason"]
        source = self.payload["records"][1]["document"]["text"]
        self.assertEqual(source[slice(*field["span"])], field["value"])

    def test_csv_quoted_source_span(self):
        record = app.extract(self.payload)["records"][2]
        field = record["fields"]["feedback"]
        source = self.payload["records"][2]["document"]["text"]
        self.assertEqual(source[slice(*field["span"])], '"Slow data, allowance is expensive"')

    def test_multiline_csv(self):
        self.payload["records"][2]["document"]["text"] = (
            'duration_seconds,data_mb,feedback\n1,2,"Slow\n""signal"""\n')
        field = app.extract(self.payload)["records"][2]["fields"]["feedback"]
        self.assertEqual(field["value"], 'Slow\n"signal"')

    def test_privacy_redaction(self):
        encoded = json.dumps(app.run(self.payload))
        for secret in ("+1-202-555-0142", "000000000000018", "demo@example.invalid",
                       "SYNTHETIC-SUB-001", "Private feedback"):
            self.assertNotIn(secret, encoded)
        self.assertIn("[REDACTED]", encoded)

    def test_opt_out_not_parsed(self):
        self.payload["records"][4]["document"]["text"] = "Billing adjustment: INVALID"
        self.assertEqual(len(app.extract(self.payload)["excluded"]), 1)

    def test_missing_field_report(self):
        self.payload["records"][1]["document"]["text"] = "Ticket: SYN-1"
        result = app.run(self.payload)
        self.assertEqual(result["extraction"]["records"][1]["missing_fields"], ["feedback"])
        self.assertEqual(result["insights"]["incomplete_record_count"], 1)

    def test_billing_requires_reason(self):
        for adjustment in ("0", "-5", "5"):
            with self.subTest(adjustment=adjustment):
                self.payload["records"][1]["document"]["text"] = (
                    f"Ticket: SYN-1\nFeedback: bill\nBilling adjustment: {adjustment}")
                with self.assertRaises(app.ValidationError):
                    app.run(self.payload)

    def test_residency_enforced_even_opt_out(self):
        for index in (0, 4):
            payload = copy.deepcopy(self.payload)
            payload["records"][index]["residency"] = "US"
            with self.assertRaises(app.ValidationError):
                app.run(payload)

    def test_lawful_basis_and_purpose(self):
        self.payload["records"][0]["privacy"]["lawful_basis"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.run(self.payload)
        self.setUp()
        self.payload["purpose"] = "advertising"
        with self.assertRaises(app.ValidationError):
            app.run(self.payload)

    def test_invalid_usage_values(self):
        for value in ("-1", "NaN", "Infinity", "not-a-number"):
            with self.subTest(value=value):
                self.payload["records"][2]["document"]["text"] = (
                    f"duration_seconds,data_mb\n{value},3\n")
                with self.assertRaises(app.ValidationError):
                    app.run(self.payload)

    def test_invalid_csv(self):
        for source in ("data_mb,data_mb\n1,2", "unknown\n1",
                       "data_mb\n1\n2", 'feedback\n"unterminated', "data_mb\n1,2"):
            with self.subTest(source=source):
                self.payload["records"][2]["document"]["text"] = source
                with self.assertRaises(app.ValidationError):
                    app.run(self.payload)

    def test_duplicate_transcript_fields(self):
        self.payload["records"][0]["document"]["text"] = "Plan: one\nPlan: two"
        with self.assertRaises(app.ValidationError):
            app.run(self.payload)

    def test_handoff_validation(self):
        for mutation in ("span", "missing", "residency", "private"):
            with self.subTest(mutation=mutation):
                extracted = app.extract(self.payload)
                record = extracted["records"][0]
                if mutation == "span":
                    record["fields"]["plan"]["span"] = [-1, 10]
                elif mutation == "missing":
                    record["missing_fields"] = ["plan"]
                elif mutation == "residency":
                    record["residency"] = "US"
                else:
                    record["fields"]["phone"]["value"] = "private"
                with self.assertRaises(app.ValidationError):
                    app.insights(extracted)

    def test_cross_stage_evidence(self):
        result = app.run(self.payload)
        records = {r["record_id"]: r for r in result["extraction"]["records"]}
        for theme in result["insights"]["themes"]:
            self.assertTrue(theme["action"])
            for evidence in theme["evidence"]:
                record = records[evidence["record_id"]]
                self.assertEqual(evidence["residency"], record["residency"])
                field = record["fields"][evidence["field"]]
                self.assertEqual(evidence["span"], field["span"])
                self.assertEqual(evidence["excerpt"], field["value"])
        themes = {t["theme"]: t for t in result["insights"]["themes"]}
        self.assertEqual(themes["network_reliability"]["record_count"], 2)
        self.assertEqual(themes["high_usage"]["record_count"], 1)

    def test_insight_tampering_rejected(self):
        result = app.run(self.payload)
        result["insights"]["record_count"] = 100
        with self.assertRaises(app.ValidationError):
            app.validate(result, "insights")

    def test_empty_input(self):
        self.payload["records"] = []
        self.assertEqual(app.run(self.payload)["insights"]["themes"], [])

    def test_seeded_random_usage(self):
        rng = random.Random(39)
        for _ in range(20):
            volume = round(rng.uniform(0, 20000), 2)
            duration = round(rng.uniform(0, 3600), 2)
            self.payload["records"] = [self.payload["records"][2]] if len(
                self.payload["records"]) > 1 else self.payload["records"]
            self.payload["records"][0]["document"]["text"] = (
                f"duration_seconds,data_mb\n{duration},{volume}\n")
            result = app.run(self.payload)
            self.assertEqual(bool(result["insights"]["themes"]), volume >= 10000)

    def test_deterministic_no_input_mutation(self):
        before = copy.deepcopy(self.payload)
        self.assertEqual(app.run(self.payload), app.run(self.payload))
        self.assertEqual(self.payload, before)

    def test_invalid_input_schema(self):
        for payload in ([], {}, {"schema_version": "2"}):
            with self.subTest(payload=payload), self.assertRaises(app.ValidationError):
                app.run(payload)
        self.payload["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.payload)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True,
                              text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for source in ('{"x":', '{"schema_version":"1","schema_version":"2"}', '[]',
                       '{"records":NaN}'):
            with patch("builtins.open", mock_open(read_data=source)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
