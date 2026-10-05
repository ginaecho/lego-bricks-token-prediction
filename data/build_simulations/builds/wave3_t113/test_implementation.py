"""Synthetic fixtures only; tests create no files and call no networks."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_categorization_priority_and_accountability(self):
        result = app.triage(self.payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([r["category"] for r in result["results"]],
                         ["billing", "technical", "general"])
        self.assertEqual([r["priority"] for r in result["results"]],
                         ["normal", "urgent", "low"])
        self.assertEqual(result["results"][1]["route"],
                         {"team": "synthetic-engineering", "owner": "synthetic-oncall"})
        self.assertEqual(result["summary"]["total"], 3)
        self.assertEqual(sum(result["summary"]["by_priority"].values()), 3)

    def test_first_category_wins_but_all_priority_rules_apply(self):
        self.payload["tickets"] = [
            {"id": "SYN-X", "subject": "OUTAGE invoice", "body": "blocked"}]
        result = app.triage(self.payload)["results"][0]
        self.assertEqual(result["category"], "billing")
        self.assertEqual(result["priority"], "urgent")
        self.assertEqual(len(result["reason"]["priority_rules"]), 2)

    def test_empty_text_routes_to_accountable_fallback(self):
        self.payload["tickets"] = [{"id": "SYN-X", "subject": "", "body": ""}]
        result = app.triage(self.payload)["results"][0]
        self.assertEqual(result["reason"]["category_selection"], "fallback")
        self.assertEqual(result["route"]["owner"], "synthetic-support-lead")

    def test_empty_batch(self):
        self.payload["tickets"] = []
        result = app.triage(self.payload)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["summary"]["total"], 0)

    def test_rules_cannot_lower_baseline(self):
        self.payload["config"]["priority_rules"] = [
            {"keywords": ["outage"], "priority": "low"}]
        self.assertEqual(app.triage(self.payload)["results"][1]["priority"], "high")

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(app.triage(self.payload), app.triage(self.payload))
        self.assertEqual(self.payload, original)

    def test_invalid_inputs_share_validation(self):
        invalids = [
            ("schema_version", True),
            ("schema_version", 1.0),
            ("tickets", {}),
            ("config", None),
            ("dataset_label", " "),
        ]
        for field, value in invalids:
            with self.subTest(field=field, value=value):
                payload = copy.deepcopy(self.payload)
                payload[field] = value
                with self.assertRaises(app.ValidationError):
                    app.triage(payload)
        with self.assertRaises(app.ValidationError):
            app.triage([])

    def test_duplicate_ids_rejected(self):
        for collection in ("tickets", "categories"):
            with self.subTest(collection=collection):
                payload = copy.deepcopy(self.payload)
                items = payload["tickets"] if collection == "tickets" else payload["config"]["categories"]
                items.append(copy.deepcopy(items[0]))
                with self.assertRaises(app.ValidationError):
                    app.triage(payload)

    def test_invalid_configuration_rejected(self):
        for field, value in (("owner", ""), ("team", None), ("base_priority", "critical"),
                             ("keywords", [""]), ("keywords", ["outage", "OUTAGE"]),
                             ("keywords", [" refund"])):
            with self.subTest(field=field, value=value):
                payload = copy.deepcopy(self.payload)
                payload["config"]["categories"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.triage(payload)
        self.payload["config"]["fallback_category"] = "missing"
        with self.assertRaises(app.ValidationError):
            app.triage(self.payload)

    def test_unknown_and_missing_fields_rejected(self):
        self.payload["tickets"][0]["extra"] = 1
        with self.assertRaises(app.ValidationError):
            app.triage(self.payload)
        del self.payload["tickets"][0]["extra"]
        del self.payload["tickets"][0]["body"]
        with self.assertRaises(app.ValidationError):
            app.triage(self.payload)

    def test_unicode_casefold_and_documented_substring(self):
        self.payload["config"]["categories"][0]["keywords"] = ["straße"]
        self.payload["tickets"] = [
            {"id": "SYN-U", "subject": "STRASSE", "body": ""}]
        self.assertEqual(app.triage(self.payload)["results"][0]["category"], "billing")
        self.payload["config"]["categories"][0]["keywords"] = ["voice"]
        self.payload["tickets"][0]["subject"] = "invoice"
        self.assertEqual(app.triage(self.payload)["results"][0]["category"], "billing")

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), app.triage(self.payload))

    def test_cli_usage_and_file_error(self):
        for args in ([], ["missing-synthetic-input.json"], ["one", "two"]):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    capture_output=True, text=True, cwd=ROOT, check=False)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_malformed_duplicate_nonfinite_and_schema_errors(self):
        for source in ("{", '{"x": 1, "x": 2}', '{"x": NaN}', "[]",
                       '{"schema_version": true}'):
            with self.subTest(source=source):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=source)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
