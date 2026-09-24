import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def pipeline(self):
        return app.run_pipeline(self.payload)

    def reject(self):
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_full_pipeline_and_usage(self):
        result = self.pipeline()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["documents"]["records"]), 2)
        eu = result["insights"]["records"][0]
        self.assertEqual(eu["usage"], {"record_count": 2, "call_minutes": 267,
                                     "data_mb": 9540, "billing_adjustment_total": "-3.25"})

    def test_seeded_randomized_fixture_volumes(self):
        generator = random.Random(13)
        expected = [(generator.randint(20, 300), generator.randint(500, 9000))
                    for _ in range(3)]
        records = app.parse_source(self.payload)["records"]
        actual = [(r["call_minutes"], r["data_mb"]) for r in records
                  if r["type"] == "call_data_usage"]
        self.assertEqual(actual, expected)

    def test_decimal_adjustment_reconciliation(self):
        self.payload["cdr_csv"] = self.payload["cdr_csv"].replace(
            "0.00,", "-0.10,goodwill")
        document = self.pipeline()["documents"]["records"][0]
        self.assertEqual(document["usage"]["billing_adjustment_total"], "-3.35")
        self.assertEqual(len(document["adjustments"]), 2)

    def test_keyword_word_boundaries(self):
        self.payload["records"][4]["text"] = "A billboard is nearby."
        themes = self.pipeline()["insights"]["records"][0]["themes"]
        self.assertNotIn("billing", [t["name"] for t in themes])

    def test_insight_themes_and_actions(self):
        insight = self.pipeline()["insights"]["records"][0]
        self.assertEqual(insight["themes"][0]["name"], "connectivity")
        self.assertEqual(insight["themes"][0]["evidence_count"], 2)
        self.assertIn("Investigate", insight["themes"][0]["action"])
        self.assertEqual(insight["themes"][1]["name"], "coverage")

    def test_document_handoff_preserves_validated_values(self):
        result = self.pipeline()
        for insight, document in zip(result["insights"]["records"], result["documents"]["records"]):
            for key in ("subscriber_ref", "residency", "themes", "ticket_refs", "usage", "adjustments"):
                self.assertEqual(insight[key], document[key])
            self.assertEqual(document["checks"]["billing_reconciliation"], "passed")
        result["documents"]["records"][0]["themes"].clear()
        self.assertTrue(result["insights"]["records"][0]["themes"])

    def test_tampered_handoff_rejected(self):
        insights = self.pipeline()["insights"]
        insights["records"][0]["usage"]["billing_adjustment_total"] = "99.00"
        with self.assertRaises(app.ValidationError):
            app.automate_documents(insights)

    def test_handoff_residency_rejected(self):
        insights = self.pipeline()["insights"]
        insights["records"][0]["adjustments"][0]["residency"] = "US"
        with self.assertRaises(app.ValidationError):
            app.automate_documents(insights)

    def test_minimized_output(self):
        output = json.dumps(self.pipeline())
        for raw in ("SYN-ALPHA", "+12025550101", "000000123456789",
                    "synthetic-demo-key-not-production", "Subscriber:", "FAULT-INVENTED"):
            self.assertNotIn(raw, output)

    def test_repeatable_without_input_mutation(self):
        before = copy.deepcopy(self.payload)
        self.assertEqual(self.pipeline(), self.pipeline())
        self.assertEqual(before, self.payload)

    def test_empty_feedback_and_usage(self):
        self.payload["records"] = self.payload["records"][:1]
        self.payload.pop("cdr_csv")
        record = self.pipeline()["documents"]["records"][0]
        self.assertEqual(record["themes"], [])
        self.assertEqual(record["usage"]["record_count"], 0)

    def test_unknown_feedback_fallback(self):
        self.payload["records"][4]["text"] = "Thank you for assisting me."
        names = [t["name"] for t in self.pipeline()["insights"]["records"][0]["themes"]]
        self.assertIn("other", names)

    def test_residency_mismatch_csv(self):
        self.payload["cdr_csv"] = self.payload["cdr_csv"].replace("SYN-ALPHA,EU", "SYN-ALPHA,US")
        self.reject()

    def test_residency_missing_and_unsupported(self):
        for value in (None, "APAC", ""):
            with self.subTest(value=value):
                self.payload["records"][0]["residency"] = value
                self.reject()

    def test_privacy_constraints(self):
        for key in ("processing_permitted", "do_not_sell"):
            with self.subTest(key=key):
                self.payload["privacy"][key] = False
                self.reject()
                self.payload["privacy"][key] = True
        self.payload["records"][0]["erasure_requested"] = True
        self.reject()

    def test_adjustment_requires_stated_reason(self):
        self.payload["cdr_csv"] = self.payload["cdr_csv"].replace("service_outage", "")
        self.reject()

    def test_invalid_usage_and_money(self):
        original = self.payload["cdr_csv"]
        for old, new in ((",152,", ",-1,"), ("-3.25", "NaN"), ("-3.25", "-3.251")):
            with self.subTest(value=new):
                self.payload["cdr_csv"] = original.replace(old, new)
                self.reject()

    def test_csv_shape_errors(self):
        for csv_text in ("wrong,header\n1,2\n", ",".join(app.CDR_FIELDS) + "\nx,y\n"):
            with self.subTest(csv=csv_text):
                self.payload["cdr_csv"] = csv_text
                self.reject()

    def test_orphan_and_duplicate_records(self):
        self.payload["records"].append(copy.deepcopy(self.payload["records"][0]))
        self.reject()
        self.payload["records"].pop()
        self.payload["records"][4]["subscriber_id"] = "UNKNOWN"
        self.reject()

    def test_cross_account_chat_link_rejected(self):
        self.payload["records"][4]["ticket_id"] = "FAULT-INVENTED-102"
        self.reject()

    def test_shared_schema_rejects_unknown_fields(self):
        self.payload["records"][0]["private_notes"] = "not allowed"
        self.reject()

    def test_wrong_schema_and_non_synthetic_input(self):
        self.payload["schema_version"] = True
        self.reject()
        self.payload["schema_version"] = 1
        self.payload["fixture_label"] = "REAL"
        self.reject()

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_argument(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_and_invalid_input(self):
        for content in ("{bad JSON", "[]", '{"stage":"source"}'):
            with patch("builtins.open", mock_open(read_data=content)):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(app.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_document_validation(self):
        documents = self.pipeline()["documents"]
        documents["records"][0]["checks"]["residency"] = "skipped"
        with self.assertRaises(app.ValidationError):
            app.validate(documents, "documents")


if __name__ == "__main__":
    unittest.main()
