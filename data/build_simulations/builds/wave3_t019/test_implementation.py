import contextlib
import copy
import io
import json
import random
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_normal_trace_and_gap(self):
        result = app.review(self.data)
        self.assertEqual(result["summary"], {"requirements": 3, "supported": 2, "gaps": 1})
        self.assertEqual(result["findings"][0]["evidence_document_ids"], ["chat_demo"])
        self.assertEqual(result["findings"][1]["inspected_document_ids"], ["chat_demo"])

    def test_deterministic_and_nonmutating(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.review(self.data), app.review(self.data))
        self.assertEqual(self.data, before)

    def test_empty_review_not_assessed(self):
        self.data["requirements"] = []
        self.assertEqual(app.review(self.data)["review_state"], "not_assessed")

    def test_missing_evidence_is_gap(self):
        self.data["requirements"][0]["document_ids"] = []
        self.assertEqual(app.review(self.data)["findings"][0]["result"], "gap")

    def test_unlinked_document_cannot_support(self):
        self.data["documents"][0]["entity_ids"] = ["acct_demo"]
        self.assertEqual(app.review(self.data)["findings"][0]["result"], "gap")

    def test_case_insensitive(self):
        self.data["requirements"][0]["required_terms"] = ["OUTAGE"]
        self.assertEqual(app.review(self.data)["findings"][0]["result"], "supported")

    def test_residency_every_collection(self):
        for name in ("accounts", "tickets", "usage_records", "billing_adjustments",
                     "documents", "requirements"):
            with self.subTest(name=name):
                self.data[name][0]["residency"] = "US"
                self.invalid()
                self.data[name][0]["residency"] = "EU"

    def test_cdr_row_residency(self):
        self.data["documents"][1]["content"] = self.data["documents"][1]["content"].replace(",EU,", ",US,")
        self.invalid()

    def test_cdr_tampering(self):
        self.data["documents"][1]["content"] = self.data["documents"][1]["content"].replace("2773", "2774")
        self.invalid()

    def test_billing_reason(self):
        self.data["billing_adjustments"][0]["reason"] = " "
        self.invalid()

    def test_privacy_protection(self):
        self.data["accounts"][0]["subscriber_data_protected"] = False
        self.invalid()

    def test_ccpa_demonstration(self):
        self.data["processing_residency"] = "US"
        for name in ("accounts", "tickets", "usage_records", "billing_adjustments", "documents", "requirements"):
            for row in self.data[name]:
                row["residency"] = "US"
        self.data["accounts"][0]["jurisdiction"] = "CCPA"
        self.data["documents"][1]["content"] = self.data["documents"][1]["content"].replace(",EU,", ",US,")
        self.assertEqual(app.review(self.data)["status"], "ok")

    def test_no_sensitive_content_output(self):
        rendered = json.dumps(app.review(self.data))
        for forbidden in ("+1-202-555-0100", "000000000000000", "My data connection stopped"):
            self.assertNotIn(forbidden, rendered)

    def test_invalid_types(self):
        for value in (None, [], "bad", True):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.review(value)
        self.data["tickets"][0]["account_id"] = []
        self.invalid()

    def test_nonfinite_and_boolean_usage(self):
        for value in (float("nan"), float("inf"), -1, True, 10 ** 1000):
            self.data["usage_records"][0]["data_mb"] = value
            self.invalid()

    def test_seeded_usage_fixture(self):
        rng = random.Random(19)
        self.assertEqual(self.data["usage_records"][0]["call_seconds"], rng.randint(0, 3600))
        self.assertEqual(self.data["usage_records"][0]["data_mb"], round(rng.uniform(0, 500), 2))

    def test_terms_do_not_combine_across_documents(self):
        doc = copy.deepcopy(self.data["documents"][0])
        doc["id"] = "second_chat"
        doc["content"] = "investigation"
        self.data["documents"].append(doc)
        self.data["documents"][0]["content"] = "outage"
        self.data["requirements"][0]["document_ids"].append("second_chat")
        self.assertEqual(app.review(self.data)["findings"][0]["result"], "gap")

    def test_duplicates_and_unknown_references(self):
        self.data["tickets"][0]["id"] = "acct_demo"
        self.invalid()
        self.data["tickets"][0]["id"] = "ticket_demo"
        self.data["requirements"][0]["document_ids"] = ["missing"]
        self.invalid()

    def test_synthetic_only(self):
        self.data["synthetic"] = False
        self.invalid()

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_error_and_usage(self):
        for args in ([], [str(ROOT / "nonexistent.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_json_and_validation_errors(self):
        for source in ("{", '{"schema_version":"1","schema_version":"1"}', "NaN",
                       '{"schema_version":"1","synthetic":false}', "[]"):
            with self.subTest(source=source):
                output = io.StringIO()
                with patch("builtins.open", return_value=io.StringIO(source)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["mocked.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
