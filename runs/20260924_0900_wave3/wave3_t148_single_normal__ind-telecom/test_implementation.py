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


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.research(self.data)

    def mutate_csv(self, field, value):
        record = self.data["records"][2]
        row = next(csv.DictReader(io.StringIO(record["cdr_csv"])))
        row[field] = value
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=app.CDR_FIELDS)
        writer.writeheader()
        writer.writerow(row)
        record["cdr_csv"] = stream.getvalue()

    def test_normal_deterministic_and_private(self):
        before = copy.deepcopy(self.data)
        result = app.research(self.data)
        self.assertEqual(result, app.research(self.data))
        self.assertEqual(self.data, before)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["findings"]), 3)
        encoded = json.dumps(result)
        self.assertNotIn(self.data["records"][0]["phone"], encoded)
        self.assertNotIn(self.data["records"][0]["imei"], encoded)

    def test_exact_source_citations(self):
        for finding in app.research(self.data)["findings"]:
            citation = finding["citation"]
            record = next(r for r in self.data["records"] if r["id"] == citation["record_id"])
            if citation["format"] == "support_chat":
                self.assertEqual(record["transcript"][citation["start"]:citation["end"]],
                                 finding["quote"])
                self.assertEqual(record["transcript"].splitlines()[citation["line"] - 1],
                                 finding["quote"])
            else:
                row = next(csv.DictReader(io.StringIO(record["cdr_csv"])))
                self.assertEqual(row[citation["column"]], finding["quote"])
                self.assertEqual(citation["row"], 2)

    def test_no_matches(self):
        self.data["query"] = "satellite"
        result = app.research(self.data)
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["outcome"], "no_matches")

    def test_limit_and_ties(self):
        self.data["limit"] = 1
        result = app.research(self.data)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["citation"]["record_id"], "ticket-demo")
        self.assertEqual(result["matched_passages"], 3)

    def test_schema_invalid(self):
        for key, value in (("limit", True), ("query", " "), ("synthetic", False),
                           ("records", []), ("schema_version", True)):
            with self.subTest(key=key):
                original = self.data[key]
                self.data[key] = value
                self.invalid()
                self.data[key] = original

    def test_all_record_residencies(self):
        for record in self.data["records"]:
            record["residency"] = "US"
            self.invalid()
            record["residency"] = "EU"
        self.mutate_csv("residency", "US")
        self.invalid()

    def test_billing_requires_reason(self):
        adjustment = self.data["records"][1]["billing_adjustment"]
        for reason in ("", " ", None):
            adjustment["reason"] = reason
            self.invalid()
        del adjustment["reason"]
        self.invalid()

    def test_adjustment_residency_and_amount(self):
        adjustment = self.data["records"][1]["billing_adjustment"]
        adjustment["residency"] = "US"
        self.invalid()
        adjustment["residency"] = "EU"
        adjustment["amount"] = float("nan")
        self.invalid()

    def test_subscriber_protection_and_policy(self):
        self.data["records"][0]["processing_allowed"] = False
        self.invalid()
        self.data["records"][0]["processing_allowed"] = True
        for key in ("purpose", "role", "lawful_basis"):
            original = self.data["policy"][key]
            self.data["policy"][key] = "unauthorized"
            self.invalid()
            self.data["policy"][key] = original

    def test_sensitive_text_rejected(self):
        for value in ("network +1-202-555-0142", "network 000000123456789",
                      "network user@example.invalid"):
            self.data["query"] = value
            self.invalid()
        self.data["query"] = "network"
        self.data["records"][1]["transcript"] += "\nuser@example.invalid"
        self.invalid()

    def test_bad_csv_usage_and_identity(self):
        original = self.data["records"][2]["cdr_csv"]
        for key, value in (("volume_mb", "-1"), ("volume_mb", "nan"),
                           ("duration_seconds", "oops"), ("id", "usage-other"),
                           ("phone", "+1-202-555-0199")):
            self.data["records"][2]["cdr_csv"] = original
            self.mutate_csv(key, value)
            self.invalid()
        self.data["records"][2]["cdr_csv"] = "wrong,header\n1,2\n"
        self.invalid()

    def test_randomized_synthetic_usage(self):
        rng = random.Random(148)
        for _ in range(10):
            self.mutate_csv("volume_mb", f"{rng.uniform(0, 1000):.3f}")
            self.mutate_csv("duration_seconds", str(rng.randrange(3600)))
            self.assertEqual(app.research(self.data)["status"], "ok")

    def test_link_and_duplicate_validation(self):
        self.data["records"][1]["account_id"] = "acct-missing"
        self.invalid()
        self.data["records"][1]["account_id"] = "acct-demo"
        self.data["records"].append(copy.deepcopy(self.data["records"][0]))
        self.invalid()

    def test_crlf_unicode_offsets(self):
        self.data["records"][1]["transcript"] = "Agent: café\r\nAgent: network outage\r\n"
        finding = next(f for f in app.research(self.data)["findings"]
                       if f["citation"]["record_id"] == "ticket-demo")
        citation = finding["citation"]
        self.assertEqual(citation["line"], 2)
        self.assertEqual(self.data["records"][1]["transcript"][citation["start"]:citation["end"]],
                         finding["quote"])

    def test_quoted_multiline_csv_citation(self):
        note = 'Synthetic network outage, with "data"\nand recovery'
        self.mutate_csv("note", note)
        finding = next(f for f in app.research(self.data)["findings"]
                       if f["citation"]["record_id"] == "usage-demo")
        self.assertEqual(finding["quote"], note)
        self.assertEqual(finding["citation"]["row"], 2)

    def test_cli_existing_invalid_files(self):
        for filename in ("implementation.py", "build_manifest.json"):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                     str(ROOT / filename)],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_without_scratch_files(self):
        for content in ("{", '{"synthetic":true,"synthetic":true}', '{"value":NaN}',
                        'null', '{"schema_version": 1}'):
            with patch.object(Path, "open", return_value=io.StringIO(content)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["example_input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
