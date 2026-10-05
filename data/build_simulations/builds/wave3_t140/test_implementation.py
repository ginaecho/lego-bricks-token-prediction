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


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_normal_ranking_and_grounding(self):
        result = app.recommend(self.data)
        self.assertEqual([r["score"] for r in result["recommendations"]], [10, 9, 4])
        self.assertEqual(result["recommendations"][0]["id"], "syn-plan-data")
        evidence = result["recommendations"][0]["explanations"]
        self.assertEqual(evidence[1]["record_ids"], ["syn-cdr-001", "syn-cdr-002"])
        self.assertTrue(all(e["residency"] == "EU" for e in evidence))

    def test_exclusions_are_absolute(self):
        self.data["exclusions"]["ids"] = ["syn-plan-data"]
        self.data["subscriber_account"]["preferences"]["roaming"] = 100
        ids = [r["id"] for r in app.recommend(self.data)["recommendations"]]
        self.assertNotIn("syn-plan-data", ids)
        self.assertNotIn("syn-plan-roaming", ids)

    def test_no_signals_returns_empty(self):
        self.data["subscriber_account"]["preferences"] = {}
        self.data["call_detail_records_csv"] = self.data["call_detail_records_csv"].splitlines()[0] + "\n"
        self.data["network_fault_tickets"] = []
        self.assertEqual(app.recommend(self.data)["recommendations"], [])

    def test_ties_and_top_k(self):
        other = copy.deepcopy(self.data["catalog"][0])
        other["id"] = "aaa-plan"
        self.data["catalog"].append(other)
        self.data["top_k"] = 1
        self.assertEqual(app.recommend(self.data)["recommendations"][0]["id"], "aaa-plan")

    def test_privacy_flags_and_purpose(self):
        for key, value in [("personalization_consent", False), ("sale_share_opt_out", True),
                           ("deletion_requested", True), ("purpose", "advertising")]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.data)
                self.data["subscriber_account"]["privacy"][key] = value
                self.invalid()
                self.data = original

    def test_residency_on_every_entity(self):
        for entity in ("subscriber_account", "catalog", "network_fault_tickets", "chat", "csv"):
            with self.subTest(entity=entity):
                original = copy.deepcopy(self.data)
                if entity == "csv":
                    self.data["call_detail_records_csv"] = self.data["call_detail_records_csv"].replace(",EU,", ",US,")
                elif entity == "chat":
                    self.data["network_fault_tickets"][0]["support_chat_transcript"][0]["residency"] = "US"
                elif entity == "subscriber_account":
                    self.data[entity]["residency"] = "US"
                else:
                    self.data[entity][0]["residency"] = "US"
                self.invalid()
                self.data = original

    def test_adjustments_need_reason_even_if_excluded(self):
        self.data["exclusions"]["ids"] = ["syn-credit-review"]
        self.data["catalog"][2]["billing_adjustment"]["reason"] = " "
        self.invalid()

    def test_bad_numeric_and_types(self):
        for value in (True, -1, float("nan"), float("inf"), "8", None, 10 ** 1000):
            with self.subTest(value=value):
                self.data["subscriber_account"]["preferences"]["data"] = value
                self.invalid()

    def test_csv_errors(self):
        for csv_text in ("bad,header\n", self.data["call_detail_records_csv"].replace("1327", "nan"),
                         self.data["call_detail_records_csv"] + "broken,row\n"):
            with self.subTest(csv=csv_text):
                self.data["call_detail_records_csv"] = csv_text
                self.invalid()

    def test_link_and_duplicate_validation(self):
        self.data["network_fault_tickets"][0]["subscriber_id"] = "other"
        self.invalid()
        self.data["network_fault_tickets"][0]["subscriber_id"] = "syn-account-001"
        self.data["catalog"][0]["id"] = "syn-account-001"
        self.invalid()

    def test_data_minimization(self):
        rendered = json.dumps(app.recommend(self.data))
        for private in ("+000123456789", "000000123456789", "Moonlight Tower", "47"):
            self.assertNotIn(private, rendered)
        self.assertFalse(app.recommend(self.data)["policy"]["billing_adjustments_executed"])

    def test_repeatability_and_seeded_random_usage(self):
        rng = random.Random(140)
        for _ in range(10):
            data = copy.deepcopy(self.data)
            data["call_detail_records_csv"] = data["call_detail_records_csv"].replace(
                "1327", str(rng.randint(0, 5000)))
            before = copy.deepcopy(data)
            self.assertEqual(app.recommend(data), app.recommend(data))
            self.assertEqual(data, before)

    def test_synthetic_identity_required(self):
        self.data["subscriber_account"]["phone"] = "+491234567890"
        self.invalid()

    def test_unknown_fields_and_invalid_top_k(self):
        self.data["top_k"] = True
        self.invalid()
        self.data["top_k"] = 1
        self.data["unexpected"] = "anything"
        self.invalid()

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], ["nonexistent-input.json"]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_bad_json_and_validation(self):
        for raw in ("{", '{"schema_version": "9"}', "null", "[]"):
            with patch("builtins.open", return_value=io.StringIO(raw)):
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    self.assertEqual(app.main(["memory.json"]), 2)
                    self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
