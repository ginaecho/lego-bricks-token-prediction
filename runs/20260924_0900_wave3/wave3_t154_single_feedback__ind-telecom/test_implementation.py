import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_normal_dedup_and_themes(self):
        result = app.analyze(self.doc)
        self.assertEqual(result["summary"], {
            "feedback_records": 3, "unique_feedback": 2, "duplicates_removed": 1})
        self.assertEqual([t["theme"] for t in result["themes"]],
                         ["billing", "connectivity", "support", "usage"])

    def test_traceable_evidence(self):
        theme = next(t for t in app.analyze(self.doc)["themes"] if t["theme"] == "connectivity")
        item = theme["evidence"][0]
        self.assertEqual(item["source_feedback_ids"], ["fb_001", "fb_002"])
        self.assertEqual(item["ticket_id"], "tkt_001")
        self.assertEqual(item["usage_record_ids"], ["cdr_001"])

    def test_private_subscriber_values_not_published(self):
        result = json.dumps(app.analyze(self.doc))
        for key in ("phone", "imei", "name", "email"):
            self.assertNotIn(self.doc["subscribers"][0][key], result)
        self.assertIn("[REDACTED]", result)

    def test_residency_on_all_entity_types(self):
        for collection in ("subscribers", "tickets", "feedback"):
            with self.subTest(collection=collection):
                doc = copy.deepcopy(self.doc)
                doc[collection][0]["residency"] = "US"
                with self.assertRaises(app.ValidationError):
                    app.analyze(doc)
        self.doc["usage_csv"] = self.doc["usage_csv"].replace("cdr_001,EU", "cdr_001,US")
        with self.assertRaises(app.ValidationError):
            app.analyze(self.doc)

    def test_billing_adjustment_reason_required(self):
        for reason in ("", "  ", None):
            self.doc["tickets"][1]["billing_adjustment"]["reason"] = reason
            with self.assertRaises(app.ValidationError):
                app.analyze(self.doc)

    def test_privacy_opt_out_and_basis(self):
        self.doc["subscribers"][0]["analytics_allowed"] = False
        with self.assertRaises(app.ValidationError):
            app.analyze(self.doc)
        self.doc["subscribers"][0]["analytics_allowed"] = True
        self.doc["subscribers"][0]["processing_basis"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.analyze(self.doc)

    def test_empty_feedback_and_unknown_theme(self):
        self.doc["feedback"] = []
        self.assertEqual(app.analyze(self.doc)["themes"], [])
        self.setUp()
        self.doc["feedback"] = [self.doc["feedback"][0]]
        self.doc["feedback"][0]["text"] = "Thank you very much"
        self.assertEqual(app.analyze(self.doc)["themes"][0]["theme"], "other")

    def test_duplicate_ids_rejected(self):
        self.doc["feedback"].append(copy.deepcopy(self.doc["feedback"][0]))
        with self.assertRaises(app.ValidationError):
            app.analyze(self.doc)

    def test_invalid_relationship_and_csv(self):
        self.doc["feedback"][0]["ticket_id"] = "tkt_999"
        with self.assertRaises(app.ValidationError):
            app.analyze(self.doc)
        self.setUp()
        self.doc["usage_csv"] = self.doc["usage_csv"].replace("482.37", "NaN")
        with self.assertRaises(app.ValidationError):
            app.analyze(self.doc)

    def test_deterministic_randomized_usage_fixture(self):
        rng = random.Random(154)
        self.doc["usage_csv"] = (
            "id,residency,subscriber_id,ticket_id,call_seconds,data_mb\n"
            f"cdr_001,EU,sub_001,tkt_001,{rng.randint(0,3600)},{rng.uniform(0,1024):.2f}\n")
        self.assertEqual(app.analyze(self.doc), app.analyze(copy.deepcopy(self.doc)))

    def test_no_cross_ticket_deduplication(self):
        self.doc["feedback"][1]["ticket_id"] = "tkt_002"
        self.assertEqual(app.analyze(self.doc)["summary"]["unique_feedback"], 3)

    def test_invalid_envelope(self):
        for value in ([], {}, None, {"synthetic": False}):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.analyze(value)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ((), ("absent.json",), ("example_input.json", "extra")):
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_wrong_schema(self):
        for filename in ("implementation.py", "build_manifest.json"):
            result = self.cli(filename)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")


if __name__ == "__main__":
    unittest.main()
