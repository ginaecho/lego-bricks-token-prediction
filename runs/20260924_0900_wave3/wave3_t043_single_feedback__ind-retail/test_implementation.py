import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_deduplication_and_traceability(self):
        result = app.analyze(self.data)
        self.assertEqual(result["counts"], {"received": 3, "unique": 2, "duplicates_removed": 1})
        self.assertEqual(result["evidence"][0]["source_feedback_ids"], ["F-01", "F-02"])
        originals = {f["feedback_id"]: f["text"] for f in self.data["feedback"]}
        for evidence in result["evidence"]:
            self.assertEqual(evidence["excerpt"], originals[evidence["canonical_feedback_id"]])
        ids = {e["canonical_feedback_id"] for e in result["evidence"]}
        self.assertTrue(all(set(t["evidence_ids"]) <= ids for t in result["themes"]))
        self.assertEqual(result["catalog"], self.data["catalog"])

    def test_themes_multilabel(self):
        themes = {t["theme"]: t["unique_feedback_count"] for t in app.analyze(self.data)["themes"]}
        self.assertEqual(themes, {"delivery": 1, "packaging": 1, "price_value": 1, "product_quality": 2})

    def test_empty_feedback(self):
        self.data["feedback"] = []
        result = app.analyze(self.data)
        self.assertEqual(result["themes"], [])
        self.assertEqual(result["counts"]["unique"], 0)

    def test_unmatched_and_negated_text_remains_verbatim(self):
        self.data["feedback"] = [self.data["feedback"][0]]
        self.data["feedback"][0]["text"] = "Nothing to add."
        self.assertEqual(app.analyze(self.data)["themes"][0]["theme"], "other")
        self.data["feedback"][0]["text"] = "Not expensive"
        result = app.analyze(self.data)
        self.assertEqual(result["evidence"][0]["excerpt"], "Not expensive")
        self.assertNotIn("sentiment", result)

    def test_same_text_other_customer_not_duplicate(self):
        self.data["feedback"][2]["text"] = self.data["feedback"][0]["text"]
        self.assertEqual(app.analyze(self.data)["counts"]["unique"], 2)

    def test_consent_required_for_both_regimes(self):
        self.data["personalize"] = True
        for consent_key in ("gdpr_personalization", "ccpa_personalization"):
            with self.subTest(key=consent_key):
                for customer in self.data["customers"]:
                    customer["consent"] = {"gdpr_personalization": True, "ccpa_personalization": True}
                self.data["customers"][0]["consent"][consent_key] = False
                with self.assertRaises(app.ValidationError):
                    app.analyze(self.data)

    def test_consented_personalization(self):
        self.data["personalize"] = True
        for customer in self.data["customers"]:
            customer["consent"] = {"gdpr_personalization": True, "ccpa_personalization": True}
        self.assertEqual(len(app.analyze(self.data)["personalized_insights"]), 2)

    def test_no_consent_no_personalization(self):
        self.assertEqual(app.analyze(self.data)["personalized_insights"], [])

    def test_catalog_snapshots(self):
        for collection in ("orders", "clickstream"):
            for field, value in (("shown_price", "9.99"), ("shown_stock", 99)):
                with self.subTest(collection=collection, field=field):
                    data = copy.deepcopy(self.data)
                    row = data[collection][0]
                    if collection == "orders":
                        row = row["items"][0]
                    row[field] = value
                    with self.assertRaises(app.ValidationError):
                        app.analyze(data)

    def test_provenance_and_label(self):
        self.data["feedback"][0]["provenance"] = "generated_endorsement"
        with self.assertRaises(app.ValidationError):
            app.analyze(self.data)
        self.data["feedback"] = []
        self.data["fixture_label"] = "Real shoppers"
        with self.assertRaises(app.ValidationError):
            app.analyze(self.data)

    def test_references_and_duplicates(self):
        for field, value in (("sku", "UNKNOWN"), ("order_id", "UNKNOWN"), ("customer_id", "SYN-B"), ("feedback_id", "F-03")):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["feedback"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.analyze(data)

    def test_strict_types(self):
        for field, value in (("stock", True), ("stock", -1), ("price", 12.50), ("price", "NaN")):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["catalog"]["products"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.analyze(data)

    def test_determinism_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.analyze(self.data)
        self.assertEqual(original, self.data)
        self.data["feedback"].reverse()
        self.assertEqual(first, app.analyze(self.data))

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")

    def test_cli_invalid_json_and_validation(self):
        for payload in ('{', '{"x":1,"x":2}', '{"x":NaN}', '[]', '{"schema_version":1}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("builtins.open", return_value=io.StringIO(payload)), patch("sys.stdout", output):
                    self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
