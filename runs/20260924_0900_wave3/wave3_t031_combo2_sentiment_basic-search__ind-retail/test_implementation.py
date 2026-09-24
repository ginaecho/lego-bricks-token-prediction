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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def test_complete_pipeline(self):
        result = self.run_data()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["search"]["results"]), 1)
        self.assertEqual(result["search"]["results"][0]["product"]["sku"], "SYN-SHOE-001")

    def test_transparent_sentiment(self):
        issues = self.run_data()["insights"]["issues"]
        self.assertEqual(issues[0]["score"], -8)
        self.assertEqual(issues[0]["priority"], 28)
        self.assertEqual(sum(e["weight"] for e in issues[0]["evidence"]), -8)
        self.assertEqual(issues[1]["score"], 3)
        self.assertTrue(issues[1]["evidence"][1]["negated"])

    def test_severity_dominates_sentiment(self):
        self.data["issues"][0]["text"] = "excellent"
        self.data["issues"][1]["text"] = "terrible " * 100
        issues = self.run_data()["insights"]["issues"]
        self.assertEqual(issues[0]["severity"], "high")
        self.assertEqual(issues[1]["priority"], 9)

    def test_synonym_search(self):
        self.assertEqual(self.run_data()["search"]["normalized_terms"], ["running", "shoe"])
        self.data["query"] = "rucksack"
        self.assertEqual(self.run_data()["search"]["results"][0]["product"]["sku"], "SYN-PACK-001")

    def test_handoff_changes_availability_and_context(self):
        first = self.run_data()["search"]
        self.assertTrue(first["urgent_in_stock_only"])
        self.assertEqual(first["results"][0]["score_components"]["issue_context"], 3)
        self.data["issues"] = []
        second = self.run_data()["search"]
        self.assertFalse(second["urgent_in_stock_only"])
        self.assertEqual(len(second["results"]), 2)
        self.assertEqual(second["results"][0]["score_components"]["issue_context"], 0)

    def test_reject_tampered_handoff(self):
        envelope = {"schema_version": 1, "status": "ok", "input": self.data}
        output = app.sentiment_stage(envelope)
        output["insights"]["urgent"] = False
        with self.assertRaises(app.ValidationError):
            app.search_stage(output)

    def test_consent_gates_all_personal_signals(self):
        self.data["customer"]["consent"]["personalization"] = False
        first = self.run_data()["search"]
        self.data["issues"] = []
        self.data["clickstream"] = []
        self.data["basket"] = []
        self.data["customer"]["preferred_categories"] = []
        self.assertEqual(first, self.run_data()["search"])
        self.assertFalse(first["urgent_in_stock_only"])
        self.assertEqual(first["source_issue_ids"], [])

    def test_exact_catalog_prices_and_stock(self):
        result = self.run_data()
        by_sku = {p["sku"]: p for p in self.data["catalog"]}
        for item in result["search"]["results"]:
            self.assertEqual(item["product"], by_sku[item["product"]["sku"]])
        result["search"]["results"][0]["product"]["price_cents"] += 1
        with self.assertRaises(app.ValidationError):
            app.validate(result, "search")

    def test_no_fabricated_reviews(self):
        self.data["catalog"][0]["reviews"] = ["Everyone loves it"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_neutral_and_empty_issues(self):
        self.data["issues"][0]["text"] = "shoe arrived"
        issue = self.run_data()["insights"]["issues"][0]
        self.assertEqual(issue["sentiment"], "neutral")
        self.assertEqual(issue["evidence"], [])
        self.data["issues"] = []
        self.assertEqual(self.run_data()["insights"]["overall_score"], 0)

    def test_no_matches(self):
        self.data["query"] = "telescope"
        result = self.run_data()["search"]
        self.assertEqual(result["results"], [])
        self.assertIn("No catalog matches", result["message"])

    def test_unknown_references(self):
        for field in ("issues", "basket", "clickstream"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data[field][0]["sku"] = "UNKNOWN"
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_catalog_numbers(self):
        for field, value in [("price_cents", True), ("price_cents", 1.5),
                             ("price_cents", -1), ("stock", -1), ("stock", False)]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                data["catalog"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_skus_and_ids(self):
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_data()
        self.data["catalog"].pop()
        self.data["issues"][1]["id"] = self.data["issues"][0]["id"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_invalid_consent_and_events(self):
        self.data["customer"]["consent"]["personalization"] = "yes"
        with self.assertRaises(app.ValidationError):
            self.run_data()
        self.data["customer"]["consent"]["personalization"] = True
        self.data["clickstream"][1]["sequence"] = 0
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_invalid_query_and_shape(self):
        for query in ("", "   ", "!!!", None, 42):
            with self.subTest(query=query):
                self.data["query"] = query
                with self.assertRaises(app.ValidationError):
                    self.run_data()
        with self.assertRaises(app.ValidationError):
            app.run_pipeline([])

    def test_determinism_and_input_immutability(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(before, self.data)

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], ["does-not-exist.json"]):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_malformed_and_invalid_json_without_scratch_files(self):
        for content in ('{', '{"schema_version":1,"schema_version":1}', 'NaN', '{}'):
            with self.subTest(content=content):
                stream = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)):
                    with contextlib.redirect_stdout(stream):
                        code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
