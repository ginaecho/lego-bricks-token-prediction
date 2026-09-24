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

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_full_pipeline(self):
        result = app.run(self.data)
        self.assertEqual(result["stage"], "research")
        self.assertEqual(len(result["records"]), 3)
        self.assertTrue(result["records"][2]["research"]["findings"])

    def test_sentiment_transparency(self):
        result = app.score_sentiment("happy but frustrated and angry")
        self.assertEqual(result["score"], -1)
        self.assertEqual(sum(e["weight"] for e in result["evidence"]), -1)

    def test_negation(self):
        self.assertEqual(app.score_sentiment("not helpful")["score"], -1)
        self.assertEqual(app.score_sentiment("not angry")["score"], 1)

    def test_severity_overrides_positive(self):
        self.assertGreater(app.score_sentiment("urgent happy excellent")["priority"],
                           app.score_sentiment("angry frustrated poor")["priority"])

    def test_neutral(self):
        self.assertEqual(app.score_sentiment("routine information")["label"], "neutral")

    def test_synonym_search(self):
        insight = app.score_sentiment("insurer preauthorization")
        result = app.search_insight(insight, self.data["catalog"])
        self.assertEqual(result["results"][0]["product_id"], "authorization-guide")
        self.assertEqual(result["results"][0]["matched_terms"], ["authorization", "insurance"])

    def test_general_purpose_search(self):
        catalog = [{"id": "shoe", "title": "Running shoe", "description": "lightweight footwear",
                    "keywords": ["sport"], "source_ids": []}]
        self.assertEqual(app.search_insight(app.score_sentiment("lightweight shoe"), catalog)
                         ["results"][0]["score"], 2)

    def test_exact_citations(self):
        result = app.run(self.data)
        passages = {(s["id"], p["id"]): p["text"]
                    for s in self.data["sources"] for p in s["passages"]}
        for row in result["records"]:
            allowed = {s for hit in row["search"]["results"] for s in hit["source_ids"]}
            for finding in row["research"]["findings"]:
                original = passages[finding["source_id"], finding["passage_id"]]
                self.assertEqual(original[finding["start"]:finding["end"]], finding["quote"])
                self.assertIn(finding["source_id"], allowed)

    def test_cross_stage_propagation(self):
        for row in app.run(self.data)["records"]:
            self.assertEqual(row["sentiment"]["query_terms"], row["search"]["query_terms"])
            self.assertEqual(row["sentiment"]["priority"], row["research"]["priority"])

    def test_patient_aggregates_notes_and_request(self):
        result = app.run(self.data)
        self.assertEqual(result["records"][0]["sentiment"]["severity"], 3)

    def test_no_match(self):
        self.data["catalog"] = []
        result = app.run(self.data)
        self.assertEqual(result["records"][1]["research"]["findings"], [])
        self.assertEqual(result["records"][1]["research"]["evidence_status"],
                         "no_matching_evidence")

    def test_empty_sources(self):
        self.data["sources"] = []
        for product in self.data["catalog"]:
            product["source_ids"] = []
        self.assertFalse(app.run(self.data)["records"][0]["research"]["findings"])

    def test_immutable_deterministic(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, original)

    def test_audit_all_changes(self):
        result = app.run(self.data)
        self.assertEqual(len(result["audit_trail"]), 9)
        self.assertEqual([a["sequence"] for a in result["audit_trail"]], list(range(1, 10)))
        self.assertEqual([r["resource"] for r in result["records"]], self.data["resources"])
        self.assertEqual(result["records"][2]["resource"]["status"], "pending")

    def test_tampered_handoff_rejected(self):
        result = app.advance(app.start(self.data), "sentiment")
        result["records"][0]["sentiment"]["query_terms"] = ["invented"]
        with self.assertRaises(app.ValidationError):
            app.advance(result, "search")

    def test_missing_audit_rejected(self):
        result = app.advance(app.start(self.data), "sentiment")
        result["audit_trail"].pop()
        with self.assertRaises(app.ValidationError):
            app.advance(result, "search")

    def test_stage_order(self):
        with self.assertRaises(app.ValidationError):
            app.advance(app.start(self.data), "research")

    def test_patient_identifiers_blocked(self):
        for key, value in [("name", "Example"), ("birthDate", "2000-01-01"),
                           ("identifier", "123")]:
            with self.subTest(key=key):
                modified = copy.deepcopy(self.data)
                modified["resources"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(modified)

    def test_text_identifier_blocked(self):
        for content in ["MRN 123456", "DOB 2000-01-01", "email a@example.org",
                        "Call 555-123-4567", "Patient name: Example"]:
            with self.subTest(content=content):
                self.data["resources"][1]["text"] = content
                self.invalid()

    def test_review_mandatory(self):
        self.data["resources"][2]["human_review_required"] = False
        self.invalid()

    def test_no_autonomous_decision(self):
        result = app.run(self.data)
        self.assertIsNone(result["clinical_decision"])
        self.assertTrue(all(r["research"]["clinical_decision"] is None
                            and r["research"]["human_review_required"]
                            for r in result["records"]))

    def test_real_data_rejected(self):
        self.data["synthetic"] = False
        self.invalid()

    def test_unknown_subject(self):
        self.data["resources"][1]["subject"] = "Patient/anon-p999"
        self.invalid()

    def test_duplicate_ids(self):
        self.data["resources"].append(copy.deepcopy(self.data["resources"][0]))
        self.invalid()

    def test_unknown_source(self):
        self.data["catalog"][0]["source_ids"] = ["missing"]
        self.invalid()

    def test_invalid_types(self):
        for key in ["resources", "catalog", "sources"]:
            with self.subTest(key=key):
                broken = copy.deepcopy(self.data)
                broken[key] = "not a list"
                with self.assertRaises(app.ValidationError):
                    app.run(broken)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "does-not-exist.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for payload in ["{", "null", '{"synthetic": true}', '{"a":1,"a":2}', '{"x":NaN}']:
            with self.subTest(payload=payload), patch("builtins.open", mock_open(read_data=payload)):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_usage(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
