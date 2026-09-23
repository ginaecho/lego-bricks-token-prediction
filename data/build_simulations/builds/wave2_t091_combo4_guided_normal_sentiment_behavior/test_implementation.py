"""Synthetic fixtures only; no network, providers, or scratch files."""
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
        self.value = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_stage(self, name):
        return app.run_pipeline(self.value)["stages"][name]["data"]

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.value)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), ["guided", "normal", "sentiment", "behavior"])
        self.assertEqual(result["stages"]["guided"]["data"]["progress"], 1)

    def test_empty_setup_is_blocked(self):
        self.value["guided"]["steps"] = []
        result = app.run_pipeline(self.value)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stages"]["guided"]["data"]["next_step"], "profile")
        self.assertIsNone(result["stages"]["normal"])

    def test_partial_progress(self):
        self.value["guided"]["steps"] = self.value["guided"]["steps"][:2]
        result = self.run_stage("guided")
        self.assertEqual(result["progress"], 2 / 3)
        self.assertEqual(result["next_step"], "research")

    def test_prerequisite_order(self):
        self.value["guided"]["steps"].reverse()
        self.invalid()

    def test_duplicate_steps(self):
        self.value["guided"]["steps"][1] = self.value["guided"]["steps"][0]
        self.invalid()

    def test_consent_required(self):
        self.value["guided"]["steps"][1]["data"]["accepted"] = False
        self.invalid()

    def test_email_validation(self):
        self.value["guided"]["steps"][0]["data"]["email"] = "invalid"
        self.invalid()

    def test_empty_query(self):
        self.value["guided"]["steps"][2]["data"]["query"] = "!!!"
        self.invalid()

    def test_exact_unicode_citations(self):
        self.value["normal"]["documents"][0]["text"] = "  Café delivery is slow! \n Delivery is bad."
        findings = self.run_stage("normal")["findings"]
        documents = {d["id"]: d["text"] for d in self.value["normal"]["documents"]}
        self.assertTrue(any("Café" in f["citation"]["quote"] for f in findings))
        for finding in findings:
            c = finding["citation"]
            self.assertEqual(documents[c["source_id"]][c["start"]:c["end"]], c["quote"])

    def test_query_handoff(self):
        self.value["guided"]["steps"][2]["data"]["query"] = "packaging"
        stages = app.run_pipeline(self.value)["stages"]
        self.assertEqual(len(stages["normal"]["data"]["findings"]), 1)
        self.assertEqual(stages["sentiment"]["data"]["query"], "packaging")
        self.assertEqual(stages["sentiment"]["data"]["issues"][0]["label"], "positive")
        self.assertTrue(all(x["issue_score"] == 0
                            for x in stages["behavior"]["data"]["ranking"]))

    def test_newline_passages_without_punctuation(self):
        self.value["normal"]["documents"] = [
            {"id": "synthetic-lines", "text": "Delivery slow\nCheckout broken\n",
             "severity": "high"}]
        findings = self.run_stage("normal")["findings"]
        self.assertEqual([f["citation"]["quote"] for f in findings],
                         ["Delivery slow", "Checkout broken"])

    def test_timestamp_utc_overflow(self):
        self.value["behavior"]["as_of"] = "0001-01-01T00:00:00+01:00"
        self.invalid()

    def test_no_matches(self):
        self.value["guided"]["steps"][2]["data"]["query"] = "unicorn"
        self.assertEqual(self.run_stage("normal")["findings"], [])
        self.assertEqual(self.run_stage("sentiment")["issues"], [])
        self.assertEqual(self.run_stage("behavior")["ranking"][0]["id"], "delivery-guide")

    def test_empty_documents_and_catalog(self):
        self.value["normal"]["documents"] = []
        self.value["behavior"].update(catalog=[], events=[])
        self.assertEqual(self.run_stage("behavior")["ranking"], [])

    def test_top_k(self):
        self.value["normal"]["top_k"] = 1
        self.assertEqual(len(self.run_stage("normal")["findings"]), 1)
        self.assertEqual(len(self.run_stage("sentiment")["issues"]), 1)

    def test_sentiment_transparency(self):
        score, contributions = app.score_sentiment("great but slow")
        self.assertAlmostEqual(score, 1 / 3)
        self.assertEqual([c["value"] for c in contributions], [2, -1])

    def test_negation_and_neutral(self):
        self.assertEqual(app.score_sentiment("not good")[0], -1)
        self.assertEqual(app.score_sentiment("not bad")[0], 1)
        self.assertEqual(app.score_sentiment("ordinary words")[0], 0)
        self.assertEqual(app.score_sentiment("good bad")[0], 0)

    def test_severity_dominates(self):
        issues = self.run_stage("sentiment")["issues"]
        self.assertEqual(issues[0]["severity"], "critical")
        self.assertGreater(app.issue_priority({"severity": "high", "relevance": 0}, 1),
                           app.issue_priority({"severity": "medium", "relevance": 1}, -1))

    def test_insight_limit_handoff(self):
        self.value["sentiment"]["max_issues"] = 1
        ranking = self.run_stage("behavior")["ranking"]
        self.assertEqual(ranking[0]["id"], "checkout-guide")
        delivery = next(x for x in ranking if x["id"] == "delivery-guide")
        self.assertEqual(delivery["issue_score"], 0)

    def test_severity_handoff_changes_ranking(self):
        self.value["behavior"]["events"] = []
        self.assertEqual(self.run_stage("behavior")["ranking"][0]["id"], "checkout-guide")
        self.value["normal"]["documents"][0]["severity"] = "critical"
        self.value["normal"]["documents"][1]["severity"] = "low"
        self.assertEqual(self.run_stage("behavior")["ranking"][0]["id"], "delivery-guide")

    def test_cold_start_with_issues(self):
        self.value["behavior"]["events"] = []
        result = self.run_stage("behavior")
        self.assertTrue(result["cold_start"])
        self.assertGreater(result["ranking"][0]["issue_score"], 0)

    def test_cold_start_without_matches_stable_ties(self):
        self.value["behavior"]["events"] = []
        self.value["normal"]["documents"] = []
        ranking = self.run_stage("behavior")["ranking"]
        self.assertEqual([x["id"] for x in ranking], sorted(x["id"] for x in ranking))

    def test_recency_and_purchase_weight(self):
        self.value["normal"]["documents"] = []
        self.value["behavior"]["events"] = [
            {"item_id": "checkout-guide", "kind": "purchase",
             "occurred_at": "2026-08-24T12:00:00Z"},
            {"item_id": "delivery-guide", "kind": "browse",
             "occurred_at": "2026-09-23T12:00:00Z"}]
        ranking = self.run_stage("behavior")["ranking"]
        self.assertEqual(ranking[0]["history_score"], 1.5)
        self.assertEqual(ranking[1]["history_score"], 1.0)

    def test_invalid_timestamp_future_unknown_event(self):
        for update in ({"occurred_at": "2026-09-23"},
                       {"occurred_at": "2027-01-01T00:00:00Z"},
                       {"item_id": "missing"}, {"kind": "click"}):
            with self.subTest(update=update):
                value = copy.deepcopy(self.value)
                value["behavior"]["events"][0].update(update)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(value)

    def test_duplicates_and_bool_integer(self):
        self.value["normal"]["documents"].append(self.value["normal"]["documents"][0])
        self.invalid()
        self.value["normal"]["documents"].pop()
        self.value["normal"]["top_k"] = True
        self.invalid()

    def test_unknown_keys_and_synthetic_label(self):
        self.value["extra"] = 1
        self.invalid()
        del self.value["extra"]
        self.value["synthetic"] = False
        self.invalid()

    def test_invalid_shapes(self):
        for key in ("guided", "normal", "sentiment", "behavior"):
            with self.subTest(key=key):
                value = copy.deepcopy(self.value)
                value[key] = []
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(value)

    def test_upstream_validation(self):
        stages = app.run_pipeline(self.value)["stages"]
        with self.assertRaises(app.ValidationError):
            app.sentiment_stage(stages["guided"], self.value["sentiment"])
        stages["sentiment"]["data"]["issues"][0]["priority"] = 999
        with self.assertRaises(app.ValidationError):
            app.behavior_stage(stages["sentiment"], self.value["behavior"])

    def test_immutable_and_deterministic(self):
        before = copy.deepcopy(self.value)
        self.assertEqual(app.run_pipeline(self.value), app.run_pipeline(self.value))
        self.assertEqual(self.value, before)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        for text in ("{", "null", '{"a":1,"a":2}', '{"a":NaN}',
                     json.dumps(dict(self.value, schema_version=2))):
            with self.subTest(text=text), patch("builtins.open", mock_open(read_data=text)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = app.main(["synthetic-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
