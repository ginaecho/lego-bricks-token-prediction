"""Synthetic fixtures only. Tests create no files and contact no providers."""

import copy
import contextlib
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

    def test_full_pipeline_and_determinism(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(original, self.data)
        self.assertEqual(app.validate(first, "research"), first)

    def test_transparent_sentiment(self):
        issue = app.sentiment_stage(self.data)["sentiment"]["issues"][0]
        self.assertEqual(issue["score"], -7)
        self.assertEqual(issue["score"], sum(e["contribution"] for e in issue["evidence"]))
        self.assertEqual(issue["priority"], 100)

    def test_negation(self):
        self.data["feedback"][0]["text"] = "not good"
        issue = app.score_sentiment(self.data)["issues"][0]
        self.assertEqual(issue["score"], -1)
        self.assertTrue(issue["evidence"][0]["negated"])
        self.data["feedback"][0]["text"] = "not never good"
        self.assertEqual(app.score_sentiment(self.data)["issues"][0]["score"], 1)

    def test_neutral_unknown_text(self):
        self.data["feedback"][0]["text"] = "ordinary object"
        self.assertEqual(app.score_sentiment(self.data)["issues"][0]["label"], "neutral")

    def test_severity_dominates_negative_magnitude(self):
        self.data["feedback"][0].update(text="good", severity="critical")
        self.data["feedback"][1].update(text="unsafe " * 20, severity="high")
        self.assertEqual(app.score_sentiment(self.data)["issues"][0]["feedback_id"], "f1")

    def test_recency_and_purchase_weights(self):
        behavior = app.run_pipeline(self.data)["behavior"]
        signals = {s["event_id"]: s for s in behavior["signals"]}
        self.assertAlmostEqual(signals["e1"]["weight"], 30 / 31, places=7)
        self.assertEqual(signals["e2"]["weight"], 1.5)
        self.data["events"][1]["at"] = self.data["as_of"]
        self.assertEqual(app.run_pipeline(self.data)["behavior"]["signals"][1]["weight"], 3)

    def test_sentiment_to_behavior_handoff(self):
        first = app.run_pipeline(self.data)
        self.data["feedback"] = []
        second = app.run_pipeline(self.data)
        rank = lambda out: next(r for r in out["behavior"]["rankings"] if r["product_id"] == "lamp")
        self.assertEqual(rank(first)["components"]["issue_attention"], 1)
        self.assertEqual(rank(second)["components"]["issue_attention"], 0)
        self.assertGreater(rank(first)["score"], rank(second)["score"])

    def test_behavior_to_research_handoff(self):
        result = app.run_pipeline(self.data)
        self.assertEqual([r["product_id"] for r in result["research"]["results"]],
                         result["behavior"]["selected_product_ids"])
        for finding in result["research"]["results"]:
            rank = next(r for r in result["behavior"]["rankings"]
                        if r["product_id"] == finding["product_id"])
            self.assertEqual(finding["behavior_score"], rank["score"])
            self.assertEqual(finding["issue_ids"], rank["issue_ids"])
        lamp = next(r for r in result["research"]["results"] if r["product_id"] == "lamp")
        self.assertIn("unsafe", lamp["query_terms"])

    def test_exact_citations_and_extraction(self):
        self.data["sources"][0]["text"] = "  Synthetic lamp safety — café.\r\n\r\n  A broken lamp needs repair.  "
        result = app.run_pipeline(self.data)
        sources = {s["id"]: s["text"] for s in self.data["sources"]}
        found = 0
        for group in result["research"]["results"]:
            for finding in group["findings"]:
                cite = finding["citation"]
                self.assertEqual(sources[cite["source_id"]][cite["start"]:cite["end"]], cite["quote"])
                self.assertEqual(finding["finding"], cite["quote"])
                found += 1
        self.assertGreater(found, 0)

    def test_cold_start_popularity_and_ties(self):
        self.data["events"] = []
        self.data["feedback"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(result["behavior"]["cold_start"])
        self.assertEqual(result["behavior"]["rankings"][0]["product_id"], "chair")
        for product in self.data["catalog"]:
            product["popularity"] = 0
        self.assertEqual(app.run_pipeline(self.data)["behavior"]["selected_product_ids"],
                         ["chair", "lamp", "lamp-b"])

    def test_no_evidence(self):
        self.data["sources"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(all(r["status"] == "no_evidence" and not r["findings"]
                            for r in result["research"]["results"]))

    def test_passage_boundaries_and_trailing_whitespace(self):
        source = {"text": "  First lamp.\r\n\r\n  Last lamp.  "}
        extracted = list(app.passages(source))
        self.assertEqual([p[2] for p in extracted], ["First lamp.", "Last lamp."])
        for start, end, quote in extracted:
            self.assertEqual(source["text"][start:end], quote)
        self.assertEqual(list(app.passages({"text": " x "})), [(1, 2, "x")])

    def test_unrelated_sources_have_no_findings(self):
        self.data["sources"] = [{"id": "none", "title": "Synthetic unrelated",
                                 "text": "volcano geology basalt magma"}]
        result = app.run_pipeline(self.data)
        self.assertTrue(all(r["status"] == "no_evidence" for r in result["research"]["results"]))

    def test_empty_collections(self):
        for key in ("feedback", "events", "catalog", "sources"):
            self.data[key] = []
        self.assertEqual(app.run_pipeline(self.data)["research"]["results"], [])

    def test_invalid_inputs(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(extra=1),
            lambda d: d.update(as_of="2026-09-23"),
            lambda d: d["feedback"][0].update(severity="urgent"),
            lambda d: d["feedback"][0].update(customer_id="another"),
            lambda d: d["feedback"][0].update(category="unknown"),
            lambda d: d["events"][0].update(product_id="missing"),
            lambda d: d["events"][0].update(kind="click"),
            lambda d: d["events"][0].update(at="2027-01-01T00:00:00Z"),
            lambda d: d["catalog"][0].update(popularity=float("nan")),
            lambda d: d["catalog"][0].update(popularity=True),
            lambda d: d["catalog"][0].update(popularity=10 ** 1000),
            lambda d: d["sources"].append(copy.deepcopy(d["sources"][0])),
            lambda d: d.update(events={}),
            lambda d: d["feedback"][0].update(text=" "),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                fixture = copy.deepcopy(self.data)
                mutate(fixture)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(fixture)

    def test_tampered_handoffs_rejected(self):
        first = app.sentiment_stage(self.data)
        first["sentiment"]["issues"][0]["priority"] = 0
        with self.assertRaises(app.ValidationError):
            app.behavior_stage(first)
        second = app.behavior_stage(app.sentiment_stage(self.data))
        second["behavior"]["selected_product_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            app.research_stage(second)

    def test_research_validation_rejects_forged_citation(self):
        result = app.run_pipeline(self.data)
        result["research"]["results"][0]["findings"][0]["citation"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.validate(result, "research")

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                     capture_output=True, text=True, cwd=ROOT, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        for content in ('{', '{"a":1,"a":2}', 'NaN', 'null', '[]', '{"schema_version":1}'):
            with self.subTest(content=content):
                stdout = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(stdout):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
