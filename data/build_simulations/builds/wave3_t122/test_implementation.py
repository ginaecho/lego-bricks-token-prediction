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

    def stages(self):
        r = app.research(self.data)
        d = app.documents(r)
        i = app.insights(d)
        return r, d, i, app.onboard(i)

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["stage"], "onboard")
        app.validate(result["data"], "onboard")

    def test_research_citations(self):
        r = app.research(self.data)["results"]["research"]
        sources = {s["id"]: s["text"] for s in self.data["sources"]}
        self.assertGreater(len(r["evidence"]), 0)
        for item in r["evidence"]:
            self.assertIn(item["quote"], sources[item["source_id"]])

    def test_research_unanswered(self):
        self.assertEqual(app.research(self.data)["results"]["research"]["unanswered_question_ids"],
                         ["q-security"])

    def test_keyword_fallback_casefold(self):
        self.data["questions"] = [{"id": "q", "text": "What is SETUP?", "keywords": []}]
        self.assertTrue(app.research(self.data)["results"]["research"]["evidence"])

    def test_no_substring_matching(self):
        self.data["questions"] = [{"id": "q", "text": "How is set?", "keywords": ["set"]}]
        self.assertEqual(app.research(self.data)["results"]["research"]["evidence"], [])

    def test_documents_collapse_multiquestion_evidence(self):
        _, d, _, _ = self.stages()
        docs = d["results"]["documents"]
        self.assertEqual(docs["checks"]["duplicates_collapsed"], 2)
        row = next(r for r in docs["rows"] if r["source_id"] == "s-alex" and "Setup" in r["text"])
        self.assertEqual(row["question_ids"], ["q-adoption", "q-setup"])
        self.assertEqual(row["word_count"], 5)

    def test_insights_only_use_feedback(self):
        _, d, i, _ = self.stages()
        self.assertEqual(i["results"]["insights"]["feedback_row_count"], 5)
        reference_ids = {r["id"] for r in d["results"]["documents"]["rows"] if r["kind"] == "reference"}
        self.assertFalse(reference_ids & {rid for t in i["results"]["insights"]["themes"] for rid in t["row_ids"]})

    def test_insights_themes_and_sentiment(self):
        _, _, i, _ = self.stages()
        themes = {t["id"]: t for t in i["results"]["insights"]["themes"]}
        self.assertEqual(themes["setup"]["sentiment_counts"]["negative"], 1)
        self.assertEqual(themes["support"]["sentiment_counts"]["positive"], 1)
        self.assertIn("other", themes)

    def test_sentiment_edge_cases(self):
        for value, expected in [("easy but slow", "mixed"), ("routine workflow", "neutral"),
                                ("excellent", "positive"), ("broken", "negative")]:
            self.assertEqual(app.sentiment(value), expected)

    def test_personal_onboarding(self):
        _, _, _, o = self.stages()
        plans = {p["customer_id"]: p for p in o["results"]["onboard"]["plans"]}
        self.assertEqual(plans["c-alex"]["basis"], "personal_feedback")
        self.assertIn("setup", plans["c-alex"]["theme_ids"])
        self.assertIn("Synthetic Alex", plans["c-alex"]["next_steps"][0])
        self.assertIn("guided introduction", plans["c-alex"]["next_steps"][0])
        self.assertIn("focused pilot", plans["c-jules"]["next_steps"][0])

    def test_shared_onboarding(self):
        plan = self.stages()[3]["results"]["onboard"]["plans"][2]
        self.assertEqual(plan["basis"], "shared_feedback")
        self.assertTrue(plan["evidence_row_ids"])

    def test_cross_stage_customer_lineage(self):
        _, d, _, o = self.stages()
        rows = {r["id"]: r for r in d["results"]["documents"]["rows"]}
        for plan in o["results"]["onboard"]["plans"][:2]:
            for rid in plan["evidence_row_ids"]:
                self.assertEqual(rows[rid]["customer_id"], plan["customer_id"])
                self.assertTrue(rows[rid]["evidence_ids"])

    def test_empty_sources_goal_only(self):
        self.data["sources"] = []
        result = app.run_pipeline(self.data)["data"]["results"]
        self.assertEqual(result["documents"]["rows"], [])
        self.assertEqual(result["insights"]["themes"], [])
        self.assertTrue(all(p["basis"] == "goal_only" for p in result["onboard"]["plans"]))

    def test_no_customers(self):
        self.data["customers"] = []
        for source in self.data["sources"]:
            source["customer_id"] = None
        self.assertEqual(app.run_pipeline(self.data)["data"]["results"]["onboard"]["plans"], [])

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_stage_order_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.documents(self.data)

    def test_invalid_schema_variants(self):
        for key, value in [("schema_version", True), ("stage", []), ("questions", []),
                           ("sources", {}), ("fixture_label", "Real data"), ("customers", None)]:
            with self.subTest(key=key):
                invalid = copy.deepcopy(self.data)
                invalid[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(invalid)

    def test_unknown_fields_rejected(self):
        self.data["surprise"] = 1
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_duplicate_ids_rejected(self):
        self.data["sources"].append(copy.deepcopy(self.data["sources"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_unknown_customer_rejected(self):
        self.data["sources"][1]["customer_id"] = "missing"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_empty_source_text_rejected(self):
        self.data["sources"][0]["text"] = " "
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_tampered_evidence_rejected(self):
        r = app.research(self.data)
        r["results"]["research"]["evidence"][0]["quote"] = "Fabricated claim"
        with self.assertRaises(app.ValidationError):
            app.documents(r)

    def test_tampered_document_rejected(self):
        d = self.stages()[1]
        d["results"]["documents"]["rows"][0]["word_count"] = 99
        with self.assertRaises(app.ValidationError):
            app.insights(d)

    def test_tampered_insights_rejected(self):
        i = self.stages()[2]
        i["results"]["insights"]["themes"][0]["customer_ids"] = ["c-sam"]
        with self.assertRaises(app.ValidationError):
            app.onboard(i)

    def test_tampered_onboarding_rejected(self):
        o = self.stages()[3]
        o["results"]["onboard"]["plans"][0]["evidence_row_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.validate(o)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "nonexistent.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")
        self.assertEqual(proc.stderr, "")

    def test_cli_bad_arguments(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out.getvalue())["status"], "error")

    def test_cli_malformed_and_invalid_json(self):
        for content in ("{broken", "null", "[]", '{"x": NaN}', '{"schema_version": 1}'):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    code = app.main(["synthetic-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")

    def test_unicode_tokenization(self):
        self.data["questions"] = [{"id": "q", "text": "Café?", "keywords": ["CAFÉ"]}]
        self.data["sources"][0]["text"] = "Le café est excellent."
        evidence = app.research(self.data)["results"]["research"]["evidence"]
        self.assertEqual(evidence[0]["matched_terms"], ["café"])


if __name__ == "__main__":
    unittest.main()
