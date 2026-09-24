"""Synthetic fixtures only; no filesystem writes or external dependencies."""

import contextlib
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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.state = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        self.request = self.state["data"]["request"]

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.state)
        self.assertEqual(result["stage"], "recommend")
        self.assertEqual(result["data"]["recommend"]["items"][0]["product_id"], "p-pack")
        self.assertIs(app.validate_state(result), result)

    def test_input_not_mutated(self):
        before = copy.deepcopy(self.state)
        app.run_pipeline(self.state)
        self.assertEqual(before, self.state)

    def test_research_provenance_and_ranking(self):
        evidence = app.research(self.state)["data"]["research"]["evidence"]
        self.assertEqual(evidence[0]["source_id"], "synthetic-lab")
        self.assertEqual(evidence[0]["excerpt"], self.request["sources"][0]["text"])
        self.assertIn("lightweight", evidence[0]["matched_terms"])

    def test_unanswered_question(self):
        self.request["questions"].append({"id": "q-other", "text": "astronomy telescope"})
        result = app.run_pipeline(self.state)["data"]
        self.assertIn("q-other", result["research"]["unanswered_question_ids"])
        self.assertEqual(result["documents"]["question_summary"][-1]["status"], "no_usable_evidence")

    def test_documents_filter_and_normalize(self):
        result = app.documents(app.research(self.state))["data"]["documents"]
        self.assertEqual(len(result["dropped_evidence"]), 1)
        self.assertTrue(all(row["source_id"] != "synthetic-rumor" for row in result["rows"]))
        self.assertTrue(all("  " not in row["text"] for row in result["rows"]))
        self.assertEqual(result["checks"]["accepted_rows"], len(result["rows"]))

    def test_threshold_propagates_to_discovery(self):
        initial = app.run_pipeline(self.state)["data"]["recommend"]["items"][0]
        self.request["document_options"]["minimum_reliability"] = 1
        result = app.run_pipeline(self.state)["data"]
        self.assertEqual(result["documents"]["discovery_terms"], [])
        self.assertEqual(result["recommend"]["items"][0]["score_components"]["research"], 0)
        self.assertGreater(initial["score_components"]["research"], 0)
        self.assertEqual(result["recommend"]["items"][0]["supporting_evidence"], [])

    def test_source_change_propagates(self):
        self.request["profile"]["interests"] = []
        before = app.run_pipeline(self.state)["data"]["recommend"]["items"][0]["score"]
        for source in self.request["sources"]:
            source["title"] = "Unrelated"
            source["text"] = "Astronomy telescope"
        after = app.run_pipeline(self.state)["data"]
        self.assertEqual(after["documents"]["rows"], [])
        self.assertEqual(after["recommend"]["items"][0]["score"], 0)
        self.assertGreater(before, 0)

    def test_recommendation_constraints(self):
        result = app.run_pipeline(self.state)["data"]["recommend"]
        excluded = {item["product_id"]: item["reasons"] for item in result["excluded"]}
        self.assertEqual(excluded["p-premium"], ["over_budget"])
        self.assertEqual(excluded["p-leather"], ["excluded_tag"])
        self.assertEqual(result["eligible_count"], 2)

    def test_limit(self):
        self.request["profile"]["limit"] = 1
        self.assertEqual(len(app.run_pipeline(self.state)["data"]["recommend"]["items"]), 1)

    def test_empty_sources_and_catalog(self):
        self.request["sources"] = []
        self.request["products"] = []
        result = app.run_pipeline(self.state)["data"]
        self.assertEqual(result["research"]["unanswered_question_ids"], ["q-travel", "q-weather"])
        self.assertEqual(result["recommend"]["items"], [])

    def test_all_products_over_budget(self):
        self.request["profile"]["budget"] = 0
        self.assertEqual(app.run_pipeline(self.state)["data"]["recommend"]["eligible_count"], 0)

    def test_stable_tie_breaking(self):
        self.request["sources"] = []
        self.request["profile"]["interests"] = []
        self.request["products"] = [
            {"id": name, "name": "Synthetic item", "description": "Storage", "tags": [], "price": price}
            for name, price in [("c", 20), ("b", 10), ("a", 10)]
        ]
        items = app.run_pipeline(self.state)["data"]["recommend"]["items"]
        self.assertEqual([item["product_id"] for item in items], ["a", "b", "c"])
        self.assertTrue(all(item["reason"] == "budget_eligible_fallback" for item in items))

    def test_zero_reliability_never_boosts(self):
        self.request["document_options"]["minimum_reliability"] = 0
        for source in self.request["sources"]:
            source["reliability"] = 0
        result = app.run_pipeline(self.state)["data"]
        self.assertTrue(result["documents"]["rows"])
        self.assertEqual(result["documents"]["discovery_terms"], [])

    def test_duplicate_ids_rejected(self):
        self.request["sources"].append(copy.deepcopy(self.request["sources"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.state)

    def test_invalid_numeric_inputs(self):
        for value in (True, -1, float("nan"), float("inf"), "100"):
            with self.subTest(value=value):
                self.request["profile"]["budget"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.state)

    def test_invalid_structure(self):
        for state in (None, [], {}, {"schema_version": 1}):
            with self.subTest(state=state):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(state)

    def test_unknown_fields_rejected(self):
        self.request["unexpected"] = "value"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.state)

    def test_blank_questions_rejected(self):
        for value in (" ", "the and of", "!!!"):
            self.request["questions"][0]["text"] = value
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.state)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.documents(self.state)
        with self.assertRaises(app.ValidationError):
            app.recommend(app.research(self.state))

    def test_tampered_research_rejected(self):
        state = app.research(self.state)
        state["data"]["research"]["evidence"][0]["source_id"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.documents(state)

    def test_tampered_documents_rejected(self):
        state = app.documents(app.research(self.state))
        state["data"]["documents"]["discovery_terms"][0]["weight"] = 100
        with self.assertRaises(app.ValidationError):
            app.recommend(state)

    def test_case_insensitive_unicode_tokens(self):
        self.assertEqual(app.tokens("CAFÉ waterproof"), {"café", "waterproof"})
        self.request["profile"]["excluded_tags"] = [" LEATHER "]
        result = app.run_pipeline(self.state)["data"]["recommend"]
        self.assertNotIn("p-leather", [item["product_id"] for item in result["items"]])

    def test_repeatability(self):
        self.assertEqual(app.run_pipeline(self.state), app.run_pipeline(self.state))

    def test_cli_success_subprocess(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["stage"], "recommend")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_subprocess(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "absent.json")],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_missing_argument(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_bad_json_and_invalid_schema(self):
        for content in ("{", "[]", '{"a":1,"a":2}', '{"x":NaN}', b"\xff"):
            stream = io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content)
            if isinstance(content, bytes):
                stream = io.TextIOWrapper(stream, encoding="utf-8")
            output = io.StringIO()
            with self.subTest(content=content), patch.object(Path, "open", return_value=stream):
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
