"""Synthetic fixtures only; tests never write files or call external services."""

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
        self.doc = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_complete_pipeline(self):
        output = app.run_pipeline(self.doc)
        self.assertEqual(output["journey"]["steps"], ["review", "compare"])
        self.assertEqual(output["triage"]["category"], "technical")
        self.assertEqual(output["triage"]["priority"], "normal")

    def test_evidence_is_exact_and_relevant(self):
        result = app.research(self.doc)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["excerpt"], self.doc["sources"][0]["text"])
        self.assertEqual(result["confidence"], 0.9)

    def test_no_evidence_blocks_and_escalates(self):
        self.doc["sources"] = []
        output = app.run_pipeline(self.doc)
        self.assertEqual(output["research"]["status"], "insufficient")
        self.assertEqual(output["journey"]["steps"], [])
        self.assertEqual(output["triage"]["priority"], "high")
        self.assertEqual(output["triage"]["evidence_ids"], [])

    def test_zero_reliability_is_excluded(self):
        self.doc["sources"][0]["reliability"] = 0
        self.assertEqual(app.research(self.doc)["evidence"], [])

    def test_cross_stage_propagation(self):
        output = app.run_pipeline(self.doc)
        self.assertEqual(output["journey"]["evidence_ids"], ["synthetic-guide"])
        self.assertEqual(output["triage"]["evidence_ids"], output["journey"]["evidence_ids"])
        self.assertEqual(output["triage"]["action_ids"], output["journey"]["steps"])
        self.assertEqual(output["triage"]["customer_id"], self.doc["ticket"]["customer_id"])

    def test_completed_action_not_repeated(self):
        self.doc["completed"] = ["review"]
        output = app.run_pipeline(self.doc)
        self.assertEqual(output["journey"]["next_actions"], ["compare"])
        self.assertEqual(output["journey"]["status"], "blocked")

    def test_unsupported_prerequisite_is_not_skipped(self):
        self.doc["actions"][0]["evidence_terms"] = ["unrelated"]
        self.assertEqual(app.run_pipeline(self.doc)["journey"]["next_actions"], [])

    def test_independent_actions_form_valid_pair(self):
        self.doc["actions"][1]["prerequisites"] = []
        result = app.run_pipeline(self.doc)["journey"]
        self.assertEqual(result["steps"], ["compare", "review"])
        self.assertEqual(result["next_actions"], ["compare", "review"])

    def test_fallback_and_accountability(self):
        self.doc["sources"] = []
        self.doc["ticket"]["text"] = "Hello"
        self.doc["routing"]["categories"]["general"]["owner"] = "synthetic-new-owner"
        result = app.run_pipeline(self.doc)["triage"]
        self.assertEqual(result["category"], "general")
        self.assertEqual(result["owner"], "synthetic-new-owner")

    def test_journey_votes_change_ticket_category(self):
        self.doc["ticket"]["text"] = "Hello"
        self.assertEqual(app.run_pipeline(self.doc)["triage"]["category"], "technical")

    def test_critical_priority_not_downgraded(self):
        self.doc["sources"] = []
        self.doc["ticket"]["severity"] = "critical"
        self.assertEqual(app.run_pipeline(self.doc)["triage"]["priority"], "urgent")

    def test_configurable_priority(self):
        self.doc["routing"]["priority_by_severity"]["standard"] = "urgent"
        self.assertEqual(app.run_pipeline(self.doc)["triage"]["priority"], "urgent")

    def test_unknown_and_cyclic_prerequisites(self):
        for dependencies in (["missing"], ["compare"], ["review"]):
            with self.subTest(dependencies=dependencies):
                candidate = copy.deepcopy(self.doc)
                candidate["actions"][0]["prerequisites"] = dependencies
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(candidate)

    def test_invalid_input_variants(self):
        cases = [
            ("schema_version", True), ("question", "  "), ("question", "the and"),
            ("sources", {}), ("completed", ["unknown"]), ("completed", ["compare"]),
            ("actions", None), ("ticket", {}), ("routing", []),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                candidate = copy.deepcopy(self.doc)
                candidate[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(candidate)

    def test_bad_reliability(self):
        for reliability in (True, -1, 1.1, float("nan"), float("inf"), "0.9"):
            with self.subTest(reliability=reliability):
                self.doc["sources"][0]["reliability"] = reliability
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.doc)

    def test_duplicate_ids_and_missing_owner(self):
        candidate = copy.deepcopy(self.doc)
        candidate["sources"].append(copy.deepcopy(candidate["sources"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(candidate)
        self.doc["routing"]["categories"]["technical"]["owner"] = " "
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.doc)

    def test_reject_fabricated_research_handoff(self):
        evidence = app.research(self.doc)
        evidence["evidence"][0]["excerpt"] = "Invented claim"
        with self.assertRaises(app.ValidationError):
            app.journey(self.doc, evidence)

    def test_reject_invalid_journey_handoff(self):
        evidence = app.research(self.doc)
        plan = app.journey(self.doc, evidence)
        plan["steps"] = ["compare", "review"]
        with self.assertRaises(app.ValidationError):
            app.triage(self.doc, evidence, plan)

    def test_reject_lost_citations_and_unaccountable_route(self):
        output = app.run_pipeline(self.doc)
        output["triage"]["evidence_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.validate("output", output, self.doc)
        output = app.run_pipeline(self.doc)
        output["triage"]["owner"] = "someone-else"
        with self.assertRaises(app.ValidationError):
            app.validate("output", output, self.doc)

    def test_deterministic_and_nonmutating(self):
        original = copy.deepcopy(self.doc)
        first = app.run_pipeline(self.doc)
        self.doc["actions"].reverse()
        self.doc["sources"].reverse()
        self.assertEqual(first, app.run_pipeline(self.doc))
        self.doc["actions"].reverse()
        self.doc["sources"].reverse()
        self.assertEqual(original, self.doc)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_usage_and_missing_file(self):
        for args in ([], [str(ROOT / "nonexistent.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_encoding_duplicate_and_invalid_schema(self):
        for content in (b"{", b"\xff", b'{"a":1,"a":2}', b"[]",
                        b'{"schema_version":1}', b"x" * 1_000_001):
            with self.subTest(content=content[:40]):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
