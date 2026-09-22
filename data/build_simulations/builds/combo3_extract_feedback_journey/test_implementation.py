"""Labeled synthetic fixtures; no external data or services."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


def synthetic_fixture(label="baseline"):
    data = json.loads(Path(__file__).with_name("example_input.json").read_text(encoding="utf-8"))
    if label == "support-only":
        data["feedback"] = data["feedback"][:1]
    elif label == "missing-channel":
        data["claims"] = [c for c in data["claims"] if c["id"] != "channel"]
    elif label == "contradictory-goal":
        data["sources"].append({"id": "synthetic-conflict", "text": "retention"})
        data["claims"].append({"id": "conflict", "field": "profile.goal", "value": "retention",
                               "evidence": [{"source_id": "synthetic-conflict",
                                             "start": 0, "end": 9}]})
    elif label != "baseline":
        raise ValueError(label)
    return data


class CompositionTests(unittest.TestCase):
    def ids(self, result):
        return [s["action_id"] for s in result["journey"]["steps"]]

    def test_baseline_two_steps_and_disagreement(self):
        result = app.run_pipeline(synthetic_fixture())
        self.assertEqual(self.ids(result), ["review-setup", "run-pilot"])
        self.assertTrue(result["feedback"]["setup_friction"]["disagreement"])
        self.assertEqual(result["feedback"]["setup_friction"]["net_support"], 0)
        self.assertEqual(result["journey"]["steps"][1]["conditional_on_completion"],
                         ["review-setup"])
        self.assertEqual(result["extraction"]["profile.role"]["candidates"][0]["evidence"][0]["quote"],
                         "owner")
        self.assertEqual(result["journey"]["steps"][0]["feedback_impacts"][0]["feedback_ids"],
                         ["confused", "not-confused"])

    def test_missing_extraction_blocks_only_affected_action(self):
        result = app.run_pipeline(synthetic_fixture("missing-channel"))
        self.assertEqual(self.ids(result), ["review-setup", "review-messaging"])
        self.assertIn("requirements.channel:missing", result["journey"]["not_selected"]["run-pilot"])

    def test_feedback_changes_priority_and_plan_order(self):
        data = synthetic_fixture("support-only")
        data["actions"][0]["priority"] = 0
        with_feedback = app.run_pipeline(data)
        data["feedback"] = []
        without_feedback = app.run_pipeline(data)
        self.assertEqual(self.ids(with_feedback), ["review-setup", "run-pilot"])
        self.assertEqual(self.ids(without_feedback), ["review-messaging", "review-setup"])

    def test_extracted_context_controls_feedback_then_priority(self):
        data = synthetic_fixture("support-only")
        data["actions"][0]["requires_fields"] = []
        data["actions"][0]["priority"] = 0
        self.assertEqual(self.ids(app.run_pipeline(data))[0], "review-setup")
        data["claims"] = [c for c in data["claims"] if c["id"] != "goal"]
        result = app.run_pipeline(data)
        self.assertEqual(self.ids(result)[0], "review-messaging")
        self.assertEqual(result["feedback"]["setup_friction"]["inactive"][0]["reasons"],
                         ["profile.goal:missing"])

    def test_context_mismatch_excludes_feedback(self):
        data = synthetic_fixture("support-only")
        data["feedback"][0]["applies_to"]["profile.goal"] = "retention"
        result = app.run_pipeline(data)
        self.assertEqual(result["feedback"]["setup_friction"]["net_support"], 0)
        self.assertIn("profile.goal:context_mismatch",
                      result["feedback"]["setup_friction"]["inactive"][0]["reasons"])

    def test_contradictions_block_context_and_prerequisites(self):
        result = app.run_pipeline(synthetic_fixture("contradictory-goal"))
        self.assertEqual(result["extraction"]["profile.goal"]["status"], "contradictory")
        self.assertEqual(len(result["extraction"]["profile.goal"]["candidates"]), 2)
        self.assertEqual(self.ids(result), ["review-messaging"])
        self.assertEqual(len(result["feedback"]["setup_friction"]["inactive"]), 2)
        self.assertIn("prerequisite_not_planned:review-setup",
                      result["journey"]["not_selected"]["run-pilot"])

    def test_absent_evidence_is_not_support(self):
        data = synthetic_fixture()
        for row in data["claims"] + data["feedback"] + data["actions"]:
            row["evidence"] = []
        result = app.run_pipeline(data)
        self.assertEqual(result["journey"]["status"], "no_eligible_action")
        self.assertEqual(result["extraction"]["profile.goal"]["status"], "missing")
        self.assertEqual(result["feedback"]["setup_friction"]["net_support"], 0)

    def test_empty_valid_input(self):
        result = app.run_pipeline({k: [] for k in ("sources", "claims", "feedback", "actions")})
        self.assertEqual(result["journey"]["status"], "no_eligible_action")

    def test_same_value_claims_merge_without_false_conflict(self):
        data = synthetic_fixture()
        row = copy.deepcopy(data["claims"][0])
        row["id"] = "same-role"
        data["claims"].append(row)
        field = app.run_pipeline(data)["extraction"]["profile.role"]
        self.assertEqual(field["status"], "resolved")
        self.assertEqual(len(field["candidates"][0]["claim_ids"]), 2)
        self.assertEqual(len(field["candidates"][0]["evidence"]), 1)

    def test_cycles_and_unknown_dependencies(self):
        for dependency in ("run-pilot", "review-setup", "unknown"):
            with self.subTest(synthetic_dependency=dependency):
                data = synthetic_fixture()
                data["actions"][0]["prerequisites"] = [dependency]
                with self.assertRaises(app.InputError):
                    app.run_pipeline(data)

    def test_duplicate_record_ids_all_collections(self):
        for collection in ("sources", "claims", "feedback", "actions"):
            with self.subTest(synthetic_duplicate=collection):
                data = synthetic_fixture()
                data[collection].append(copy.deepcopy(data[collection][0]))
                with self.assertRaises(app.InputError):
                    app.run_pipeline(data)

    def test_malformed_schema_and_spans(self):
        cases = [
            ("sources", 0, "text", None),
            ("claims", 0, "field", []),
            ("claims", 0, "value", "not in source"),
            ("feedback", 0, "stance", []),
            ("feedback", 0, "applies_to", {"unknown": "x"}),
            ("actions", 0, "priority", True),
            ("actions", 0, "theme_weights", {"setup_friction": 1.2}),
            ("actions", 0, "requires_fields", ["unknown"]),
            ("actions", 0, "prerequisites", ["run-pilot", "run-pilot"]),
            ("claims", 0, "evidence", [{"source_id": "unknown", "start": 0, "end": 1}]),
            ("claims", 0, "evidence", [{"source_id": "synthetic-profile", "start": True, "end": 5}]),
            ("claims", 0, "evidence", [{"source_id": "synthetic-profile", "start": 0, "end": 999}]),
            ("claims", 0, "evidence", [{"source_id": "synthetic-profile", "start": 5, "end": 0}]),
        ]
        for collection, index, key, value in cases:
            with self.subTest(synthetic_malformed=(collection, key, value)):
                data = synthetic_fixture()
                data[collection][index][key] = value
                with self.assertRaises(app.InputError):
                    app.run_pipeline(data)
        for data in (None, [], {}, {"sources": None, "claims": [], "feedback": [], "actions": []}):
            with self.assertRaises(app.InputError):
                app.run_pipeline(data)

    def test_duplicate_evidence_rejected(self):
        data = synthetic_fixture()
        data["claims"][0]["evidence"] *= 2
        with self.assertRaises(app.InputError):
            app.run_pipeline(data)

    def test_grounding_includes_both_stages_and_action(self):
        result = app.run_pipeline(synthetic_fixture())
        step = result["journey"]["steps"][1]
        self.assertEqual(step["source_ids"], ["synthetic-catalog", "synthetic-feedback-a",
                                              "synthetic-feedback-b", "synthetic-profile"])
        self.assertEqual(step["required_context"]["requirements.channel"], "email")

    def test_deterministic_tie_break_and_no_input_mutation(self):
        data = synthetic_fixture()
        data["feedback"] = []
        for action in data["actions"]:
            action["priority"] = 0
        before = copy.deepcopy(data)
        self.assertEqual(self.ids(app.run_pipeline(data))[0], "review-messaging")
        self.assertEqual(data, before)

    def callback(self, result):
        return {"steps": [{"action_id": s["action_id"], "source_ids": s["source_ids"],
                           "explanation": "Synthetic explanation."}
                          for s in result["journey"]["steps"]]}

    def test_callback_preserves_decisions_and_sources(self):
        result = app.run_pipeline(synthetic_fixture(), self.callback)
        self.assertEqual(self.ids(result), ["review-setup", "run-pilot"])
        self.assertEqual(result["journey"]["steps"][0]["supplemental_explanation"]["semantic_verification"],
                         "not_performed")

    def test_callback_rejects_changed_ids_sources_order_and_length(self):
        for mutation in ("id", "sources", "order", "length", "text"):
            def callback(result):
                output = self.callback(result)
                if mutation == "id":
                    output["steps"][0]["action_id"] = "invented"
                elif mutation == "sources":
                    output["steps"][0]["source_ids"] = ["invented"]
                elif mutation == "order":
                    output["steps"].reverse()
                elif mutation == "length":
                    output["steps"].pop()
                else:
                    output["steps"][0]["explanation"] = ""
                return output
            with self.subTest(synthetic_callback=mutation), self.assertRaises(app.InputError):
                app.run_pipeline(synthetic_fixture(), callback)

    def test_callback_mutation_cannot_change_actual_plan(self):
        def callback(result):
            result["journey"]["steps"][0]["score"] = 999
            return self.callback(result)
        result = app.run_pipeline(synthetic_fixture(), callback)
        self.assertEqual(result["journey"]["steps"][0]["score"], 3)

    def test_cli_invalid_json_duplicate_keys_constants_and_missing_file(self):
        for raw in ('{', '{"sources":[],"sources":[]}', '{"sources":NaN}'):
            with self.subTest(synthetic_json=raw), patch("builtins.open", mock_open(read_data=raw)):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(app.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["error"]["type"], "invalid_input")
        for argv in ([], ["a", "b"], ["nonexistent-synthetic-file.json"]):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(argv), 2)

    def test_exact_cli(self):
        process = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                                 cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(self.ids(json.loads(process.stdout)), ["review-setup", "run-pilot"])


if __name__ == "__main__":
    unittest.main()
