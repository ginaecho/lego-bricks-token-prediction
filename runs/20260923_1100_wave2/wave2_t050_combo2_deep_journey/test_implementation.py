"""Tests use only the labeled synthetic fixture; no providers or networks."""

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


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_multidocument_synthesis(self):
        research = app.deep_research(self.source)
        basics = research["findings"][0]
        self.assertEqual(basics["assessment"], "supported")
        self.assertEqual(basics["document_count"], 2)
        self.assertEqual(len(basics["evidence"]), 2)

    def test_disagreement_and_missing_evidence(self):
        research = app.deep_research(self.source)
        self.assertEqual(research["disagreements"][0]["question_id"], "q-format")
        self.assertEqual([item["reason"] for item in research["unresolved_questions"]],
                         ["conflicting evidence", "no evidence"])

    def test_valid_two_step_dependency(self):
        result = app.run_pipeline(self.source)
        journey = result["journey"]
        self.assertEqual(journey["status"], "ready")
        self.assertEqual([step["action_id"] for step in journey["steps"]],
                         ["intro", "compare-formats"])
        self.assertEqual(journey["steps"][1]["prerequisites_satisfied"], ["intro"])
        self.assertEqual(journey["next_actions"], ["intro"])

    def test_cross_stage_provenance_and_unresolved_propagation(self):
        result = app.run_pipeline(self.source)
        research, journey = result["research"], result["journey"]
        self.assertEqual(journey["unresolved_questions"], research["unresolved_questions"])
        finding = next(item for item in research["findings"] if item["question_id"] == "q-format")
        rationale = journey["steps"][1]["rationale"][0]
        self.assertEqual(rationale["assessment"], finding["assessment"])
        self.assertEqual(rationale["evidence"], finding["evidence"])
        unsafe = next(item for item in journey["blocked_actions"]
                      if item["action_id"] == "commit-video")
        self.assertEqual(unsafe["unsupported_questions"], ["q-format"])

    def test_research_change_affects_action_safety(self):
        self.source["documents"][1]["claims"][1]["stance"] = "support"
        journey = app.run_pipeline(self.source)["journey"]
        self.assertNotIn("commit-video", [item["action_id"] for item in journey["blocked_actions"]
                                        if item["unsupported_questions"]])
        self.assertEqual(journey["steps"][1]["rationale"][0]["assessment"], "supported")

    def test_completed_actions_unlock_followups(self):
        self.source["user"]["completed_actions"] = ["intro"]
        journey = app.run_pipeline(self.source)["journey"]
        self.assertEqual([item["action_id"] for item in journey["steps"]],
                         ["compare-formats", "retention-check"])

    def test_empty_documents_block_application(self):
        self.source["documents"] = []
        result = app.run_pipeline(self.source)
        self.assertTrue(all(item["assessment"] == "unresolved" for item in result["research"]["findings"]))
        self.assertEqual(result["journey"]["status"], "blocked")
        self.assertEqual(result["journey"]["steps"], [])

    def test_single_available_action_not_fabricated_pair(self):
        self.source["actions"] = self.source["actions"][:1]
        journey = app.run_pipeline(self.source)["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual(journey["steps"], [])
        self.assertEqual(journey["next_actions"], ["intro"])

    def test_uncertain_claim_prevents_overstatement(self):
        self.source["documents"][1]["claims"][0]["stance"] = "uncertain"
        self.assertEqual(app.deep_research(self.source)["findings"][0]["assessment"], "unresolved")

    def test_opposition_is_not_support(self):
        for document in self.source["documents"]:
            document["claims"][0]["stance"] = "oppose"
        result = app.run_pipeline(self.source)
        self.assertEqual(result["research"]["findings"][0]["assessment"], "opposed")
        self.assertEqual(result["journey"]["status"], "blocked")

    def test_invalid_inputs(self):
        variants = []
        for key, value in [("schema_version", True), ("synthetic", False), ("questions", []),
                           ("documents", {}), ("actions", None)]:
            source = copy.deepcopy(self.source)
            source[key] = value
            variants.append(source)
        source = copy.deepcopy(self.source)
        source["documents"][0]["claims"][0]["question_id"] = "missing"
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["actions"][0]["prerequisites"] = ["unknown"]
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["actions"][0]["prerequisites"] = ["compare-formats"]
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["questions"].append(source["questions"][0])
        variants.append(source)
        source = copy.deepcopy(self.source)
        source["user"]["completed_actions"] = ["compare-formats"]
        variants.append(source)
        for source in variants:
            with self.subTest(source=source), self.assertRaises(app.ValidationError):
                app.run_pipeline(source)

    def test_tampered_research_handoff_rejected(self):
        research = app.deep_research(self.source)
        research["findings"][0]["evidence"][0]["quote"] = "Invented quotation"
        with self.assertRaises(app.ValidationError):
            app.recommend_journey(self.source, research)

    def test_tampered_journey_rejected(self):
        result = app.run_pipeline(self.source)
        result["journey"]["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate(result, "output", self.source)

    def test_personalization_changes_selection(self):
        self.source["actions"].append({
            "id": "retention-now", "title": "Explore retention", "kind": "research",
            "question_ids": ["q-retention"], "prerequisites": []})
        self.source["user"]["interests"] = ["q-retention"]
        journey = app.run_pipeline(self.source)["journey"]
        self.assertEqual(journey["steps"][0]["action_id"], "retention-now")

    def test_deterministic_and_nonmutating(self):
        original = copy.deepcopy(self.source)
        expected = app.run_pipeline(self.source)
        self.assertEqual(self.source, original)
        self.source["documents"].reverse()
        self.source["questions"].reverse()
        self.source["actions"].reverse()
        self.assertEqual(app.run_pipeline(self.source), expected)

    def test_no_actions(self):
        self.source["actions"] = []
        self.assertEqual(app.run_pipeline(self.source)["journey"]["next_actions"], [])

    def test_same_document_not_counted_twice(self):
        self.source["documents"][0]["claims"].append(
            {"question_id": "q-basics", "stance": "support", "quote": "Another synthetic observation."})
        self.assertEqual(app.deep_research(self.source)["findings"][0]["document_count"], 2)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "nonexistent.json")], ["one", "two"]]:
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                        cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for payload in ["{", "[]", '{"schema_version":1,"schema_version":1}', '{"x":NaN}']:
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), redirect_stdout(output):
                    status = app.main(["synthetic-in-memory.json"])
                self.assertEqual(status, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
