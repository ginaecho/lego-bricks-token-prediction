import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_full_pipeline(self):
        output = app.run_pipeline(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertTrue(output["synthetic"])
        self.assertEqual(output["schema_version"], 1)

    def test_progress_and_ready_blocked_steps(self):
        guided = app.guided_stage(self.data)["guided"]
        self.assertEqual(guided["progress"], {"completed": 2, "total": 4, "fraction": 0.5})
        self.assertEqual(guided["next_step_ids"], ["tutorial"])
        self.assertEqual(guided["steps"][2]["missing_prerequisites"], ["billing_enabled"])

    def test_prerequisite_completion_rejected(self):
        self.data["steps"][2]["completed"] = True
        with self.assertRaisesRegex(app.ValidationError, "blocked step"):
            app.run_pipeline(self.data)

    def test_dependency_completion_rejected(self):
        self.data["steps"][0]["completed"] = False
        with self.assertRaisesRegex(app.ValidationError, "blocked step"):
            app.run_pipeline(self.data)

    def test_waiting_dependency_is_tracked(self):
        self.data["steps"][0]["completed"] = False
        self.data["steps"][1]["completed"] = False
        self.data["feedback"] = []
        self.data["research_questions"] = []
        guided = app.run_pipeline(self.data)["guided"]
        self.assertEqual(guided["steps"][1]["waiting_for"], ["account"])

    def test_dependency_cycle_rejected(self):
        self.data["steps"][0]["depends_on"] = ["catalog"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            app.run_pipeline(self.data)

    def test_unknown_references(self):
        for field, value in (("depends_on", ["missing"]), ("prerequisites", ["missing"])):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["steps"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_positive_negative_and_unknown_scores(self):
        for phrase, score in (("GREAT helpful", 1), ("broken awful", -1),
                              ("ordinary screen", 0), ("good bad", 0)):
            with self.subTest(phrase=phrase):
                self.assertEqual(app.score_sentiment(phrase)["score"], score)

    def test_negation_and_punctuation(self):
        self.assertEqual(app.score_sentiment("not good")["score"], -1)
        self.assertEqual(app.score_sentiment("not bad")["score"], 1)
        self.assertEqual(app.score_sentiment("not. good")["score"], 1)
        self.assertEqual(app.score_sentiment("not never bad")["score"], -1)
        self.assertEqual(app.score_sentiment("not one two three good")["score"], 1)

    def test_priority_severity_dominates(self):
        issues = app.run_pipeline(self.data)["sentiment"]["issues"]
        self.assertEqual([row["id"] for row in issues], ["f3", "f1", "f2"])
        self.assertEqual(issues[1]["priority_score"], 90)
        self.assertEqual(issues[1]["priority_explanation"]["negative_sentiment_bonus"], 20)

    def test_priority_tie_is_stable(self):
        self.data["feedback"][0]["severity"] = "low"
        self.data["feedback"][0]["text"] = "good"
        issues = app.run_pipeline(self.data)["sentiment"]["issues"]
        self.assertEqual([row["id"] for row in issues], ["f3", "f1", "f2"])

    def test_feedback_on_incomplete_step_rejected(self):
        self.data["feedback"][0]["step_id"] = "checkout"
        with self.assertRaisesRegex(app.ValidationError, "completed onboarding"):
            app.run_pipeline(self.data)

    def test_cross_stage_provenance(self):
        output = app.run_pipeline(self.data)
        self.assertEqual(output["sentiment"]["source_completed_step_ids"],
                         output["guided"]["completed_step_ids"])
        finding = output["deep"]["findings"][0]
        self.assertEqual(finding["source_issue_ids"], ["f3", "f1"])
        self.assertEqual(finding["source_step_ids"], ["catalog"])
        self.assertEqual(finding["priority_score"], 100)
        self.assertEqual(finding["source_sentiment_labels"],
                         {"f3": "positive", "f1": "negative"})

    def test_multi_document_disagreement(self):
        finding = app.run_pipeline(self.data)["deep"]["findings"][0]
        self.assertTrue(finding["disagreement"])
        self.assertEqual(finding["document_ids"], ["doc1", "doc2"])
        self.assertEqual(finding["document_counts_by_position"],
                         {"support": 1, "oppose": 1, "neutral": 0})
        self.assertIn("require reconciliation", finding["unresolved_questions"][0])
        self.assertEqual(finding["questions"][0]["status"], "needs_human_review")

    def test_citations_are_exact(self):
        finding = app.run_pipeline(self.data)["deep"]["findings"][0]
        docs = {doc["id"]: doc for doc in self.data["documents"]}
        for evidence in finding["evidence"]:
            original = docs[evidence["document_id"]]["claims"][evidence["claim_index"]]
            self.assertEqual(evidence["quote"], original["evidence"])

    def test_no_evidence_and_unused_documents(self):
        self.data["documents"] = []
        finding = app.run_pipeline(self.data)["deep"]["findings"][0]
        self.assertEqual(finding["agreement"], "no_evidence")
        self.assertIn("No supplied evidence", finding["unresolved_questions"][0])
        self.assertEqual(app.run_pipeline(json.loads(
            (ROOT / "example_input.json").read_text(encoding="utf-8")
        ))["deep"]["unused_document_ids"], ["doc3"])

    def test_repeated_claims_do_not_inflate_document_count(self):
        self.data["documents"][0]["claims"].append(
            copy.deepcopy(self.data["documents"][0]["claims"][0]))
        finding = app.run_pipeline(self.data)["deep"]["findings"][0]
        self.assertEqual(finding["document_counts_by_position"]["support"], 1)
        self.assertEqual(len(finding["evidence"]), 3)

    def test_single_document_caveat(self):
        finding = app.run_pipeline(self.data)["deep"]["findings"][1]
        self.assertFalse(finding["disagreement"])
        self.assertIn("independent corroboration", finding["unresolved_questions"][0])

    def test_empty_pipeline(self):
        for key in ("steps", "feedback", "documents", "research_questions"):
            self.data[key] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["guided"]["progress"]["fraction"], 1)
        self.assertEqual(result["deep"]["findings"], [])

    def test_duplicate_ids_and_invalid_values(self):
        mutations = [
            lambda d: d["steps"].append(copy.deepcopy(d["steps"][0])),
            lambda d: d["feedback"][0].update(severity="urgent"),
            lambda d: d["steps"][0].update(completed=1),
            lambda d: d.update(schema_version=True),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(extra="unknown"),
            lambda d: d["feedback"][0].update(text=" "),
            lambda d: d["documents"][0]["claims"][0].update(position="maybe"),
            lambda d: d["research_questions"][0].update(topic="unknown"),
            lambda d: d.update(steps=None),
            lambda d: d["feedback"][0].update(severity=[]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_handoff_tampering_rejected(self):
        state = app.guided_stage(self.data)
        state["guided"]["completed_step_ids"].append("checkout")
        with self.assertRaisesRegex(app.ValidationError, "handoff.guided"):
            app.sentiment_stage(state)
        state = app.sentiment_stage(app.guided_stage(self.data))
        state["sentiment"]["issues"][0]["priority_score"] = 999
        with self.assertRaisesRegex(app.ValidationError, "handoff.sentiment"):
            app.deep_stage(state)

    def test_stage_order_rejected(self):
        with self.assertRaisesRegex(app.ValidationError, "expected sentiment"):
            app.deep_stage(app.guided_stage(self.data))

    def test_deterministic_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(before, self.data)

    def invoke_cli(self, *args):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(process.stderr, "")
        return process.returncode, json.loads(process.stdout)

    def test_cli_success(self):
        code, result = self.invoke_cli(str(ROOT / "example_input.json"))
        self.assertEqual(code, 0)
        self.assertEqual(result, app.run_pipeline(self.data))

    def test_cli_missing_file(self):
        code, result = self.invoke_cli(str(ROOT / "nonexistent.json"))
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "error")

    def test_cli_usage(self):
        for args in ((), ("one", "two")):
            code, result = self.invoke_cli(*args)
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "error")

    def test_cli_malformed_and_invalid_json_without_scratch_files(self):
        for raw in ("{", "[]", '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(raw=raw):
                with patch.object(Path, "open", return_value=io.StringIO(raw)):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_utf8(self):
        with patch.object(Path, "open", side_effect=UnicodeError("invalid UTF-8")):
            output = io.StringIO()
            with redirect_stdout(output):
                code = app.main(["synthetic.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
