import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def stages(self):
        return [p["data"] for p in app.run(self.value)["stages"]]

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.value)

    def test_integrated_stage_order(self):
        result = app.run(self.value)
        self.assertEqual([s["stage"] for s in result["stages"]], list(app.STAGES))
        self.assertEqual([s["input_stage"] for s in result["stages"]],
                         ["input", "feedback", "faq", "guided"])

    def test_dedup_retains_all_ids(self):
        group = self.stages()[0]["groups"][0]
        self.assertEqual(group["feedback_ids"], ["f1", "f2"])
        self.assertEqual(group["evidence"]["excerpt"], self.value["feedback"][0]["text"])

    def test_theme_support_and_unmatched(self):
        feedback = self.stages()[0]
        self.assertEqual(feedback["themes"][0]["feedback_ids"], ["f1", "f2"])
        self.assertEqual(feedback["unmatched_feedback_ids"], ["f5"])

    def test_grounded_faq(self):
        answer = self.stages()[1]["answers"][0]
        self.assertEqual(answer["answer"], self.value["knowledge"][0]["text"])
        self.assertEqual(answer["answer"], answer["citations"][0]["excerpt"])

    def test_abstention_propagates(self):
        _, faq, guided, research = self.stages()
        self.assertEqual(faq["answers"][2]["status"], "abstained")
        self.assertIsNone(faq["answers"][2]["answer"])
        self.assertEqual(guided["steps"][2]["status"], "needs_answer")
        self.assertNotIn("future-step", [r["id"] for r in research["findings"]])

    def test_prerequisites_and_progress(self):
        guided = self.stages()[2]
        self.assertEqual([s["status"] for s in guided["steps"]],
                         ["completed", "ready", "needs_answer"])
        self.assertEqual(guided["progress"], {"completed": 1, "total": 3})

    def test_blocked_steps_not_researched(self):
        self.value["completed_steps"] = []
        _, _, guided, research = self.stages()
        self.assertEqual(guided["steps"][1]["status"], "blocked")
        self.assertEqual([r["id"] for r in research["findings"]], ["verify"])

    def test_invalid_completed_prerequisite(self):
        self.value["completed_steps"] = ["sync"]
        self.invalid()

    def test_unsupported_completion_rejected(self):
        self.value["completed_steps"].append("future-step")
        self.invalid()

    def test_exact_research_citation(self):
        sources = {s["id"]: s["text"] for s in self.value["research_sources"]}
        for row in self.stages()[3]["findings"]:
            c = row["citations"][0]
            self.assertEqual(row["finding"], sources[c["source_id"]][c["start"]:c["end"]])

    def test_end_to_end_evidence_lineage(self):
        feedback, faq, guided, research = self.stages()
        self.assertEqual(feedback["themes"][0]["feedback_ids"], research["findings"][0]["feedback_ids"])
        self.assertEqual(faq["answers"][0]["citations"], research["findings"][0]["faq_citations"])
        self.assertEqual(guided["steps"][0]["faq_citations"], research["findings"][0]["faq_citations"])

    def test_research_abstains_on_empty_sources(self):
        self.value["research_sources"] = []
        for row in self.stages()[3]["findings"]:
            self.assertEqual(row["status"], "abstained")
            self.assertEqual(row["citations"], [])
            self.assertIsNone(row["finding"])

    def test_empty_feedback(self):
        self.value["feedback"] = []
        self.value["completed_steps"] = []
        result = self.stages()
        self.assertEqual(result[1]["answers"], [])
        self.assertEqual(result[2]["progress"], {"completed": 0, "total": 0})
        self.assertEqual(result[3]["findings"], [])

    def test_empty_knowledge(self):
        self.value["knowledge"] = []
        self.value["completed_steps"] = []
        self.assertTrue(all(a["status"] == "abstained" for a in self.stages()[1]["answers"]))

    def test_cycle(self):
        self.value["steps"][0]["requires"] = ["sync"]
        self.invalid()

    def test_unknown_prerequisite(self):
        self.value["steps"][0]["requires"] = ["missing"]
        self.invalid()

    def test_duplicate_id(self):
        self.value["feedback"].append(copy.deepcopy(self.value["feedback"][0]))
        self.invalid()

    def test_unknown_field(self):
        self.value["unexpected"] = 42
        self.invalid()

    def test_wrong_type(self):
        self.value["feedback"] = "not a list"
        self.invalid()

    def test_non_synthetic_input(self):
        self.value["synthetic"] = False
        self.invalid()

    def test_blank_text(self):
        self.value["feedback"][0]["text"] = " "
        self.invalid()

    def test_unknown_topic(self):
        self.value["steps"][0]["topic"] = "missing"
        self.invalid()

    def test_unicode_offsets(self):
        source = {"id": "unicode", "text": "  Café verification works.  Next sentence!"}
        c = app.retrieve("verification", [source])
        app.validate_citation(c, {"unicode": source})
        self.assertEqual(c["start"], 2)
        self.assertEqual(c["excerpt"], "Café verification works.")

    def test_deterministic_ties(self):
        sources = [{"id": "z", "text": "Verification works."},
                   {"id": "a", "text": "Verification succeeds."}]
        self.assertEqual(app.retrieve("verification", sources)["source_id"], "a")
        self.assertEqual(app.retrieve("verification", sources), app.retrieve("verification", sources[::-1]))

    def test_input_unchanged_and_repeatable(self):
        before = copy.deepcopy(self.value)
        self.assertEqual(app.run(self.value), app.run(self.value))
        self.assertEqual(self.value, before)

    def test_tampered_feedback_rejected_at_handoff(self):
        result = app.feedback_stage(self.value)
        result["data"]["themes"][0]["evidence"][0]["excerpt"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.faq_stage(self.value, result)

    def test_tampered_faq_rejected_at_handoff(self):
        result = app.run(self.value)["stages"]
        result[1]["data"]["answers"][0]["answer"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.guided_stage(self.value, result[1], result[0])

    def test_tampered_guided_rejected_at_handoff(self):
        result = app.run(self.value)["stages"]
        result[2]["data"]["steps"][1]["requires"] = []
        with self.assertRaises(app.ValidationError):
            app.normal_stage(self.value, result[2], result[1])

    def test_prerequisite_closure(self):
        self.value["feedback"] = [self.value["feedback"][2]]
        self.value["completed_steps"] = []
        guided = self.stages()[2]
        self.assertEqual([s["id"] for s in guided["steps"]], ["verify", "sync"])
        self.assertEqual([s["status"] for s in guided["steps"]], ["needs_answer", "blocked"])

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "absent.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_usage(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for content in ("{", '{"a":1,"a":2}', '{"schema_version":NaN}', "[]",
                        '{"synthetic":true}', "\ud800"):
            stream = io.StringIO()
            with mock.patch("builtins.open", mock.mock_open(read_data=content)):
                with contextlib.redirect_stdout(stream):
                    code = app.main(["fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
