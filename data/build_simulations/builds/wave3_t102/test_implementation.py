import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_pipeline(self):
        return app.run_pipeline(self.source)

    def test_full_pipeline(self):
        result = self.run_pipeline()
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["id"] for s in result["journey"]["steps"]], ["gather", "contact"])

    def test_multi_document_agreement(self):
        findings = {f["topic"]: f for f in self.run_pipeline()["deep"]["findings"]}
        self.assertEqual(findings["return_days"]["document_count"], 2)
        self.assertEqual(findings["return_days"]["value"], "30")

    def test_disagreement_preserved(self):
        finding = next(f for f in self.run_pipeline()["deep"]["findings"]
                       if f["topic"] == "shipping_days")
        self.assertEqual(finding["alternatives"], ["3", "5"])
        self.assertEqual(finding["status"], "disputed")
        self.assertIsNone(finding["value"])

    def test_unresolved_questions(self):
        questions = self.run_pipeline()["deep"]["unresolved_questions"]
        self.assertEqual(len(questions), 3)
        self.assertTrue(any("refund_days" in q for q in questions))

    def test_document_spans(self):
        result = self.run_pipeline()
        documents = {d["id"]: d["text"] for d in self.source["documents"]}
        for finding in result["deep"]["findings"]:
            for evidence in finding["evidence"]:
                self.assertEqual(documents[evidence["document_id"]][evidence["start"]:evidence["end"]],
                                 evidence["quote"])

    def test_no_feasible_two_steps(self):
        self.source["completed_actions"] = ["gather"]
        result = self.run_pipeline()
        self.assertEqual(result["journey"]["status"], "blocked")
        self.assertEqual(result["journey"]["steps"], [])
        self.assertIn("No feasible", result["support"]["answer"])

    def test_prerequisites_are_satisfied_in_order(self):
        done = set(self.source["completed_actions"])
        for step in self.run_pipeline()["journey"]["steps"]:
            self.assertLessEqual(set(step["prerequisites"]), done)
            done.add(step["id"])

    def test_disputed_topic_blocks_action(self):
        journey = self.run_pipeline()["journey"]
        blocked = next(a for a in journey["blocked_actions"] if a["action_id"] == "shipping")
        self.assertEqual(blocked["unresolved_topics"], ["shipping_days"])

    def test_offline_help_is_honest(self):
        support = self.run_pipeline()["support"]
        self.assertIn("team is offline", support["answer"])
        self.assertIn("no ticket has been created", support["answer"])
        self.assertTrue(support["escalation_needed"])
        self.assertIn("30", support["answer"])

    def test_online_help(self):
        self.source["request"]["team_online"] = True
        self.assertIn("team is online", self.run_pipeline()["support"]["answer"])

    def test_cross_stage_propagation(self):
        result = self.run_pipeline()
        self.assertEqual(result["journey"]["research"], result["deep"])
        self.assertEqual(result["support"]["journey"], result["journey"])
        self.assertNotIn("contact", self.source["request"]["topics"])
        self.assertEqual(result["extract"]["fields"]["support_address"]["value"],
                         "help@example.invalid")

    def test_extraction_types_and_spans(self):
        result = self.run_pipeline()
        fields = result["extract"]["fields"]
        self.assertIs(type(fields["return_window"]["value"]), int)
        self.assertIs(fields["needs_receipt"]["value"], True)
        for field in fields.values():
            if field["source_span"]:
                span = field["source_span"]
                self.assertEqual(result["support"]["answer"][span["start"]:span["end"]], span["text"])
                self.assertTrue(field["citations"])

    def test_missing_fields(self):
        extraction = self.run_pipeline()["extract"]
        self.assertEqual(extraction["status"], "incomplete")
        self.assertEqual({f["name"] for f in extraction["missing_fields"]},
                         {"shipping_estimate", "refund_estimate"})

    def test_optional_missing_is_complete(self):
        for field in self.source["extraction_schema"]:
            field["required"] = False
        self.assertEqual(self.run_pipeline()["extract"]["status"], "complete")

    def test_conversion_failure_reported(self):
        self.source["extraction_schema"][2]["type"] = "integer"
        self.assertIsNone(self.run_pipeline()["extract"]["fields"]["support_address"]["value"])

    def test_enum_failure_reported(self):
        self.source["extraction_schema"][0]["enum"] = [14]
        self.assertIsNone(self.run_pipeline()["extract"]["fields"]["return_window"]["value"])

    def test_empty_documents_rejected(self):
        self.source["documents"] = []
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_fabricated_quote_rejected(self):
        self.source["documents"][0]["claims"][0]["quote"] = "30 fabricated"
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_duplicate_documents_rejected(self):
        self.source["documents"][1]["id"] = self.source["documents"][0]["id"]
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_cycle_rejected(self):
        self.source["actions"][0]["prerequisites"] = ["contact"]
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_unknown_prerequisite_rejected(self):
        self.source["actions"][0]["prerequisites"] = ["unknown"]
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_repeated_quote_requires_span(self):
        doc = self.source["documents"][0]
        doc["text"] += " Return window days: 30."
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()
        doc["claims"][0]["start"] = 0
        self.assertEqual(self.run_pipeline()["status"], "ok")

    def test_invalid_span_rejected(self):
        self.source["documents"][0]["claims"][0]["start"] = True
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_null_span_uses_unique_quote(self):
        self.source["documents"][0]["claims"][0]["start"] = None
        self.assertEqual(self.run_pipeline()["status"], "ok")

    def test_unicode_offsets(self):
        self.source["documents"][0]["text"] = "\u2603 \U0001f600 " + self.source["documents"][0]["text"]
        self.test_document_spans()
        self.test_extraction_types_and_spans()

    def test_handoff_boolean_is_not_numeric_span(self):
        result = self.run_pipeline()
        changed = copy.deepcopy(result["deep"])
        changed["findings"][2]["evidence"][0]["start"] = False
        with self.assertRaises(app.ValidationError):
            app.validate(changed, "deep", (self.source,))

    def test_handoff_tampering_rejected_at_each_stage(self):
        result = self.run_pipeline()
        contexts = {"deep": (self.source,), "journey": (self.source, result["deep"]),
                    "support": (self.source, result["journey"]),
                    "extract": (self.source, result["support"])}
        for stage, context in contexts.items():
            with self.subTest(stage=stage):
                altered = copy.deepcopy(result[stage])
                altered["fabricated"] = True
                with self.assertRaises(app.ValidationError):
                    app.validate(altered, stage, context)

    def test_empty_evidence_safe(self):
        for doc in self.source["documents"]:
            doc["claims"] = []
        result = self.run_pipeline()
        self.assertEqual(result["deep"]["findings"], [])
        self.assertEqual(result["support"]["facts"], [])
        self.assertEqual(len(result["extract"]["missing_fields"]), 5)

    def test_no_actions_safe(self):
        self.source["actions"] = []
        self.assertEqual(self.run_pipeline()["journey"]["status"], "blocked")

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.source)
        self.assertEqual(self.run_pipeline(), self.run_pipeline())
        self.assertEqual(self.source, original)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        completed = self.cli("example_input.json")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")

    def test_cli_file_error(self):
        completed = self.cli("nonexistent.json")
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_usage_error(self):
        completed = self.cli()
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_invalid_json(self):
        completed = self.cli("implementation.py")
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_invalid_schema(self):
        completed = self.cli("build_manifest.json")
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_invalid_schema_variants(self):
        for key, value in [("schema_version", "2"), ("synthetic", False),
                           ("request", []), ("actions", None), ("extraction_schema", [])]:
            with self.subTest(key=key):
                source = copy.deepcopy(self.source)
                source[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(source)


if __name__ == "__main__":
    unittest.main()
