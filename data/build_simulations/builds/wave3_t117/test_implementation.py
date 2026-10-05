"""All examples in this suite are synthetic. No test creates files."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def result(self):
        return impl.run_pipeline(self.data)["results"]

    def invalid(self):
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(self.data)

    def test_example_pipeline(self):
        output = impl.run_pipeline(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(list(output["results"]), list(impl.STAGES))
        self.assertEqual(output["results"]["review"]["summary"], {"met": 1, "gaps": 1})

    def test_research_evidence_offsets(self):
        finding = self.result()["research"]["findings"][0]
        self.assertEqual(finding["status"], "answered")
        evidence = finding["evidence"][0]
        source = self.data["research"]["sources"][0]
        self.assertEqual(source["text"][evidence["start"]:evidence["end"]], evidence["quote"])

    def test_research_conflict_blocks_and_propagates(self):
        source = self.data["research"]["sources"][0]
        source["text"] += " No owner email is required."
        source["evidence"].append({"question_id": "q-access", "quote": "No owner email is required.",
                                   "stance": "opposes"})
        result = self.result()
        self.assertEqual(result["research"]["findings"][0]["status"], "conflicted")
        self.assertEqual(result["guided"]["progress"]["completed"], 0)
        self.assertEqual(result["feedback"]["counts"]["deferred"], 4)
        self.assertEqual(result["review"]["summary"], {"met": 0, "gaps": 2})
        self.assertTrue(any(g["reason"] == "conflicted"
                            for g in result["review"]["requirements"][0]["gaps"]))

    def test_context_is_not_an_answer(self):
        self.data["research"]["sources"][0]["evidence"][0]["stance"] = "context"
        self.assertEqual(self.result()["research"]["findings"][0]["status"], "unanswered")

    def test_opposing_evidence_is_an_answer_not_an_approval(self):
        self.data["research"]["sources"][0]["evidence"][0]["stance"] = "opposes"
        self.assertEqual(self.result()["research"]["findings"][0]["status"], "answered")

    def test_no_sources_propagates(self):
        self.data["research"]["sources"] = []
        result = self.result()
        self.assertEqual(result["guided"]["progress"]["percent"], 0.0)
        self.assertEqual(result["feedback"]["counts"]["eligible"], 0)

    def test_fabricated_quote_rejected(self):
        self.data["research"]["sources"][0]["evidence"][0]["quote"] = "fabricated"
        self.invalid()

    def test_unknown_question_reference(self):
        self.data["research"]["sources"][0]["evidence"][0]["question_id"] = "unknown"
        self.invalid()

    def test_duplicate_ids(self):
        self.data["research"]["questions"].append(copy.deepcopy(self.data["research"]["questions"][0]))
        self.invalid()

    def test_out_of_order_dependencies(self):
        self.data["guided"]["steps"].reverse()
        result = self.result()["guided"]
        self.assertEqual(result["execution_order"], ["account", "export"])
        self.assertEqual(result["progress"]["percent"], 100.0)

    def test_missing_field_blocks_dependents(self):
        self.data["guided"]["fields"].pop("owner_email")
        result = self.result()["guided"]
        self.assertEqual(result["steps"][0]["reasons"], [{"kind": "field", "id": "owner_email"}])
        self.assertIn({"kind": "dependency", "id": "account"}, result["steps"][1]["reasons"])

    def test_blank_field_is_missing(self):
        self.data["guided"]["fields"]["owner_email"] = " \n "
        self.assertEqual(self.result()["guided"]["progress"]["completed"], 0)

    def test_field_type(self):
        self.data["guided"]["fields"]["owner_email"] = False
        self.invalid()

    def test_cycle(self):
        self.data["guided"]["steps"][0]["depends_on"] = ["export"]
        self.invalid()

    def test_unknown_dependency(self):
        self.data["guided"]["steps"][0]["depends_on"] = ["missing"]
        self.invalid()

    def test_feedback_dedup_support_and_themes(self):
        feedback = self.result()["feedback"]
        self.assertEqual(feedback["counts"], {"received": 4, "eligible": 4, "unique": 3,
                                             "duplicates": 1, "deferred": 0})
        self.assertEqual([s["item_id"] for s in feedback["groups"][0]["support"]], ["f1", "f2"])
        for group in feedback["groups"]:
            for support in group["support"]:
                original = next(x["text"] for x in self.data["feedback"]["items"]
                                if x["id"] == support["item_id"])
                self.assertEqual(original[support["start"]:support["end"]], support["quote"])
        self.assertEqual(feedback["themes"][-1]["group_ids"], ["f4"])

    def test_word_boundaries(self):
        self.data["feedback"]["items"][2]["text"] = "A slowdown was noticed."
        self.assertEqual(self.result()["feedback"]["themes"][1]["unique_count"], 0)

    def test_multi_theme_membership(self):
        self.data["feedback"]["items"][0]["text"] = "Clear and fast."
        feedback = self.result()["feedback"]
        self.assertIn("f1", feedback["themes"][0]["group_ids"])
        self.assertIn("f1", feedback["themes"][1]["group_ids"])

    def test_blocked_duplicate_does_not_contribute(self):
        self.data["guided"]["fields"]["export_acknowledged"] = ""
        self.data["feedback"]["items"][2]["text"] = self.data["feedback"]["items"][0]["text"]
        feedback = self.result()["feedback"]
        self.assertEqual(feedback["counts"]["eligible"], 2)
        self.assertEqual(feedback["counts"]["deferred"], 2)
        self.assertNotIn("f3", [s["item_id"] for g in feedback["groups"] for s in g["support"]])

    def test_empty_feedback(self):
        self.data["feedback"]["items"] = []
        result = self.result()
        self.assertEqual(result["feedback"]["counts"]["unique"], 0)
        self.assertEqual(result["review"]["summary"]["gaps"], 2)

    def test_empty_steps(self):
        self.data["guided"]["steps"] = []
        self.data["feedback"]["items"] = []
        for requirement in self.data["review"]["requirements"]:
            requirement["step_ids"] = []
        self.assertEqual(self.result()["guided"]["progress"],
                         {"completed": 0, "total": 0, "percent": 100.0})

    def test_review_literal_term_case_and_offsets(self):
        self.data["review"]["document"]["text"] = "OWNER EMAIL: synthetic."
        requirement = self.result()["review"]["requirements"][0]
        self.assertEqual(requirement["status"], "met")
        evidence = next(e for e in requirement["evidence"] if e["kind"] == "document")
        self.assertEqual(evidence["quote"], "OWNER EMAIL")
        self.assertEqual(self.data["review"]["document"]["text"][evidence["start"]:evidence["end"]],
                         evidence["quote"])

    def test_review_traces_feedback(self):
        review = self.result()["review"]
        evidence = next(e for e in review["requirements"][0]["evidence"] if e["kind"] == "feedback")
        self.assertEqual([s["item_id"] for s in evidence["support"]], ["f1", "f2"])
        self.assertIn("not certification", review["disclaimer"])

    def test_empty_document_gaps(self):
        self.data["review"]["document"]["text"] = ""
        self.assertEqual(self.result()["review"]["summary"]["gaps"], 2)

    def test_requirement_must_check_something(self):
        requirement = self.data["review"]["requirements"][0]
        for key in ("question_ids", "step_ids", "theme_ids", "required_terms"):
            requirement[key] = []
        self.invalid()

    def test_unknown_review_theme(self):
        self.data["review"]["requirements"][0]["theme_ids"] = ["missing"]
        self.invalid()

    def test_strict_schema_and_synthetic_flag(self):
        for key, value in (("unexpected", "x"), ("schema_version", True), ("synthetic", False)):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(data)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(impl.ValidationError):
            impl.feedback_stage(impl.research_stage(self.data))

    def test_tampered_handoff_rejected(self):
        state = impl.research_stage(self.data)
        state["results"]["research"]["findings"][0]["evidence"][0]["quote"] = "invented"
        with self.assertRaises(impl.ValidationError):
            impl.guided_stage(state)

    def test_stale_handoff_rejected(self):
        state = impl.guided_stage(impl.research_stage(self.data))
        state["input"]["guided"]["fields"]["owner_email"] = ""
        with self.assertRaises(impl.ValidationError):
            impl.feedback_stage(state)

    def test_determinism_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(impl.run_pipeline(self.data), impl.run_pipeline(self.data))
        self.assertEqual(self.data, original)
        research = impl.research_stage(self.data)
        saved = copy.deepcopy(research)
        impl.guided_stage(research)
        self.assertEqual(research, saved)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        result = self.cli("does-not-exist.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_usage(self):
        for args in ((), ("one", "two")):
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_and_invalid_json_without_disk_writes(self):
        for payload in ('{', '[]', '{"a": 1, "a": 2}', '{"x": NaN}', '{"schema_version": 1}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.StringIO(payload)):
                    with contextlib.redirect_stdout(output):
                        code = impl.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_unicode_read_error(self):
        output = io.StringIO()
        with patch.object(Path, "open", side_effect=UnicodeError("synthetic decoding failure")):
            with contextlib.redirect_stdout(output):
                code = impl.main(["synthetic-invalid.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
