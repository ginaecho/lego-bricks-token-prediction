"""Synthetic fixtures only; tests never write files or invoke providers."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as impl


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_request(self):
        return impl.run_pipeline(self.request)

    def test_integrated_example(self):
        result = self.run_request()
        self.assertEqual(list(result["stages"]), list(impl.STAGES))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stages"]["journey"]["status"], "complete")

    def test_sentiment_contributions_and_negation(self):
        row = self.run_request()["stages"]["sentiment"]["feedback"][0]
        self.assertEqual(row["score"], -6)
        self.assertEqual(row["priority"], 106)
        self.assertTrue(row["contributions"][-1]["negated"])
        for hit in row["contributions"]:
            a, b = hit["span"]
            self.assertEqual(self.request["feedback"][0]["text"][a:b], hit["token"])

    def test_neutral_and_case_insensitive_sentiment(self):
        self.request["feedback"][0]["text"] = "ordinary checkout."
        self.request["feedback"][1]["text"] = "NOT BAD; GREAT"
        rows = self.run_request()["stages"]["sentiment"]["feedback"]
        self.assertEqual(rows[0]["sentiment"], "neutral")
        self.assertEqual(rows[1]["score"], 3)

    def test_severity_applies_even_to_positive_feedback(self):
        self.request["feedback"][0]["text"] = "excellent"
        issues = self.run_request()["stages"]["sentiment"]["issues"]
        self.assertEqual(issues[0]["priority"], 100)
        self.assertEqual(issues[0]["highest_severity"], "critical")

    def test_multidocument_disagreement_and_exact_spans(self):
        report = self.run_request()["stages"]["deep"]["topics"][0]
        self.assertEqual(report["document_count"], 2)
        self.assertTrue(report["disagreement"])
        self.assertEqual(report["conclusion"], "disputed")
        self.assertTrue(report["unresolved_questions"])
        docs = {doc["id"]: doc["text"] for doc in self.request["documents"]}
        for item in report["evidence"]:
            a, b = item["span"]
            self.assertEqual(docs[item["document_id"]][a:b], item["quote"])

    def test_uncertainty_and_absent_evidence(self):
        reports = self.run_request()["stages"]["deep"]["topics"]
        self.assertEqual(reports[1]["conclusion"], "inconclusive")
        self.request["documents"] = []
        reports = self.run_request()["stages"]["deep"]["topics"]
        self.assertTrue(all(r["conclusion"] == "no_evidence" for r in reports))
        self.assertTrue(all(r["unresolved_questions"] for r in reports))

    def test_duplicate_claims_not_double_counted(self):
        doc = self.request["documents"][0]
        doc["claims"].append(copy.deepcopy(doc["claims"][0]))
        report = self.run_request()["stages"]["deep"]["topics"][0]
        self.assertEqual(report["stance_counts"]["supports"], 1)

    def test_one_sided_research_does_not_invent_disagreement(self):
        for stance, conclusion in (("supports", "supported"), ("opposes", "opposed")):
            with self.subTest(stance=stance):
                for doc in self.request["documents"]:
                    for claim in doc["claims"]:
                        claim["stance"] = stance
                report = self.run_request()["stages"]["deep"]["topics"][0]
                self.assertEqual(report["conclusion"], conclusion)
                self.assertFalse(report["disagreement"])
                self.assertEqual(report["unresolved_questions"], [])

    def test_two_step_prerequisites(self):
        journey = self.run_request()["stages"]["journey"]
        first, second = journey["steps"]
        self.assertEqual(first["id"], "a-checkout-audit")
        self.assertEqual(second["id"], "b-delivery-review")
        self.assertEqual(second["prerequisites_satisfied"], [first["id"]])
        self.assertEqual(journey["plan_priority"], 136)

    def test_completed_actions_not_recommended(self):
        self.request["profile"]["completed"] = ["a-checkout-audit"]
        steps = self.run_request()["stages"]["journey"]["steps"]
        self.assertEqual(len(steps), 2)
        self.assertNotIn("a-checkout-audit", [step["id"] for step in steps])

    def test_cycle_is_blocked_not_fabricated(self):
        self.request["actions"][0]["prerequisites"] = ["b-delivery-review"]
        journey = self.run_request()["stages"]["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual(journey["steps"], [])

    def test_one_remaining_action_is_partial_blocked(self):
        self.request["profile"]["completed"] = ["a-checkout-audit", "c-checkout-fix"]
        journey = self.run_request()["stages"]["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual(len(journey["steps"]), 1)

    def test_prerequisite_only_action_can_enable_relevant_action(self):
        self.request["actions"][0]["topic"] = "setup"
        journey = self.run_request()["stages"]["journey"]
        self.assertEqual(journey["status"], "complete")
        self.assertEqual(journey["steps"][0]["research_conclusion"], "prerequisite_only")
        self.assertEqual(journey["steps"][1]["id"], "c-checkout-fix")

    def test_unrelated_actions_do_not_pad_journey(self):
        self.request["actions"][1]["topic"] = "unrelated"
        self.request["actions"][2]["topic"] = "unrelated"
        journey = self.run_request()["stages"]["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual([step["id"] for step in journey["steps"]], ["a-checkout-audit"])

    def test_extraction_values_spans_and_optional_missing(self):
        extraction = self.run_request()["stages"]["extract"]
        owner, eta, approval = extraction["fields"]
        self.assertEqual(owner["value"], "Synthetic Support")
        self.assertEqual(eta["value"], 2)
        self.assertEqual(owner["match_count"], 2)
        for item in (owner, eta):
            source = item["source"]
            start, end = source["span"]
            self.assertEqual(source["text"][start:end], str(item["value"]))
        self.assertIsNone(approval["value"])
        self.assertEqual(extraction["missing_fields"], ["approval"])
        self.assertTrue(extraction["complete"])

    def test_required_missing_report(self):
        self.request["extraction_schema"][2]["required"] = True
        extraction = self.run_request()["stages"]["extract"]
        self.assertFalse(extraction["complete"])
        self.assertEqual(extraction["required_missing"], ["approval"])

    def test_invalid_integer_reports_missing_with_provenance(self):
        self.request["actions"][0]["description"] = "Owner: Test\nETA days: tomorrow"
        extraction = self.run_request()["stages"]["extract"]
        eta = extraction["fields"][1]
        self.assertEqual(eta["missing_reason"], "invalid_integer")
        self.assertIsNotNone(eta["source"])
        self.assertIsNone(eta["value"])
        self.assertFalse(extraction["complete"])

    def test_unicode_whitespace_literal_label_spans(self):
        self.request["actions"][0]["description"] = "  Ow.ner:   Zoë 🧱  \r\nETA days: 2"
        self.request["extraction_schema"][0]["label"] = "Ow.ner"
        field = self.run_request()["stages"]["extract"]["fields"][0]
        self.assertEqual(field["value"], "Zoë 🧱")
        start, end = field["source"]["span"]
        self.assertEqual(field["source"]["text"][start:end], "Zoë 🧱")

    def test_title_source_and_blank_occurrences(self):
        self.request["actions"][0]["title"] = "Owner:   \nOwner: Synthetic Reviewer"
        self.request["extraction_schema"][0]["source"] = "title"
        field = self.run_request()["stages"]["extract"]["fields"][0]
        self.assertEqual(field["value"], "Synthetic Reviewer")
        self.assertEqual(field["match_count"], 1)
        self.assertEqual(field["source"]["field"], "title")

    def test_cross_stage_propagation(self):
        result = self.run_request()["stages"]
        issue = result["sentiment"]["issues"][0]
        report = result["deep"]["topics"][0]
        step = result["journey"]["steps"][0]
        field = result["extract"]["fields"][0]
        self.assertEqual(issue["priority"], report["priority"])
        self.assertEqual(report["priority"], step["priority"])
        self.assertEqual(report["unresolved_questions"], step["unresolved_questions"])
        self.assertEqual(field["source"]["action_id"], step["id"])
        self.request["profile"]["completed"] = ["a-checkout-audit"]
        changed = self.run_request()["stages"]["extract"]["fields"][0]
        self.assertEqual(changed["value"], "Synthetic Operations")

    def test_changed_sentiment_reaches_journey_priority(self):
        self.request["feedback"][1]["severity"] = "critical"
        self.request["feedback"][1]["text"] = "terrible broken failed"
        stages = self.run_request()["stages"]
        self.assertEqual(stages["deep"]["topics"][0]["topic"], "delivery")
        delivery = next(s for s in stages["journey"]["steps"] if s["topic"] == "delivery")
        self.assertEqual(delivery["priority"], 107)

    def test_empty_collections_have_explicit_missing_and_blocked(self):
        for name in ("feedback", "documents", "actions"):
            self.request[name] = []
        stages = self.run_request()["stages"]
        self.assertEqual(stages["deep"]["topics"], [])
        self.assertEqual(stages["journey"]["status"], "blocked")
        self.assertEqual(stages["extract"]["required_missing"], ["owner", "eta_days"])

    def test_invalid_inputs(self):
        mutations = (
            lambda r: r.update(schema_version=True),
            lambda r: r.update(synthetic=False),
            lambda r: r.update(extra=1),
            lambda r: r["feedback"][0].update(severity="urgent"),
            lambda r: r["feedback"].append(copy.deepcopy(r["feedback"][0])),
            lambda r: r["documents"][0]["claims"][0].update(quote="invented quote"),
            lambda r: r["actions"][0].update(prerequisites=["absent"]),
            lambda r: r["actions"][0].update(prerequisites=["a-checkout-audit"]),
            lambda r: r["profile"].update(completed=["absent"]),
            lambda r: r["extraction_schema"][0].update(required="yes"),
            lambda r: r["extraction_schema"][0].update(type="date"),
            lambda r: r["extraction_schema"][0].update(label="bad\nlabel"),
            lambda r: r["extraction_schema"].append(copy.deepcopy(r["extraction_schema"][0])),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                request = copy.deepcopy(self.request)
                mutation(request)
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(request)

    def test_stage_boundary_rejects_skips_and_tampering(self):
        empty = {"schema_version": 1, "status": "ok", "input": self.request, "stages": {}}
        with self.assertRaises(impl.ValidationError):
            impl.advance(empty, "deep")
        scored = impl.advance(empty, "sentiment")
        scored["stages"]["sentiment"]["issues"][0]["priority"] = -1
        with self.assertRaises(impl.ValidationError):
            impl.advance(scored, "deep")

    def test_each_stage_rejects_semantic_tampering(self):
        result = self.run_request()
        for stage in impl.STAGES:
            with self.subTest(stage=stage):
                changed = copy.deepcopy(result)
                changed["stages"][stage]["unexpected"] = True
                with self.assertRaises(impl.ValidationError):
                    impl.validate_envelope(changed)

    def test_determinism_and_no_input_mutation(self):
        before = copy.deepcopy(self.request)
        first = self.run_request()
        self.assertEqual(first, self.run_request())
        self.assertEqual(before, self.request)
        self.assertEqual(impl.validate_envelope(json.loads(json.dumps(first))), first)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"), str(HERE / "example_input.json")],
            cwd=HERE, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file_and_bad_arguments(self):
        for args in ([], [str(HERE / "nonexistent-input.json")], ["one", "two"]):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(HERE / "implementation.py")] + args,
                    cwd=HERE, capture_output=True, text=True, check=False,
                )
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_duplicate_keys_and_nonfinite(self):
        for payload in ("{broken", '{"x":1,"x":2}', '{"x":NaN}', "null", "[]"):
            with self.subTest(payload=payload):
                out = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), redirect_stdout(out):
                    code = impl.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")

    def test_cli_encoding_and_permission_errors(self):
        for error in (PermissionError("denied"), UnicodeError("bad encoding")):
            with self.subTest(error=error):
                out = io.StringIO()
                with patch("builtins.open", side_effect=error), redirect_stdout(out):
                    code = impl.main(["synthetic-unreadable.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
