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
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_case(self):
        return app.run_pipeline(self.raw)

    def test_grounded_faq(self):
        faq = self.run_case()["data"]["faq"]
        self.assertEqual(faq["status"], "answered")
        self.assertEqual(faq["facts"]["income_limit"], 2400)
        self.assertEqual(len(faq["evidence_ids"]), 4)
        self.assertIn("2400", faq["answer"])

    def test_unknown_question_abstains(self):
        self.raw["service_request"]["question"] = "Can I appeal a parking ticket?"
        data = self.run_case()["data"]
        self.assertEqual(data["faq"]["status"], "abstained")
        self.assertEqual(data["guided"]["status"], "blocked")
        self.assertIn("Can a case worker answer the original service question?",
                      data["deep"]["unresolved_questions"])

    def test_documents_question(self):
        self.raw["service_request"]["question"] = "What documents do I need?"
        faq = self.run_case()["data"]["faq"]
        self.assertEqual(faq["facts"]["required_documents"], ["identity", "income", "residency"])

    def test_missing_documents_prerequisites(self):
        guide = self.run_case()["data"]["guided"]
        self.assertEqual(guide["missing_documents"], ["income"])
        self.assertEqual([s["status"] for s in guide["steps"]], ["complete", "pending", "blocked"])
        self.assertEqual(guide["completed_steps"], 1)

    def test_ready_for_human_review_not_approved(self):
        self.raw["benefits_application"]["documents"].append("income")
        guide = self.run_case()["data"]["guided"]
        self.assertEqual(guide["status"], "ready_for_review")
        self.assertEqual(guide["completed_steps"], 2)
        self.assertEqual(guide["decision"], "not_made")
        self.assertEqual(guide["steps"][-1]["status"], "pending")

    def test_rule_boundary_is_inclusive(self):
        self.raw["benefits_application"].update(monthly_income=2400, residency_months=6)
        checks = self.run_case()["data"]["guided"]["checks"]
        self.assertTrue(all(c["result"] == "meets_stated_rule" for c in checks))

    def test_rule_failure_is_not_denial(self):
        self.raw["benefits_application"]["monthly_income"] = 2401
        data = self.run_case()["data"]
        self.assertEqual(data["guided"]["checks"][0]["result"], "needs_case_worker_review")
        self.assertEqual(data["guided"]["decision"], "not_made")
        self.assertIn("Are exceptions or other benefits available for this application?",
                      data["deep"]["unresolved_questions"])

    def test_multi_document_synthesis(self):
        deep = self.run_case()["data"]["deep"]
        self.assertEqual(deep["documents_used"], 2)
        self.assertEqual(len(deep["agreements"]), 3)
        self.assertTrue(all(len(x["evidence_ids"]) == 2 for x in deep["agreements"]))
        self.assertEqual(deep["disagreements"], [])

    def test_disagreement_survives_handoffs(self):
        policy = self.raw["policy_documents"][1]
        policy["text"] = policy["text"].replace("2400", "2100")
        data = self.run_case()["data"]
        self.assertEqual(data["faq"]["status"], "abstained")
        self.assertEqual(data["guided"]["status"], "blocked")
        conflict = data["deep"]["disagreements"][0]
        self.assertEqual(conflict["rule"], "income_limit")
        self.assertEqual({v["value"] for v in conflict["alternatives"]}, {2400, 2100})

    def test_missing_evidence(self):
        self.raw["policy_documents"] = []
        data = self.run_case()["data"]
        self.assertEqual(data["faq"]["status"], "abstained")
        self.assertEqual(data["deep"]["documents_used"], 0)
        self.assertEqual(len(data["deep"]["agreements"]), 0)
        self.assertTrue(any("Monthly" in q or "monthly" in q
                            for q in data["deep"]["unresolved_questions"]))

    def test_other_program_not_used(self):
        for policy in self.raw["policy_documents"]:
            policy["program"] = "synthetic-other"
        self.assertEqual(self.run_case()["evidence"], [])

    def test_cross_stage_tracking(self):
        faq = app.faq_stage(self.raw)
        guided = app.guided_stage(faq)
        deep = app.deep_stage(guided)
        self.assertEqual(deep["data"]["faq"], faq["data"]["faq"])
        self.assertEqual(deep["data"]["guided"], guided["data"]["guided"])
        self.assertEqual(deep["data"]["deep"]["source_guided_status"], "in_progress")
        self.assertEqual(deep["context"]["case_number"], "SYN-CASE-0720")
        self.assertNotIn("guided", faq["data"])

    def test_tampered_faq_rejected(self):
        output = app.faq_stage(self.raw)
        output["data"]["faq"]["answer"] = "You are approved."
        with self.assertRaises(app.ValidationError):
            app.guided_stage(output)

    def test_tampered_guided_rejected(self):
        output = app.guided_stage(app.faq_stage(self.raw))
        output["data"]["guided"]["completed_steps"] = 3
        with self.assertRaises(app.ValidationError):
            app.deep_stage(output)

    def test_cannot_skip_stage(self):
        with self.assertRaises(app.ValidationError):
            app.deep_stage(app.faq_stage(self.raw))

    def test_invalid_inputs(self):
        invalids = []
        for key, value in (("monthly_income", -1), ("monthly_income", True),
                           ("residency_months", 1.5), ("documents", ["identity", "identity"]),
                           ("documents", ["passport"]), ("documents", "income")):
            raw = copy.deepcopy(self.raw)
            raw["benefits_application"][key] = value
            invalids.append(raw)
        invalids.extend([None, [], {}, {"schema_version": 1}])
        for raw in invalids:
            with self.subTest(raw=raw), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)

    def test_case_mismatch(self):
        self.raw["benefits_application"]["case_number"] = "SYN-CASE-9999"
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_unknown_field_rejected(self):
        self.raw["service_request"]["citizen"]["social_security_number"] = "not-stored"
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_real_data_not_accepted(self):
        self.raw["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_identity_removed(self):
        output = json.dumps(self.run_case())
        citizen = self.raw["service_request"]["citizen"]
        self.assertNotIn(citizen["display_name"], output)
        self.assertNotIn(citizen["address"], output)
        self.assertNotIn("display_name", output)

    def test_private_text_rejected(self):
        for private in ("citizen@example.invalid", "123-45-6789",
                        self.raw["service_request"]["citizen"]["display_name"]):
            with self.subTest(private=private):
                raw = copy.deepcopy(self.raw)
                raw["service_request"]["question"] += " " + private
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(raw)

    def test_malformed_and_duplicate_policy_rules(self):
        for suffix in ("\nincome_limit: 12", "\nresidency_months: NaN"):
            raw = copy.deepcopy(self.raw)
            raw["policy_documents"][0]["text"] += suffix
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)

    def test_duplicate_policy_ids(self):
        self.raw["policy_documents"][1]["id"] = self.raw["policy_documents"][0]["id"]
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_injected_callable_is_grounded_and_private(self):
        captured = []

        def hook(request):
            captured.append(copy.deepcopy(request))
            return {"answer": request["expected_answer"],
                    "evidence_ids": request["expected_evidence_ids"]}

        output = app.run_pipeline(self.raw, hook)
        self.assertEqual(output, self.run_case())
        self.assertNotIn("Synthetic Rowan Example", json.dumps(captured))
        self.assertNotIn("monthly_income", json.dumps(captured))

    def test_ungrounded_callable_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.raw, lambda _: {"answer": "Approved", "evidence_ids": []})

    def test_callable_exception_sanitized(self):
        def hook(_):
            raise RuntimeError("Private information")

        with self.assertRaisesRegex(app.ValidationError, "^The optional answer hook failed.$"):
            app.run_pipeline(self.raw, hook)

    def test_evidence_quotes_and_readable_explanations(self):
        output = self.run_case()
        texts = {p["id"]: p["text"] for p in self.raw["policy_documents"]}
        for evidence in output["evidence"]:
            self.assertIn(evidence["quote"], texts[evidence["policy_id"]])
        self.assertIn("case worker", output["data"]["deep"]["summary"])
        self.assertTrue(output["explanation"])

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.raw)
        self.assertEqual(self.run_case(), self.run_case())
        self.assertEqual(self.raw, original)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["stage"], "deep")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for arguments in ([], ["absent-input.json"]):
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_schema_without_extra_files(self):
        for content in ("{", "null", '{"a":1,"a":2}', '{"synthetic":NaN}',
                        json.dumps({**self.raw, "synthetic": False})):
            with self.subTest(content=content), patch.object(Path, "read_text", return_value=content):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = app.main([str(ROOT / "example_input.json")])
                self.assertEqual(status, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
