"""Standard-library tests using clearly labeled synthetic fixtures."""

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
        self.value = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_value(self):
        return app.run_pipeline(self.value)

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            self.run_value()

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, text=True, capture_output=True, check=False)

    def test_example_grounded_answer(self):
        result = self.run_value()
        faq = result["stages"]["faq"]
        self.assertEqual(faq["data"]["answer"], self.value["knowledge_base"][0]["text"])
        self.assertEqual(faq["data"]["citations"][0]["article_id"], "faq_payouts")
        self.assertEqual(result["status"], "needs_attention")

    def test_no_match_abstains_and_skips(self):
        self.value["query"] = "SYNTHETIC unrelated shipping question"
        stages = self.run_value()["stages"]
        self.assertEqual(stages["faq"]["status"], "abstained")
        self.assertIsNone(stages["faq"]["data"]["answer"])
        self.assertEqual(stages["guided"]["status"], "skipped")
        self.assertEqual(stages["review"]["status"], "skipped")
        self.assertEqual(stages["review"]["data"]["checks"], [])

    def test_ambiguous_matches_abstain(self):
        duplicate = copy.deepcopy(self.value["knowledge_base"][0])
        duplicate["id"] = "alternate"
        self.value["knowledge_base"].append(duplicate)
        self.assertEqual(self.run_value()["stages"]["faq"]["data"]["reason"], "ambiguous_matches")

    def test_matching_uses_token_boundaries_and_case(self):
        self.value["query"] = "SELLER, PAYOUTS?"
        self.assertEqual(self.run_value()["stages"]["faq"]["status"], "complete")
        self.value["query"] = "reseller payouts"
        self.assertEqual(self.run_value()["stages"]["faq"]["status"], "abstained")

    def test_handoff_expands_prerequisites_and_requirements(self):
        stages = self.run_value()["stages"]
        self.assertEqual(stages["faq"]["step_ids"], ["configure_payouts"])
        self.assertEqual(stages["guided"]["step_ids"], ["verify_account", "configure_payouts"])
        self.assertEqual(stages["guided"]["requirement_ids"], ["account_record", "payout_record"])
        self.assertEqual(stages["review"]["requirement_ids"], stages["guided"]["requirement_ids"])
        self.assertEqual(stages["review"]["article_ids"], stages["faq"]["article_ids"])

    def test_guided_progress_and_ready(self):
        guided = self.run_value()["stages"]["guided"]["data"]
        self.assertEqual(guided["progress"], {"completed": 1, "total": 2, "fraction": 0.5})
        self.assertEqual(guided["next_step_ids"], ["configure_payouts"])

    def test_blocking_propagates_to_review(self):
        self.value["setup"][0]["completed"] = False
        stages = self.run_value()["stages"]
        self.assertEqual(stages["guided"]["data"]["steps"][1]["status"], "blocked")
        gap = next(g for g in stages["review"]["data"]["gaps"]
                   if g.get("step_id") == "configure_payouts")
        self.assertEqual(gap["blocked_by"], ["verify_account"])

    def test_missing_evidence_is_traceable(self):
        review = self.run_value()["stages"]["review"]["data"]
        gap = next(g for g in review["gaps"] if g["kind"] == "missing_evidence")
        self.assertEqual(gap["requirement_id"], "payout_record")
        self.assertEqual(gap["step_ids"], ["configure_payouts"])
        self.assertEqual(gap["article_ids"], ["faq_payouts"])
        check = review["checks"][0]
        self.assertEqual(check["status"], "evidence_match")
        evidence = check["evidence"][0]
        document = self.value["documents"][0]["text"]
        self.assertEqual(document[evidence["start"]:evidence["end"]], evidence["quote"])
        self.assertIn("not certification", review["disclaimer"])

    def test_insufficient_evidence(self):
        self.value["evidence"].append({
            "id": "payout_quote", "requirement_id": "payout_record",
            "document_id": "setup_notes", "quote": "Payout pending.", "start": 37,
        })
        review = self.run_value()["stages"]["review"]["data"]
        check = review["checks"][1]
        self.assertEqual(check["evidence"][0]["missing_terms"], ["weekly"])
        self.assertEqual(review["gaps"][-1]["kind"], "insufficient_evidence")

    def test_complete_pipeline(self):
        self.value["setup"][1]["completed"] = True
        self.value["documents"][0]["text"] += " Payout weekly."
        body = self.value["documents"][0]["text"]
        self.value["evidence"].append({
            "id": "payout_quote", "requirement_id": "payout_record",
            "document_id": "setup_notes", "quote": "Payout weekly.",
            "start": body.index("Payout weekly."),
        })
        result = self.run_value()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["stages"]["review"]["data"]["gaps"], [])
        self.assertEqual(result["stages"]["guided"]["data"]["progress"]["fraction"], 1.0)

    def test_empty_catalogs_abstain(self):
        for key in ("knowledge_base", "setup", "requirements", "documents", "evidence"):
            self.value[key] = []
        self.assertEqual(self.run_value()["stages"]["faq"]["status"], "abstained")

    def test_faq_without_setup_has_empty_review(self):
        self.value["knowledge_base"][0]["step_ids"] = []
        stages = self.run_value()["stages"]
        self.assertEqual(stages["guided"]["data"]["progress"]["total"], 0)
        self.assertEqual(stages["review"]["data"]["checks"], [])

    def test_unselected_requirement_is_not_reviewed(self):
        self.value["requirements"].append({
            "id": "unselected", "description": "SYNTHETIC unrelated", "required_terms": ["other"],
        })
        self.assertNotIn("unselected", self.run_value()["stages"]["review"]["requirement_ids"])

    def test_repeatability_and_input_not_mutated(self):
        before = copy.deepcopy(self.value)
        self.assertEqual(self.run_value(), self.run_value())
        self.assertEqual(self.value, before)

    def test_missing_field_unknown_field_and_wrong_type(self):
        original = copy.deepcopy(self.value)
        for alteration in ("missing", "unknown", "type", "not_synthetic", "version"):
            with self.subTest(alteration=alteration):
                self.value = copy.deepcopy(original)
                if alteration == "missing":
                    del self.value["query"]
                elif alteration == "unknown":
                    self.value["extra"] = 1
                elif alteration == "type":
                    self.value["setup"][0]["completed"] = 1
                elif alteration == "not_synthetic":
                    self.value["synthetic"] = False
                else:
                    self.value["schema_version"] = "2.0"
                self.invalid()

    def test_duplicate_and_unknown_references(self):
        original = copy.deepcopy(self.value)
        for refs in (["absent"], ["configure_payouts", "configure_payouts"]):
            self.value = copy.deepcopy(original)
            self.value["knowledge_base"][0]["step_ids"] = refs
            self.invalid()
        self.value = original
        self.value["setup"].append(copy.deepcopy(self.value["setup"][0]))
        self.invalid()

    def test_prerequisite_cycle(self):
        self.value["setup"][0]["prerequisites"] = ["configure_payouts"]
        self.invalid()

    def test_completed_step_requires_completed_prerequisites(self):
        self.value["setup"][0]["completed"] = False
        self.value["setup"][1]["completed"] = True
        self.invalid()

    def test_fabricated_quote_bad_offset_and_boolean_offset(self):
        original = copy.deepcopy(self.value)
        for field, value in (("quote", "Fabricated quote"), ("start", 0), ("start", True)):
            with self.subTest(field=field, value=value):
                self.value = copy.deepcopy(original)
                self.value["evidence"][0][field] = value
                self.invalid()

    def test_empty_terms_invalid(self):
        self.value["requirements"][0]["required_terms"] = []
        self.invalid()

    def test_tampered_faq_handoff_rejected(self):
        indexes = app.validate_input(self.value)
        faq = app.answer_faq(self.value["query"], indexes)
        faq["data"]["answer"] = "Unsupported invented claim"
        with self.assertRaises(app.ValidationError):
            app.guided_setup(faq, indexes)

    def test_tampered_guided_handoff_rejected(self):
        indexes = app.validate_input(self.value)
        faq = app.answer_faq(self.value["query"], indexes)
        guided = app.guided_setup(faq, indexes)
        guided["requirement_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.review_documents(guided, indexes)

    def test_cli_success_single_json(self):
        completed = self.cli("example_input.json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(json.loads(completed.stdout), self.run_value())
        self.assertEqual(len(completed.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in ((), ("nonexistent_synthetic_input.json",), ("a", "b")):
            with self.subTest(args=args):
                completed = self.cli(*args)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stderr, "")
                self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_invalid_schema(self):
        completed = self.cli("build_manifest.json")
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_main_malformed_json_duplicate_keys_and_nonfinite(self):
        for payload in ('{', '{"x": 1, "x": 2}', '{"x": NaN}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.StringIO(payload)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
