"""Synthetic fixtures only; subprocess tests make no network calls."""

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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def triage(self):
        return app.triage_ticket(self.data)

    def researched(self):
        return app.research_ticket(self.data, self.triage())

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_full_pipeline_and_handoffs(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["ticket_id"], self.data["ticket"]["id"])
        for field in ("category", "routing"):
            self.assertEqual(result["triage"][field], result["research"][field])
            self.assertEqual(result["research"][field], result["onboarding"][field])
        self.assertEqual(result["triage"]["query_terms"], result["research"]["query_terms"])
        self.assertEqual(result["onboarding"]["evidence_finding_ids"], ["F1", "F2"])

    def test_category_priority_and_accountability(self):
        triage = self.triage()["triage"]
        self.assertEqual(triage["category"], "billing")
        self.assertEqual(triage["priority"], "urgent")
        self.assertEqual(triage["routing"]["owner"], "synthetic-billing-owner")

    def test_default_category(self):
        self.data["ticket"].update(subject="Unrecognized question", body="")
        triage = self.triage()["triage"]
        self.assertEqual(triage["category"], "general")
        self.assertEqual(triage["priority"], "low")

    def test_tie_uses_configuration_order(self):
        self.data["ticket"].update(subject="invoice password", body="")
        self.assertEqual(self.triage()["triage"]["category"], "billing")

    def test_keywords_are_token_based_case_insensitive(self):
        self.data["ticket"].update(subject="PASSWORD", body="login")
        self.assertEqual(self.triage()["triage"]["category"], "account")
        self.data["ticket"].update(subject="passwordless", body="")
        self.assertEqual(self.triage()["triage"]["category"], "general")

    def test_configurable_priority_and_route(self):
        self.data["triage_config"]["categories"][0]["routing"]["owner"] = "new-owner"
        self.data["triage_config"]["priority_rules"] = []
        triage = self.triage()["triage"]
        self.assertEqual(triage["priority"], "normal")
        self.assertEqual(triage["routing"]["owner"], "new-owner")

    def test_exact_citations_and_extracts(self):
        findings = self.researched()["research"]["findings"]
        self.assertEqual(len(findings), 2)
        for finding in findings:
            citation = finding["citation"]
            document = next(d for d in self.data["corpus"] if d["id"] == citation["document_id"])
            passage = document["passages"][citation["passage_index"]]
            self.assertEqual(finding["text"], passage[citation["start"]:citation["end"]])
            self.assertEqual(citation["quote"], finding["text"])

    def test_unicode_exact_offsets(self):
        self.data["corpus"][0]["passages"][0] = "  Invoice café — refund 🔎.\n"
        research = self.researched()["research"]
        finding = next(f for f in research["findings"] if "café" in f["text"])
        self.assertEqual(finding["citation"]["end"], len(finding["text"]))
        self.assertTrue(finding["text"].startswith("  "))

    def test_research_limit_and_order(self):
        self.data["research_config"]["max_findings"] = 1
        research = self.researched()["research"]
        self.assertEqual(len(research["findings"]), 1)
        self.assertGreaterEqual(research["retrieved_passages_count"], 2)
        self.assertEqual(research["findings"][0]["id"], "F1")

    def test_tied_retrieval_order_uses_document_id(self):
        self.data["corpus"] = [
            {"id": "z", "title": "Synthetic Z", "passages": ["invoice"]},
            {"id": "a", "title": "Synthetic A", "passages": ["invoice"]},
        ]
        findings = self.researched()["research"]["findings"]
        self.assertEqual([f["citation"]["document_id"] for f in findings], ["a", "z"])

    def test_empty_corpus_no_invented_findings(self):
        self.data["corpus"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(result["research"]["no_evidence"])
        self.assertEqual(result["research"]["findings"], [])
        self.assertEqual([s["id"] for s in result["onboarding"]["steps"]], ["orientation"])
        self.assertIn("No retrieved evidence", result["onboarding"]["explanation"])

    def test_unrelated_passages_do_not_become_findings(self):
        self.data["corpus"] = [{"id": "x", "title": "Synthetic", "passages": ["Zebras gallop."]}]
        self.assertEqual(self.researched()["research"]["findings"], [])

    def test_prerequisites_and_preference(self):
        result = app.run_pipeline(self.data)["onboarding"]
        self.assertEqual([s["id"] for s in result["steps"]],
                         ["orientation", "review-invoice", "request-refund"])
        done = set()
        for step in result["steps"]:
            self.assertTrue(set(step["prerequisites"]) <= done)
            self.assertEqual(step["format"], "hands_on")
            self.assertIn("novice", step["explanation"])
            done.add(step["id"])

    def test_expert_still_gets_required_prerequisite(self):
        self.data["profile"] = {"experience": "expert", "preference": "reading"}
        steps = app.run_pipeline(self.data)["onboarding"]["steps"]
        self.assertEqual(steps[0]["id"], "orientation")
        self.assertIn("Required prerequisite", steps[0]["explanation"])
        self.assertTrue(all(s["format"] == "reading" for s in steps))
        self.assertTrue(steps[1]["instruction"].startswith("Read"))

    def test_evidence_propagates_to_step_citations(self):
        result = app.run_pipeline(self.data)
        findings = {f["id"]: f for f in result["research"]["findings"]}
        step = result["onboarding"]["steps"][-1]
        self.assertTrue(step["finding_ids"])
        self.assertEqual(step["citations"], [findings[key]["citation"] for key in step["finding_ids"]])

    def test_account_route_changes_research_and_onboarding(self):
        self.data["ticket"].update(subject="Password login", body="")
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["category"], "account")
        self.assertEqual(result["onboarding"]["routing"]["owner"], "synthetic-account-owner")
        self.assertEqual([s["id"] for s in result["onboarding"]["steps"]],
                         ["orientation", "reset-password"])

    def test_empty_catalog(self):
        self.data["onboarding_catalog"] = []
        result = app.run_pipeline(self.data)["onboarding"]
        self.assertEqual(result["steps"], [])
        self.assertIn("No eligible", result["explanation"])

    def test_unknown_and_duplicate_categories(self):
        self.data["triage_config"]["default_category"] = "missing"
        self.invalid()
        self.setUp()
        self.data["triage_config"]["categories"].append(copy.deepcopy(
            self.data["triage_config"]["categories"][0]))
        self.invalid()

    def test_missing_accountable_owner(self):
        self.data["triage_config"]["categories"][0]["routing"]["owner"] = " "
        self.invalid()

    def test_invalid_profile_and_boolean_limit(self):
        self.data["profile"]["experience"] = "wizard"
        self.invalid()
        self.setUp()
        self.data["research_config"]["max_findings"] = True
        self.invalid()

    def test_unknown_fields_and_nonsynthetic_input(self):
        self.data["extra"] = "unsupported"
        self.invalid()
        self.setUp()
        self.data["synthetic"] = False
        self.invalid()

    def test_duplicate_document_ids(self):
        self.data["corpus"].append(copy.deepcopy(self.data["corpus"][0]))
        self.invalid()

    def test_missing_prerequisite_and_cycle(self):
        self.data["onboarding_catalog"][0]["prerequisites"] = ["missing"]
        self.invalid()
        self.setUp()
        self.data["onboarding_catalog"][0]["prerequisites"] = ["request-refund"]
        self.invalid()

    def test_stage_transition_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.onboard_ticket(self.data, self.triage())

    def test_tampered_handoff_rejected(self):
        previous = self.triage()
        previous["triage"]["routing"]["owner"] = "wrong-owner"
        with self.assertRaises(app.ValidationError):
            app.research_ticket(self.data, previous)
        previous = self.researched()
        previous["research"]["category"] = "account"
        with self.assertRaises(app.ValidationError):
            app.onboard_ticket(self.data, previous)

    def test_fabricated_citation_rejected(self):
        previous = self.researched()
        previous["research"]["findings"][0]["citation"]["quote"] = "Invented quotation"
        with self.assertRaises(app.ValidationError):
            app.onboard_ticket(self.data, previous)

    def test_injected_classifier_fixture(self):
        result = app.run_pipeline(self.data, lambda ticket, categories: "account")
        self.assertEqual(result["triage"]["category"], "account")
        self.assertEqual(result["triage"]["classification_method"], "injected_callable")
        self.assertEqual(result["onboarding"]["category"], "account")

    def test_invalid_injected_classifier(self):
        for response in ("unknown", {"category": "account"}, None):
            with self.subTest(response=response), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, lambda ticket, categories: response)
        def failing(ticket, categories):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "classifier failed"):
            app.run_pipeline(self.data, failing)

    def test_determinism_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(original, self.data)
        before = self.triage()
        saved = copy.deepcopy(before)
        app.research_ticket(self.data, before)
        self.assertEqual(before, saved)

    def test_classifier_cannot_mutate_configuration(self):
        def classifier(ticket, categories):
            categories[0]["routing"]["owner"] = "mutated"
            ticket["id"] = "mutated"
            return "billing"
        result = app.run_pipeline(self.data, classifier)
        self.assertEqual(result["ticket_id"], "SYNTHETIC-1001")
        self.assertEqual(result["triage"]["routing"]["owner"], "synthetic-billing-owner")

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "complete")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in ((), ("missing-synthetic-fixture.json",), ("one", "two")):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_schema(self):
        for text in ("{broken", '{"x": 1}', "[]", '{"x": NaN}', '{"x": 1, "x": 2}'):
            with self.subTest(text=text), patch("builtins.open", mock_open(read_data=text)):
                output = io.StringIO()
                with redirect_stdout(output):
                    status = app.main(["synthetic-bad-input.json"])
                self.assertEqual(status, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_utf8(self):
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid fixture")
        with patch("builtins.open", side_effect=error), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(app.main(["synthetic-invalid-encoding.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
