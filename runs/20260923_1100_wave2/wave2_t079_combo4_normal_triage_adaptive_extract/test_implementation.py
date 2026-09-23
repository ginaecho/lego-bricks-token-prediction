"""Standard-library synthetic contract and CLI tests; no temporary files."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
import contextlib
import io

import implementation as impl


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return impl.run_pipeline(self.data)

    def initial(self):
        return {"schema_version": 1, "status": "ok", "synthetic": True,
                "input": copy.deepcopy(self.data), "stages": {}}

    def test_full_pipeline_validates(self):
        result = self.run_data()
        self.assertEqual(list(result["stages"]), list(impl.PHASES))
        self.assertIs(impl.validate(result, 4), result)

    def test_deterministic_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(before, self.data)

    def test_research_exact_citations_and_ranking(self):
        findings = self.run_data()["stages"]["research"]["findings"]
        self.assertTrue(findings)
        self.assertEqual([f["score"] for f in findings],
                         sorted([f["score"] for f in findings], reverse=True))
        docs = {d["id"]: d["text"] for d in self.data["documents"]}
        for finding in findings:
            source = finding["source"]
            self.assertEqual(finding["finding"], docs[source["document_id"]][source["start"]:source["end"]])

    def test_research_no_matches_and_default_triage(self):
        self.data["query"] = "nonexistentword"
        result = self.run_data()["stages"]
        self.assertEqual(result["research"]["findings"], [])
        self.assertEqual(result["triage"]["tickets"][0]["category"], "general")
        self.assertEqual(result["triage"]["tickets"][0]["priority"], "normal")

    def test_research_limit(self):
        self.data["policy"]["research_limit"] = 1
        self.assertEqual(len(self.run_data()["stages"]["research"]["findings"]), 1)

    def test_research_retrieves_unpunctuated_lines_and_trailing_text(self):
        self.data["documents"][0]["text"] = "  invoice first\ninvoice second\ninvoice final"
        findings = self.run_data()["stages"]["research"]["findings"]
        self.assertEqual([finding["finding"] for finding in findings],
                         ["invoice first", "invoice second", "invoice final"])
        self.assertEqual(findings[0]["source"]["start"], 2)

    def test_research_drives_category_priority_and_routing(self):
        route = self.run_data()["stages"]["triage"]["tickets"][0]
        self.assertEqual((route["category"], route["priority"], route["queue"], route["owner"]),
                         ("billing", "urgent", "accounts", "synthetic-billing-lead"))
        self.assertTrue(route["finding_ids"])

    def test_unrelated_ticket_does_not_receive_other_document_evidence(self):
        self.data["tickets"].append({"id": "isolated", "title": "A question",
                                     "description": "Need information", "document_ids": []})
        stages = self.run_data()["stages"]
        route = stages["triage"]["tickets"][1]
        self.assertEqual(route["finding_ids"], [])
        self.assertEqual(route["category"], "general")
        self.assertTrue(stages["extract"]["records"][1]["missing_required_fields"])

    def test_category_tie_uses_configuration_order(self):
        self.data["tickets"][0].update(title="shipping invoice", description="question", document_ids=[])
        self.assertEqual(self.run_data()["stages"]["triage"]["tickets"][0]["category"], "billing")
        self.data["policy"]["categories"].reverse()
        self.assertEqual(self.run_data()["stages"]["triage"]["tickets"][0]["category"], "shipping")

    def test_priority_uses_configured_order(self):
        self.data["policy"]["priorities"].reverse()
        self.assertEqual(self.run_data()["stages"]["triage"]["tickets"][0]["priority"], "high")

    def test_keyword_boundaries(self):
        self.data["tickets"][0].update(title="invoiced shippingly", description="question", document_ids=[])
        self.assertEqual(self.run_data()["stages"]["triage"]["tickets"][0]["category"], "general")

    def test_prerequisites_inserted_before_target_even_for_other_audience(self):
        steps = self.run_data()["stages"]["adaptive"]["assignments"][0]["steps"]
        self.assertEqual([s["step_id"] for s in steps], ["verify", "billing-intro"])
        self.assertIn("prerequisite", steps[0]["explanation"])

    def test_preference_and_fallback_explained(self):
        steps = self.run_data()["stages"]["adaptive"]["assignments"][0]["steps"]
        self.assertEqual([s["mode"] for s in steps], ["text", "interactive"])
        self.assertIn("unavailable", steps[0]["explanation"])
        self.assertIn("preferred", steps[1]["explanation"])

    def test_experience_changes_path_and_downstream_fields(self):
        self.data["profile"]["experience"] = "advanced"
        result = self.run_data()["stages"]
        self.assertEqual([s["step_id"] for s in result["adaptive"]["assignments"][0]["steps"]],
                         ["billing-shortcut"])
        self.assertIsNone(result["extract"]["records"][0]["fields"]["plan"])

    def test_handoffs_preserve_accountability_and_evidence(self):
        stages = self.run_data()["stages"]
        route = stages["triage"]["tickets"][0]
        assignment = stages["adaptive"]["assignments"][0]
        record = stages["extract"]["records"][0]
        for key in ("category", "priority", "queue", "owner", "finding_ids"):
            self.assertEqual(route[key], assignment[key])
            self.assertEqual(assignment[key], record[key])
        self.assertIn("synthetic-onboarding", record["document_ids"])
        self.assertEqual(record["fields"]["plan"]["value"], "Starter")

    def test_extraction_exact_span_and_missing_optional(self):
        record = self.run_data()["stages"]["extract"]["records"][0]
        self.assertEqual(record["fields"]["invoice_id"]["value"], "SYN-1042")
        self.assertEqual(record["missing_fields"], ["optional_email"])
        self.assertEqual(record["missing_required_fields"], [])
        self.assertTrue(record["complete"])
        self.assertEqual(record["fields"]["plan"]["source"]["document_id"], "synthetic-onboarding")

    def test_missing_required_is_reported_not_fabricated(self):
        self.data["extraction_schema"][0]["pattern"] = r"Absent: (?P<value>\w+)"
        record = self.run_data()["stages"]["extract"]["records"][0]
        self.assertIsNone(record["fields"]["invoice_id"])
        self.assertEqual(record["missing_required_fields"], ["invoice_id"])
        self.assertFalse(record["complete"])

    def test_unicode_offsets(self):
        self.data["documents"][0]["text"] = "🧱 café. " + self.data["documents"][0]["text"]
        record = self.run_data()["stages"]["extract"]["records"][0]
        source = record["fields"]["invoice_id"]["source"]
        self.assertEqual(self.data["documents"][0]["text"][source["start"]:source["end"]], "SYN-1042")

    def test_first_match_uses_assigned_document_order(self):
        self.data["documents"][1]["text"] += "Invoice: SYN-9999"
        record = self.run_data()["stages"]["extract"]["records"][0]
        self.assertEqual(record["fields"]["invoice_id"]["value"], "SYN-1042")

    def test_empty_optional_capture_is_missing(self):
        self.data["extraction_schema"][0]["pattern"] = r"(?P<value>)"
        self.assertIsNone(self.run_data()["stages"]["extract"]["records"][0]["fields"]["invoice_id"])

    def test_empty_onboarding_allowed(self):
        self.data["onboarding_steps"] = []
        self.assertEqual(self.run_data()["stages"]["adaptive"]["assignments"][0]["steps"], [])

    def test_invalid_root_and_required_fields(self):
        for data in (None, [], {}, {"schema_version": True}):
            with self.subTest(data=data), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(data)

    def test_invalid_document_reference_and_duplicate_ids(self):
        self.data["tickets"][0]["document_ids"] = ["missing"]
        with self.assertRaises(impl.ValidationError):
            self.run_data()
        self.setUp()
        self.data["documents"].append(copy.deepcopy(self.data["documents"][0]))
        with self.assertRaises(impl.ValidationError):
            self.run_data()

    def test_cycle_and_unknown_prerequisite(self):
        self.data["onboarding_steps"][0]["prerequisites"] = ["billing-intro"]
        with self.assertRaisesRegex(impl.ValidationError, "cycle"):
            self.run_data()
        self.data["onboarding_steps"][0]["prerequisites"] = ["unknown"]
        with self.assertRaisesRegex(impl.ValidationError, "unknown prerequisite"):
            self.run_data()

    def test_invalid_regex_and_missing_capture(self):
        for pattern in ("[", "invoice"):
            with self.subTest(pattern=pattern), self.assertRaises(impl.ValidationError):
                self.data["extraction_schema"][0]["pattern"] = pattern
                self.run_data()

    def test_invalid_profile_and_priority(self):
        self.data["profile"]["preference"] = "telepathy"
        with self.assertRaises(impl.ValidationError):
            self.run_data()
        self.setUp()
        self.data["policy"]["default_priority"] = "unknown"
        with self.assertRaises(impl.ValidationError):
            self.run_data()

    def test_required_boolean_and_research_limit_types(self):
        self.data["extraction_schema"][0]["required"] = 1
        with self.assertRaises(impl.ValidationError):
            self.run_data()
        self.setUp()
        self.data["policy"]["research_limit"] = True
        with self.assertRaises(impl.ValidationError):
            self.run_data()

    def test_stage_order_enforced(self):
        with self.assertRaises(impl.ValidationError):
            impl.triage(self.initial())
        result = self.run_data()
        del result["stages"]["triage"]
        with self.assertRaises(impl.ValidationError):
            impl.validate(result)

    def test_tampered_source_rejected_before_triage(self):
        state = impl.research(self.initial())
        state["stages"]["research"]["findings"][0]["source"]["text"] = "fabricated"
        with self.assertRaisesRegex(impl.ValidationError, "span"):
            impl.triage(state)

    def test_tampered_routing_rejected_before_onboarding(self):
        state = impl.triage(impl.research(self.initial()))
        state["stages"]["triage"]["tickets"][0]["owner"] = "unassigned"
        with self.assertRaisesRegex(impl.ValidationError, "routing"):
            impl.adaptive(state)

    def test_tampered_prerequisite_order_rejected(self):
        state = impl.adaptive(impl.triage(impl.research(self.initial())))
        state["stages"]["adaptive"]["assignments"][0]["steps"].reverse()
        with self.assertRaisesRegex(impl.ValidationError, "prerequisite"):
            impl.extract(state)

    def test_cli_success_json_only(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")],
                                 capture_output=True, text=True, cwd=HERE, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["nonexistent-input.json"], ["one", "two"]):
            process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                     capture_output=True, text=True, cwd=HERE, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_invalid_duplicate_and_nonstandard_json(self):
        for content in ("{broken", "[]", '{"x":1,"x":2}', '{"x":NaN}'):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                code = impl.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_bad_encoding(self):
        output = io.StringIO()
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
        with patch("builtins.open", side_effect=error), contextlib.redirect_stdout(output):
            self.assertEqual(impl.main(["bad-encoding.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
