import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_case(self):
        return app.run_pipeline(self.payload)

    def complete_evidence(self):
        self.payload["data"]["customer"]["kyc"]["screening_flags"] = []
        for tx in self.payload["data"]["ledger"]["data"]:
            tx["screening_flags"] = []
        self.payload["data"]["issues"] = []
        doc = copy.deepcopy(self.payload["data"]["documents"][0])
        doc.update(id="doc_income", type="income_proof", fields={"annual_income": "48000.00"})
        self.payload["data"]["documents"].append(doc)

    def assert_invalid(self):
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_example_and_shared_envelope(self):
        result = self.run_case()
        self.assertEqual(set(result), set(self.payload))
        self.assertEqual(result["stage"], "faq")
        self.assertEqual(list(result["results"]), list(app.STAGES[1:]))
        app.validate_state(result, "faq")

    def test_sentiment_contributions_explain_score(self):
        item = self.run_case()["results"]["sentiment"]["issues"][0]
        self.assertEqual(item["sentiment"]["score"], -5)
        self.assertEqual(sum(c["weight"] for c in item["sentiment"]["explanation"]["contributions"]), -5)
        self.assertEqual(item["priority"]["score"], 100)
        self.assertEqual(item["severity"], "critical")

    def test_positive_and_neutral_sentiment(self):
        self.payload["data"]["issues"] = [
            {"id": "positive", "text": "excellent helpful", "severity": "low"},
            {"id": "neutral", "text": "Account statement", "severity": "low"}]
        result = self.run_case()["results"]["sentiment"]["issues"]
        by_id = {i["id"]: i for i in result}
        self.assertEqual(by_id["positive"]["sentiment"]["score"], 3)
        self.assertEqual(by_id["neutral"]["sentiment"]["label"], "neutral")
        self.assertEqual([i["id"] for i in result], ["neutral", "positive"])

    def test_high_severity_outweighs_negative_low(self):
        self.payload["data"]["issues"] = [
            {"id": "high", "text": "excellent", "severity": "high"},
            {"id": "low", "text": "terrible angry bad", "severity": "low"}]
        result = self.run_case()["results"]["sentiment"]["issues"]
        self.assertEqual(result[0]["id"], "high")

    def test_score_clamping(self):
        self.payload["data"]["issues"][0]["text"] = "fraud " * 20
        item = self.run_case()["results"]["sentiment"]["issues"][0]
        self.assertEqual(item["sentiment"]["explanation"]["raw_sum"], -60)
        self.assertEqual(item["sentiment"]["score"], -5)
        self.assertEqual(item["priority"]["score"], 100)

    def test_sentiment_to_adaptive_priority_and_incident(self):
        results = self.run_case()["results"]
        self.assertEqual(results["adaptive"]["focus_issue_ids"][0], "issue_fraud")
        self.assertIn("req_incident_report", [r["id"] for r in results["adaptive"]["requirements"]])
        self.assertEqual(results["sentiment"]["urgent_issue_ids"], results["adaptive"]["urgent_issue_ids"])

    def test_experience_and_preference_adaptation(self):
        self.assertIn("orientation", [m["id"] for m in self.run_case()["results"]["adaptive"]["modules"]])
        customer = self.payload["data"]["customer"]
        customer["experience"] = "experienced"
        customer["preferences"] = {"format": "narrative", "pace": "fast"}
        plan = self.run_case()["results"]["adaptive"]
        self.assertEqual(plan["format"], "narrative")
        self.assertNotIn("orientation", [m["id"] for m in plan["modules"]])
        self.assertIn("req_identity_document", [r["id"] for r in plan["requirements"]])

    def test_prerequisites_and_holds(self):
        plan = self.run_case()["results"]["adaptive"]
        modules = {m["id"]: m for m in plan["modules"]}
        self.assertEqual(modules["screening"]["status"], "needs_review")
        self.assertEqual(modules["ledger"]["blocked_by"], ["screening"])
        self.assertEqual(modules["loan_preparation"]["status"], "blocked")

    def test_kyc_false_propagation(self):
        self.payload["data"]["customer"]["kyc"]["identity_verified"] = False
        results = self.run_case()["results"]
        screen = next(m for m in results["adaptive"]["modules"] if m["id"] == "screening")
        self.assertEqual(screen["blocked_by"], ["identity"])
        self.assertIn("identity_unverified", results["faq"]["case_context"]["hold_reasons"])

    def test_flags_have_sources_and_survive_all_handoffs(self):
        results = self.run_case()["results"]
        for stage in ("sentiment", "adaptive", "review"):
            self.assertEqual(results[stage]["screening_flags"], ["aml_alert", "pep_review"])
        self.assertIn({"source": "transaction:tx_02", "flag": "aml_alert"},
                      results["sentiment"]["flag_sources"])
        self.assertEqual(results["faq"]["case_context"]["hold_reasons"], ["aml_alert", "pep_review"])

    def test_adaptive_to_review_exact_requirements(self):
        results = self.run_case()["results"]
        self.assertEqual([r["id"] for r in results["adaptive"]["requirements"]],
                         [r["requirement_id"] for r in results["review"]["checks"]])
        identity = results["review"]["checks"][0]
        self.assertEqual(identity["evidence_ids"], ["doc_identity"])
        self.assertIn("req_income_proof", results["review"]["gap_ids"])

    def test_expired_draft_mismatch_and_missing_evidence(self):
        doc = self.payload["data"]["documents"][0]
        doc.update(expires_on="2026-09-23", status="draft", customer_id="other_customer", fields={})
        check = self.run_case()["results"]["review"]["checks"][0]
        self.assertEqual(check["status"], "gap")
        reasons = check["evaluations"][0]["reasons"]
        for reason in ("expired", "not_final", "owner_or_loan_mismatch", "missing_field:legal_name"):
            self.assertIn(reason, reasons)

    def test_expiry_day_is_inclusive(self):
        self.payload["data"]["documents"][0]["expires_on"] = "2026-09-24"
        self.assertEqual(self.run_case()["results"]["review"]["checks"][0]["status"], "satisfied")

    def test_future_issued_document_is_gap(self):
        self.payload["data"]["documents"][0]["issued_on"] = "2026-10-01"
        self.assertIn("future_issued", self.run_case()["results"]["review"]["checks"][0]["evaluations"][0]["reasons"])

    def test_loan_fields_must_match(self):
        self.payload["data"]["documents"][2]["fields"]["requested_amount"] = "999.00"
        check = self.run_case()["results"]["review"]["checks"][2]
        self.assertIn("loan_field_mismatch:requested_amount", check["evaluations"][0]["reasons"])

    def test_multiple_evidence_candidates(self):
        doc = copy.deepcopy(self.payload["data"]["documents"][0])
        doc["id"] = "alternate_identity"
        self.payload["data"]["documents"][0]["status"] = "draft"
        self.payload["data"]["documents"].append(doc)
        check = self.run_case()["results"]["review"]["checks"][0]
        self.assertEqual(check["evidence_ids"], ["alternate_identity"])
        self.assertEqual(len(check["evaluations"]), 2)

    def test_complete_evidence_is_not_approval(self):
        self.complete_evidence()
        results = self.run_case()["results"]
        self.assertEqual(results["review"]["status"], "evidence_complete")
        self.assertEqual(results["review"]["decision"], "not_a_loan_decision")
        self.assertEqual(results["faq"]["case_context"]["gap_ids"], [])

    def test_documents_do_not_clear_screening_flags(self):
        self.complete_evidence()
        self.payload["data"]["customer"]["kyc"]["screening_flags"] = ["sanctions_match"]
        for kind, fields in (("source_of_funds", {"source": "Synthetic savings"}),
                             ("manual_screening_review", {"reviewer": "Synthetic reviewer", "outcome": "reviewed"})):
            doc = copy.deepcopy(self.payload["data"]["documents"][0])
            doc.update(id="doc_" + kind, type=kind, fields=fields)
            self.payload["data"]["documents"].append(doc)
        review = self.run_case()["results"]["review"]
        self.assertEqual(review["gap_ids"], [])
        self.assertEqual(review["hold_reasons"], ["sanctions_match"])
        self.assertEqual(review["status"], "needs_action")

    def test_faq_is_grounded_and_explained(self):
        result = self.run_case()["results"]["faq"]
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "\n\n".join(c["excerpt"] for c in result["citations"]))
        sources = {k["id"]: k["text"] for k in self.payload["data"]["knowledge_base"]}
        for citation in result["citations"]:
            self.assertEqual(citation["excerpt"], sources[citation["knowledge_id"]])
        for candidate in result["retrieval"]:
            e = candidate["explanation"]
            self.assertEqual(candidate["score"], 2 * len(e["title_matches"]) +
                             len(e["body_tag_matches"]) + len(e["review_context_matches"]))

    def test_review_to_faq_applicability(self):
        self.payload["data"]["question"] = "Evidence complete"
        result = self.run_case()["results"]["faq"]
        self.assertNotIn("kb_complete", [c["knowledge_id"] for c in result["citations"]])
        self.complete_evidence()
        result = self.run_case()["results"]["faq"]
        self.assertIn("kb_complete", [c["knowledge_id"] for c in result["citations"]])

    def test_abstention_does_not_retrieve_on_context_alone(self):
        self.payload["data"]["question"] = "Quantum penguin astronomy"
        result = self.run_case()["results"]["faq"]
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertTrue(result["abstention_reason"])

    def test_empty_collections_are_valid(self):
        for name in ("issues", "documents", "knowledge_base"):
            self.payload["data"][name] = []
        self.payload["data"]["ledger"]["data"] = []
        result = self.run_case()["results"]
        self.assertEqual(result["sentiment"]["issues"], [])
        self.assertEqual(result["faq"]["status"], "abstained")
        self.assertTrue(result["review"]["gap_ids"])

    def test_masking_card_fields_and_free_text(self):
        self.payload["data"]["knowledge_base"][0]["text"] += " Card 5555-5555-5555-4444."
        output = json.dumps(self.run_case())
        for raw in ("4111111111111111", "4111 1111 1111 1111", "5555-5555-5555-4444"):
            self.assertNotIn(raw, output)
        self.assertIn("****1111", output)
        self.assertIn("****4444", output)

    def test_csv_matches_json(self):
        expected = self.run_case()
        rows = self.payload["data"]["ledger"]["data"]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "screening_flags": "|".join(row["screening_flags"])})
        self.payload["data"]["ledger"] = {"format": "csv", "data": buffer.getvalue()}
        self.assertEqual(self.run_case(), expected)

    def test_iso20022_style_matches_json(self):
        expected = self.run_case()
        rows = self.payload["data"]["ledger"]["data"]
        self.payload["data"]["ledger"] = {"format": "iso20022", "data": {
            "MsgId": "synthetic_message",
            "PmtInf": [{"DbtrAcct": {"Id": rows[0]["account"]}, "CdtTrfTxInf": [
                {"PmtId": {"EndToEndId": row["id"]},
                 "Amt": {"InstdAmt": {"Value": row["amount"], "Ccy": row["currency"]}},
                 "CdtrAcct": {"IBAN": row["counterparty_iban"]},
                 "CardNumber": row["card_number"], "ScreeningFlags": row["screening_flags"]}
                for row in rows]}]}}
        self.assertEqual(self.run_case(), expected)

    def test_seeded_randomized_synthetic_histories(self):
        rng = random.Random(112)
        template = self.payload["data"]["ledger"]["data"][0]
        self.payload["data"]["ledger"]["data"] = [
            {**template, "id": "random_tx_" + str(i),
             "amount": format(app.Decimal(rng.randrange(1, 900000)) / 100, ".2f"),
             "screening_flags": ["aml_alert"] if rng.randrange(7) == 0 else []}
            for i in range(50)]
        self.assertEqual(self.run_case(), self.run_case())
        self.assertEqual(len(self.run_case()["data"]["ledger"]["data"]), 50)

    def test_input_not_mutated(self):
        original = copy.deepcopy(self.payload)
        self.run_case()
        self.assertEqual(self.payload, original)

    def test_handoff_tampering_rejected(self):
        state = app.advance(self.payload, "sentiment")
        state["results"]["sentiment"]["issues"][0]["priority"]["score"] = 0
        with self.assertRaises(app.ValidationError):
            app.advance(state, "adaptive")

    def test_each_stage_rejects_missing_predecessor(self):
        for stage in ("adaptive", "review", "faq"):
            with self.subTest(stage=stage), self.assertRaises(app.ValidationError):
                app.advance(self.payload, stage)

    def test_numeric_handoff_types_are_strict(self):
        state = app.advance(self.payload, "sentiment")
        state["results"]["sentiment"]["issues"][0]["priority"]["score"] = 100.0
        with self.assertRaises(app.ValidationError):
            app.advance(state, "adaptive")

    def test_review_and_faq_tampering_rejected(self):
        result = self.run_case()
        result["results"]["review"]["gap_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.validate_state(result, "faq")
        result = self.run_case()
        result["results"]["faq"]["answer"] = "Invented unsupported approval."
        with self.assertRaises(app.ValidationError):
            app.validate_state(result, "faq")

    def test_changed_context_rejects_stale_results(self):
        state = app.advance(self.payload, "sentiment")
        state["data"]["issues"][0]["text"] = "excellent"
        with self.assertRaises(app.ValidationError):
            app.advance(state, "adaptive")

    def test_invalid_amounts(self):
        for value in (True, 0, -1, "NaN", "Infinity", "12.123", [], "1000000001"):
            with self.subTest(value=value):
                self.payload["data"]["ledger"]["data"][0]["amount"] = value
                self.assert_invalid()

    def test_invalid_flags_boolean_and_duplicate_ids(self):
        baseline = copy.deepcopy(self.payload)
        self.payload["data"]["customer"]["kyc"]["identity_verified"] = "true"
        self.assert_invalid()
        self.payload = copy.deepcopy(baseline)
        self.payload["data"]["customer"]["kyc"]["screening_flags"] = ["unknown"]
        self.assert_invalid()
        self.payload = copy.deepcopy(baseline)
        self.payload["data"]["issues"].append(copy.deepcopy(self.payload["data"]["issues"][0]))
        self.assert_invalid()

    def test_requires_synthetic_and_fake_accounts(self):
        self.payload["synthetic"] = False
        self.assert_invalid()
        self.payload["synthetic"] = True
        self.payload["data"]["ledger"]["data"][0]["counterparty_iban"] = "DE89370400440532013000"
        self.assert_invalid()

    def test_unknown_fields_and_wrong_schema_rejected(self):
        self.payload["unexpected"] = True
        self.assert_invalid()
        del self.payload["unexpected"]
        self.payload["schema_version"] = "2"
        self.assert_invalid()

    def test_invalid_csv_and_iso_shapes(self):
        self.payload["data"]["ledger"] = {"format": "csv", "data": "bad,header\n1,2"}
        self.assert_invalid()
        self.payload["data"]["ledger"] = {"format": "iso20022", "data": {"MsgId": "m"}}
        self.assert_invalid()

    def test_cli_success_actual_process(self):
        proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                               str(HERE / "example_input.json")],
                              cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["stage"], "faq")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_missing_file_actual_process(self):
        proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                               str(HERE / "does_not_exist.json")],
                              cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")
        self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_schema_without_extra_files(self):
        for source in ("{", '{"x": NaN}', '{"a": 1, "a": 2}', "null",
                       json.dumps({**self.payload, "synthetic": False})):
            with self.subTest(source=source[:30]):
                stream = io.StringIO()
                with patch("builtins.open", mock_open(read_data=source)), contextlib.redirect_stdout(stream):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_usage(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_oversized_and_large_integer_json_errors(self):
        for raw in (" " * 1000001, "9" * 5000):
            with self.subTest(length=len(raw)):
                stream = io.StringIO()
                with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(stream):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_malformed_schema_types_are_controlled_errors(self):
        baseline = copy.deepcopy(self.payload)
        paths = []

        def visit(value, path=()):
            paths.append(path)
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, path + (key,))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, path + (index,))

        visit(baseline)
        for path in paths:
            for replacement in (None, True, 0, -1, 1.5, "", [], {}, "NaN"):
                with self.subTest(path=path, replacement=replacement):
                    sample = copy.deepcopy(baseline)
                    if not path:
                        sample = replacement
                    else:
                        node = sample
                        for key in path[:-1]:
                            node = node[key]
                        node[path[-1]] = replacement
                    try:
                        result = app.run_pipeline(sample)
                    except app.ValidationError:
                        continue
                    app.validate_state(result, "faq")


if __name__ == "__main__":
    unittest.main()
