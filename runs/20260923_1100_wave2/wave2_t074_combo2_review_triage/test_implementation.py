import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_review_outcomes(self):
        result = app.review_stage(self.data)
        self.assertEqual([r["outcome"] for r in result["review"]["checks"]],
                         ["satisfied", "insufficient_evidence", "missing_evidence"])
        self.assertIn("not certification", result["review"]["notice"])

    def test_accountable_routes_and_priorities(self):
        tickets = app.run(self.data)["triage"]["tickets"]
        self.assertEqual([(t["category"], t["priority"]) for t in tickets],
                         [("security", 2), ("documentation", 4)])
        self.assertEqual(tickets[0]["owner"], "synthetic-security-lead")
        self.assertEqual(tickets[0]["team"], "synthetic-security")

    def test_cross_stage_traceability(self):
        result = app.run(self.data)
        for gap, ticket in zip(result["review"]["gaps"], result["triage"]["tickets"]):
            self.assertEqual(ticket["gap_id"], gap["id"])
            for key in ("document_id", "requirement_id", "evidence_ids", "missing_terms", "reason"):
                self.assertEqual(ticket[key], gap[key])
        self.assertEqual(len(result["triage"]["tickets"]), len(result["review"]["gaps"]))

    def test_forged_review_rejected(self):
        result = app.review_stage(self.data)
        result["review"]["gaps"][0]["severity"] = "low"
        with self.assertRaises(app.ValidationError):
            app.triage_stage(result)

    def test_forged_ticket_rejected(self):
        result = app.run(self.data)
        result["triage"]["tickets"][0]["owner"] = "unaccountable"
        with self.assertRaises(app.ValidationError):
            app.validate(result, "complete")

    def test_derived_types_are_strict(self):
        result = app.run(self.data)
        result["triage"]["tickets"][0]["matched_rule_index"] = False
        with self.assertRaises(app.ValidationError):
            app.validate(result, "complete")

    def test_unanchored_evidence_does_not_satisfy(self):
        self.data["input"]["evidence"][0]["quote"] = "backups daily fabricated"
        check = app.run(self.data)["review"]["checks"][0]
        self.assertEqual(check["outcome"], "invalid_evidence")
        self.assertEqual(check["accepted_evidence_ids"], [])
        self.assertEqual(check["rejected_evidence_ids"], ["E1"])

    def test_terms_can_span_multiple_valid_evidence_records(self):
        self.data["input"]["evidence"][0]["quote"] = "Backups"
        self.data["input"]["evidence"].append(
            {"id": "E3", "requirement_id": "R1", "quote": "daily"})
        self.assertEqual(app.run(self.data)["review"]["checks"][0]["outcome"], "satisfied")

    def test_empty_requirements_are_valid(self):
        self.data["input"]["requirements"] = []
        self.data["input"]["evidence"] = []
        result = app.run(self.data)
        self.assertEqual(result["review"]["checks"], [])
        self.assertEqual(result["triage"]["tickets"], [])

    def test_all_satisfied_produces_no_tickets(self):
        self.data["input"]["requirements"] = self.data["input"]["requirements"][:1]
        self.data["input"]["evidence"] = self.data["input"]["evidence"][:1]
        self.assertEqual(app.run(self.data)["triage"]["tickets"], [])

    def test_config_first_matching_rule_and_priority_floor(self):
        cfg = self.data["input"]["routing"]
        cfg["rules"].insert(0, {"keywords": ["SECURITY"], "category": "documentation"})
        cfg["categories"]["documentation"]["priority_floor"] = 1
        ticket = app.run(self.data)["triage"]["tickets"][0]
        self.assertEqual((ticket["category"], ticket["priority"], ticket["matched_rule_index"]),
                         ("documentation", 1, 0))

    def test_determinism_and_no_input_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, before)

    def test_invalid_input_variants(self):
        variants = []
        def variant():
            data = copy.deepcopy(self.data)
            variants.append(data)
            return data
        variant()["input"]["requirements"][0]["severity"] = []
        variant()["input"]["requirements"][0]["terms"] = []
        variant()["input"]["requirements"][0]["terms"] = ["x", "X"]
        variant()["input"]["requirements"][0]["id"] = "R2"
        variant()["input"]["evidence"][0]["requirement_id"] = "unknown"
        variant()["input"]["evidence"][0]["id"] = "E2"
        variant()["input"]["routing"]["default_category"] = "unknown"
        variant()["input"]["routing"]["rules"][0]["category"] = "unknown"
        variant()["input"]["routing"]["categories"]["security"]["owner"] = " "
        variant()["input"]["routing"]["severity_priority"]["high"] = True
        variant()["input"]["routing"]["severity_priority"]["high"] = 0
        variant()["synthetic"] = False
        variant()["schema_version"] = True
        variant()["extra"] = 1
        variant()["input"]["document"]["text"] = ""
        variants.extend([None, [], {}, 42])
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(app.ValidationError):
                    app.run(value)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.triage_stage(self.data)

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "complete")
        self.assertEqual(completed.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["one", "two"]):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        for payload in ('{', '{"x":1,"x":2}', 'NaN', 'null', '{"status":"input"}'):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=payload)):
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_encoding(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("invalid encoding")):
            with contextlib.redirect_stdout(output):
                code = app.main(["synthetic-invalid.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
