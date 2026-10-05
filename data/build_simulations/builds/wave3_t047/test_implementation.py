import contextlib
import copy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app

HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def bundle(self):
        return self.data["sources"][0]["content"]["ACORD"]

    def test_research_normalizes_monthly_cost_and_citations(self):
        output = app.research_stage(self.data)
        self.assertIsNone(output["comparison"])
        fact = output["research"]["evidence"][0]
        self.assertEqual(fact["value"], 1200)
        self.assertEqual(fact["source_ids"], ["SRC_JSON", "SRC_FORM"])

    def test_comparison_preference_ranking(self):
        output = app.run_pipeline(self.data)
        self.assertEqual(output["comparison"]["ranking"], ["POL_B", "POL_A"])
        self.assertEqual(output["comparison"]["side_by_side"][0]["attributes"]["coverage_limit"], 50000)

    def test_cross_stage_preserves_claim_reasons_and_factors(self):
        researched = app.research_stage(self.data)
        original = copy.deepcopy(researched)
        output = app.compare_stage(researched)
        self.assertEqual(researched, original)
        self.assertEqual(output["research"], researched["research"])
        row = output["comparison"]["side_by_side"][0]
        record = researched["research"]["records"][0]
        self.assertEqual(row["disclosed_underwriting_factors"], record["underwriting_submission"]["factors"])
        self.assertEqual(row["claim_decisions"][0]["reasons"], record["claims"][0]["reasons"])

    def test_minimization_removes_synthetic_identity_data(self):
        output = json.dumps(app.run_pipeline(self.data))
        for private in ("Invented Mira", "SYNTHVIN", "Fiction Lane", "policyholder", "Property Address"):
            self.assertNotIn(private, output)

    def test_xml_and_claim_form_parse(self):
        records = app.research_stage(self.data)["research"]["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["claims"][0]["decision"], "review")
        self.assertEqual(records[0]["claims"][0]["decision"], "approve")

    def test_disallows_hidden_and_sensitive_factors(self):
        factor = self.bundle()["underwriting_submission"]["factors"][0]
        factor["disclosed"] = False
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        factor["disclosed"] = True
        factor["name"] = "ethnicity"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_incomplete_evidence_cannot_be_denied(self):
        source = self.data["sources"][1]
        source["content"] = source["content"].replace('decision="review"', 'decision="deny"')
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_stated_claim_reason_required(self):
        form = self.data["sources"][2]
        form["content"] = form["content"].replace("Reasons: covered_loss", "Reasons: ")
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_rejects_tampered_handoff(self):
        output = app.research_stage(self.data)
        output["research"]["evidence"][0]["value"] = 1
        with self.assertRaises(app.ValidationError):
            app.compare_stage(output)

    def test_budget_excludes_without_fabricated_recommendation(self):
        self.data["preferences"]["max_annual_premium"] = 1
        output = app.run_pipeline(self.data)["comparison"]
        self.assertEqual(output["ranking"], [])
        self.assertIsNone(output["recommended_policy_id"])
        self.assertTrue(all(row["score"] is None for row in output["side_by_side"]))

    def test_single_candidate_has_finite_score(self):
        self.data["sources"].pop(1)
        output = app.run_pipeline(self.data)["comparison"]
        self.assertEqual(output["ranking"], ["POL_A"])
        self.assertEqual(output["side_by_side"][0]["score"], 1)

    def test_mixed_currency_rejected(self):
        self.bundle()["policy"]["currency"] = "USD"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_invalid_money(self):
        for value in (True, -1, float("nan"), float("inf"), 1.234):
            with self.subTest(value=value):
                self.bundle()["policy"]["premium"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_invalid_weights(self):
        for weights in ({}, {"annual_premium": 0}, {"secret": 1}, {"deductible": -1}):
            with self.subTest(weights=weights):
                self.data["preferences"]["weights"] = weights
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_duplicate_source_rejected(self):
        self.data["sources"].append(copy.deepcopy(self.data["sources"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_missing_policy_reference_rejected(self):
        self.data["sources"][2]["content"] = self.data["sources"][2]["content"].replace("POL_A", "UNKNOWN")
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_invalid_loss_date(self):
        source = self.data["sources"][2]
        source["content"] = re.sub(r"Loss Date: \d{4}-\d{2}-\d{2}", "Loss Date: 2025-02-30", source["content"])
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_denial_requires_supported_reason(self):
        source = self.data["sources"][2]
        source["content"] = (source["content"].replace("Covered: true", "Covered: false")
                             .replace("Decision: approve", "Decision: deny")
                             .replace("Reasons: covered_loss", "Reasons: excluded_peril"))
        output = app.run_pipeline(self.data)
        claim = output["comparison"]["side_by_side"][0]["claim_decisions"][0]
        self.assertEqual(claim["decision"], "deny")
        self.assertEqual(claim["reasons"], ["excluded_peril"])
        self.assertIsNone(claim["illustrative_payment"])

    def test_preference_changes_propagate(self):
        self.data["preferences"]["weights"] = {"coverage_limit": 1}
        output = app.run_pipeline(self.data)
        self.assertEqual(output["comparison"]["ranking"], ["POL_A", "POL_B"])

    def test_ties_are_stable(self):
        self.bundle()["policy"]["premium"] = 75
        self.data["preferences"]["weights"] = {"annual_premium": 1}
        output = app.run_pipeline(self.data)["comparison"]
        self.assertEqual(output["ranking"], ["POL_A", "POL_B"])
        self.assertEqual([row["score"] for row in output["side_by_side"]], [1, 1])

    def test_complete_output_revalidation_detects_tampering(self):
        output = app.run_pipeline(self.data)
        output["comparison"]["ranking"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate_document(output, "complete")

    def test_xml_entity_declaration_rejected(self):
        self.data["sources"][1]["content"] = '<!DOCTYPE ACORD [<!ENTITY x "bad">]><ACORD/>'
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_injected_offline_selector_is_validated(self):
        output = app.run_pipeline(self.data, lambda question, facts: [facts[-1]["evidence_id"]])
        self.assertEqual(len(output["research"]["answer"]["evidence_ids"]), 1)
        for selector in (lambda q, e: ["invented"], lambda q, e: "not a list"):
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, selector)

    def test_general_purpose_evidence_selection(self):
        facts = [{"evidence_id": "E1", "entity_id": "THING",
                  "attribute": "energy", "value": 10, "unit": "kWh", "source_ids": ["S"]}]
        answer = app.research_answer("What energy is required?", facts)
        self.assertEqual(answer["evidence_ids"], ["E1"])
        self.assertIn("energy=10", answer["finding"])

    def test_deterministic_and_nonmutating(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(before, self.data)

    def test_synthetic_marker_required(self):
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")],
                                 cwd=HERE, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage_errors(self):
        for args in ([], [str(HERE / "does_not_exist.json")]):
            process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                     cwd=HERE, text=True, capture_output=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_json_and_invalid_schema(self):
        for text in ("{", '{"schema_version":"1.0","schema_version":"2.0"}', "null",
                     json.dumps(dict(self.data, synthetic=False))):
            with patch("builtins.open", mock_open(read_data=text)), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(app.main(["virtual.json"]), 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
