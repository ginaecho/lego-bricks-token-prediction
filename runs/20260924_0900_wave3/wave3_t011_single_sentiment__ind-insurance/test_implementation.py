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


HERE = Path(__file__).resolve().parent


class ImplementationTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def reject(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.doc)

    def test_all_entities_and_formats(self):
        result = app.run(self.doc)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([r["record_id"] for r in result["results"]],
                         ["REC-CLAIM-01", "REC-UW-01", "REC-POLICY-01"])
        self.assertEqual(result["results"][0]["sentiment"]["score"], -5)
        self.assertEqual(result["results"][0]["priority"]["score"], 35)

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.doc)
        self.assertEqual(app.run(self.doc), app.run(self.doc))
        self.assertEqual(self.doc, original)

    def test_transparent_negation_neutral_and_clipping(self):
        self.assertEqual(app.sentiment("not helpful")["score"], -2)
        self.assertEqual(app.sentiment("not. helpful")["score"], 2)
        self.assertEqual(app.sentiment("not very helpful")["score"], 2)
        self.assertEqual(app.sentiment("ordinary information")["score"], 0)
        self.assertEqual(app.sentiment("excellent excellent excellent")["score"], 5)
        match = app.sentiment("not poor")["matches"][0]
        self.assertEqual(match, {"term": "poor", "base_weight": -2,
                                 "negated": True, "contribution": 2})

    def test_severity_dominates_sentiment(self):
        self.doc["records"][0]["severity"] = "critical"
        self.doc["records"][1]["severity"] = "high"
        self.assertEqual(app.run(self.doc)["results"][0]["record_id"], "REC-POLICY-01")

    def test_stable_ties(self):
        first = self.doc["records"][0]
        second = copy.deepcopy(first)
        first["record_id"], second["record_id"] = "REC-Z", "REC-A"
        self.doc["records"] = [first, second]
        self.assertEqual(app.run(self.doc)["results"][0]["record_id"], "REC-A")

    def test_output_minimizes_personal_data(self):
        output = json.dumps(app.run(self.doc))
        for forbidden in ["Mira", "Leon", "Nora", "Imaginary", "Fabricated",
                          "1SYNTHET1C0000001", "SYN-HOLDER", "inspection report"]:
            self.assertNotIn(forbidden, output)

    def test_claims_require_reasons_even_pending(self):
        self.doc["records"][1]["claim_handling"]["reasons"] = []
        self.reject()

    def test_no_adverse_automated_claim_decision(self):
        self.doc["records"][1]["claim_handling"]["decision"] = "denied"
        claim = app.run(self.doc)["results"][0]["claim_handling"]
        self.assertEqual(claim["action"], "human_review")
        self.assertNotIn("decision", claim)

    def test_hidden_and_protected_underwriting_factors(self):
        factor = self.doc["records"][2]["underwriting_factors"][0]
        factor["disclosed"] = False
        self.reject()
        factor["disclosed"], factor["name"] = True, "ethnicity"
        self.reject()

    def test_underwriting_factors_do_not_determine_queue(self):
        baseline = app.run(self.doc)["results"][1]["priority"]
        self.doc["records"][2]["underwriting_factors"][0]["value"] = 200
        self.assertEqual(app.run(self.doc)["results"][1]["priority"], baseline)

    def test_minimization_rejects_extra_policyholder_fields(self):
        self.doc["records"][0]["submission"]["content"]["ACORD"]["email"] = "synthetic"
        self.reject()

    def test_invalid_envelopes(self):
        for value in [None, [], {}, {"schema_version": "1.0", "synthetic": True, "records": []}]:
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run(value)
        self.doc["synthetic"] = False
        self.reject()

    def test_duplicate_ids_and_invalid_severity(self):
        self.doc["records"][1]["record_id"] = self.doc["records"][0]["record_id"]
        self.reject()
        self.doc["records"][1]["record_id"] = "REC-CLAIM-01"
        self.doc["records"][0]["severity"] = []
        self.reject()

    def test_invalid_source_values(self):
        payload = self.doc["records"][0]["submission"]["content"]["ACORD"]
        for value in [True, -1, float("inf"), float("nan"), "2"]:
            payload["loss_amount"] = value
            self.reject()
        payload["loss_amount"] = 10
        payload["loss_date"] = "2025-02-30"
        self.reject()

    def test_xml_rejects_entities_and_duplicate_fields(self):
        source = self.doc["records"][2]["submission"]
        for value in ["<ACORD>", '<!DOCTYPE ACORD [<!ENTITY x "abc">]><ACORD/>',
                      "<ACORD><Feedback>a</Feedback><Feedback>b</Feedback></ACORD>"]:
            source["content"] = value
            self.reject()

    def test_claim_form_rejects_duplicate_fields(self):
        self.doc["records"][1]["submission"]["content"] += "\nFeedback: helpful"
        self.reject()

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")],
                                 capture_output=True, text=True, cwd=HERE)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in [[], ["absent-input.json"]]:
            process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                                     capture_output=True, text=True, cwd=HERE)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_schema_without_extra_files(self):
        for content in ["{", '{"x":1,"x":2}', '{"schema_version":"1.0"}', "null"]:
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=content), contextlib.redirect_stdout(output):
                code = app.main([str(HERE / "example_input.json")])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
