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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_support_grounded_in_policy(self):
        output = app.support_stage(self.data)
        self.assertEqual(output["support"]["citations"][0]["policy_id"], "POL-HOUSING")
        self.assertIn(output["support"]["answer"], self.data["policies"][0]["text"])
        self.assertFalse(output["support"]["needs_human"])

    def test_search_understands_synonym(self):
        output = app.run_pipeline(self.data)
        self.assertEqual(output["search"]["results"][0]["service_id"], "SVC-HOUSING")
        self.assertEqual(output["search"]["results"][0]["matched_terms"], ["housing"])

    def test_cross_stage_propagation(self):
        stage = app.support_stage(self.data)
        output = app.search_stage(stage)
        self.assertEqual(output["search"]["source_query_terms"], stage["support"]["query_terms"])
        self.assertEqual(output["search"]["source_policy_ids"], ["POL-HOUSING"])
        self.assertEqual(output["search"]["results"][0]["support_policy_ids"], ["POL-HOUSING"])
        self.assertEqual(output["benefits_application"], self.data["benefits_application"])
        self.assertEqual(output["request"]["case_number"], "SYN-CASE-0042")

    def test_no_mutation_and_deterministic(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_privacy_minimization(self):
        self.data["request"]["question"] += " Address: " + self.data["request"]["citizen"]["address"]
        output = app.run_pipeline(self.data)
        serialized = json.dumps(output)
        for private in self.data["request"]["citizen"].values():
            self.assertNotIn(private, serialized)
        self.assertIsNone(output["request"]["citizen"])
        self.assertIn("[redacted]", output["request"]["question"])

    def test_unknown_pii_patterns_redacted(self):
        self.data["request"]["question"] = "Rent help. Contact other@example.invalid or 202-555-0109. ID 123-45-6789."
        serialized = json.dumps(app.run_pipeline(self.data))
        for private in ("other@example.invalid", "202-555-0109", "123-45-6789"):
            self.assertNotIn(private, serialized)

    def test_public_policy_cannot_leak_citizen(self):
        self.data["policies"][0]["text"] += " Mira Exampleperson"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_unsupported_question_escalates(self):
        self.data["request"]["question"] = "Astronomy observatory"
        output = app.run_pipeline(self.data)
        self.assertTrue(output["support"]["needs_human"])
        self.assertEqual(output["support"]["citations"], [])
        self.assertEqual(output["search"]["results"], [])

    def test_empty_catalog(self):
        self.data["policies"] = []
        self.data["services"] = []
        output = app.run_pipeline(self.data)
        self.assertTrue(output["support"]["needs_human"])
        self.assertEqual(output["search"]["results"], [])

    def test_only_stop_words(self):
        self.data["request"]["question"] = "Can I please"
        output = app.run_pipeline(self.data)
        self.assertEqual(output["support"]["query_terms"], [])
        self.assertEqual(output["search"]["results"], [])

    def test_tie_breaking_is_explainable(self):
        service = copy.deepcopy(self.data["services"][0])
        service["id"] = "SVC-AAA"
        self.data["services"].append(service)
        output = app.run_pipeline(self.data)
        self.assertEqual([r["service_id"] for r in output["search"]["results"]],
                         ["SVC-AAA", "SVC-HOUSING"])
        self.assertEqual(output["search"]["results"][0]["score"], 3)
        self.assertIn("housing", output["search"]["results"][0]["explanation"])

    def test_invalid_input_variations(self):
        for field, value in (("schema_version", "9"), ("fixture_label", "real"),
                             ("status", "ok"), ("request", []), ("policies", "text")):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.data)
                bad[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)

    def test_unknown_fields_rejected(self):
        self.data["request"]["national_id"] = "private"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_broken_policy_reference_rejected(self):
        self.data["services"][0]["policy_ids"] = ["POL-MISSING"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_duplicate_ids_rejected(self):
        self.data["policies"].append(copy.deepcopy(self.data["policies"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_plain_language_constraint(self):
        sentence = " ".join(["word"] * 29)
        self.data["policies"][0]["plain_language_summary"] = sentence
        self.data["policies"][0]["text"] = sentence
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_ungrounded_summary_rejected(self):
        self.data["policies"][0]["plain_language_summary"] = "Everyone is approved."
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_no_eligibility_decision(self):
        output = app.run_pipeline(self.data)
        self.assertIn("not a benefits decision", output["support"]["notice"])
        self.data["benefits_application"]["status"] = "approved"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_tampered_handoff_rejected(self):
        for field, value in (("query_terms", ["food"]), ("answer", "You qualify."),
                             ("citations", [])):
            with self.subTest(field=field):
                stage = app.support_stage(self.data)
                stage["support"][field] = value
                with self.assertRaises(app.ValidationError):
                    app.search_stage(stage)

    def test_stage_order_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.search_stage(self.data)

    def test_final_output_validation(self):
        output = app.run_pipeline(self.data)
        app.validate(output, "ok")
        output["search"]["results"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.validate(output, "ok")

    def test_citizen_request_food_route(self):
        self.data["request"]["kind"] = "citizen_service_request"
        self.data["request"]["question"] = "I need groceries"
        output = app.run_pipeline(self.data)
        self.assertEqual(output["search"]["results"][0]["service_id"], "SVC-FOOD")
        self.assertEqual(output["support"]["citations"][0]["policy_id"], "POL-FOOD")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_file_error_and_usage(self):
        for args in ([], [str(ROOT / "missing.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")
            self.assertNotIn("missing.json", result.stdout)

    def test_cli_bad_json_and_validation_errors(self):
        for raw in ("{", '{"status":"input","status":"ok"}', "NaN", "[]",
                    json.dumps({**self.data, "schema_version": "invalid"})):
            with self.subTest(raw=raw[:20]):
                output = io.StringIO()
                with patch("builtins.open", return_value=io.StringIO(raw)), patch("sys.stdout", output):
                    self.assertEqual(app.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
