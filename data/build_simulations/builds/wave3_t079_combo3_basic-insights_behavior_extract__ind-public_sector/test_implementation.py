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
        self.input = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_full_pipeline(self):
        result = app.run(self.input)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stage"], "extract")
        self.assertEqual(len(result["data"]["extractions"]), 2)

    def test_theme_counts_and_actions(self):
        themes = {t["name"]: t for t in app.insights(self.input)["data"]["themes"]}
        self.assertEqual(themes["paperwork"]["count"], 2)
        self.assertEqual(themes["access"]["count"], 2)
        self.assertTrue(all(t["action"].endswith(".") for t in themes.values()))

    def test_unclassified_feedback(self):
        self.input["service_requests"][0]["feedback"] = "Thank you."
        themes = app.insights(self.input)["data"]["themes"]
        self.assertTrue(any(t["name"] == "other" for t in themes))

    def test_private_details_do_not_propagate(self):
        self.input["policy_documents"][0]["text"] += "\nContact: Invented Mira Example, mira@example.invalid"
        self.input["policy_documents"][0]["title"] += " Invented Mira Example"
        output = json.dumps(app.run(self.input))
        for private in app.secrets(self.input):
            self.assertNotIn(private, output)
        self.assertIn("[REDACTED]", output)

    def test_extra_contact_patterns(self):
        safe = app.redact("Other: person@fiction.invalid, +1 (202) 555-0199; date 2026-10-01", ())
        self.assertNotIn("person@", safe)
        self.assertNotIn("555", safe)
        self.assertIn("2026-10-01", safe)

    def test_recency_purchase_weights(self):
        result = app.behavior(app.insights(self.input))["data"]
        scores = {r["document"]["id"]: r["score"] for r in result["ranked_documents"]}
        self.assertEqual(scores["SYN-P-002"], 3.5)
        self.assertEqual(scores["SYN-P-001"], 4)

    def test_other_citizens_history_is_not_used(self):
        before = app.run(self.input)
        self.input["events"] = [e for e in self.input["events"] if e["citizen_id"] != "SYN-C-002"]
        self.assertEqual(before, app.run(self.input))

    def test_cold_start(self):
        self.input["events"] = []
        data = app.run(self.input)["data"]
        self.assertTrue(data["cold_start"])
        self.assertIn("No activity", data["rankings"][0]["explanation"])
        self.assertEqual(data["rankings"][0]["document_id"], "SYN-P-001")

    def test_ties_are_stable(self):
        self.input["events"] = []
        self.input["service_requests"] = []
        self.input["benefits_applications"] = []
        self.input["policy_documents"].reverse()
        data = app.run(self.input)["data"]
        self.assertEqual([r["document_id"] for r in data["rankings"]], ["SYN-P-001", "SYN-P-002"])

    def test_cross_stage_order_and_theme_propagation(self):
        first = app.insights(self.input)
        first["data"]["themes"][0]["action"] = "Ask the service team for help."
        second = app.behavior(first)
        last = app.extract(second)
        self.assertEqual(first["data"]["themes"], last["data"]["themes"])
        self.assertEqual([r["document"]["id"] for r in second["data"]["ranked_documents"]],
                         [r["document_id"] for r in last["data"]["extractions"]])

    def test_schema_propagates(self):
        first = app.insights(self.input)
        first["data"]["extraction_schema"] = [{"name": "help", "label": "Help", "type": "string", "required": True}]
        last = app.extract(app.behavior(first))
        self.assertEqual(set(last["data"]["extractions"][0]["fields"]), {"help"})

    def test_extraction_values_and_spans(self):
        self.input["policy_documents"][0]["text"] = "政策 — synthetic guide\n" + self.input["policy_documents"][0]["text"]
        result = app.run(self.input)
        first = result["data"]["extractions"][0]
        self.assertEqual(first["fields"]["response_days"]["value"], 14)
        self.assertEqual(first["fields"]["effective_date"]["value"], "2026-10-01")
        for record in result["data"]["extractions"]:
            for field in record["fields"].values():
                start, end = field["source_span"]
                self.assertEqual(record["source_text"][start:end], field["source_text"])

    def test_missing_required_and_optional(self):
        result = app.run(self.input)["data"]["extractions"][1]
        self.assertIn({"name": "effective_date", "required": True}, result["missing_fields"])
        self.assertIn({"name": "appeal", "required": False}, result["missing_fields"])

    def test_invalid_typed_value(self):
        self.input["policy_documents"][0]["text"] = "Service: Example\nResponse days: many\nEffective date: 2026-02-30"
        first = app.run(self.input)["data"]["extractions"][0]
        self.assertEqual({f["name"] for f in first["invalid_fields"]}, {"response_days", "effective_date"})

    def test_duplicate_field_not_silently_chosen(self):
        self.input["policy_documents"][0]["text"] += "\nResponse days: 99"
        first = app.run(self.input)["data"]["extractions"][0]
        self.assertNotIn("response_days", first["fields"])
        self.assertEqual(first["invalid_fields"][0]["name"], "response_days")

    def test_redacted_fields_are_not_extracted(self):
        self.input["policy_documents"][0]["text"] = "Service: Invented Mira Example"
        first = app.run(self.input)["data"]["extractions"][0]
        self.assertEqual(first["fields"], {})
        self.assertEqual(first["invalid_fields"][0]["name"], "service")

    def test_empty_policy_collection(self):
        self.input["policy_documents"] = []
        self.input["events"] = []
        self.assertEqual(app.run(self.input)["data"]["extractions"], [])

    def test_determinism_and_no_input_mutation(self):
        saved = copy.deepcopy(self.input)
        self.assertEqual(app.run(self.input), app.run(self.input))
        self.assertEqual(saved, self.input)

    def test_future_event_rejected(self):
        self.input["events"][0]["at"] = "2027-01-01T00:00:00Z"
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_unzoned_time_rejected(self):
        self.input["now"] = "2026-09-24T12:00:00"
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_unknown_event_document_rejected(self):
        self.input["events"][0]["document_id"] = "SYN-P-999"
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_duplicate_id_rejected(self):
        self.input["policy_documents"].append(copy.deepcopy(self.input["policy_documents"][0]))
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_synthetic_guard(self):
        self.input["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_nontraceable_contacts_required(self):
        self.input["personas"][0]["pii"]["email"] = "person@example.com"
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_unknown_input_fields_rejected(self):
        self.input["eligibility_decision"] = "approved"
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_invalid_schema_type_rejected(self):
        self.input["extraction_schema"][0]["type"] = "python"
        with self.assertRaises(app.ValidationError):
            app.run(self.input)

    def test_handoff_validation(self):
        first = app.insights(self.input)
        first["stage"] = "extract"
        with self.assertRaises(app.ValidationError):
            app.behavior(first)

    def test_ranking_validation(self):
        second = app.behavior(app.insights(self.input))
        second["data"]["ranked_documents"][0]["score"] = float("nan")
        with self.assertRaises(app.ValidationError):
            app.extract(second)

    def test_invalid_schema_handoff(self):
        first = app.insights(self.input)
        first["data"]["extraction_schema"][0]["type"] = "unsafe"
        with self.assertRaises(app.ValidationError):
            app.behavior(first)

    def test_invalid_document_handoff(self):
        second = app.behavior(app.insights(self.input))
        second["data"]["ranked_documents"][0]["document"]["text"] = None
        with self.assertRaises(app.ValidationError):
            app.extract(second)

    def test_span_validation(self):
        last = app.run(self.input)
        last["data"]["extractions"][0]["fields"]["service"]["source_span"] = [0, 1]
        with self.assertRaises(app.ValidationError):
            app.validate(last, "extract")

    def test_plain_language_and_review_explanations(self):
        result = app.run(self.input)
        self.assertIn("not decide benefit eligibility", result["notice"])
        for ranked in result["data"]["rankings"]:
            self.assertIn("30 days", ranked["explanation"])
            self.assertIn("Feedback match", ranked["explanation"])

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "does-not-exist.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_duplicate_keys(self):
        for data in ("{", '{"schema_version":"1.0","schema_version":"1.0"}', "[]", '{"synthetic":false}'):
            with self.subTest(data=data), patch("builtins.open", return_value=io.StringIO(data)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(app.main(["input.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_argument_error(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(app.main([]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
