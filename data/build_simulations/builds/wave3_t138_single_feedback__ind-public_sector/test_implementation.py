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


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def single(self, text):
        return {"schema_version": 1, "synthetic": True,
                "records": [{"entity_type": "citizen_service_request", "text": text}]}

    def test_normal_example(self):
        result = app.analyze(self.data)
        self.assertEqual(result["summary"]["unique_feedback"], 3)
        self.assertEqual(result["summary"]["duplicate_records"], 1)
        self.assertEqual({t["theme_id"] for t in result["themes"]},
                         {"access", "clarity", "delay", "positive"})

    def test_deterministic_and_does_not_mutate(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.analyze(self.data), app.analyze(self.data))
        self.assertEqual(self.data, before)

    def test_excerpts_trace_to_exact_input(self):
        output = app.analyze(self.data)
        for theme in output["themes"]:
            for item in theme["supporting_excerpts"]:
                source = self.data["records"][int(item["record_ref"].split("_")[1]) - 1]["text"]
                self.assertEqual(source[item["start"]:item["end"]], item["excerpt"])

    def test_no_private_values_in_output(self):
        self.data["records"][0]["text"] += " Email zora@example.invalid or call 555-010-9876; SSN 123-45-6789."
        text = json.dumps(app.analyze(self.data))
        for forbidden in ("Zora", "Exampleperson", "Fiction Walk", "SYN-CASE-001",
                          "zora@example.invalid", "555-010-9876", "123-45-6789"):
            self.assertNotIn(forbidden, text)

    def test_theme_word_inside_private_field_is_not_evidence(self):
        data = self.single("Helpful lives at 1 Clear Lane.")
        data["records"][0]["citizen"] = {"name": "Helpful", "address": "1 Clear Lane"}
        self.assertEqual(app.analyze(data)["themes"], [])

    def test_unclassified_and_whole_word_matching(self):
        output = app.analyze(self.single("Unclearance is a fictional term."))
        self.assertEqual(output["themes"], [])
        self.assertEqual(output["summary"]["unclassified_unique_feedback"], 1)

    def test_private_only_records_not_merged(self):
        data = self.single("111-22-3333")
        data["records"].append({"entity_type": "benefits_application", "text": "999-88-7777"})
        self.assertEqual(app.analyze(data)["summary"]["unique_feedback"], 2)

    def test_counts_once_per_unique_feedback(self):
        data = self.single("Wait, wait; delayed!")
        data["records"].append({"entity_type": "policy_document", "text": "WAIT wait delayed"})
        output = app.analyze(data)
        self.assertEqual(output["themes"][0]["unique_feedback_count"], 1)
        self.assertEqual(output["deduplication"][0]["record_refs"],
                         ["record_0001", "record_0002"])

    def test_invalid_top_level_and_industry_label(self):
        for value in ([], {}, None, {"schema_version": True, "synthetic": True, "records": []}):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.analyze(value)
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.analyze(self.data)

    def test_invalid_records(self):
        for record in (None, {}, {"entity_type": [], "text": "wait"},
                       {"entity_type": "private", "text": "wait"},
                       {"entity_type": "policy_document", "text": " "},
                       {"entity_type": "policy_document", "text": "wait", "ssn": "private"},
                       {"entity_type": "policy_document", "text": "wait", "citizen": {}},
                       {"entity_type": "policy_document", "text": "wait", "case_number": 5}):
            self.data["records"] = [record]
            with self.subTest(record=record), self.assertRaises(app.ValidationError):
                app.analyze(self.data)

    def test_limits(self):
        for text in ("x" * 10001, "bad\x00text"):
            with self.assertRaises(app.ValidationError):
                app.analyze(self.single(text))
        for records in ([], [{"entity_type": "policy_document", "text": "wait"}] * 201):
            self.data["records"] = records
            with self.assertRaises(app.ValidationError):
                app.analyze(self.data)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], ["missing-input.json"], ["example_input.json", "extra"]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_malformed_and_duplicate_json(self):
        for content in ('{broken', '{"synthetic":true,"synthetic":false}', '{"x":NaN}',
                        '{"schema_version":1,"synthetic":false,"records":[]}'):
            with patch.object(Path, "open", return_value=io.StringIO(content)):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    code = app.main(["example_input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
