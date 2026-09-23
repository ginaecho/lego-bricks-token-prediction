import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_two_step_journey(self):
        result = app.pipeline(self.document)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["action_id"] for s in result["journey"]["steps"]],
                         ["clean-orders", "build-dashboard"])
        self.assertTrue(result["journey"]["goals_met"])

    def test_exact_extractive_citations(self):
        result = app.run_research(self.document)
        self.assertEqual(len(result["findings"]), 2)
        for finding in result["findings"]:
            citation = finding["citation"]
            source = next(p for p in self.document["passages"] if p["id"] == citation["passage_id"])
            self.assertEqual(citation["quote"], source["text"][citation["start"]:citation["end"]])
            self.assertEqual(citation["source_uri"], source["source_uri"])
        self.assertGreater(result["findings"][0]["citation"]["start"], 0)

    def test_cross_stage_evidence_and_skills(self):
        result = app.pipeline(self.document)
        findings = {f["id"] for f in result["research"]["findings"]}
        first, second = result["journey"]["steps"]
        self.assertEqual(first["skills_after"], second["skills_before"])
        self.assertIn("clean-orders", second["skills_before"])
        for step in (first, second):
            self.assertTrue(set(step["finding_ids"]) <= findings)
        self.assertEqual([r["action_id"] for r in result["journey"]["recommendations"]], ["clean-orders"])

    def test_no_relevant_evidence(self):
        self.document["query"] = "astronomy"
        result = app.pipeline(self.document)
        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["journey"]["steps"], [])
        self.assertEqual(result["journey"]["recommendations"], [])

    def test_missing_prerequisites(self):
        self.document["profile"]["skills"] = []
        result = app.pipeline(self.document)
        self.assertEqual(result["status"], "no_journey")
        self.assertEqual(result["journey"]["recommendations"], [])

    def test_unretrieved_action_evidence_is_excluded(self):
        self.document["actions"][1]["evidence_passage_ids"].append("synthetic-unrelated")
        result = app.pipeline(self.document)
        self.assertEqual(result["status"], "no_journey")
        self.assertEqual(len(result["journey"]["recommendations"]), 1)

    def test_single_action_is_not_a_two_step_journey(self):
        self.document["actions"] = self.document["actions"][:1]
        self.document["profile"]["goals"] = ["clean-orders"]
        result = app.pipeline(self.document)
        self.assertEqual(result["status"], "no_journey")
        self.assertEqual(result["journey"]["steps"], [])

    def test_already_met_goals(self):
        self.document["profile"]["skills"].append("sales-dashboard")
        result = app.pipeline(self.document)
        self.assertEqual(result["journey"]["reason"], "goals_already_met")
        self.assertTrue(result["journey"]["goals_met"])

    def test_circular_prerequisites(self):
        self.document["actions"][0]["requires"] = ["sales-dashboard"]
        self.assertEqual(app.pipeline(self.document)["status"], "no_journey")

    def test_empty_collections(self):
        self.document["passages"] = []
        self.document["actions"] = []
        self.assertEqual(app.pipeline(self.document)["status"], "no_evidence")

    def test_input_order_does_not_change_output(self):
        expected = app.pipeline(self.document)
        self.document["passages"].reverse()
        self.document["actions"].reverse()
        self.assertEqual(app.pipeline(self.document), expected)

    def test_invalid_inputs(self):
        variants = [
            ("query", "  "), ("query", "the and"), ("schema_version", True),
            ("synthetic_data", False), ("profile", []), ("actions", {}), ("extra", 1)
        ]
        for key, value in variants:
            with self.subTest(key=key, value=value):
                invalid = copy.deepcopy(self.document)
                invalid[key] = value
                with self.assertRaises(app.ValidationError):
                    app.pipeline(invalid)

    def test_duplicate_ids_and_unknown_evidence(self):
        duplicate = copy.deepcopy(self.document)
        duplicate["passages"].append(copy.deepcopy(duplicate["passages"][0]))
        with self.assertRaises(app.ValidationError):
            app.pipeline(duplicate)
        self.document["actions"][0]["evidence_passage_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            app.pipeline(self.document)

    def test_tampered_research_is_rejected_before_journey(self):
        for key, value in (("quote", "fabricated"), ("start", -1),
                           ("source_uri", "synthetic:wrong"), ("end", True)):
            with self.subTest(key=key):
                research = app.run_research(self.document)
                research["findings"][0]["citation"][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_journey(self.document, research)

    def test_tampered_journey_is_rejected(self):
        research = app.run_research(self.document)
        journey = app.run_journey(self.document, research)
        journey["steps"][1]["skills_before"] = []
        with self.assertRaises(app.ValidationError):
            app.validate("journey", journey, self.document, research)
        journey = app.run_journey(self.document, research)
        journey["steps"][0]["finding_ids"] = ["invented"]
        with self.assertRaises(app.ValidationError):
            app.validate("journey", journey, self.document, research)

    def test_unicode_offsets_and_sentence_tie(self):
        self.document["passages"][0]["text"] = "Résumé café.  Marketplace sales analytics works! Marketplace sales analytics again."
        findings = app.run_research(self.document)["findings"]
        citation = next(f["citation"] for f in findings if f["citation"]["passage_id"] == "synthetic-cleaning")
        self.assertEqual(citation["start"], 14)
        self.assertEqual(citation["quote"], "Marketplace sales analytics works!")

    def test_cli_success(self):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                    str(ROOT / "example_input.json")], cwd=ROOT,
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)

    def test_cli_file_and_argument_errors(self):
        for arguments in ([], [str(ROOT / "missing-input.json")], ["one", "two"]):
            with self.subTest(arguments=arguments):
                completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                                           cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(json.loads(completed.stdout)["status"], "error")
                self.assertEqual(completed.stderr, "")

    def test_cli_malformed_json_without_scratch_files(self):
        for content in ("{broken", '{"query":"a","query":"b"}', '{"schema_version": NaN}', '[]', "9" * 5000):
            with self.subTest(content=content), patch.object(Path, "open", return_value=io.StringIO(content)):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["synthetic-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_error(self):
        with patch.object(Path, "open", side_effect=UnicodeError("invalid UTF-8")):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(app.main(["synthetic-input.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
