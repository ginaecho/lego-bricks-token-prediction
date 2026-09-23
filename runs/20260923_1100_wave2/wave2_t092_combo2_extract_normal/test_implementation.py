import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    return {
        "version": 1, "fixture_label": "SYNTHETIC unittest fixture",
        "documents": [{"id": "d1", "text": "Project: Aurora\nOwner: Mira\nAurora delivery approved.\n"},
                      {"id": "d2", "text": "Aurora delivery delayed.\n"}],
        "fields": [{"name": "project", "label": "Project", "required": True},
                   {"name": "owner", "label": "Owner", "required": False}],
        "queries": [{"id": "q1", "text": "delivery", "fields": ["project"]}],
        "top_k": 3
    }


class PipelineTests(unittest.TestCase):
    def test_extraction_values_and_offsets(self):
        state = fixture()
        result = app.extract(state)
        match = result["extraction"][0]["fields"]["project"][0]
        self.assertEqual(match["value"], "Aurora")
        self.assertEqual(match["source"],
                         {"document_id": "d1", "start": 9, "end": 15, "quote": "Aurora"})

    def test_missing_required_and_optional(self):
        record = app.extract(fixture())["extraction"][1]
        self.assertFalse(record["complete"])
        self.assertEqual(record["missing_fields"],
                         [{"name": "project", "required": True},
                          {"name": "owner", "required": False}])

    def test_optional_missing_keeps_complete(self):
        state = fixture()
        state["documents"][0]["text"] = "Project: Aurora"
        self.assertTrue(app.extract(state)["extraction"][0]["complete"])

    def test_unicode_crlf_whitespace_and_repeated_labels(self):
        state = fixture()
        state["documents"][0]["text"] = "🌟\r\n  pRoJeCt :  Étoile  \r\nProject: Lune\r\nProject:   \r\n"
        result = app.extract(state)["extraction"][0]["fields"]["project"]
        self.assertEqual([m["value"] for m in result], ["Étoile", "Lune"])
        for match in result:
            source = match["source"]
            self.assertEqual(state["documents"][0]["text"][source["start"]:source["end"]],
                             match["value"])

    def test_ranking_citations_and_stable_ties(self):
        state = fixture()
        findings = app.run_pipeline(state)["findings"][0]["passages"]
        self.assertEqual([p["score"] for p in findings], [3, 3, 1])
        self.assertEqual([p["source"]["document_id"] for p in findings], ["d1", "d2", "d1"])
        documents = {d["id"]: d["text"] for d in state["documents"]}
        for passage in findings:
            source = passage["source"]
            self.assertEqual(passage["text"], source["quote"])
            self.assertEqual(source["quote"],
                             documents[source["document_id"]][source["start"]:source["end"]])

    def test_extraction_drives_research_expansion(self):
        state = fixture()
        state["queries"][0]["text"] = "overview"
        result = app.run_pipeline(state)
        self.assertEqual(result["findings"][0]["status"], "found")
        self.assertEqual(result["findings"][0]["expanded_from"][0]["value"], "Aurora")
        state["fields"][0]["label"] = "Missing label"
        result = app.run_pipeline(state)
        self.assertEqual(result["findings"][0]["status"], "no_evidence")
        self.assertEqual(result["findings"][0]["expanded_from"], [])

    def test_tampered_handoff_is_rejected(self):
        extracted = app.extract(fixture())
        extracted["extraction"][0]["fields"]["project"][0]["source"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.research(extracted)

    def test_tampered_final_result_is_rejected(self):
        result = app.run_pipeline(fixture())
        result["findings"][0]["passages"][0]["text"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.validate(result, "researched")

    def test_handoff_and_output_types_are_strict(self):
        extracted = app.extract(fixture())
        extracted["extraction"][0]["complete"] = 1
        with self.assertRaises(app.ValidationError):
            app.research(extracted)
        result = app.run_pipeline(fixture())
        result["findings"][0]["passages"][0]["score"] = 3.0
        with self.assertRaises(app.ValidationError):
            app.validate(result, "researched")

    def test_empty_document_and_no_evidence(self):
        state = fixture()
        state["documents"] = [{"id": "empty", "text": ""}]
        result = app.run_pipeline(state)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["findings"][0]["passages"], [])

    def test_no_field_expansion_and_top_k(self):
        state = fixture()
        state["queries"][0]["fields"] = []
        state["top_k"] = 1
        result = app.run_pipeline(state)["findings"][0]
        self.assertEqual(result["expanded_from"], [])
        self.assertEqual(len(result["passages"]), 1)
        self.assertEqual(result["passages"][0]["score"], 2)

    def test_invalid_schema_variations(self):
        cases = []
        for key, value in [("top_k", True), ("top_k", 0), ("version", True),
                           ("documents", []), ("fields", "wrong"), ("extra", 1)]:
            state = fixture()
            state[key] = value
            cases.append(state)
        state = fixture()
        state["queries"][0]["fields"] = ["unknown"]
        cases.append(state)
        state = fixture()
        state["documents"][1]["id"] = "d1"
        cases.append(state)
        state = fixture()
        state["fields"][1]["label"] = "PROJECT"
        cases.append(state)
        state = fixture()
        state["fields"][0]["required"] = "yes"
        cases.append(state)
        cases.extend([None, [], 42])
        for state in cases:
            with self.subTest(state=state), self.assertRaises(app.ValidationError):
                app.run_pipeline(state)

    def test_determinism_and_no_input_mutation(self):
        state = fixture()
        before = copy.deepcopy(state)
        self.assertEqual(app.run_pipeline(state), app.run_pipeline(state))
        self.assertEqual(state, before)

    def test_exact_label_not_substring(self):
        state = fixture()
        state["documents"][0]["text"] = "Old Project: Fake\nProject summary: Fake\nProject: Real"
        matches = app.extract(state)["extraction"][0]["fields"]["project"]
        self.assertEqual([m["value"] for m in matches], ["Real"])

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        app.validate(json.loads(completed.stdout), "researched")

    def test_cli_missing_file_and_usage(self):
        for arguments in [[], ["nonexistent-input.json"], ["one", "two"]]:
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_invalid_json_and_schema_without_scratch_files(self):
        for contents in ['{', '{"version":1,"version":1}', '{"top_k":NaN}', '[]']:
            output = io.StringIO()
            with patch.object(Path, "open", return_value=io.StringIO(contents)):
                with redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
