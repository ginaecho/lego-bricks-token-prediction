"""All fixtures are synthetic; no networks, providers or third-party packages."""

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


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def test_full_pipeline_and_shared_schema(self):
        result = self.run_data()
        self.assertEqual(result["status"], "ok")
        for stage in ("extraction", "feedback", "journey"):
            self.assertEqual(result[stage]["schema_version"], result["schema_version"])
            self.assertEqual(result[stage]["stage"], stage)
        self.assertEqual(len(result["journey"]["steps"]), 2)
        self.assertIs(app.validate("result", result, self.data), result)

    def test_schema_extraction_and_integer_value(self):
        document = self.run_data()["extraction"]["documents"][0]
        self.assertEqual(document["fields"]["rating"]["value"], 2)
        self.assertEqual(document["fields"]["customer"]["value"], "Synthetic Ada")
        self.assertEqual(document["missing_fields"], [])

    def test_source_spans_unicode_crlf_and_whitespace(self):
        self.data["documents"][0]["text"] = (
            "Customer:\t  Synthetic Zoë 🧱 \r\n"
            "Feedback:  Setup is confusing!  \r\nGoal: Learn\r\nRating: +03\r\n")
        result = self.run_data()
        originals = {doc["id"]: doc["text"] for doc in self.data["documents"]}
        for doc in result["extraction"]["documents"]:
            for field in doc["fields"].values():
                source = field["source"]
                self.assertEqual(originals[source["document_id"]][source["start"]:source["end"]],
                                 source["text"])
        self.assertEqual(result["extraction"]["documents"][0]["fields"]["rating"]["value"], 3)

    def test_missing_required_and_optional_fields(self):
        result = self.run_data()
        last = result["extraction"]["documents"][-1]
        self.assertEqual(last["missing_fields"], ["feedback", "rating"])
        self.assertEqual(last["missing_required_fields"], ["feedback"])
        self.assertEqual(result["feedback"]["skipped_documents"], [
            {"document_id": "synthetic-004", "reason": "missing_feedback"}])

    def test_blank_field_does_not_consume_next_line(self):
        self.data["documents"] = [{"id": "blank", "text": "Feedback:   \nGoal: Learn\n"}]
        doc = self.run_data()["extraction"]["documents"][0]
        self.assertNotIn("feedback", doc["fields"])
        self.assertEqual(doc["fields"]["goal"]["value"], "Learn")
        self.assertEqual(doc["missing_required_fields"], ["customer", "feedback"])

    def test_duplicate_field_label_rejected(self):
        self.data["documents"][0]["text"] += "Feedback: A second value\n"
        with self.assertRaisesRegex(app.ValidationError, "duplicate field"):
            self.run_data()

    def test_invalid_extracted_integer_rejected(self):
        self.data["documents"][0]["text"] = "Feedback: setup\nRating: 2.5\n"
        with self.assertRaisesRegex(app.ValidationError, "invalid integer"):
            self.run_data()

    def test_custom_schema_field_propagates(self):
        self.data["field_schema"].append(
            {"name": "region", "label": "Region", "type": "string", "required": False})
        self.data["documents"][0]["text"] += "Region: Synthetic North\n"
        doc = self.run_data()["extraction"]["documents"][0]
        self.assertEqual(doc["fields"]["region"]["value"], "Synthetic North")

    def test_dedup_preserves_every_source_and_context(self):
        feedback = self.run_data()["feedback"]
        self.assertEqual(len(feedback["items"]), 2)
        self.assertEqual(feedback["duplicate_count"], 1)
        occurrences = feedback["items"][0]["occurrences"]
        self.assertEqual([entry["customer"] for entry in occurrences],
                         ["Synthetic Ada", "Synthetic Ben"])
        self.assertEqual([entry["source"]["document_id"] for entry in occurrences],
                         ["synthetic-001", "synthetic-002"])

    def test_unicode_normalization_deduplicates(self):
        self.data["documents"] = [
            {"id": "a", "text": "Feedback: ＳＥＴＵＰ guide!\n"},
            {"id": "b", "text": "Feedback: setup   guide.\n"},
        ]
        self.assertEqual(self.run_data()["feedback"]["duplicate_count"], 1)

    def test_theme_excerpts_are_traceable_to_both_duplicate_sources(self):
        result = self.run_data()
        theme = result["feedback"]["themes"][0]
        self.assertEqual(theme["feedback_ids"], ["F0001"])
        self.assertEqual(len(theme["supporting_excerpts"]), 2)
        originals = {doc["id"]: doc["text"] for doc in self.data["documents"]}
        for excerpt in theme["supporting_excerpts"]:
            self.assertEqual(excerpt["feedback_id"], "F0001")
            self.assertEqual(originals[excerpt["document_id"]][excerpt["start"]:excerpt["end"]],
                             excerpt["text"])

    def test_keywords_match_whole_words_not_substrings(self):
        self.data["documents"] = [{"id": "x", "text": "Feedback: presetup analyticsish\n"}]
        feedback = self.run_data()["feedback"]
        self.assertEqual(feedback["themes"], [])
        self.assertEqual(feedback["unmatched_feedback_ids"], ["F0001"])

    def test_punctuation_only_feedback_skipped(self):
        self.data["documents"] = [{"id": "x", "text": "Feedback: !!!\n"}]
        feedback = self.run_data()["feedback"]
        self.assertEqual(feedback["items"], [])
        self.assertEqual(feedback["skipped_documents"][0]["reason"], "no_word_content")

    def test_prerequisite_aware_two_step_plan(self):
        journey = self.run_data()["journey"]
        self.assertEqual([step["action_id"] for step in journey["steps"]],
                         ["01-profile", "02-guide"])
        self.assertEqual([action["action_id"] for action in journey["next_actions"]],
                         ["01-profile"])
        done = set()
        for step in journey["steps"]:
            self.assertTrue(set(step["prerequisites"]) <= done)
            self.assertNotIn(step["action_id"], done)
            done.add(step["action_id"])
        self.assertEqual(journey["steps"][1]["supporting_feedback_ids"], ["F0001"])
        self.assertEqual(journey["steps"][1]["score"], 1)

    def test_completed_action_unlocks_relevant_journey(self):
        self.data["profile"]["completed_action_ids"] = ["01-profile"]
        journey = self.run_data()["journey"]
        self.assertEqual([step["action_id"] for step in journey["steps"]],
                         ["02-guide", "03-report"])
        self.assertNotIn("01-profile", [step["action_id"] for step in journey["steps"]])

    def test_feedback_change_propagates_into_journey(self):
        self.data["documents"] = [
            {"id": "report-only", "text": "Feedback: Where is the sales report?\n"}]
        result = self.run_data()
        self.assertEqual([theme["id"] for theme in result["feedback"]["themes"]], ["reporting"])
        self.assertEqual(result["journey"]["steps"][1]["action_id"], "03-report")
        self.assertEqual(result["journey"]["steps"][1]["supporting_feedback_ids"], ["F0001"])

    def test_no_documents_has_explicit_zero_evidence_fallback(self):
        self.data["documents"] = []
        result = self.run_data()
        self.assertEqual(result["extraction"]["documents"], [])
        self.assertEqual(result["feedback"]["items"], [])
        self.assertEqual(result["journey"]["status"], "ready")
        self.assertTrue(all(step["score"] == 0 for step in result["journey"]["steps"]))

    def test_insufficient_remaining_actions_blocks_without_partial_journey(self):
        self.data["profile"]["completed_action_ids"] = ["01-profile", "02-guide", "03-report"]
        journey = self.run_data()["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual(journey["steps"], [])
        self.assertEqual(len(journey["next_actions"]), 1)
        self.assertIsNotNone(journey["reason"])

    def test_no_actions_or_all_completed_are_blocked(self):
        for empty in (True, False):
            with self.subTest(empty_catalog=empty):
                data = copy.deepcopy(self.data)
                if empty:
                    data["actions"] = []
                else:
                    data["profile"]["completed_action_ids"] = [a["id"] for a in data["actions"]]
                result = app.run_pipeline(data)
                self.assertEqual(result["journey"]["status"], "blocked")
                self.assertEqual(result["journey"]["next_actions"], [])

    def test_cycle_and_unknown_prerequisite_rejected(self):
        for dependency in ("02-guide", "unknown"):
            with self.subTest(dependency=dependency):
                self.data["actions"][0]["prerequisites"] = [dependency]
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_invalid_input_shapes_and_references(self):
        mutations = [
            lambda d: d.update(schema_version="2.0"),
            lambda d: d.update(unexpected=True),
            lambda d: d.update(documents={}),
            lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
            lambda d: d["field_schema"][0].update(required=1),
            lambda d: d["field_schema"][1].update(type="integer"),
            lambda d: d["field_schema"][0].update(label="Bad:Label"),
            lambda d: d["themes"][0].update(keywords=["!!!"]),
            lambda d: d["actions"][0].update(theme_ids=["unknown"]),
            lambda d: d["profile"].update(completed_action_ids=["unknown"]),
            lambda d: d["actions"].append(copy.deepcopy(d["actions"][0])),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                mutation(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        for invalid in (None, [], 2, "string", True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(invalid)

    def test_tampered_extraction_is_rejected_at_feedback_boundary(self):
        extraction = app.extract(self.data)
        extraction["documents"][0]["fields"]["feedback"]["source"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.analyze_feedback(extraction, self.data)

    def test_tampered_feedback_is_rejected_at_journey_boundary(self):
        extraction = app.extract(self.data)
        feedback = app.analyze_feedback(extraction, self.data)
        feedback["themes"][0]["supporting_excerpts"][0]["text"] = "Fabricated excerpt"
        with self.assertRaises(app.ValidationError):
            app.recommend_journey(feedback, extraction, self.data)

    def test_tampered_journey_is_rejected_by_shared_validation(self):
        result = self.run_data()
        result["journey"]["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate("result", result, self.data)

    def test_output_validation_rejects_boolean_integer_confusion(self):
        result = self.run_data()
        result["feedback"]["duplicate_count"] = True
        with self.assertRaises(app.ValidationError):
            app.validate("result", result, self.data)

    def test_deterministic_and_no_input_mutation(self):
        before = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(first, self.run_data())
        self.assertEqual(self.data, before)

    def invoke_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"), *arguments],
            cwd=HERE, capture_output=True, text=True, check=False)

    def test_cli_success_is_one_json_object(self):
        process = self.invoke_cli(str(HERE / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_file_error(self):
        process = self.invoke_cli(str(HERE / "nonexistent-input.json"))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_usage_errors(self):
        for args in ((), ("a", "b")):
            with self.subTest(args=args):
                process = self.invoke_cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_real_malformed_json_file(self):
        process = self.invoke_cli(str(HERE / "implementation.py"))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_real_invalid_schema_file(self):
        process = self.invoke_cli(str(HERE / "build_manifest.json"))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_json_parse_schema_and_encoding_errors(self):
        cases = ("{", "[]", '{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}',
                 json.dumps({**self.data, "schema_version": "unknown"}))
        for content in cases:
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")):
            with contextlib.redirect_stdout(output):
                code = app.main(["synthetic-input.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
