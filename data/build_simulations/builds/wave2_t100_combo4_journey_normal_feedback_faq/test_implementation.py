"""Tests use only the four deliverables; no temporary files or live providers."""

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
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_fixture(self):
        return app.run_pipeline(self.payload)

    def stage(self, result, name):
        return next(stage for stage in result["stages"] if stage["stage"] == name)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def assert_main_error(self, text):
        stream = io.StringIO()
        with patch.object(app.Path, "read_text", return_value=text), contextlib.redirect_stdout(stream):
            code = app.main(["example_input.json"])
        self.assertEqual(code, 2)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["status"], "error")
        return result

    def test_integrated_order_and_shared_envelopes(self):
        result = self.run_fixture()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["synthetic"])
        self.assertEqual([stage["stage"] for stage in result["stages"]], list(app.STAGES))
        for stage in result["stages"]:
            self.assertEqual(set(stage), {"schema_version", "status", "stage", "data"})
            self.assertEqual(stage["schema_version"], 1)
            self.assertEqual(stage["status"], "ok")

    def test_journey_prerequisites_and_personalized_next_actions(self):
        journey = self.stage(self.run_fixture(), "journey")["data"]
        self.assertEqual(journey["next_actions"], ["a_setup", "c_ads"])
        self.assertEqual([step["id"] for step in journey["steps"]], ["a_setup", "b_shipping"])
        self.assertNotIn("b_shipping", journey["next_actions"])

    def test_completed_actions_are_not_recommended(self):
        self.payload["profile"]["completed"].append("a_setup")
        journey = self.stage(self.run_fixture(), "journey")["data"]
        self.assertEqual([step["id"] for step in journey["steps"]], ["b_shipping", "c_ads"])
        self.assertNotIn("a_setup", journey["next_actions"])

    def test_personalization_propagates_through_every_stage(self):
        self.payload["profile"]["goals"] = ["advertising"]
        result = self.run_fixture()
        journey, normal, feedback, faq = result["stages"]
        self.assertEqual(journey["data"]["steps"][0]["id"], "c_ads")
        self.assertEqual(normal["data"]["findings"][0]["action_id"], "c_ads")
        self.assertEqual(normal["data"]["findings"][0]["citations"][0]["id"], "p_ads")
        self.assertNotIn("p_shipping", feedback["data"]["research_passage_ids"])
        self.assertIn("f5", [group["id"] for group in feedback["data"]["groups"]])
        self.assertNotIn("shipping", [answer["theme_id"] for answer in faq["data"]["answers"]])

    def test_no_feasible_two_step_journey_errors(self):
        self.payload["actions"] = self.payload["actions"][:1]
        with self.assertRaisesRegex(app.ValidationError, "two-step"):
            self.run_fixture()

    def test_cycle_without_eligible_actions_errors(self):
        self.payload["actions"] = self.payload["actions"][:2]
        self.payload["actions"][0]["prerequisites"] = ["b_shipping"]
        with self.assertRaisesRegex(app.ValidationError, "two-step"):
            self.run_fixture()

    def test_journey_tampering_rejected_before_research(self):
        validator = app.Validator(self.payload)
        journey = app.journey_stage(validator)
        journey["data"]["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.normal_stage(validator, journey)

    def test_research_exact_source_and_offsets(self):
        normal = self.stage(self.run_fixture(), "normal")["data"]
        passages = app.indexed(self.payload["passages"])
        for finding in normal["findings"]:
            cite = finding["citations"][0]
            record = passages[cite["id"]]
            self.assertEqual(finding["text"], record["text"][cite["start"]:cite["end"]])
            self.assertEqual(cite["quote"], finding["text"])
            self.assertEqual(cite["source"], record["source"])
        self.assertGreater(normal["findings"][0]["citations"][0]["start"], 0)

    def test_research_unicode_offsets_use_codepoints(self):
        self.payload["passages"][0]["text"] = "Café 🧱.  Shop setup checklist uses an address."
        finding = self.stage(self.run_fixture(), "normal")["data"]["findings"][0]
        self.assertEqual(finding["citations"][0]["start"], 9)
        self.assertEqual(finding["text"], "Shop setup checklist uses an address.")

    def test_absent_research_evidence_propagates_to_empty_faq(self):
        self.payload["passages"] = []
        self.payload["feedback"] = []
        result = self.run_fixture()
        self.assertTrue(all(item["status"] == "no_evidence"
                            for item in result["stages"][1]["data"]["findings"]))
        self.assertEqual(result["stages"][2]["data"]["research_passage_ids"], [])
        self.assertEqual(result["stages"][3]["data"], {"status": "no_feedback", "answers": []})

    def test_irrelevant_research_excludes_all_feedback(self):
        for passage in self.payload["passages"]:
            passage["text"] = "Unrelated astronomy observations."
        result = self.run_fixture()
        feedback = result["stages"][2]["data"]
        self.assertEqual(feedback["groups"], [])
        self.assertEqual(feedback["excluded_feedback_ids"], ["f1", "f2", "f3", "f4", "f5", "f6"])
        self.assertEqual(result["stages"][3]["data"]["status"], "no_feedback")

    def test_research_tampering_rejected_before_feedback(self):
        validator = app.Validator(self.payload)
        journey = app.journey_stage(validator)
        normal = app.normal_stage(validator, journey)
        normal["data"]["findings"][0]["citations"][0]["source"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.feedback_stage(validator, normal, journey)

    def test_duplicate_feedback_counts_once_and_retains_members(self):
        feedback = self.stage(self.run_fixture(), "feedback")["data"]
        groups = app.indexed(feedback["groups"])
        self.assertEqual(groups["f1"]["member_ids"], ["f1", "f2"])
        self.assertEqual(groups["f1"]["excerpt"]["quote"], "Shipping delivery was late!")
        shipping = app.indexed(feedback["themes"])["shipping"]
        self.assertEqual(shipping["unique_count"], 1)
        self.assertEqual(shipping["group_ids"], ["f1"])
        self.assertEqual(shipping["excerpts"], [groups["f1"]["excerpt"]])

    def test_feedback_scope_and_unclassified_excerpts(self):
        feedback = self.stage(self.run_fixture(), "feedback")["data"]
        self.assertEqual(feedback["research_passage_ids"], ["p_setup", "p_shipping"])
        self.assertEqual(feedback["excluded_feedback_ids"], ["f5"])
        self.assertEqual(feedback["unclassified_group_ids"], ["f6"])
        source = app.indexed(self.payload["feedback"])
        for group in feedback["groups"]:
            cite = group["excerpt"]
            self.assertEqual(cite["quote"], source[cite["id"]]["text"])

    def test_unlinked_feedback_is_excluded_even_if_topical(self):
        self.payload["feedback"].append({"id": "f7", "text": "Shipping is late.", "passage_ids": []})
        feedback = self.stage(self.run_fixture(), "feedback")["data"]
        self.assertIn("f7", feedback["excluded_feedback_ids"])
        self.assertEqual(app.indexed(feedback["themes"])["shipping"]["unique_count"], 1)

    def test_duplicates_merge_retrieved_passage_links(self):
        self.payload["feedback"][1]["passage_ids"] = ["p_setup"]
        feedback = self.stage(self.run_fixture(), "feedback")["data"]
        self.assertEqual(app.indexed(feedback["groups"])["f1"]["passage_ids"], ["p_setup", "p_shipping"])

    def test_feedback_tampering_rejected_before_faq(self):
        validator = app.Validator(self.payload)
        journey, normal, feedback, _ = self.run_fixture()["stages"]
        feedback["data"]["themes"][0]["unique_count"] = 900
        with self.assertRaises(app.ValidationError):
            app.faq_stage(validator, feedback, normal, journey)

    def test_faq_grounded_answers_and_explicit_abstention(self):
        faq = self.stage(self.run_fixture(), "faq")["data"]
        self.assertEqual(faq["status"], "partial")
        answers = {answer["theme_id"]: answer for answer in faq["answers"]}
        shipping = answers["shipping"]
        self.assertEqual(shipping["supporting_feedback_ids"], ["f1"])
        self.assertEqual(shipping["answer"], "Shipping delivery usually takes three business days.")
        self.assertEqual(shipping["citations"][0]["collection"], "knowledge_base")
        billing = answers["billing"]
        self.assertEqual(billing["status"], "abstained")
        self.assertIsNone(billing["answer"])
        self.assertEqual(billing["citations"], [])
        self.assertTrue(billing["reason"])

    def test_faq_does_not_treat_feedback_as_policy(self):
        self.payload["knowledge_base"] = []
        faq = self.stage(self.run_fixture(), "faq")["data"]
        self.assertEqual(faq["status"], "abstained")
        self.assertTrue(all(answer["status"] == "abstained" for answer in faq["answers"]))

    def test_faq_rejects_single_keyword_overlap(self):
        self.payload["knowledge_base"] = [{
            "id": "weak", "source": "synthetic://kb/weak", "text": "Shipping."
        }]
        faq = self.stage(self.run_fixture(), "faq")["data"]
        self.assertEqual(faq["status"], "abstained")

    def test_faq_requires_theme_anchor_not_question_only(self):
        self.payload["knowledge_base"] = [{
            "id": "weak", "source": "synthetic://kb/weak", "text": "Requests are handled elsewhere."
        }]
        faq = self.stage(self.run_fixture(), "faq")["data"]
        billing = next(answer for answer in faq["answers"] if answer["theme_id"] == "billing")
        self.assertEqual(billing["status"], "abstained")

    def test_faq_tampered_answer_rejected(self):
        validator = app.Validator(self.payload)
        _, _, feedback, faq = self.run_fixture()["stages"]
        faq["data"]["answers"][1]["answer"] = "Unsupported claim."
        with self.assertRaises(app.ValidationError):
            validator.stage(faq, "faq", feedback)

    def test_empty_feedback_or_themes(self):
        for key in ("feedback", "themes"):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload[key] = []
                faq = app.run_pipeline(payload)["stages"][3]["data"]
                self.assertEqual(faq, {"status": "no_feedback", "answers": []})

    def test_determinism_and_input_immutability(self):
        before = copy.deepcopy(self.payload)
        first = self.run_fixture()
        second = self.run_fixture()
        self.assertEqual(first, second)
        self.assertEqual(self.payload, before)

    def test_retrieval_ties_are_stable_by_source_id(self):
        self.payload["passages"].append({
            "id": "a_first", "source": "synthetic://research/tie",
            "text": self.payload["passages"][0]["text"],
        })
        first = self.run_fixture()
        self.payload["passages"].reverse()
        second = self.run_fixture()
        self.assertEqual(first, second)
        self.assertEqual(first["stages"][1]["data"]["findings"][0]["citations"][0]["id"], "a_first")

    def test_invalid_input_shapes_and_types(self):
        invalid = [
            None, [], {}, {"schema_version": 1},
        ]
        for key, value in (
            ("schema_version", True), ("schema_version", 2), ("synthetic", False),
            ("profile", []), ("actions", {}), ("feedback", None),
        ):
            case = copy.deepcopy(self.payload)
            case[key] = value
            invalid.append(case)
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)

    def test_unknown_fields_and_references_rejected(self):
        mutations = [
            lambda p: p.update(extra=True),
            lambda p: p["actions"][0].update(prerequisites=["missing"]),
            lambda p: p["actions"][0].update(prerequisites=["a_setup"]),
            lambda p: p["feedback"][0].update(passage_ids=["missing"]),
            lambda p: p["themes"][0].update(keywords=[]),
            lambda p: p["profile"].update(goals=[]),
            lambda p: p["profile"].update(goals=["and the"]),
            lambda p: p["actions"][0].update(research_query="the and"),
            lambda p: p["feedback"][0].update(text="!!!"),
        ]
        for mutation in mutations:
            payload = copy.deepcopy(self.payload)
            mutation(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)

    def test_duplicate_ids_and_values_rejected(self):
        for key in ("actions", "passages", "feedback", "themes", "knowledge_base"):
            payload = copy.deepcopy(self.payload)
            payload[key].append(copy.deepcopy(payload[key][0]))
            with self.subTest(collection=key):
                with self.assertRaisesRegex(app.ValidationError, "duplicate"):
                    app.run_pipeline(payload)
        self.payload["profile"]["goals"] = ["setup", "setup"]
        with self.assertRaisesRegex(app.ValidationError, "duplicates"):
            self.run_fixture()

    def test_citation_validator_rejects_bad_offsets_quote_and_source(self):
        validator = app.Validator(self.payload)
        record = self.payload["passages"][0]
        valid = app.citation("passages", record, 0, 9)
        for field, value in (
            ("start", -1), ("start", True), ("end", 999999),
            ("quote", "fabricated"), ("source", "fabricated"), ("id", "missing"),
            ("collection", "knowledge_base"),
        ):
            cite = dict(valid)
            cite[field] = value
            with self.subTest(field=field):
                with self.assertRaises(app.ValidationError):
                    validator.cite(cite, "passages")

    def test_wrong_handoff_stage_rejected(self):
        validator = app.Validator(self.payload)
        journey, normal, _, faq = self.run_fixture()["stages"]
        with self.assertRaisesRegex(app.ValidationError, "handoff"):
            validator.stage(faq, "faq", journey)
        with self.assertRaisesRegex(app.ValidationError, "previous"):
            validator.stage(normal, "normal")

    def test_cli_success_is_one_json_object(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout), self.run_fixture())

    def test_cli_missing_file_returns_json_exit_two(self):
        process = self.cli("nonexistent_input.json")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_usage_errors_return_json_exit_two(self):
        for args in ((), ("example_input.json", "extra")):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_directory_input_is_file_error(self):
        process = self.cli(".")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_main_malformed_json_duplicate_keys_and_nan(self):
        for text in (
            "{", '{"schema_version":1,"schema_version":1}', '{"value":NaN}', "null",
            '{"schema_version":' + "9" * 5000 + "}",
        ):
            with self.subTest(text=text):
                self.assert_main_error(text)

    def test_cli_malformed_json_exit_code_in_real_process(self):
        script = (
            "from unittest.mock import patch; import implementation as app; "
            "p=patch.object(app.Path, 'read_text', return_value='{'); p.start(); "
            "raise SystemExit(app.main(['example_input.json']))"
        )
        process = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_unicode_and_permission_file_errors_return_json(self):
        for error in (UnicodeError("bad encoding"), PermissionError("access denied")):
            stream = io.StringIO()
            with patch.object(app.Path, "read_text", side_effect=error), contextlib.redirect_stdout(stream):
                code = app.main(["example_input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
