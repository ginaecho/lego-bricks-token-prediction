"""Offline synthetic fixtures, including complete extract -> behavior -> FAQ paths."""

import copy
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

    def run_data(self):
        return app.run_pipeline(self.data)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               *map(str, args)], cwd=ROOT, text=True,
                              capture_output=True, check=False)

    def test_complete_handoffs(self):
        result = self.run_data()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["behavior"]["context"]["customer_id"], "synthetic-alex")
        self.assertEqual(result["behavior"]["recommendations"][0]["id"], "toy-1")
        self.assertEqual(result["faq"]["citation"]["id"], "kb-toys")
        self.assertEqual(result["faq"]["answer"], self.data["knowledge_base"][1]["answer"])

    def test_unicode_spans_and_whitespace(self):
        self.data["document"] = "Note: 🧱\r\nCustomer:   Élodie  \r\nQuestion: return window\r\n"
        result = self.run_data()["extraction"]
        field = result["fields"]["customer_id"]
        self.assertEqual(field["value"], "Élodie")
        start, end = field["span"]
        self.assertEqual(self.data["document"][start:end], "Élodie")

    def test_custom_label_and_numeric_field(self):
        self.data["fields"]["customer_id"]["label"] = "Client.ID"
        self.data["document"] = self.data["document"].replace("Customer:", "client.id:")
        extraction = self.run_data()["extraction"]
        self.assertEqual(extraction["fields"]["customer_id"]["value"], "synthetic-alex")
        self.assertEqual(extraction["fields"]["budget"]["value"], 35.5)
        self.assertEqual(extraction["fields"]["budget"]["raw"], "35.50")

    def test_missing_required_fields_continue_safely(self):
        self.data["document"] = "Interest: books"
        result = self.run_data()
        self.assertEqual(result["extraction"]["missing_required"], ["customer_id", "question"])
        self.assertTrue(result["behavior"]["cold_start"])
        self.assertEqual(result["faq"]["reason"], "missing_question")

    def test_blank_field_is_missing(self):
        self.data["document"] = "Customer:  \nInterest: books\nQuestion: return window"
        result = self.run_data()
        self.assertIn("customer_id", result["extraction"]["missing_fields"])
        self.assertIsNone(result["behavior"]["context"]["customer_id"])

    def test_duplicate_field_is_ambiguous(self):
        self.data["document"] += "Customer: another\n"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_malformed_number(self):
        self.data["document"] = self.data["document"].replace("35.50", "35 USD")
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_recency_and_purchase_weights(self):
        rows = {r["id"]: r for r in self.run_data()["behavior"]["recommendations"]}
        self.assertEqual(rows["toy-1"]["history_score"], 3)
        self.assertEqual(rows["book-1"]["history_score"], 0.5)
        self.assertEqual(rows["book-1"]["score"], 1)
        self.assertEqual(rows["book-2"]["history_score"], 0)

    def test_cold_start_interest_changes_faq_category(self):
        self.data["events"] = []
        result = self.run_data()
        self.assertTrue(result["behavior"]["cold_start"])
        self.assertEqual(result["behavior"]["recommendations"][0]["id"], "book-1")
        self.assertEqual(result["faq"]["citation"]["id"], "kb-books")

    def test_customer_extraction_changes_behavior_and_faq(self):
        self.data["document"] = self.data["document"].replace("synthetic-alex", "synthetic-other")
        result = self.run_data()
        self.assertEqual(result["behavior"]["matched_events"], 1)
        self.assertEqual(result["behavior"]["recommendations"][0]["id"], "book-2")
        self.assertEqual(result["faq"]["category"], "books")

    def test_no_customer_no_interest_stable_id_ties(self):
        self.data["document"] = "Question: return window"
        result = self.run_data()
        self.assertEqual([r["id"] for r in result["behavior"]["recommendations"]],
                         ["book-1", "book-2", "toy-1"])
        self.assertTrue(result["behavior"]["cold_start"])

    def test_top_k(self):
        self.data["config"]["top_k"] = 1
        self.assertEqual(len(self.run_data()["behavior"]["recommendations"]), 1)

    def test_future_event_rejected(self):
        self.data["events"][0]["at"] = "2027-01-01T00:00:00Z"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_naive_timestamp_rejected(self):
        self.data["as_of"] = "2026-09-23T12:00:00"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_timezone_equivalence(self):
        self.data["as_of"] = "2026-09-23T14:00:00+02:00"
        self.assertEqual(self.run_data()["behavior"]["recommendations"][0]["score"], 3)

    def test_unknown_event_item_rejected(self):
        self.data["events"][0]["item_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_duplicate_catalog_id_rejected(self):
        self.data["items"].append(copy.deepcopy(self.data["items"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_invalid_configuration_types_and_ranges(self):
        for key, value in [("half_life_days", 0), ("half_life_days", float("nan")),
                           ("top_k", True), ("top_k", 1.5), ("faq_min_score", 1.1)]:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["config"][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_empty_catalog_abstention(self):
        self.data["items"] = []
        self.data["events"] = []
        result = self.run_data()
        self.assertEqual(result["behavior"]["recommendations"], [])
        self.assertEqual(result["faq"]["reason"], "no_recommendations")

    def test_no_knowledge_abstention(self):
        self.data["knowledge_base"] = []
        result = self.run_data()["faq"]
        self.assertEqual(result["reason"], "insufficient_evidence")
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["citation"])

    def test_irrelevant_question_never_fabricates(self):
        self.data["document"] = self.data["document"].replace(
            "What is the return window?", "Who invented quantum gravity?")
        self.data["config"]["faq_min_score"] = 0
        self.assertEqual(self.run_data()["faq"]["status"], "abstained")

    def test_empty_query_tokens_abstain(self):
        self.data["document"] = "Question: What is the"
        self.assertEqual(self.run_data()["faq"]["status"], "abstained")

    def test_faq_threshold_is_inclusive(self):
        self.data["document"] = self.data["document"].replace(
            "What is the return window?", "return unknown")
        self.assertEqual(self.run_data()["faq"]["status"], "answered")
        self.data["config"]["faq_min_score"] = 0.51
        self.assertEqual(self.run_data()["faq"]["status"], "abstained")

    def test_category_filter_prevents_wrong_policy(self):
        self.data["knowledge_base"] = self.data["knowledge_base"][:1]
        self.assertEqual(self.run_data()["faq"]["status"], "abstained")

    def test_article_ties_resolved_by_id(self):
        article = copy.deepcopy(self.data["knowledge_base"][1])
        article.update(id="aaa", answer="Synthetic alternate policy.")
        self.data["knowledge_base"].append(article)
        self.assertEqual(self.run_data()["faq"]["citation"]["id"], "aaa")

    def test_tampered_extraction_handoff_rejected(self):
        extracted = app.extract(self.data)
        extracted["fields"]["customer_id"]["span"] = [0, 1]
        with self.assertRaises(app.ValidationError):
            app.personalize(self.data, extracted)

    def test_tampered_behavior_handoff_rejected(self):
        behavior = self.run_data()["behavior"]
        behavior["recommendations"][0]["score"] = -1
        with self.assertRaises(app.ValidationError):
            app.answer_faq(self.data, behavior)

    def test_ungrounded_answer_rejected(self):
        faq = self.run_data()["faq"]
        faq["answer"] = "Unsubstantiated policy"
        with self.assertRaises(app.ValidationError):
            app.validate(faq, "faq", self.data)

    def test_input_is_not_mutated_and_deterministic(self):
        original = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(self.data, original)
        self.assertEqual(first, self.run_data())

    def test_unknown_keys_rejected(self):
        self.data["extra"] = 1
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_cli_success_single_json_object(self):
        completed = self.cli(ROOT / "example_input.json")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")

    def test_cli_missing_file_and_usage(self):
        for args in [(), ("does-not-exist.json",), ("one", "two")]:
            with self.subTest(args=args):
                completed = self.cli(*args)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(json.loads(completed.stdout)["status"], "error")
                self.assertEqual(completed.stderr, "")

    def test_cli_json_parse_and_validation_errors(self):
        for text in ['{"broken":', "[]", '{"x":1,"x":2}', "\ufeff{}"]:
            with self.subTest(text=text):
                with patch.object(Path, "read_text", return_value=text):
                    with patch("builtins.print") as output:
                        self.assertEqual(app.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")

    def test_cli_invalid_utf8_error(self):
        with patch.object(Path, "read_text",
                          side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")):
            with patch("builtins.print") as output:
                self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")


if __name__ == "__main__":
    unittest.main()
