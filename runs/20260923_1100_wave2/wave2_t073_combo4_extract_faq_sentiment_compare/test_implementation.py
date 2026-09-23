"""Standard-library tests using only labeled synthetic fixtures."""

import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def pipeline(self):
        return app.run_pipeline(self.request)

    def question(self, value):
        self.request["document"] = "Customer: Demo\nOrder: 1042\nQuestion: " + value + "\n"

    def predecessor(self, stage):
        result = self.pipeline()
        result["stage"] = stage
        index = app.STAGES.index(stage)
        result["data"] = {name: result["data"][name] for name in app.STAGES[1:index + 1]}
        return result

    def test_complete_pipeline(self):
        result = self.pipeline()
        self.assertEqual(result["stage"], "compare")
        self.assertEqual(list(result["data"]), ["extract", "faq", "sentiment", "compare"])
        self.assertEqual(result["data"]["compare"]["recommended_product_id"], "care")

    def test_source_spans(self):
        for field in self.pipeline()["data"]["extract"]["fields"].values():
            start, end = field["span"]
            self.assertEqual(field["source_text"], self.request["document"][start:end])

    def test_integer_conversion(self):
        self.assertEqual(self.pipeline()["data"]["extract"]["fields"]["order"]["value"], 1042)

    def test_missing_optional(self):
        extraction = self.pipeline()["data"]["extract"]
        self.assertEqual(extraction["missing_fields"], ["email"])
        self.assertEqual(extraction["missing_required"], [])

    def test_missing_required_abstains(self):
        self.request["document"] = ""
        result = self.pipeline()["data"]
        self.assertIn("question", result["extract"]["missing_required"])
        self.assertEqual(result["faq"]["status"], "abstained")
        self.assertEqual(result["sentiment"]["faq_status"], "abstained")
        self.assertEqual(result["compare"]["faq_status"], "abstained")

    def test_failed_conversion(self):
        self.request["document"] = self.request["document"].replace("1042", "not-a-number")
        extraction = self.pipeline()["data"]["extract"]
        self.assertIn("order", extraction["missing_required"])
        self.assertEqual(extraction["invalid_fields"][0]["field"], "order")

    def test_unicode_offsets_and_trim(self):
        self.request["document"] = "Customer:  Élodie 😀  \nQuestion: refund\n"
        field = self.pipeline()["data"]["extract"]["fields"]["customer"]
        self.assertEqual(field["value"], "Élodie 😀")
        self.assertEqual(self.request["document"][slice(*field["span"])], field["value"])

    def test_whole_match_pattern(self):
        self.request["fields"][0]["pattern"] = "Demo Person"
        self.assertEqual(self.pipeline()["data"]["extract"]["fields"]["customer"]["value"], "Demo Person")

    def test_unmatched_optional_capture(self):
        self.request["fields"][0]["pattern"] = r"Customer:(?P<value>ZZZ)?"
        self.assertIn("customer", self.pipeline()["data"]["extract"]["missing_fields"])

    def test_grounded_answer_and_handoff(self):
        data = self.pipeline()["data"]
        self.assertEqual(data["faq"]["query"], data["extract"]["fields"]["question"]["value"])
        self.assertEqual(data["faq"]["answer"], self.request["knowledge_base"][0]["answer"])
        self.assertEqual(data["faq"]["citation"], "kb-refund")

    def test_irrelevant_abstention(self):
        self.question("astronomy telescope")
        answer = self.pipeline()["data"]["faq"]
        self.assertEqual(answer["status"], "abstained")
        self.assertIsNone(answer["answer"])

    def test_empty_kb(self):
        self.request["knowledge_base"] = []
        self.assertEqual(self.pipeline()["data"]["faq"]["status"], "abstained")

    def test_retrieval_ties_use_id(self):
        article = copy.deepcopy(self.request["knowledge_base"][0])
        article["id"] = "aaa"
        self.request["knowledge_base"].append(article)
        self.assertEqual(self.pipeline()["data"]["faq"]["citation"], "aaa")

    def test_sentiment_contributions(self):
        sentiment = self.pipeline()["data"]["sentiment"]
        self.assertEqual(sentiment["score"], -6)
        self.assertEqual(sum(c["score"] for c in sentiment["contributions"]), -6)
        self.assertEqual(sentiment["label"], "negative")

    def test_negation(self):
        self.question("not good never bad")
        insight = self.pipeline()["data"]["sentiment"]
        self.assertEqual(insight["score"], 0)
        self.assertTrue(all(c["negated"] for c in insight["contributions"]))

    def test_severity_overrides_positive(self):
        self.question("great happy love smoke")
        insight = self.pipeline()["data"]["sentiment"]
        self.assertEqual(insight["label"], "positive")
        self.assertEqual(insight["severity"], "critical")
        self.assertEqual(insight["priority"], 100)

    def test_kb_answer_not_scored_as_customer(self):
        self.question("delivery shipment tracking")
        self.request["knowledge_base"][1]["answer"] = "unsafe angry broken"
        insight = self.pipeline()["data"]["sentiment"]
        self.assertEqual(insight["score"], 0)
        self.assertEqual(insight["severity"], "normal")

    def test_unanswered_priority_boost(self):
        self.question("delivery shipment tracking")
        answered = self.pipeline()["data"]["sentiment"]["priority"]
        self.request["knowledge_base"] = []
        self.assertEqual(self.pipeline()["data"]["sentiment"]["priority"], answered + 10)

    def test_units(self):
        rows = self.pipeline()["data"]["compare"]["side_by_side"]
        self.assertEqual(rows[0]["attributes"], {"price": 50, "battery": 2, "warranty": 6})
        self.assertEqual(rows[1]["attributes"]["warranty"], 24)

    def test_severity_changes_ranking(self):
        self.request["preferences"] = [{"attribute": "price", "direction": "min", "weight": 1}]
        self.assertEqual(self.pipeline()["data"]["compare"]["recommended_product_id"], "care")
        self.question("delivery shipment tracking")
        comparison = self.pipeline()["data"]["compare"]
        self.assertEqual(comparison["recommended_product_id"], "budget")
        self.assertEqual(comparison["adjustments"], [])

    def test_missing_attributes_penalized(self):
        self.request["products"][1]["attributes"] = {}
        comparison = self.pipeline()["data"]["compare"]
        row = next(r for r in comparison["ranking"] if r["id"] == "care")
        self.assertEqual(row["score"], 0)
        self.assertTrue(all(c["missing"] for c in row["contributions"]))

    def test_equal_products_tie_by_id(self):
        self.request["products"][1]["attributes"] = copy.deepcopy(self.request["products"][0]["attributes"])
        comparison = self.pipeline()["data"]["compare"]
        self.assertEqual(comparison["ranking"][0]["id"], "budget")
        self.assertEqual(comparison["ranking"][0]["score"], 1)

    def test_empty_products(self):
        self.request["products"] = []
        comparison = self.pipeline()["data"]["compare"]
        self.assertEqual(comparison["ranking"], [])
        self.assertIsNone(comparison["recommended_product_id"])

    def test_input_unchanged_and_deterministic(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(self.pipeline(), self.pipeline())
        self.assertEqual(original, self.request)

    def test_invalid_request_variants(self):
        mutations = [
            lambda r: r.update(schema_version="2"),
            lambda r: r.update(retrieval_threshold=True),
            lambda r: r.update(retrieval_threshold=0),
            lambda r: r.update(question_field="unknown"),
            lambda r: r["fields"][0].update(pattern="["),
            lambda r: r["fields"][0].update(required=1),
            lambda r: r["fields"].append(copy.deepcopy(r["fields"][0])),
            lambda r: r["products"][0]["attributes"]["price"].update(unit="EUR"),
            lambda r: r["products"][0]["attributes"]["price"].update(value=float("nan")),
            lambda r: r["products"][0]["attributes"]["price"].update(value=True),
            lambda r: r["products"].append(copy.deepcopy(r["products"][0])),
            lambda r: r["preferences"][0].update(weight=-1),
            lambda r: r.update(preferences=[]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                invalid = copy.deepcopy(self.request)
                mutate(invalid)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(invalid)

    def test_wrong_predecessor_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.compare(self.predecessor("extract"))

    def test_tampered_source_rejected(self):
        previous = self.predecessor("extract")
        previous["data"]["extract"]["fields"]["question"]["span"] = [0, 1]
        with self.assertRaises(app.ValidationError):
            app.faq(previous)

    def test_tampered_grounding_rejected(self):
        previous = self.predecessor("faq")
        previous["data"]["faq"]["answer"] = "Unsupported promise"
        with self.assertRaises(app.ValidationError):
            app.sentiment(previous)

    def test_tampered_sentiment_rejected(self):
        previous = self.predecessor("sentiment")
        previous["data"]["sentiment"]["priority"] = 0
        with self.assertRaises(app.ValidationError):
            app.compare(previous)

    def cli(self, *arguments):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                              cwd=ROOT, text=True, capture_output=True, check=False)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["stage"], "compare")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = self.cli("synthetic-nonexistent.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_argument_error(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_and_validation_errors(self):
        for content in ("{", "[]", '{"x": NaN}', '{"x": 1, "x": 2}'):
            with self.subTest(content=content):
                output = io.StringIO()
                with mock.patch("builtins.open", mock.mock_open(read_data=content)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
