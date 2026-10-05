"""All fixtures are synthetic; tests do not write any files or call networks."""

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
        self.input = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_case(self):
        return app.run_pipeline(self.input)["stages"]

    def test_full_pipeline(self):
        output = app.run_pipeline(self.input)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(list(output["stages"]), list(app.STAGES))
        app.validate(output, 4)

    def test_personalized_onboarding(self):
        result = self.run_case()["onboarding"]
        self.assertIn("Alex Example", result["greeting"])
        self.assertIn("drawing", result["next_step"]["instruction"])

    def test_missing_contact(self):
        self.input["customer"]["email"] = ""
        result = self.run_case()
        self.assertEqual(result["onboarding"]["next_step"]["code"], "add_contact")
        self.assertFalse(result["support"]["contact_available"])

    def test_missing_goals(self):
        self.input["customer"]["goals"] = []
        self.assertEqual(self.run_case()["onboarding"]["next_step"]["code"], "choose_goal")

    def test_grounded_support(self):
        support = self.run_case()["support"]
        self.assertEqual(support["source_ids"], ["SYN-KB-PORTABLE"])
        self.assertIn(self.input["knowledge"][0]["answer"], support["answer"])

    def test_unknown_support_question(self):
        self.input["question"] = "Quantum entanglement?"
        support = self.run_case()["support"]
        self.assertEqual(support["source_ids"], [])
        self.assertEqual(support["resolution"], "human_review_needed")
        self.assertIn("no ticket has been submitted", support["answer"])

    def test_empty_knowledge(self):
        self.input["knowledge"] = []
        self.assertEqual(self.run_case()["support"]["topic_tags"], [])

    def test_onboarding_support_handoff(self):
        stages = self.run_case()
        self.assertEqual(stages["support"]["onboarding_next_step"],
                         stages["onboarding"]["next_step"])

    def test_support_recommendation_handoff(self):
        stages = self.run_case()
        recommendation = stages["recommendations"]
        self.assertEqual(recommendation["support_source_ids"], stages["support"]["source_ids"])
        self.assertEqual(recommendation["items"][0]["product_id"], "SYN-P001")
        self.assertEqual(recommendation["items"][0]["matched_terms"]["support_topics"],
                         ["drawing", "portable"])

    def test_support_topics_change_discovery(self):
        self.input["customer"]["goals"] = []
        self.input["customer"]["interests"] = []
        self.input["catalog"][1]["tags"] = ["returns"]
        self.input["question"] = "refund"
        self.assertEqual(self.run_case()["recommendations"]["items"][0]["product_id"],
                         "SYN-P002")

    def test_filters_and_empty_recommendations(self):
        self.input["catalog"][0]["active"] = False
        self.input["catalog"][1]["stock"] = 0
        result = self.run_case()
        self.assertEqual(result["recommendations"]["items"], [])
        self.assertEqual(result["documents"]["lines"], [])
        self.assertEqual(result["documents"]["review_status"], "needs_review")

    def test_budget_boundary(self):
        self.input["customer"]["budget_cents"] = 1500
        items = self.run_case()["recommendations"]["items"]
        self.assertEqual([item["product_id"] for item in items], ["SYN-P002"])

    def test_zero_budget(self):
        self.input["customer"]["budget_cents"] = 0
        self.assertEqual(self.run_case()["recommendations"]["items"], [])

    def test_free_product(self):
        self.input["customer"]["budget_cents"] = 0
        self.input["catalog"][0]["price_cents"] = 0
        self.assertEqual(self.run_case()["documents"]["total_cents"], 0)
        self.assertEqual(self.run_case()["documents"]["review_status"], "draft_ready")

    def test_deterministic_tie_breaking(self):
        self.input["customer"]["goals"] = []
        self.input["customer"]["interests"] = []
        self.input["knowledge"] = []
        for product in self.input["catalog"]:
            product["price_cents"] = 100
        self.input["catalog"].reverse()
        self.assertEqual([item["product_id"] for item in
                          self.run_case()["recommendations"]["items"]],
                         ["SYN-P001", "SYN-P002", "SYN-P003"])

    def test_document_extraction_and_handoff(self):
        stages = self.run_case()
        document = stages["documents"]
        self.assertEqual(document["reference"], "SYN-PROPOSAL-001")
        self.assertEqual(document["lines"][0]["product_id"],
                         stages["recommendations"]["items"][0]["product_id"])
        self.assertEqual(document["support_source_ids"], stages["support"]["source_ids"])
        self.assertEqual(document["total_cents"], 5000)
        self.assertIn("USD 50.00", document["rendered_text"])
        self.assertTrue(all(document["checks"].values()))

    def test_document_total_budget_check(self):
        self.input["document_text"] = "Reference: SYN-OVER\nQuantity: 5"
        document = self.run_case()["documents"]
        self.assertFalse(document["checks"]["within_total_budget"])
        self.assertEqual(document["review_status"], "needs_review")

    def test_document_inventory_check(self):
        self.input["catalog"][0]["stock"] = 1
        self.assertFalse(self.run_case()["documents"]["checks"]["inventory_sufficient"])

    def test_document_invalid_formats(self):
        for source in ("Reference: X", "Reference: X\nQuantity: 0",
                       "Reference: X\nQuantity: -2", "Reference: X\nQuantity: 2.5",
                       "Reference: X\nQuantity: 1001", "Reference: X\nQuantity: 2\nQuantity: 3",
                       "Reference: X\nQuantity: 2\nSecret: nope", "bad line",
                       "Reference: ../escape\nQuantity: 2"):
            with self.subTest(source=source), self.assertRaises(app.ValidationError):
                app.parse_document(source)

    def test_source_immutability_and_repeatability(self):
        before = copy.deepcopy(self.input)
        first = app.run_pipeline(self.input)
        self.assertEqual(self.input, before)
        self.assertEqual(first, app.run_pipeline(self.input))

    def test_tampered_handoff_rejected(self):
        state = app.advance(self.input, "onboarding")
        state["stages"]["onboarding"]["customer_id"] = "OTHER"
        with self.assertRaises(app.ValidationError):
            app.advance(state, "support")

    def test_tampered_recommendation_rejected(self):
        state = self.input
        for stage in app.STAGES[:3]:
            state = app.advance(state, stage)
        state["stages"]["recommendations"]["items"][0]["price_cents"] = 1
        with self.assertRaises(app.ValidationError):
            app.advance(state, "documents")

    def test_out_of_order_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.advance(self.input, "support")

    def test_invalid_input_variations(self):
        cases = []
        for field, value in (("budget_cents", True), ("budget_cents", -1),
                             ("budget_cents", 1.5), ("email", "bad"),
                             ("goals", "drawing"), ("name", " "),
                             ("interests", ["art", "ART"])):
            state = copy.deepcopy(self.input)
            state["customer"][field] = value
            cases.append(state)
        for state in cases:
            with self.subTest(customer=state["customer"]), self.assertRaises(app.ValidationError):
                app.run_pipeline(state)

    def test_invalid_envelopes(self):
        for value in (None, [], {}, {"schema_version": 1}):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(value)

    def test_duplicate_catalog_id(self):
        self.input["catalog"].append(copy.deepcopy(self.input["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_invalid_stock_active_and_price(self):
        for field, value in (("stock", True), ("stock", -1), ("active", 1),
                             ("price_cents", "2500")):
            state = copy.deepcopy(self.input)
            state["catalog"][0][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(state)

    def test_unknown_field_and_non_synthetic(self):
        self.input["extra"] = 1
        with self.assertRaises(app.ValidationError):
            self.run_case()
        del self.input["extra"]
        self.input["fixture_label"] = "real"
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"),
             str(HERE / "example_input.json")],
            cwd=HERE, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file(self):
        process = subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"),
             str(HERE / "nonexistent-input.json")],
            cwd=HERE, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_usage_error(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(app.main([]), 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for raw in ("{", "null", '{"x": 1, "x": 2}', '{"x": NaN}',
                    json.dumps({**self.input, "question": ""})):
            stream = io.StringIO()
            with self.subTest(raw=raw), patch("builtins.open", mock_open(read_data=raw)):
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_decode_error(self):
        stream = io.StringIO()
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
        with patch("builtins.open", side_effect=error), contextlib.redirect_stdout(stream):
            self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_huge_integer_and_input_size(self):
        for raw in ('{"budget": ' + "9" * 5000 + "}", " " * 1_000_001):
            stream = io.StringIO()
            with self.subTest(length=len(raw)), patch("builtins.open", mock_open(read_data=raw)):
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_invalid_filename(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(app.main(["invalid\0name"]), 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
