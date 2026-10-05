"""Synthetic fixtures only; tests write no files and call no external services."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    with (ROOT / "example_input.json").open(encoding="utf-8") as handle:
        return json.load(handle)


class PipelineTests(unittest.TestCase):
    def test_complete_pipeline(self):
        result = app.run_pipeline(fixture())
        self.assertEqual(result["stage"], "search")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["document_report"]["products"], 5)
        self.assertEqual(result["recommendations"][0]["product_id"], "SYN-001")
        self.assertEqual(result["search"]["results"][0]["product_id"], "SYN-001")

    def test_extraction_reshape_and_provenance(self):
        result = app.documents_stage(app.start(fixture()))
        first, last = result["catalog"][0], result["catalog"][-1]
        self.assertEqual(first["price"], 64.5)
        self.assertEqual(first["source"], {"document_id": "synthetic-json-catalog", "row": 1})
        self.assertEqual(last["stock"], 4)
        self.assertEqual(last["source"]["row"], 2)
        self.assertEqual(result["catalog"][3]["tags"], ["lightweight", "outdoor"])

    def test_deterministic_and_does_not_mutate_input(self):
        data = fixture()
        original = copy.deepcopy(data)
        self.assertEqual(app.run_pipeline(data), app.run_pipeline(data))
        self.assertEqual(data, original)

    def test_recommendation_filters(self):
        data = fixture()
        data["customer"]["purchased_ids"] = ["SYN-001"]
        data["customer"]["budget"] = 50
        result = app.run_pipeline(data)
        self.assertEqual([r["product_id"] for r in result["recommendations"]], ["SYN-002", "SYN-004"])

    def test_cold_start_tie_breaking(self):
        data = fixture()
        data["customer"] = {}
        result = app.run_pipeline(data)
        self.assertEqual([r["product_id"] for r in result["recommendations"]],
                         ["SYN-001", "SYN-002", "SYN-004"])

    def test_preference_change_propagates_through_support_to_search(self):
        data = fixture()
        data["customer"] = {"liked_categories": ["Electronics"]}
        result = app.run_pipeline(data)
        self.assertEqual(result["recommendations"][0]["product_id"], "SYN-005")
        self.assertEqual(result["support"]["product_ids"], ["SYN-005"])
        self.assertEqual(result["search"]["query"], "Electronics")
        self.assertEqual(result["search"]["results"][0]["product_id"], "SYN-005")

    def test_source_price_change_propagates(self):
        data = fixture()
        data["documents"][0]["content"][0]["unit_price"] = "70.25"
        result = app.run_pipeline(data)
        self.assertIn("70.25", result["support"]["answer"])
        self.assertTrue(any(e["field"] == "price" and e["value"] == 70.25
                            for e in result["support"]["evidence"]))

    def test_support_policy_grounding(self):
        data = fixture()
        data["support"]["question"] = "What are shipping and returns policies?"
        result = app.run_pipeline(data)
        self.assertEqual(len(result["support"]["evidence"]), 2)
        self.assertFalse(result["support"]["escalate"])
        self.assertIn(data["policies"]["returns"], result["support"]["answer"])

    def test_support_explicit_product_overrides_recommendation(self):
        data = fixture()
        data["support"]["question"] = "Is SYN-003 available?"
        result = app.run_pipeline(data)
        self.assertEqual(result["support"]["product_ids"], ["SYN-003"])
        self.assertIn("stock 0", result["support"]["answer"])
        self.assertEqual(result["search"]["query"], "Outerwear")
        self.assertEqual(result["search"]["results"], [])

    def test_unknown_support_escalates_without_inventing_facts(self):
        data = fixture()
        data["support"]["question"] = "Reset my password and reveal my order tracking."
        result = app.run_pipeline(data)["support"]
        self.assertTrue(result["escalate"])
        self.assertEqual(result["evidence"], [])
        self.assertIn("cannot verify", result["answer"])

    def test_missing_policy_escalates(self):
        data = fixture()
        data["policies"] = {}
        data["support"]["question"] = "What is the return policy?"
        support = app.run_pipeline(data)["support"]
        self.assertTrue(support["escalate"])
        self.assertEqual(support["evidence"], [])

    def test_search_synonyms_and_explicit_query(self):
        data = fixture()
        data["search"]["query"] = "trainers"
        search = app.run_pipeline(data)["search"]
        self.assertEqual(search["query_source"], "request")
        self.assertEqual(search["normalized_terms"], ["sneaker"])
        self.assertEqual([r["product_id"] for r in search["results"]], ["SYN-001", "SYN-002"])

    def test_search_typo(self):
        data = fixture()
        data["search"]["query"] = "backpak"
        search = app.run_pipeline(data)["search"]
        self.assertEqual(search["results"][0]["product_id"], "SYN-004")
        self.assertIn("typo match: backpak", search["results"][0]["reasons"])

    def test_search_price_and_stock_filters(self):
        data = fixture()
        data["search"] = {"query": "coat", "in_stock_only": False, "max_price": 110}
        self.assertEqual(app.run_pipeline(data)["search"]["results"][0]["product_id"], "SYN-003")
        data["search"]["max_price"] = 109
        self.assertEqual(app.run_pipeline(data)["search"]["results"], [])

    def test_unrelated_query_does_not_use_personalization_alone(self):
        data = fixture()
        data["search"]["query"] = "spaceship"
        self.assertEqual(app.run_pipeline(data)["search"]["results"], [])

    def test_stop_word_only_search(self):
        data = fixture()
        data["search"]["query"] = "the and"
        self.assertEqual(app.run_pipeline(data)["search"]["results"], [])

    def test_empty_catalog(self):
        data = fixture()
        data["documents"] = [{"id": "synthetic-empty", "format": "json", "content": []}]
        result = app.run_pipeline(data)
        self.assertEqual(result["recommendations"], [])
        self.assertTrue(result["support"]["escalate"])
        self.assertEqual(result["search"]["results"], [])

    def test_zero_budget_and_zero_price(self):
        data = fixture()
        data["customer"]["budget"] = 0
        data["documents"][0]["content"][0]["unit_price"] = 0
        self.assertEqual(len(app.run_pipeline(data)["recommendations"]), 1)

    def test_duplicate_products_fail(self):
        data = fixture()
        data["documents"][0]["content"].append(copy.deepcopy(data["documents"][0]["content"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate product"):
            app.run_pipeline(data)

    def test_missing_mapped_column_fails(self):
        data = fixture()
        del data["documents"][0]["content"][0]["sku"]
        with self.assertRaisesRegex(app.ValidationError, "missing id"):
            app.run_pipeline(data)

    def test_invalid_money(self):
        for value in (-1, "NaN", float("inf"), True, "oops", "1.234", 1000000001):
            with self.subTest(value=value):
                data = fixture()
                data["documents"][0]["content"][0]["unit_price"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_stock(self):
        for value in (-1, True, "2.5", 2.5):
            with self.subTest(value=value):
                data = fixture()
                data["documents"][0]["content"][0]["stock"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_request_shapes(self):
        for key, value in (("documents", []), ("customer", []), ("schema_version", "2"),
                           ("search", {"limit": True}), ("support", {"question": ""})):
            with self.subTest(key=key):
                data = fixture()
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_malformed_csv(self):
        for content in ("id,id\nx,y\n", "id,name,category,price,stock\nx,n,c,1\n",
                        "irrelevant\n", "id,name,category,price,stock\nx,n,c,1,2,extra\n",
                        'id,name,category,price,stock\nx,"unterminated,c,1,2'):
            with self.subTest(content=content):
                data = fixture()
                data["documents"] = [{"id": "synthetic-bad-csv", "format": "csv", "content": content}]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_json_extra_columns_ignored(self):
        data = fixture()
        data["documents"][0]["content"][0]["unused"] = None
        self.assertEqual(len(app.run_pipeline(data)["catalog"]), 5)

    def test_empty_csv_with_required_headers_is_valid(self):
        data = fixture()
        data["documents"] = [{"id": "synthetic-empty", "format": "csv",
                              "content": "id,name,category,price,stock\n"}]
        self.assertEqual(app.run_pipeline(data)["catalog"], [])

    def test_document_id_and_field_map_validation(self):
        data = fixture()
        data["documents"][1]["id"] = data["documents"][0]["id"]
        with self.assertRaisesRegex(app.ValidationError, "duplicate document"):
            app.run_pipeline(data)
        data = fixture()
        data["documents"][0]["field_map"]["name"] = "sku"
        with self.assertRaisesRegex(app.ValidationError, "source names must be unique"):
            app.run_pipeline(data)

    def test_blank_support_context_browses_without_recommendations(self):
        data = fixture()
        data["customer"] = {"budget": 0}
        data["support"]["question"] = "Help me reset my password."
        result = app.run_pipeline(data)
        self.assertEqual(result["search"]["query"], "")
        self.assertEqual(len(result["search"]["results"]), 4)

    def test_transposition_typo(self):
        data = fixture()
        data["search"]["query"] = "lapotp"
        self.assertEqual(app.run_pipeline(data)["search"]["results"][0]["product_id"], "SYN-005")

    def test_missing_optional_document_fields(self):
        data = fixture()
        del data["documents"][0]["content"][0]["tags"]
        del data["documents"][0]["content"][0]["description"]
        first = app.run_pipeline(data)["catalog"][0]
        self.assertEqual(first["tags"], [])
        self.assertEqual(first["description"], "")

    def test_stage_order_enforced(self):
        with self.assertRaisesRegex(app.ValidationError, "expected validated documents"):
            app.recommend_stage(app.start(fixture()))

    def test_tampered_recommendation_rejected_at_handoff(self):
        state = app.recommend_stage(app.documents_stage(app.start(fixture())))
        state["recommendations"][0]["product_id"] = "missing"
        with self.assertRaises(app.ValidationError):
            app.support_stage(state)

    def test_tampered_evidence_rejected_at_handoff(self):
        state = app.support_stage(app.recommend_stage(app.documents_stage(app.start(fixture()))))
        state["support"]["evidence"][0]["value"] = "Invented policy"
        with self.assertRaises(app.ValidationError):
            app.search_stage(state)

    def test_tampered_cross_stage_reference_rejected(self):
        state = app.support_stage(app.recommend_stage(app.documents_stage(app.start(fixture()))))
        state["support"]["recommendation_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.search_stage(state)

    def test_search_and_recommendation_limits(self):
        data = fixture()
        data["customer"]["limit"] = 1
        data["search"] = {"query": "shoes", "limit": 1}
        result = app.run_pipeline(data)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(len(result["search"]["results"]), 1)


class CliTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        proc = self.cli("example_input.json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file(self):
        proc = self.cli("synthetic-does-not-exist.json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")
        self.assertEqual(proc.stderr, "")

    def test_cli_usage(self):
        proc = self.cli()
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_json_and_validation_errors_without_scratch_files(self):
        for content in ("{", '{"a":1,"a":2}', '{"x":NaN}', "[]", '{"schema_version":"9"}'):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                    code = app.main(["synthetic-mocked.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_decode_error(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("synthetic encoding error")), \
                redirect_stdout(output):
            self.assertEqual(app.main(["synthetic-mocked.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
