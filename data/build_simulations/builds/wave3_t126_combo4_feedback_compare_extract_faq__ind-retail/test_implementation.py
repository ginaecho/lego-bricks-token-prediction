import contextlib
import copy
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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_case(self):
        return app.run_pipeline(self.data)

    def reject(self):
        with self.assertRaises(app.ValidationError):
            self.run_case()

    def test_complete_pipeline(self):
        result = self.run_case()
        self.assertEqual(list(result["stages"]), list(app.STAGES))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["synthetic"])
        self.assertFalse(result["stages"]["faq"]["abstained"])

    def test_deterministic_and_input_immutable(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.run_case(), self.run_case())
        self.assertEqual(self.data, original)

    def test_dedup_preserves_all_source_ids(self):
        stage = self.run_case()["stages"]["feedback"]
        self.assertEqual(len(stage["groups"]), 2)
        self.assertEqual(stage["groups"][0]["source_ids"], ["SYN-F1", "SYN-F2"])
        self.assertEqual(len(stage["themes"]["price"]), 1)

    def test_dedup_never_merges_distinct_products(self):
        self.data["feedback"][1]["sku"] = "SYN-BOTTLE-B"
        self.assertEqual(len(self.run_case()["stages"]["feedback"]["groups"]), 3)

    def test_feedback_exact_excerpts(self):
        feedback = self.run_case()["stages"]["feedback"]
        sources = {f["id"]: f["text"] for f in self.data["feedback"]}
        for supports in feedback["themes"].values():
            for support in supports:
                start, end = support["span"]
                self.assertEqual(support["excerpt"], sources[support["source_id"]][start:end])

    def test_no_feedback_is_valid(self):
        self.data["feedback"] = []
        self.assertEqual(self.run_case()["stages"]["feedback"], {"groups": [], "themes": {}})

    def test_unknown_theme(self):
        self.data["feedback"][0]["text"] = "Synthetic greeting."
        self.assertIn("other", self.run_case()["stages"]["feedback"]["themes"])

    def test_attribute_normalization(self):
        rows = self.run_case()["stages"]["compare"]["rows"]
        self.assertEqual(rows[0]["attributes"]["capacity_ml"], 500)
        self.assertEqual(rows[0]["attributes"]["color"], "gray")

    def test_catalog_values_preserved(self):
        result = self.run_case()["stages"]
        catalog = {p["sku"]: p for p in self.data["catalog"]}
        for row in result["compare"]["rows"]:
            for key in ("price", "stock", "currency"):
                self.assertEqual(row[key], catalog[row["sku"]][key])
        facts = result["faq"]["catalog_facts"]
        self.assertEqual(facts["price"], catalog[facts["sku"]]["price"])
        self.assertEqual(facts["stock"], catalog[facts["sku"]]["stock"])

    def test_feedback_affects_ranking_score(self):
        first = self.run_case()["stages"]["compare"]
        self.data["feedback"] = []
        second = self.run_case()["stages"]["compare"]
        self.assertEqual(first["rows"][0]["score"], second["rows"][0]["score"] + 1)
        self.assertEqual(first["feedback_theme_counts"]["price"], 1)

    def test_consent_required_for_each_regime(self):
        for regime in ("gdpr", "ccpa"):
            with self.subTest(regime=regime):
                self.data["customer"]["consent"] = {"gdpr": True, "ccpa": True}
                self.data["customer"]["consent"][regime] = False
                result = self.run_case()["stages"]["compare"]
                self.assertFalse(result["personalized"])
                self.assertTrue(all(row["score"] == 0 for row in result["rows"]))
                self.assertTrue(all(not row["reasons"] for row in result["rows"]))

    def test_no_consent_ignores_changed_personal_signals(self):
        self.data["customer"]["consent"]["gdpr"] = False
        first = self.run_case()["stages"]["compare"]
        self.data["clickstream"] = []
        self.data["preferences"] = {"max_price": 1000, "in_stock": False,
                                    "attributes": {"color": "Blue"}}
        self.assertEqual(first, self.run_case()["stages"]["compare"])

    def test_ranking_selection_propagates_to_extraction_and_faq(self):
        self.data["preferences"] = {"max_price": 30, "in_stock": True,
                                    "attributes": {"color": "Blue", "capacity": "750 ml"}}
        self.data["clickstream"] = []
        self.data["question"] = "Is this bottle dishwasher safe?"
        result = self.run_case()["stages"]
        for stage in ("compare", "extract", "faq"):
            self.assertEqual(result[stage]["selected_sku"], "SYN-BOTTLE-B")
        self.assertEqual(result["extract"]["fields"]["price"]["value"], 26)
        self.assertEqual(result["faq"]["citations"][0]["source_id"], "SYN-KB-CARE-B")

    def test_extraction_spans_and_types(self):
        fields = self.run_case()["stages"]["extract"]["fields"]
        document = self.data["documents"][0]["text"]
        for field in fields.values():
            self.assertEqual(document[slice(*field["span"])], field["raw"])
        self.assertIs(type(fields["stock"]["value"]), int)
        self.assertIs(type(fields["price"]["value"]), float)

    def test_missing_optional_field(self):
        result = self.run_case()["stages"]["extract"]
        self.assertEqual(result["missing_fields"], ["warranty"])
        self.assertEqual(result["missing_required"], [])

    def test_windows_document_line_endings(self):
        self.data["documents"][0]["text"] = self.data["documents"][0]["text"].replace("\n", "\r\n")
        stage = self.run_case()["stages"]["extract"]
        self.assertEqual(stage["missing_required"], [])
        self.assertEqual(stage["fields"]["sku"]["value"], "SYN-BOTTLE-A")

    def test_missing_required_sku_causes_abstention(self):
        self.data["documents"][0]["text"] = self.data["documents"][0]["text"].replace(
            "SKU: SYN-BOTTLE-A\n", "")
        result = self.run_case()["stages"]
        self.assertIn("sku", result["extract"]["missing_required"])
        self.assertTrue(result["faq"]["abstained"])
        self.assertEqual(result["faq"]["reason"], "missing_extracted_sku")

    def test_missing_document_reports_all_fields(self):
        self.data["documents"] = []
        result = self.run_case()["stages"]
        self.assertFalse(result["extract"]["document_found"])
        self.assertEqual(len(result["extract"]["missing_fields"]), 7)
        self.assertTrue(result["faq"]["abstained"])

    def test_schema_driven_custom_field(self):
        self.data["extraction_schema"].append(
            {"name": "volume", "label": "Volume (ml)", "type": "integer", "required": True})
        self.data["documents"][0]["text"] += "Volume (ml): 500   \n"
        self.assertEqual(self.run_case()["stages"]["extract"]["fields"]["volume"]["value"], 500)

    def test_duplicate_document_fields_rejected(self):
        self.data["documents"][0]["text"] += "Price: 18.50\n"
        self.reject()

    def test_document_price_and_stock_must_match_catalog(self):
        original = self.data["documents"][0]["text"]
        for old, new in (("18.50", "19.50"), ("Stock: 12", "Stock: 99")):
            with self.subTest(field=old):
                self.data["documents"][0]["text"] = original.replace(old, new)
                self.reject()

    def test_order_context_must_match(self):
        self.data["documents"][0]["text"] = self.data["documents"][0]["text"].replace(
            "SYN-ORDER-1", "SYN-ORDER-OTHER")
        self.reject()

    def test_invalid_extracted_number_rejected(self):
        self.data["documents"][0]["text"] = self.data["documents"][0]["text"].replace("18.50", "free")
        self.reject()

    def test_grounded_faq_has_exact_source(self):
        result = self.run_case()["stages"]["faq"]
        self.assertEqual(result["answer"], self.data["knowledge_base"][0]["text"])
        self.assertEqual(result["answer"], result["citations"][0]["excerpt"])

    def test_irrelevant_question_abstains_even_with_common_product_word(self):
        self.data["question"] = "Does this bottle support teleportation?"
        result = self.run_case()["stages"]["faq"]
        self.assertTrue(result["abstained"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])

    def test_empty_knowledge_base_abstains(self):
        self.data["knowledge_base"] = []
        self.assertTrue(self.run_case()["stages"]["faq"]["abstained"])

    def test_wrong_sku_knowledge_never_used(self):
        self.data["knowledge_base"] = [self.data["knowledge_base"][1]]
        self.assertTrue(self.run_case()["stages"]["faq"]["abstained"])

    def test_generic_article_is_retrievable(self):
        self.data["question"] = "What is the returns policy?"
        self.assertEqual(self.run_case()["stages"]["faq"]["citations"][0]["source_id"], "SYN-KB-RETURNS")

    def test_article_catalog_claim_mismatch_rejected(self):
        self.data["knowledge_base"][0]["claims"]["stock"] = 999
        self.reject()

    def test_article_commercial_prose_rejected(self):
        self.data["knowledge_base"][0]["text"] = "The price is 999."
        self.reject()

    def test_fabricated_feedback_excerpt_rejected_at_handoff(self):
        state = self.run_case()
        state["stages"] = {"feedback": state["stages"]["feedback"]}
        state["stages"]["feedback"]["themes"]["price"][0]["excerpt"] = "A fake endorsement"
        with self.assertRaises(app.ValidationError):
            app.compare_stage(state)

    def test_modified_comparison_rejected_at_handoff(self):
        state = self.run_case()
        del state["stages"]["faq"]
        del state["stages"]["extract"]
        state["stages"]["compare"]["rows"][0]["price"] = 0
        with self.assertRaises(app.ValidationError):
            app.extract_stage(state)

    def test_modified_extraction_rejected_at_handoff(self):
        state = self.run_case()
        del state["stages"]["faq"]
        state["stages"]["extract"]["fields"]["sku"]["span"] = [0, 3]
        with self.assertRaises(app.ValidationError):
            app.faq_stage(state)

    def test_out_of_order_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.compare_stage(self.run_case())

    def test_invalid_inputs(self):
        mutations = [
            lambda d: d.update(synthetic=False),
            lambda d: d.update(catalog=[]),
            lambda d: d["catalog"].append(copy.deepcopy(d["catalog"][0])),
            lambda d: d["catalog"][0].update(price=-1),
            lambda d: d["catalog"][0].update(price=float("nan")),
            lambda d: d["catalog"][0].update(stock=True),
            lambda d: d["catalog"][0].update(stock=1.5),
            lambda d: d["catalog"][1].update(currency="USD"),
            lambda d: d["catalog"][0]["attributes"].update(capacity="one bottle"),
            lambda d: d["customer"]["consent"].update(gdpr="yes"),
            lambda d: d["basket"]["items"][0].update(quantity=99),
            lambda d: d["basket"].update(customer_id="stranger"),
            lambda d: d["feedback"][0].update(synthetic=False),
            lambda d: d["feedback"][0].update(sku="unknown"),
            lambda d: d["clickstream"][0].update(customer_id="stranger"),
            lambda d: d["extraction_schema"][0].update(type="number"),
        ]
        original = copy.deepcopy(self.data)
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.data = copy.deepcopy(original)
                mutate(self.data)
                self.reject()

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "absent.json")],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_argument(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_and_duplicate_keys_without_files(self):
        for payload in ("{broken", '{"schema_version": "1.0", "schema_version": "2.0"}',
                        '{"unexpected": 1}', "null"):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with mock.patch("builtins.open", mock.mock_open(read_data=payload)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["virtual-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
