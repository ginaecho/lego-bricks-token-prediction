"""All product names, documents, and claims in these tests are synthetic."""

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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_example(self):
        return app.run_pipeline(self.data)

    def test_end_to_end_shared_output_schema(self):
        result = self.run_example()
        app.validate(result, app.OUTPUT_SCHEMA)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["comparison"]["winner_id"], "synthetic-a")

    def test_numeric_alias_unit_and_text_normalization(self):
        product = app.compare(self.data)["products"][0]
        self.assertEqual(product["attributes"]["battery"], {"value": 10.0, "unit": "h"})
        self.assertEqual(product["attributes"]["mass"]["value"], 500)
        self.assertEqual(product["attributes"]["finish"]["value"], "blue")

    def test_side_by_side_and_weighted_ranking(self):
        result = app.compare(self.data)
        self.assertEqual([p["score"] for p in result["products"]], [0.8, 0.2])
        battery = next(r for r in result["side_by_side"] if r["attribute"] == "battery")
        self.assertEqual(battery["values"], {"synthetic-a": 10, "synthetic-b": 12})
        self.assertEqual(result["columns"], ["synthetic-a", "synthetic-b"])

    def test_compare_to_research_to_extraction_propagates_winner(self):
        self.data["preferences"] = [{"attribute": "battery", "direction": "max", "weight": 1}]
        result = self.run_example()
        self.assertEqual(result["research"]["winner_id"], "synthetic-b")
        self.assertEqual(result["research"]["products"][0]["score"], 1)
        self.assertEqual(result["extraction"]["fields"]["recommended_product"]["value"],
                         "Synthetic Orbit Speaker")
        self.assertEqual(result["extraction"]["fields"]["battery_hours"]["value"], 12)
        self.assertTrue(result["extraction"]["complete"])

    def test_multidocument_synthesis_and_disagreement(self):
        research = self.run_example()["research"]
        a = research["products"][0]
        price, battery = a["findings"][:2]
        self.assertEqual(price["document_count"], 2)
        self.assertEqual(price["status"], "supported")
        self.assertEqual(battery["status"], "disputed")
        self.assertEqual(battery["distinct_values"], [10, 9])
        self.assertIsNone(battery["consensus_value"])
        self.assertEqual(len(research["disagreements"]), 1)
        self.assertEqual(len(a["unresolved_questions"]), 2)

    def test_missing_required_and_optional_fields(self):
        extraction = self.run_example()["extraction"]
        self.assertFalse(extraction["complete"])
        self.assertEqual(extraction["missing_fields"], ["battery_hours", "mass_grams"])
        self.assertEqual(extraction["required_missing"], ["battery_hours"])
        self.assertEqual(extraction["fields"]["battery_hours"]["reason"], "conflicting evidence")
        self.assertEqual(extraction["fields"]["mass_grams"]["reason"],
                         "no usable documentary evidence")
        self.assertEqual(extraction["fields"]["alternative_battery"]["value"], 12)

    def test_every_extracted_source_span_round_trips(self):
        docs = {d["id"]: d["text"] for d in self.data["documents"]}
        fields = self.run_example()["extraction"]["fields"]
        self.assertEqual(len(fields["verified_price"]["source_spans"]), 2)
        for field in fields.values():
            for span in field["source_spans"]:
                self.assertEqual(docs[span["document_id"]][span["start"]:span["end"]],
                                 span["quote"])
        self.assertEqual(fields["verified_price"]["source_spans"][0]["quote"], "80 USD")

    def test_unicode_crlf_and_whitespace_spans(self):
        self.data["documents"][0]["text"] = "SYNTHETIC café 🎵\r\n  Cost:  80 USD  \r\n"
        result = self.run_example()
        span = result["extraction"]["fields"]["verified_price"]["source_spans"][0]
        self.assertEqual(span["quote"], "80 USD")
        self.assertEqual(span["start"], self.data["documents"][0]["text"].index("80"))

    def test_no_documents_does_not_invent_catalog_evidence(self):
        self.data["documents"] = []
        result = self.run_example()
        self.assertEqual(result["research"]["documents_used"], [])
        self.assertIsNone(result["extraction"]["fields"]["verified_price"]["value"])
        self.assertEqual(result["extraction"]["fields"]["evidence_count"]["value"], 0)
        self.assertEqual(result["extraction"]["fields"]["open_questions"]["value"], 4)

    def test_document_changes_propagate_to_extraction(self):
        self.data["documents"][1]["text"] = self.data["documents"][1]["text"].replace("9 h", "600 min")
        result = self.run_example()
        self.assertEqual(result["extraction"]["fields"]["battery_hours"]["value"], 10)
        self.assertTrue(result["extraction"]["complete"])
        self.assertEqual(result["extraction"]["missing_fields"], ["mass_grams"])

    def test_catalog_document_conflict_even_with_single_document(self):
        self.data["documents"][2]["text"] = "SYNTHETIC\nbattery: 11 h\n"
        result = self.run_example()
        self.assertEqual(result["research"]["products"][1]["findings"][1]["status"], "disputed")
        self.assertIsNone(result["extraction"]["fields"]["alternative_battery"]["value"])

    def test_invalid_document_quantity_is_unresolved_not_invented(self):
        self.data["documents"][0]["text"] += "mass: unknown\n"
        result = self.run_example()
        questions = result["research"]["products"][0]["unresolved_questions"]
        self.assertTrue(any("invalid mass claim" in question for question in questions))
        self.assertIsNone(result["extraction"]["fields"]["mass_grams"]["value"])

    def test_ties_use_product_id_and_singleton_scores_are_one(self):
        self.data["preferences"] = [{"attribute": "price", "direction": "min", "weight": 1}]
        self.data["products"][1]["attributes"]["price"] = 80
        self.data["products"].reverse()
        result = app.compare(self.data)
        self.assertEqual(result["winner_id"], "synthetic-a")
        self.assertEqual([p["score"] for p in result["products"]], [1, 1])
        self.data["products"] = [self.data["products"][1]]
        self.data["documents"] = []
        self.data["extraction_schema"] = self.data["extraction_schema"][:1]
        self.assertEqual(app.compare(self.data)["products"][0]["score"], 1)

    def test_missing_comparison_attribute_is_zero_not_best(self):
        del self.data["products"][0]["attributes"]["Cost"]
        result = app.compare(self.data)
        a = next(p for p in result["products"] if p["id"] == "synthetic-a")
        self.assertEqual(a["preference_scores"]["price"], 0)
        self.assertIsNone(result["side_by_side"][0]["values"]["synthetic-a"])

    def test_unknown_keys_references_duplicates_and_empty_products_fail(self):
        changes = [
            lambda d: d.update({"unexpected": True}),
            lambda d: d.update({"products": []}),
            lambda d: d["documents"][0].update({"product_id": "absent"}),
            lambda d: d["products"][1].update({"id": "synthetic-a"}),
            lambda d: d["attributes"][1]["aliases"].append("Cost"),
            lambda d: d["products"][0]["attributes"].update({"price": 80}),
            lambda d: d["extraction_schema"][0].update({"product_id": "missing"}),
            lambda d: d["extraction_schema"][0].update({"type": "number"}),
            lambda d: d["extraction_schema"][2].update({"attribute": "missing"}),
        ]
        for change in changes:
            with self.subTest(change=change):
                invalid = copy.deepcopy(self.data)
                change(invalid)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(invalid)

    def test_booleans_nonfinite_numbers_units_and_weights_rejected(self):
        for value in [True, float("inf"), float("nan"), "12 weeks", {"value": 1, "unit": "days"}]:
            with self.subTest(value=value):
                invalid = copy.deepcopy(self.data)
                invalid["products"][0]["attributes"]["Battery life"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(invalid)
        for weight in [0, -1, True, float("inf")]:
            with self.subTest(weight=weight):
                self.data["preferences"][0]["weight"] = weight
                with self.assertRaises(app.ValidationError):
                    self.run_example()

    def test_stage_handoffs_reject_malformed_or_invalid_references(self):
        comparison = app.compare(self.data)
        bad = copy.deepcopy(comparison)
        bad["products"][0]["rank"] = 99
        with self.assertRaises(app.ValidationError):
            app.deep_research(self.data, bad)
        bad = copy.deepcopy(comparison)
        bad["products"][0]["score"] = "not numeric"
        with self.assertRaises(app.ValidationError):
            app.deep_research(self.data, bad)
        research = app.deep_research(self.data, comparison)
        research["products"][0]["findings"][0]["evidence"][0]["source_span"]["start"] = 0
        with self.assertRaises(app.ValidationError):
            app.extract(self.data, research)

    def test_dimensionless_quantities_and_large_finite_weights(self):
        self.data["attributes"][0]["unit"] = None
        self.data["attributes"][0]["units"] = {}
        self.data["products"][0]["attributes"]["Cost"] = 80
        self.data["documents"] = []
        for preference in self.data["preferences"]:
            preference["weight"] = 1e308
        result = self.run_example()
        self.assertEqual(result["comparison"]["products"][0]["score"], 0.666666666667)

    def test_extreme_numeric_ranges_and_close_large_integers(self):
        self.data["preferences"] = [{"attribute": "price", "direction": "min", "weight": 1}]
        for low, high in [(-1e308, 1e308), (10 ** 100, 10 ** 100 + 1)]:
            with self.subTest(low=low):
                self.data["products"][0]["attributes"]["Cost"] = low
                self.data["products"][1]["attributes"]["price"] = high
                result = app.compare(self.data)
                self.assertEqual(result["winner_id"], "synthetic-a")
                self.assertEqual([p["score"] for p in result["products"]], [1, 0])

    def test_research_handoff_rejects_fabricated_consensus_and_evidence(self):
        research = self.run_example()["research"]
        changes = [
            lambda r: r["products"][0]["findings"][0].update({"consensus_value": 999}),
            lambda r: r["products"][0]["findings"][0].update({"status": "disputed"}),
            lambda r: r["products"][0]["findings"][0].update({"document_count": 55}),
            lambda r: r["products"][0]["findings"][0]["evidence"][0].update({"value": 999}),
            lambda r: r.update({"disagreements": []}),
            lambda r: r.update({"documents_used": []}),
        ]
        for change in changes:
            with self.subTest(change=change):
                invalid = copy.deepcopy(research)
                change(invalid)
                with self.assertRaises(app.ValidationError):
                    app.extract(self.data, invalid)

    def test_comparison_handoff_rejects_inconsistent_table(self):
        comparison = app.compare(self.data)
        comparison["side_by_side"][0]["values"]["synthetic-a"] = 999
        with self.assertRaises(app.ValidationError):
            app.deep_research(self.data, comparison)

    def test_deterministic_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(self.run_example(), self.run_example())
        self.assertEqual(self.data, before)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_single_json_object(self):
        completed = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")

    def test_cli_usage_and_file_errors_are_json_exit_two(self):
        for args in [(), ("missing-synthetic-input.json",), ("a", "b"), (str(ROOT),)]:
            with self.subTest(args=args):
                completed = self.cli(*args)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stderr, "")
                self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_bad_json_validation_encoding_and_duplicate_keys(self):
        for text in ["{", "[]", '{"schema_version":"1.0","schema_version":"1.0"}',
                     '{"value":NaN}', '{"value":Infinity}', "null"]:
            with self.subTest(text=text):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=text), redirect_stdout(output):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")
        output = io.StringIO()
        with patch.object(Path, "read_text", side_effect=UnicodeError("bad UTF-8")), redirect_stdout(output):
            self.assertEqual(app.main(["synthetic-fixture.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
