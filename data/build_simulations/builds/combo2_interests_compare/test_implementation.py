"""Labeled synthetic fixtures exercise the actual integrated pipeline."""

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest

import implementation as app


def fixture():
    """Fixture OUTDOOR: two hiking choices, travel alternative, excluded bargain."""
    return json.loads(Path(__file__).with_name("example_input.json").read_text(encoding="utf-8"))


class IntegratedBuildTests(unittest.TestCase):
    def test_rank_then_compare_with_grounded_tradeoffs(self):
        result = app.build(fixture())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["recommendation"]["selected_ids"], ["trail-light", "trail-long"])
        self.assertEqual(result["comparison"]["candidate_ids"], ["trail-light", "trail-long"])
        attrs = {a["attribute"]: a for a in result["comparison"]["attributes"]}
        self.assertEqual(attrs["battery"]["preferred_ids"], ["trail-long"])
        self.assertEqual(attrs["weight"]["preferred_ids"], ["trail-light"])
        self.assertEqual(attrs["price"]["preferred_ids"], ["trail-light"])
        self.assertIn("24 hours", attrs["battery"]["explanation"])
        self.assertEqual(result["recommendation"]["selected"][0]["score"], 4)
        self.assertEqual(result["comparison"]["facts"][0]["provenance"]["input_index"], 0)
        self.assertTrue(result["comparison"]["facts"][0]["provenance"]["source"].startswith("fixture:"))
        tradeoff = result["comparison"]["tradeoffs"][0]
        self.assertEqual(tradeoff["advantages"],
                         {"trail-light": ["weight", "price"], "trail-long": ["battery"]})
        self.assertTrue(all(fact in result["comparison"]["facts"] for fact in tradeoff["facts"]))

    def test_exclusions_propagate_for_id_brand_and_tag(self):
        for kind, value, blocked in [
            ("product_ids", ["trail-light"], {"trail-light"}),
            ("brands", [" NORTH ", "Blocked"], {"trail-light", "blocked-pro"}),
            ("tags", [" TRAVEL "], {"trail-light", "city-mini", "blocked-pro"}),
        ]:
            with self.subTest(kind=kind):
                data = fixture()
                data["preferences"]["exclusions"][kind] = value
                result = app.build(data)
                self.assertFalse(blocked & set(result["recommendation"]["selected_ids"]))
                self.assertFalse(blocked & set(result["comparison"]["candidate_ids"]))
                self.assertFalse(blocked & {f["product_id"] for f in result["comparison"]["facts"]})
                for tradeoff in result["comparison"]["tradeoffs"]:
                    self.assertFalse(blocked & set(tradeoff["product_ids"]))
                for item in result["comparison"]["attributes"]:
                    self.assertFalse(blocked & set(item["preferred_ids"]))

    def test_interest_change_changes_downstream_comparison(self):
        data = fixture()
        first = app.build(data)
        data["preferences"]["interests"] = {"travel": 5, "hiking": 1}
        second = app.build(data)
        self.assertEqual(second["comparison"]["candidate_ids"], ["trail-light", "city-mini"])
        self.assertNotEqual(first["comparison"], second["comparison"])
        self.assertEqual(second["comparison"]["attributes"][0]["preferred_ids"], ["trail-light"])

    def test_product_fact_change_changes_downstream_explanation(self):
        data = fixture()
        first = app.build(data)
        data["products"][1]["attributes"]["battery"]["value"] = 6
        second = app.build(data)
        self.assertEqual(first["comparison"]["candidate_ids"], second["comparison"]["candidate_ids"])
        self.assertEqual(second["comparison"]["attributes"][0]["preferred_ids"], ["trail-light"])
        self.assertNotEqual(first["comparison"]["attributes"][0]["explanation"],
                            second["comparison"]["attributes"][0]["explanation"])

    def test_common_constraints_filter_before_comparison(self):
        for change, reason in [
            ({"price": 999}, "over_budget"),
            ({"price": None}, "unknown_price"),
            ({"currency": "EUR"}, "currency_mismatch"),
        ]:
            with self.subTest(reason=reason):
                data = fixture()
                data["products"][0].update(change)
                result = app.build(data)
                self.assertNotIn("trail-light", result["comparison"]["candidate_ids"])
                rejection = next(r for r in result["recommendation"]["rejected"]
                                 if r["product_id"] == "trail-light")
                self.assertIn(reason, rejection["reasons"])
        data = fixture()
        data["preferences"]["constraints"]["required_tags"] = ["travel"]
        result = app.build(data)
        self.assertNotIn("trail-long", result["comparison"]["candidate_ids"])

    def test_stable_ties_across_catalog_reordering(self):
        data = fixture()
        data["preferences"]["interests"] = {"hiking": 1}
        first = app.build(data)
        data["products"].reverse()
        second = app.build(data)
        self.assertEqual(first["recommendation"]["selected_ids"], ["trail-light", "trail-long"])
        self.assertEqual(first["comparison"]["candidate_ids"], second["comparison"]["candidate_ids"])

    def test_missing_noncomparable_and_no_unit_conversion(self):
        data = fixture()
        del data["products"][0]["attributes"]["battery"]
        data["products"][0]["attributes"]["weight"] = {"value": 0.25, "unit": "kg"}
        result = app.build(data)
        battery, weight = result["comparison"]["attributes"][:2]
        self.assertEqual(battery["status"], "insufficient_data")
        self.assertEqual(battery["missing_ids"], ["trail-light"])
        self.assertEqual(weight["status"], "noncomparable")
        self.assertEqual(weight["preferred_ids"], [])
        data["products"][0]["attributes"]["weight"] = {"value": "light"}
        self.assertEqual(app.build(data)["comparison"]["attributes"][1]["status"], "noncomparable")

    def test_partial_numeric_comparison_limits_conclusions(self):
        data = fixture()
        data["candidate_limit"] = 3
        data["products"][2]["attributes"]["battery"]["value"] = None
        battery = app.build(data)["comparison"]["attributes"][0]
        self.assertEqual(battery["status"], "comparable")
        self.assertEqual(battery["missing_ids"], ["city-mini"])
        self.assertIn("among known values", battery["explanation"])

    def test_no_match_empty_catalog_and_single_candidate(self):
        data = fixture()
        data["preferences"]["interests"] = {"astronomy": 1}
        result = app.build(data)
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["comparison"]["candidate_ids"], [])
        self.assertEqual(result["comparison"]["facts"], [])
        data["products"] = []
        self.assertEqual(app.build(data)["status"], "no_match")
        data = fixture()
        data["candidate_limit"] = 1
        result = app.build(data)
        self.assertTrue(all(a["status"] == "insufficient_data"
                            for a in result["comparison"]["attributes"]))

    def test_normalization_and_no_mutation(self):
        data = fixture()
        data["products"][0]["tags"] = [" HIKING ", "Travel"]
        data["products"][0]["attributes"]["battery"]["unit"] = " HOURS "
        before = copy.deepcopy(data)
        result = app.build(data)
        self.assertEqual(data, before)
        self.assertEqual(result["recommendation"]["selected_ids"][0], "trail-light")
        self.assertEqual(result["comparison"]["attributes"][0]["status"], "comparable")

    def test_invalid_and_duplicate_inputs(self):
        mutations = [
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["products"][0].update(id=" "),
            lambda d: d["products"][0].update(tags=["hiking", " HIKING "]),
            lambda d: d["products"][0].update(price=True),
            lambda d: d["products"][0].update(price=-1),
            lambda d: d["products"][0].update(currency="dollars"),
            lambda d: d["products"][0]["attributes"].update(BATTERY={"value": 2}),
            lambda d: d["products"][0]["attributes"].update(price={"value": 2}),
            lambda d: d["products"][0]["attributes"].update(battery={"value": []}),
            lambda d: d["products"][0]["attributes"].update(battery={"value": True, "unit": "h"}),
            lambda d: d["products"][0]["attributes"].update(battery={"value": float("nan")}),
            lambda d: d["preferences"].update(interests={}),
            lambda d: d["preferences"].update(interests={"hiking": -1}),
            lambda d: d["preferences"].update(interests={"hiking": float("inf")}),
            lambda d: d["preferences"].update(interests={"hiking": 1e308, "travel": 1e308}),
            lambda d: d["preferences"]["constraints"].update(max_price=False),
            lambda d: d["preferences"]["attribute_preferences"].update(battery="best"),
            lambda d: d["preferences"]["attribute_preferences"].update(unknown="higher"),
            lambda d: d.update(candidate_limit=True),
            lambda d: d.update(candidate_limit=0),
            lambda d: d.update(comparison_attributes=["price", " PRICE "]),
            lambda d: d.update(unknown=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(invalid_fixture=index):
                data = fixture()
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.build(data)

    def test_callback_only_receives_selection_and_preserves_data(self):
        def callback(context):
            self.assertEqual(context["candidate_ids"], ["trail-light", "trail-long"])
            claim = copy.deepcopy(context["facts"][0])
            context["facts"].clear()
            context["candidate_ids"].clear()
            return {"claims": [claim]}
        result = app.build(fixture(), callback)
        self.assertEqual(len(result["comparison"]["facts"]), 8)
        self.assertEqual(result["recommendation"]["selected_ids"], ["trail-light", "trail-long"])
        self.assertIn("12 hours", result["callback_explanations"][0])

    def test_callback_rejects_invented_ids_facts_provenance_and_prose(self):
        mutations = [
            lambda c: c.update(product_id="blocked-pro"),
            lambda c: c.update(product_id="not-in-catalog"),
            lambda c: c.update(value=999),
            lambda c: c.update(value=True),
            lambda c: c.update(unit="days"),
            lambda c: c.update(attribute="invented"),
            lambda c: c.update(provenance={"input_index": 0, "source": "invented"}),
            lambda c: c.update(text="This is objectively the best."),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(callback_fixture=index):
                def callback(context):
                    claim = context["facts"][0]
                    mutate(claim)
                    return {"claims": [claim]}
                with self.assertRaises(app.ValidationError):
                    app.build(fixture(), callback)
        with self.assertRaises(app.ValidationError):
            app.build(fixture(), lambda c: {"claims": [c["facts"][0], c["facts"][0]]})

    def test_json_loader_rejects_duplicate_keys(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"products": [], "products": []}', object_pairs_hook=app.unique_object)

    def test_cli_success_and_errors_are_json(self):
        for args, expected_code, expected_status in [
            ([str(Path(__file__).with_name("example_input.json"))], 0, "ok"),
            ([], 2, "error"),
            ([str(Path(__file__).with_name("does-not-exist.json"))], 2, "error"),
        ]:
            with self.subTest(args=args):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = app.main(args)
                self.assertEqual(code, expected_code)
                self.assertEqual(json.loads(stream.getvalue())["status"], expected_status)


if __name__ == "__main__":
    unittest.main()
