import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, compare, quantity, unique_object


ROOT = Path(__file__).resolve().parent


def fixture():
    return json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))


class ComparisonTests(unittest.TestCase):
    def test_example_normalization_ranking_and_tradeoffs(self):
        result = compare(fixture())
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["normalized_products"]["fixture-light"]["mass"], .90718474)
        self.assertEqual(result["normalized_products"]["fixture-light"]["battery"], 60)
        self.assertEqual([p["product_id"] for p in result["ranking"]],
                         ["fixture-budget", "fixture-light"])
        self.assertEqual([p["score"] for p in result["ranking"]], [.5, .5])
        self.assertEqual(result["excluded"][0]["product_id"], "fixture-over-budget")
        self.assertEqual({t["assessment"] for t in result["tradeoffs"]},
                         {"advantage", "disadvantage"})
        self.assertTrue(all(t["product_id"] == "fixture-budget" and
                            t["compared_to"] == "fixture-light" for t in result["tradeoffs"]))

    def test_stable_ties_follow_input_order(self):
        data = fixture()
        data["products"].reverse()
        self.assertEqual(compare(data)["ranking"][0]["product_id"], "fixture-light")

    def test_common_units(self):
        for source, target, value, expected in [
            ("cm", "m", 100, 1), ("in", "cm", 1, 2.54),
            ("g", "kg", 1000, 1), ("mL", "L", 1000, 1),
            ("kW", "W", 2, 2000), ("kWh", "Wh", 2, 2000),
            ("min", "s", 2, 120), ("GiB", "MiB", 2, 2048),
            ("GB", "MB", 2, 2000), ("USD", "USD", 2, 2),
        ]:
            with self.subTest(source=source):
                self.assertAlmostEqual(quantity({"value": value, "unit": source},
                                                {"type": "numeric", "unit": target}, "test"),
                                       expected)

    def test_missing_and_noncomparable(self):
        result = compare(fixture())
        self.assertEqual(result["missing_attributes"]["fixture-budget"], ["warranty"])
        self.assertEqual(result["noncomparable_attributes"],
                         [{"attribute": "warranty", "reason": "fewer_than_two_known_values"}])
        self.assertFalse(result["comparison"][-1]["comparable"])
        self.assertIsNone(result["comparison"][-1]["values"]["fixture-budget"])

    def test_missing_preference_is_penalized_not_invented(self):
        data = fixture()
        data["preferences"] = {"warranty": {"direction": "max", "weight": 1}}
        result = compare(data)
        self.assertEqual([p["score"] for p in result["ranking"]], [1, 0])
        self.assertEqual(result["tradeoffs"], [])

    def test_missing_constraint_excludes(self):
        data = fixture()
        data["constraints"] = {"warranty": {"min": 1}}
        result = compare(data)
        self.assertEqual([p["product_id"] for p in result["ranking"]], ["fixture-light"])
        self.assertEqual(result["excluded"][0]["reasons"][0]["reason"], "missing")

    def test_empty_products(self):
        data = fixture()
        data["products"] = []
        result = compare(data)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["ranking"], [])
        self.assertEqual(result["tradeoffs"], [])

    def test_no_compatible_products(self):
        data = fixture()
        data["constraints"] = {"price": {"max": 1}}
        result = compare(data)
        self.assertEqual(result["status"], "no_compatible_products")
        self.assertEqual(result["ranking"], [])
        self.assertEqual(len(result["excluded"]), 3)

    def test_empty_preferences_and_schema(self):
        result = compare({"fixture": True, "schema": {}, "products": [
            {"id": "b", "attributes": {}}, {"id": "a", "attributes": {}}]})
        self.assertEqual([p["product_id"] for p in result["ranking"]], ["b", "a"])
        self.assertEqual([p["score"] for p in result["ranking"]], [0, 0])

    def test_categorical_and_unit_constraints(self):
        data = fixture()
        data["constraints"] = {"finish": {"equals": "silver"},
                               "mass": {"max": {"value": 1000, "unit": "g"}}}
        self.assertEqual(compare(data)["ranking"][0]["product_id"], "fixture-light")

    def test_bad_units(self):
        for unit in ["watts", "USD", [], None]:
            data = fixture()
            data["products"][0]["attributes"]["mass"] = {"value": 1, "unit": unit}
            with self.subTest(unit=unit), self.assertRaises(ValidationError):
                compare(data)

    def test_conflicting_schema(self):
        for spec in [{"type": "categorical", "unit": "kg"},
                     {"type": "numeric", "unit": "unknown"},
                     {"type": "numeric", "units": "kg"}, {"type": "other"}]:
            data = fixture()
            data["schema"]["mass"] = spec
            with self.subTest(spec=spec), self.assertRaises(ValidationError):
                compare(data)

    def test_bad_numbers(self):
        for value in [True, "3 kg", float("nan"), float("inf"), 10 ** 1000]:
            data = fixture()
            data["products"][0]["attributes"]["mass"] = value
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValidationError):
                compare(data)

    def test_invalid_policies(self):
        mutations = [
            ("preferences", {"mass": {"direction": "min", "weight": 0}}),
            ("preferences", {"mass": {"direction": "sideways", "weight": 1}}),
            ("preferences", {"ghost": {"direction": "min", "weight": 1}}),
            ("preferences", {"finish": {"target": 3, "weight": 1}}),
            ("constraints", {"mass": {"min": 2, "max": 1}}),
            ("constraints", {"mass": {}}),
            ("constraints", {"finish": {"equals": 3}}),
        ]
        for field, value in mutations:
            data = fixture()
            data[field] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                compare(data)

    def test_product_validation(self):
        for product in [
            {"id": "", "attributes": {}}, {"id": "fixture-budget", "attributes": {}},
            {"id": "new", "attributes": {"ghost": 1}},
            {"id": "new", "attributes": {"finish": 2}},
        ]:
            data = fixture()
            data["products"].append(product)
            with self.subTest(product=product), self.assertRaises(ValidationError):
                compare(data)

    def test_fixture_label_required(self):
        data = fixture()
        data["fixture"] = False
        with self.assertRaises(ValidationError):
            compare(data)

    def test_deterministic_and_no_input_mutation(self):
        data = fixture()
        before = copy.deepcopy(data)
        self.assertEqual(compare(data), compare(data))
        self.assertEqual(data, before)

    def test_large_values_and_weights_stay_finite(self):
        data = {"fixture": True, "schema": {"x": {"type": "numeric"}},
                "products": [{"id": "low", "attributes": {"x": -1e308}},
                             {"id": "high", "attributes": {"x": 1e308}}],
                "preferences": {"x": {"direction": "max", "weight": 1e308}}}
        result = compare(data)
        self.assertEqual([p["score"] for p in result["ranking"]], [1, 0])
        json.dumps(result, allow_nan=False)

    def test_valid_injected_explanation(self):
        def callback(result):
            result["ranking"].clear()
            return {"claims": [{"product_id": "fixture-budget", "attribute": "mass",
                                "value": 1.5}]}
        result = compare(fixture(), callback)
        self.assertEqual(len(result["ranking"]), 2)
        self.assertEqual(result["explanation"]["claims"][0]["value"], 1.5)

    def test_reject_ungrounded_explanations(self):
        claims = [
            {"product_id": "invented", "attribute": "mass", "value": 1.5},
            {"product_id": "fixture-budget", "attribute": "invented", "value": 1},
            {"product_id": "fixture-budget", "attribute": "mass", "value": 2},
            {"product_id": "fixture-budget", "attribute": "warranty", "value": None},
            {"product_id": "fixture-budget", "attribute": "mass", "value": True},
        ]
        for claim in claims:
            with self.subTest(claim=claim), self.assertRaises(ValidationError):
                compare(fixture(), lambda result: {"claims": [claim]})
        with self.assertRaises(ValidationError):
            compare(fixture(), lambda result: {"claims": [], "prose": "invented"})

    def test_duplicate_json_keys(self):
        with self.assertRaises(ValidationError):
            json.loads('{"mass": 1, "mass": 2}', object_pairs_hook=unique_object)

    def test_cli_fixture(self):
        process = subprocess.run([sys.executable, str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_bad_json_and_validation(self):
        for text in ['{', '{"fixture":true,"fixture":false}',
                     '{"fixture":true,"schema":{},"products":[],"preferences":NaN}',
                     '{"fixture":false,"schema":{},"products":[]}']:
            process = subprocess.run([sys.executable, str(ROOT / "implementation.py"), "-"],
                                     input=text, capture_output=True, text=True, check=False)
            with self.subTest(text=text):
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stdout, "")
                self.assertIn("error", json.loads(process.stderr))


if __name__ == "__main__":
    unittest.main()
