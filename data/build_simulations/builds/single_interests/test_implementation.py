"""Acceptance tests. All catalogs and explanation callables below are fixtures.

No fixture represents an actual LLM invocation or a production catalog.
"""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as impl


HERE = Path(__file__).resolve().parent


def catalog_fixture():
    return {
        "products": [
            {"id": "b", "name": "Beta", "attributes": {"color": "Blue", "tags": ["walk", "trail"]}},
            {"id": "a", "name": "Alpha", "attributes": {"color": "blue", "tags": ["trail"]}},
            {"id": "c", "name": "Gamma", "attributes": {"color": "red", "tags": ["city"]}},
        ],
        "preferences": [{"attribute": "color", "value": "BLUE", "weight": 2},
                        {"attribute": "tags", "value": "trail"}],
    }


def explanation_fixture(envelope):
    """Injected structured-response fixture; deliberately not a real LLM."""
    return {"explanations": [
        {"product_id": product["id"],
         "facts": [{"attribute": "color", "value": product["attributes"]["color"]}]}
        for product in envelope["products"]
    ]}


class RankingTests(unittest.TestCase):
    def test_weighted_ranking_and_stable_ties(self):
        data = catalog_fixture()
        result = impl.recommend(data)
        self.assertEqual([x["product_id"] for x in result["recommendations"]], ["a", "b"])
        self.assertEqual([x["score"] for x in result["recommendations"]], [3, 3])
        data["products"].reverse()
        self.assertEqual(impl.recommend(data), result)

    def test_unequal_scores_and_limit(self):
        data = catalog_fixture()
        data["preferences"].append({"attribute": "tags", "value": "walk", "weight": 5})
        data["limit"] = 1
        result = impl.recommend(data)
        self.assertEqual(result["recommendations"][0]["product_id"], "b")
        self.assertEqual(result["recommendations"][0]["score"], 8)
        self.assertEqual(result["total_matches"], 2)

    def test_exclusions_override_all_preferences(self):
        data = catalog_fixture()
        data["exclusions"] = [{"attribute": "tags", "value": " WALK "}]
        data["excluded_product_ids"] = ["a"]
        result = impl.recommend(data)
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["reason"], "no_positive_preference_matches")
        self.assertEqual(result["excluded_count"], 2)
        self.assertEqual(result["recommendations"], [])

    def test_all_excluded_is_explicit(self):
        data = catalog_fixture()
        data["excluded_product_ids"] = ["a", "b", "c"]
        self.assertEqual(impl.recommend(data)["reason"], "all_products_excluded")

    def test_missing_attribute_is_not_invented(self):
        data = catalog_fixture()
        data["preferences"] = [{"attribute": "waterproof", "value": True}]
        result = impl.recommend(data)
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["total_matches"], 0)

    def test_explanations_use_actual_values_only(self):
        result = impl.recommend(catalog_fixture())
        beta = result["recommendations"][1]
        self.assertIn('color = "Blue"', beta["explanation"])
        self.assertIn('tags = "trail"', beta["explanation"])
        self.assertNotIn("waterproof", beta["explanation"])
        self.assertEqual(beta["matched_preferences"][0]["product_value"], "Blue")

    def test_bool_does_not_match_numeric_but_numbers_match(self):
        data = {"products": [
            {"id": "bool", "name": "Boolean", "attributes": {"x": True}},
            {"id": "number", "name": "Number", "attributes": {"x": 1.0}}],
            "preferences": [{"attribute": "x", "value": 1}]}
        self.assertEqual([x["product_id"] for x in impl.recommend(data)["recommendations"]],
                         ["number"])

    def test_false_zero_and_list_values_are_matches(self):
        for value in (False, 0, "hello"):
            with self.subTest(value=value):
                data = {"products": [{"id": "x", "name": "X",
                                      "attributes": {"x": [value]}}],
                        "preferences": [{"attribute": "x", "value": value}]}
                self.assertEqual(impl.recommend(data)["recommendations"][0]["score"], 1)

    def test_repeated_list_values_do_not_multiply_weight(self):
        data = catalog_fixture()
        data["products"][0]["attributes"]["tags"] = ["trail", "trail"]
        self.assertEqual(impl.recommend(data)["recommendations"][1]["score"], 3)

    def test_inputs_not_mutated(self):
        data = catalog_fixture()
        before = copy.deepcopy(data)
        impl.recommend(data, explanation_fixture)
        self.assertEqual(data, before)


class ValidationTests(unittest.TestCase):
    def test_invalid_catalogs(self):
        invalid = [None, [], {}, {"products": [], "preferences": []}]
        for replacement in ([], {}, None, [None], [
            {"id": "a", "name": "A", "attributes": {"x": 1}},
            {"id": "a", "name": "B", "attributes": {"x": 2}},
        ]):
            data = catalog_fixture()
            data["products"] = replacement
            invalid.append(data)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(impl.ValidationError):
                impl.recommend(data)

    def test_invalid_product_fields(self):
        for key, value in (
            ("id", ""), ("id", 3), ("name", " "), ("attributes", {}),
            ("attributes", []), ("attributes", {"": "x"}),
            ("attributes", {"x": []}), ("attributes", {"x": None}),
            ("attributes", {"x": {"nested": True}}),
            ("attributes", {"x": [["nested"]]}),
            ("attributes", {"x": float("nan")}),
            ("attributes", {"x": float("inf")}),
        ):
            data = catalog_fixture()
            data["products"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(impl.ValidationError):
                impl.recommend(data)

    def test_invalid_rules_and_options(self):
        for field, value in (
            ("preferences", []), ("preferences", "blue"),
            ("preferences", [{"attribute": "x"}]),
            ("preferences", [{"attribute": "x", "value": []}]),
            ("preferences", [{"attribute": "x", "value": 1, "weight": True}]),
            ("preferences", [{"attribute": "x", "value": 1, "weight": 0}]),
            ("preferences", [{"attribute": "x", "value": 1, "weight": 1.5}]),
            ("exclusions", [{"attribute": "x", "value": 1, "weight": 2}]),
            ("excluded_product_ids", [1]), ("excluded_product_ids", "a"),
            ("limit", False), ("limit", 0), ("limit", 101), ("limit", 2.2),
            ("unexpected", 1),
        ):
            data = catalog_fixture()
            data[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(impl.ValidationError):
                impl.recommend(data)

    def test_duplicate_normalized_rules_rejected(self):
        data = catalog_fixture()
        data["preferences"].append({"attribute": "color", "value": " blue "})
        with self.assertRaisesRegex(impl.ValidationError, "duplicate"):
            impl.recommend(data)

    def test_bounds(self):
        data = catalog_fixture()
        data["products"] *= 3334
        with self.assertRaisesRegex(impl.ValidationError, "10000"):
            impl.recommend(data)
        data = catalog_fixture()
        data["products"][0]["attributes"]["tags"] = ["x"] * 101
        with self.assertRaisesRegex(impl.ValidationError, "100"):
            impl.recommend(data)


class AdapterTests(unittest.TestCase):
    def test_fixture_adapter_gets_supplied_grounding(self):
        captured = []

        def fixture(envelope):
            captured.append(envelope)
            return explanation_fixture(envelope)

        data = catalog_fixture()
        result = impl.recommend(data, fixture)
        self.assertEqual(result["explanation_source"], "validated_adapter_facts")
        self.assertEqual({x["id"] for x in captured[0]["products"]}, {"a", "b"})
        self.assertEqual(captured[0]["products"][0]["attributes"], data["products"][0]["attributes"])
        self.assertEqual(result["recommendations"][0]["adapter_explanation"],
                         'Supplied product facts: color = "blue".')
        self.assertEqual([x["product_id"] for x in result["recommendations"]], ["a", "b"])

    def test_bad_adapter_responses_rejected(self):
        responses = [
            None, "not JSON", {}, {"explanations": []},
            {"explanations": [{"product_id": "unknown", "facts": []}]},
            {"explanations": [{"product_id": 3, "facts": []}]},
            {"explanations": [{"product_id": "a", "facts": []}]},
            {"explanations": [{"product_id": "a", "facts":
                              [{"attribute": "color", "value": "invented"}]}]},
            {"explanations": [{"product_id": "a", "facts":
                              [{"attribute": "waterproof", "value": True}]}]},
            {"explanations": [{"product_id": "a", "facts":
                              [{"attribute": "color", "value": "blue"}],
                              "prose": "invented claims"}]},
            {"explanations": [{"product_id": "a", "facts":
                              [{"attribute": "color", "value": "blue"}]}]},
        ]
        valid = explanation_fixture({"products": catalog_fixture()["products"][:2]})
        duplicate = copy.deepcopy(valid)
        duplicate["explanations"].append(copy.deepcopy(duplicate["explanations"][0]))
        responses.append(duplicate)
        duplicate_fact = copy.deepcopy(valid)
        facts = duplicate_fact["explanations"][0]["facts"]
        facts.append(copy.deepcopy(facts[0]))
        responses.append(duplicate_fact)
        for response in responses:
            with self.subTest(response=response), self.assertRaises(impl.ExplanationAdapterError):
                impl.recommend(catalog_fixture(), lambda envelope: response)

    def test_excluded_or_truncated_ids_cannot_be_reintroduced(self):
        for option in ({"excluded_product_ids": ["b"]}, {"limit": 1}):
            data = catalog_fixture()
            data.update(option)
            response = {"explanations": [
                {"product_id": "b", "facts": [{"attribute": "color", "value": "Blue"}]}]}
            with self.subTest(option=option), self.assertRaisesRegex(
                    impl.ExplanationAdapterError, "unlisted"):
                impl.recommend(data, lambda envelope: response)

    def test_callback_errors_are_explicit(self):
        def failing_fixture(envelope):
            raise RuntimeError("fixture provider unavailable")

        with self.assertRaisesRegex(impl.ExplanationAdapterError, "RuntimeError.*unavailable"):
            impl.recommend(catalog_fixture(), failing_fixture)
        with self.assertRaisesRegex(impl.ExplanationAdapterError, "callable"):
            impl.recommend(catalog_fixture(), "not callable")

    def test_callback_cannot_mutate_grounding(self):
        def malicious_fixture(envelope):
            envelope["products"][0]["attributes"]["color"] = "invented"
            return explanation_fixture(envelope)

        with self.assertRaisesRegex(impl.ExplanationAdapterError, "ungrounded"):
            impl.recommend(catalog_fixture(), malicious_fixture)

    def test_no_match_does_not_call_adapter(self):
        data = catalog_fixture()
        data["excluded_product_ids"] = ["a", "b", "c"]

        def forbidden_fixture(envelope):
            self.fail("adapter must not be invoked for no-match")

        self.assertEqual(impl.recommend(data, forbidden_fixture)["status"], "no_match")


class CliTests(unittest.TestCase):
    def test_real_cli_example(self):
        process = subprocess.run(
            [sys.executable, str(HERE / "implementation.py"), str(HERE / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual([x["product_id"] for x in result["recommendations"]], ["p01", "p02"])
        self.assertEqual([x["score"] for x in result["recommendations"]], [5, 3])
        self.assertEqual(result["excluded_count"], 1)

    def test_missing_file_cli_error(self):
        process = subprocess.run(
            [sys.executable, str(HERE / "implementation.py"), str(HERE / "missing-fixture.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_json_parser_rejects_malformed_duplicate_and_nonfinite(self):
        for text in ("{", '{"products": [], "products": []}',
                     '{"products": NaN}', '{"products": Infinity}',
                     '{"products": [], "preferences": []}'):
            output = io.StringIO()
            with self.subTest(text=text), patch.object(
                    Path, "open", return_value=io.StringIO(text)), redirect_stdout(output):
                self.assertEqual(impl.main(["fixture.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
