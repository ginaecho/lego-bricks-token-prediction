"""Standard-library tests using only clearly labeled synthetic fixtures."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


def fixture():
    return json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def result(self):
        return app.run_pipeline(self.data)["stages"]

    def test_grounded_answer_is_exact_source_text(self):
        faq = self.result()["faq"]
        self.assertEqual(faq["status"], "answered")
        self.assertEqual(faq["answer"], self.data["request"]["knowledge_base"][0]["answer"])
        self.assertEqual(faq["sources"], ["faq-runtime"])
        self.assertEqual(faq["confidence"], 1.0)

    def test_no_match_explicitly_abstains_and_propagates(self):
        self.data["request"]["question"] = "Explain quantum entanglement"
        stages = self.result()
        self.assertEqual(stages["faq"]["status"], "abstained")
        self.assertIsNone(stages["faq"]["answer"])
        self.assertEqual(stages["faq"]["sources"], [])
        self.assertEqual(stages["adaptive"]["topics"], [])
        self.assertEqual([step["id"] for step in stages["adaptive"]["steps"]], ["account"])
        self.assertIn("FAQ abstained", stages["adaptive"]["explanation"])
        self.assertEqual(stages["adaptive"]["weight_reasons"]["weight"], "Default balanced priority.")

    def test_stopwords_only_abstain(self):
        self.data["request"]["question"] = "What is the?"
        self.assertEqual(self.result()["faq"]["reason"], "no_match")

    def test_empty_knowledge_base_abstains(self):
        self.data["request"]["knowledge_base"] = []
        self.assertEqual(self.result()["faq"]["status"], "abstained")

    def test_conflicting_top_matches_abstain(self):
        other = copy.deepcopy(self.data["request"]["knowledge_base"][0])
        other.update(id="faq-conflict", answer="A conflicting synthetic answer.")
        self.data["request"]["knowledge_base"].append(other)
        self.assertEqual(self.result()["faq"]["reason"], "ambiguous_match")

    def test_single_keyword_question_can_match(self):
        self.data["request"]["question"] = "runtime"
        self.assertEqual(self.result()["faq"]["status"], "answered")

    def test_faq_topics_select_only_relevant_onboarding(self):
        adaptive = self.result()["adaptive"]
        self.assertEqual([step["id"] for step in adaptive["steps"]], ["account", "brightness"])
        self.assertEqual(adaptive["topics"], ["portable-lighting"])
        self.assertTrue(all(step["explanation"] and step["selection_reason"] for step in adaptive["steps"]))

    def test_expert_includes_prerequisites_outside_audience(self):
        self.data["request"]["profile"]["experience"] = "expert"
        steps = self.result()["adaptive"]["steps"]
        self.assertEqual([step["id"] for step in steps], ["account", "brightness", "advanced-profile"])
        self.assertIn("Required prerequisite", steps[0]["selection_reason"])

    def test_completed_prerequisites_are_not_repeated(self):
        self.data["request"]["profile"]["completed_steps"] = ["account"]
        steps = self.result()["adaptive"]["steps"]
        self.assertEqual([step["id"] for step in steps], ["brightness"])
        self.assertEqual(steps[0]["prerequisites"], ["account"])

    def test_self_service_preference_changes_delivery(self):
        self.data["request"]["profile"]["onboarding_mode"] = "self_service"
        adaptive = self.result()["adaptive"]
        self.assertEqual(adaptive["mode"], "self_service")
        self.assertTrue(all("independently" in step["delivery"] for step in adaptive["steps"]))

    def test_priorities_override_faq_hints_and_flow_to_compare(self):
        stages = self.result()
        adaptive, compare = stages["adaptive"], stages["compare"]
        self.assertAlmostEqual(adaptive["weights"]["battery"], 5 / 11)
        self.assertAlmostEqual(adaptive["weights"]["weight"], 2 / 11)
        self.assertEqual(adaptive["weight_reasons"]["battery"], "Explicit user priority.")
        self.assertEqual(adaptive["weight_reasons"]["weight"], "Grounded FAQ attribute hint.")
        self.assertEqual(compare["weights"], adaptive["weights"])
        self.assertEqual(compare["max_price_usd"], adaptive["max_price_usd"])

    def test_faq_change_propagates_to_plan_and_ranking(self):
        self.data["request"]["profile"]["priorities"] = {}
        self.data["request"]["knowledge_base"][0]["attribute_hints"] = ["battery"]
        first = self.result()
        self.data["request"]["question"] = "What is the synthetic return policy?"
        second = self.result()
        self.assertNotEqual(first["adaptive"]["topics"], second["adaptive"]["topics"])
        self.assertEqual(second["adaptive"]["topics"], ["returns"])
        self.assertNotEqual(first["compare"]["weights"], second["compare"]["weights"])
        self.assertNotEqual(first["compare"]["ranking"][0]["product_id"],
                            second["compare"]["ranking"][0]["product_id"])

    def test_attribute_aliases_and_units_normalize(self):
        normalized = self.result()["compare"]["normalized_products"][0]["attributes"]
        self.assertEqual(normalized, {"price": 40.0, "weight": 250.0, "battery": 4.0, "rating": 4.0})
        raw = {"weight": {"value": 1, "unit": "lb"}}
        self.assertAlmostEqual(app.normalize_attributes(raw)["weight"], 453.59237)

    def test_side_by_side_and_budget_exclusions(self):
        compare = self.result()["compare"]
        self.assertEqual(compare["excluded"], [{"product_id": "light-c", "reason": "over_budget"}])
        self.assertEqual(compare["ranking"][0]["product_id"], "light-b")
        self.assertAlmostEqual(compare["ranking"][0]["score"], 6 / 11, places=6)
        self.assertEqual(compare["side_by_side"][2]["values"], {"light-a": 4.0, "light-b": 10.0})
        self.assertEqual([row["rank"] for row in compare["ranking"]], [1, 2])

    def test_preferences_change_ranking(self):
        self.data["request"]["profile"]["priorities"] = {"price": 100}
        self.assertEqual(self.result()["compare"]["ranking"][0]["product_id"], "light-a")

    def test_missing_values_receive_zero_contribution(self):
        self.data["request"]["products"][1]["attributes"].pop("battery")
        ranked = self.result()["compare"]["ranking"]
        row = next(item for item in ranked if item["product_id"] == "light-b")
        self.assertEqual(row["missing_attributes"], ["battery"])
        self.assertEqual(row["contributions"]["battery"], 0.0)

    def test_unknown_price_excluded_only_with_budget(self):
        self.data["request"]["products"][0]["attributes"].pop("cost")
        compare = self.result()["compare"]
        self.assertIn({"product_id": "light-a", "reason": "price_unknown_under_budget_constraint"},
                      compare["excluded"])
        self.data["request"]["profile"]["max_price_usd"] = None
        self.assertEqual(len(self.result()["compare"]["ranking"]), 3)

    def test_all_excluded_and_empty_catalog_are_valid(self):
        self.data["request"]["profile"]["max_price_usd"] = 0
        self.assertEqual(self.result()["compare"]["ranking"], [])
        self.data["request"]["products"] = []
        compare = self.result()["compare"]
        self.assertEqual(compare["ranking"], [])
        self.assertEqual(compare["excluded"], [])
        self.assertTrue(all(row["values"] == {} for row in compare["side_by_side"]))

    def test_equal_values_and_ties_are_deterministic(self):
        products = self.data["request"]["products"]
        products[1]["attributes"] = copy.deepcopy(products[0]["attributes"])
        ranked = self.result()["compare"]["ranking"]
        self.assertEqual([row["product_id"] for row in ranked], ["light-a", "light-b"])
        self.assertEqual([row["score"] for row in ranked], [1.0, 1.0])

    def test_empty_onboarding_and_all_missing_attributes(self):
        self.data["request"]["onboarding_steps"] = []
        self.data["request"]["profile"]["max_price_usd"] = None
        self.data["request"]["products"] = [
            {"id": "empty", "name": "SYNTHETIC unknown product", "attributes": {}}
        ]
        stages = self.result()
        self.assertEqual(stages["adaptive"]["steps"], [])
        row = stages["compare"]["ranking"][0]
        self.assertEqual(row["score"], 0.0)
        self.assertEqual(row["missing_attributes"], list(app.ATTRIBUTES))

    def test_equivalent_evidence_tie_is_stable(self):
        duplicate = copy.deepcopy(self.data["request"]["knowledge_base"][0])
        duplicate["id"] = "aaa-equivalent"
        self.data["request"]["knowledge_base"].append(duplicate)
        self.assertEqual(self.result()["faq"]["sources"], ["aaa-equivalent"])

    def test_final_comparison_tampering_is_rejected(self):
        output = app.run_pipeline(self.data)
        output["stages"]["compare"]["ranking"][0]["score"] = 99
        with self.assertRaises(app.ValidationError):
            app.validate(output)

    def test_pipeline_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        first, second = app.run_pipeline(self.data), app.run_pipeline(self.data)
        self.assertEqual(self.data, before)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(app.validate(first), first)

    def test_handoffs_reject_tampered_grounding_and_preferences(self):
        faq = app.answer_faq(self.data)
        faq["stages"]["faq"]["answer"] = "Invented claim."
        with self.assertRaises(app.ValidationError):
            app.adapt_onboarding(faq)
        adaptive = app.adapt_onboarding(app.answer_faq(self.data))
        adaptive["stages"]["adaptive"]["weights"]["price"] = 1000
        with self.assertRaises(app.ValidationError):
            app.compare_products(adaptive)

    def test_stage_order_is_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.compare_products(self.data)
        with self.assertRaises(app.ValidationError):
            app.adapt_onboarding(self.data)

    def test_invalid_schema_and_profile(self):
        cases = [
            ("schema_version", "2.0"),
            ("fixture_label", "Real product data"),
            ("status", "ok"),
            ("request", None),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                data = fixture()
                data[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        self.data["request"]["profile"]["experience"] = "wizard"
        with self.assertRaises(app.ValidationError):
            self.result()

    def test_invalid_numbers_and_units(self):
        for value in (True, -1, float("nan"), float("inf"), "100"):
            with self.subTest(value=value):
                data = fixture()
                data["request"]["products"][0]["attributes"]["cost"]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        self.data["request"]["products"][0]["attributes"]["cost"]["unit"] = "EUR"
        with self.assertRaises(app.ValidationError):
            self.result()

    def test_invalid_rating_and_priority(self):
        self.data["request"]["products"][1]["attributes"]["rating"]["value"] = 101
        with self.assertRaises(app.ValidationError):
            self.result()
        self.data = fixture()
        self.data["request"]["profile"]["priorities"]["price"] = 0
        with self.assertRaises(app.ValidationError):
            self.result()

    def test_alias_collision_and_duplicate_ids_rejected(self):
        raw = {"price": {"value": 1, "unit": "USD"}, "cost": {"value": 2, "unit": "USD"}}
        with self.assertRaises(app.ValidationError):
            app.normalize_attributes(raw)
        self.data["request"]["products"][1]["id"] = "light-a"
        with self.assertRaises(app.ValidationError):
            self.result()

    def test_cycles_unknown_prerequisites_and_completed_steps_rejected(self):
        self.data["request"]["onboarding_steps"][0]["prerequisites"] = ["brightness"]
        with self.assertRaises(app.ValidationError):
            self.result()
        self.data = fixture()
        self.data["request"]["onboarding_steps"][0]["prerequisites"] = ["absent"]
        with self.assertRaises(app.ValidationError):
            self.result()
        self.data = fixture()
        self.data["request"]["profile"]["completed_steps"] = ["absent"]
        with self.assertRaises(app.ValidationError):
            self.result()

    def test_unknown_fields_and_whitespace_question_rejected(self):
        self.data["request"]["profile"]["unexpected"] = 1
        with self.assertRaises(app.ValidationError):
            self.result()
        self.data = fixture()
        self.data["request"]["question"] = "   "
        with self.assertRaises(app.ValidationError):
            self.result()


class CliTests(unittest.TestCase):
    def cli(self, *args):
        process = subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"), *args],
            cwd=HERE, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        return process.returncode, json.loads(process.stdout)

    def test_cli_success(self):
        code, output = self.cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(set(output["stages"]), {"faq", "adaptive", "compare"})

    def test_cli_missing_file(self):
        code, output = self.cli("synthetic-file-that-does-not-exist.json")
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_wrong_argument_count(self):
        for args in ((), ("example_input.json", "example_input.json")):
            code, output = self.cli(*args)
            self.assertEqual(code, 2)
            self.assertEqual(output["status"], "error")

    def test_cli_malformed_json_existing_python_file(self):
        code, output = self.cli("implementation.py")
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_valid_json_invalid_schema(self):
        code, output = self.cli("build_manifest.json")
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        for payload in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(payload=payload):
                with self.assertRaises(app.ValidationError):
                    json.loads(payload, object_pairs_hook=app.unique_object,
                               parse_constant=app.reject_constant)

    def test_cli_read_error_emits_json(self):
        import contextlib
        import io
        stream = io.StringIO()
        with patch.object(Path, "open", side_effect=PermissionError("synthetic denied")), \
                contextlib.redirect_stdout(stream):
            code = app.main([str(HERE / "example_input.json")])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
