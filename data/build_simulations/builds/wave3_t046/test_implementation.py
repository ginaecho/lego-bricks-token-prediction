import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_integrated_example(self):
        result = app.run_pipeline(self.request)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["search"]["matches"][0]["product"]["id"], "SYN-AUDIO-01")
        self.assertEqual(result["onboarding"]["selected_product_id"], "SYN-AUDIO-01")
        self.assertEqual(result["onboarding"]["search_result_ids"],
                         [m["product"]["id"] for m in result["search"]["matches"]])
        self.assertEqual(result["fixture_label"], self.request["fixture_label"])

    def test_synonym_and_typo(self):
        result = app.smart_search(self.request)
        self.assertEqual(result["normalized_terms"], ["cordless", "headphnes", "run"])
        self.assertEqual(result["matches"][0]["matched_terms"], result["normalized_terms"])
        self.assertGreater(result["matches"][0]["score"], result["matches"][1]["score"])

    def test_filters_are_hard_constraints(self):
        self.request["filters"]["max_price_usd"] = 50
        result = app.run_pipeline(self.request)
        self.assertEqual([m["product"]["id"] for m in result["search"]["matches"]], ["SYN-AUDIO-02"])
        self.assertEqual(result["onboarding"]["selected_product_id"], "SYN-AUDIO-02")
        self.assertEqual([s["id"] for s in result["onboarding"]["steps"]], ["basics", "compare", "ready"])

    def test_general_catalog_category_and_alias(self):
        self.request["query"] = "TRAINERS"
        self.request["filters"] = {"category": None, "max_price_usd": None}
        result = app.run_pipeline(self.request)
        self.assertEqual(result["onboarding"]["selected_product_id"], "SYN-SHOE-01")

    def test_beginner_prerequisite_order_and_preference(self):
        result = app.run_pipeline(self.request)["onboarding"]
        seen = set()
        for step in result["steps"]:
            self.assertTrue(set(step["prerequisites"]) <= seen)
            self.assertEqual(step["format"], "interactive")
            self.assertEqual(step["product_id"], result["selected_product_id"])
            self.assertIn("Detailed guidance", step["detail"])
            self.assertTrue(step["explanation"])
            seen.add(step["id"])
        self.assertEqual([s["id"] for s in result["steps"]],
                         ["basics", "compare", "safety", "configure", "ready"])
        self.assertIn("safety", result["steps"][3]["prerequisites"])

    def test_advanced_waives_only_intro_not_safety(self):
        self.request["profile"] = {"experience": "advanced", "preferred_format": "video"}
        result = app.run_pipeline(self.request)["onboarding"]
        self.assertEqual([w["id"] for w in result["waived_steps"]], ["basics"])
        self.assertEqual([s["id"] for s in result["steps"]], ["compare", "safety", "configure", "ready"])
        self.assertTrue(all(s["format"] == "video" and "text only" in s["detail"] for s in result["steps"]))

    def test_intermediate_is_concise(self):
        self.request["profile"] = {"experience": "intermediate", "preferred_format": "text"}
        result = app.run_pipeline(self.request)["onboarding"]
        self.assertEqual(result["waived_steps"], [])
        self.assertIn("Concise guidance", result["steps"][0]["detail"])

    def test_no_matches_and_empty_catalog(self):
        for change in ({"query": "zzzzzzzz"}, {"catalog": []}):
            request = dict(self.request, **change)
            result = app.run_pipeline(request)["onboarding"]
            self.assertEqual(result["status"], "no_matches")
            self.assertIsNone(result["selected_product_id"])
            self.assertEqual(result["steps"], [])
            self.assertTrue(result["explanation"])

    def test_ties_and_limit_are_deterministic(self):
        product = copy.deepcopy(self.request["catalog"][0])
        product["id"] = "AAA"
        self.request["catalog"].append(product)
        self.request["limit"] = 1
        first = app.run_pipeline(self.request)
        self.request["catalog"].reverse()
        self.assertEqual(first, app.run_pipeline(self.request))
        self.assertEqual(first["onboarding"]["selected_product_id"], "AAA")

    def test_unicode_normalization(self):
        self.request["query"] = "CÓRDLESS"
        self.assertEqual(app.smart_search(self.request)["normalized_terms"], ["cordless"])

    def test_extreme_number_rejected_without_overflow(self):
        self.request["filters"]["max_price_usd"] = 10 ** 1000
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.request)

    def test_invalid_input_shapes(self):
        mutations = [
            {"query": ""}, {"query": "!!!"}, {"query": "the and for"},
            {"query": 42}, {"limit": True}, {"limit": 0}, {"limit": 21},
            {"schema_version": True}, {"catalog": None}, {"extra": 1},
            {"profile": {"experience": "expert", "preferred_format": "text"}},
            {"profile": {"experience": "beginner", "preferred_format": []}},
            {"filters": {"category": None, "max_price_usd": float("nan")}},
            {"filters": {"category": None, "max_price_usd": -1}},
            {"filters": {"category": None, "max_price_usd": True}},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(app.ValidationError):
                app.run_pipeline(dict(self.request, **mutation))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline([])

    def test_invalid_products(self):
        for field, value in (("price_usd", float("inf")), ("price_usd", True),
                             ("requires_setup", 1), ("tags", ["same", "same"]), ("id", "")):
            request = copy.deepcopy(self.request)
            request["catalog"][0][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)
        self.request["catalog"].append(copy.deepcopy(self.request["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.request)

    def test_handoff_rejects_forged_product_or_ranking(self):
        for field, value in (("id", "forged"), ("requires_setup", False), ("price_usd", 0)):
            search = app.smart_search(self.request)
            search["matches"][0]["product"][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.adaptive_onboarding(self.request, search)
        search = app.smart_search(self.request)
        search["matches"].reverse()
        with self.assertRaises(app.ValidationError):
            app.adaptive_onboarding(self.request, search)

    def test_handoff_rejects_forged_evidence(self):
        search = app.smart_search(self.request)
        search["matches"][0]["score"] += 1
        with self.assertRaises(app.ValidationError):
            app.adaptive_onboarding(self.request, search)

    def test_onboarding_validation_rejects_broken_prerequisites(self):
        search = app.smart_search(self.request)
        result = app.adaptive_onboarding(self.request, search)
        result["steps"][0], result["steps"][1] = result["steps"][1], result["steps"][0]
        with self.assertRaises(app.ValidationError):
            app.validate("onboarding", result, self.request, search)
        result = app.adaptive_onboarding(self.request, search)
        result["steps"][0]["product_id"] = "other"
        with self.assertRaises(app.ValidationError):
            app.validate("onboarding", result, self.request, search)

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(app.run_pipeline(self.request), app.run_pipeline(self.request))
        self.assertEqual(original, self.request)

    def test_cli_success_single_json(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file_arguments_and_invalid_schema(self):
        for args in ((), ("missing-input.json",), ("example_input.json", "extra"),
                     ("build_manifest.json",)):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stderr, "")
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_json_without_extra_files(self):
        for content in ("{", '{"x": NaN}', '{"x": 1, "x": 2}', "[]"):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)), \
                    patch("builtins.print") as output:
                self.assertEqual(app.main(["invalid.json"]), 2)
                self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")


if __name__ == "__main__":
    unittest.main()
