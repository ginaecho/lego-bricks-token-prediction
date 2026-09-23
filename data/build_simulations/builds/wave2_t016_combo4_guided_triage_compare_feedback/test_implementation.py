"""Tests use synthetic fixtures only and never write files or call networks."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def result(self):
        return app.run_pipeline(self.data)["results"]

    def guided(self):
        return app.guided_setup(app.initialize(self.data))

    def test_integrated_result_and_determinism(self):
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(first["stage"], "feedback")
        self.assertEqual(list(first["results"]), ["guided", "triage", "compare", "feedback"])
        self.assertEqual(first["status"], "ok")

    def test_inputs_and_stage_inputs_not_mutated(self):
        original = copy.deepcopy(self.data)
        guided = self.guided()
        before = copy.deepcopy(guided)
        app.triage_tickets(guided)
        app.run_pipeline(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(guided, before)

    def test_guided_progress_and_optional_step(self):
        guided = self.guided()["results"]["guided"]
        self.assertEqual(guided["completed_count"], 2)
        self.assertEqual(guided["progress"], 0.666667)
        self.assertEqual(guided["steps"][2]["status"], "ready")
        self.assertTrue(guided["ready_for_triage"])

    def test_incomplete_required_step_blocks_triage(self):
        self.data["guided"]["completed_steps"] = ["profile"]
        guided = self.guided()
        self.assertEqual(guided["results"]["guided"]["required_remaining"], ["preferences"])
        self.assertEqual(guided["results"]["guided"]["steps"][2]["status"], "blocked")
        with self.assertRaisesRegex(app.ValidationError, "incomplete"):
            app.triage_tickets(guided)

    def test_prerequisite_completion_order(self):
        self.data["guided"]["completed_steps"] = ["preferences", "profile"]
        with self.assertRaisesRegex(app.ValidationError, "completed first"):
            app.initialize(self.data)

    def test_prerequisite_cycle(self):
        self.data["guided"]["steps"][0]["requires"] = ["newsletter"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            app.initialize(self.data)

    def test_unknown_prerequisite(self):
        self.data["guided"]["steps"][1]["requires"] = ["missing"]
        with self.assertRaisesRegex(app.ValidationError, "unknown prerequisite"):
            app.initialize(self.data)

    def test_guided_preferences_propagate_and_route_overrides(self):
        result = self.result()
        routed = result["triage"]["tickets"][0]
        compared = result["compare"]["comparisons"][0]
        self.assertEqual(routed["preferences"]["min_memory_gb"], 16)
        self.assertEqual(routed["preferences"]["max_price_usd"], 1200)
        self.assertEqual(compared["preferences"], routed["preferences"])
        self.assertEqual(compared["owner"], routed["owner"])
        self.assertEqual(compared["priority"], "high")

    def test_categorization_and_accountable_owner(self):
        rows = self.result()["triage"]["tickets"]
        self.assertEqual(rows[0]["category"], "performance")
        self.assertEqual(rows[0]["matched_keywords"], ["slow", "memory"])
        self.assertEqual(rows[1]["owner"], "synthetic-mobile-team")

    def test_rule_tie_uses_priority_then_configuration_order(self):
        self.data["triage"]["tickets"][0]["text"] = "slow travel"
        self.assertEqual(self.result()["triage"]["tickets"][0]["category"], "performance")
        self.data["triage"]["rules"][1]["priority"] = "high"
        self.assertEqual(self.result()["triage"]["tickets"][0]["category"], "performance")
        self.data["triage"]["rules"].reverse()
        self.assertEqual(self.result()["triage"]["tickets"][0]["category"], "portability")

    def test_default_route_and_word_boundaries(self):
        self.data["triage"]["tickets"][0]["text"] = "slower traveling"
        row = self.result()["triage"]["tickets"][0]
        self.assertEqual(row["category"], "general")
        self.assertEqual(row["owner"], "synthetic-help-team")
        self.assertEqual(row["matched_keywords"], [])

    def test_normalization_units_and_brand(self):
        products = self.result()["compare"]["comparisons"][0]["products"]
        self.assertEqual(products[0]["attributes"], {
            "price_usd": 900, "memory_gb": 16, "weight_kg": 1.4, "brand": "exampleco"
        })
        self.data["products"][0]["attributes"]["ram"] = {"value": 0.015625, "unit": "TB"}
        self.assertEqual(self.result()["compare"]["comparisons"][0]["products"][0]
                         ["attributes"]["memory_gb"], 16)

    def test_side_by_side_and_ranking(self):
        result = self.result()["compare"]["comparisons"][0]
        self.assertEqual(result["columns"], list(app.ATTRIBUTES))
        self.assertEqual(len(result["products"]), 3)
        self.assertEqual(result["ranked_product_ids"], ["product-a", "product-b"])
        self.assertEqual(result["recommended_product_id"], "product-a")
        self.assertEqual(result["products"][2]["rejection_reasons"], ["min_memory_gb"])
        self.assertIsNone(result["products"][2]["rank"])

    def test_converted_units_at_exact_constraint_boundary(self):
        self.data["guided"]["preferences"]["max_weight_kg"] = 1.4
        product = self.result()["compare"]["comparisons"][0]["products"][0]
        self.assertTrue(product["eligible"])
        self.assertEqual(product["attributes"]["weight_kg"], 1.4)

    def test_weights_change_recommendation(self):
        self.data["guided"]["preferences"]["weights"] = {
            "price_usd": 0, "memory_gb": 1, "weight_kg": 0, "brand": 0
        }
        self.assertEqual(self.result()["compare"]["comparisons"][0]
                         ["recommended_product_id"], "product-b")

    def test_equal_scores_use_product_id_not_input_order(self):
        self.data["guided"]["preferences"]["weights"] = {
            "price_usd": 0, "memory_gb": 0, "weight_kg": 0, "brand": 1
        }
        del self.data["guided"]["preferences"]["preferred_brand"]
        self.data["products"].reverse()
        self.assertEqual(self.result()["compare"]["comparisons"][0]
                         ["ranked_product_ids"], ["product-a", "product-b"])

    def test_no_eligible_products_preserves_comparison_and_feedback(self):
        self.data["guided"]["preferences"]["max_price_usd"] = 1
        result = self.result()
        row = result["compare"]["comparisons"][0]
        self.assertEqual(row["outcome"], "no_eligible_products")
        self.assertEqual(row["ranked_product_ids"], [])
        self.assertIsNone(row["recommended_product_id"])
        self.assertIsNone(result["feedback"]["groups"][0]["comparison_rank"])
        self.assertFalse(result["feedback"]["groups"][0]["was_recommended"])

    def test_feedback_deduplication_and_traceable_excerpts(self):
        result = self.result()["feedback"]
        self.assertEqual(result["raw_count"], 4)
        self.assertEqual(result["unique_count"], 3)
        self.assertEqual(result["duplicates_removed"], 1)
        self.assertEqual(result["groups"][0]["source_ids"], ["feedback-1", "feedback-2"])
        originals = {row["id"]: row["text"] for row in self.data["feedback"]}
        for theme in result["themes"]:
            for evidence in theme["evidence"]:
                self.assertEqual(evidence["excerpt"], originals[evidence["source_id"]])
        value = next(row for row in result["themes"] if row["theme"] == "value")
        self.assertEqual(value["unique_feedback_count"], 2)
        self.assertEqual(len(value["evidence"]), 3)

    def test_compare_to_feedback_context_propagation(self):
        result = self.result()
        group = result["feedback"]["groups"][0]
        product = result["compare"]["comparisons"][0]["products"][0]
        self.assertEqual(group["comparison_rank"], product["rank"])
        self.assertEqual(group["comparison_score"], product["score"])
        self.assertTrue(group["was_recommended"])
        self.assertEqual(group["owner"], "synthetic-performance-team")

    def test_same_text_different_ticket_not_deduplicated(self):
        extra = copy.deepcopy(self.data["feedback"][0])
        extra.update({"id": "feedback-5", "ticket_id": "ticket-2"})
        self.data["feedback"].append(extra)
        self.assertEqual(self.result()["feedback"]["unique_count"], 4)

    def test_same_text_different_product_not_deduplicated(self):
        extra = copy.deepcopy(self.data["feedback"][0])
        extra.update({"id": "feedback-5", "product_id": "product-b"})
        self.data["feedback"].append(extra)
        self.assertEqual(self.result()["feedback"]["unique_count"], 4)

    def test_unthemed_and_empty_feedback(self):
        self.data["feedback_themes"] = {}
        self.assertEqual(self.result()["feedback"]["unthemed_count"], 3)
        self.data["feedback"] = []
        feedback = self.result()["feedback"]
        self.assertEqual(feedback["raw_count"], 0)
        self.assertEqual(feedback["unique_count"], 0)

    def test_empty_ticket_batch(self):
        self.data["triage"]["tickets"] = []
        self.data["feedback"] = []
        result = self.result()
        self.assertEqual(result["triage"]["tickets"], [])
        self.assertEqual(result["compare"]["comparisons"], [])

    def test_invalid_input_variants(self):
        variants = []
        for key, value in [("schema_version", True), ("schema_version", 2),
                           ("synthetic", False), ("products", []), ("feedback", {})]:
            bad = copy.deepcopy(self.data)
            bad[key] = value
            variants.append(bad)
        bad = copy.deepcopy(self.data)
        bad["unknown"] = 1
        variants.append(bad)
        variants.extend([None, [], "not an object"])
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(app.ValidationError):
                app.run_pipeline(bad)

    def test_duplicate_identifiers_and_completed_steps(self):
        for section in ("products", "feedback"):
            bad = copy.deepcopy(self.data)
            bad[section].append(copy.deepcopy(bad[section][0]))
            with self.subTest(section=section), self.assertRaisesRegex(app.ValidationError, "duplicate"):
                app.initialize(bad)
        self.data["guided"]["completed_steps"].append("profile")
        with self.assertRaisesRegex(app.ValidationError, "duplicates"):
            app.initialize(self.data)

    def test_invalid_numbers_units_and_aliases(self):
        for value in (True, -1, 0, float("nan"), float("inf"), "16"):
            bad = copy.deepcopy(self.data)
            bad["products"][0]["attributes"]["ram"]["value"] = value
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.initialize(bad)
        self.data["products"][0]["attributes"]["ram"]["unit"] = "bytes"
        with self.assertRaisesRegex(app.ValidationError, "unsupported unit"):
            app.initialize(self.data)
        self.data["products"][0]["attributes"]["ram"]["unit"] = "MB"
        self.data["products"][0]["attributes"]["memory"] = {"value": 16, "unit": "GB"}
        with self.assertRaisesRegex(app.ValidationError, "duplicate alias"):
            app.initialize(self.data)

    def test_bad_references_and_missing_owner(self):
        for key in ("ticket_id", "product_id"):
            bad = copy.deepcopy(self.data)
            bad["feedback"][0][key] = "missing"
            with self.subTest(key=key), self.assertRaisesRegex(app.ValidationError, "unknown"):
                app.initialize(bad)
        self.data["triage"]["default"]["owner"] = " "
        with self.assertRaises(app.ValidationError):
            app.initialize(self.data)

    def test_customer_mismatch(self):
        self.data["triage"]["tickets"][0]["customer_id"] = "someone-else"
        with self.assertRaisesRegex(app.ValidationError, "onboarded customer"):
            app.initialize(self.data)

    def test_weights_validation_and_large_finite_values(self):
        weights = self.data["guided"]["preferences"]["weights"]
        for key in weights:
            weights[key] = 0
        with self.assertRaisesRegex(app.ValidationError, "at least one"):
            app.initialize(self.data)
        for key in weights:
            weights[key] = 1e308
        score = self.result()["compare"]["comparisons"][0]["products"][0]["score"]
        self.assertTrue(0 <= score <= 1)

    def test_wrong_stage_and_tampered_handoffs_rejected(self):
        with self.assertRaisesRegex(app.ValidationError, "expected guided"):
            app.triage_tickets(app.initialize(self.data))
        guided = self.guided()
        guided["results"]["guided"]["preferences"]["max_price_usd"] = 1
        with self.assertRaisesRegex(app.ValidationError, "handoff differs"):
            app.triage_tickets(guided)
        routed = app.triage_tickets(self.guided())
        routed["results"]["triage"]["tickets"][0]["owner"] = "tampered"
        with self.assertRaisesRegex(app.ValidationError, "handoff differs"):
            app.compare_products(routed)
        compared = app.compare_products(app.triage_tickets(self.guided()))
        compared["results"]["compare"]["comparisons"][0]["products"][0]["rank"] = 999
        with self.assertRaisesRegex(app.ValidationError, "handoff differs"):
            app.analyze_feedback(compared)

    def test_cli_success_subprocess(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(json.loads(proc.stdout), app.run_pipeline(self.data))

    def test_cli_missing_file_and_usage_subprocess(self):
        for arguments in ([], ["nonexistent-synthetic-file.json"],
                          ["example_input.json", "extra"]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                                  cwd=ROOT, capture_output=True, text=True, check=False)
            with self.subTest(arguments=arguments):
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stderr, "")
                self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema_without_creating_files(self):
        for content in ('{', '{"schema_version": NaN}', '{"a":1,"a":2}', '{}',
                        json.dumps({**self.data, "synthetic": False})):
            with self.subTest(content=content), patch.object(
                Path, "open", return_value=io.StringIO(content)
            ), redirect_stdout(io.StringIO()) as stdout:
                code = app.main(["synthetic-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_cli_unicode_and_file_errors(self):
        for error in (PermissionError("synthetic denied"),
                      UnicodeDecodeError("utf-8", b"\xff", 0, 1, "synthetic invalid encoding")):
            with self.subTest(error=error), patch.object(Path, "open", side_effect=error), \
                    redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(app.main(["synthetic-input.json"]), 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
