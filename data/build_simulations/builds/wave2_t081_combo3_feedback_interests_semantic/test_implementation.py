"""Standard-library fixture tests; no network, providers or external dependencies."""

import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def test_example_integrated(self):
        out = self.run_data()
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["interests"]["recommendations"][0]["product_id"], "bottle")
        self.assertEqual(out["search"]["results"][0]["product_id"], "bottle")

    def test_feedback_dedup_and_exact_evidence(self):
        feedback = self.run_data()["feedback"]
        self.assertEqual(feedback["duplicates_removed"], 1)
        evidence = feedback["themes"]["sustainability"]["evidence"][0]
        self.assertEqual(evidence["source_ids"], ["f1", "f2"])
        self.assertEqual(evidence["excerpt"], self.data["feedback"][0]["text"])
        self.assertEqual(feedback["themes"]["travel"]["count"], 1)

    def test_distinct_customers_not_deduplicated(self):
        self.data["feedback"][1]["customer_id"] = "synthetic-c3"
        self.assertEqual(self.run_data()["feedback"]["duplicates_removed"], 0)

    def test_conflicting_duplicate_rating_rejected(self):
        self.data["feedback"][1]["rating"] = 1
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_feedback_flows_into_profile_and_grounding(self):
        out = self.run_data()
        profile = out["interests"]["profile"]
        self.assertEqual(profile["travel"]["weight"], 4)
        self.assertEqual(profile["travel"]["source_ids"], ["f1", "f2"])
        self.assertEqual(profile["comfort"]["feedback_weight"], -0.5)
        bottle = out["interests"]["recommendations"][0]
        self.assertEqual(bottle["score"], sum(e["weight"] for e in bottle["evidence"]))
        self.assertIn("feedback=1", bottle["explanation"])

    def test_rating_change_propagates(self):
        before = self.run_data()
        self.data["feedback"][0]["rating"] = 1
        self.data["feedback"][1]["rating"] = 1
        after = self.run_data()
        self.assertLess(after["interests"]["profile"]["travel"]["weight"],
                        before["interests"]["profile"]["travel"]["weight"])
        self.assertNotEqual(after["search"]["results"][0]["interest_score"],
                            before["search"]["results"][0]["interest_score"])

    def test_all_exclusion_types_enforced(self):
        self.data["preferences"]["exclude"] = {
            "product_ids": ["bottle"], "categories": ["HOME"], "tags": ["Leather"]}
        self.data["search"]["query"] = "bottle cushion leather"
        out = self.run_data()
        self.assertEqual(out["interests"]["eligible_product_ids"], ["pack"])
        self.assertEqual(list(out["search"]["index"]), ["pack"])

    def test_recommendation_limit_is_search_boundary(self):
        self.data["preferences"]["limit"] = 1
        self.data["search"]["query"] = "soft cushion"
        out = self.run_data()
        self.assertEqual(list(out["search"]["index"]), ["bottle"])
        self.assertEqual(len(out["search"]["results"]), 1)

    def test_lexical_query_reranks_shortlist(self):
        self.data["search"]["query"] = "soft cushion"
        out = self.run_data()
        first = out["search"]["results"][0]
        self.assertEqual(first["product_id"], "cushion")
        self.assertEqual(first["matched_terms"], ["cushion", "soft"])
        self.assertGreater(out["search"]["index"]["cushion"]["soft"], 0)

    def test_empty_query_falls_back_to_interests(self):
        self.data["search"]["query"] = ""
        out = self.run_data()
        expected = [r["product_id"] for r in out["interests"]["recommendations"]][:2]
        self.assertEqual([r["product_id"] for r in out["search"]["results"]], expected)

    def test_empty_catalog_and_feedback(self):
        self.data["products"] = []
        self.data["feedback"] = []
        out = self.run_data()
        self.assertEqual(out["search"]["results"], [])
        self.assertEqual(out["search"]["index"], {})

    def test_all_products_excluded(self):
        self.data["preferences"]["exclude"]["product_ids"] = [
            p["id"] for p in self.data["products"]]
        self.assertEqual(self.run_data()["search"]["results"], [])

    def test_no_feedback_no_preferences_deterministic_fallback(self):
        self.data["feedback"] = []
        self.data["preferences"]["interests"] = {}
        self.data["search"]["query"] = "zzzz"
        out = self.run_data()
        self.assertEqual(out["interests"]["eligible_product_ids"],
                         ["bottle", "cushion", "pack"])
        self.assertTrue(all(r["score"] == 0 for r in out["search"]["results"]))

    def test_injected_embedding_ranks_synonym(self):
        self.data["search"]["query"] = "ergonomic"
        calls = []

        def fixture_embedder(texts):
            calls.append(texts)
            return [[1, 0] if i == 0 or "cushion" in t.lower() else [0, 1]
                    for i, t in enumerate(texts)]

        out = app.run_pipeline(self.data, fixture_embedder)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)
        self.assertEqual(out["search"]["mode"], "injected_embedding")
        self.assertEqual(out["search"]["results"][0]["product_id"], "cushion")
        self.assertEqual(out["search"]["results"][0]["embedding_score"], 1)

    def test_invalid_embeddings(self):
        invalid = [
            None, [], [[1]] * 3, [[1], [1, 2], [1], [1]],
            [[math.nan]] * 4, [[math.inf]] * 4, [[True]] * 4,
            [["1"]] * 4, [[]] * 4,
        ]
        for vectors in invalid:
            with self.subTest(vectors=vectors), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, lambda texts: vectors)

    def test_provider_exception_is_validation_error(self):
        def broken(texts):
            raise RuntimeError("fixture error")
        with self.assertRaisesRegex(app.ValidationError, "embedder failed"):
            app.run_pipeline(self.data, broken)

    def test_zero_and_huge_vectors(self):
        self.assertEqual(app.cosine([0, 0], [1, 2]), 0)
        self.assertAlmostEqual(app.cosine([1e308, 1e308], [1e308, 1e308]), 1)

    def test_invalid_input_variants(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(products={}),
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["feedback"][0].update(product_id="unknown"),
            lambda d: d["feedback"][0].update(rating=True),
            lambda d: d["feedback"][0].update(rating=6),
            lambda d: d["feedback"][0].update(rating=math.nan),
            lambda d: d["preferences"].update(limit=0),
            lambda d: d["preferences"].update(limit=True),
            lambda d: d["preferences"]["interests"].update(travel=math.inf),
            lambda d: d["preferences"]["interests"].update(TRAVEL=2),
            lambda d: d["search"].update(query=2),
            lambda d: d.update(unexpected=True),
            lambda d: d["products"][0].update(tags="travel"),
            lambda d: d["products"][0].update(category="!!!"),
        ]
        original = copy.deepcopy(self.data)
        for mutate in mutations:
            self.data = copy.deepcopy(original)
            mutate(self.data)
            with self.subTest(data=self.data), self.assertRaises(app.ValidationError):
                self.run_data()

    def test_feedback_handoff_rejects_tampered_excerpt(self):
        feedback = app.analyze_feedback(self.data)
        feedback["records"][0]["excerpt"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.validate("feedback", feedback, self.data)

    def test_interest_handoff_rejects_excluded_product(self):
        feedback = app.analyze_feedback(self.data)
        interests = app.rank_interests(self.data, feedback)
        interests["recommendations"][0]["product_id"] = "case"
        with self.assertRaises(app.ValidationError):
            app.validate("interests", interests, (self.data, feedback))

    def test_search_handoff_rejects_ineligible_product(self):
        out = self.run_data()
        out["search"]["results"][0]["product_id"] = "case"
        with self.assertRaises(app.ValidationError):
            app.validate("search", out["search"], (self.data, out["interests"]))

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(first, self.run_data())
        self.assertEqual(original, self.data)
        self.data["products"].reverse()
        self.data["feedback"].reverse()
        self.assertEqual(first, self.run_data())

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_single_json(self):
        result = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_missing_file(self):
        result = self.cli(str(ROOT / "does-not-exist.json"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage_errors(self):
        for args in ((), ("one", "two")):
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_schema_actual_process(self):
        result = self.cli(str(ROOT / "build_manifest.json"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_malformed_duplicate_keys_and_bad_encoding(self):
        for payload in ("{", '{"schema_version":1,"schema_version":1}', "[]"):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=payload)), redirect_stdout(output):
                self.assertEqual(app.main(["fixture.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("invalid encoding")), \
                redirect_stdout(output):
            self.assertEqual(app.main(["fixture.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
