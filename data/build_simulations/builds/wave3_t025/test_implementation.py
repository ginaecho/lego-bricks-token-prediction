"""All catalog and customer fixtures are synthetic; tests write no files."""

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def product(identifier, **changes):
    result = {"id": identifier, "title": "Synthetic " + identifier,
              "category": "Books", "tags": ["science"], "price": 10,
              "popularity": 0.5}
    result.update(changes)
    return result


class DiscoveryTests(unittest.TestCase):
    def test_example_personalization_and_filters(self):
        payload = json.loads((ROOT / "example_input.json").read_text())
        result = app.recommend(payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "personalized")
        self.assertEqual(result["recommendations"][0]["product_id"], "synthetic-trail-pack")
        self.assertEqual(result["recommendations"][0]["score"], 9.6)
        self.assertEqual(result["summary"]["filtered"]["over_budget"], 1)
        self.assertEqual(result["summary"]["filtered"]["out_of_stock"], 1)

    def test_cold_start_popularity_and_stable_ties(self):
        items = [product("z"), product("b", price=9), product("a", price=9),
                 product("popular", popularity=1)]
        result = app.recommend({"products": items})
        self.assertEqual(result["mode"], "cold_start")
        self.assertEqual([p["product_id"] for p in result["recommendations"]],
                         ["popular", "a", "b", "z"])
        self.assertEqual(result, app.recommend({"products": list(reversed(items))}))

    def test_history_affinity_and_purchased_exclusion(self):
        result = app.recommend({
            "products": [product("owned"), product("similar"), product("other", category="Toys", tags=[])],
            "customer": {"purchased_product_ids": ["owned"]},
        })
        self.assertEqual(result["recommendations"][0]["product_id"], "similar")
        self.assertEqual(result["recommendations"][0]["score"], 3.5)
        self.assertEqual(result["summary"]["filtered"]["already_purchased"], 1)

    def test_empty_and_all_filtered(self):
        self.assertEqual(app.recommend({"products": []})["recommendations"], [])
        result = app.recommend({"products": [product("a")],
                                "customer": {"excluded_categories": [" BOOKS "]}})
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["summary"]["filtered"]["excluded_category"], 1)

    def test_zero_budget_and_limit(self):
        result = app.recommend({"products": [product("free", price=0), product("paid")],
                                "customer": {"max_price": 0}, "limit": 1})
        self.assertEqual([p["product_id"] for p in result["recommendations"]], ["free"])

    def test_normalization_and_no_input_mutation(self):
        payload = {"products": [product("a")],
                   "customer": {"preferred_categories": [" BOOKS "],
                                "preferred_tags": ["SCIENCE", "science"]}}
        original = copy.deepcopy(payload)
        self.assertEqual(app.recommend(payload)["recommendations"][0]["score"], 7.5)
        self.assertEqual(payload, original)

    def test_invalid_top_level_and_customer(self):
        for payload in (None, [], {}, {"products": {}, "limit": 2},
                        {"products": [], "surprise": True},
                        {"products": [], "customer": None},
                        {"products": [], "customer": {"preferred_tags": "science"}},
                        {"products": [], "customer": {"viewed_product_ids": ["missing"]}}):
            with self.subTest(payload=payload), self.assertRaises(app.ValidationError):
                app.recommend(payload)

    def test_invalid_numbers_and_flags(self):
        for value in (True, -1, float("nan"), float("inf"), "10", 10**400):
            with self.subTest(value=str(value)), self.assertRaises(app.ValidationError):
                app.recommend({"products": [product("a", price=value)]})
        for limit in (0, 51, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(app.ValidationError):
                app.recommend({"products": [], "limit": limit})
        for changes in ({"popularity": 1.1}, {"in_stock": 1}, {"title": " "},
                        {"tags": [1]}, {"category": None}, {"unknown": 1}):
            with self.subTest(changes=changes), self.assertRaises(app.ValidationError):
                app.recommend({"products": [product("a", **changes)]})

    def test_duplicate_ids(self):
        with self.assertRaises(app.ValidationError):
            app.recommend({"products": [product("a"), product(" a ")]})

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_usage_and_json_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")],
                     [str(ROOT / "implementation.py")], ["a", "b"]):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    capture_output=True, text=True, cwd=ROOT, check=False,
                )
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_validation_duplicate_keys_and_nonfinite_json(self):
        for raw in ('{"products":[],"limit":false}', '{"products":[],"products":[]}',
                    '{"products":[],"customer":{"max_price":NaN}}',
                    '{"products":[],"customer":{"max_price":1e999}}'):
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["synthetic-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
