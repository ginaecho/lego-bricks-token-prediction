"""Synthetic labeled fixtures; callbacks below are local stubs, not providers."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, rank_products, unique_object


HERE = Path(__file__).resolve().parent


def labeled_fixture():
    return {
        "as_of": "2026-09-21T12:00:00Z",
        "products": [
            {"id": "a", "category": "books", "tags": ["python"]},
            {"id": "b", "category": "books", "tags": ["python"]},
            {"id": "c", "category": "home", "tags": ["ceramic"]},
        ],
        "events": [
            {"id": "e1", "product_id": "a", "type": "purchase", "count": 1, "timestamp": "2026-09-21T12:00:00Z"}
        ],
    }


class BehavioralTests(unittest.TestCase):
    def test_labeled_purchase_and_related_item_ranking(self):
        result = rank_products(labeled_fixture())
        self.assertEqual(result["ranked_product_ids"], ["a", "b", "c"])
        self.assertEqual([row["score"] for row in result["ranking"]], [6, 2, 0])
        self.assertFalse(result["cold_start"])
        self.assertEqual(result["history_events_used"], 1)

    def test_half_life_exact_decay(self):
        data = labeled_fixture()
        data["events"][0]["timestamp"] = "2026-08-22T12:00:00Z"
        self.assertEqual(rank_products(data)["ranking"][0]["score"], 3)

    def test_recent_browse_outranks_old_purchase(self):
        data = labeled_fixture()
        data["events"][0]["timestamp"] = "2026-05-24T12:00:00Z"
        data["events"].append({"id": "e2", "product_id": "c", "type": "browse", "count": 1, "timestamp": data["as_of"]})
        self.assertEqual(rank_products(data)["ranked_product_ids"], ["c", "a", "b"])

    def test_cold_start_explicit_preferences(self):
        data = labeled_fixture()
        data["events"] = []
        data["preferences"] = {"categories": ["home"], "tags": ["ceramic"]}
        result = rank_products(data)
        self.assertTrue(result["cold_start"])
        self.assertEqual(result["ranked_product_ids"], ["c", "a", "b"])
        self.assertEqual(result["ranking"][0]["score"], 3)
        self.assertIn("cold_start", result["ranking"][0]["claims"])

    def test_cold_start_stable_ties(self):
        data = labeled_fixture()
        data["events"] = []
        data["products"].reverse()
        self.assertEqual(rank_products(data)["ranked_product_ids"], ["a", "b", "c"])

    def test_zero_counts_are_cold_start(self):
        data = labeled_fixture()
        data["events"][0]["count"] = 0
        self.assertTrue(rank_products(data)["cold_start"])

    def test_exclusions_remove_history_and_override_preferences(self):
        for exclusions in ({"product_ids": ["a"]}, {"categories": ["books"]}, {"tags": ["python"]}):
            with self.subTest(exclusions=exclusions):
                data = labeled_fixture()
                data["exclusions"] = exclusions
                data["preferences"] = {"categories": ["books"]}
                result = rank_products(data)
                self.assertTrue(result["cold_start"])
                self.assertNotIn("a", result["ranked_product_ids"])
                self.assertTrue(all(row["components"]["category_affinity"] == 0 for row in result["ranking"]))

    def test_empty_catalog(self):
        self.assertEqual(rank_products({"as_of": "2026-09-21T00:00:00Z", "products": []})["ranking"], [])

    def test_all_products_excluded(self):
        data = labeled_fixture()
        data["exclusions"] = {"product_ids": ["a", "b", "c"]}
        result = rank_products(data)
        self.assertEqual(result["ranking"], [])
        self.assertTrue(result["cold_start"])

    def test_underflow_does_not_claim_cold_start(self):
        data = labeled_fixture()
        data["half_life_days"] = 0.000001
        data["events"][0]["timestamp"] = "2026-08-22T12:00:00Z"
        result = rank_products(data)
        self.assertFalse(result["cold_start"])
        self.assertEqual([row["score"] for row in result["ranking"]], [0, 0, 0])

    def test_overflow_rejected(self):
        data = labeled_fixture()
        data["events"][0]["count"] = 10 ** 308
        with self.assertRaises(ValidationError):
            rank_products(data)

    def test_unknown_ids(self):
        for change in ("event", "exclusion"):
            data = labeled_fixture()
            if change == "event":
                data["events"][0]["product_id"] = "missing"
            else:
                data["exclusions"] = {"product_ids": ["missing"]}
            with self.subTest(change=change), self.assertRaises(ValidationError):
                rank_products(data)

    def test_bad_timestamps(self):
        for value in ("yesterday", "2026-02-30T12:00:00Z", "2026-09-21", "2026-09-21T12:00:00", "2027-01-01T00:00:00Z", None):
            data = labeled_fixture()
            data["events"][0]["timestamp"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                rank_products(data)

    def test_timezone_normalization(self):
        data = labeled_fixture()
        data["events"][0]["timestamp"] = "2026-09-21T14:00:00+02:00"
        self.assertEqual(rank_products(data)["ranking"][0]["score"], 6)

    def test_bad_counts(self):
        for value in (-1, 1.2, True, "2", float("inf"), 10 ** 1000):
            data = labeled_fixture()
            data["events"][0]["count"] = value
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValidationError):
                rank_products(data)

    def test_bad_half_lives(self):
        for value in (0, -1, True, float("nan"), float("inf"), "30"):
            data = labeled_fixture()
            data["half_life_days"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                rank_products(data)

    def test_duplicate_products_events_and_values(self):
        for kind in ("product", "event_id", "event_content", "tags", "preferences", "exclusions"):
            data = labeled_fixture()
            if kind == "product":
                data["products"].append(copy.deepcopy(data["products"][0]))
            elif kind.startswith("event"):
                duplicate = copy.deepcopy(data["events"][0])
                if kind == "event_content":
                    duplicate["id"] = "another-id"
                data["events"].append(duplicate)
            elif kind == "tags":
                data["products"][0]["tags"] = ["python", "python"]
            elif kind == "preferences":
                data["preferences"] = {"tags": ["python", "python"]}
            else:
                data["exclusions"] = {"product_ids": ["a", "a"]}
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                rank_products(data)

    def test_duplicate_json_keys(self):
        with self.assertRaises(ValidationError):
            json.loads('{"products": [], "products": []}', object_pairs_hook=unique_object)

    def test_malformed_schema(self):
        for value in ([], {}, {"as_of": "2026-09-21T00:00:00Z", "products": "wrong"}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                rank_products(value)
        data = labeled_fixture()
        data["events"][0]["type"] = "click"
        with self.assertRaises(ValidationError):
            rank_products(data)

    def test_input_order_independence_and_no_mutation(self):
        data = labeled_fixture()
        data["events"].append({"id": "e2", "product_id": "b", "type": "browse", "count": 2, "timestamp": data["as_of"]})
        original = copy.deepcopy(data)
        expected = rank_products(data)
        self.assertEqual(data, original)
        data["events"].reverse()
        data["products"].reverse()
        self.assertEqual(rank_products(data), expected)

    def test_grounded_callback_fixture_not_provider(self):
        seen = []

        def local_fixture_callback(payload):
            seen.append(payload["product_id"])
            return {"product_id": payload["product_id"], "claim_ids": [next(iter(payload["claims"]))]}

        result = rank_products(labeled_fixture(), local_fixture_callback)
        self.assertEqual(seen, ["a", "b", "c"])
        for row in result["ranking"]:
            self.assertEqual(row["explanation_source"], "injected_callback")
            self.assertTrue(set(row["explanations"]) <= set(row["claims"].values()))

    def test_callback_rejects_ungrounded_outputs(self):
        for output in (
            {"product_id": "a", "claim_ids": ["invented"]},
            {"product_id": "wrong", "claim_ids": ["direct_behavior"]},
            {"product_id": "a", "claim_ids": []},
            {"product_id": "a", "claim_ids": ["direct_behavior", "direct_behavior"]},
            {"product_id": "a", "claim_ids": ["direct_behavior"], "text": "Bestseller"},
            "An unsupported claim",
        ):
            with self.subTest(output=output), self.assertRaises(ValidationError):
                rank_products(labeled_fixture(), lambda payload: output)

    def test_callback_cannot_mutate_grounding(self):
        def callback(payload):
            payload["claims"]["invented"] = "Unsupported."
            return {"product_id": payload["product_id"], "claim_ids": ["invented"]}

        with self.assertRaises(ValidationError):
            rank_products(labeled_fixture(), callback)

    def test_callback_exceptions_propagate(self):
        def callback(payload):
            raise RuntimeError("Local fixture failure")

        with self.assertRaisesRegex(RuntimeError, "Local fixture failure"):
            rank_products(labeled_fixture(), callback)

    def test_cli_example_labeled_result(self):
        process = subprocess.run(
            [sys.executable, str(HERE / "implementation.py"), str(HERE / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["ranked_product_ids"], ["book-a", "book-b"])
        self.assertEqual([row["score"] for row in result["ranking"]], [8.375, 6.5])

    def test_cli_missing_file_error(self):
        process = subprocess.run(
            [sys.executable, str(HERE / "implementation.py"), str(HERE / "nonexistent-input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("error:", process.stderr)


if __name__ == "__main__":
    unittest.main()
