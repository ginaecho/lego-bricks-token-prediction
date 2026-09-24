"""Synthetic fixtures; tests never write files or use network services."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

from implementation import ValidationError, discover, main


ROOT = Path(__file__).resolve().parent


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_purchase_and_recency(self):
        output = discover(self.data)
        self.assertFalse(output["cold_start"])
        self.assertEqual(output["history_events_used"], 2)
        self.assertEqual([r["item_id"] for r in output["recommendations"]],
                         ["synthetic-book", "synthetic-lamp", "synthetic-ball"])
        self.assertAlmostEqual(output["recommendations"][0]["score"], 3 / 3.5 + 0.03)

    def test_recency_changes_ranking(self):
        self.data["popularity_weight"] = 0
        self.data["events"][0]["kind"] = "browse"
        self.data["events"][0]["timestamp"] = "2025-09-24T12:00:00Z"
        self.data["events"][1]["timestamp"] = self.data["as_of"]
        self.assertEqual(discover(self.data)["recommendations"][0]["item_id"], "synthetic-ball")

    def test_cold_start_and_availability(self):
        self.data["events"] = []
        output = discover(self.data)
        self.assertTrue(output["cold_start"])
        self.assertEqual(output["recommendations"][0]["item_id"], "synthetic-ball")
        self.assertNotIn("synthetic-sold-out", [r["item_id"] for r in output["recommendations"]])

    def test_other_users_do_not_personalize(self):
        self.data["user_id"] = "synthetic-new-user"
        output = discover(self.data)
        self.assertTrue(output["cold_start"])
        self.assertEqual(output["history_events_used"], 0)

    def test_empty_catalog_and_zero_limit(self):
        self.data["limit"] = 0
        self.assertEqual(discover(self.data)["recommendations"], [])
        self.data["catalog"] = []
        self.data["events"] = []
        self.assertEqual(discover(self.data)["recommendations"], [])

    def test_ties_and_input_order_are_deterministic(self):
        self.data["events"] = []
        for item in self.data["catalog"]:
            item["popularity"] = 0.5
        original = copy.deepcopy(self.data)
        output = discover(self.data)
        self.assertEqual(self.data, original)
        self.data["catalog"].reverse()
        self.assertEqual(output, discover(self.data))
        ids = [r["item_id"] for r in output["recommendations"]]
        self.assertEqual(ids, sorted(ids))

    def test_event_order_invariance(self):
        output = discover(self.data)
        self.data["events"].reverse()
        self.assertEqual(output, discover(self.data))

    def test_invalid_scalar_values(self):
        for key, value in [("limit", True), ("limit", -1), ("half_life_days", 0),
                           ("popularity_weight", float("nan")), ("synthetic", "yes"),
                           ("schema_version", True), ("as_of", "2026-09-24"),
                           ("user_id", ""), ("unknown", 1)]:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(ValidationError):
                    discover(data)

    def test_invalid_events(self):
        for key, value in [("timestamp", "2027-01-01T00:00:00Z"),
                           ("item_id", "missing"), ("kind", "click"), ("kind", [])]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["events"][0][key] = value
                with self.assertRaises(ValidationError):
                    discover(data)

    def test_duplicate_ids_and_wrong_shapes(self):
        for collection in ["catalog", "events"]:
            data = copy.deepcopy(self.data)
            data[collection].append(copy.deepcopy(data[collection][0]))
            with self.assertRaises(ValidationError):
                discover(data)
            data[collection] = {}
            with self.assertRaises(ValidationError):
                discover(data)
        with self.assertRaises(ValidationError):
            discover([])

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), discover(self.data))
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in [[], [str(ROOT / "nonexistent-input.json")]]:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        import io
        for content in ['{', '{"a":1,"a":2}', '[]', '{"schema_version":1}']:
            with self.subTest(content=content):
                out = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), patch("sys.stdout", out):
                    self.assertEqual(main(["synthetic-invalid.json"]), 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")

    def test_timezone_equivalence_and_underflow_fallback(self):
        original = discover(self.data)
        self.data["events"][0]["timestamp"] = "2026-09-24T14:00:00+02:00"
        self.assertEqual(discover(self.data), original)
        self.data["half_life_days"] = 0.001
        self.data["events"][0]["timestamp"] = "2020-01-01T00:00:00Z"
        self.assertTrue(discover(self.data)["cold_start"])


if __name__ == "__main__":
    unittest.main()
