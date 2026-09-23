"""All test data is synthetic; no providers or filesystem scratch files."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import contextlib
import io

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    return json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))


class PersonalizationTests(unittest.TestCase):
    def test_normal_personalization(self):
        result = app.personalize(fixture())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["strategy"], "behavioral")
        self.assertEqual(result["recommendations"][0]["item_id"], "book-b")
        self.assertEqual(result["excluded_purchase_count"], 1)
        self.assertNotIn("book-a", [r["item_id"] for r in result["recommendations"]])

    def test_cold_start_popularity(self):
        payload = fixture()
        payload["user_id"] = "synthetic-new-user"
        result = app.personalize(payload)
        self.assertEqual(result["strategy"], "cold_start")
        self.assertEqual(result["matched_event_count"], 0)
        self.assertEqual(result["excluded_purchase_count"], 0)
        self.assertEqual(result["recommendations"][0]["item_id"], "game-a")
        self.assertEqual(result["recommendations"][0]["score"], 1)

    def test_exact_recency_and_purchase_weight(self):
        payload = fixture()
        payload["exclude_purchased"] = False
        payload["limit"] = 4
        payload["events"] = [
            {"user_id": payload["user_id"], "item_id": "book-a", "type": "purchase",
             "timestamp": "2026-08-24T12:00:00Z"},
            {"user_id": payload["user_id"], "item_id": "tool-a", "type": "browse",
             "timestamp": payload["now"]},
        ]
        rows = {row["item_id"]: row for row in app.personalize(payload)["recommendations"]}
        self.assertEqual(rows["book-a"]["signals"]["item_affinity"], 0.6)
        self.assertEqual(rows["tool-a"]["signals"]["item_affinity"], 0.4)
        self.assertAlmostEqual(rows["book-a"]["score"], 0.58)

    def test_order_independence_and_no_mutation(self):
        payload = fixture()
        saved = copy.deepcopy(payload)
        expected = app.personalize(payload)
        self.assertEqual(payload, saved)
        payload["items"].reverse()
        payload["events"].reverse()
        self.assertEqual(app.personalize(payload), expected)

    def test_ties_use_id(self):
        payload = fixture()
        payload["events"] = []
        payload["limit"] = 4
        for item in payload["items"]:
            item["popularity"] = 0.5
        ids = [row["item_id"] for row in app.personalize(payload)["recommendations"]]
        self.assertEqual(ids, sorted(ids))

    def test_empty_catalog_zero_limit_and_all_purchased(self):
        payload = fixture()
        payload["items"], payload["events"] = [], []
        self.assertEqual(app.personalize(payload)["recommendations"], [])
        payload = fixture()
        payload["limit"] = 0
        self.assertEqual(app.personalize(payload)["recommendations"], [])
        payload = fixture()
        payload["events"] = [
            {"user_id": payload["user_id"], "item_id": item["id"], "type": "purchase",
             "timestamp": payload["now"]} for item in payload["items"]
        ]
        self.assertEqual(app.personalize(payload)["recommendations"], [])

    def test_expired_weights_fall_back(self):
        payload = fixture()
        payload["half_life_days"] = 0.000001
        payload["events"] = [payload["events"][-1]]
        self.assertEqual(app.personalize(payload)["strategy"], "cold_start")

    def test_timezone_equivalence(self):
        payload = fixture()
        expected = app.personalize(payload)
        payload["now"] = "2026-09-23T14:00:00+02:00"
        self.assertEqual(app.personalize(payload), expected)

    def test_invalid_top_level_and_configuration(self):
        cases = [None, [], {}, {**fixture(), "unknown": 1}]
        for field, values in {
            "user_id": ["", None], "now": ["bad", "2026-09-23"],
            "limit": [True, -1, 1001, 1.5], "half_life_days": [0, -1, True, float("nan"), float("inf")],
            "exclude_purchased": ["true", 1], "items": [{}], "events": [{}],
        }.items():
            cases.extend({**fixture(), field: value} for value in values)
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(app.ValidationError):
                    app.personalize(payload)

    def test_invalid_items_and_events(self):
        cases = []
        for field, value in [("popularity", 2), ("popularity", True), ("category", ""), ("id", [])]:
            payload = fixture()
            payload["items"][0][field] = value
            cases.append(payload)
        payload = fixture()
        payload["items"].append(copy.deepcopy(payload["items"][0]))
        cases.append(payload)
        for field, value in [
            ("type", "click"), ("type", {}), ("item_id", "missing"),
            ("timestamp", "2026-09-24T00:00:00Z"), ("timestamp", "2026-09-22T12:00:00"),
        ]:
            payload = fixture()
            payload["events"][0][field] = value
            cases.append(payload)
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(app.ValidationError):
                    app.personalize(payload)

    def test_cli_success_and_file_errors(self):
        for arguments, code, status in [
            (["example_input.json"], 0, "ok"),
            (["nonexistent-synthetic-input.json"], 2, "error"),
            ([], 2, "error"),
            (["example_input.json", "extra"], 2, "error"),
            (["implementation.py"], 2, "error"),
            (["."], 2, "error"),
        ]:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-B", "implementation.py", *arguments],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads(result.stdout)["status"], status)
                self.assertEqual(result.stderr, "")

    def test_cli_shared_validation_and_strict_json(self):
        for raw in ['{"user_id": "a", "user_id": "b"}', '{"x": NaN}', "null",
                    json.dumps({**fixture(), "limit": True}), "{"]:
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=raw):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
