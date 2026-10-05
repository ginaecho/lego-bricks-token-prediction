"""Synthetic fixtures only. Tests do not create files or call networks."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_personalized_and_nonmutating(self):
        original = copy.deepcopy(self.data)
        output = app.recommend(self.data)
        self.assertEqual([p["product_id"] for p in output["recommendations"]],
                         ["synthetic-p2", "synthetic-p3"])
        self.assertEqual(output["summary"]["excluded"]["purchased"], 1)
        self.assertEqual(output["summary"]["excluded"]["out_of_stock"], 1)
        self.assertEqual(output["recommendations"][0]["score"], 7.7)
        self.assertEqual(self.data, original)

    def test_empty_catalog(self):
        result = app.recommend({"schema_version": 1, "catalog": [], "customer": {"id": "synthetic"}})
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["summary"]["mode"], "cold_start")

    def test_cold_start_and_deterministic_ties(self):
        self.data["customer"] = {"id": "synthetic"}
        for product in self.data["catalog"]:
            product["popularity"] = 0.5
        first = app.recommend(self.data)
        self.data["catalog"].reverse()
        self.assertEqual(first, app.recommend(self.data))
        self.assertEqual(first["recommendations"][0]["product_id"], "synthetic-p1")

    def test_budget_blocking_and_dislike(self):
        self.data["customer"]["budget"] = {"min": 18, "max": 24}
        self.data["customer"]["interactions"] = [{"product_id": "synthetic-p1", "type": "dislike"}]
        result = app.recommend(self.data)
        self.assertEqual([p["product_id"] for p in result["recommendations"]], ["synthetic-p3"])
        self.data["customer"]["blocked_categories"] = ["home"]
        self.assertEqual(app.recommend(self.data)["recommendations"], [])

    def test_allow_purchased(self):
        self.data["options"]["exclude_purchased"] = False
        self.assertEqual(app.recommend(self.data)["recommendations"][0]["product_id"], "synthetic-p1")

    def test_duplicate_events_are_ignored(self):
        expected = app.recommend(self.data)
        self.data["customer"]["interactions"] *= 5
        self.assertEqual(app.recommend(self.data), expected)

    def test_diversity_changes_order(self):
        self.data["customer"] = {"id": "synthetic"}
        self.data["catalog"][2]["popularity"] = 0.6
        self.data["options"]["diversity"] = 0
        self.assertEqual(app.recommend(self.data)["recommendations"][1]["product_id"], "synthetic-p2")
        self.data["options"]["diversity"] = 1
        self.assertEqual(app.recommend(self.data)["recommendations"][1]["product_id"], "synthetic-p3")

    def test_invalid_root_and_options(self):
        for value in (None, [], {}, {"schema_version": True, "catalog": [], "customer": {"id": "x"}}):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.recommend(value)
        for options in ({"limit": True}, {"limit": 0}, {"diversity": float("nan")},
                        {"diversity": 2}, {"exclude_purchased": 1}, {"unknown": 0}):
            with self.subTest(options=options), self.assertRaises(app.ValidationError):
                app.recommend(dict(self.data, options=options))

    def test_invalid_catalog(self):
        for key, value in (("price", -1), ("price", True), ("price", float("inf")),
                           ("tags", ["same", "same"]), ("in_stock", 1), ("id", " ")):
            bad = copy.deepcopy(self.data)
            bad["catalog"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.recommend(bad)
        self.data["catalog"].append(self.data["catalog"][0])
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_invalid_customer(self):
        for changes in ({"budget": {"min": 3, "max": 1}},
                        {"interactions": [{"product_id": "missing", "type": "view"}]},
                        {"interactions": [{"product_id": "synthetic-p1", "type": "bad"}]},
                        {"preferred_tags": "lightweight"}):
            bad = copy.deepcopy(self.data)
            bad["customer"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(app.ValidationError):
                app.recommend(bad)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), app.recommend(self.data))
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_bad_json_and_validation(self):
        for content in ("{", '{"schema_version":1,"schema_version":1}',
                        '{"value":NaN}', "[]", '{"schema_version":2}'):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                code = app.main(["synthetic-fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
