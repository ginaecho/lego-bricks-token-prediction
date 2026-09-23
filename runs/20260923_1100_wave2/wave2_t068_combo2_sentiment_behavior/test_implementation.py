"""Tests use synthetic fixtures only; no network, providers, or scratch files."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def alex(self, result):
        return next(r for r in result["personalization"] if r["id"] == "synthetic-alex")

    def test_sentiment_negation_and_evidence(self):
        result = self.run_data()
        row = result["sentiment"][0]
        self.assertEqual(row["score"], -1)
        self.assertEqual(row["priority"], 100)
        self.assertEqual(row["matches"][-1],
                         {"token": "good", "index": 4, "negated": True, "contribution": -1})
        self.assertEqual(result["sentiment"][1]["label"], "positive")
        self.assertEqual(result["sentiment"][1]["priority"], 0)

    def test_severity_can_override_neutral_sentiment(self):
        self.data["feedback"][0].update(text="Package missing", severity="critical")
        row = self.run_data()["sentiment"][0]
        self.assertEqual(row["score"], 0)
        self.assertTrue(row["is_issue"])
        self.assertEqual(row["priority"], 80)

    def test_neutral_empty_and_negated_negative(self):
        for value, expected in [("", 0), ("unrecognized vocabulary", 0),
                                ("not bad", 1), ("good bad", 0), ("GREAT!", 1)]:
            with self.subTest(value=value):
                self.data["feedback"][0].update(text=value, severity="low")
                row = next(r for r in self.run_data()["sentiment"]
                           if r["id"] == "synthetic-f1")
                self.assertEqual(row["score"], expected)

    def test_purchase_and_recency_weighting(self):
        rows = {r["id"]: r for r in self.alex(self.run_data())["recommendations"]}
        # A 30-day-old purchase weighs 3 * 0.5 = 1.5, versus today's view = 1.
        self.assertAlmostEqual(rows["synthetic-camera"]["item_affinity"], 2 / 3, places=6)
        self.assertEqual(rows["synthetic-book"]["item_affinity"], 1)
        self.assertEqual(rows["synthetic-headphones"]["item_affinity"], 0)
        self.assertAlmostEqual(rows["synthetic-headphones"]["category_affinity"],
                               2 / 3, places=6)

    def test_cross_stage_category_penalty_and_customer_isolation(self):
        result = self.run_data()
        alex = {r["id"]: r for r in self.alex(result)["recommendations"]}
        for product in ("synthetic-camera", "synthetic-headphones"):
            self.assertEqual(alex[product]["issue_penalty"], 0.5)
            self.assertEqual(alex[product]["issue_ids"], ["synthetic-f1"])
        self.assertEqual(alex["synthetic-book"]["issue_penalty"], 0)
        sam = next(r for r in result["personalization"] if r["id"] == "synthetic-sam")
        self.assertTrue(sam["cold_start"])
        self.assertTrue(all(r["issue_penalty"] == 0 for r in sam["recommendations"]))

    def test_handoff_changes_ranking(self):
        self.data["events"] = []
        with_issue = self.alex(self.run_data())["recommendations"]
        self.data["feedback"] = []
        without_issue = self.alex(self.run_data())["recommendations"]
        self.assertEqual(with_issue[0]["id"], "synthetic-book")
        self.assertEqual(without_issue[0]["id"], "synthetic-camera")

    def test_cold_start_and_stable_ties(self):
        self.data["events"] = []
        self.data["feedback"] = []
        for product in self.data["products"]:
            product["popularity"] = 0.5
        result = self.alex(self.run_data())
        self.assertTrue(result["cold_start"])
        self.assertEqual([r["id"] for r in result["recommendations"]],
                         sorted(p["id"] for p in self.data["products"]))

    def test_empty_collections(self):
        self.data.update(products=[], events=[], feedback=[])
        result = self.run_data()
        self.assertEqual(result["sentiment"], [])
        self.assertEqual(self.alex(result)["recommendations"], [])
        self.data["customers"] = []
        self.assertEqual(self.run_data()["personalization"], [])

    def test_invalid_inputs(self):
        mutations = [
            lambda d: d.update(top_k=True),
            lambda d: d.update(top_k=0),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(schema_version="2"),
            lambda d: d.update(extra=1),
            lambda d: d["products"][0].update(popularity=float("nan")),
            lambda d: d["products"][0].update(popularity=2),
            lambda d: d["feedback"][0].update(customer_id="unknown"),
            lambda d: d["events"][0].update(product_id="unknown"),
            lambda d: d["feedback"][0].update(severity=[]),
            lambda d: d["events"][0].update(kind="click"),
            lambda d: d["events"][0].update(timestamp="2027-01-01T00:00:00Z"),
            lambda d: d.update(as_of="2026-09-23"),
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_handoff_is_validated(self):
        initial = {"schema_version": "1.0", "status": "ok", "input": self.data,
                   "sentiment": None, "personalization": None}
        handoff = app.sentiment_stage(initial)
        for field, value in [("priority", 999), ("customer_id", "synthetic-sam"),
                             ("score", 0.25), ("category", "wrong")]:
            with self.subTest(field=field):
                invalid = copy.deepcopy(handoff)
                invalid["sentiment"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.behavior_stage(invalid)

    def test_repeated_issues_use_max_not_sum(self):
        duplicate = copy.deepcopy(self.data["feedback"][0])
        duplicate.update(id="synthetic-f3", severity="low")
        self.data["feedback"].append(duplicate)
        recommendations = self.alex(self.run_data())["recommendations"]
        camera = next(r for r in recommendations if r["id"] == "synthetic-camera")
        self.assertEqual(camera["issue_penalty"], 0.5)
        self.assertEqual(camera["issue_ids"], ["synthetic-f1", "synthetic-f3"])

    def test_deterministic_nonmutating_top_k(self):
        self.data["top_k"] = 1
        original = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(first, self.run_data())
        self.assertEqual(self.data, original)
        self.assertEqual(len(self.alex(first)["recommendations"]), 1)
        self.data["events"].reverse()
        self.data["products"].reverse()
        self.assertEqual(first["personalization"], self.run_data()["personalization"])

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout), self.run_data())

    def test_cli_missing_file_and_usage(self):
        for arguments in [[], [str(ROOT / "nonexistent.json")],
                          ["example_input.json", "extra"]]:
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, "-B",
                                         str(ROOT / "implementation.py"), *arguments],
                                        cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stderr, "")
                self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for contents in ["{", "[]", '{"x": NaN}', '{"x":1,"x":2}',
                         json.dumps({**self.data, "top_k": False})]:
            with self.subTest(contents=contents):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=contents)):
                    with redirect_stdout(output):
                        code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
