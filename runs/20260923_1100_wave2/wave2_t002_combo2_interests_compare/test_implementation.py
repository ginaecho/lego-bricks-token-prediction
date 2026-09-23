import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_interest_ranking_and_grounding(self):
        result = app.pipeline(self.data)
        items = result["interests"]["recommendations"]
        self.assertEqual([p["product"]["id"] for p in items], ["audio-a", "audio-b", "audio-c"])
        self.assertEqual([p["interest_score"] for p in items], [6, 4, 1])
        self.assertEqual(items[0]["matched_interests"], ["audio", "outdoors", "travel"])
        self.assertIn("matches category", items[0]["explanation"][0])
        self.assertIn("matches tag", items[0]["explanation"][1])

    def test_exclusions_reach_comparison(self):
        self.data["preferences"]["exclude_ids"] = ["audio-b"]
        result = app.pipeline(self.data)
        self.assertEqual(result["comparison"]["candidate_ids"], ["audio-a", "audio-c"])
        self.assertEqual({p["product_id"] for p in result["interests"]["excluded"]},
                         {"audio-b", "audio-d", "audio-e"})
        self.assertEqual({p["product_id"] for p in result["comparison"]["ranking"]},
                         {"audio-a", "audio-c"})

    def test_attribute_normalization(self):
        context = app.Validation.input(self.data)
        product = context["products"][0]
        self.assertEqual(product["weight_g"], 250)
        self.assertEqual(product["battery_hours"], 10)
        rows = {r["attribute"]: r for r in app.pipeline(self.data)["comparison"]["side_by_side"]}
        self.assertEqual(rows["weight_g"]["values"]["audio-b"], 400)
        self.assertEqual(rows["battery_hours"]["unit"], "h")
        self.assertIsNone(rows["weight_g"]["values"]["audio-c"])

    def test_comparison_weighted_ranking(self):
        result = app.pipeline(self.data)["comparison"]["ranking"]
        self.assertEqual([p["product_id"] for p in result], ["audio-a", "audio-b", "audio-c"])
        self.assertAlmostEqual(result[0]["score"], 0.78)
        self.assertAlmostEqual(result[1]["score"], 0.34)
        self.assertAlmostEqual(result[2]["score"], 0.30)
        self.assertEqual([p["rank"] for p in result], [1, 2, 3])

    def test_preferences_can_change_comparison_winner(self):
        self.data["preferences"]["comparison_weights"] = {"price_usd": 1}
        ranking = app.pipeline(self.data)["comparison"]["ranking"]
        self.assertEqual(ranking[0]["product_id"], "audio-c")

    def test_limit_is_validated_cross_stage_boundary(self):
        self.data["limit"] = 2
        result = app.pipeline(self.data)
        self.assertEqual(result["interests"]["eligible_count"], 3)
        self.assertEqual(result["comparison"]["candidate_ids"], ["audio-a", "audio-b"])
        self.assertEqual(result["comparison"]["ranking"][0]["matched_interests"],
                         result["interests"]["recommendations"][0]["matched_interests"])
        for row in result["comparison"]["side_by_side"]:
            self.assertEqual(set(row["values"]), {"audio-a", "audio-b"})

    def test_mutated_handoff_rejected(self):
        context = app.Validation.input(self.data)
        for mutation in ("score", "product", "explanation", "exclusion"):
            with self.subTest(mutation=mutation):
                handoff = app.recommend(context)
                first = handoff["recommendations"][0]
                if mutation == "score":
                    first["interest_score"] += 1
                elif mutation == "product":
                    first["product"]["price_usd"] = 0
                elif mutation == "explanation":
                    first["explanation"] = ["Invented claim"]
                else:
                    first["product"] = context["products"][3]
                with self.assertRaises(app.ValidationError):
                    app.compare(context, handoff)

    def test_missing_attributes_receive_zero_utility(self):
        last = app.pipeline(self.data)["comparison"]["ranking"][-1]
        self.assertEqual(last["utilities"]["weight_g"], 0)
        self.assertEqual(last["utilities"]["battery_hours"], 0)
        self.assertTrue(any("missing" in text for text in last["explanation"]))

    def test_empty_and_all_excluded_catalog(self):
        for empty_catalog in (True, False):
            with self.subTest(empty_catalog=empty_catalog):
                data = copy.deepcopy(self.data)
                if empty_catalog:
                    data["products"] = []
                else:
                    data["preferences"]["exclude_ids"] = [p["id"] for p in data["products"]]
                result = app.pipeline(data)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["comparison"]["ranking"], [])
                self.assertEqual(result["comparison"]["candidate_ids"], [])

    def test_ties_and_single_candidate(self):
        self.data["preferences"]["interests"] = {}
        self.data["preferences"]["comparison_weights"] = {"interest": 1}
        ranking = app.pipeline(self.data)["comparison"]["ranking"]
        self.assertEqual([p["product_id"] for p in ranking], ["audio-a", "audio-b", "audio-c"])
        self.data["limit"] = 1
        self.assertEqual(app.pipeline(self.data)["comparison"]["ranking"][0]["score"], 1)

    def test_case_normalization_no_double_counting(self):
        self.data["preferences"]["interests"] = {" AUDIO ": 2}
        self.data["products"][0]["tags"].append("Audio")
        item = app.pipeline(self.data)["interests"]["recommendations"][0]
        self.assertEqual(item["interest_score"], 2)
        self.assertEqual(item["matched_interests"], ["audio"])

    def test_budget_inclusive(self):
        self.data["preferences"]["max_price_usd"] = 60
        ids = app.pipeline(self.data)["comparison"]["candidate_ids"]
        self.assertIn("audio-a", ids)
        self.assertNotIn("audio-b", ids)

    def test_invalid_inputs(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(limit=0),
            lambda d: d.update(limit=True),
            lambda d: d.update(extra="unknown"),
            lambda d: d.update(products={}),
            lambda d: d["fixture"].update(synthetic=False),
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["products"][0]["price"].update(currency="EUR"),
            lambda d: d["products"][0]["price"].update(amount=0.001),
            lambda d: d["products"][0]["price"].update(amount=-1),
            lambda d: d["products"][0]["price"].update(amount=float("nan")),
            lambda d: d["products"][0]["weight"].update(unit="lb"),
            lambda d: d["products"][0]["weight"].update(value=True),
            lambda d: d["products"][0].update(tags=["Travel", "travel"]),
            lambda d: d["preferences"].update(comparison_weights={}),
            lambda d: d["preferences"].update(comparison_weights={"unknown": 1}),
            lambda d: d["preferences"].update(interests={"travel": -1}),
            lambda d: d["preferences"].update(interests={"Travel": 1, "travel": 2}),
            lambda d: d["preferences"].update(exclude_ids="audio-a"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.pipeline(data)

    def test_deterministic_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        result = app.pipeline(self.data)
        self.assertEqual(before, self.data)
        self.data["products"].reverse()
        self.assertEqual(result, app.pipeline(self.data))

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["does-not-exist.json"], ["a", "b"]):
            with self.subTest(args=args):
                process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                         cwd=ROOT, text=True, capture_output=True, check=False)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stderr, "")
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_json_and_validation_errors(self):
        for text in ('{', '{"x": 1, "x": 2}', '{"x": NaN}', '[]', '{}',
                     json.dumps({**self.data, "limit": -1})):
            with self.subTest(text=text):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=text), redirect_stdout(output):
                    status = app.main(["fixture.json"])
                self.assertEqual(status, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
