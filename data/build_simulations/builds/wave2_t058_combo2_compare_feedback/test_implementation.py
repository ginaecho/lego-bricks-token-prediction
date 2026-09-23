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

    def test_unit_normalization(self):
        attrs = app.compare(self.data)["products"][0]["attributes"]
        self.assertEqual(attrs, {"price": 80, "weight": 0.5, "battery_hours": 10})

    def test_preference_ranking(self):
        rows = app.compare(self.data)["comparison"]
        self.assertEqual([r["product_id"] for r in rows], ["speaker-a", "speaker-c", "speaker-b"])
        self.assertAlmostEqual(rows[0]["score"], (1.5 + 2 + 2 / 7) / 6)
        self.data["preferences"] = {"battery_hours": {"weight": 1, "direction": "max"}}
        self.assertEqual(app.run(self.data)["insights"][0]["product_id"], "speaker-b")

    def test_deduplication_and_traceable_themes(self):
        insight = app.run(self.data)["insights"][0]
        self.assertEqual((insight["raw_count"], insight["unique_count"]), (3, 2))
        themes = {t["theme"]: t for t in insight["themes"]}
        self.assertEqual(themes["value"]["count"], 1)
        support = themes["value"]["support"][0]
        self.assertEqual(support["feedback_ids"], ["f1", "f2"])
        self.assertEqual(support["excerpts"][1]["text"], "LIGHT and good value.")

    def test_cross_stage_metadata_and_validation(self):
        comparison = app.compare(self.data)
        output = app.feedback_analysis(comparison)
        for row, insight in zip(comparison["comparison"], output["insights"]):
            for key in ("product_id", "rank", "score"):
                self.assertEqual(row[key], insight[key])
        comparison["comparison"][0]["attributes"]["price"] = 0
        with self.assertRaises(app.ValidationError):
            app.feedback_analysis(comparison)

    def test_output_trace_validation(self):
        output = app.run(self.data)
        output["insights"][0]["themes"][0]["support"][0]["excerpts"][0]["text"] = "Invented"
        with self.assertRaises(app.ValidationError):
            app.validate(output, "final")

    def test_empty_feedback(self):
        self.data["feedback"] = []
        for insight in app.run(self.data)["insights"]:
            self.assertEqual(insight["themes"], [])
            self.assertEqual(insight["unique_count"], 0)

    def test_single_product_equal_bounds(self):
        self.data["products"] = self.data["products"][:1]
        self.data["feedback"] = []
        self.assertEqual(app.run(self.data)["comparison"][0]["score"], 1)

    def test_ties_use_product_id(self):
        for product in self.data["products"]:
            product["attributes"] = {"price": 10}
        self.data["preferences"] = {"price": {"weight": 1, "direction": "min"}}
        self.data["products"].reverse()
        self.assertEqual([r["product_id"] for r in app.run(self.data)["comparison"]],
                         ["speaker-a", "speaker-b", "speaker-c"])

    def test_duplicates_do_not_cross_products(self):
        self.data["feedback"] = [
            {"id": "a", "product_id": "speaker-a", "text": "Nice sound"},
            {"id": "b", "product_id": "speaker-b", "text": "Nice sound"},
        ]
        result = {x["product_id"]: x for x in app.run(self.data)["insights"]}
        self.assertEqual(result["speaker-a"]["unique_count"], 1)
        self.assertEqual(result["speaker-b"]["themes"][0]["theme"], "general")

    def test_invalid_inputs(self):
        cases = []
        for key, value in [("products", []), ("preferences", {}), ("schema_version", True), ("feedback", {})]:
            data = copy.deepcopy(self.data)
            data[key] = value
            cases.append(data)
        for value in [True, -1, float("nan"), float("inf"), "10 EUR", "heavy", {}]:
            data = copy.deepcopy(self.data)
            data["products"][0]["attributes"]["cost"] = value
            cases.append(data)
        data = copy.deepcopy(self.data)
        data["products"][0]["attributes"]["price"] = 50
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["feedback"][0]["product_id"] = "missing"
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["feedback"][1]["id"] = "f1"
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["products"][1]["id"] = "speaker-a"
        cases.append(data)
        data = copy.deepcopy(self.data)
        del data["products"][0]["attributes"]["cost"]
        cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_invalid_preferences(self):
        for weight, direction in [(0, "min"), (-1, "min"), (True, "min"), (1, "ascending")]:
            self.data["preferences"] = {"price": {"weight": weight, "direction": direction}}
            with self.subTest(weight=weight, direction=direction), self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_large_finite_weights(self):
        for preference in self.data["preferences"].values():
            preference["weight"] = 1e308
        self.assertEqual(app.run(self.data)["status"], "ok")

    def test_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, before)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_and_argument_errors(self):
        for args in [[], ["not-present.json"], ["one", "two"]]:
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_bad_schema(self):
        for raw in ["{", "null", '{"schema_version": 1, "schema_version": 1}', '{"x": NaN}']:
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
