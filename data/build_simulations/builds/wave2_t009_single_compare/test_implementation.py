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


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_normalization_and_ranking(self):
        result = app.compare(self.data)
        self.assertEqual(result["status"], "ok")
        alpha = result["normalized_products"][0]["attributes"]
        self.assertEqual(alpha["mass"], 1.5)
        self.assertEqual(alpha["storage"], 500)
        self.assertEqual(alpha["color"], "silver")
        self.assertIs(alpha["touch"], True)
        self.assertEqual([p["product_id"] for p in result["ranking"]], ["alpha", "gamma", "beta"])
        self.assertAlmostEqual(result["ranking"][0]["score"], 0.566667)
        self.assertEqual(result["comparison"]["rows"][0]["values"]["gamma"], 600)

    def test_no_input_mutation_and_determinism(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.compare(self.data), app.compare(self.data))
        self.assertEqual(self.data, before)

    def test_missing_all_values_scores_zero(self):
        for p in self.data["products"]:
            p["attributes"] = {}
        result = app.compare(self.data)
        self.assertTrue(all(p["score"] == 0 for p in result["ranking"]))
        self.assertEqual([p["product_id"] for p in result["ranking"]], ["alpha", "beta", "gamma"])

    def test_equal_numeric_values_and_single_preference(self):
        self.data["preferences"] = [{"attribute": "storage", "mode": "higher", "weight": 1}]
        for p in self.data["products"]:
            p["attributes"]["storage"] = 500
        self.assertTrue(all(p["score"] == 1 for p in app.compare(self.data)["ranking"]))

    def test_target_units(self):
        self.data["preferences"] = [{"attribute": "mass", "mode": "target", "value": "1500 g", "weight": 1}]
        result = app.compare(self.data)["ranking"]
        self.assertEqual(result[0]["product_id"], "alpha")
        self.assertEqual(result[0]["score"], 1)
        self.assertAlmostEqual(result[1]["score"], 0.666667)

    def test_invalid_values(self):
        for invalid in [True, "NaN", float("inf"), "2 cm", "2 EUR", [], ""]:
            with self.subTest(invalid=invalid):
                data = copy.deepcopy(self.data)
                data["products"][0]["attributes"]["cost"] = invalid
                with self.assertRaises(app.ValidationError):
                    app.compare(data)

    def test_invalid_schema(self):
        cases = []
        for field, value in [("schema_version", True), ("products", []), ("preferences", []), ("extra", 1)]:
            data = copy.deepcopy(self.data)
            data[field] = value
            cases.append(data)
        data = copy.deepcopy(self.data)
        data["products"][1]["id"] = "alpha"
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["attributes"][1]["aliases"] = ["cost"]
        cases.append(data)
        data = copy.deepcopy(self.data)
        data["products"][0]["attributes"]["price"] = 123
        cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.compare(data)

    def test_invalid_preferences(self):
        for preference in [
            {"attribute": "price", "mode": "lower", "weight": 0},
            {"attribute": "color", "mode": "higher", "weight": 1},
            {"attribute": "mass", "mode": "target", "weight": 1},
            {"attribute": "unknown", "mode": "lower", "weight": 1},
            {"attribute": "price", "mode": "lower", "weight": 1, "value": 2},
        ]:
            with self.subTest(preference=preference):
                self.data["preferences"] = [preference]
                with self.assertRaises(app.ValidationError):
                    app.compare(self.data)

    def test_large_finite_numbers(self):
        self.data["preferences"] = [
            {"attribute": "price", "mode": "higher", "weight": 1e308},
            {"attribute": "mass", "mode": "lower", "weight": 1e308},
        ]
        self.data["products"][0]["attributes"]["cost"] = -1e308
        self.data["products"][1]["attributes"]["price"] = 1e308
        result = app.compare(self.data)
        json.dumps(result, allow_nan=False)
        self.assertTrue(all(0 <= p["score"] <= 1 for p in result["ranking"]))

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(json.loads(proc.stdout), app.compare(self.data))

    def test_cli_file_and_usage_errors(self):
        for args in [[], ["does-not-exist.json"], ["a", "b"], ["."]]:
            with self.subTest(args=args):
                proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_cli_malformed_and_duplicate_json(self):
        for raw in ['{', '{"a":1,"a":2}', '{"a":NaN}', '[]', '{"schema_version":2}']:
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
