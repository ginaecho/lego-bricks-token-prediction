import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from implementation import ValidationError, compare, main


ROOT = Path(__file__).resolve().parent


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_ranking_and_comparison(self):
        result = compare(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([p["id"] for p in result["ranking"]], ["a", "c", "b"])
        self.assertAlmostEqual(result["ranking"][0]["score"], 4.1666666667 / 7, places=6)
        self.assertEqual(result["comparison"][0]["values"], {"a": 80, "b": 100, "c": 60})

    def test_units_aliases_categories(self):
        result = compare(self.data)
        values = result["products"][0]["attributes"]
        self.assertEqual(values, {"price": 80, "weight": 120, "capacity": 1000,
                                  "finish": "matte"})

    def test_missing_is_penalized(self):
        result = compare(self.data)
        row = next(p for p in result["ranking"] if p["id"] == "c")
        self.assertEqual(row["contributions"][1]["utility"], 0)
        self.assertTrue(row["contributions"][1]["missing"])
        self.assertIsNone(result["comparison"][1]["values"]["c"])

    def test_single_product(self):
        self.data["products"] = self.data["products"][:1]
        self.assertEqual(compare(self.data)["ranking"][0]["score"], 1)

    def test_ties_use_identifier_not_input_order(self):
        first = self.data["products"][0]
        second = copy.deepcopy(first)
        second["id"] = "z"
        self.data["products"] = [second, first]
        self.assertEqual([p["id"] for p in compare(self.data)["ranking"]], ["a", "z"])

    def test_all_missing_attribute(self):
        for product in self.data["products"]:
            product["attributes"].pop("weight", None)
        self.data["preferences"] = [{"attribute": "weight", "weight": 1, "direction": "lower"}]
        self.assertEqual([p["score"] for p in compare(self.data)["ranking"]], [0, 0, 0])

    def test_unlisted_category(self):
        self.data["products"][0]["attributes"]["finish"] = "brushed"
        result = compare(self.data)
        row = next(p for p in result["ranking"] if p["id"] == "a")
        self.assertEqual(row["contributions"][-1]["utility"], 0)

    def test_incompatible_units(self):
        self.data["products"][0]["attributes"]["weight"]["unit"] = "TB"
        with self.assertRaises(ValidationError):
            compare(self.data)

    def test_invalid_numeric_and_weights(self):
        for value in [True, "10", float("nan"), float("inf"), 10**400]:
            with self.subTest(value=str(value)[:20]):
                data = copy.deepcopy(self.data)
                data["products"][0]["attributes"]["cost"] = value
                del data["products"][0]["attributes"][" COST "]
                with self.assertRaises(ValidationError):
                    compare(data)
        for value in [0, -1, True, float("nan")]:
            self.data["preferences"][0]["weight"] = value
            with self.assertRaises(ValidationError):
                compare(self.data)

    def test_invalid_schema(self):
        for field, value in [("products", []), ("attributes", []), ("preferences", []),
                             ("fixture_label", "real catalog")]:
            data = copy.deepcopy(self.data)
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                compare(data)
        self.data["unexpected"] = 1
        with self.assertRaises(ValidationError):
            compare(self.data)

    def test_duplicate_alias_id_preference(self):
        mutations = [
            lambda d: d["attributes"][1].update(aliases=["cost"]),
            lambda d: d["products"][1].update(id="a"),
            lambda d: d["products"][0]["attributes"].update(price=20),
            lambda d: d["preferences"].append(copy.deepcopy(d["preferences"][0])),
            lambda d: d["preferences"][-1].update(order=["matte", " MATTE "]),
        ]
        for mutate in mutations:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(ValidationError):
                compare(data)

    def test_deterministic_and_does_not_mutate(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(compare(self.data), compare(self.data))
        self.assertEqual(self.data, original)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in [[], [str(ROOT / "missing-input.json")],
                     ["example_input.json", "extra"]]:
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                      *args], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_malformed_json_and_encoding_without_extra_files(self):
        for payload in [b"{", b'{"x":1,"x":2}', b"NaN", b"\xff",
                        b"[]" , b"x" * 1_000_001]:
            with patch("implementation.Path.open", return_value=io.BytesIO(payload)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["synthetic-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
