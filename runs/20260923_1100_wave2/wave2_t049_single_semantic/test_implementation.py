"""Synthetic fixtures only; no network or live embedding provider."""

import copy
import json
import pathlib
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

from implementation import ValidationError, main, search

ROOT = pathlib.Path(__file__).resolve().parent


def fixture():
    return {"schema_version": 1, "dataset_label": "SYNTHETIC test catalog",
            "query": "couch", "products": [
                {"id": "b", "title": "Sofa"},
                {"id": "a", "title": "Couch"},
                {"id": "c", "title": "Garden shovel"}]}


class SearchTests(unittest.TestCase):
    def test_exact_before_synonym(self):
        result = search(fixture())
        self.assertEqual([r["id"] for r in result["results"]], ["a", "b"])
        self.assertEqual(result["results"][1]["matched_terms"], ["sofa"])
        self.assertFalse(result["embedding_used"])

    def test_deterministic_ties(self):
        data = fixture()
        data["products"][0]["title"] = "Couch"
        self.assertEqual([r["id"] for r in search(data)["results"]], ["a", "b"])
        self.assertEqual(search(data), search(copy.deepcopy(data)))

    def test_empty_catalog(self):
        data = fixture()
        data["products"] = []
        self.assertEqual(search(data)["results"], [])
        self.assertEqual(search(data)["indexed_count"], 0)

    def test_no_match_and_punctuation(self):
        for query in ["xylophone", "!!!"]:
            data = fixture()
            data["query"] = query
            self.assertEqual(search(data)["matched_count"], 0)

    def test_case_unicode_and_tags(self):
        data = fixture()
        data["query"] = "CAFÉ"
        data["products"][2]["tags"] = ["café"]
        self.assertEqual(search(data)["results"][0]["id"], "c")

    def test_limit_and_threshold(self):
        data = fixture()
        data["options"] = {"limit": 1}
        result = search(data)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["matched_count"], 2)
        data["options"]["min_score"] = 1
        self.assertEqual(search(data)["results"], [])

    def test_invalid_schema(self):
        variants = [None, [], {}, {"schema_version": True}]
        for key, value in [("query", ""), ("query", 2), ("products", {}),
                           ("options", {"limit": True}), ("options", {"limit": 0}),
                           ("options", {"min_score": float("nan")}),
                           ("options", {"embedding_weight": float("inf")}),
                           ("options", {"unknown": 2}), ("extra", 1)]:
            data = fixture()
            data[key] = value
            variants.append(data)
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                search(data)

    def test_invalid_products(self):
        for replacement in [None, {}, {"id": "x", "title": " "},
                            {"id": "x", "title": "X", "tags": [1]},
                            {"id": "x", "title": "X", "description": None}]:
            data = fixture()
            data["products"][0] = replacement
            with self.subTest(product=replacement), self.assertRaises(ValidationError):
                search(data)
        data = fixture()
        data["products"][0]["id"] = "a"
        with self.assertRaises(ValidationError):
            search(data)

    def test_embedding_injection(self):
        data = fixture()
        original = copy.deepcopy(data)
        data["options"] = {"embedding_weight": 1}
        calls = []

        def synthetic_embedder(texts):
            calls.append(texts)
            return [[1, 0], [0, 1], [-1, 0], [1, 0]]

        result = search(data, synthetic_embedder)
        self.assertEqual([r["id"] for r in result["results"]], ["c"])
        self.assertEqual(result["results"][0]["embedding_score"], 1)
        self.assertTrue(result["embedding_used"])
        self.assertEqual(calls[0][0], "couch")
        self.assertEqual(len(calls[0]), 4)
        self.assertEqual(data["products"], original["products"])

    def test_invalid_embedding_fixtures(self):
        outputs = [None, [], [[1]] * 3, [[0]] * 4,
                   [[1], [1, 0], [1], [1]], [[True]] * 4,
                   [[float("nan")]] * 4, [[float("inf")]] * 4,
                   [["bad"]] * 4]
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(ValidationError):
                search(fixture(), lambda texts: output)
        with self.assertRaises(ValidationError):
            search(fixture(), "not callable")

        def broken(texts):
            raise RuntimeError("synthetic failure")

        with self.assertRaisesRegex(ValidationError, "embedding callable failed"):
            search(fixture(), broken)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=str(ROOT), capture_output=True, text=True, check=False)

    def test_cli_success(self):
        completed = self.run_cli(str(ROOT / "example_input.json"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["results"][0]["id"], "SYN-002")
        self.assertEqual(completed.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in [(), ("missing-synthetic-input.json",),
                     ("example_input.json", "extra"), ("build_manifest.json",)]:
            completed = self.run_cli(*args)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_malformed_json_error(self):
        with patch("builtins.open", mock_open(read_data="{broken")), patch("builtins.print") as output:
            self.assertEqual(main(["synthetic-invalid.json"]), 2)
        self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")

    def test_unreadable_encoding_error(self):
        with patch("builtins.open", side_effect=UnicodeError("synthetic invalid UTF-8")), \
                patch("builtins.print") as output:
            self.assertEqual(main(["synthetic-invalid.json"]), 2)
        self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")


if __name__ == "__main__":
    unittest.main()
