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


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_preference_ranking(self):
        result = app.recommend(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [(r["item_id"], r["score"], r["rank"]) for r in result["recommendations"]],
            [("planter", 8, 1), ("seed-kit", 5, 2), ("craft-kit", 2, 3)],
        )
        self.assertEqual(result["audit"]["unmatched_count"], 1)

    def test_all_exclusions_override_preferences(self):
        result = app.recommend(self.data)
        excluded = {item["item_id"]: item["reasons"] for item in result["audit"]["excluded"]}
        self.assertEqual(set(excluded), {"blocked-kit", "fertilizer", "disposable-pot"})
        self.assertEqual(excluded["blocked-kit"], [{"field": "id", "value": "blocked-kit"}])
        self.assertEqual(excluded["fertilizer"], [{"field": "category", "value": "chemicals"}])
        self.assertEqual(excluded["disposable-pot"], [{"field": "tags", "value": "single-use"}])
        self.assertTrue(set(excluded).isdisjoint(r["item_id"] for r in result["recommendations"]))

    def test_grounded_explanations(self):
        catalog = {item["id"]: item for item in self.data["catalog"]}
        for rec in app.recommend(self.data)["recommendations"]:
            expected = []
            for match in rec["matched_interests"]:
                self.assertIn(match["tag"], catalog[rec["item_id"]]["tags"])
                self.assertEqual(match["weight"], self.data["user"]["interests"][match["tag"]])
                self.assertEqual(match["source"], "catalog.tags")
                expected.append(f"{match['tag']} (weight {match['weight']})")
            self.assertEqual(rec["score"], sum(m["weight"] for m in rec["matched_interests"]))
            self.assertEqual(rec["explanation"],
                             f"Catalog tags match your interests: {', '.join(expected)}. Total score: {rec['score']}.")

    def test_tie_breaking_and_catalog_permutation(self):
        self.data["catalog"].append({
            "id": "aaa", "title": "Synthetic second seeds", "category": "garden",
            "tags": ["gardening"],
        })
        before = app.recommend(self.data)
        self.assertEqual([r["item_id"] for r in before["recommendations"]],
                         ["planter", "aaa", "seed-kit"])
        self.data["catalog"].reverse()
        self.assertEqual(app.recommend(self.data), before)

    def test_normalization_without_mutation(self):
        self.data["user"]["interests"] = {" Gardening ": 5}
        self.data["catalog"][0]["tags"] = [" GARDENING "]
        self.data["user"]["exclusions"]["categories"] = [" CHEMICALS "]
        original = copy.deepcopy(self.data)
        result = app.recommend(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(result["recommendations"][0]["matched_interests"][0]["tag"], "gardening")
        self.assertIn("fertilizer", [r["item_id"] for r in result["audit"]["excluded"]])

    def test_empty_interests_catalog_and_zero_limit(self):
        for field in ("interests", "catalog", "limit"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                if field == "interests":
                    data["user"]["interests"] = {}
                elif field == "catalog":
                    data["catalog"] = []
                else:
                    data["limit"] = 0
                self.assertEqual(app.recommend(data)["recommendations"], [])

    def test_all_excluded(self):
        self.data["user"]["exclusions"]["item_ids"] = [i["id"] for i in self.data["catalog"]]
        result = app.recommend(self.data)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(len(result["audit"]["excluded"]), 7)
        self.assertEqual(result["audit"]["unmatched_count"], 0)

    def test_invalid_schema(self):
        invalid = [
            None, [], {},
            {**self.data, "limit": True},
            {**self.data, "limit": -1},
            {**self.data, "limit": 101},
            {**self.data, "limit": 1.5},
            {**self.data, "unexpected": "field"},
            {**self.data, "fixture_label": "real data"},
            {**self.data, "catalog": {}},
            {**self.data, "catalog": self.data["catalog"] * 200},
        ]
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(app.ValidationError):
                    app.recommend(value)

    def test_invalid_interests_and_items(self):
        for interests in ([], {"gardening": 0}, {"gardening": True},
                          {"gardening": 101}, {" ": 2},
                          {"Gardening": 1, " gardening ": 2}):
            with self.subTest(interests=interests):
                self.data["user"]["interests"] = interests
                with self.assertRaises(app.ValidationError):
                    app.recommend(self.data)
        self.setUp()
        for change in ({"tags": ["Garden", "garden"]}, {"id": " "},
                       {"title": 5}, {"tags": "gardening"}, {"category": None}):
            data = copy.deepcopy(self.data)
            data["catalog"][0].update(change)
            with self.subTest(change=change), self.assertRaises(app.ValidationError):
                app.recommend(data)
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_invalid_exclusions(self):
        for value in (None, {"item_ids": []},
                      {"item_ids": [], "categories": [], "tags": ["X", " x "]},
                      {"item_ids": [False], "categories": [], "tags": []}):
            self.data["user"]["exclusions"] = value
            with self.assertRaises(app.ValidationError):
                app.recommend(self.data)

    def test_cli_success(self):
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), app.recommend(self.data))
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in ([], ["missing-synthetic-input.json"], ["one", "two"]):
            with self.subTest(args=args):
                proc = subprocess.run([sys.executable, "-B", "implementation.py", *args],
                                      cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        for payload in ('{', '{"a": 1, "a": 2}', 'NaN', 'Infinity', '[]',
                        json.dumps({**self.data, "limit": False})):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.StringIO(payload)), redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_utf8(self):
        output = io.StringIO()
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        with patch.object(Path, "open", side_effect=error), redirect_stdout(output):
            self.assertEqual(app.main(["synthetic-invalid.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
