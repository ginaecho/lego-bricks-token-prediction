import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_weighted_ranking_and_grounding(self):
        result = app.recommend(self.data)
        rows = result["recommendations"]
        self.assertEqual([r["item_id"] for r in rows], ["demo-1", "demo-2"])
        self.assertEqual([r["score"] for r in rows], [1.0, 0.6])
        self.assertEqual(rows[0]["explanation"]["matched_interests"], [
            {"interest": "outdoors", "weight": 3, "sources": ["tags"]},
            {"interest": "photography", "weight": 2, "sources": ["category", "tags"]}
        ])
        self.assertEqual([r["rank"] for r in rows], [1, 2])

    def test_all_exclusion_types_override_preferences(self):
        result = app.recommend(self.data)
        excluded = result["audit"]["excluded"]
        self.assertEqual([x["item_id"] for x in excluded], ["demo-3", "demo-4", "demo-6"])
        self.assertEqual([x["reasons"][0]["kind"] for x in excluded], ["tag", "item_id", "category"])
        self.assertEqual(result["audit"]["unmatched_item_ids"], ["demo-5"])

    def test_fallback_respects_exclusions(self):
        self.data["preferences"]["interests"] = {"outdoors": 0}
        result = app.recommend(self.data)
        self.assertEqual(result["mode"], "popularity_fallback")
        self.assertEqual([x["item_id"] for x in result["recommendations"]], ["demo-2", "demo-5", "demo-1"])
        self.assertTrue(all(x["score"] == 0 for x in result["recommendations"]))

    def test_empty_catalog_and_zero_limit(self):
        self.data["limit"] = 0
        result = app.recommend(self.data)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["audit"]["omitted_by_limit"], 2)
        self.data["catalog"] = []
        self.assertEqual(app.recommend(self.data)["audit"]["eligible_count"], 0)

    def test_no_matching_interests_does_not_silently_fallback(self):
        self.data["preferences"]["interests"] = {"unknown-interest": 5}
        result = app.recommend(self.data)
        self.assertEqual(result["mode"], "personalized")
        self.assertEqual(result["recommendations"], [])

    def test_ties_are_deterministic_and_input_is_unchanged(self):
        other = copy.deepcopy(self.data["catalog"][0])
        other["id"] = "demo-0"
        self.data["catalog"].append(other)
        original = copy.deepcopy(self.data)
        first = app.recommend(self.data)
        self.assertEqual(self.data, original)
        self.data["catalog"].reverse()
        self.assertEqual(first, app.recommend(self.data))
        self.assertEqual(first["recommendations"][0]["item_id"], "demo-0")

    def test_invalid_root_and_limits(self):
        for root in (None, [], "text", {}, {"schema_version": 1}):
            with self.subTest(root=root), self.assertRaises(app.ValidationError):
                app.recommend(root)
        for limit in (True, -1, 101, 1.5, "3"):
            with self.subTest(limit=limit), self.assertRaises(app.ValidationError):
                app.recommend(dict(self.data, limit=limit))

    def test_invalid_catalog_and_preferences(self):
        for mutate in (
            lambda d: d["catalog"].append(copy.deepcopy(d["catalog"][0])),
            lambda d: d["catalog"][0].update(popularity=float("nan")),
            lambda d: d["catalog"][0].update(popularity=True),
            lambda d: d["catalog"][0].update(tags=["outdoors", "outdoors"]),
            lambda d: d["catalog"][0].update(id=" "),
            lambda d: d["preferences"]["interests"].update(outdoors=-1),
            lambda d: d["preferences"]["interests"].update(outdoors=float("inf")),
            lambda d: d["preferences"]["exclusions"].update(tags="bad"),
            lambda d: d.update(unknown="field"),
            lambda d: d.update(fixture_label="real"),
        ):
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.recommend(data)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), app.recommend(self.data))
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], [str(ROOT / "missing-input.json")], ["a", "b"]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_duplicate_and_nonfinite_json(self):
        for raw in ("{", '{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', "[]"):
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["synthetic-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
