import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_document_reshape_and_provenance(self):
        item = app.automate_documents(self.payload["documents"])[0]
        self.assertEqual(item["title"], "Garden Planning")
        self.assertEqual(item["tags"], ["gardening", "outdoors"])
        self.assertEqual(item["source"]["row"], 0)
        self.assertEqual(item["source"]["fields"]["title"], "details.name")

    def test_ranking_and_grounding(self):
        result = app.run(self.payload)
        self.assertEqual([x["id"] for x in result["recommendations"]], ["r1", "r2", "r4"])
        best = result["recommendations"][0]
        self.assertEqual(best["score"], 4.8)
        self.assertEqual(best["explanation"]["matched_interests"],
                         [{"tag": "gardening", "weight": 3.0}, {"tag": "outdoors", "weight": 1.0}])
        self.assertTrue(result["recommendations"][-1]["explanation"]["fallback"])

    def test_cross_stage_tag_and_title_propagation(self):
        row = self.payload["documents"][0]["rows"][1]
        row["topics"] = ["gardening", "outdoors"]
        row["details"]["name"] = " Revised   resource "
        result = app.run(self.payload)
        self.assertEqual(result["recommendations"][0]["id"], "r2")
        self.assertEqual(result["recommendations"][0]["title"], "Revised resource")
        self.assertEqual(result["recommendations"][0]["score"], 4.9)

    def test_exclusions_override_high_preferences(self):
        self.payload["profile"]["weights"]["extreme"] = 10
        exclusions = self.payload["profile"]["exclusions"]
        exclusions["ids"] = ["r1"]
        exclusions["terms"] = ["OUTDOOR SCENES"]
        result = app.run(self.payload)
        self.assertEqual([x["id"] for x in result["recommendations"]], ["r4"])
        self.assertEqual(len(result["excluded"]), 3)
        self.assertEqual(result["excluded"][1]["reasons"][0]["field"], "description")

    def test_empty_documents_and_zero_limit(self):
        self.payload["documents"] = []
        self.assertEqual(app.run(self.payload)["recommendations"], [])
        self.setUp()
        self.payload["profile"]["limit"] = 0
        result = app.run(self.payload)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["eligible_count"], 3)

    def test_default_quality_and_negative_preference(self):
        del self.payload["documents"][0]["fields"]["quality"]
        self.payload["profile"]["weights"] = {"gardening": -2}
        result = app.run(self.payload)
        self.assertEqual([x["id"] for x in result["recommendations"]], ["r2", "r4", "r1"])
        self.assertEqual(result["catalog"][0]["quality"], 0)

    def test_missing_source_field_rejected(self):
        del self.payload["documents"][0]["rows"][0]["details"]["name"]
        with self.assertRaisesRegex(app.ValidationError, "missing source"):
            app.run(self.payload)

    def test_duplicate_items_across_documents_rejected(self):
        second = copy.deepcopy(self.payload["documents"][0])
        second["document_id"] = "second"
        self.payload["documents"].append(second)
        with self.assertRaisesRegex(app.ValidationError, "duplicate item"):
            app.run(self.payload)

    def test_invalid_numeric_values(self):
        for value in (True, "1", float("inf"), float("nan"), -1, 2):
            with self.subTest(value=value):
                self.payload["documents"][0]["rows"][0]["quality"] = value
                with self.assertRaises(app.ValidationError):
                    app.run(self.payload)

    def test_shared_contract_rejects_noncanonical_handoff(self):
        items = app.automate_documents(self.payload["documents"])
        items[0]["tags"] = ["Gardening"]
        with self.assertRaisesRegex(app.ValidationError, "canonical"):
            app.recommend(items, self.payload["profile"])

    def test_bad_profile_and_unknown_fields(self):
        for key, value in (("limit", True), ("limit", 101), ("weights", {"x": "2"}),
                           ("weights", {" Gardening ": 1, "gardening": 2})):
            with self.subTest(key=key, value=value):
                candidate = copy.deepcopy(self.payload)
                candidate["profile"][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(candidate)
        self.payload["unexpected"] = 1
        with self.assertRaises(app.ValidationError):
            app.run(self.payload)

    def test_invalid_document_shapes(self):
        for replacement in (None, {}, "records"):
            with self.subTest(replacement=replacement):
                self.payload["documents"] = replacement
                with self.assertRaises(app.ValidationError):
                    app.run(self.payload)

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.payload)
        first = app.run(self.payload)
        self.assertEqual(first, app.run(self.payload))
        self.assertEqual(original, self.payload)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], cwd=ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "nonexistent.json")], ["a", "b"]):
            with self.subTest(args=args):
                proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_schema_without_scratch_files(self):
        for raw in ("{", '{"a":1,"a":2}', '{"value":NaN}', '{}', '[]'):
            with self.subTest(raw=raw):
                out = io.StringIO()
                with patch.object(Path, "read_text", return_value=raw), contextlib.redirect_stdout(out):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
