import copy
import io
import json
import pathlib
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = pathlib.Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def envelope(self):
        return {"schema_version": 1, "status": "ok", "stage": "input",
                "data": {"request": copy.deepcopy(self.request)}}

    def test_research_relevance_and_provenance(self):
        research = app.research_stage(self.envelope())["data"]["research"]
        self.assertEqual([x["id"] for x in research["evidence"]], ["s1", "s2"])
        self.assertIn("comfort", research["needs"])
        self.assertEqual(research["evidence"][0]["excerpt"], self.request["sources"][0]["text"])

    def test_synonyms_and_typo(self):
        search = app.run(self.request)["data"]["search"]
        self.assertEqual(search["interpreted_terms"], ["bluetooth", "comfort", "headphone"])
        self.assertEqual(set(x["id"] for x in search["results"]), {"p1", "p2"})

    def test_budget_availability_and_limit(self):
        self.request["products"][0]["available"] = False
        self.request["query"]["limit"] = 1
        results = app.run(self.request)["data"]["search"]["results"]
        self.assertEqual([x["id"] for x in results], ["p2"])

    def test_cross_stage_research_drives_empty_query(self):
        self.request["query"]["text"] = ""
        output = app.run(self.request)["data"]
        self.assertTrue(output["search"]["results"])
        self.assertEqual(output["search"]["research_source_ids"], ["s1", "s2"])
        self.request["sources"] = []
        output = app.run(self.request)["data"]
        self.assertEqual(output["search"]["results"], [])
        self.assertEqual(output["insights"]["feedback_ids"], [])

    def test_feedback_scoped_to_search(self):
        insights = app.run(self.request)["data"]["insights"]
        self.assertEqual(insights["feedback_ids"], ["f1", "f2", "f3"])
        battery = next(t for t in insights["themes"] if t["id"] == "battery")
        self.assertEqual((battery["count"], battery["negative"], battery["positive"]), (2, 1, 1))
        self.assertIn("Investigate", battery["action"])

    def test_research_rejects_zero_trust(self):
        for row in self.request["sources"]:
            row["credibility"] = 0
        self.assertEqual(app.run(self.request)["data"]["research"]["evidence"], [])

    def test_empty_collections(self):
        self.request.update(sources=[], products=[], feedback=[])
        output = app.run(self.request)["data"]
        self.assertEqual(output["search"]["results"], [])
        self.assertEqual(output["insights"]["themes"], [])

    def test_no_matching_query(self):
        self.request["sources"] = []
        self.request["query"]["text"] = "xyzunknown"
        self.assertEqual(app.run(self.request)["data"]["search"]["results"], [])

    def test_rating_precedence_and_lexical_fallback(self):
        self.assertEqual(app.sentiment({"rating": 1, "text": "great excellent"}), "negative")
        self.assertEqual(app.sentiment({"rating": None, "text": "great easy"}), "positive")
        self.assertEqual(app.sentiment({"rating": None, "text": "bad broken"}), "negative")
        self.assertEqual(app.sentiment({"rating": None, "text": "average"}), "neutral")
        self.assertEqual(app.sentiment({"rating": 3, "text": "great"}), "neutral")

    def test_other_theme(self):
        self.request["feedback"][0]["text"] = "Purple packaging"
        themes = app.run(self.request)["data"]["insights"]["themes"]
        self.assertIn("other", [row["id"] for row in themes])

    def test_determinism_and_input_immutability(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(app.run(self.request), app.run(self.request))
        self.assertEqual(self.request, original)

    def test_duplicate_and_unknown_references(self):
        for mutation in ("duplicate", "unknown"):
            request = copy.deepcopy(self.request)
            if mutation == "duplicate":
                request["products"].append(copy.deepcopy(request["products"][0]))
            else:
                request["feedback"][0]["product_id"] = "missing"
            with self.subTest(mutation=mutation), self.assertRaises(app.ValidationError):
                app.run(request)

    def test_invalid_numeric_inputs(self):
        for value in (True, -1, float("nan"), float("inf"), "20"):
            request = copy.deepcopy(self.request)
            request["products"][0]["price"] = value
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run(request)
        for value in (True, 0, 101, 1.5):
            self.request["query"]["limit"] = value
            with self.subTest(limit=value), self.assertRaises(app.ValidationError):
                app.run(self.request)

    def test_invalid_structure(self):
        for request in (None, [], {}, {"schema_version": 1}):
            with self.subTest(request=request), self.assertRaises(app.ValidationError):
                app.run(request)
        self.request["question"] = "  "
        with self.assertRaises(app.ValidationError):
            app.run(self.request)

    def test_handoff_rejects_wrong_stage_and_forgery(self):
        with self.assertRaises(app.ValidationError):
            app.search_stage(self.envelope())
        research = app.research_stage(self.envelope())
        research["data"]["research"]["evidence"][0]["excerpt"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.search_stage(research)
        search = app.search_stage(app.research_stage(self.envelope()))
        search["data"]["search"]["research_source_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.insights_stage(search)

    def test_final_validation_rejects_feedback_leak(self):
        output = app.run(self.request)
        output["data"]["insights"]["feedback_ids"].append("f4")
        with self.assertRaises(app.ValidationError):
            app.validate(output, "insights")

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_single_json(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_usage_and_file_errors(self):
        for args in ((), ("nonexistent-input.json",), ("example_input.json", "extra")):
            result = self.cli(*args)
            with self.subTest(args=args):
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_and_invalid_schema(self):
        for content in ("{broken", "[]", '{"schema_version":1}', '{"x":NaN}'):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                code = app.main(["injected.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_price_boundary_and_tie_break(self):
        self.request["sources"] = []
        self.request["query"].update(text="headphones", max_price=65)
        product = self.request["products"][1]
        product["price"] = 65
        results = app.run(self.request)["data"]["search"]["results"]
        self.assertEqual([r["id"] for r in results], ["p1", "p2"])

    def test_large_integer_price_and_out_of_range_credibility(self):
        self.request["products"][0]["price"] = 10 ** 400
        results = app.run(self.request)["data"]["search"]["results"]
        self.assertNotIn("p1", [row["id"] for row in results])
        self.request["sources"][0]["credibility"] = 10 ** 400
        with self.assertRaises(app.ValidationError):
            app.run(self.request)


if __name__ == "__main__":
    unittest.main()
