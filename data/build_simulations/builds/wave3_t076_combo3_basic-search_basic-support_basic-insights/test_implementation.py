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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_handoffs(self):
        output = app.run(self.data)
        self.assertEqual(output["search"], output["support"]["search"])
        self.assertEqual(output["support"]["feedback"], self.data["feedback"])
        self.assertEqual(output["insights"]["matched_product_ids"], ["demo-a"])
        self.assertEqual(output["insights"]["feedback_count"], 2)
        self.assertEqual(output["insights"]["excluded_feedback_count"], 1)

    def test_synonyms(self):
        self.data["query"] = "sneakers"
        self.assertEqual(app.run(self.data)["search"]["matches"][0]["product"]["id"], "demo-b")

    def test_typo(self):
        self.data["query"] = "bluetoth"
        self.assertEqual(app.run(self.data)["search"]["matches"][0]["product"]["id"], "demo-a")

    def test_filters(self):
        self.data["options"]["max_price"] = 10
        self.assertEqual(app.run(self.data)["search"]["matches"], [])

    def test_limit_and_stock(self):
        self.data["options"].update(in_stock_only=False, limit=1)
        self.assertEqual(len(app.run(self.data)["search"]["matches"]), 1)

    def test_deterministic(self):
        self.assertEqual(app.run(self.data), app.run(copy.deepcopy(self.data)))

    def test_grounded_policy_and_product(self):
        supported = app.run(self.data)["support"]
        self.assertTrue(supported["resolved"])
        self.assertEqual({c["source"] for c in supported["citations"]},
                         {"policy:demo-returns", "product:demo-a"})
        self.assertEqual(supported["answer"], "\n".join(c["text"] for c in supported["citations"]))

    def test_unknown_question_gap(self):
        self.data["question"] = "Can this cure a headache?"
        output = app.run(self.data)
        self.assertFalse(output["support"]["resolved"])
        self.assertEqual(output["insights"]["support_gap"]["question"], self.data["question"])

    def test_empty_collections(self):
        self.data.update(catalog=[], feedback=[], policies=[])
        output = app.run(self.data)
        self.assertFalse(output["support"]["resolved"])
        self.assertEqual(output["insights"]["themes"], [])

    def test_stopwords_query(self):
        self.data["query"] = "the and"
        self.assertEqual(app.run(self.data)["search"]["matches"], [])

    def test_theme_and_sentiment(self):
        analysis = app.run(self.data)["insights"]
        self.assertEqual(analysis["sentiment"], {"positive": 0, "negative": 1, "mixed": 1, "neutral": 0})
        self.assertEqual({t["theme"] for t in analysis["themes"]}, {"delivery", "quality", "usability"})
        self.assertTrue(all(t["feedback_ids"] for t in analysis["themes"]))

    def test_valid_injected_selector(self):
        output = app.run(self.data, lambda evidence: [evidence[0]["source"]])
        self.assertEqual(len(output["support"]["citations"]), 1)

    def test_invalid_injected_selector(self):
        for result in (["made-up"], "bad", [1], ["policy:demo-returns"] * 2):
            with self.subTest(result=result), self.assertRaises(app.ValidationError):
                app.run(self.data, lambda evidence: result)

    def test_bad_input(self):
        for field, value in (("query", ""), ("catalog", {}), ("synthetic", False),
                             ("schema_version", True), ("feedback", None)):
            data = copy.deepcopy(self.data)
            data[field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_bad_prices_and_options(self):
        for price in (-1, float("nan"), float("inf"), True, "4"):
            data = copy.deepcopy(self.data)
            data["catalog"][0]["price"] = price
            with self.subTest(price=price), self.assertRaises(app.ValidationError):
                app.run(data)
        self.data["options"]["limit"] = True
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_duplicate_ids_and_dangling_feedback(self):
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.run(self.data)
        self.data["catalog"].pop()
        self.data["feedback"][0]["product_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_tampered_handoff_rejected(self):
        result = app.search(self.data)
        result["query"] = "changed"
        with self.assertRaises(app.ValidationError):
            app.support(self.data, result)
        supported = app.run(self.data)["support"]
        supported["answer"] = "Invented guarantee."
        with self.assertRaises(app.ValidationError):
            app.validate(supported, "support", self.data)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "nonexistent.json")]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_bad_json_and_schema(self):
        for raw in ("{bad", "{}", '{"query": "x", "query": "y"}', "null"):
            with patch("builtins.open", return_value=io.StringIO(raw)), redirect_stdout(io.StringIO()) as out:
                code = app.main(["fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
