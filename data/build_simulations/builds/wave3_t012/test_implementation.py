import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_insight_multitheme_and_counts(self):
        result = impl.customer_insights(self.data)
        themes = {t["theme"]: t for t in result["themes"]}
        self.assertEqual(result["feedback_count"], 5)
        self.assertEqual(themes["Usability"]["mentions"], 2)
        self.assertEqual(themes["Usability"]["sentiments"],
                         {"positive": 1, "neutral": 0, "negative": 1})
        self.assertEqual(themes["Reliability"]["evidence"][0]["id"], "SYN-001")
        self.assertEqual(themes["Usability"]["priority"], "high")

    def test_rating_precedes_lexicon(self):
        self.assertEqual(impl.sentiment({"text": "good great", "rating": 1}), "negative")
        self.assertEqual(impl.sentiment({"text": "excellent helpful"}), "positive")
        self.assertEqual(impl.sentiment({"text": "slow broken"}), "negative")
        self.assertEqual(impl.sentiment({"text": "good bad"}), "neutral")

    def test_custom_phrase_and_word_boundaries(self):
        self.data["theme_rules"] = [
            {"name": "Billing", "keywords": ["DOUBLE charge"], "action": "Review invoices."}
        ]
        self.data["feedback"] = [
            {"id": "a", "customer": "SYN", "text": "Double, charge appeared."},
            {"id": "b", "customer": "SYN", "text": "Double charger appeared."},
        ]
        result = impl.customer_insights(self.data)
        self.assertEqual([t["theme"] for t in result["themes"]], ["Billing", "Other"])
        self.assertEqual(result["themes"][0]["mentions"], 1)

    def test_cross_stage_propagation(self):
        result = impl.run_pipeline(self.data)
        row = next(r for r in result["documents"]["rows"] if r["Theme"] == "Usability")
        self.assertEqual(row["Mentions"], 2)
        self.assertEqual(row["Priority"], "HIGH")
        self.assertEqual(row["Evidence IDs"], "SYN-001, SYN-004")
        self.assertEqual(result["documents"]["checks"]["source_feedback_count"], 5)
        self.data["feedback"].append(
            {"id": "SYN-006", "customer": "Synthetic Ada", "text": "Navigation is easy."})
        new = impl.run_pipeline(self.data)
        row = next(r for r in new["documents"]["rows"] if r["Theme"] == "Usability")
        self.assertEqual(row["Mentions"], 3)
        self.assertEqual(row["Customers"], 2)
        self.assertIn("SYN-006", row["Evidence IDs"])

    def test_json_document_filter_and_transform(self):
        self.data["document"] = {
            "format": "json", "min_mentions": 2,
            "columns": [{"label": "name", "source": "theme", "transform": "lower"},
                        {"label": "count", "source": "mentions"}],
        }
        document = impl.run_pipeline(self.data)["documents"]
        self.assertEqual(document["rows"], [{"name": "usability", "count": 2}])
        self.assertEqual(json.loads(document["content"]), document["rows"])
        self.assertEqual(document["checks"]["excluded_theme_count"], 4)

    def test_csv_quotes_newlines_and_formula_protection(self):
        self.data["theme_rules"] = [
            {"name": "=SYNTHETIC", "keywords": ["navigation"],
             "action": " \t=SUM(1,2)\nReview"}
        ]
        result = impl.run_pipeline(self.data)
        document = result["documents"]
        rows = list(csv.DictReader(io.StringIO(document["content"])))
        row = next(r for r in rows if r["Theme"] == "'=SYNTHETIC")
        self.assertEqual(row["Recommended action"], "' \t=SUM(1,2)\nReview")

    def test_empty_input(self):
        self.data["feedback"] = []
        result = impl.run_pipeline(self.data)
        self.assertEqual(result["insights"], {"feedback_count": 0, "themes": []})
        self.assertEqual(result["documents"]["rows"], [])
        self.assertEqual(len(list(csv.reader(io.StringIO(result["documents"]["content"])))), 1)

    def test_invalid_input(self):
        for mutate in [
            lambda d: d.update(schema_version="2"),
            lambda d: d.update(feedback="not a list"),
            lambda d: d["feedback"].append(d["feedback"][0]),
            lambda d: d["feedback"][0].update(rating=True),
            lambda d: d["feedback"][0].update(rating=6),
            lambda d: d["feedback"][0].update(text=" "),
            lambda d: d.update(unexpected=1),
            lambda d: d.update(theme_rules=[{"name": "Other", "keywords": ["x"], "action": "x"}]),
            lambda d: d.update(theme_rules=[{"name": "X", "keywords": ["!!!"], "action": "x"}]),
        ]:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(data)

    def test_invalid_document_config(self):
        for config in [
            {"format": "pdf"}, {"min_mentions": -1}, {"columns": []},
            {"columns": [{"label": "x", "source": "missing"}]},
            {"columns": [{"label": "x", "source": "mentions", "transform": "upper"}]},
            {"columns": [{"label": "x", "source": "theme"},
                         {"label": "x", "source": "action"}]},
        ]:
            with self.subTest(config=config):
                self.data["document"] = config
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(self.data)

    def test_invalid_stage_handoff_rejected(self):
        insights = impl.customer_insights(self.data)
        insights["themes"][0]["mentions"] += 1
        with self.assertRaises(impl.ValidationError):
            impl.document_automation(insights, {})

    def test_invalid_document_output_rejected(self):
        insights = impl.customer_insights(self.data)
        config = self.data["document"]
        documents = impl.document_automation(insights, config)
        documents["rows"][0]["Mentions"] = 999
        with self.assertRaises(impl.ValidationError):
            impl.validate("documents", documents, insights=insights, config=config)

    def test_determinism_and_input_not_mutated(self):
        before = copy.deepcopy(self.data)
        first = impl.run_pipeline(self.data)
        self.assertEqual(self.data, before)
        self.data["feedback"].reverse()
        self.assertEqual(first, impl.run_pipeline(self.data))

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_file_usage_and_validation_errors(self):
        for args in [(), ("missing-synthetic-input.json",), ("implementation.py",),
                     ("build_manifest.json",), ("example_input.json", "extra")]:
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_malformed_json_without_writing_files(self):
        for payload in ['{"x":1,"x":2}', '{"x":NaN}', '{"broken"', '[]']:
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.StringIO(payload)):
                    with patch("sys.stdout", output):
                        self.assertEqual(impl.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
