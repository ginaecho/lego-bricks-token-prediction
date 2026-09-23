import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_exact_extractive_citations(self):
        result = app.research(self.data)
        self.assertEqual(len(result["findings"]), 2)
        sources = {s["id"]: s["text"] for s in self.data["sources"]}
        for finding in result["findings"]:
            cite = finding["citation"]
            self.assertEqual(finding["text"], cite["quote"])
            self.assertEqual(cite["quote"], sources[cite["source_id"]][cite["start"]:cite["end"]])

    def test_unicode_and_whitespace_offsets(self):
        self.data["sources"][0]["text"] = "  Café. \n   Portable battery — très bien!"
        result = app.research(self.data)
        cite = next(f["citation"] for f in result["findings"]
                    if f["citation"]["source_id"] == "source-a")
        self.assertEqual(cite["quote"], "Portable battery — très bien!")
        self.assertEqual(cite["start"], 12)

    def test_normalization_and_side_by_side(self):
        result = app.run_pipeline(self.data)["comparison"]
        self.assertEqual(result["columns"], list(app.ATTRIBUTES))
        self.assertEqual(result["rows"][0]["attributes"],
                         {"price_usd": 120.0, "weight_g": 800.0, "battery_hours": 10.0})
        self.assertTrue(all(row["finding_ids"] for row in result["rows"]))

    def test_preference_ranking(self):
        ranks = app.run_pipeline(self.data)["comparison"]["ranking"]
        self.assertEqual([r["product_id"] for r in ranks], ["aero", "breeze"])
        self.assertAlmostEqual(ranks[0]["score"], 2 / 3)
        self.data["preferences"][0]["weight"] = 20
        ranks = app.run_pipeline(self.data)["comparison"]["ranking"]
        self.assertEqual(ranks[0]["product_id"], "breeze")

    def test_research_filters_comparison_and_feedback(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["product_ids"], ["aero", "breeze"])
        self.assertNotIn("desk", result["feedback"]["selected_product_ids"])
        excerpts = [e for theme in result["feedback"]["themes"]
                    for e in theme["supporting_excerpts"]]
        self.assertNotIn("f5", [e["feedback_id"] for e in excerpts])

    def test_retrieval_limit_propagates(self):
        self.data["max_findings"] = 1
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["product_ids"], ["aero"])
        self.assertEqual(len(result["comparison"]["rows"]), 1)
        self.assertEqual(result["feedback"]["selected_product_ids"], ["aero"])

    def test_feedback_dedup_and_trace(self):
        feedback = app.run_pipeline(self.data)["feedback"]
        self.assertEqual(feedback["unique_feedback_count"], 4)
        self.assertEqual(feedback["duplicate_count"], 1)
        battery = next(t for t in feedback["themes"] if t["theme"] == "battery")
        self.assertEqual(battery["count"], 2)
        excerpt = battery["supporting_excerpts"][0]
        self.assertEqual(excerpt["feedback_id"], "f1")
        self.assertEqual(excerpt["duplicate_ids"], ["f2"])
        self.assertEqual(excerpt["text"], self.data["feedback"][0]["text"])
        self.assertEqual(excerpt["comparison_rank"], 1)

    def test_dedup_does_not_merge_products(self):
        self.data["feedback"].append(
            {"id": "f7", "product_id": "breeze", "text": "Great battery, easy setup!"})
        result = app.run_pipeline(self.data)["feedback"]
        self.assertEqual(result["unique_feedback_count"], 5)
        self.assertEqual(result["duplicate_count"], 1)

    def test_rank_change_reaches_feedback(self):
        self.data["preferences"][0]["weight"] = 20
        result = app.run_pipeline(self.data)
        self.assertEqual(result["feedback"]["selected_product_ids"], ["breeze", "aero"])
        battery = next(t for t in result["feedback"]["themes"] if t["theme"] == "battery")
        self.assertEqual(battery["supporting_excerpts"][0]["product_id"], "breeze")
        self.assertEqual(battery["supporting_excerpts"][0]["comparison_rank"], 1)

    def test_no_matches(self):
        self.data["query"] = "unfindable"
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["findings"], [])
        self.assertEqual(result["comparison"]["ranking"], [])
        self.assertEqual(result["feedback"]["themes"], [])
        self.assertEqual(result["feedback"]["unique_feedback_count"], 0)

    def test_empty_collections(self):
        for field in ("sources", "products", "feedback"):
            self.data[field] = []
        self.assertEqual(app.run_pipeline(self.data)["status"], "ok")

    def test_missing_attributes_receive_zero_utility(self):
        self.data["products"][0]["attributes"] = {}
        result = app.run_pipeline(self.data)["comparison"]
        self.assertTrue(all(v is None for v in result["rows"][0]["attributes"].values()))
        self.assertEqual(result["ranking"][-1]["product_id"], "aero")
        self.assertEqual(result["ranking"][-1]["score"], 0)

    def test_ties_have_stable_order(self):
        self.data["products"][1]["attributes"] = copy.deepcopy(
            self.data["products"][0]["attributes"])
        result = app.run_pipeline(self.data)["comparison"]["ranking"]
        self.assertEqual([r["product_id"] for r in result], ["aero", "breeze"])
        self.assertEqual(result[0]["score"], 1)
        self.assertEqual(result[1]["score"], 1)

    def test_unknown_theme_is_preserved(self):
        themes = app.run_pipeline(self.data)["feedback"]["themes"]
        other = next(t for t in themes if t["theme"] == "other")
        self.assertEqual(other["supporting_excerpts"][0]["feedback_id"], "f6")

    def test_invalid_input_shapes(self):
        changes = [
            ("schema_version", True), ("synthetic", False), ("query", "and the"),
            ("sources", None), ("products", {}), ("feedback", "bad"),
            ("preferences", []), ("max_findings", 0), ("max_findings", True),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(self.data)
                bad[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)

    def test_duplicate_ids_and_dangling_references(self):
        for field in ("sources", "products", "feedback"):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.data)
                bad[field].append(copy.deepcopy(bad[field][0]))
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)
        self.data["products"][0]["source_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_invalid_measurements(self):
        for value in (-1, True, "120", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=str(value)):
                bad = copy.deepcopy(self.data)
                bad["products"][0]["attributes"]["price"]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)
        self.data["products"][0]["attributes"]["weight"]["unit"] = "lb"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_currency_is_not_silently_converted(self):
        self.data["products"][0]["attributes"]["price"]["currency"] = "EUR"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_invalid_nested_units_return_json_errors(self):
        for unit in (None, [], {}, 100, True):
            with self.subTest(unit=unit):
                self.data["products"][0]["attributes"]["weight"]["unit"] = unit
                output = io.StringIO()
                with mock.patch.object(app.Path, "open",
                                       return_value=io.StringIO(json.dumps(self.data))):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["invalid-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_invalid_feedback_reference_and_blank_text(self):
        self.data["feedback"][0]["product_id"] = "absent"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data["feedback"][0]["product_id"] = "aero"
        for body in ("", "   ", "!!!", None, []):
            with self.subTest(body=body):
                self.data["feedback"][0]["text"] = body
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_normalization_overflow_is_rejected(self):
        self.data["products"][0]["attributes"]["weight"]["value"] = 1e308
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_invalid_preferences(self):
        for field, value in (("weight", 0), ("weight", -1), ("weight", True),
                             ("attribute", "speed"), ("attribute", []),
                             ("direction", "sideways")):
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(self.data)
                bad["preferences"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)

    def test_handoff_rejects_tampered_citation(self):
        research = app.research(self.data)
        research["findings"][0]["citation"]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.compare(self.data, research)

    def test_handoff_rejects_unsupported_product(self):
        research = app.research(self.data)
        research["product_ids"].append("desk")
        with self.assertRaises(app.ValidationError):
            app.compare(self.data, research)

    def test_handoff_rejects_tampered_rank(self):
        research = app.research(self.data)
        comparison = app.compare(self.data, research)
        comparison["ranking"][0]["rank"] = 99
        with self.assertRaises(app.ValidationError):
            app.analyze_feedback(self.data, research, comparison)

    def test_feedback_validation_rejects_fabricated_excerpt(self):
        research = app.research(self.data)
        comparison = app.compare(self.data, research)
        feedback = app.analyze_feedback(self.data, research, comparison)
        feedback["themes"][0]["supporting_excerpts"][0]["text"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.validate("feedback", feedback, self.data, comparison)

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        expected = app.run_pipeline(self.data)
        self.assertEqual(app.run_pipeline(self.data), expected)
        self.assertEqual(self.data, original)
        for field in ("sources", "products", "feedback"):
            self.data[field].reverse()
        self.assertEqual(app.run_pipeline(self.data), expected)

    def test_large_finite_preference_weights(self):
        for pref in self.data["preferences"]:
            pref["weight"] = 1e308
        self.assertEqual(app.run_pipeline(self.data)["status"], "ok")

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_file_error(self):
        result = self.cli("nonexistent-fixture.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_usage_error(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json(self):
        result = self.cli("implementation.py")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_schema(self):
        result = self.cli("build_manifest.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_json_parser_rejects_nonfinite_and_duplicate_keys(self):
        for value in ('{"query":"a","query":"b"}', '{"value":NaN}', '{"value":Infinity}'):
            with self.subTest(value=value):
                with self.assertRaises(app.ValidationError):
                    json.loads(value, object_pairs_hook=app.unique_object,
                               parse_constant=app.reject_constant)


if __name__ == "__main__":
    unittest.main()
