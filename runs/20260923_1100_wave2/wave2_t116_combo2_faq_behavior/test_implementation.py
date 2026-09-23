"""Standard-library tests using explicitly synthetic fixtures."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_cli(self, *args):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.stderr, "")
        return result.returncode, json.loads(result.stdout)

    def test_grounded_answer(self):
        faq = app.answer_faq(self.data)
        self.assertEqual(faq["answer"], self.data["knowledge_base"][0]["answer"])
        self.assertEqual(faq["source_ids"], ["faq-bottle"])

    def test_abstention(self):
        self.data["question"] = "Quantum insurance warranty?"
        output = app.run_pipeline(self.data)
        self.assertEqual(output["faq"]["status"], "abstained")
        self.assertIsNone(output["faq"]["answer"])
        self.assertTrue(all(r["components"]["faq"] == 0
                            for r in output["behavior"]["recommendations"]))

    def test_empty_query_tokens(self):
        self.data["question"] = "the and?"
        self.assertEqual(app.answer_faq(self.data)["status"], "abstained")

    def test_cross_stage_boost_and_provenance(self):
        self.data["events"] = []
        output = app.run_pipeline(self.data)
        top = output["behavior"]["recommendations"][0]
        self.assertEqual(top["product_id"], "bottle")
        self.assertEqual(top["faq_source_ids"], output["faq"]["source_ids"])
        self.data["question"] = "day pack capacity"
        self.assertEqual(app.run_pipeline(self.data)["behavior"]["recommendations"][0]["product_id"], "pack")

    def test_handoff_rejects_tampering(self):
        faq = app.answer_faq(self.data)
        faq["product_ids"] = ["hat"]
        with self.assertRaises(app.ValidationError):
            app.personalize(self.data, faq)
        faq = app.answer_faq(self.data)
        faq["answer"] = "Unsupported promise"
        with self.assertRaises(app.ValidationError):
            app.personalize(self.data, faq)

    def test_half_life_and_purchase_weights(self):
        rows = app.run_pipeline(self.data)["behavior"]["recommendations"]
        scores = {r["product_id"]: r["components"]["history"] for r in rows}
        self.assertEqual(scores["bottle"], 1.0)
        self.assertEqual(scores["pack"], 1.5)

    def test_cold_start_popularity(self):
        self.data["events"] = []
        self.data["knowledge_base"] = []
        output = app.run_pipeline(self.data)
        self.assertTrue(output["behavior"]["cold_start"])
        self.assertEqual(output["behavior"]["recommendations"][0]["product_id"], "pack")

    def test_deterministic_ties_and_limit(self):
        self.data["events"] = []
        self.data["knowledge_base"] = []
        for p in self.data["products"]:
            p["popularity"] = 0.5
        self.data["limit"] = 2
        first = app.run_pipeline(self.data)
        self.data["products"].reverse()
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual([r["product_id"] for r in first["behavior"]["recommendations"]],
                         ["bottle", "hat"])

    def test_empty_catalog(self):
        self.data.update(products=[], knowledge_base=[], events=[])
        output = app.run_pipeline(self.data)
        self.assertEqual(output["behavior"]["recommendations"], [])
        self.assertTrue(output["behavior"]["cold_start"])

    def test_invalid_inputs(self):
        variants = [
            ("limit", True), ("limit", 0), ("events", None),
            ("question", ""), ("now", "2026-09-23"),
            ("schema_version", 2), ("products", {}),
        ]
        for key, value in variants:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_references_and_events(self):
        for changes in ({"product_id": "missing"}, {"kind": "click"},
                        {"at": "2027-01-01T00:00:00Z"}):
            data = copy.deepcopy(self.data)
            data["events"][0].update(changes)
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_duplicate_ids_and_nonfinite(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)
        self.data["products"].pop()
        self.data["products"][0]["popularity"] = float("nan")
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_timezone_equivalence(self):
        first = app.run_pipeline(self.data)
        self.data["now"] = "2026-09-23T14:00:00+02:00"
        self.assertEqual(first, app.run_pipeline(self.data))

    def test_pipeline_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, before)

    def test_cli_success(self):
        code, output = self.run_cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(output, app.run_pipeline(self.data))

    def test_cli_file_error(self):
        code, output = self.run_cli("nonexistent-input.json")
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_malformed_json(self):
        code, output = self.run_cli("test_implementation.py")
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_usage_error(self):
        code, output = self.run_cli()
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_schema_error(self):
        code, output = self.run_cli("build_manifest.json")
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_strict_json_hooks(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"x": 1, "x": 2}', object_pairs_hook=app.unique_object)
        with self.assertRaises(app.ValidationError):
            json.loads('{"x": NaN}', parse_constant=app.reject_constant)


if __name__ == "__main__":
    unittest.main()
