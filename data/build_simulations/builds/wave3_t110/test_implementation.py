import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def stages(self):
        return app.run_pipeline(self.data)["stages"]

    def test_complete_pipeline(self):
        outputs = self.stages()
        self.assertEqual([s["stage"] for s in outputs], list(app.STAGES))
        self.assertTrue(all(s["status"] == "ok" for s in outputs))

    def test_synonym_search(self):
        self.data["query"] = "rucksack"
        self.assertEqual(len(self.stages()[0]["records"]), 3)

    def test_typo_search(self):
        self.data["query"] = "backpak"
        self.assertEqual(len(self.stages()[0]["records"]), 3)

    def test_no_search_match_propagates(self):
        self.data["query"] = "xylophone"
        result = app.run_pipeline(self.data)
        self.assertTrue(all(not s["records"] for s in result["stages"]))
        self.assertEqual(result["answer"]["status"], "abstained")

    def test_stopwords_do_not_match(self):
        self.data["query"] = "the and for"
        self.assertEqual(self.stages()[0]["status"], "empty")

    def test_preference_ranking(self):
        self.data["preferences"]["budget_max"] = None
        ranked = self.stages()[1]["records"]
        self.assertEqual(ranked[0]["product_id"], "p1")
        self.assertIn("Preferred tags: recycled", ranked[0]["text"])

    def test_exclusions_and_budget_propagate(self):
        stages = self.stages()
        self.assertEqual(stages[0]["product_ids"], ["p1", "p2", "p3"])
        for stage in stages[1:]:
            self.assertEqual(stage["product_ids"], ["p1"])

    def test_explicit_exclusion(self):
        self.data["preferences"]["excluded_product_ids"] = ["p1"]
        self.assertTrue(all(not s["records"] for s in self.stages()[1:]))

    def test_exclusion_case_insensitive(self):
        self.data["preferences"]["excluded_tags"] = ["RECYCLED", "LEATHER"]
        self.assertEqual(self.stages()[1]["status"], "empty")

    def test_budget_boundary(self):
        self.data["preferences"]["budget_max"] = 60
        self.assertEqual(self.stages()[1]["product_ids"], ["p1"])

    def test_exact_research_citations(self):
        sources = {s["id"]: s["text"] for s in self.data["research"]}
        for r in self.stages()[2]["records"]:
            for c in r["citations"]:
                self.assertEqual(c["quote"], sources[c["source_id"]][c["start"]:c["end"]])
                self.assertEqual(r["text"], c["quote"])

    def test_faq_grounded_negative_answer(self):
        answer = app.run_pipeline(self.data)["answer"]
        self.assertIn("not waterproof", answer["text"])
        self.assertNotIn("unsealed seams", answer["text"])

    def test_missing_research_blocks_faq(self):
        self.data["research"] = []
        self.assertEqual(self.stages()[3]["status"], "abstained")

    def test_missing_kb_abstains(self):
        self.data["knowledge_base"] = []
        self.assertEqual(self.stages()[3]["status"], "abstained")

    def test_unrelated_question_abstains(self):
        self.data["question"] = "What orbital velocity?"
        self.assertEqual(self.stages()[3]["status"], "abstained")

    def test_product_overlap_alone_abstains(self):
        self.data["question"] = "Backpack warranty?"
        self.assertEqual(self.stages()[3]["status"], "abstained")

    def test_newline_passages_without_punctuation(self):
        self.data["research"][0]["text"] = "Light hiking backpack\nBackpack is not waterproof"
        findings = self.stages()[2]["records"]
        self.assertEqual(len(findings), 2)
        self.assertEqual({r["text"] for r in findings},
                         {"Light hiking backpack", "Backpack is not waterproof"})

    def test_upstream_references(self):
        stages = self.stages()
        for prior, current in zip(stages, stages[1:]):
            parents = {r["id"]: r for r in prior["records"]}
            for r in current["records"]:
                self.assertTrue(r["upstream_ids"])
                for ref in r["upstream_ids"]:
                    self.assertEqual(parents[ref]["product_id"], r["product_id"])

    def test_search_limit_constrains_downstream(self):
        self.data["limits"]["search"] = 1
        stages = self.stages()
        for stage in stages[1:]:
            self.assertTrue(set(stage["product_ids"]) <= set(stages[0]["product_ids"]))

    def test_deterministic_and_nonmutating(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, before)

    def test_empty_catalog(self):
        self.data["products"] = []
        self.data["research"] = []
        self.data["knowledge_base"] = []
        self.assertEqual(app.run_pipeline(self.data)["answer"]["status"], "abstained")

    def test_invalid_inputs(self):
        for field, value in [("query", ""), ("synthetic", False), ("products", None),
                             ("question", 2), ("preferences", [])]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_numbers(self):
        for value in [True, -1, float("nan"), float("inf"), "50", 10 ** 500]:
            with self.subTest(value=value):
                self.data["products"][0]["price"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_duplicate_product(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        with self.assertRaises(app.ValidationError):
            self.stages()

    def test_unknown_source_product(self):
        self.data["research"][0]["product_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            self.stages()

    def test_invalid_limits(self):
        for value in [True, 0, 21, 1.5]:
            self.data["limits"]["faq"] = value
            with self.assertRaises(app.ValidationError):
                self.stages()

    def test_forged_citation_rejected(self):
        stages = self.stages()
        stages[2]["records"][0]["citations"][0]["quote"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.validate_stage(stages[2], self.data, stages[1])

    def test_forged_handoff_rejected(self):
        stages = self.stages()
        stages[1]["records"][0]["upstream_ids"] = ["unknown"]
        with self.assertRaises(app.ValidationError):
            app.validate_stage(stages[1], self.data, stages[0])

    def test_unicode_citation_offsets(self):
        self.data["research"][0]["text"] = "  Café hiking backpack.  Waterproof backpack?  "
        stages = self.stages()
        for r in stages[2]["records"]:
            c = r["citations"][0]
            self.assertEqual(c["quote"], self.data["research"][0]["text"][c["start"]:c["end"]])

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "does-not-exist.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_usage_error(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(app.main([]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_bad_json_and_duplicate_keys(self):
        for raw in ["{", '{"query":"a","query":"b"}', '{"x":NaN}', "[]"]:
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch.object(Path, "open", mock_open(read_data=raw)):
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
