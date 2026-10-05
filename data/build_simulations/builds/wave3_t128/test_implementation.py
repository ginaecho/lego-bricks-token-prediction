"""Synthetic fixtures only; tests never create files or call external services."""

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


def fixture():
    return json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))


def initial(data):
    return {"status": "ok", "schema_version": 1, "stage": "input",
            "request": app.validate_request(data)}


class PipelineTests(unittest.TestCase):
    def test_complete_pipeline(self):
        result = app.run_pipeline(fixture())
        self.assertEqual(result["stage"], "research")
        self.assertEqual(result["research"]["decision"]["product_id"], "p1")
        self.assertEqual(result["research"]["decision"]["status"], "evidence_available")
        self.assertIs(app.validate_state(result, "research"), result)

    def test_personalized_onboarding(self):
        result = app.onboard(initial(fixture()))["onboarding"]
        self.assertIn("Alex", result["greeting"])
        self.assertIn("within your budget", result["next_step"])
        self.assertEqual(result["missing_fields"], [])

    def test_cold_start(self):
        data = fixture()
        data["customer"] = {"id": "new"}
        data["query"] = ""
        result = app.run_pipeline(data)
        self.assertIn("query_or_interests", result["onboarding"]["missing_fields"])
        self.assertIn("budget", result["onboarding"]["missing_fields"])
        self.assertEqual(len(result["search"]["items"]), 4)

    def test_interests_fallback_propagates(self):
        data = fixture()
        data["query"] = "  "
        result = app.run_pipeline(data)
        self.assertEqual(result["onboarding"]["effective_query"], "travel quiet")
        self.assertEqual(result["search"]["query"], "travel quiet")

    def test_synonyms_and_typo(self):
        result = app.run_pipeline(fixture())["search"]
        self.assertIn("bluetooth", result["query_tokens"])
        self.assertIn("travel", result["query_tokens"])
        self.assertEqual(result["corrections"]["headphons"], "headphone")
        self.assertEqual(result["items"][0]["product_id"], "p1")

    def test_budget_filters_all_handoffs(self):
        result = app.run_pipeline(fixture())
        self.assertEqual(result["search"]["excluded_over_budget"], 1)
        for stage in ("search", "recommendations"):
            self.assertNotIn("p4", [p["product_id"] for p in result[stage]["items"]])
        self.assertNotIn("p4", result["research"]["coverage"]["with_evidence"])

    def test_zero_budget(self):
        data = fixture()
        data["customer"]["budget"] = 0
        data["catalog"][0]["price"] = 0
        result = app.run_pipeline(data)
        self.assertEqual([i["product_id"] for i in result["recommendations"]["items"]], ["p1"])

    def test_explicit_preferences_rank(self):
        data = fixture()
        data["query"] = "bluetooth"
        data["customer"]["interests"] = ["budget"]
        result = app.run_pipeline(data)
        self.assertEqual(result["recommendations"]["items"][0]["product_id"], "p2")
        self.assertTrue(any("budget" in reason for reason in
                            result["recommendations"]["items"][0]["reasons"]))

    def test_recommendations_are_search_subset(self):
        data = fixture()
        data["limits"]["search"] = 1
        result = app.run_pipeline(data)
        self.assertEqual(len(result["recommendations"]["items"]), 1)
        self.assertEqual(result["research"]["coverage"]["with_evidence"], ["p1"])

    def test_research_quotes_are_exact(self):
        result = app.run_pipeline(fixture())
        sources = {s["id"]: s for s in result["request"]["research"]["sources"]}
        for evidence in result["research"]["evidence"]:
            self.assertIn(evidence["quote"], sources[evidence["source_id"]]["text"])
        self.assertEqual(result["research"]["coverage"]["without_evidence"], ["p3"])

    def test_missing_sources(self):
        data = fixture()
        data["research"]["sources"] = []
        result = app.run_pipeline(data)["research"]
        self.assertEqual(result["decision"]["status"], "insufficient_evidence")
        self.assertEqual(result["decision"]["source_ids"], [])

    def test_irrelevant_question_not_claimed_supported(self):
        data = fixture()
        data["research"]["question"] = "Waterproof certification?"
        result = app.run_pipeline(data)["research"]
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["decision"]["status"], "insufficient_evidence")

    def test_no_matches(self):
        data = fixture()
        data["query"] = "telescope"
        result = app.run_pipeline(data)
        self.assertEqual(result["search"]["items"], [])
        self.assertEqual(result["research"]["decision"]["status"], "no_candidates")
        self.assertIsNone(result["research"]["decision"]["product_id"])

    def test_empty_catalog(self):
        data = fixture()
        data["catalog"] = []
        data["research"]["sources"] = []
        self.assertEqual(app.run_pipeline(data)["recommendations"]["items"], [])

    def test_stable_nonmutating_run(self):
        data = fixture()
        before = copy.deepcopy(data)
        self.assertEqual(app.run_pipeline(data), app.run_pipeline(data))
        self.assertEqual(data, before)
        state = app.onboard(initial(data))
        original = copy.deepcopy(state)
        app.search(state)
        self.assertEqual(state, original)

    def test_limits(self):
        data = fixture()
        data["limits"] = {"search": 2, "recommendations": 1, "evidence": 1}
        result = app.run_pipeline(data)
        self.assertLessEqual(len(result["search"]["items"]), 2)
        self.assertEqual(len(result["recommendations"]["items"]), 1)
        self.assertEqual(len(result["research"]["evidence"]), 1)

    def test_invalid_number_variants(self):
        for value in (-1, True, "12", float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(value=str(value)[:30]):
                data = fixture()
                data["customer"]["budget"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_product(self):
        data = fixture()
        data["catalog"].append(copy.deepcopy(data["catalog"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)

    def test_unknown_source_reference(self):
        data = fixture()
        data["research"]["sources"][0]["product_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)

    def test_currency_mismatch(self):
        data = fixture()
        data["catalog"][1]["currency"] = "EUR"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)

    def test_invalid_input_shapes(self):
        variants = [None, [], {}, {"data_label": "synthetic"}]
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(app.ValidationError):
                app.run_pipeline(raw)

    def test_unknown_field_and_invalid_limit(self):
        data = fixture()
        data["unexpected"] = 1
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)
        for value in (0, 51, True, 1.5):
            data = fixture()
            data["limits"]["search"] = value
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.recommend(app.onboard(initial(fixture())))

    def test_tampered_handoff_rejected(self):
        state = app.search(app.onboard(initial(fixture())))
        state["search"]["items"][0]["product_id"] = "p4"
        with self.assertRaises(app.ValidationError):
            app.recommend(state)
        state = app.onboard(initial(fixture()))
        state["onboarding"]["customer_id"] = "other"
        with self.assertRaises(app.ValidationError):
            app.search(state)

    def test_forged_evidence_rejected(self):
        result = app.run_pipeline(fixture())
        result["research"]["evidence"][0]["quote"] = "Invented claim."
        with self.assertRaises(app.ValidationError):
            app.validate_state(result, "research")

    def test_tampered_onboarding_query_rejected(self):
        state = app.onboard(initial(fixture()))
        state["onboarding"]["effective_query"] = "unrelated"
        with self.assertRaises(app.ValidationError):
            app.search(state)

    def test_tampered_decision_citations_rejected(self):
        result = app.run_pipeline(fixture())
        result["research"]["decision"]["source_ids"] = ["unknown"]
        with self.assertRaises(app.ValidationError):
            app.validate_state(result, "research")

    def test_invalid_reliability(self):
        for value in (-0.1, 1.1, True, None, "high"):
            data = fixture()
            data["research"]["sources"][0]["reliability"] = value
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_conflicting_sources_preserved(self):
        data = fixture()
        data["research"]["sources"].append({
            "id": "synthetic-conflict", "title": "Synthetic conflicting report",
            "text": "Battery life was only 2 hours.", "product_ids": ["p1"],
            "reliability": 0.7,
        })
        result = app.run_pipeline(data)["research"]
        self.assertIn("synthetic-conflict", result["decision"]["source_ids"])
        self.assertTrue(any("Conflicting" in item for item in result["limitations"]))

    def test_unicode_normalization(self):
        self.assertEqual(app.tokens("CAFÉ wireless"), {"cafe", "bluetooth"})

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "nonexistent.json")],
                                 capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_usage(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for payload in ("{", "[]", '{"a": 1, "a": 2}', '{"value": NaN}'):
            output = io.StringIO()
            with patch("builtins.open", return_value=io.StringIO(payload)), \
                    patch("sys.stdout", output):
                code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_duplicate_source_ids(self):
        data = fixture()
        data["research"]["sources"].append(copy.deepcopy(data["research"]["sources"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)


if __name__ == "__main__":
    unittest.main()
