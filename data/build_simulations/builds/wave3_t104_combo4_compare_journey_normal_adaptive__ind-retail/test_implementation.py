import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_stage(self, name):
        return next(r["data"] for r in app.run_pipeline(self.value)["stages"] if r["stage"] == name)

    def test_pipeline_order(self):
        result = app.run_pipeline(self.value)
        self.assertEqual([r["stage"] for r in result["stages"]], list(app.STAGES))

    def test_normalization(self):
        rows = self.run_stage("compare")["comparison"]
        self.assertEqual(rows[0]["attributes"]["weight_g"], 650)
        self.assertEqual(rows[1]["attributes"]["capacity_ml"], 18000)

    def test_preference_ranking(self):
        self.value["customer"]["preferences"]["waterproof_required"] = False
        self.value["research_query"] = "pocket"
        self.assertEqual(self.run_stage("compare")["selected_sku"], "FICTION-PACK-202")

    def test_exclusion_reasons(self):
        rows = self.run_stage("compare")["comparison"]
        self.assertIn("not_waterproof", rows[1]["exclusions"])
        self.assertIn("no_unallocated_stock", rows[2]["exclusions"])
        self.assertIn("over_budget", rows[2]["exclusions"])

    def test_weight_priority(self):
        self.value["customer"]["preferences"].update(waterproof_required=False, priority="weight")
        self.assertEqual(self.run_stage("compare")["selected_sku"], "FICTION-PACK-101")

    def test_capacity_priority(self):
        self.value["customer"]["preferences"].update(waterproof_required=False, priority="capacity")
        self.assertEqual(self.run_stage("compare")["ranking"][0], "FICTION-PACK-101")

    def test_clickstream_tie_break(self):
        self.value["customer"]["preferences"]["waterproof_required"] = False
        self.value["research_query"] = "pocket"
        self.value["catalog"]["products"][1]["price_cents"] = 4900
        self.value["clickstream"] = [{"sequence": 1, "event": "view", "sku": "FICTION-PACK-202"}]
        self.assertEqual(self.run_stage("compare")["ranking"][0], "FICTION-PACK-202")
        self.value["clickstream"] = []
        self.assertEqual(self.run_stage("compare")["ranking"][0], "FICTION-PACK-101")

    def test_two_step_journey(self):
        journey = self.run_stage("journey")
        self.assertEqual(len(journey["actions"]), 2)
        self.assertEqual(journey["actions"][1]["requires"], journey["actions"][0]["provides"])
        self.assertTrue(all(a["status"] == "planned" for a in journey["actions"]))

    def test_prerequisite_failure(self):
        with self.assertRaises(app.ValidationError):
            app.plan([], [("bad", ["missing"], [], "invalid")])

    def test_exact_citations(self):
        research = self.run_stage("normal")
        passages = {p["source_id"]: p["text"] for p in self.value["catalog"]["products"][0]["passages"]}
        self.assertEqual(len(research["findings"]), 2)
        for finding in research["findings"]:
            cite = finding["citation"]
            self.assertEqual(finding["quote"], passages[cite["source_id"]][cite["start"]:cite["end"]])
            self.assertEqual(cite["sku"], research["sku"])

    def test_no_evidence_fails_closed(self):
        self.value["research_query"] = "unfindable"
        with self.assertRaisesRegex(app.ValidationError, "No matching"):
            app.run_pipeline(self.value)

    def test_adaptive_experience(self):
        self.assertEqual(self.run_stage("adaptive")["steps"][0]["action"], "guided_evidence_walkthrough")
        self.value["customer"]["experience"] = "experienced"
        self.assertEqual(self.run_stage("adaptive")["steps"][0]["action"], "quick_evidence_check")

    def test_adaptive_preference_explanation(self):
        self.value["customer"]["preferences"]["priority"] = "weight"
        self.assertIn("weight", self.run_stage("adaptive")["steps"][0]["explanation"])

    def test_cross_stage_handoffs(self):
        stages = app.run_pipeline(self.value)["stages"]
        self.assertEqual(stages[0]["input_digest"], app.digest(self.value))
        for previous, current in zip(stages, stages[1:]):
            self.assertEqual(current["input_digest"], app.digest(previous))
        self.assertEqual(stages[0]["data"]["selected_sku"], stages[1]["data"]["sku"])
        self.assertEqual(stages[1]["data"]["research_request"]["query"], stages[2]["data"]["query"])
        self.assertEqual(stages[2]["data"]["findings"], stages[3]["data"]["evidence"])
        self.assertEqual(stages[1]["data"]["actions"], stages[3]["data"]["journey_actions"])

    def test_changed_selection_propagates(self):
        self.value["customer"]["preferences"]["waterproof_required"] = False
        self.value["research_query"] = "pocket"
        stages = app.run_pipeline(self.value)["stages"]
        self.assertTrue(all(r["data"]["sku"] == "FICTION-PACK-202" for r in stages[1:]))
        self.assertEqual(stages[-1]["data"]["basket_preview"]["total_cents"], 7800)

    def test_offer_and_basket_match_catalog(self):
        final = self.run_stage("adaptive")
        self.assertEqual(final["offer"], {"price_cents": 4900, "stock": 7})
        preview = final["basket_preview"]
        self.assertEqual(preview["total_cents"], 8800)
        self.assertFalse(preview["order_placed"])
        catalog = {p["sku"]: p for p in self.value["catalog"]["products"]}
        for line in preview["lines"]:
            self.assertEqual(line["unit_price_cents"], catalog[line["sku"]]["price_cents"])
            self.assertEqual(line["stock"], catalog[line["sku"]]["stock"])

    def test_consent_required_before_personalization(self):
        for consent in (False, None, 1, "yes"):
            with self.subTest(consent=consent):
                self.value["customer"]["consent"]["personalization"] = consent
                with patch.dict(app.BUILDERS, {"compare": lambda *_: self.fail("Personalization ran")}):
                    with self.assertRaises(app.ValidationError):
                        app.run_pipeline(self.value)

    def test_no_reviews_or_endorsements(self):
        self.value["catalog"]["products"][0]["reviews"] = ["invented praise"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)
        del self.value["catalog"]["products"][0]["reviews"]
        self.value["catalog"]["products"][0]["passages"][0]["kind"] = "review"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)

    def test_review_text_rejected(self):
        self.value["catalog"]["products"][0]["passages"][0]["text"] = "Five stars from invented shoppers"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)

    def test_no_eligible_product(self):
        self.value["customer"]["preferences"]["budget_cents"] = 0
        with self.assertRaisesRegex(app.ValidationError, "No eligible"):
            app.run_pipeline(self.value)

    def test_basket_stock_accounted_for(self):
        self.value["basket"]["lines"] = [{"sku": "FICTION-PACK-101", "quantity": 7}]
        with self.assertRaisesRegex(app.ValidationError, "No eligible"):
            app.run_pipeline(self.value)

    def test_invalid_basket(self):
        for line in ({"sku": "unknown", "quantity": 1},
                     {"sku": "FICTION-PACK-101", "quantity": 8},
                     {"sku": "FICTION-PACK-101", "quantity": True}):
            with self.subTest(line=line):
                self.value["basket"]["lines"] = [line]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.value)

    def test_duplicate_sku(self):
        self.value["catalog"]["products"][1]["sku"] = "FICTION-PACK-101"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)

    def test_invalid_units_and_values(self):
        weight = self.value["catalog"]["products"][0]["attributes"]["weight"]
        for invalid in (float("nan"), float("inf"), True, -1, "light"):
            weight["value"] = invalid
            with self.subTest(value=invalid), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.value)
        weight.update(value=2, unit="stone")
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)

    def test_invalid_prices_and_stock(self):
        product = self.value["catalog"]["products"][0]
        for field in ("price_cents", "stock"):
            original = product[field]
            for invalid in (-1, 1.5, True, None):
                product[field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.value)
            product[field] = original

    def test_clickstream_order(self):
        self.value["clickstream"][1]["sequence"] = 1
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)

    def test_tampered_stage_rejected(self):
        records = app.run_pipeline(self.value)["stages"]
        for index, stage in enumerate(app.STAGES):
            broken = copy.deepcopy(records[index])
            broken["data"]["unexpected"] = "fabricated"
            with self.subTest(stage=stage), self.assertRaises(app.ValidationError):
                app.Validation.stage(stage, broken, self.value, records[index - 1] if index else None)

    def test_broken_handoff_rejected(self):
        stages = app.run_pipeline(self.value)["stages"]
        stages[1]["input_digest"] = "wrong"
        with self.assertRaises(app.ValidationError):
            app.Validation.stage("journey", stages[1], self.value, stages[0])

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.value)
        self.assertEqual(app.run_pipeline(self.value), app.run_pipeline(self.value))
        self.assertEqual(self.value, original)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(len(run.stdout.splitlines()), 1)
        self.assertEqual(run.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_malformed_duplicate_and_invalid_json(self):
        for content in ("{", '{"x": 1, "x": 2}', "null", "[]",
                        json.dumps(dict(self.value, synthetic=False))):
            with self.subTest(content=content[:30]), patch("builtins.open", mock_open(read_data=content)):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_synthetic_marker_required(self):
        self.value["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.value)


if __name__ == "__main__":
    unittest.main()
