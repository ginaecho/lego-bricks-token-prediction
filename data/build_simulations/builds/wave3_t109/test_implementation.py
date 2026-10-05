import copy
import csv
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
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def stage(self, name):
        result = app.normalize(self.raw)
        for stage in (app.research, app.search, app.documents, app.insights):
            if result["stage"] == name:
                break
            result = stage(result)
        return result

    def test_full_pipeline(self):
        result = app.run(self.raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["results"]), list(app.STAGES[1:]))
        app.validate(result, "insights")

    def test_deterministic_and_no_input_mutation(self):
        before = copy.deepcopy(self.raw)
        self.assertEqual(app.run(self.raw), app.run(self.raw))
        self.assertEqual(self.raw, before)

    def test_research_cites_exact_sources(self):
        result = self.stage("research")
        sources = {s["id"]: s["text"] for s in result["data"]["sources"]}
        for evidence in result["results"]["research"]["evidence"]:
            self.assertIn(evidence["excerpt"], sources[evidence["source_id"]])

    def test_insufficient_evidence(self):
        self.raw["sources"] = []
        self.assertEqual(self.stage("research")["results"]["research"]["evidence_status"],
                         "insufficient")

    def test_synonym_search(self):
        self.raw["question"] = "blackout"
        self.raw["sources"] = []
        result = self.stage("search")["results"]["search"]["matches"]
        self.assertEqual(result[0]["product_id"], "recovery-kit")
        self.assertIn("outage", result[0]["matched_terms"])

    def test_search_propagates_evidence(self):
        result = self.stage("search")["results"]
        ids = [e["source_id"] for e in result["research"]["evidence"]]
        self.assertTrue(ids)
        self.assertTrue(all(m["evidence_ids"] == ids for m in result["search"]["matches"]))

    def test_safety_compatibility_filter(self):
        result = self.stage("search")["results"]["search"]["matches"]
        monitor = next(m for m in result if m["product_id"] == "meter-monitor")
        self.assertNotIn("outage-1", monitor["outage_ids"])
        self.assertIn("outage-2", monitor["outage_ids"])

    def test_no_match(self):
        self.raw["products"] = []
        result = app.run(self.raw)["results"]
        self.assertEqual(result["search"]["status"], "no_match")
        self.assertTrue(all(r["recommended_product_id"] is None
                            for r in result["documents"]["outage_rows"]))

    def test_document_meter_total(self):
        result = self.stage("documents")["results"]["documents"]
        self.assertAlmostEqual(result["meter_totals"][0]["value"], 6.4)
        self.assertEqual(result["meter_totals"][0]["unit"], "kWh")
        self.assertEqual(result["meter_totals"][0]["intervals"], 3)

    def test_document_csv_and_recommendations(self):
        result = self.stage("documents")["results"]
        doc = result["documents"]
        parsed = list(csv.DictReader(io.StringIO(doc["outage_csv"])))
        self.assertEqual(parsed[0]["priority"], "P1")
        self.assertEqual(parsed[0]["recommended_product_id"],
                         result["search"]["matches"][0]["product_id"])

    def test_emissions_provenance(self):
        result = self.stage("documents")
        emissions = result["results"]["documents"]["telemetry"][0]["emissions"]
        self.assertEqual(emissions, self.raw["telemetry"]["samples"][0]["emissions"])

    def test_insights_propagate_priority_and_product(self):
        result = app.run(self.raw)["results"]
        safety = next(t for t in result["insights"]["themes"] if t["theme"] == "safety")
        self.assertEqual(safety["priority"], "P1")
        self.assertEqual(safety["product_ids"], ["recovery-kit"])
        self.assertEqual(safety["feedback_ids"], ["feedback-1"])

    def test_empty_feedback(self):
        self.raw["feedback"] = []
        self.assertEqual(app.run(self.raw)["results"]["insights"]["themes"], [])

    def test_empty_optional_collections(self):
        self.raw["sources"] = []
        self.raw["outages"] = []
        self.raw["feedback"] = []
        self.raw["telemetry"]["samples"] = []
        self.raw["meter_csv"] = "timestamp,meter_id,asset_id,kwh\n"
        self.assertEqual(app.run(self.raw)["results"]["documents"]["meter_totals"], [])

    def test_protection_in_references_and_text(self):
        rendered = json.dumps(app.run(self.raw))
        for identity in ("FICT-SUB-ALPHA", "FICT-FEEDER-BETA", "FICT-METER-1"):
            self.assertNotIn(identity, rendered)
        self.assertIn("ASSET-0001", rendered)

    def test_reject_priority_downgrade(self):
        self.raw["outages"][0]["declared_priority"] = "P3"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_priority_rules_all_branches(self):
        for safety, critical, count, expected in [
            (True, False, 0, "P1"), (False, True, 1, "P2"),
            (False, False, 100, "P2"), (False, False, 99, "P3")]:
            self.assertEqual(app.priority({"life_safety": safety, "critical_service": critical,
                                           "customers_affected": count}), expected)

    def test_invalid_emissions(self):
        for value in [
            {"value": 1, "unit": "kgCO2e"},
            {"value": 1, "unit": "tons", "source": "synthetic"},
            {"value": -1, "unit": "kgCO2e", "source": "synthetic"},
            {"value": float("nan"), "unit": "kgCO2e", "source": "synthetic"}]:
            with self.subTest(value=value):
                self.raw["telemetry"]["samples"][0]["emissions"] = value
                with self.assertRaises(app.ValidationError):
                    app.run(self.raw)

    def test_unknown_asset(self):
        self.raw["outages"][0]["asset_id"] = "not-in-topology"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_cycle(self):
        self.raw["telemetry"]["assets"][0]["parent_id"] = "FICT-FEEDER-BETA"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_duplicate_interval(self):
        self.raw["meter_csv"] += self.raw["meter_csv"].splitlines()[1] + "\n"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_equivalent_timezone_duplicates(self):
        self.raw["meter_csv"] += self.raw["meter_csv"].splitlines()[1].replace(
            "18:00:00Z", "19:00:00+01:00") + "\n"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_bad_csv(self):
        for value in ["bad,header\n", "timestamp,meter_id,asset_id,kwh\nx,y,z\n",
                      self.raw["meter_csv"].replace(",2.4", ",NaN"),
                      self.raw["meter_csv"].replace(",2.4", ",-2")]:
            with self.subTest(value=value):
                self.raw["meter_csv"] = value
                with self.assertRaises(app.ValidationError):
                    app.run(self.raw)

    def test_timestamp_timezone_required(self):
        self.raw["telemetry"]["samples"][0]["timestamp"] = "2026-01-01T00:00:00"
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_boolean_number_rejected(self):
        self.raw["outages"][0]["customers_affected"] = True
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_synthetic_label_required(self):
        self.raw["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.raw)

    def test_bad_citation_blocks_search(self):
        state = self.stage("research")
        state["results"]["research"]["evidence"][0]["source_id"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.search(state)

    def test_bad_match_blocks_documents(self):
        state = self.stage("search")
        state["results"]["search"]["matches"][0]["evidence_ids"] = []
        with self.assertRaises(app.ValidationError):
            app.documents(state)

    def test_tampered_document_blocks_insights(self):
        state = self.stage("documents")
        state["results"]["documents"]["outage_rows"][0]["priority"] = "P3"
        with self.assertRaises(app.ValidationError):
            app.insights(state)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.documents(app.normalize(self.raw))

    def test_csv_formula_defense(self):
        self.raw["outages"][0]["id"] = "=DANGEROUS()"
        self.raw["feedback"][0]["outage_id"] = "=DANGEROUS()"
        doc = self.stage("documents")["results"]["documents"]
        row = list(csv.DictReader(io.StringIO(doc["outage_csv"])))[0]
        self.assertEqual(row["outage_id"], "'=DANGEROUS()")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "missing-input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertNotIn(str(ROOT), result.stdout)

    def test_cli_usage(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(app.main([]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for content in ("{", "null", "[]", '{"schema_version":"invalid"}'):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)), \
                    patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(app.main(["virtual.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
