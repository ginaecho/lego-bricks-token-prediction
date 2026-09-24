"""All fixtures are synthetic; tests create no files and never call networks."""

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
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def assert_invalid(self, raw=None):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.raw if raw is None else raw)

    def test_end_to_end_and_shared_schema(self):
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stage"], "research")
        self.assertIs(app.validate(result, "research"), result)
        self.assertTrue(result["synthetic"])

    def test_search_synonym_not_literal(self):
        self.raw["query"] = "blackout"
        matches = app.search(self.raw)["search"]["matches"]
        self.assertEqual(matches[0]["id"], "product-alerts")

    def test_search_typo(self):
        self.raw["query"] = "consumptoin"
        self.assertEqual(app.search(self.raw)["search"]["matches"][0]["id"], "product-insights")

    def test_search_general_purpose_catalog(self):
        self.raw["catalog"] = [
            {"id": "mobile", "name": "Affordable smartphone", "description": "A mobile device",
             "tags": ["phone"], "source_id": "source-plans"},
            {"id": "book", "name": "Hardcover novel", "description": "Fiction reading",
             "tags": ["literature"], "source_id": "source-plans"},
        ]
        self.raw["query"] = "cheap phone"
        self.assertEqual(app.search(self.raw)["search"]["matches"][0]["id"], "mobile")

    def test_search_no_match_and_research_uncertainty(self):
        self.raw["query"] = "zyxwvu"
        self.raw["support_question"] = "zyxwvu"
        self.raw["research_question"] = "zyxwvu"
        result = app.run_pipeline(self.raw)
        self.assertTrue(result["search"]["no_match"])
        self.assertEqual(result["support"]["matched_product_ids"], [])
        self.assertEqual(result["research"]["answerability"], "insufficient-evidence")
        self.assertTrue(any("No catalog match" in item for item in result["support"]["limitations"]))

    def test_empty_catalog_is_valid(self):
        self.raw["catalog"] = []
        self.assertTrue(app.run_pipeline(self.raw)["search"]["no_match"])

    def test_search_limit_and_stable_tie_break(self):
        self.raw["catalog"] = [
            {"id": key, "name": "Energy plan", "description": "Energy", "tags": [],
             "source_id": "source-plans"} for key in ["z", "a", "b"]
        ]
        self.raw["query"] = "energy"
        self.raw["top_k"] = 2
        self.assertEqual([p["id"] for p in app.search(self.raw)["search"]["matches"]], ["a", "b"])

    def test_support_grounded_claim_citations(self):
        result = app.support(app.search(self.raw))
        sources = {s["id"] for s in result["context"]["sources"]}
        self.assertTrue(result["support"]["claims"])
        for claim in result["support"]["claims"]:
            self.assertTrue(set(claim["citations"]) <= sources)
            self.assertIn(claim["text"], result["support"]["answer"])
        self.assertIn("not guaranteed", result["support"]["answer"])

    def test_meter_aggregation_has_units_and_sampling_caveat(self):
        result = app.run_pipeline(self.raw)
        meter = next(c for c in result["support"]["claims"] if c["kind"] == "meter")
        self.assertIn("5.600 kWh", meter["text"])
        self.assertIn("not a full billing-period", meter["text"])

    def test_cross_stage_product_and_evidence_propagation(self):
        state1 = app.search(self.raw)
        state2 = app.support(state1)
        state3 = app.research(state2)
        identifiers = [p["id"] for p in state1["search"]["matches"]]
        self.assertEqual(state2["support"]["matched_product_ids"], identifiers)
        self.assertEqual(state3["research"]["recommended_product_ids"], identifiers)
        self.assertEqual(state2["support"]["consumes"], "validated:search")
        self.assertEqual(state3["research"]["consumes"], "validated:support")
        claims = {claim["id"]: claim for claim in state2["support"]["claims"]}
        for finding in state3["research"]["findings"]:
            origin = claims[finding["claim_id"]]
            self.assertEqual(finding["finding"], origin["text"])
            self.assertEqual(finding["citations"], origin["citations"])

    def test_pipeline_deterministic_and_does_not_mutate_inputs(self):
        original = copy.deepcopy(self.raw)
        one = app.run_pipeline(self.raw)
        self.assertEqual(one, app.run_pipeline(self.raw))
        self.assertEqual(original, self.raw)
        search = app.search(self.raw)
        original_search = copy.deepcopy(search)
        app.support(search)
        self.assertEqual(search, original_search)

    def test_tampered_search_is_rejected_before_support(self):
        state = app.search(self.raw)
        state["search"]["matches"][0]["description"] = "Guaranteed free power"
        with self.assertRaises(app.ValidationError):
            app.support(state)

    def test_tampered_support_is_rejected_before_research(self):
        state = app.support(app.search(self.raw))
        state["support"]["claims"][0]["citations"] = ["unknown"]
        with self.assertRaises(app.ValidationError):
            app.research(state)

    def test_tampered_industry_context_is_rejected(self):
        state = app.search(self.raw)
        state["context"]["outage_reports"][0]["priority"] = "P3"
        with self.assertRaises(app.ValidationError):
            app.support(state)

    def test_wrong_stage_is_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.research(app.search(self.raw))

    def test_unknown_envelope_fields_are_rejected(self):
        state = app.search(self.raw)
        state["raw_asset"] = "private"
        with self.assertRaises(app.ValidationError):
            app.support(state)

    def test_critical_identifiers_are_withheld_everywhere(self):
        self.raw["support_question"] += " at SYN-DEVICE-FEEDER-01 and SYN-METER-HOUSE-01"
        result = app.run_pipeline(self.raw)
        serialized = json.dumps(result)
        for identifier in ["SYN-DEVICE-FEEDER-01", "SYN-DEVICE-TRANSFORMER-02",
                           "SYN-SUB-ALDER-01", "SYN-METER-HOUSE-01"]:
            self.assertNotIn(identifier, serialized)
        self.assertEqual(result["context"]["topology"], [["asset-001", "asset-002"]])
        self.assertIn("station-001", serialized)
        self.assertIn("Not CIP certification", serialized)

    def test_safety_overrides_small_customer_count(self):
        result = app.run_pipeline(self.raw)
        self.assertTrue(result["support"]["human_escalation_required"])
        self.assertIn("Stay clear of downed lines", result["support"]["answer"])
        self.assertIn("before commercial recommendations", result["research"]["decision"])

    def test_short_raw_meter_id_does_not_corrupt_alias(self):
        self.raw["meter_interval_csv"] = self.raw["meter_interval_csv"].replace("SYN-METER-HOUSE-01", "m")
        result = app.run_pipeline(self.raw)
        self.assertEqual(result["context"]["meter_readings"][0]["meter_ref"], "meter-001")

    def test_priority_rule_cases(self):
        for flags, customers, expected in [
            ((True, False, False), 1, "P1"),
            ((False, True, True), 1, "P1"),
            ((False, False, True), 1, "P2"),
            ((False, False, False), 1000, "P2"),
            ((False, False, False), 999, "P3"),
        ]:
            with self.subTest(flags=flags, customers=customers):
                raw = copy.deepcopy(self.raw)
                outage = raw["outage_reports"][0]
                for key, value in zip(("life_safety", "downed_line", "critical_service"), flags):
                    outage[key] = value
                outage["customers_affected"] = customers
                outage["priority"] = expected
                self.assertEqual(app.run_pipeline(raw)["context"]["outage_reports"][0]["priority"], expected)

    def test_declared_threshold_applies(self):
        self.raw["safety_rules"]["widespread_customer_threshold"] = 20
        outage = self.raw["outage_reports"][0]
        outage["downed_line"] = False
        outage["priority"] = "P2"
        self.assertEqual(app.run_pipeline(self.raw)["context"]["outage_reports"][0]["priority"], "P2")

    def test_unsafe_outage_priority_rejected(self):
        self.raw["outage_reports"][0]["priority"] = "P3"
        self.assert_invalid()

    def test_unsafe_declared_safety_rules_rejected(self):
        self.raw["safety_rules"]["life_safety"] = "P3"
        self.assert_invalid()

    def test_emissions_units_source_and_value_propagate(self):
        result = app.run_pipeline(self.raw)
        claim = next(c for c in result["support"]["claims"] if c["kind"] == "emissions")
        self.assertIn("42.5 kgCO2e", claim["text"])
        self.assertEqual(claim["citations"], ["source-emissions"])
        self.assertIn("source-emissions", [s["id"] for s in result["research"]["citations"]])

    def test_missing_emissions_unit_or_source_rejected(self):
        for key in ("unit", "source_id"):
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw)
                del raw["emissions"][0][key]
                self.assert_invalid(raw)

    def test_latest_telemetry_clears_stale_warning(self):
        self.raw["telemetry"]["samples"].append({
            "asset_id": "SYN-DEVICE-FEEDER-01", "timestamp": "2026-01-15T18:00:00Z",
            "voltage_kv": 11, "load_pct": 75, "status": "normal"})
        result = app.run_pipeline(self.raw)
        self.assertFalse(any(c["kind"] == "telemetry" for c in result["support"]["claims"]))

    def test_unknown_asset_in_telemetry_topology_outage(self):
        for location in ("telemetry", "topology", "outage"):
            with self.subTest(location=location):
                raw = copy.deepcopy(self.raw)
                if location == "telemetry":
                    raw["telemetry"]["samples"][0]["asset_id"] = "SYN-UNKNOWN"
                elif location == "topology":
                    raw["topology"][0][1] = "SYN-UNKNOWN"
                else:
                    raw["outage_reports"][0]["asset_id"] = "SYN-UNKNOWN"
                self.assert_invalid(raw)

    def test_negative_nonfinite_and_boolean_numbers_rejected(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw)
                raw["emissions"][0]["value"] = value
                self.assert_invalid(raw)

    def test_csv_invalid_cases(self):
        for value in [
            "meter_id,timestamp,watts\nm,2026-01-01T00:00:00Z,1",
            "meter_id,timestamp,kwh\nm,2026-01-01T00:00:00Z,-1",
            "meter_id,timestamp,kwh\nm,2026-01-01T00:00:00Z,nan",
            "meter_id,timestamp,kwh\nm,2026-01-01T00:00:00,1",
            "meter_id,timestamp,kwh\nm,2026-01-01T00:00:00Z,1,extra",
            "meter_id,timestamp,kwh\nm,2026-01-01T00:00:00Z",
            "meter_id,timestamp,kwh\n",
        ]:
            with self.subTest(csv=value):
                raw = copy.deepcopy(self.raw)
                raw["meter_interval_csv"] = value
                self.assert_invalid(raw)

    def test_duplicate_csv_interval_equivalent_timezone_rejected(self):
        self.raw["meter_interval_csv"] += "SYN-METER-HOUSE-01,2026-01-15T18:00:00+01:00,1.1\n"
        self.assert_invalid()

    def test_duplicate_entity_ids_rejected(self):
        self.raw["grid_assets"].append(copy.deepcopy(self.raw["grid_assets"][0]))
        self.assert_invalid()

    def test_unknown_and_ungrounded_knowledge_source_rejected(self):
        for key, value in [("source_id", "missing"), ("text", "Guaranteed zero bill")]:
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw)
                raw["knowledge"][0][key] = value
                self.assert_invalid(raw)

    def test_no_outages_no_knowledge_escalates_without_inventing_outage(self):
        self.raw["outage_reports"] = []
        self.raw["knowledge"] = []
        result = app.run_pipeline(self.raw)
        self.assertIn("human support representative", result["research"]["decision"])
        self.assertFalse(any(c["kind"] == "outage" for c in result["support"]["claims"]))

    def test_required_and_fixture_fields(self):
        for key, value in [("schema_version", "2"), ("synthetic", False), ("query", " "),
                           ("top_k", 0), ("sources", None), ("outage_reports", "bad")]:
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw)
                raw[key] = value
                self.assert_invalid(raw)

    def test_cli_success_single_json(self):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                    str(ROOT / "example_input.json")],
                                   cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(json.loads(completed.stdout)["stage"], "research")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "nonexistent-input.json")]):
            with self.subTest(args=args):
                completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                           cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(json.loads(completed.stdout)["status"], "error")
                self.assertEqual(completed.stderr, "")
                self.assertNotIn(str(ROOT), completed.stdout)

    def test_cli_malformed_json_duplicate_keys_nonfinite_and_bad_schema(self):
        for content in ["{", '{"a":1,"a":2}', '{"x":NaN}', "[]",
                        json.dumps({**self.raw, "synthetic": False})]:
            with self.subTest(content=content[:30]):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=content), contextlib.redirect_stdout(output):
                    code = app.main(["ignored-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
