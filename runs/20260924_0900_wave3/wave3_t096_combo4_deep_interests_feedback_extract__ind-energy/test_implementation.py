"""Run with python -B -m unittest discover -s . -p test_implementation.py."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class EnergyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_pipeline(self):
        return app.pipeline(self.source)

    def initial(self):
        return {"schema_version": app.VERSION, "status": "ok", "stage": "input",
                "data": {"source": app.protect(self.source)}}

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_full_pipeline_and_shared_schema(self):
        result = self.run_pipeline()
        self.assertEqual(result["stage"], "extract")
        self.assertIs(app.validate(result, "extract"), result)
        self.assertEqual(set(result["data"]),
                         {"source", "research", "recommendations", "feedback_analysis", "extraction"})

    def test_multidocument_disagreement(self):
        research = self.run_pipeline()["data"]["research"]
        self.assertEqual(len(research["disagreements"]), 1)
        conflict = research["disagreements"][0]
        self.assertEqual(conflict["topic"], "reliability")
        evidence = {e["id"]: e for e in research["evidence"]}
        self.assertEqual({evidence[i]["document_id"] for i in conflict["evidence_ids"]},
                         {"doc-a", "doc-b"})

    def test_unresolved_preference_topic(self):
        questions = self.run_pipeline()["data"]["research"]["unresolved_questions"]
        self.assertTrue(any("flood_resilience" in q for q in questions))

    def test_evidence_source_spans(self):
        data = self.run_pipeline()["data"]
        docs = {d["id"]: d["text"] for d in data["source"]["documents"]}
        for evidence in data["research"]["evidence"]:
            s = evidence["span"]
            self.assertEqual(docs[evidence["document_id"]][s["start"]:s["end"]], s["quote"])

    def test_weighted_ranking(self):
        ranked = self.run_pipeline()["data"]["recommendations"]
        self.assertEqual([r["score"] for r in ranked], [8, 2])
        self.assertIn("reliability", ranked[0]["explanation"])
        self.assertEqual(len(ranked[0]["uncertainty"]), 1)

    def test_exclusions_propagate_to_feedback_and_extraction(self):
        self.source["preferences"]["exclude_assets"].append("FICT-SUB-NORTH-01")
        data = self.run_pipeline()["data"]
        self.assertEqual(len(data["recommendations"]), 1)
        self.assertEqual(len(data["feedback_analysis"]["records"]), 1)
        self.assertEqual(data["feedback_analysis"]["records"][0]["outage_id"], "OUT-SYN-2")
        self.assertEqual(len(data["extraction"]), 1)
        self.assertEqual(data["feedback_analysis"]["excluded_feedback_count"], 3)

    def test_research_changes_ranking_downstream(self):
        self.source["documents"] = [
            d for d in self.source["documents"] if d["asset_id"] != "FICT-SUB-NORTH-01"]
        data = self.run_pipeline()["data"]
        self.assertEqual([r["score"] for r in data["recommendations"]], [2])
        self.assertEqual(len(data["extraction"]), 1)
        self.assertTrue(any("documentary" in q for q in data["research"]["unresolved_questions"]))

    def test_deduplication_preserves_provenance(self):
        analysis = self.run_pipeline()["data"]["feedback_analysis"]
        self.assertEqual(analysis["duplicates_removed"], 1)
        self.assertEqual(len(analysis["records"]), 2)
        self.assertEqual([s["feedback_id"] for s in analysis["records"][0]["supports"]],
                         ["response-a", "response-b"])

    def test_deduplication_whitespace(self):
        self.source["feedback"][1]["text"] = "  " + self.source["feedback"][0]["text"].replace(
            "winter", "winter   ") + "  "
        analysis = self.run_pipeline()["data"]["feedback_analysis"]
        self.assertEqual(analysis["duplicates_removed"], 1)

    def test_themes_count_unique_feedback_and_trace_excerpts(self):
        data = self.run_pipeline()["data"]
        theme = next(t for t in data["feedback_analysis"]["themes"] if t["name"] == "safety")
        self.assertEqual(theme["count"], 1)
        support = theme["supporting_excerpts"][0]
        text = next(f["text"] for f in data["source"]["feedback"]
                    if f["id"] == support["feedback_id"])
        app.verify_span(text, support["span"])
        self.assertIn("live wire", support["span"]["quote"])

    def test_schema_extraction_and_missing_fields(self):
        extracted = self.run_pipeline()["data"]["extraction"]
        self.assertEqual(extracted[0]["fields"]["consumption"]["value"], 3.7)
        self.assertEqual(extracted[0]["fields"]["emissions"]["unit"], "kgCO2e")
        self.assertEqual(extracted[0]["fields"]["emissions"]["source"], "response-a")
        self.assertEqual(extracted[1]["missing_fields"], ["consumption", "emissions"])
        self.assertEqual(extracted[1]["missing_required"], ["emissions"])

    def test_extraction_spans_reference_protected_source(self):
        data = self.run_pipeline()["data"]
        feedback = {f["id"]: f["text"] for f in data["source"]["feedback"]}
        for result in data["extraction"]:
            for value in result["fields"].values():
                if value is not None:
                    app.verify_span(feedback[value["source"]], value["span"])

    def test_critical_asset_device_and_meter_identifiers_protected(self):
        text = json.dumps(self.run_pipeline())
        identifiers = [a["id"] for a in self.source["assets"]] + [
            d["device_id"] for d in self.source["telemetry"]["devices"]] + ["FICT-METER-01"]
        for identifier in identifiers:
            self.assertNotIn(identifier, text)
        self.assertIn("protected-", text)

    def test_safety_priority_enforced(self):
        self.source["outages"][0]["priority"] = "routine"
        self.invalid()

    def test_safety_rule_tampering_rejected(self):
        self.source["safety_rules"]["live_wire"] = "routine"
        self.invalid()

    def test_unknown_safety_flag_rejected(self):
        self.source["outages"][0]["safety_flags"] = ["unrecognized"]
        self.invalid()

    def test_extracted_priority_enforced(self):
        self.source["feedback"][0]["text"] = self.source["feedback"][0]["text"].replace(
            "Priority: critical", "Priority: routine")
        self.invalid()

    def test_emissions_source_required(self):
        del self.source["documents"][0]["claims"][1]["source"]
        self.invalid()

    def test_emissions_units_required(self):
        self.source["documents"][0]["claims"][1]["unit"] = "tons"
        self.invalid()

    def test_emissions_unit_conversion_avoids_false_conflict(self):
        document = copy.deepcopy(self.source["documents"][0])
        document["id"] = "doc-e"
        document["text"] = "SYNTHETIC emissions: 0.012 tCO2e."
        document["claims"] = [{"topic": "emissions", "value": 0.012,
                                "unit": "tCO2e", "source": "doc-e",
                                "quote": "0.012 tCO2e"}]
        self.source["documents"].append(document)
        self.assertEqual(len(self.run_pipeline()["data"]["research"]["disagreements"]), 1)

    def test_extracted_emissions_wrong_units_rejected(self):
        self.source["feedback"][0]["text"] = self.source["feedback"][0]["text"].replace(
            "12 kgCO2e", "12 kg")
        self.invalid()

    def test_negative_extracted_emissions_rejected(self):
        self.source["feedback"][0]["text"] = self.source["feedback"][0]["text"].replace(
            "12 kgCO2e", "-12 kgCO2e")
        self.invalid()

    def test_smart_meter_and_scada_formats(self):
        readings = self.run_pipeline()["data"]["research"]["meter_readings"]
        self.assertEqual([r["consumption_kwh"] for r in readings], [3.2, 3.7, 1.8])
        self.assertTrue(all(r["unit"] == "kWh" for r in readings))

    def test_negative_meter_consumption_rejected(self):
        self.source["meter_csv"] = self.source["meter_csv"].replace(",3.2", ",-3.2")
        self.invalid()

    def test_duplicate_meter_interval_rejected(self):
        self.source["meter_csv"] += self.source["meter_csv"].splitlines()[1] + "\n"
        self.invalid()

    def test_nonfinite_scada_rejected(self):
        self.source["telemetry"]["devices"][0]["voltage_kv"] = float("nan")
        self.invalid()

    def test_unknown_topology_asset_rejected(self):
        self.source["topology"][0][1] = "unknown"
        self.invalid()

    def test_naive_timestamp_rejected(self):
        self.source["telemetry"]["devices"][0]["timestamp"] = "2026-01-15T17:30:00"
        self.invalid()

    def test_false_evidence_quote_rejected(self):
        self.source["documents"][0]["claims"][0]["quote"] = "fabricated"
        self.invalid()

    def test_no_recommendations_has_empty_downstream(self):
        self.source["preferences"]["weights"] = {}
        data = self.run_pipeline()["data"]
        self.assertEqual(data["recommendations"], [])
        self.assertEqual(data["feedback_analysis"]["records"], [])
        self.assertEqual(data["extraction"], [])

    def test_empty_documents_has_unresolved_questions(self):
        self.source["documents"] = []
        data = self.run_pipeline()["data"]
        self.assertEqual(data["research"]["evidence"], [])
        self.assertGreaterEqual(len(data["research"]["unresolved_questions"]), 3)
        self.assertEqual(data["extraction"], [])

    def test_empty_feedback_is_supported(self):
        self.source["feedback"] = []
        self.assertEqual(self.run_pipeline()["data"]["extraction"], [])

    def test_synthetic_label_mandatory(self):
        self.source["synthetic"] = False
        self.invalid()

    def test_duplicate_schema_name_rejected(self):
        self.source["extraction_schema"].append(copy.deepcopy(self.source["extraction_schema"][0]))
        self.invalid()

    def test_repeated_extraction_label_rejected(self):
        self.source["feedback"][0]["text"] += "; Priority: critical"
        self.invalid()

    def test_wrong_stage_handoff_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.feedback(self.initial())

    def test_tampered_evidence_handoff_rejected(self):
        result = app.deep(self.initial())
        result["data"]["research"]["evidence"][0]["value"] = "unsupported"
        with self.assertRaises(app.ValidationError):
            app.interests(result)

    def test_tampered_recommendation_handoff_rejected(self):
        result = app.interests(app.deep(self.initial()))
        result["data"]["recommendations"][0]["evidence_ids"] = ["made-up"]
        with self.assertRaises(app.ValidationError):
            app.feedback(result)

    def test_tampered_feedback_handoff_rejected(self):
        result = app.feedback(app.interests(app.deep(self.initial())))
        result["data"]["feedback_analysis"]["records"][0]["text"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.extract(result)

    def test_pipeline_deterministic_and_input_unchanged(self):
        before = copy.deepcopy(self.source)
        self.assertEqual(self.run_pipeline(), self.run_pipeline())
        self.assertEqual(self.source, before)

    def test_cli_success_one_json_object(self):
        process = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file_json_error(self):
        process = subprocess.run([sys.executable, "-B", "implementation.py", "nonexistent.json"],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_error_without_writing_file(self):
        # The implementation itself is an existing non-JSON fixture.
        process = subprocess.run([sys.executable, "-B", "implementation.py", "implementation.py"],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_schema_error_without_writing_file(self):
        process = subprocess.run([sys.executable, "-B", "implementation.py", "build_manifest.json"],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_usage_error(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
