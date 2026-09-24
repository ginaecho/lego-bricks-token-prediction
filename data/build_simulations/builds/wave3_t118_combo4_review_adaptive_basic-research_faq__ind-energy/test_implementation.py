import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))
        self.context = self.data["context"]

    def reject(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def cli(self, content=None, args=None):
        return subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] +
                              (args if args is not None else ["-"]),
                              input=content, text=True, capture_output=True, cwd=HERE)

    def test_full_pipeline_and_shared_schema(self):
        output = app.run(self.data)
        self.assertEqual(output["stage"], "faq")
        self.assertEqual(set(output), set(self.data))
        self.assertEqual(list(output["results"]), list(app.STAGES[1:]))
        app.validate(output)

    def test_review_traceable_supported_and_gap(self):
        result = app.review(self.data)["results"]["review"]
        self.assertEqual(result["gaps"], ["REQ-CALIBRATION"])
        self.assertEqual(result["checks"][0]["status"], "supported")
        self.assertEqual(result["checks"][0]["evidence"][0]["source_id"], "SYN-S2")
        self.assertEqual(result["checks"][2]["missing_terms"], ["calibration certificate"])

    def test_smart_meter_seasonal_totals(self):
        summary = app.review(self.data)["results"]["review"]["meter_summary"]
        self.assertEqual(summary["seasonal_consumption"], {"winter": 7.9, "summer": 2.2})
        self.assertEqual(summary["unit"], "kWh")

    def test_critical_identifiers_protected_everywhere(self):
        output = app.run(self.data)
        serialized = json.dumps(output)
        self.assertNotIn("FICT-SUB-001", serialized)
        self.assertNotIn("FICT-RTU-001", serialized)
        self.assertIn("protected-asset-0002", serialized)
        app.validate(output)

    def test_inputs_not_mutated_and_deterministic(self):
        original = copy.deepcopy(self.data)
        first = app.run(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(first, app.run(self.data))

    def test_prerequisites_are_ordered(self):
        result = app.adaptive(app.review(self.data))["results"]["adaptive"]
        topics = [module["topic"] for module in result["modules"]]
        for module in result["modules"]:
            for prerequisite in module["prerequisites"]:
                self.assertLess(topics.index(prerequisite), topics.index(module["topic"]))
        self.assertEqual(result["review_gap_ids"], ["REQ-CALIBRATION"])

    def test_experience_and_preference(self):
        self.context["profile"].update(experience="expert", preference="concise")
        result = app.adaptive(app.review(self.data))["results"]["adaptive"]
        self.assertTrue(all(m["depth"] == "accelerated" for m in result["modules"]))
        self.assertTrue(all("listed prerequisites" not in m["explanation"] for m in result["modules"]))

    def test_completed_topics_skipped(self):
        self.context["profile"]["completed_topics"] = list(app.PREREQUISITES)
        result = app.adaptive(app.review(self.data))["results"]["adaptive"]
        self.assertEqual(result["modules"], [])
        self.assertEqual(result["review_gap_ids"], ["REQ-CALIBRATION"])

    def test_research_consumes_generated_gap_question(self):
        result = app.run(self.data)["results"]
        finding = next(f for f in result["research"]["findings"]
                       if f["question_id"] == "gap:REQ-CALIBRATION")
        self.assertEqual(finding["origin"], "REQ-CALIBRATION")
        self.assertEqual(finding["status"], "insufficient")
        self.assertEqual(result["research"]["review_gap_ids"], result["review"]["gaps"])
        self.assertEqual(result["faq"]["unresolved_review_gaps"], result["review"]["gaps"])

    def test_faq_grounded_verbatim(self):
        output = app.run(self.data)
        answer = output["results"]["faq"]["answers"][0]
        self.assertEqual(answer["status"], "answered")
        sources = {s["id"]: s["text"] for s in output["context"]["sources"]}
        for citation in answer["citations"]:
            self.assertIn(sources[citation], answer["answer"])

    def test_faq_explicit_abstention(self):
        answer = app.run(self.data)["results"]["faq"]["answers"][2]
        self.assertEqual(answer["status"], "abstained")
        self.assertEqual(answer["citations"], [])
        self.assertIn("cannot", answer["answer"])

    def test_conflicts_abstain(self):
        self.context["sources"].append({"id": "SYN-CONFLICT", "title": "Synthetic conflict",
                                       "kind": "secondary", "text": "Winter consumption falls.",
                                       "claims": {"peak_season": "summer"}})
        result = app.run(self.data)["results"]
        self.assertEqual(result["research"]["findings"][0]["status"], "conflicting")
        self.assertEqual(result["faq"]["answers"][0]["status"], "abstained")

    def test_empty_source_corpus_abstains(self):
        self.context["sources"] = []
        self.context["emissions"] = []
        for requirement in self.context["requirements"]:
            requirement["source_ids"] = []
        output = app.run(self.data)
        self.assertTrue(all(a["status"] == "abstained" for a in output["results"]["faq"]["answers"]))

    def test_no_sources_are_invented_for_gaps(self):
        output = app.run(self.data)
        self.assertEqual(output["results"]["review"]["checks"][2]["evidence"], [])

    def test_tampered_handoff_rejected(self):
        output = app.review(self.data)
        output["results"]["review"]["gaps"] = []
        with self.assertRaises(app.ValidationError):
            app.adaptive(output)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.faq(app.review(self.data))

    def test_safety_priority_violation(self):
        self.context["outages"][0]["priority"] = "P3"
        self.reject()

    def test_essential_service_priority(self):
        self.context["outages"][0]["safety"]["life_safety"] = False
        self.context["outages"][0]["priority"] = "P2"
        self.assertEqual(app.run(self.data)["results"]["review"]["outage_priorities"][0]["priority"], "P2")

    def test_safety_rules_cannot_be_weakened(self):
        self.context["safety_rules"]["life_safety"] = "P3"
        self.reject()

    def test_emissions_units_and_source_required(self):
        for field, value in (("unit", "tons"), ("source_id", "unknown")):
            with self.subTest(field=field):
                saved = self.context["emissions"][0][field]
                self.context["emissions"][0][field] = value
                self.reject()
                self.context["emissions"][0][field] = saved

    def test_invalid_numeric_values(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.subTest(value=value):
                self.context["emissions"][0]["value"] = value
                self.reject()

    def test_unknown_grid_reference(self):
        self.context["topology"][0][1] = "unknown"
        self.reject()

    def test_telemetry_units_and_device(self):
        self.context["telemetry"]["observations"][0]["unit"] = "kWh"
        self.reject()
        self.context["telemetry"]["observations"][0]["unit"] = "V"
        self.context["telemetry"]["observations"][0]["device_id"] = "wrong"
        self.reject()

    def test_invalid_csv(self):
        self.context["meter_csv"] = "wrong,header\n1,2\n"
        self.reject()

    def test_duplicate_intervals(self):
        self.context["meter_csv"] += self.context["meter_csv"].splitlines()[1] + "\n"
        self.reject()

    def test_equivalent_timezone_duplicate_intervals(self):
        self.context["meter_csv"] += self.context["meter_csv"].splitlines()[1].replace(
            "2026-01-15T18:00:00Z", "2026-01-15T19:00:00+01:00") + "\n"
        self.reject()

    def test_timezone_required(self):
        self.context["meter_csv"] = self.context["meter_csv"].replace("18:00:00Z", "18:00:00")
        self.reject()

    def test_synthetic_label_required(self):
        self.context["synthetic"] = False
        self.reject()

    def test_duplicate_ids_rejected(self):
        self.context["sources"].append(copy.deepcopy(self.context["sources"][0]))
        self.reject()

    def test_nonempty_questions_required(self):
        self.context["questions"] = []
        self.reject()

    def test_profile_invalid(self):
        self.context["profile"]["experience"] = "unlimited"
        self.reject()

    def test_cli_example(self):
        process = self.cli(args=["example_input.json"])
        self.assertEqual(process.returncode, 0, process.stdout)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["stage"], "faq")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_error(self):
        process = self.cli(args=["does-not-exist.json"])
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_validation_and_json_errors(self):
        for content in ("{", "null", '{"status": "ok", "status": "error"}',
                        '{"value": NaN}', json.dumps({**self.data, "schema_version": 99})):
            with self.subTest(content=content[:20]):
                process = self.cli(content)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_usage_error(self):
        process = self.cli(args=[])
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_error_does_not_expose_identifiers(self):
        self.context["outages"][0]["priority"] = "FICT-SUB-001"
        process = self.cli(json.dumps(self.data))
        self.assertEqual(process.returncode, 2)
        self.assertNotIn("FICT-SUB-001", process.stdout)

    def test_overflowing_consumption_total(self):
        self.context["meter_csv"] = self.context["meter_csv"].replace(",3.8,", ",1e308,").replace(
            ",4.1,", ",1e308,")
        self.reject()
        process = self.cli(json.dumps(self.data))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_nonfinite_unknown_metadata(self):
        self.context["metadata"] = {"untrusted": float("inf")}
        self.reject()
        content = json.dumps(self.data).replace("Infinity", "1e999")
        process = self.cli(content)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_extremely_large_integer_cli(self):
        process = self.cli('{"schema_version":' + "9" * 5000 + "}")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_general_purpose_research_not_energy_keyword_specific(self):
        self.context["sources"].append({
            "id": "SYN-GENERAL", "title": "Synthetic archive study", "kind": "primary",
            "text": "Archive retention lasts twelve months for synthetic records.", "claims": {}
        })
        self.context["questions"] = [{"id": "Q-ARCHIVE", "text": "How long is archive retention?"}]
        answer = app.run(self.data)["results"]["faq"]["answers"][0]
        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["citations"], ["SYN-GENERAL"])

    def test_conflict_outside_three_excerpts_still_blocks_answer(self):
        for index in range(4):
            self.context["sources"].append({
                "id": "Z-CONFLICT-%d" % index, "title": "Synthetic source",
                "kind": "secondary", "text": "Winter consumption changes.",
                "claims": {"peak_season": "summer" if index == 3 else "winter"}
            })
        result = app.run(self.data)["results"]
        self.assertEqual(len(result["research"]["findings"][0]["evidence"]), 3)
        self.assertEqual(result["faq"]["answers"][0]["status"], "abstained")


if __name__ == "__main__":
    unittest.main()
