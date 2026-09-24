import copy
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

    def test_integrated_example(self):
        output = app.run(self.data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["research"]["findings"]), 3)
        self.assertEqual(output["entities"]["meter_readings"][0]["consumption"], 2.4)

    def test_novice_prerequisites_block_research(self):
        self.data["profile"]["completed"] = []
        output = app.run(self.data)
        self.assertEqual(output["status"], "needs_prerequisites")
        self.assertIn("energy_basics", output["onboarding"]["pending"])
        self.assertEqual(output["research"]["findings"], [])
        self.assertTrue(output["onboarding"]["lessons"][0]["explanation"])

    def test_experienced_skips_basics_not_safety(self):
        self.data["profile"].update(experience="experienced", completed=[])
        guide = app.onboard(self.data)
        self.assertNotIn("energy_basics", guide["required"])
        self.assertIn("safety_rules", guide["pending"])

    def test_exact_citations(self):
        output = app.run(self.data)
        sources = {p["source_id"]: p["text"] for p in self.data["passages"]}
        for finding in output["research"]["findings"]:
            citation = finding["citation"]
            self.assertEqual(citation["quote"],
                             sources[citation["source_id"]][citation["start"]:citation["end"]])
            self.assertEqual(finding["finding"], citation["quote"])

    def test_preference_and_topics_propagate(self):
        self.data["profile"].update(topics=["outages"], detail="brief")
        duplicate = copy.deepcopy(self.data["passages"][1])
        duplicate["source_id"] = "z-other"
        self.data["passages"].append(duplicate)
        output = app.run(self.data)
        self.assertEqual(output["research"]["topics"], ["outages"])
        self.assertEqual(len(output["research"]["findings"]), 1)
        self.data["profile"]["detail"] = "guided"
        self.assertEqual(len(app.run(self.data)["research"]["findings"]), 2)

    def test_tampered_handoff_rejected(self):
        guide = app.onboard(self.data)
        guide["topics"] = ["outages"]
        with self.assertRaises(app.ValidationError):
            app.research(guide, self.data)

    def test_forged_readiness_rejected(self):
        self.data["profile"]["completed"] = []
        guide = app.onboard(self.data)
        guide["ready"] = True
        with self.assertRaises(app.ValidationError):
            app.research(guide, self.data)

    def test_forged_citation_rejected(self):
        guide = app.onboard(self.data)
        result = app.research(guide, self.data)
        result["findings"][0]["citation"]["end"] -= 1
        with self.assertRaises(app.ValidationError):
            app.validate("research", result, (guide, self.data))

    def test_unknown_topic_rejected(self):
        self.data["profile"]["topics"] = ["trading"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_no_matches_explicit(self):
        self.data["passages"][0]["topics"] = ["outages"]
        self.assertEqual(app.run(self.data)["research"]["unanswered_topics"], ["consumption"])

    def test_identifiers_protected(self):
        serialized = json.dumps(app.run(self.data))
        for asset in self.data["grid_assets"]:
            self.assertNotIn(asset["internal_id"], serialized)
        self.data["passages"][0]["text"] += " SYN-SUBSTATION-001"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_all_outage_priorities(self):
        outage = self.data["outage_reports"][0]
        for life, essential, priority in [(True, True, "P1"), (True, False, "P1"),
                                          (False, True, "P2"), (False, False, "P3")]:
            outage.update(life_safety=life, essential_service=essential, priority=priority)
            self.assertEqual(app.run(self.data)["entities"]["outage_reports"][0]["priority"], priority)
            outage["priority"] = "P3" if priority != "P3" else "P1"
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_declared_safety_policy_not_overridable(self):
        self.data["safety_rules"]["life_safety"] = "P3"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_emission_units_source_and_finiteness(self):
        for key, value in [("unit", ""), ("source_id", "missing"), ("value", float("nan")),
                           ("value", -1)]:
            data = copy.deepcopy(self.data)
            data["telemetry"]["samples"][0]["emissions"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_meter_invalid_and_overlap(self):
        original = self.data["meter_csv"]
        for raw in [original.replace("kWh", "kW"), original.replace(",2.4,", ",nan,"),
                    original.replace("18:30:00", "18:00:00"), "wrong,header\n"]:
            self.data["meter_csv"] = raw
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_topology_and_synthetic_label(self):
        self.data["topology"][0][1] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)
        self.data["topology"] = []
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_timezone_required(self):
        self.data["telemetry"]["samples"][0]["timestamp"] = "2026-01-01T12:00:00"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_cli_success_and_determinism(self):
        command = [sys.executable, "-B", str(ROOT / "implementation.py"),
                   str(ROOT / "example_input.json")]
        first = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        second = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stderr, "")
        self.assertEqual(json.loads(first.stdout)["status"], "ok")
        self.assertEqual(first.stdout, second.stdout)

    def test_cli_file_and_usage_errors(self):
        for args in [[], ["absent.json"], ["example_input.json", "extra"]]:
            completed = subprocess.run([sys.executable, "-B", "implementation.py", *args],
                                       capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_malformed_json_and_schema_errors(self):
        for raw in ["{invalid", "null", '{"synthetic":true}']:
            with patch("builtins.open", mock_open(read_data=raw)), patch("sys.stdout", new_callable=app.io.StringIO) as output:
                self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
