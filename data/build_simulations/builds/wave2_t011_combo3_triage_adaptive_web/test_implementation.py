"""All fixtures are synthetic; tests perform no network calls or external writes."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as impl


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_pipeline(self):
        return impl.run_pipeline(self.data)

    def assert_invalid(self):
        with self.assertRaises(impl.ValidationError):
            self.run_pipeline()

    def test_complete_pipeline_and_determinism(self):
        before = copy.deepcopy(self.data)
        first = self.run_pipeline()
        self.assertEqual(first, self.run_pipeline())
        self.assertEqual(self.data, before)
        self.assertEqual(first["stage"], "web")
        self.assertEqual(first["status"], "ok")

    def test_triage_category_and_accountability(self):
        triage = self.run_pipeline()["triage"]
        self.assertEqual(triage["category"], "billing")
        self.assertEqual(triage["owner"], "synthetic-billing-owner")
        self.assertEqual(triage["team"], "Synthetic Billing Support")

    def test_fallback_and_tie_are_deterministic(self):
        self.data["ticket"].update(subject="unrecognized", body="synthetic nonsense")
        self.assertEqual(self.run_pipeline()["triage"]["category"], "general")
        self.data["ticket"]["body"] = "invoice help"
        self.assertEqual(self.run_pipeline()["triage"]["category"], "billing")

    def test_priority_rule_and_severity_floor(self):
        self.data["ticket"]["body"] += " outage"
        self.assertEqual(self.run_pipeline()["triage"]["priority"], "urgent")
        self.data["ticket"]["body"] = "invoice"
        self.data["ticket"]["severity"] = "high"
        self.assertEqual(self.run_pipeline()["triage"]["priority"], "high")

    def test_keyword_boundaries(self):
        self.data["ticket"].update(subject="unrelated", body="invoices billingish helpless")
        self.assertEqual(self.run_pipeline()["triage"]["category"], "general")

    def test_onboarding_prerequisites_and_preferences(self):
        plan = self.run_pipeline()["adaptive"]["plan"]
        self.assertEqual([s["id"] for s in plan], ["workspace-basics", "configure-billing"])
        self.assertEqual([s["format"] for s in plan], ["text", "interactive"])
        self.assertIn("fallback", plan[0]["reason"])
        self.assertIn("prerequisite", plan[0]["reason"])

    def test_completed_prerequisite_is_skipped(self):
        self.data["profile"]["completed_steps"] = ["workspace-basics"]
        adaptive = self.run_pipeline()["adaptive"]
        self.assertEqual([s["id"] for s in adaptive["plan"]], ["configure-billing"])
        self.assertEqual(adaptive["skipped_completed"], ["workspace-basics"])

    def test_experience_blocks_dependent_steps(self):
        self.data["config"]["steps"][0]["min_experience"] = "advanced"
        adaptive = self.run_pipeline()["adaptive"]
        self.assertEqual(adaptive["plan"], [])
        self.assertEqual(len(adaptive["blocked"]), 2)
        self.assertEqual(adaptive["research_terms"], ["invoice"])

    def test_advanced_experience_unlocks_steps(self):
        self.data["config"]["steps"][1]["min_experience"] = "advanced"
        self.data["profile"]["experience"] = "advanced"
        self.assertEqual(len(self.run_pipeline()["adaptive"]["plan"]), 2)

    def test_handoff_identity_route_priority_and_terms(self):
        output = self.run_pipeline()
        for field in ("ticket_id", "category", "owner", "priority"):
            self.assertEqual(output["triage"][field], output["adaptive"][field])
            self.assertEqual(output["adaptive"][field], output["web"][field])
        self.assertIn("configuration", output["web"]["query_terms"])
        self.assertEqual(output["web"]["plan_step_ids"],
                         [s["id"] for s in output["adaptive"]["plan"]])

    def test_retrieval_preserves_provenance(self):
        web = self.run_pipeline()["web"]
        self.assertEqual(web["status"], "found")
        for finding in web["findings"]:
            source = next(s for s in self.data["sources"] if s["id"] == finding["source_id"])
            provenance = finding["provenance"]
            self.assertEqual(finding["finding"],
                             source["content"][provenance["start"]:provenance["end"]])
            self.assertEqual(finding["url"], source["url"])
            self.assertTrue(provenance["synthetic"])
            self.assertEqual(finding["score"], len(finding["matched_terms"]))

    def test_retrieval_ranking_and_limit(self):
        self.data["config"]["max_findings"] = 1
        findings = self.run_pipeline()["web"]["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["source_id"], "synthetic-guide-1")

    def test_no_sources_and_no_matching_content(self):
        for source in self.data["sources"]:
            source["content"] = "Synthetic unrelated zebras"
        self.assertEqual(self.run_pipeline()["web"]["status"], "no_results")
        self.data["sources"] = []
        self.assertEqual(self.run_pipeline()["web"]["findings"], [])

    def test_allowlist_rejects_unsafe_urls(self):
        urls = ["http://docs.synthetic.example/a", "https://evil.example/a",
                "https://docs.synthetic.example.evil.example/a",
                "https://docs.synthetic.example@evil.example/a",
                "https://user@docs.synthetic.example/a",
                "https://docs.synthetic.example:8080/a",
                "https://docs.synthetic.example/a#fragment",
                "https://docs.synthetic.example\\evil/a",
                "https://docs.synthetic.example/\nfoo",
                "https://docs.synthetic.example:bad/a"]
        for url in urls:
            with self.subTest(url=url):
                self.data["sources"][0]["url"] = url
                self.assert_invalid()

    def test_url_canonicalization_and_duplicate_rejection(self):
        self.data["sources"][0]["url"] = "https://DOCS.SYNTHETIC.EXAMPLE:443/billing"
        self.assertEqual(self.run_pipeline()["web"]["findings"][0]["url"],
                         "https://docs.synthetic.example/billing")
        self.data["sources"][1]["url"] = "https://docs.synthetic.example/billing"
        self.assert_invalid()

    def test_graph_cycle_and_unknown_reference(self):
        self.data["config"]["steps"][0]["prerequisites"] = ["configure-billing"]
        self.assert_invalid()
        self.data["config"]["steps"][0]["prerequisites"] = ["missing"]
        self.assert_invalid()

    def test_invalid_shapes_types_and_unknown_fields(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(unknown="field"),
            lambda d: d["ticket"].update(severity="critical"),
            lambda d: d["config"].update(max_findings=True),
            lambda d: d["config"].update(categories=[]),
            lambda d: d["profile"].update(completed_steps=["missing"]),
            lambda d: d["sources"][0].update(synthetic=False),
            lambda d: d["config"]["categories"][0].update(owner=" "),
            lambda d: d["config"].update(allowlisted_hosts=["*.example"]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(self.data)
                mutation(candidate)
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(candidate)

    def test_valid_injected_classifier_and_mutation_isolation(self):
        def classifier(ticket, categories):
            ticket["id"] = "changed"
            self.assertIn("general", categories)
            return {"category": "general", "priority": "high"}
        output = impl.run_pipeline(self.data, classifier)
        self.assertEqual(output["triage"]["category"], "general")
        self.assertEqual(output["web"]["owner"], "synthetic-general-owner")
        self.assertEqual(output["web"]["priority"], "high")
        self.assertEqual(output["web"]["ticket_id"], self.data["ticket"]["id"])

    def test_invalid_injected_classifier_and_failure(self):
        for proposal in [None, {}, {"category": "unknown", "priority": "normal"},
                         {"category": "general", "priority": "bad"},
                         {"category": "general", "priority": "low", "owner": "injected"}]:
            with self.subTest(proposal=proposal), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(self.data, lambda *_: proposal)

        def failed(*_):
            raise RuntimeError("synthetic failure")
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(self.data, failed)

    def test_handoff_validation_rejects_tampering(self):
        triaged = impl.triage_stage(impl.initial_envelope(self.data))
        triaged["triage"]["owner"] = "wrong"
        with self.assertRaises(impl.ValidationError):
            impl.adaptive_stage(triaged)
        adaptive = impl.adaptive_stage(impl.triage_stage(impl.initial_envelope(self.data)))
        adaptive["adaptive"]["research_terms"].append("injected")
        with self.assertRaises(impl.ValidationError):
            impl.web_stage(adaptive)
        output = self.run_pipeline()
        output["web"]["findings"][0]["finding"] = "fabricated"
        with self.assertRaises(impl.ValidationError):
            impl.validate_envelope(output, "web")

    def test_out_of_order_stage_is_rejected(self):
        with self.assertRaises(impl.ValidationError):
            impl.web_stage(impl.initial_envelope(self.data))

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                 str(HERE / "example_input.json")],
                                capture_output=True, text=True, check=False, cwd=HERE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], ["absent-synthetic-input.json"]]:
            result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                    capture_output=True, text=True, check=False, cwd=HERE)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_input(self):
        for raw in ["{", '{"schema_version":1,"schema_version":1}', "NaN", "[]",
                    json.dumps(dict(self.data, synthetic=False))]:
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = impl.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_empty_recommendations(self):
        self.data["config"]["categories"][0]["onboarding_steps"] = []
        self.data["config"]["categories"][0]["research_terms"] = []
        output = self.run_pipeline()
        self.assertEqual(output["adaptive"]["plan"], [])
        self.assertEqual(output["web"]["status"], "no_results")


if __name__ == "__main__":
    unittest.main()
