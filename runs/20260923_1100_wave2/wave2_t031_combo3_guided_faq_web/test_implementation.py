"""Offline synthetic fixtures; no network, provider, or third-party packages."""

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
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_stages(self):
        return app.run_pipeline(self.request)["stages"]

    def test_happy_path_and_cross_stage_provenance(self):
        stages = self.run_stages()
        self.assertEqual([s["stage"] for s in stages], ["guided", "faq", "web"])
        self.assertEqual([s["status"] for s in stages], ["ready", "answered", "answered"])
        self.assertEqual(stages[0]["context"], stages[1]["context"])
        self.assertEqual(stages[1]["context"], stages[2]["context"])
        self.assertEqual(stages[0]["handoff"]["query"], stages[2]["handoff"]["query"])
        self.assertEqual(stages[2]["handoff"]["upstream_source_ids"], ["faq-export"])
        self.assertEqual(stages[2]["evidence"][0]["url"],
                         self.request["research"]["pages"][0]["url"])

    def test_progress_optional_step_does_not_block(self):
        guided = self.run_stages()[0]
        self.assertEqual(guided["progress"], {
            "completed": ["account", "workspace"], "remaining": ["tour"],
            "available": ["tour"], "total": 3, "completed_count": 2})
        self.assertTrue(guided["handoff"]["ready"])

    def test_incomplete_onboarding_blocks_downstream(self):
        self.request["setup"]["completed"] = []
        result = app.run_pipeline(self.request)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stages"][0]["progress"]["available"], ["account"])
        for stage in result["stages"][1:]:
            self.assertEqual(stage["reason"], "onboarding_incomplete")
            self.assertIsNone(stage["answer"])
            self.assertFalse(stage["handoff"]["ready"])

    def test_completed_step_requires_prerequisites(self):
        self.request["setup"]["completed"] = ["workspace"]
        with self.assertRaisesRegex(app.ValidationError, "incomplete prerequisites"):
            self.run_stages()

    def test_cycles_and_unknown_prerequisites(self):
        for dependency in ("tour", "missing"):
            with self.subTest(dependency=dependency):
                request = copy.deepcopy(self.request)
                request["setup"]["steps"][0]["requires"] = [dependency]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_duplicate_and_unknown_completed_steps(self):
        for completed in (["account", "account"], ["unknown"]):
            with self.subTest(completed=completed):
                self.request["setup"]["completed"] = completed
                with self.assertRaises(app.ValidationError):
                    self.run_stages()

    def test_faq_grounding_and_product_scope(self):
        faq = self.run_stages()[1]
        self.assertEqual(faq["handoff"]["source_ids"], ["faq-export"])
        self.assertEqual(faq["answer"], self.request["knowledge"][0]["text"])
        self.assertIsNone(faq["evidence"][0]["url"])

    def test_faq_abstention_still_hands_off_to_web(self):
        self.request["knowledge"] = []
        _, faq, web = self.run_stages()
        self.assertEqual(faq["status"], "abstained")
        self.assertEqual(faq["reason"], "no_relevant_evidence")
        self.assertTrue(faq["handoff"]["ready"])
        self.assertEqual(web["status"], "answered")
        self.assertEqual(web["handoff"]["upstream_source_ids"], [])

    def test_no_relevant_evidence_explicit_abstention(self):
        self.request["question"] = "Quantum teleportation?"
        for stage in self.run_stages()[1:]:
            self.assertEqual(stage["status"], "abstained")
            self.assertEqual(stage["evidence"], [])
            self.assertIsNone(stage["answer"])

    def test_empty_web_corpus(self):
        self.request["research"]["pages"] = []
        self.assertEqual(self.run_stages()[2]["status"], "abstained")

    def test_url_policy_rejects_host_confusion_and_unsafe_urls(self):
        urls = [
            "http://docs.demodesk.example/export",
            "https://docs.demodesk.example.evil.example/export",
            "https://sub.docs.demodesk.example/export",
            "https://docs.demodesk.example@evil.example/export",
            "https://user@docs.demodesk.example/export",
            "https://docs.demodesk.example:444/export",
            "https://docs.demodesk.example:bad/export",
            "https://docs.demodesk.example/export#fragment",
            "https://docs.demodesk.example\\evil/export",
            "https://docs.demodesk.example/\nexport",
            "file:///export",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.request["research"]["pages"][0]["url"] = url
                with self.assertRaises(app.ValidationError):
                    self.run_stages()

    def test_url_policy_accepts_exact_https_and_preserves_provenance(self):
        url = "https://DOCS.DEMODESK.EXAMPLE:443/export?edition=synthetic"
        self.request["research"]["pages"][0]["url"] = url
        self.assertEqual(self.run_stages()[2]["evidence"][0]["url"], url)

    def test_duplicate_corpus_identifiers_and_urls(self):
        for field in ("id", "url"):
            with self.subTest(field=field):
                request = copy.deepcopy(self.request)
                request["research"]["pages"][1][field] = request["research"]["pages"][0][field]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)
        self.request["knowledge"][1]["id"] = self.request["knowledge"][0]["id"]
        with self.assertRaises(app.ValidationError):
            self.run_stages()

    def test_shared_input_validation(self):
        changes = [
            ("schema_version", True), ("schema_version", 2), ("question", " "),
            ("question", []), ("fixture_label", "real"), ("knowledge", None),
            ("unknown", 42),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                request = copy.deepcopy(self.request)
                request[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_malformed_nested_values(self):
        for value in (None, 3, [], "setup"):
            with self.subTest(value=value):
                self.request["setup"] = value
                with self.assertRaises(app.ValidationError):
                    self.run_stages()

    def test_shared_stage_validation_rejects_corrupt_handoff(self):
        guided, faq, _ = self.run_stages()
        guided["handoff"]["ready"] = "yes"
        with self.assertRaises(app.ValidationError):
            app.faq_stage(guided, self.request["knowledge"])
        faq["handoff"]["source_ids"] = ["invented"]
        with self.assertRaises(app.ValidationError):
            app.web_stage(faq, self.request["research"])

    def test_injected_answerer_validated_and_isolated(self):
        calls = []

        def fixture_answerer(stage, query, evidence):
            calls.append((stage, query))
            answer = {"answer": evidence[0]["text"], "source_ids": [evidence[0]["id"]]}
            evidence[0]["text"] = "malicious mutation"
            return answer

        output = app.run_pipeline(self.request, fixture_answerer)
        self.assertEqual([stage for stage, _ in calls], ["faq", "web"])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertNotIn("malicious", output["stages"][1]["answer"])

    def test_injected_hallucinations_and_unknown_citations_rejected(self):
        proposals = [
            {"answer": "Invented recommendation", "source_ids": ["faq-export"]},
            {"answer": "Synthetic text", "source_ids": ["fake"]},
            {"answer": "", "source_ids": []},
            {"answer": "text"},
        ]
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.request, lambda *_: proposal)

    def test_injected_answerer_exception_is_validation_error(self):
        def broken(*_):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "injected answerer failed"):
            app.run_pipeline(self.request, broken)

    def test_input_is_not_mutated_and_runs_are_deterministic(self):
        before = copy.deepcopy(self.request)
        self.assertEqual(app.run_pipeline(self.request), app.run_pipeline(self.request))
        self.assertEqual(self.request, before)

    def test_retrieval_ties_sorted_by_id_and_limited(self):
        candidates = [{"id": name, "text": "export"} for name in ("z", "d", "a", "c")]
        self.assertEqual([x["id"] for x in app.retrieve("export", candidates)], ["a", "c", "d"])

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        result = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ((), ("nonexistent-synthetic-input.json",), ("one", "two")):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_file(self):
        # This existing Python source is intentionally not valid JSON.
        result = self.cli(str(ROOT / "test_implementation.py"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_valid_json_invalid_schema(self):
        result = self.cli(str(ROOT / "build_manifest.json"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_duplicate_json_keys_and_nonfinite_numbers(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"x":1,"x":2}', object_pairs_hook=app.reject_duplicates)
        with self.assertRaises(app.ValidationError):
            json.loads('{"x":NaN}', parse_constant=app.reject_constant)


if __name__ == "__main__":
    unittest.main()
