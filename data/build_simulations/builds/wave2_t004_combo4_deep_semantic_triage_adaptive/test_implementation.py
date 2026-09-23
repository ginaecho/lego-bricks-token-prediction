"""Synthetic fixtures only; tests never use a provider or write scratch files."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / "example_input.json").open(encoding="utf-8") as handle:
            self.data = json.load(handle)

    def run_stages(self):
        return app.run_pipeline(self.data)["stages"]

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

    def mocked_cli(self, text):
        output = io.StringIO()
        with patch.object(Path, "open", return_value=io.StringIO(text)):
            with redirect_stdout(output):
                code = app.main(["synthetic.json"])
        return code, json.loads(output.getvalue())

    def test_example_runs_all_four_stages(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), ["research", "semantic", "triage", "adaptive"])
        self.assertTrue(result["fixture_label"].startswith("SYNTHETIC"))

    def test_research_synthesizes_disagreement_and_missing_evidence(self):
        research = self.run_stages()["research"]
        offline, export, retention = research["findings"]
        self.assertEqual(offline["status"], "disputed")
        self.assertEqual(offline["source_count"], 2)
        self.assertEqual(export["status"], "supported")
        self.assertEqual(retention["status"], "unanswered")
        self.assertEqual(research["unresolved_question_ids"], ["offline", "retention"])
        self.assertEqual([e["evidence_id"] for e in offline["evidence"]], ["guide:1", "trial:1"])

    def test_sources_not_counted_twice_for_duplicate_source_documents(self):
        duplicate = copy.deepcopy(self.data["documents"][0])
        duplicate["id"] = "guide-copy"
        self.data["documents"].append(duplicate)
        self.assertEqual(self.run_stages()["research"]["findings"][0]["source_count"], 2)

    def test_uncertain_and_opposing_only_findings_remain_unresolved(self):
        self.data["documents"] = [self.data["documents"][0]]
        self.data["documents"][0]["evidence"][0]["stance"] = "uncertain"
        self.data["documents"][0]["evidence"][1]["stance"] = "oppose"
        findings = self.run_stages()["research"]["findings"]
        self.assertEqual([f["status"] for f in findings], ["uncertain", "contested", "unanswered"])

    def test_support_plus_uncertain_does_not_hide_uncertainty(self):
        self.data["documents"][1]["evidence"][0]["stance"] = "uncertain"
        self.assertEqual(self.run_stages()["research"]["findings"][0]["status"], "uncertain")

    def test_research_text_enriches_search_catalog(self):
        result = self.run_stages()["semantic"]["results"][0]
        self.assertEqual(result["product_id"], "trail-kit")
        self.assertGreater(result["score"], 0)
        self.assertIn("trial:1", result["evidence_ids"])
        self.data["documents"] = []
        self.assertEqual(self.run_stages()["semantic"]["results"], [])

    def test_synonym_expansion_matches_query(self):
        self.data["search"]["query"] = "SYNC"
        self.assertEqual(self.run_stages()["semantic"]["results"][0]["product_id"], "trail-kit")
        self.data["search"]["synonyms"] = []
        self.assertEqual(self.run_stages()["semantic"]["results"], [])

    def test_ranking_ties_break_by_product_id(self):
        self.data["documents"] = []
        self.data["search"]["query"] = "matching"
        self.data["products"] = [
            {"id": "trail-kit", "name": "Matching", "description": "Same content", "tags": []},
            {"id": "desk-kit", "name": "Matching", "description": "Same content", "tags": []},
        ]
        ids = [p["product_id"] for p in self.run_stages()["semantic"]["results"]]
        self.assertEqual(ids, ["desk-kit", "trail-kit"])
        self.data["search"]["top_k"] = 1
        self.assertEqual(len(self.run_stages()["semantic"]["results"]), 1)

    def test_unrelated_query_has_no_fabricated_result(self):
        self.data["search"]["query"] = "xyzzyqz"
        stages = self.run_stages()
        self.assertEqual(stages["semantic"]["results"], [])
        self.assertIsNone(stages["triage"]["selected_product_id"])
        self.assertEqual(stages["triage"]["priority"], "normal")
        self.assertFalse(stages["adaptive"]["needs_human_review"])
        self.assertEqual(stages["adaptive"]["steps"], [])

    def test_embedding_injection_is_local_validated_and_affects_ranking(self):
        batches = []

        def synthetic_embedder(texts):
            batches.append(texts)
            return [[1, 0], [0, 1], [1, 0]]

        self.data["search"]["embedding_weight"] = 1
        result = app.run_pipeline(self.data, embedder=synthetic_embedder)["stages"]
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)
        self.assertIn("Offline synchronization", batches[0][1])
        self.assertEqual(result["semantic"]["results"][0]["product_id"], "desk-kit")
        self.assertTrue(result["semantic"]["embedding_used"])
        self.assertFalse(result["adaptive"]["needs_human_review"])

    def test_disabled_embedding_never_calls_injection(self):
        def forbidden(_):
            self.fail("disabled embedder must not be called")
        app.run_pipeline(self.data, embedder=forbidden)

    def test_positive_embedding_weight_requires_callable(self):
        self.data["search"]["embedding_weight"] = 0.5
        with self.assertRaisesRegex(app.ValidationError, "injected embedder"):
            app.run_pipeline(self.data)

    def test_invalid_embedding_outputs_rejected(self):
        self.data["search"]["embedding_weight"] = 0.5
        fixtures = [
            None, [], [[1, 0]], [[1, 0], [1], [1, 0]],
            [[1, 0], [0, 0], [1, 0]], [[1, 0], [True, 1], [1, 0]],
            [[1, 0], [float("nan"), 1], [1, 0]],
            [[1, 0], [float("inf"), 1], [1, 0]], [[], [], []],
            [[1, 0], ["1", 0], [1, 0]],
        ]
        for vectors in fixtures:
            with self.subTest(vectors=vectors), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, embedder=lambda _, v=vectors: v)

    def test_embedding_failure_is_a_validation_error(self):
        self.data["search"]["embedding_weight"] = 0.5
        def broken(_):
            raise RuntimeError("synthetic provider-free failure")
        with self.assertRaisesRegex(app.ValidationError, "embedder failed"):
            app.run_pipeline(self.data, embedder=broken)

    def test_extreme_finite_embedding_vectors_normalize_without_overflow(self):
        self.data["search"]["embedding_weight"] = 1
        result = app.run_pipeline(self.data, embedder=lambda _: [
            [1e308, 1e308], [1e308, 1e308], [-1e308, -1e308]
        ])
        self.assertEqual(result["stages"]["semantic"]["results"][0]["score"], 1.0)

    def test_cross_stage_uncertainty_drives_priority_and_review(self):
        stages = self.run_stages()
        result = stages["semantic"]["results"][0]
        triage = stages["triage"]
        onboarding = stages["adaptive"]
        self.assertEqual(result["attention_question_ids"], ["offline"])
        self.assertEqual(triage["attention_question_ids"], result["attention_question_ids"])
        self.assertEqual(triage["evidence_ids"], result["evidence_ids"])
        self.assertEqual(triage["priority"], "high")
        self.assertEqual(onboarding["attention_question_ids"], ["offline"])
        self.assertEqual(onboarding["accountable_owner"], triage["owner"])
        self.assertTrue(onboarding["needs_human_review"])
        self.assertIn("review", [s["step_id"] for s in onboarding["steps"]])

    def test_resolving_disagreement_removes_review_and_escalation(self):
        self.data["documents"][1]["evidence"][0]["stance"] = "support"
        stages = self.run_stages()
        self.assertEqual(stages["triage"]["priority"], "normal")
        self.assertEqual(stages["triage"]["attention_question_ids"], [])
        self.assertNotIn("review", [s["step_id"] for s in stages["adaptive"]["steps"]])
        self.assertFalse(stages["adaptive"]["needs_human_review"])
        self.assertIn("retention", stages["research"]["unresolved_question_ids"])

    def test_urgent_keywords_override_lower_priorities(self):
        self.data["ticket"]["body"] += " We have DATA LOSS."
        self.assertEqual(self.run_stages()["triage"]["priority"], "urgent")

    def test_routing_most_matches_and_tie_order(self):
        self.data["ticket"] = {"id": "SYNTHETIC-2", "subject": "setup refund", "body": "help"}
        self.assertEqual(self.run_stages()["triage"]["rule_id"], "setup")
        self.data["ticket"]["body"] = "invoice"
        triage = self.run_stages()["triage"]
        self.assertEqual(triage["rule_id"], "billing")
        self.assertEqual(triage["owner"], "synthetic-billing-owner")
        self.assertEqual(triage["onboarding_track"], "account")

    def test_routing_uses_word_boundaries_and_fallback(self):
        self.data["ticket"] = {"id": "SYNTHETIC-3", "subject": "preconfiguration", "body": "refundish"}
        result = self.run_stages()["triage"]
        self.assertIsNone(result["rule_id"])
        self.assertEqual(result["owner"], "synthetic-help-owner")

    def test_account_route_propagates_to_adaptive_track(self):
        self.data["ticket"] = {"id": "SYNTHETIC-4", "subject": "invoice", "body": "refund"}
        stages = self.run_stages()
        self.assertEqual(stages["adaptive"]["track"], "account")
        self.assertEqual(stages["adaptive"]["accountable_owner"], "synthetic-billing-owner")
        ids = [s["step_id"] for s in stages["adaptive"]["steps"]]
        self.assertIn("account", ids)
        self.assertNotIn("connect", ids)

    def test_beginner_formats_and_prerequisites_have_explanations(self):
        steps = self.run_stages()["adaptive"]["steps"]
        self.assertEqual([s["step_id"] for s in steps], ["welcome", "review", "connect"])
        self.assertEqual([s["format"] for s in steps], ["video", "text", "interactive"])
        self.assertIn("prerequisite", steps[0]["reason"])
        self.assertIn("No preferred format", steps[1]["reason"])

    def test_advanced_profile_still_gets_required_beginner_prerequisites(self):
        self.data["profile"]["experience"] = "advanced"
        steps = self.run_stages()["adaptive"]["steps"]
        self.assertEqual([s["step_id"] for s in steps], ["welcome", "review", "connect", "automate"])
        self.assertIn("overrides", next(s["reason"] for s in steps if s["step_id"] == "connect"))

    def test_completed_steps_are_omitted_but_human_review_remains(self):
        self.data["profile"]["completed_step_ids"] = ["welcome", "review", "connect"]
        result = self.run_stages()["adaptive"]
        self.assertEqual(result["steps"], [])
        self.assertTrue(result["needs_human_review"])

    def test_empty_preference_list_uses_supported_format(self):
        self.data["profile"]["preferred_formats"] = []
        self.assertEqual(self.run_stages()["adaptive"]["steps"][0]["format"], "text")

    def test_no_research_documents_is_supported(self):
        self.data["documents"] = []
        self.data["search"]["query"] = "Trail"
        stages = self.run_stages()
        self.assertTrue(all(f["status"] == "unanswered" for f in stages["research"]["findings"]))
        self.assertEqual(stages["semantic"]["results"][0]["evidence_ids"], [])

    def test_empty_catalog_and_empty_research_are_supported(self):
        self.data["questions"] = []
        self.data["documents"] = []
        self.data["products"] = []
        for step in self.data["onboarding"]["steps"]:
            step["product_ids"] = []
        stages = self.run_stages()
        self.assertEqual(stages["research"]["findings"], [])
        self.assertEqual(stages["semantic"]["index"], [])
        self.assertIsNone(stages["triage"]["selected_product_id"])

    def test_input_is_not_mutated_and_outputs_are_repeatable(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(first, app.run_pipeline(self.data))
        first["stages"]["research"]["findings"][0]["evidence"][0]["product_ids"].clear()
        self.assertEqual(self.data, original)

    def test_duplicate_ids_and_unknown_references_rejected(self):
        mutations = [
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["documents"][0]["evidence"][0].update(question_id="missing"),
            lambda d: d["documents"][0]["evidence"][0].update(product_ids=["missing"]),
            lambda d: d["onboarding"]["steps"][0].update(prerequisites=["missing"]),
            lambda d: d["profile"].update(completed_step_ids=["missing"]),
            lambda d: d["routing"]["fallback"].update(onboarding_track="missing"),
            lambda d: d["onboarding"].update(review_step_id="missing"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_cycles_rejected_even_when_completed(self):
        self.data["onboarding"]["steps"][0]["prerequisites"] = ["connect"]
        self.data["profile"]["completed_step_ids"] = ["welcome", "connect"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            self.run_stages()

    def test_schema_type_ranges_and_unknown_fields_rejected(self):
        mutations = [
            lambda d: d.update(unexpected=True),
            lambda d: d.update(schema_version="2.0"),
            lambda d: d["search"].update(top_k=True),
            lambda d: d["search"].update(top_k=0),
            lambda d: d["search"].update(top_k=101),
            lambda d: d["search"].update(embedding_weight=-0.1),
            lambda d: d["search"].update(embedding_weight=1.1),
            lambda d: d["search"].update(embedding_weight=float("nan")),
            lambda d: d["search"].update(embedding_weight=10 ** 400),
            lambda d: d["search"].update(query="!!!"),
            lambda d: d["routing"]["fallback"].update(owner=" "),
            lambda d: d["routing"]["rules"][0].update(keywords=[]),
            lambda d: d["onboarding"]["steps"][0].update(formats=[]),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_research_handoff_stops_before_search(self):
        corrupt = self.run_stages()["research"]
        corrupt["findings"][0]["evidence"][0]["excerpt"] = "Fabricated synthetic evidence"
        with patch.object(app, "research", return_value=corrupt):
            with patch.object(app, "semantic") as next_stage:
                with self.assertRaisesRegex(app.ValidationError, "provenance"):
                    app.run_pipeline(self.data)
                next_stage.assert_not_called()

    def test_invalid_semantic_handoff_stops_before_triage(self):
        corrupt = self.run_stages()["semantic"]
        corrupt["results"][0]["attention_question_ids"] = []
        with patch.object(app, "semantic", return_value=corrupt):
            with patch.object(app, "triage") as next_stage:
                with self.assertRaisesRegex(app.ValidationError, "provenance"):
                    app.run_pipeline(self.data)
                next_stage.assert_not_called()

    def test_invalid_triage_handoff_stops_before_onboarding(self):
        corrupt = self.run_stages()["triage"]
        corrupt["owner"] = "unaccountable-owner"
        with patch.object(app, "triage", return_value=corrupt):
            with patch.object(app, "adaptive") as next_stage:
                with self.assertRaisesRegex(app.ValidationError, "owner"):
                    app.run_pipeline(self.data)
                next_stage.assert_not_called()

    def test_invalid_adaptive_output_is_not_emitted(self):
        corrupt = self.run_stages()["adaptive"]
        corrupt["steps"].reverse()
        with patch.object(app, "adaptive", return_value=corrupt):
            with self.assertRaisesRegex(app.ValidationError, "order"):
                app.run_pipeline(self.data)

    def test_cli_success_prints_one_json_object(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_missing_file_returns_json_and_exit_two(self):
        result = self.cli("definitely_missing_synthetic_fixture.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_directory_input_returns_json_and_exit_two(self):
        result = self.cli(".")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_returns_json_and_exit_two(self):
        result = self.cli("test_implementation.py")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_argument_errors(self):
        for args in ((), ("example_input.json", "extra")):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_rejects_duplicate_keys_nonfinite_values_and_bad_shapes(self):
        for text in ('{"a":1,"a":2}', '{"value":NaN}', '{"value":Infinity}',
                     '[]', '{}', 'null', '1'):
            with self.subTest(text=text):
                code, result = self.mocked_cli(text)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_schema_failure_is_json_error(self):
        self.data["search"]["top_k"] = 0
        code, result = self.mocked_cli(json.dumps(self.data))
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "error")
        self.assertIn("top_k", result["error"])

    def test_cli_invalid_encoding_is_json_error(self):
        output = io.StringIO()
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "synthetic invalid encoding")
        with patch.object(Path, "open", side_effect=error):
            with redirect_stdout(output):
                code = app.main(["synthetic.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
