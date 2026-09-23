"""All fixtures in this suite are synthetic; no network or provider is used."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / "example_input.json").open(encoding="utf-8") as handle:
            self.data = json.load(handle)

    def run_pipeline(self):
        return impl.run_pipeline(self.data)

    def report(self, result, topic):
        return next(item for item in result["research"]["topics"] if item["topic"] == topic)

    def ticket(self, result, topic):
        return next(item for item in result["triage"]["tickets"] if item["topic"] == topic)

    def assert_invalid(self, data=None):
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(self.data if data is None else data)

    def test_two_step_journey_checks_prerequisites_in_order(self):
        result = self.run_pipeline()
        journey = result["journey"]
        self.assertEqual([step["action_id"] for step in journey["steps"]],
                         ["enable-checkout", "launch-store"])
        completed = set(self.data["user"]["completed_actions"])
        for step in journey["steps"]:
            self.assertTrue(set(step["prerequisites"]) <= completed)
            self.assertNotIn(step["action_id"], completed)
            completed.add(step["action_id"])
        self.assertNotIn("launch-store", [item["action_id"] for item in journey["next_actions"]])

    def test_personalized_goals_change_journey_and_downstream_topics(self):
        self.data["user"]["goals"] = ["analytics"]
        result = self.run_pipeline()
        self.assertEqual(result["journey"]["steps"][0]["action_id"], "view-analytics")
        self.assertIn("analytics-ready", result["journey"]["research_topics"])
        self.assertEqual(self.ticket(result, "analytics-ready")["category"], "analytics")
        self.assertNotIn("region-ready", result["journey"]["research_topics"])

    def test_unrelated_prerequisite_can_unlock_relevant_second_step(self):
        self.data["actions"][1]["goals"] = ["setup"]
        self.data["user"]["goals"] = ["safe-launch"]
        result = self.run_pipeline()
        self.assertEqual(result["journey"]["steps"][0]["matching_goals"], [])
        self.assertEqual(result["journey"]["steps"][1]["action_id"], "launch-store")

    def test_multi_document_disagreement_retains_both_citations(self):
        report = self.report(self.run_pipeline(), "checkout-ready")
        self.assertEqual(report["source_count"], 2)
        self.assertEqual(report["disagreement"]["supports"], ["SYNTHETIC-guide"])
        self.assertEqual(report["disagreement"]["opposes"], ["SYNTHETIC-test-report"])
        self.assertTrue(report["unresolved_questions"])
        self.assertIn("do not establish truth", report["synthesis"])

    def test_agreement_is_separate_from_missing_evidence(self):
        result = self.run_pipeline()
        report = self.report(result, "refunds-ready")
        self.assertEqual(report["reason_codes"], ["resolved"])
        self.assertEqual(report["unresolved_questions"], [])
        self.assertIsNone(report["disagreement"])
        missing = self.report(result, "region-ready")
        self.assertEqual(missing["source_count"], 0)
        self.assertEqual(missing["reason_codes"], ["missing_evidence"])

    def test_research_only_uses_selected_journey_topics(self):
        result = self.run_pipeline()
        self.assertNotIn("analytics-ready", [item["topic"] for item in result["research"]["topics"]])
        self.assertEqual(result["research"]["journey_action_ids"],
                         [step["action_id"] for step in result["journey"]["steps"]])

    def test_single_uncertain_source_creates_two_unresolved_questions(self):
        self.data["documents"] = [self.data["documents"][0]]
        self.data["documents"][0]["claims"][0]["stance"] = "uncertain"
        report = self.report(self.run_pipeline(), "checkout-ready")
        self.assertEqual(report["reason_codes"], ["uncertain", "limited_sources"])
        self.assertEqual(len(report["unresolved_questions"]), 2)
        self.assertIsNone(report["disagreement"])

    def test_empty_documents_are_valid_and_route_missing_evidence(self):
        self.data["documents"] = []
        result = self.run_pipeline()
        self.assertEqual(result["research"]["document_ids"], [])
        for ticket in result["triage"]["tickets"]:
            self.assertEqual(ticket["priority"], "high")
            self.assertEqual(ticket["reason_codes"], ["missing_evidence"])
            self.assertEqual(ticket["document_ids"], [])

    def test_triage_preserves_actions_sources_questions_and_summary(self):
        result = self.run_pipeline()
        for report, ticket in zip(result["research"]["topics"], result["triage"]["tickets"]):
            self.assertEqual(ticket["topic"], report["topic"])
            self.assertEqual(ticket["action_ids"], report["action_ids"])
            self.assertEqual(ticket["document_ids"],
                             [item["document_id"] for item in report["evidence"]])
            self.assertEqual(ticket["questions"], report["unresolved_questions"])
            self.assertEqual(ticket["summary"], report["synthesis"])
            self.assertEqual(ticket["reason_codes"], report["reason_codes"])
        checkout = self.ticket(result, "checkout-ready")
        self.assertEqual(checkout["action_ids"], ["enable-checkout", "launch-store"])

    def test_configurable_priority_category_and_accountable_fallback(self):
        self.data["routing"]["priorities"]["disagreement"] = "normal"
        self.data["routing"]["categories"][0]["owner"] = "SYNTHETIC-new-owner"
        result = self.run_pipeline()
        checkout = self.ticket(result, "checkout-ready")
        self.assertEqual(checkout["priority"], "normal")
        self.assertEqual(checkout["owner"], "SYNTHETIC-new-owner")
        fallback = self.ticket(result, "region-ready")
        self.assertEqual(fallback["category"], "general")
        self.assertEqual(fallback["owner"], "SYNTHETIC-support-lead")
        self.assertEqual(fallback["queue"], "SYNTHETIC-support-review")

    def test_overlapping_routes_use_first_configured_category(self):
        self.data["routing"]["categories"].insert(0, {
            "id": "first", "topics": ["checkout-ready"],
            "owner": "SYNTHETIC-first-owner", "queue": "SYNTHETIC-first-queue",
        })
        self.assertEqual(self.ticket(self.run_pipeline(), "checkout-ready")["category"], "first")

    def test_highest_configured_priority_wins_multiple_reasons(self):
        self.data["documents"][1]["claims"][0]["stance"] = "uncertain"
        self.data["documents"] = [self.data["documents"][1]]
        self.data["routing"]["priorities"]["uncertain"] = "low"
        self.data["routing"]["priorities"]["limited_sources"] = "urgent"
        self.assertEqual(self.ticket(self.run_pipeline(), "checkout-ready")["priority"], "urgent")

    def test_deterministic_and_does_not_mutate_input(self):
        original = copy.deepcopy(self.data)
        first = self.run_pipeline()
        self.assertEqual(first, self.run_pipeline())
        self.assertEqual(self.data, original)
        self.data["actions"].reverse()
        self.data["documents"].reverse()
        self.assertEqual(first, self.run_pipeline())

    def test_no_two_step_plan_is_validation_error(self):
        self.data["user"]["completed_actions"] = ["create-account", "enable-checkout", "launch-store"]
        self.assert_invalid()

    def test_no_matching_goal_is_validation_error(self):
        self.data["user"]["goals"] = ["unavailable-goal"]
        self.assert_invalid()

    def test_cycles_and_unknown_prerequisites_are_rejected(self):
        for prerequisites in (["launch-store"], ["not-an-action"], ["enable-checkout"]):
            with self.subTest(prerequisites=prerequisites):
                data = copy.deepcopy(self.data)
                data["actions"][1]["prerequisites"] = prerequisites
                self.assert_invalid(data)

    def test_completed_actions_require_known_ids_and_prerequisite_closure(self):
        for completed in (["missing"], ["launch-store"], ["create-account", "create-account"]):
            with self.subTest(completed=completed):
                data = copy.deepcopy(self.data)
                data["user"]["completed_actions"] = completed
                self.assert_invalid(data)

    def test_duplicate_ids_and_claim_topics_are_rejected(self):
        for field in ("actions", "documents"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data[field].append(copy.deepcopy(data[field][0]))
                self.assert_invalid(data)
        self.data["documents"][0]["claims"].append(copy.deepcopy(self.data["documents"][0]["claims"][0]))
        self.assert_invalid()

    def test_invalid_structures_and_unknown_fields_are_rejected(self):
        for invalid in (None, [], True, {}, {"schema_version": "2.0"}):
            with self.subTest(invalid=invalid):
                self.assert_invalid_value(invalid)
        for field, value in (("synthetic", False), ("schema_version", 1), ("request_id", " "),
                             ("actions", {}), ("documents", None)):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data[field] = value
                self.assert_invalid(data)
        self.data["unexpected"] = True
        self.assert_invalid()

    def assert_invalid_value(self, value):
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(value)

    def test_invalid_routing_and_claim_stance_are_rejected(self):
        for modify in (
            lambda d: d["routing"].update(fallback_category="unknown"),
            lambda d: d["routing"]["categories"][0].update(owner=""),
            lambda d: d["routing"]["priorities"].update(disagreement="critical"),
            lambda d: d["documents"][0]["claims"][0].update(stance="guessed"),
        ):
            data = copy.deepcopy(self.data)
            modify(data)
            self.assert_invalid(data)

    def test_tampered_journey_is_rejected_by_research_consumer(self):
        journey = impl.recommend_journey(self.data)
        journey["steps"].reverse()
        with self.assertRaises(impl.ValidationError):
            impl.deep_research(self.data, journey)

    def test_tampered_evidence_is_rejected_by_triage_consumer(self):
        result = self.run_pipeline()
        result["research"]["topics"][0]["evidence"][0]["text"] = "Fabricated quote."
        with self.assertRaises(impl.ValidationError):
            impl.triage_tickets(self.data, result["journey"], result["research"])

    def test_stale_research_handoff_is_rejected(self):
        result = self.run_pipeline()
        self.data["documents"][0]["claims"][0]["stance"] = "uncertain"
        with self.assertRaises(impl.ValidationError):
            impl.triage_tickets(self.data, result["journey"], result["research"])

    def test_wrong_owner_and_bool_source_count_are_rejected(self):
        result = self.run_pipeline()
        triage = copy.deepcopy(result["triage"])
        triage["tickets"][0]["owner"] = "unaccountable"
        with self.assertRaises(impl.ValidationError):
            impl.validate("triage", triage, request=self.data,
                          journey=result["journey"], research=result["research"])
        result["research"]["topics"][0]["source_count"] = True
        with self.assertRaises(impl.ValidationError):
            impl.validate("research", result["research"], request=self.data, journey=result["journey"])

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=15)

    def test_cli_success_one_json_object(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout), self.run_pipeline())

    def test_cli_missing_file_directory_and_usage_errors(self):
        for args in ((), ("does-not-exist.json",), (".",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stderr, "")
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_bad_json_duplicate_keys_and_nonfinite_values(self):
        for payload in ("{", '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', "null", "[]"):
            with self.subTest(payload=payload):
                stdout = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)):
                    with contextlib.redirect_stdout(stdout):
                        code = impl.main(["synthetic-memory-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_cli_encoding_and_permission_errors_are_json(self):
        for error in (PermissionError("Synthetic access denial"),
                      UnicodeDecodeError("utf-8", b"\xff", 0, 1, "synthetic invalid encoding")):
            stdout = io.StringIO()
            with patch("builtins.open", side_effect=error):
                with contextlib.redirect_stdout(stdout):
                    code = impl.main(["synthetic-memory-fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
