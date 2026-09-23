"""Tests use only clearly labeled synthetic fixtures; no external services."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def output(self):
        return app.run_pipeline(self.payload)

    def triage(self):
        return self.output()["stages"]["triage"]["tickets"]

    def issues(self):
        return self.output()["stages"]["sentiment"]["issues"]

    def one_ticket(self, text="Please help", severity="low", action_id=None, blocked=False):
        self.payload["tickets"] = [
            {"id": "synthetic-only", "text": text, "severity": severity, "action_id": action_id, "blocked": blocked}
        ]

    def test_example_is_complete_and_valid(self):
        result = self.output()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(tuple(result["stages"]), app.STAGE_NAMES)
        self.assertIs(app.validate(result, 4), result)

    def test_journey_is_two_steps_with_simulated_prerequisites(self):
        result = self.output()
        journey = result["stages"]["journey"]
        self.assertEqual([s["action_id"] for s in journey["steps"]], ["payout", "listing"])
        self.assertEqual(journey["steps"][1]["assumed_completed_before"], ["account", "payout"])
        self.assertTrue(journey["planning_only"])
        self.assertEqual(result["profile"]["completed"], ["account"])

    def test_next_actions_exclude_locked_and_completed_actions(self):
        available = self.output()["stages"]["journey"]["next_actions"]
        self.assertEqual([a["action_id"] for a in available], ["payout", "tour"])
        self.assertEqual(available[0]["interest_matches"], ["payments", "selling"])

    def test_interest_change_changes_journey_and_onboarding(self):
        self.payload["profile"]["interests"] = ["discovery"]
        result = self.output()["stages"]
        self.assertEqual(result["journey"]["steps"][0]["action_id"], "tour")
        self.assertEqual(result["adaptive"]["steps"][0]["action_id"], "tour")

    def test_empty_interests_have_stable_id_tie_break(self):
        self.payload["profile"]["interests"] = []
        expected = self.output()["stages"]["journey"]
        self.payload["actions"].reverse()
        self.assertEqual(self.output()["stages"]["journey"], expected)
        self.assertEqual(expected["steps"][0]["action_id"], "payout")

    def test_no_two_step_plan_is_validation_error(self):
        self.payload["profile"]["completed"] = ["account", "payout", "listing"]
        with self.assertRaisesRegex(app.ValidationError, "two-step"):
            self.output()

    def test_no_actions_is_validation_error(self):
        self.payload["actions"] = []
        self.payload["profile"]["completed"] = []
        self.payload["tickets"] = []
        with self.assertRaisesRegex(app.ValidationError, "two-step"):
            self.output()

    def test_beginner_hands_on_onboarding_explains_its_choices(self):
        adaptive = self.output()["stages"]["adaptive"]
        self.assertEqual(adaptive["experience"], "beginner")
        for step in adaptive["steps"]:
            self.assertEqual(step["format"], "hands_on")
            self.assertIn("detailed walkthrough", step["guidance"])
            self.assertIn("synthetic exercise", step["guidance"])
            self.assertIn("beginner", step["explanation"])
            self.assertTrue(step["prerequisites_met"])

    def test_all_experiences_and_preferences_are_supported(self):
        for experience in app.EXPERIENCES:
            for preference in app.FORMATS:
                with self.subTest(experience=experience, preference=preference):
                    self.payload["profile"].update(experience=experience, preference=preference)
                    step = self.output()["stages"]["adaptive"]["steps"][0]
                    self.assertIn(app.EXPERIENCES[experience], step["guidance"])
                    self.assertIn(app.FORMATS[preference], step["guidance"])

    def test_profile_preference_reaches_triage(self):
        self.payload["profile"]["preference"] = "video"
        ticket = self.triage()[0]
        self.assertEqual(ticket["onboarding_format"], "video")
        self.assertIn("narrated demonstration", ticket["onboarding_guidance"])

    def test_onboarding_route_and_blocked_priority(self):
        ticket = self.triage()[0]
        self.assertEqual(ticket["category"], "billing")
        self.assertEqual(ticket["category_source"], "onboarding")
        self.assertEqual(ticket["onboarding_step"], 1)
        self.assertEqual(ticket["priority"], 2)
        self.assertTrue(ticket["blocked_onboarding_boost"])
        self.assertEqual(ticket["owner"], "synthetic-billing-lead")

    def test_keyword_rule_overrides_onboarding_context(self):
        self.one_ticket("A service outage blocks the payout", action_id="payout")
        ticket = self.triage()[0]
        self.assertEqual(ticket["category"], "technical")
        self.assertEqual(ticket["category_source"], "keyword")
        self.assertEqual(ticket["matched_keywords"], ["service outage"])

    def test_rule_order_is_configurable(self):
        self.one_ticket("REFUND for a service outage")
        self.assertEqual(self.triage()[0]["category"], "technical")
        self.payload["config"]["rules"].reverse()
        self.assertEqual(self.triage()[0]["category"], "billing")

    def test_phrase_matching_does_not_use_substrings(self):
        self.one_ticket("The refunding dashboard is not an invoicebook")
        self.assertEqual(self.triage()[0]["category_source"], "default")
        self.one_ticket("SERVICE, OUTAGE!")
        self.assertEqual(self.triage()[0]["category"], "technical")

    def test_accountable_default_route_and_no_onboarding_boost(self):
        self.one_ticket(blocked=True)
        ticket = self.triage()[0]
        self.assertEqual(ticket["category_source"], "default")
        self.assertEqual(ticket["owner"], "synthetic-support-lead")
        self.assertEqual(ticket["queue"], "synthetic-general")
        self.assertIsNone(ticket["onboarding_step"])
        self.assertEqual(ticket["priority"], 4)
        self.assertFalse(ticket["blocked_onboarding_boost"])

    def test_known_but_unplanned_action_does_not_supply_context(self):
        self.one_ticket(action_id="tour", blocked=True)
        ticket = self.triage()[0]
        self.assertIsNone(ticket["onboarding_step"])
        self.assertEqual(ticket["category_source"], "default")
        self.assertEqual(ticket["priority"], 4)

    def test_all_severities_have_priority_floors(self):
        for severity, priority in app.SEVERITIES.items():
            with self.subTest(severity=severity):
                self.one_ticket(severity=severity)
                self.assertEqual(self.triage()[0]["priority"], priority)

    def test_route_priority_and_owner_are_configurable(self):
        self.one_ticket()
        self.payload["config"]["routes"]["general"].update(base_priority=1, owner="synthetic-on-call")
        result = self.output()["stages"]
        self.assertEqual(result["triage"]["tickets"][0]["priority"], 1)
        self.assertEqual(result["sentiment"]["issues"][0]["owner"], "synthetic-on-call")

    def test_score_is_sum_of_exposed_contributions(self):
        self.one_ticket("I hate this broken step")
        issue = self.issues()[0]
        self.assertEqual(issue["sentiment"]["score"], -6)
        self.assertEqual(issue["sentiment"]["label"], "negative")
        self.assertEqual(sum(x["contribution"] for x in issue["sentiment"]["contributions"]), -6)
        self.assertEqual(issue["priority"], 2)
        self.assertTrue(issue["negative_escalation"])
        self.assertEqual(issue["urgency"], 315)

    def test_negation_is_transparent(self):
        self.one_ticket("not good never bad")
        sentiment = self.issues()[0]["sentiment"]
        self.assertEqual(sentiment["score"], 0)
        self.assertEqual(sentiment["label"], "neutral")
        self.assertEqual([c["contribution"] for c in sentiment["contributions"]], [-2, 2])
        self.assertTrue(all(c["negated"] for c in sentiment["contributions"]))

    def test_sentiment_whole_tokens_casefold_and_repetition(self):
        self.one_ticket("GOOD goodness good")
        sentiment = self.issues()[0]["sentiment"]
        self.assertEqual(sentiment["score"], 4)
        self.assertEqual([c["index"] for c in sentiment["contributions"]], [0, 2])

    def test_empty_lexicon_and_unknown_words_are_neutral(self):
        self.payload["config"]["sentiment_lexicon"] = {}
        for issue in self.issues():
            self.assertEqual(issue["sentiment"]["score"], 0)
            self.assertEqual(issue["sentiment"]["contributions"], [])

    def test_positive_critical_issues_stay_urgent(self):
        issues = self.issues()
        self.assertEqual(issues[0]["ticket_id"], "synthetic-ticket-002")
        self.assertEqual(issues[0]["sentiment"]["label"], "positive")
        self.assertEqual(issues[0]["priority"], 1)

    def test_severity_precedes_negativity_within_same_priority(self):
        self.payload["tickets"] = [
            {"id": "synthetic-low", "text": "hate hate", "severity": "low", "action_id": None, "blocked": False},
            {"id": "synthetic-high", "text": "good", "severity": "high", "action_id": None, "blocked": False},
        ]
        issues = self.issues()
        self.assertEqual([i["priority"] for i in issues], [2, 2])
        self.assertEqual([i["ticket_id"] for i in issues], ["synthetic-high", "synthetic-low"])

    def test_equal_urgency_uses_ticket_id_and_contiguous_ranks(self):
        self.one_ticket()
        self.payload["tickets"][0]["id"] = "synthetic-z"
        duplicate = copy.deepcopy(self.payload["tickets"][0])
        duplicate["id"] = "synthetic-a"
        self.payload["tickets"].append(duplicate)
        issues = self.issues()
        self.assertEqual([i["ticket_id"] for i in issues], ["synthetic-a", "synthetic-z"])
        self.assertEqual([i["rank"] for i in issues], [1, 2])

    def test_full_cross_stage_identity_and_context_propagation(self):
        stages = self.output()["stages"]
        journey_step = stages["journey"]["steps"][0]
        onboarding_step = stages["adaptive"]["steps"][0]
        ticket = stages["triage"]["tickets"][0]
        issue = next(i for i in stages["sentiment"]["issues"] if i["ticket_id"] == ticket["ticket_id"])
        self.assertEqual(journey_step["action_id"], onboarding_step["action_id"])
        self.assertEqual(onboarding_step["action_id"], ticket["action_id"])
        for key in ("action_id", "category", "owner", "queue", "severity", "onboarding_step"):
            self.assertEqual(ticket[key], issue[key])
        self.assertEqual(issue["triage_priority"], ticket["priority"])

    def test_no_tickets_is_valid(self):
        self.payload["tickets"] = []
        result = self.output()["stages"]
        self.assertEqual(result["triage"]["tickets"], [])
        self.assertEqual(result["sentiment"]["issues"], [])

    def test_deterministic_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.payload)
        first = self.output()
        self.assertEqual(first, self.output())
        self.assertEqual(self.payload, before)

    def test_transition_rejects_missing_predecessor(self):
        state = dict(copy.deepcopy(self.payload), status="ok", stages={})
        with self.assertRaises(app.ValidationError):
            app.advance(state, "triage")

    def test_transition_rejects_tampered_journey(self):
        state = dict(copy.deepcopy(self.payload), status="ok", stages={})
        state = app.advance(state, "journey")
        state["stages"]["journey"]["steps"][0]["action_id"] = "listing"
        with self.assertRaisesRegex(app.ValidationError, "journey"):
            app.advance(state, "adaptive")

    def test_transition_rejects_tampered_onboarding(self):
        result = self.output()
        result["stages"].pop("sentiment")
        result["stages"].pop("triage")
        result["stages"]["adaptive"]["steps"][0]["support_category"] = "general"
        with self.assertRaisesRegex(app.ValidationError, "adaptive"):
            app.advance(result, "triage")

    def test_transition_rejects_tampered_triage_owner(self):
        result = self.output()
        result["stages"].pop("sentiment")
        result["stages"]["triage"]["tickets"][0]["owner"] = ""
        with self.assertRaisesRegex(app.ValidationError, "triage"):
            app.advance(result, "sentiment")

    def test_output_validation_rejects_boolean_for_integer(self):
        result = self.output()
        result["stages"]["journey"]["steps"][0]["position"] = True
        with self.assertRaises(app.ValidationError):
            app.validate(result, 4)

    def test_output_validation_rejects_tampered_sentiment(self):
        result = self.output()
        result["stages"]["sentiment"]["issues"][0]["priority"] = 4
        with self.assertRaisesRegex(app.ValidationError, "sentiment"):
            app.validate(result, 4)

    def test_duplicate_and_dangling_action_ids_rejected(self):
        original = copy.deepcopy(self.payload)
        for change in ("duplicate", "dangling", "completed", "ticket"):
            with self.subTest(change=change):
                self.payload = copy.deepcopy(original)
                if change == "duplicate":
                    self.payload["actions"].append(copy.deepcopy(self.payload["actions"][0]))
                elif change == "dangling":
                    self.payload["actions"][1]["prerequisites"] = ["missing"]
                elif change == "completed":
                    self.payload["profile"]["completed"] = ["missing"]
                else:
                    self.payload["tickets"][0]["action_id"] = "missing"
                with self.assertRaises(app.ValidationError):
                    self.output()

    def test_prerequisite_cycles_are_rejected_even_if_completed(self):
        self.payload["actions"][0]["prerequisites"] = ["payout"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            self.output()

    def test_inconsistent_completed_history_is_rejected(self):
        self.payload["profile"]["completed"] = ["listing"]
        with self.assertRaisesRegex(app.ValidationError, "completed prerequisites"):
            self.output()

    def test_invalid_types_unknown_keys_and_values_are_rejected(self):
        cases = [
            (("schema_version",), True),
            (("fixture",), "real"),
            (("profile", "experience"), "unknown"),
            (("profile", "preference"), []),
            (("profile", "interests"), ["selling", "selling"]),
            (("tickets", 0, "blocked"), 1),
            (("tickets", 0, "severity"), "urgent"),
            (("tickets", 0, "text"), " "),
            (("config", "routes", "billing", "owner"), ""),
            (("config", "routes", "billing", "base_priority"), True),
            (("config", "default_category"), "missing"),
            (("config", "sentiment_lexicon"), {"GOOD": 2}),
            (("config", "sentiment_lexicon"), {"good": 6}),
            (("config", "rules", 0, "keywords"), ["!!!"]),
        ]
        original = copy.deepcopy(self.payload)
        for path, value in cases:
            with self.subTest(path=path):
                self.payload = copy.deepcopy(original)
                parent = self.payload
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                with self.assertRaises(app.ValidationError):
                    self.output()
        self.payload = copy.deepcopy(original)
        self.payload["unknown"] = 1
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_duplicate_ticket_ids_rejected(self):
        self.payload["tickets"][1]["id"] = self.payload["tickets"][0]["id"]
        with self.assertRaisesRegex(app.ValidationError, "duplicate id"):
            self.output()

    def test_input_rejects_output_and_non_object_payload(self):
        for payload in (self.output(), [], None, True, "text"):
            with self.subTest(payload_type=type(payload).__name__):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"), *map(str, args)],
            cwd=HERE, capture_output=True, text=True, timeout=30, check=False,
        )

    def test_cli_success_one_json_object(self):
        result = self.cli(HERE / "example_input.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout), self.output())

    def test_cli_file_and_argument_errors(self):
        for args in ((), ("a", "b"), (HERE / "synthetic-does-not-exist.json",), (HERE,)):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stderr, "")
                self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_file(self):
        result = self.cli(HERE / "implementation.py")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_schema_error_duplicate_keys_and_nonfinite_values(self):
        for raw in ('{}', '[]', '{"fixture": "synthetic", "fixture": "synthetic"}', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(raw=raw):
                stdout = io.StringIO()
                with patch.object(app.Path, "read_text", return_value=raw), contextlib.redirect_stdout(stdout):
                    code = app.main(["synthetic-mocked-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_cli_invalid_utf8_is_json_error(self):
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        stdout = io.StringIO()
        with patch.object(app.Path, "read_text", side_effect=error), contextlib.redirect_stdout(stdout):
            code = app.main(["synthetic-mocked-input.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
