"""Tests use only clearly labeled synthetic fixture data."""

import contextlib
import copy
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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_pipeline(self):
        return app.run(self.data)["results"]

    def envelope(self):
        return {"schema_version": 1, "status": "ok", "stage": "input",
                "input": copy.deepcopy(self.data), "results": {}}

    def test_integrated_output(self):
        result = app.run(self.data)
        self.assertEqual(result["stage"], "sentiment")
        self.assertEqual(list(result["results"]), list(app.STAGES))
        self.assertIs(app.validate(result, "sentiment"), result)

    def test_journey_prerequisites(self):
        journey = self.run_pipeline()["journey"]
        self.assertEqual([s["action_id"] for s in journey["steps"]],
                         ["account", "listing"])
        self.assertEqual(journey["steps"][1]["satisfied_prerequisites"], ["account"])
        self.assertNotIn("listing", [s["action_id"] for s in journey["next_actions"]])

    def test_completed_actions_are_excluded(self):
        self.data["profile"]["completed"] = ["account"]
        journey = self.run_pipeline()["journey"]
        self.assertEqual(journey["steps"][0]["action_id"], "listing")
        self.assertNotIn("account", [s["action_id"] for s in journey["steps"]])

    def test_preference_changes_discovery(self):
        self.data["profile"]["preferences"] = ["buying", "video"]
        self.assertEqual(self.run_pipeline()["journey"]["steps"][0]["action_id"], "browse")

    def test_insufficient_remaining_actions(self):
        self.data["profile"]["completed"] = ["account", "listing", "analytics"]
        with self.assertRaisesRegex(app.ValidationError, "two-step"):
            app.run(self.data)

    def test_adaptive_consumes_journey(self):
        results = self.run_pipeline()
        self.assertEqual([m["action_id"] for m in results["adaptive"]["modules"]],
                         [s["action_id"] for s in results["journey"]["steps"]])
        module = results["adaptive"]["modules"][1]
        self.assertEqual(module["format"], "interactive")
        self.assertEqual(module["depth"], "guided")
        self.assertIn("prerequisites", module["explanation"])

    def test_expert_fast_track(self):
        self.data["profile"]["experience"] = "expert"
        self.assertTrue(all(m["depth"] == "fast-track"
                            for m in self.run_pipeline()["adaptive"]["modules"]))

    def test_no_format_preference_uses_default(self):
        self.data["profile"]["preferences"] = ["selling"]
        self.assertEqual(self.run_pipeline()["adaptive"]["modules"][1]["format"], "text")

    def test_accountable_categorization(self):
        ticket = self.run_pipeline()["triage"]["tickets"][0]
        self.assertEqual(ticket["category"], "technical")
        self.assertEqual(ticket["owner"], "synthetic-platform-team")
        self.assertEqual(ticket["matched_keywords"], ["broken", "upload"])
        self.assertEqual(ticket["priority"], "high")

    def test_adaptive_context_propagates_to_triage(self):
        results = self.run_pipeline()
        ticket = results["triage"]["tickets"][1]
        self.assertEqual(ticket["onboarding"], results["adaptive"]["modules"][0])
        self.assertEqual(ticket["priority"], "medium")
        self.assertIsNone(results["triage"]["tickets"][2]["onboarding"])

    def test_routing_fallback(self):
        ticket = self.run_pipeline()["triage"]["tickets"][2]
        self.assertEqual(ticket["category"], "general")
        self.assertEqual(ticket["owner"], "synthetic-support-team")

    def test_routing_is_configurable(self):
        self.data["routing"]["rules"][0]["owner"] = "synthetic-new-owner"
        self.data["routing"]["rules"][0]["priority"] = "critical"
        ticket = self.run_pipeline()["triage"]["tickets"][0]
        self.assertEqual(ticket["owner"], "synthetic-new-owner")
        self.assertEqual(ticket["priority"], "critical")

    def test_rule_tie_uses_configuration_order(self):
        self.data["tickets"][0]["text"] = "upload payment"
        self.assertEqual(self.run_pipeline()["triage"]["tickets"][0]["category"], "technical")

    def test_word_boundaries_not_substrings(self):
        self.data["tickets"][0]["text"] = "uploading payments"
        self.assertEqual(self.run_pipeline()["triage"]["tickets"][0]["category"], "general")

    def test_transparent_sentiment(self):
        score = app.sentiment("Great, but broken and confusing.")
        self.assertEqual(score["score"], -1)
        self.assertEqual(score["label"], "negative")
        self.assertEqual(sum(c["value"] for c in score["contributions"]), -1)

    def test_negation(self):
        self.assertEqual(app.sentiment("not good never broken")["score"], 1)
        self.assertTrue(app.sentiment("not good")["contributions"][0]["negated"])

    def test_unknown_words_and_empty_sentiment(self):
        self.assertEqual(app.sentiment("metrics xyz")["label"], "neutral")
        self.assertEqual(app.sentiment("")["score"], 0)

    def test_sentiment_escalation(self):
        issue = self.run_pipeline()["sentiment"]["issues"][0]
        self.assertEqual(issue["ticket_id"], "SYN-1")
        self.assertEqual(issue["sentiment"]["score"], -3)
        self.assertEqual(issue["final_priority"], "critical")

    def test_positive_sentiment_cannot_lower_severity(self):
        self.data["tickets"][0]["text"] = "excellent great love"
        self.data["tickets"][0]["severity"] = "critical"
        issue = self.run_pipeline()["sentiment"]["issues"][0]
        self.assertEqual(issue["sentiment"]["label"], "positive")
        self.assertEqual(issue["final_priority"], "critical")

    def test_triage_handoff_preserved(self):
        results = self.run_pipeline()
        issues = {i["ticket_id"]: i for i in results["sentiment"]["issues"]}
        for ticket in results["triage"]["tickets"]:
            for key, value in ticket.items():
                self.assertEqual(issues[ticket["ticket_id"]][key], value)

    def test_empty_tickets(self):
        self.data["tickets"] = []
        self.assertEqual(self.run_pipeline()["sentiment"]["issues"], [])

    def test_deterministic_and_nonmutating(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(original, self.data)

    def test_reject_missing_and_unknown_fields(self):
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                if change == "missing":
                    del data["profile"]
                else:
                    data["unexpected"] = 1
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_invalid_scalar_values(self):
        for path, value in [
            (("schema_version",), True), (("synthetic",), False),
            (("profile", "experience"), "guru"),
            (("actions", 0, "difficulty"), True),
            (("tickets", 0, "severity"), "urgent"),
            (("routing", "fallback", "owner"), " "),
        ]:
            with self.subTest(path=path):
                data = copy.deepcopy(self.data)
                target = data
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_unknown_references(self):
        self.data["tickets"][0]["action_id"] = "missing"
        with self.assertRaisesRegex(app.ValidationError, "unknown action"):
            app.run(self.data)

    def test_unknown_prerequisite(self):
        self.data["actions"][0]["prerequisites"] = ["missing"]
        with self.assertRaisesRegex(app.ValidationError, "unknown prerequisite"):
            app.run(self.data)

    def test_cycle(self):
        self.data["actions"][0]["prerequisites"] = ["listing"]
        with self.assertRaisesRegex(app.ValidationError, "cyclic"):
            app.run(self.data)

    def test_duplicate_ids(self):
        self.data["actions"][1]["id"] = "account"
        with self.assertRaisesRegex(app.ValidationError, "duplicate"):
            app.run(self.data)

    def test_tampered_journey_rejected_by_next_stage(self):
        envelope = app.advance(self.envelope(), "journey")
        envelope["results"]["journey"]["steps"].reverse()
        with self.assertRaisesRegex(app.ValidationError, "journey"):
            app.advance(envelope, "adaptive")

    def test_tampered_adaptive_rejected_by_triage(self):
        envelope = app.advance(app.advance(self.envelope(), "journey"), "adaptive")
        envelope["results"]["adaptive"]["modules"][0]["depth"] = "unvalidated"
        with self.assertRaisesRegex(app.ValidationError, "adaptive"):
            app.advance(envelope, "triage")

    def test_tampered_triage_rejected_by_sentiment(self):
        envelope = self.envelope()
        for stage in app.STAGES[:3]:
            envelope = app.advance(envelope, stage)
        envelope["results"]["triage"]["tickets"][0]["owner"] = ""
        with self.assertRaisesRegex(app.ValidationError, "triage"):
            app.advance(envelope, "sentiment")

    def test_cannot_skip_stages(self):
        with self.assertRaisesRegex(app.ValidationError, "stage"):
            app.advance(self.envelope(), "triage")

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        result = self.cli("does-not-exist.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_directory_error(self):
        result = self.cli(".")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for raw in ("{", "null", "[]", '{"x": NaN}', '{"x":1,"x":2}',
                    '{"schema_version":false}'):
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=raw)), \
                        contextlib.redirect_stdout(output):
                    status = app.main(["synthetic-invalid.json"])
                self.assertEqual(status, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
