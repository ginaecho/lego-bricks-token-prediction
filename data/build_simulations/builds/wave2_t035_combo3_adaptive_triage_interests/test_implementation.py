"""Standard-library tests using explicitly synthetic fixture data."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.state = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        self.request = self.state["request"]

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.state)
        self.assertEqual(result["stage"], "interests")
        self.assertEqual([x["id"] for x in result["recommendations"]["items"]], ["synthetic-a", "synthetic-d"])
        self.assertEqual(result["recommendations"]["items"][0]["score"], 8)
        app.validate_state(result, "interests")

    def test_beginner_plan_orders_prerequisites(self):
        plan = app.adaptive(self.state)["onboarding"]["plan"]
        self.assertEqual([x["id"] for x in plan], ["privacy", "workspace"])
        self.assertTrue(plan[0]["available_now"])
        self.assertFalse(plan[1]["available_now"])
        self.assertIn("privacy", plan[1]["prerequisites"])

    def test_expert_plan_does_not_waive_prerequisites(self):
        self.request["profile"]["experience"] = "expert"
        result = app.adaptive(self.state)["onboarding"]
        self.assertEqual(result["plan"][-1]["id"], "automation")
        self.assertFalse(result["plan"][-1]["available_now"])
        self.assertEqual(result["completed_steps"], ["basics"])

    def test_preferred_presentation_changes_explanations(self):
        guided = app.adaptive(self.state)["onboarding"]["plan"][0]["explanation"]
        self.request["profile"]["presentation"] = "concise"
        concise = app.adaptive(self.state)["onboarding"]["plan"][0]["explanation"]
        self.assertTrue(guided.startswith(concise))
        self.assertGreater(len(guided), len(concise))

    def test_completed_steps_change_accountable_route(self):
        blocked = app.run_pipeline(self.state)["triage"]["tickets"][0]
        self.assertTrue(blocked["onboarding_blocked"])
        self.assertEqual(blocked["owner"], "synthetic-onboarding-team")
        self.assertEqual(blocked["category_owner"], "synthetic-product-team")
        self.request["profile"]["completed_steps"] = ["basics", "privacy", "workspace"]
        ready = app.run_pipeline(self.state)["triage"]["tickets"][0]
        self.assertFalse(ready["onboarding_blocked"])
        self.assertEqual(ready["owner"], "synthetic-product-team")

    def test_rule_priority_and_configuration_order(self):
        self.request["tickets"][0]["text"] = "BILLING automation"
        rule = self.request["routing"]["rules"][1]
        rule["priority"] = "urgent"
        result = app.run_pipeline(self.state)["triage"]
        self.assertEqual(result["tickets"][0]["category"], "billing")
        self.assertEqual(result["tickets"][0]["priority"], "urgent")
        self.assertEqual(set(result["suppressed_tags"]), {"automation", "analytics"})
        rule["priority"] = "high"
        self.assertEqual(app.run_pipeline(self.state)["triage"]["tickets"][0]["category"], "setup")

    def test_default_routing(self):
        self.request["tickets"][0]["text"] = "Unrecognized synthetic request"
        ticket = app.run_pipeline(self.state)["triage"]["tickets"][0]
        self.assertIsNone(ticket["rule_id"])
        self.assertEqual(ticket["owner"], "synthetic-support-team")
        self.assertEqual(ticket["priority"], "normal")

    def test_resolved_tickets_do_not_suppress(self):
        result = app.run_pipeline(self.state)
        self.assertNotIn("analytics", result["triage"]["suppressed_tags"])
        self.request["tickets"][0]["status"] = "resolved"
        result = app.run_pipeline(self.state)
        self.assertFalse(result["triage"]["tickets"][0]["onboarding_blocked"])
        self.assertIn("synthetic-b", [x["id"] for x in result["recommendations"]["items"]])

    def test_exclusions_and_ticket_provenance(self):
        result = app.run_pipeline(self.state)
        omitted = {item["id"]: item for item in result["recommendations"]["excluded_items"]}
        self.assertEqual(omitted["synthetic-b"]["ticket_ids"], ["synthetic-ticket-1"])
        self.assertEqual(omitted["synthetic-c"]["excluded_tags"], ["advertising"])
        self.assertEqual(result["recommendations"]["items"][0]["matched_interests"], {"analytics": 3, "design": 5})

    def test_preference_propagation_and_tie_break(self):
        self.request["profile"]["interests"] = {"design": 1}
        result = app.run_pipeline(self.state)
        self.assertEqual(result["onboarding"]["interests"], result["triage"]["interests"])
        self.assertEqual([x["id"] for x in result["recommendations"]["items"]], ["synthetic-a", "synthetic-d"])
        self.assertEqual([x["score"] for x in result["recommendations"]["items"]], [1, 1])

    def test_zero_limit_and_empty_preferences(self):
        self.request["limit"] = 0
        self.assertEqual(app.run_pipeline(self.state)["recommendations"]["items"], [])
        self.request["limit"] = 3
        self.request["profile"]["interests"] = {}
        self.assertEqual(app.run_pipeline(self.state)["recommendations"]["items"], [])

    def test_empty_tickets_catalog_and_rules(self):
        self.request["tickets"] = []
        self.request["catalog"] = []
        self.request["routing"]["rules"] = []
        result = app.run_pipeline(self.state)
        self.assertEqual(result["triage"]["tickets"], [])
        self.assertEqual(result["recommendations"]["items"], [])

    def test_all_steps_completed(self):
        self.request["profile"]["experience"] = "expert"
        self.request["profile"]["completed_steps"] = list(app.STEPS)
        self.assertEqual(app.adaptive(self.state)["onboarding"]["plan"], [])

    def test_reject_inconsistent_completed_prerequisites(self):
        self.request["profile"]["completed_steps"] = ["workspace"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.state)

    def test_reject_invalid_inputs(self):
        mutations = [
            lambda x: x.update(schema_version=True),
            lambda x: x.update(synthetic=False),
            lambda x: x.update(extra=1),
            lambda x: x["request"].update(limit=True),
            lambda x: x["request"].update(limit=-1),
            lambda x: x["request"]["profile"].update(experience="wizard"),
            lambda x: x["request"]["profile"].update(interests={"design": 0}),
            lambda x: x["request"]["profile"].update(interests={"design": True}),
            lambda x: x["request"]["routing"]["rules"][0].update(owner=" "),
            lambda x: x["request"]["routing"]["rules"][0].update(prerequisites=["unknown"]),
            lambda x: x["request"]["tickets"].append(copy.deepcopy(x["request"]["tickets"][0])),
            lambda x: x["request"]["catalog"][0].update(tags=["design", "design"]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                state = copy.deepcopy(self.state)
                mutation(state)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(state)

    def test_reject_out_of_order_stage(self):
        with self.assertRaises(app.ValidationError):
            app.triage(self.state)
        with self.assertRaises(app.ValidationError):
            app.interests(app.adaptive(self.state))

    def test_reject_tampered_handoffs(self):
        state = app.adaptive(self.state)
        state["onboarding"]["completed_steps"].append("workspace")
        with self.assertRaises(app.ValidationError):
            app.triage(state)
        state = app.triage(app.adaptive(self.state))
        state["triage"]["suppressed_tags"] = {}
        with self.assertRaises(app.ValidationError):
            app.interests(state)

    def test_determinism_and_input_immutability(self):
        original = copy.deepcopy(self.state)
        self.assertEqual(app.run_pipeline(self.state), app.run_pipeline(self.state))
        self.assertEqual(self.state, original)

    def test_cli_success_single_json(self):
        result = self.run_cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in [(), ("nonexistent-synthetic-file.json",), ("example_input.json", "extra")]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        # In-memory fixtures exercise main without leaving extra deliverables.
        for source in ["{", "[]", '{"stage":"input","stage":"triage"}', '{"x":NaN}', '{"stage":"unknown"}']:
            with self.subTest(source=source), mock.patch("builtins.open", mock.mock_open(read_data=source)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
