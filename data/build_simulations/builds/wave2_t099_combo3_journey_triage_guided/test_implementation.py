import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))
        self.req = self.data["request"]

    def run_stages(self):
        return app.run_pipeline(self.data)["stages"]

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_personalized_two_step_journey(self):
        journey = self.run_stages()["journey"]
        self.assertEqual([s["action_id"] for s in journey["steps"]], ["verify", "list"])
        self.assertEqual(journey["eligible_next_actions"], ["verify", "billing"])
        self.assertEqual(journey["steps"][0]["matched_goals"], ["selling"])

    def test_goal_change_changes_both_handoffs(self):
        self.req["user"]["goals"] = ["payments"]
        self.req["setup_events"] = []
        stages = self.run_stages()
        for name in ("triage", "guided"):
            self.assertEqual(stages[name]["journey_action_ids"], ["billing", "payout"])

    def test_selected_journey_drives_context_rule(self):
        self.req["ticket"]["text"] = "Please assist"
        triage = self.run_stages()["triage"]
        self.assertEqual(triage["route"]["category"], "verification")
        self.assertEqual(triage["matched_rule"], 1)

    def test_highest_priority_matching_rule_wins(self):
        triage = self.run_stages()["triage"]
        self.assertEqual(triage["route"]["priority"], "high")
        self.assertEqual(triage["matched_rule"], 0)

    def test_equal_priority_uses_configuration_order(self):
        self.req["routing"]["rules"][1]["priority"] = "high"
        self.assertEqual(self.run_stages()["triage"]["matched_rule"], 0)

    def test_case_insensitive_keyword(self):
        self.req["ticket"]["text"] = "PAYOUT"
        self.assertEqual(self.run_stages()["triage"]["matched_rule"], 0)

    def test_default_route(self):
        self.req["routing"]["rules"] = []
        triage = self.run_stages()["triage"]
        self.assertIsNone(triage["matched_rule"])
        self.assertEqual(triage["route"]["category"], "general")

    def test_guided_dependency_closure_and_progress(self):
        guided = self.run_stages()["guided"]
        steps = {s["action_id"]: s for s in guided["steps"]}
        self.assertEqual(steps["payout"]["state"], "blocked")
        self.assertEqual(steps["payout"]["missing_prerequisites"], ["billing"])
        self.assertEqual(steps["billing"]["state"], "ready")
        self.assertEqual(steps["list"]["state"], "ready")
        self.assertEqual(guided["progress"], {"completed": 2, "total": 5, "percent": 40.0})

    def test_accountability_and_ticket_propagation(self):
        stages = self.run_stages()
        self.assertEqual(stages["guided"]["ticket_id"], self.req["ticket"]["id"])
        self.assertEqual(stages["guided"]["user_id"], self.req["user"]["id"])
        self.assertEqual(stages["guided"]["priority"], "high")
        for step in stages["guided"]["steps"]:
            self.assertEqual(step["owner"], stages["triage"]["route"]["owner"])
            self.assertEqual(step["team"], stages["triage"]["route"]["team"])

    def test_complete_plan(self):
        self.req["setup_events"] = ["verify", "list", "billing", "payout"]
        guided = self.run_stages()["guided"]
        self.assertEqual(guided["status"], "complete")
        self.assertEqual(guided["progress"]["percent"], 100)
        self.assertEqual(guided["next_actions"], [])

    def test_prerequisites_enforced_for_events(self):
        self.req["setup_events"] = ["payout"]
        self.invalid()

    def test_outside_plan_event_rejected(self):
        self.req["routing"]["rules"] = []
        self.req["setup_events"] = ["billing"]
        self.invalid()

    def test_duplicate_event_rejected(self):
        self.req["setup_events"] = ["verify", "verify"]
        self.invalid()

    def test_repeated_completed_action_rejected(self):
        self.req["setup_events"] = ["register"]
        self.invalid()

    def test_cycle_rejected(self):
        self.req["actions"][0]["prerequisites"] = ["list"]
        self.invalid()

    def test_unknown_dependency_rejected(self):
        self.req["actions"][1]["prerequisites"] = ["missing"]
        self.invalid()

    def test_unknown_route_action_rejected(self):
        self.req["routing"]["default"]["setup_actions"] = ["missing"]
        self.invalid()

    def test_completed_requires_ancestor_closure(self):
        self.req["user"]["completed"] = ["verify"]
        self.invalid()

    def test_insufficient_remaining_journey_rejected(self):
        self.req["user"]["completed"] = ["register", "verify", "list", "billing"]
        self.req["setup_events"] = []
        self.invalid()

    def test_all_completed_rejected(self):
        self.req["user"]["completed"] = [a["id"] for a in self.req["actions"]]
        self.req["setup_events"] = []
        self.invalid()

    def test_no_goals_still_deterministic(self):
        self.req["user"]["goals"] = []
        self.assertEqual(self.run_stages()["journey"]["steps"][0]["action_id"], "verify")

    def test_input_not_mutated_and_repeatable(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))
        self.assertEqual(self.data, before)

    def test_unknown_field_rejected(self):
        self.req["surprise"] = True
        self.invalid()

    def test_boolean_weight_rejected(self):
        self.req["actions"][0]["weight"] = True
        self.invalid()

    def test_blank_owner_rejected(self):
        self.req["routing"]["default"]["owner"] = " "
        self.invalid()

    def test_invalid_priority_rejected(self):
        self.req["routing"]["default"]["priority"] = "critical"
        self.invalid()

    def test_non_synthetic_label_rejected(self):
        self.data["fixture_label"] = "production"
        self.invalid()

    def test_malformed_nested_inputs(self):
        for key in ("user", "actions", "ticket", "routing", "setup_events"):
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.data)
                candidate["request"][key] = None
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(candidate)

    def test_forged_journey_handoff_rejected(self):
        output = app.run_pipeline(self.data)
        output["status"] = "processing"
        output["stages"] = {"journey": output["stages"]["journey"]}
        output["stages"]["journey"]["steps"][0]["action_id"] = "payout"
        with self.assertRaises(app.ValidationError):
            app.validate(output, "journey")

    def test_forged_routing_handoff_rejected(self):
        output = app.run_pipeline(self.data)
        output["status"] = "processing"
        del output["stages"]["guided"]
        output["stages"]["triage"]["route"]["owner"] = "wrong"
        with self.assertRaises(app.ValidationError):
            app.validate(output, "triage")

    def test_forged_progress_rejected(self):
        output = app.run_pipeline(self.data)
        output["stages"]["guided"]["progress"]["percent"] = 100
        with self.assertRaises(app.ValidationError):
            app.validate(output, "guided")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py", "nonexistent.json"],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_wrong_arguments(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py"],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_input(self):
        for raw in ("{", "null", '{"x":1,"x":2}', "NaN", '{"schema_version":1}'):
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_cli_encoding_failure(self):
        with patch.object(Path, "read_text", side_effect=UnicodeError("invalid encoding")):
            stream = io.StringIO()
            with redirect_stdout(stream):
                self.assertEqual(app.main(["fixture.json"]), 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
