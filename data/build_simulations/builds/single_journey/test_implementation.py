import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, _unique_object, recommend


ROOT = Path(__file__).resolve().parent


def fixture(label):
    """Labeled, hand-authored expected scenarios; no provider evaluation."""
    payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
    if label == "new_customer":
        payload["user"]["completed_stages"] = []
    elif label == "returning_customer":
        pass
    elif label == "completed_journey":
        payload["user"]["completed_stages"] = ["discover", "purchase", "onboard"]
    elif label == "excluded_checkout":
        payload["user"]["excluded_actions"] = ["checkout"]
    elif label == "future_exclusion":
        payload["user"]["excluded_actions"] = ["setup"]
    elif label == "preference_ranking":
        payload["journey"]["actions"].append({
            "id": "consult", "name": "Consult buying guide", "stage_id": "purchase",
            "tags": ["expert"], "priority": 3,
        })
        payload["user"]["preferences"] = ["expert"]
    elif label == "action_dependency":
        payload["journey"]["actions"].append({
            "id": "address", "name": "Add delivery address", "stage_id": "purchase",
            "prerequisites": ["checkout"],
        })
    else:
        raise AssertionError(f"Unknown fixture label: {label}")
    return payload


class JourneyTests(unittest.TestCase):
    def test_labeled_next_action_and_plan_fixtures(self):
        cases = [
            ("new_customer", "ready", ["compare"], ["compare", "checkout"], True),
            ("returning_customer", "ready", ["checkout"], ["checkout", "setup"], True),
            ("completed_journey", "completed", [], [], True),
            ("excluded_checkout", "blocked", [], [], False),
            ("future_exclusion", "ready", ["checkout"], ["checkout"], False),
            ("preference_ranking", "ready", ["consult", "checkout"], ["consult", "checkout"], True),
            ("action_dependency", "ready", ["checkout"], ["checkout", "address"], True),
        ]
        for label, status, expected_next, expected_plan, possible in cases:
            with self.subTest(label=label):
                result = recommend(fixture(label))
                self.assertEqual(result["status"], status)
                self.assertEqual([a["action_id"] for a in result["next_actions"]], expected_next)
                self.assertEqual([a["action_id"] for a in result["two_step_plan"]], expected_plan)
                self.assertEqual(result["progression_possible"], possible)

    def test_explanations_and_hypothetical_second_step(self):
        result = recommend(fixture("returning_customer"))
        first, second = result["two_step_plan"]
        self.assertEqual(first["prerequisite_explanation"]["completed_required_stages"], ["discover"])
        self.assertEqual(second["prerequisite_explanation"]["completed_required_stages"], ["purchase"])
        self.assertEqual(second["prerequisite_explanation"]["completed_required_actions"], ["checkout"])
        self.assertEqual(second["assumed_completed_plan_actions"], ["checkout"])
        self.assertEqual(first["assumed_completed_plan_actions"], [])
        self.assertEqual(result["completed_stages"], ["discover"])

    def test_no_prerequisite_explanation(self):
        action = recommend(fixture("new_customer"))["next_actions"][0]
        self.assertEqual(action["prerequisite_explanation"]["text"], "No prerequisites.")

    def test_completed_actions_advance_stage(self):
        payload = fixture("returning_customer")
        payload["user"]["completed_actions"] = ["checkout"]
        result = recommend(payload)
        self.assertEqual(result["completed_stages"], ["discover", "purchase"])
        self.assertEqual([a["action_id"] for a in result["next_actions"]], ["setup"])

    def test_completed_stage_implies_its_actions(self):
        payload = fixture("returning_customer")
        payload["user"]["completed_stages"].append("purchase")
        self.assertEqual(recommend(payload)["next_actions"][0]["action_id"], "setup")

    def test_all_actions_needed_to_complete_stage(self):
        payload = fixture("preference_ranking")
        result = recommend(payload)
        self.assertNotIn("setup", [item["action_id"] for item in result["two_step_plan"]])
        payload["user"]["completed_actions"] = ["consult", "checkout"]
        self.assertEqual(recommend(payload)["next_actions"][0]["action_id"], "setup")

    def test_exclusion_blocks_descendants(self):
        result = recommend(fixture("excluded_checkout"))
        self.assertEqual(result["unreachable_stages"], ["onboard", "purchase"])
        self.assertEqual(result["blocked_actions"][0]["reasons"],
                         [{"kind": "excluded_action", "id": "checkout"}])

    def test_excluded_tag_overrides_preferences(self):
        payload = fixture("returning_customer")
        payload["user"]["excluded_tags"] = ["self_service"]
        result = recommend(payload)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["next_actions"], [])

    def test_completed_excluded_action_remains_complete(self):
        payload = fixture("excluded_checkout")
        payload["user"]["completed_actions"] = ["checkout"]
        self.assertEqual(recommend(payload)["next_actions"][0]["action_id"], "setup")

    def test_deterministic_ties_independent_of_input_order(self):
        payload = fixture("preference_ranking")
        payload["user"]["preferences"] = []
        for action in payload["journey"]["actions"]:
            action["priority"] = 0
        first = recommend(payload)
        payload["journey"]["actions"].reverse()
        payload["journey"]["stages"].reverse()
        self.assertEqual(first, recommend(payload))
        self.assertEqual([a["action_id"] for a in first["next_actions"]], ["checkout", "consult"])

    def test_preference_scores(self):
        result = recommend(fixture("preference_ranking"))
        self.assertEqual(result["next_actions"][0]["score"], 13)
        self.assertEqual(result["next_actions"][0]["matched_preferences"], ["expert"])

    def test_input_not_mutated(self):
        payload = fixture("returning_customer")
        original = copy.deepcopy(payload)
        recommend(payload)
        self.assertEqual(payload, original)

    def test_unknown_reference_fixtures(self):
        cases = [
            ("stage_prerequisite", lambda p: p["journey"]["stages"][0].update(prerequisites=["unknown"])),
            ("action_prerequisite", lambda p: p["journey"]["actions"][0].update(prerequisites=["unknown"])),
            ("action_stage", lambda p: p["journey"]["actions"][0].update(stage_id="unknown")),
            ("completed_stage", lambda p: p["user"].update(completed_stages=["unknown"])),
            ("completed_action", lambda p: p["user"].update(completed_actions=["unknown"])),
            ("excluded_action", lambda p: p["user"].update(excluded_actions=["unknown"])),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                payload = fixture("returning_customer")
                mutate(payload)
                with self.assertRaisesRegex(ValidationError, "unknown references"):
                    recommend(payload)

    def test_cycle_fixtures(self):
        cases = [
            ("stage_cycle", lambda p: p["journey"]["stages"][0].update(prerequisites=["onboard"])),
            ("action_self_cycle", lambda p: p["journey"]["actions"][0].update(prerequisites=["compare"])),
            ("mixed_cycle", lambda p: p["journey"]["actions"][0].update(prerequisites=["setup"])),
            ("stage_self_cycle", lambda p: p["journey"]["stages"][0].update(prerequisites=["discover"])),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                payload = fixture("returning_customer")
                mutate(payload)
                with self.assertRaisesRegex(ValidationError, "cyclic"):
                    recommend(payload)

    def test_two_action_cycle(self):
        payload = fixture("action_dependency")
        payload["journey"]["actions"][1]["prerequisites"] = ["address"]
        with self.assertRaisesRegex(ValidationError, "cyclic"):
            recommend(payload)

    def test_invalid_schema_fixtures(self):
        cases = [
            ("duplicate_action", lambda p: p["journey"]["actions"].append(copy.deepcopy(p["journey"]["actions"][0]))),
            ("duplicate_stage", lambda p: p["journey"]["stages"].append(copy.deepcopy(p["journey"]["stages"][0]))),
            ("empty_stage", lambda p: p["journey"]["actions"].pop()),
            ("empty_stages", lambda p: p["journey"].update(stages=[])),
            ("wrong_list_type", lambda p: p["user"].update(preferences="self_service")),
            ("duplicate_list_member", lambda p: p["user"].update(preferences=["x", "x"])),
            ("bad_list_member", lambda p: p["user"].update(preferences=[3])),
            ("blank_id", lambda p: p["journey"]["actions"][0].update(id=" ")),
            ("unknown_field", lambda p: p["user"].update(prefrences=[])),
            ("missing_field", lambda p: p["user"].pop("completed_stages")),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                payload = fixture("returning_customer")
                mutate(payload)
                with self.assertRaises(ValidationError):
                    recommend(payload)

    def test_invalid_priorities(self):
        for priority in (True, "1", None, float("nan"), float("inf"), -float("inf"), 1_000_001):
            with self.subTest(priority=priority):
                payload = fixture("returning_customer")
                payload["journey"]["actions"][0]["priority"] = priority
                with self.assertRaises(ValidationError):
                    recommend(payload)

    def test_optional_user_fields(self):
        payload = fixture("new_customer")
        payload["user"] = {"completed_stages": []}
        self.assertEqual(recommend(payload)["next_actions"][0]["action_id"], "compare")

    def test_callback_accepts_known_plan_action(self):
        result = recommend(fixture("returning_customer"), lambda result: {
            "explanations": [{"action_id": "setup", "text": "Available after the first plan step."}]
        })
        self.assertEqual(result["explanation_status"], "accepted")
        self.assertEqual(result["supplemental_explanations"][0]["action_id"], "setup")

    def test_callback_rejects_invented_or_nonrecommended_ids(self):
        for label, action_id in (("invented", "magic_discount"), ("already_completed", "compare")):
            with self.subTest(label=label):
                result = recommend(fixture("returning_customer"), lambda result: {
                    "explanations": [{"action_id": action_id, "text": "Do this."}]
                })
                self.assertEqual(result["explanation_status"], "rejected")
                self.assertEqual(result["supplemental_explanations"], [])
                self.assertEqual(result["next_actions"][0]["action_id"], "checkout")

    def test_callback_rejects_entire_mixed_response(self):
        result = recommend(fixture("returning_customer"), lambda result: {
            "explanations": [
                {"action_id": "checkout", "text": "Known action."},
                {"action_id": "invented", "text": "Unknown action."},
            ]
        })
        self.assertEqual(result["explanation_status"], "rejected")
        self.assertEqual(result["supplemental_explanations"], [])

    def test_callback_malformed_fixtures(self):
        entry = {"action_id": "checkout", "text": "Proceed."}
        cases = [
            None, {}, {"explanations": "bad"}, {"explanations": [entry, entry]},
            {"explanations": [{"action_id": "checkout", "text": ""}]},
            {"explanations": [{"action_id": [], "text": "bad"}]},
            {"explanations": [dict(entry, injected=True)]},
        ]
        for response in cases:
            with self.subTest(response=response):
                result = recommend(fixture("returning_customer"), lambda result: response)
                self.assertEqual(result["explanation_status"], "rejected")

    def test_callback_exception_falls_back(self):
        def broken(result):
            raise RuntimeError("Provider unavailable")
        result = recommend(fixture("returning_customer"), broken)
        self.assertEqual(result["explanation_status"], "rejected")
        self.assertEqual(result["next_actions"][0]["action_id"], "checkout")

    def test_callback_cannot_modify_ranking(self):
        def malicious(result):
            result["next_actions"].clear()
            result["two_step_plan"][0]["action_id"] = "invented"
            return {"explanations": []}
        result = recommend(fixture("returning_customer"), malicious)
        self.assertEqual(result["next_actions"][0]["action_id"], "checkout")
        self.assertEqual(result["two_step_plan"][0]["action_id"], "checkout")

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate JSON key"):
            json.loads('{"user": 1, "user": 2}', object_pairs_hook=_unique_object)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        result = json.loads(process.stdout)
        self.assertEqual(result["status"], "ready")
        self.assertEqual([a["action_id"] for a in result["two_step_plan"]], ["checkout", "setup"])

    def test_cli_missing_file(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "missing_fixture.json")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, "")
        self.assertIn("error", json.loads(process.stderr))


if __name__ == "__main__":
    unittest.main()
