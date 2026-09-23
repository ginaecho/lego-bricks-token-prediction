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
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_stages(self):
        return app.run_pipeline(self.request)["stages"]

    def test_grounded_faq(self):
        faq = self.run_stages()["faq"]
        self.assertEqual(faq["status"], "answered")
        self.assertEqual(faq["citations"], ["travel", "returns"])
        self.assertEqual(faq["answer"], " ".join(e["text"] for e in faq["evidence"]))

    def test_abstention(self):
        self.request["question"] = "Does it support quantum teleportation?"
        stages = self.run_stages()
        self.assertEqual(stages["faq"]["status"], "abstained")
        self.assertEqual(stages["faq"]["citations"], [])
        self.assertEqual(stages["compare"]["faq_status"], "abstained")
        self.assertEqual(stages["compare"]["ranking"][0], "beacon")

    def test_token_boundaries(self):
        self.request["question"] = "travelogue"
        self.assertEqual(self.run_stages()["faq"]["status"], "abstained")

    def test_case_and_punctuation(self):
        self.request["question"] = "ESIM?! RETURNS!"
        self.assertEqual(self.run_stages()["faq"]["citations"], ["travel", "returns"])

    def test_normalization(self):
        rows = self.run_stages()["compare"]["rows"]
        self.assertEqual(rows[0]["weight_g"], 180)
        self.assertEqual(rows[1]["storage_gb"], 256)
        self.assertEqual(rows[0]["battery_mah"], 4000)

    def test_side_by_side_and_exclusion(self):
        compare = self.run_stages()["compare"]
        self.assertEqual(len(compare["columns"]), 4)
        self.assertEqual(len(compare["rows"]), 3)
        self.assertEqual(compare["ranking"], ["beacon", "atlas"])
        self.assertEqual(compare["rows"][2]["exclusions"], ["insufficient_storage"])

    def test_ranking_preferences_propagate(self):
        self.request["preferences"]["priorities"] = ["price"]
        stages = self.run_stages()
        self.assertEqual(stages["compare"]["ranking"][0], "atlas")
        self.assertEqual(stages["journey"]["product_id"], "atlas")
        self.assertEqual(stages["adaptive"]["product_id"], "atlas")

    def test_budget_boundary(self):
        self.request["preferences"]["budget_usd"] = 480
        self.assertEqual(self.run_stages()["compare"]["ranking"], ["atlas"])

    def test_no_eligible_products(self):
        self.request["preferences"]["budget_usd"] = 1
        stages = self.run_stages()
        self.assertEqual(stages["compare"]["status"], "no_match")
        self.assertEqual(stages["journey"]["status"], "blocked")
        self.assertEqual(stages["adaptive"]["status"], "blocked")
        self.assertEqual(stages["adaptive"]["steps"], [])

    def test_product_subset_propagates(self):
        self.request["product_ids"] = ["atlas"]
        stages = self.run_stages()
        self.assertEqual(stages["faq"]["product_ids"], ["atlas"])
        self.assertEqual(stages["compare"]["ranking"], ["atlas"])
        self.assertEqual(stages["adaptive"]["product_id"], "atlas")

    def test_citations_propagate_all_stages(self):
        stages = self.run_stages()
        for name in app.STAGE_NAMES:
            self.assertEqual(stages[name]["citations"], stages["faq"]["citations"])

    def test_two_step_journey(self):
        journey = self.run_stages()["journey"]
        self.assertEqual([s["action_id"] for s in journey["steps"]],
                         ["choose_product", "learn_basics"])
        self.assertEqual([s["action_id"] for s in journey["next_actions"]], ["choose_product"])
        self.assertEqual(journey["completed_after_journey"], ["choose_product", "learn_basics"])

    def test_expert_skips_optional_basics(self):
        self.request["profile"]["experience"] = "expert"
        stages = self.run_stages()
        self.assertEqual(stages["journey"]["steps"][1]["action_id"], "create_account")
        self.assertNotIn("learn_basics", [s["action_id"] for s in stages["adaptive"]["steps"]])

    def test_adaptive_interests_and_explanations(self):
        adaptive = self.run_stages()["adaptive"]
        ids = [step["action_id"] for step in adaptive["steps"]]
        self.assertIn("enable_travel", ids)
        self.assertIn("explore_camera", ids)
        self.assertNotIn("enable_backup", ids)
        for step in adaptive["steps"]:
            self.assertIn("novice", step["explanation"])
            self.assertIn("Prerequisites:", step["explanation"])

    def test_onboarding_uses_simulated_journey_completion(self):
        stages = self.run_stages()
        self.assertEqual(stages["adaptive"]["starting_completed_actions"],
                         stages["journey"]["completed_after_journey"])
        app.validate_sequence(stages["adaptive"]["steps"],
                              stages["journey"]["completed_after_journey"], "beacon")

    def test_already_complete(self):
        self.request["profile"]["completed_actions"] = list(app.ACTIONS)
        stages = self.run_stages()
        self.assertEqual(stages["journey"]["status"], "exhausted")
        self.assertEqual(stages["journey"]["steps"], [])
        self.assertEqual(stages["adaptive"]["status"], "complete")

    def test_one_remaining_action(self):
        self.request["profile"]["completed_actions"] = [
            a for a in app.ACTIONS if a != "enable_travel"]
        journey = self.run_stages()["journey"]
        self.assertEqual(journey["status"], "exhausted")
        self.assertEqual(len(journey["steps"]), 1)

    def test_no_interests(self):
        self.request["profile"]["interests"] = []
        self.assertEqual(self.run_stages()["adaptive"]["steps"][-1]["action_id"], "set_preferences")

    def test_invalid_input_shapes(self):
        for value in (None, [], {}, "input"):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(value)

    def test_invalid_numbers(self):
        for value in (True, 0, -1, float("nan"), float("inf"), "700", None):
            with self.subTest(value=value):
                self.request["preferences"]["budget_usd"] = value
                with self.assertRaises(app.ValidationError):
                    self.run_stages()

    def test_invalid_enums_duplicates_and_empty(self):
        for key, value in (("product_ids", ["unknown"]), ("product_ids", []),
                           ("product_ids", ["atlas", "atlas"]), ("question", " "),
                           ("schema_version", True), ("synthetic_fixture", False)):
            with self.subTest(key=key, value=value):
                invalid = copy.deepcopy(self.request)
                invalid[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(invalid)

    def test_unknown_fields(self):
        self.request["unexpected"] = 1
        with self.assertRaises(app.ValidationError):
            self.run_stages()

    def test_invalid_completed_prerequisites(self):
        self.request["profile"]["completed_actions"] = ["configure_device"]
        with self.assertRaises(app.ValidationError):
            self.run_stages()

    def test_deterministic_and_does_not_mutate_input(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(app.run_pipeline(self.request), app.run_pipeline(self.request))
        self.assertEqual(original, self.request)

    def test_reject_forged_grounding(self):
        output = self.run_stages()["faq"]
        output["answer"] = "Unsupported warranty promise"
        with self.assertRaises(app.ValidationError):
            app.validate_stage("faq", output, self.request, None)

    def test_reject_forged_ranking(self):
        stages = self.run_stages()
        stages["compare"]["ranking"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate_stage("compare", stages["compare"], self.request, stages["faq"])

    def test_reject_broken_prerequisite_sequence(self):
        step = app.action_record("configure_device", self.request["profile"], "atlas")
        with self.assertRaises(app.ValidationError):
            app.validate_sequence([step], [], "atlas")

    def test_reject_broken_handoff(self):
        stages = self.run_stages()
        stages["adaptive"]["starting_completed_actions"] = []
        with self.assertRaises(app.ValidationError):
            app.validate_stage("adaptive", stages["adaptive"], self.request, stages["journey"])

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["stage_order"], list(app.STAGE_NAMES))

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "nonexistent.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_wrong_argument_count(self):
        for args in ([], ["a", "b"]):
            with self.subTest(args=args), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(app.main(args), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_bad_json_and_file_encoding(self):
        for text in ("{", '{"a":1,"a":2}', '{"budget": NaN}', '{"budget": Infinity}',
                     "[]", "{}"):
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(app.main(["fixture.json"]), 2)
                    self.assertEqual(json.loads(output.getvalue())["status"], "error")
        with patch.object(Path, "read_text", side_effect=UnicodeError("bad encoding")):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
