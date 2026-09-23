import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from implementation import ValidationError, main, run_pipeline


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def preferences(self):
        return self.data["onboarding"]["completed_steps"][2]["answers"]

    def test_integrated_success_and_grounding(self):
        output = run_pipeline(self.data)
        self.assertEqual(output["onboarding"]["progress_percent"], 100)
        discovery = output["discovery"]
        self.assertEqual(discovery["user_id"], "synthetic-user-001")
        self.assertEqual(discovery["excluded_count"], 2)
        self.assertEqual(discovery["unmatched_count"], 1)
        first = discovery["recommendations"][0]
        self.assertEqual(first["score"], 5)
        self.assertEqual(first["explanation"]["matches"], [
            {"interest": "science", "weight": 3, "sources": ["category", "tags"]},
            {"interest": "art", "weight": 2, "sources": ["tags"]},
        ])

    def test_empty_onboarding(self):
        self.data["onboarding"]["completed_steps"] = []
        output = run_pipeline(self.data)
        self.assertEqual(output["onboarding"]["next_step"], "profile")
        self.assertEqual(output["onboarding"]["progress_percent"], 0)
        self.assertEqual(output["discovery"]["reason"], "onboarding_incomplete")

    def test_partial_progress(self):
        self.data["onboarding"]["completed_steps"] = self.data["onboarding"]["completed_steps"][:1]
        output = run_pipeline(self.data)
        self.assertEqual(output["onboarding"]["next_step"], "consent")
        self.assertEqual(output["onboarding"]["progress_percent"], 33.33)
        self.assertEqual(output["onboarding"]["steps"][2]["status"], "locked")

    def test_prerequisite_order(self):
        self.data["onboarding"]["completed_steps"].reverse()
        with self.assertRaisesRegex(ValidationError, "prerequisite"):
            run_pipeline(self.data)

    def test_consent_blocking(self):
        self.data["onboarding"]["completed_steps"] = self.data["onboarding"]["completed_steps"][:2]
        self.data["onboarding"]["completed_steps"][1]["answers"]["personalized_discovery"] = False
        output = run_pipeline(self.data)
        self.assertEqual(output["onboarding"]["status"], "blocked")
        self.assertIsNone(output["onboarding"]["next_step"])
        self.assertEqual(output["discovery"]["reason"], "consent_required")

    def test_cannot_submit_preferences_without_consent(self):
        self.data["onboarding"]["completed_steps"][1]["answers"]["personalized_discovery"] = False
        with self.assertRaisesRegex(ValidationError, "consent required"):
            run_pipeline(self.data)

    def test_preference_handoff_changes_ranking(self):
        self.preferences()["ranked_interests"] = ["music", "art", "science"]
        output = run_pipeline(self.data)["discovery"]
        # Both score 3, so stable item-id ordering applies.
        self.assertEqual(output["recommendations"][1]["score"], 3)
        self.preferences()["ranked_interests"] = ["music", "science"]
        output = run_pipeline(self.data)["discovery"]
        self.assertEqual(output["recommendations"][0]["item_id"], "synthetic-02")
        self.assertEqual(output["applied_preferences"], self.preferences())

    def test_exclusions_override_matches(self):
        self.preferences()["excluded_categories"].append("science")
        self.preferences()["excluded_item_ids"].append("synthetic-02")
        output = run_pipeline(self.data)["discovery"]
        self.assertEqual(output["recommendations"], [])
        self.assertEqual(output["excluded_count"], 4)

    def test_limit_and_ties_are_deterministic(self):
        duplicate = copy.deepcopy(self.data["discovery"]["catalog"][0])
        duplicate["id"] = "synthetic-00"
        self.data["discovery"]["catalog"].append(duplicate)
        self.data["discovery"]["limit"] = 1
        output = run_pipeline(self.data)["discovery"]
        self.assertEqual(output["recommendations"][0]["item_id"], "synthetic-00")
        self.assertEqual(output["eligible_count"], 3)
        self.data["discovery"]["catalog"].reverse()
        self.assertEqual(run_pipeline(self.data)["discovery"], output)

    def test_empty_catalog_and_unknown_interest(self):
        self.data["discovery"]["catalog"] = []
        self.assertEqual(run_pipeline(self.data)["discovery"]["recommendations"], [])
        self.setUp()
        self.preferences()["ranked_interests"] = ["unknown-interest"]
        self.assertEqual(run_pipeline(self.data)["discovery"]["eligible_count"], 0)

    def test_bad_preferences(self):
        for invalid in ([], ["science", "science"], [" science"], [1], "science"):
            with self.subTest(invalid=invalid):
                self.preferences()["ranked_interests"] = invalid
                with self.assertRaises(ValidationError):
                    run_pipeline(self.data)

    def test_invalid_types_and_fields(self):
        for key, value in (("schema_version", True), ("fixture_label", "real"), ("extra", 1)):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(ValidationError):
                    run_pipeline(data)
        for limit in (0, 101, True, "3", 2.5):
            self.data["discovery"]["limit"] = limit
            with self.assertRaises(ValidationError):
                run_pipeline(self.data)

    def test_duplicate_catalog_and_steps(self):
        self.data["discovery"]["catalog"].append(self.data["discovery"]["catalog"][0])
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            run_pipeline(self.data)
        self.setUp()
        self.data["onboarding"]["completed_steps"][1] = self.data["onboarding"]["completed_steps"][0]
        with self.assertRaises(ValidationError):
            run_pipeline(self.data)

    def test_no_input_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(run_pipeline(self.data), run_pipeline(self.data))
        self.assertEqual(self.data, before)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), run_pipeline(self.data))

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["one", "two"]):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    capture_output=True, text=True, cwd=ROOT, check=False)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for payload in ('{', '{"a":1,"a":2}', 'NaN', 'null', '{"schema_version":1}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("implementation.Path.read_text", return_value=payload):
                    with contextlib.redirect_stdout(output):
                        code = main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_decode_and_file_errors(self):
        for error in (PermissionError("denied"), UnicodeError("invalid encoding")):
            output = io.StringIO()
            with patch("implementation.Path.read_text", side_effect=error):
                with contextlib.redirect_stdout(output):
                    code = main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
