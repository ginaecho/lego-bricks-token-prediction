import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class JourneyTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_unlocks_personalized_second_step(self):
        before = copy.deepcopy(self.data)
        result = impl.recommend(self.data)
        self.assertEqual([s["action_id"] for s in result["journey"]["steps"]],
                         ["choose-planters", "plant-herbs"])
        self.assertEqual([a["action_id"] for a in result["recommendations"]],
                         ["browse-tools", "choose-planters"])
        self.assertEqual(result["journey"]["steps"][1]["score"], 2)
        self.assertEqual(self.data, before)

    def test_order_invariance_and_no_interests(self):
        self.data["profile"]["interests"] = []
        result = impl.recommend(self.data)
        self.data["actions"].reverse()
        self.assertEqual(result, impl.recommend(self.data))
        self.assertEqual(result["journey"]["steps"][0]["action_id"], "browse-tools")

    def test_empty_catalog(self):
        self.data["actions"] = []
        self.data["profile"]["completed_actions"] = []
        result = impl.recommend(self.data)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["journey"]["status"], "unavailable")
        self.assertEqual(result["journey"]["steps"], [])

    def test_only_one_remaining_returns_no_partial_journey(self):
        self.data["profile"]["completed_actions"] = [
            "orientation", "choose-planters", "plant-herbs"]
        result = impl.recommend(self.data)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["journey"]["steps"], [])

    def test_all_completed(self):
        self.data["profile"]["completed_actions"] = [a["id"] for a in self.data["actions"]]
        self.assertEqual(impl.recommend(self.data)["recommendations"], [])

    def test_invalid_schema_types_and_unknown_fields(self):
        for key, value in [("schema_version", True), ("synthetic", False),
                           ("actions", {}), ("profile", None), ("extra", 1)]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(impl.ValidationError):
                    impl.recommend(data)

    def test_invalid_references_and_completed_history(self):
        for completed in [["missing"], ["plant-herbs"], ["orientation", "orientation"]]:
            with self.subTest(completed=completed):
                self.data["profile"]["completed_actions"] = completed
                with self.assertRaises(impl.ValidationError):
                    impl.recommend(self.data)

    def test_duplicate_ids_unknown_prerequisite_and_cycle(self):
        variants = []
        data = copy.deepcopy(self.data)
        data["actions"].append(copy.deepcopy(data["actions"][0]))
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["actions"][2]["prerequisites"] = ["missing"]
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["actions"][1]["prerequisites"] = ["plant-herbs"]
        variants.append(data)
        for data in variants:
            with self.subTest(actions=data["actions"]):
                with self.assertRaises(impl.ValidationError):
                    impl.recommend(data)

    def test_conjunctive_prerequisites(self):
        self.data["actions"][2]["prerequisites"].append("browse-tools")
        result = impl.recommend(self.data)
        self.assertNotIn("plant-herbs", [s["action_id"] for s in result["journey"]["steps"]])

    def test_output_validator_rejects_invalid_handoff(self):
        result = impl.recommend(self.data)
        result["journey"]["steps"].reverse()
        with self.assertRaises(impl.ValidationError):
            impl.validate_output(result, self.data)
        result = impl.recommend(self.data)
        result["journey"]["steps"][1]["action_id"] = "orientation"
        with self.assertRaises(impl.ValidationError):
            impl.validate_output(result, self.data)

    def test_cli_success(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), impl.recommend(self.data))

    def test_cli_file_and_usage_errors(self):
        for args in [[], [str(ROOT / "nonexistent.json")], ["a", "b"]]:
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    capture_output=True, text=True, check=False, cwd=ROOT)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_malformed_json_and_validation_errors_without_extra_files(self):
        for raw in ['{', '{"schema_version":1,"schema_version":1}', 'NaN', '[]',
                    '{"schema_version":true}', '\ufeff{}']:
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=raw), redirect_stdout(output):
                    code = impl.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_bound_and_invalid_topic(self):
        self.data["actions"][0]["topics"] = [False]
        with self.assertRaises(impl.ValidationError):
            impl.recommend(self.data)
        self.data["actions"] = [
            {"id": str(i), "title": "Synthetic action", "topics": [], "prerequisites": []}
            for i in range(101)]
        with self.assertRaises(impl.ValidationError):
            impl.recommend(self.data)


if __name__ == "__main__":
    unittest.main()
