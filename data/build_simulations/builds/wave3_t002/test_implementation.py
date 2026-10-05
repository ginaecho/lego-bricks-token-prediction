import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_cli(self, *args):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        return process.returncode, json.loads(process.stdout)

    def test_novice_prerequisites_and_explanations(self):
        result = app.response(self.payload)
        self.assertEqual(result["status"], "ok")
        data = result["data"]
        self.assertEqual(data["next_actions"], ["orientation"])
        self.assertEqual(data["progress"], {"completed": 0, "total": 5, "percent": 0.0})
        catalog = next(step for step in data["steps"] if step["id"] == "catalog")
        self.assertEqual(catalog["blocked_by"], ["orientation"])
        self.assertEqual(catalog["state"], "blocked")
        self.assertIn("Practice", catalog["guidance"])
        self.assertTrue(all(step["explanation"] for step in data["steps"]))

    def test_expert_batch_and_preference(self):
        self.payload["experience"] = "expert"
        self.payload["preferences"] = {"format": "explanation", "pace": "batch"}
        data = app.onboarding(self.payload)
        self.assertEqual(data["next_actions"], ["catalog", "payments"])
        self.assertNotIn("orientation", [step["id"] for step in data["steps"]])
        self.assertTrue(all(step["guidance_level"] == "concise" for step in data["steps"]))
        self.assertIn("Read the rationale", data["steps"][0]["guidance"])

    def test_intermediate_focused(self):
        self.payload["experience"] = "intermediate"
        data = app.onboarding(self.payload)
        self.assertEqual(data["next_actions"], ["catalog"])
        self.assertEqual(data["steps"][0]["guidance_level"], "standard")

    def test_completed_orientation_unlocks_parallel_steps(self):
        self.payload["completed_steps"] = ["orientation"]
        self.payload["preferences"]["pace"] = "batch"
        data = app.onboarding(self.payload)
        self.assertEqual(data["next_actions"], ["catalog", "payments"])
        self.assertEqual(data["progress"]["percent"], 20.0)

    def test_full_completion(self):
        self.payload["goals"] = ["launch_store", "improve_sales"]
        self.payload["completed_steps"] = list(app.CATALOG)
        data = app.onboarding(self.payload)
        self.assertTrue(data["onboarding_complete"])
        self.assertEqual(data["next_actions"], [])
        self.assertEqual(data["progress"]["percent"], 100.0)

    def test_goal_union_is_order_independent_and_topological(self):
        self.payload["goals"] = ["improve_sales", "launch_store"]
        first = app.onboarding(self.payload)
        self.payload["goals"].reverse()
        self.assertEqual(first, app.onboarding(self.payload))
        seen = set()
        for step in first["steps"]:
            self.assertTrue(set(step["prerequisites"]) <= seen)
            seen.add(step["id"])
        self.assertEqual(len(seen), 7)

    def test_completed_step_cannot_bypass_prerequisites(self):
        self.payload["completed_steps"] = ["publish"]
        result = app.response(self.payload)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"][0]["field"], "completed_steps")

    def test_invalid_root_and_fields(self):
        invalid = [None, [], {}, True]
        for key, value in [
            ("schema_version", True), ("experience", "guru"), ("experience", []),
            ("fixture", {"kind": "real", "id": "customer"}),
            ("goals", []), ("goals", ["launch_store", "launch_store"]),
            ("goals", [False]), ("completed_steps", ["unknown"]),
            ("completed_steps", ["orientation", "orientation"]),
            ("preferences", {"format": "video", "pace": "batch"}),
            ("preferences", {"format": "practice", "pace": 1}),
        ]:
            candidate = copy.deepcopy(self.payload)
            candidate[key] = value
            invalid.append(candidate)
        extra = copy.deepcopy(self.payload)
        extra["extra"] = "not allowed"
        invalid.append(extra)
        for payload in invalid:
            with self.subTest(payload=payload):
                result = app.response(payload)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["data"])

    def test_deterministic_and_does_not_mutate(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(app.response(self.payload), app.response(self.payload))
        self.assertEqual(self.payload, original)

    def test_cli_success(self):
        code, result = self.run_cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(result, app.response(self.payload))

    def test_cli_missing_file_usage_and_invalid_schema(self):
        for args in [(), ("missing-synthetic-input.json",),
                     ("example_input.json", "extra"), ("build_manifest.json",)]:
            with self.subTest(args=args):
                code, result = self.run_cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_malformed_encoding_json_and_duplicate_keys(self):
        failures = ["{", '{"schema_version": 1, "schema_version": 1}', "NaN"]
        for text in failures:
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")
        with patch.object(Path, "read_text",
                          side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
