import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_and_handoff(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["id"] for s in result["journey"]["steps"]], ["catalog", "publish"])
        for source, target in zip(result["journey"]["steps"], result["guided"]["steps"]):
            self.assertEqual(source, {k: v for k, v in target.items() if k != "status"})
        self.assertEqual(result["guided"]["goal"], result["journey"]["goal"])
        self.assertEqual(result["guided"]["completed_ids"], ["account", "catalog"])
        self.assertEqual(result["guided"]["progress"]["percent"], 50)

    def test_blocked_and_ready(self):
        self.data["progress_events"] = []
        guided = app.run_pipeline(self.data)["guided"]
        self.assertEqual([s["status"] for s in guided["steps"]], ["ready", "blocked"])
        self.assertEqual(guided["progress"], {"completed": 0, "total": 2, "percent": 0})

    def test_full_progress(self):
        self.data["progress_events"][-1]["status"] = "completed"
        guided = app.run_pipeline(self.data)["guided"]
        self.assertEqual(guided["progress"]["percent"], 100)
        self.assertEqual([s["status"] for s in guided["steps"]], ["completed", "completed"])

    def test_start_then_complete(self):
        self.data["progress_events"].insert(0, {"action_id": "catalog", "status": "in_progress"})
        self.assertEqual(app.run_pipeline(self.data)["guided"]["progress"]["completed"], 1)

    def test_event_prerequisites(self):
        self.data["progress_events"].reverse()
        with self.assertRaisesRegex(app.ValidationError, "unmet prerequisites"):
            app.run_pipeline(self.data)

    def test_outside_journey_event(self):
        self.data["progress_events"] = [{"action_id": "analytics", "status": "completed"}]
        with self.assertRaisesRegex(app.ValidationError, "outside"):
            app.run_pipeline(self.data)

    def test_duplicate_completion_and_start(self):
        for status in ("completed", "in_progress"):
            with self.subTest(status=status):
                self.data["progress_events"] = [{"action_id": "catalog", "status": status}] * 2
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_personalization_and_determinism(self):
        self.data["progress_events"] = []
        self.data["actions"].extend([
            {"id": "a-catalog", "title": "Metrics setup", "prerequisites": ["account"],
             "tags": ["metrics"], "goals": ["launch-store"]},
            {"id": "a-publish", "title": "Metrics launch", "prerequisites": ["a-catalog"],
             "tags": ["metrics"], "goals": ["launch-store"]},
        ])
        self.assertEqual(app.recommend(self.data)["steps"][0]["id"], "catalog")
        self.data["profile"]["interests"] = ["metrics"]
        plan = app.recommend(self.data)
        self.assertEqual([s["id"] for s in plan["steps"]], ["a-catalog", "a-publish"])
        self.data["actions"].reverse()
        self.assertEqual(app.recommend(self.data), plan)

    def test_prerequisite_bridge(self):
        self.data["actions"][1]["goals"] = []
        self.data["progress_events"] = []
        self.assertEqual([s["id"] for s in app.recommend(self.data)["steps"]], ["catalog", "publish"])

    def test_tampered_handoff(self):
        plan = app.recommend(self.data)
        for key, value in (("goal", "other"), ("initial_completed", []), ("schema_version", True)):
            tampered = copy.deepcopy(plan)
            tampered[key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.guided_setup(self.data, tampered)
        plan["steps"][1]["prerequisites"] = []
        with self.assertRaises(app.ValidationError):
            app.guided_setup(self.data, plan)

    def test_cycle_and_missing_prerequisite(self):
        for dependency in ("publish", "missing"):
            with self.subTest(dependency=dependency):
                self.data["actions"][1]["prerequisites"] = [dependency]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_completed_context_validation(self):
        for completed in (["missing"], ["catalog"], ["account", "account"]):
            with self.subTest(completed=completed):
                self.data["profile"]["completed"] = completed
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_no_two_step_journey(self):
        self.data["profile"]["completed"] = ["account", "catalog"]
        self.data["actions"] = self.data["actions"][:3]
        with self.assertRaisesRegex(app.ValidationError, "no feasible"):
            app.recommend(self.data)

    def test_invalid_schema(self):
        for key, value in (("schema_version", True), ("actions", {}), ("profile", None),
                           ("fixture_label", "real"), ("progress_events", "invalid")):
            invalid = copy.deepcopy(self.data)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.run_pipeline(invalid)
        self.data["unknown"] = 1
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_duplicate_id_and_invalid_event(self):
        self.data["actions"].append(copy.deepcopy(self.data["actions"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate action"):
            app.run_pipeline(self.data)
        self.data["actions"].pop()
        self.data["progress_events"][0]["status"] = "skipped"
        with self.assertRaisesRegex(app.ValidationError, "event status"):
            app.run_pipeline(self.data)

    def test_input_is_not_mutated(self):
        original = copy.deepcopy(self.data)
        app.run_pipeline(self.data)
        self.assertEqual(self.data, original)

    def cli(self, *arguments):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), app.run_pipeline(self.data))
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in ((), ("does-not-exist.json",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_bad_files(self):
        path = ROOT / "_invalid_test_input.json"
        try:
            for content in (b"{", b"[]", b'{"a":1,"a":2}', b"NaN", b"\xff",
                            json.dumps({**self.data, "schema_version": False}).encode()):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    process = self.cli(str(path))
                    self.assertEqual(process.returncode, 2, process.stderr)
                    self.assertEqual(json.loads(process.stdout)["status"], "error")
                    self.assertEqual(process.stderr, "")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
