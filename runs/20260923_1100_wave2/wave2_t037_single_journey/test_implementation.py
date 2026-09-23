import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class JourneyTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_personalized_chain(self):
        result = app.recommend(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["id"] for s in result["journey"]["steps"]],
                         ["seller-profile", "first-listing"])
        self.assertEqual(result["journey"]["total_minutes"], 25)
        self.assertTrue(result["journey"]["validated"])
        self.assertIn("seller-profile", result["journey"]["steps"][1]["completed_before"])
        self.assertNotIn("first-listing", [s["id"] for s in result["recommendations"]])
        self.assertEqual(result["blocked"][0]["missing_prerequisites"], ["seller-profile"])

    def test_deterministic_and_nonmutating(self):
        original = copy.deepcopy(self.data)
        expected = app.recommend(self.data)
        self.assertEqual(original, self.data)
        self.data["actions"].reverse()
        self.data["profile"]["goals"].reverse()
        self.assertEqual(expected, app.recommend(self.data))

    def test_zero_budget(self):
        self.data["profile"]["available_minutes"] = 0
        result = app.recommend(self.data)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["journey"]["status"], "unavailable")
        self.assertFalse(result["journey"]["validated"])

    def test_empty_catalog(self):
        self.data["actions"] = []
        self.data["profile"]["completed"] = []
        result = app.recommend(self.data)
        self.assertEqual(result["blocked"], [])
        self.assertEqual(result["journey"]["steps"], [])

    def test_all_completed(self):
        self.data["profile"]["completed"] = [x["id"] for x in self.data["actions"]]
        result = app.recommend(self.data)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["journey"]["status"], "unavailable")

    def test_one_action_fits_not_pair(self):
        self.data["profile"]["available_minutes"] = 10
        result = app.recommend(self.data)
        self.assertEqual(len(result["recommendations"]), 2)
        self.assertEqual(result["journey"]["steps"], [])

    def test_no_goals_uses_shortest_feasible_pair(self):
        self.data["profile"]["goals"] = []
        self.data["limit"] = 1
        result = app.recommend(self.data)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0]["id"], "browse")
        self.assertEqual([s["id"] for s in result["journey"]["steps"]],
                         ["browse", "seller-profile"])

    def test_invalid_schema(self):
        for field, value in [("schema_version", True), ("synthetic", False),
                             ("limit", 0), ("limit", True), ("actions", None)]:
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(self.data)
                bad[field] = value
                with self.assertRaises(app.ValidationError):
                    app.recommend(bad)
        self.data["extra"] = 1
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_invalid_action(self):
        for field, value in [("minutes", True), ("minutes", -1), ("title", " "),
                             ("tags", ["same", "same"]), ("prerequisites", ["unknown"])]:
            with self.subTest(field=field):
                bad = copy.deepcopy(self.data)
                bad["actions"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.recommend(bad)

    def test_cycles_and_duplicate_ids(self):
        self.data["actions"][0]["prerequisites"] = ["first-listing"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            app.recommend(self.data)
        self.data["actions"][0]["prerequisites"] = []
        self.data["actions"].append(copy.deepcopy(self.data["actions"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate action"):
            app.recommend(self.data)

    def test_invalid_completion_history(self):
        for completed in [["unknown"], ["first-listing"], ["orientation", "orientation"]]:
            with self.subTest(completed=completed):
                self.data["profile"]["completed"] = completed
                with self.assertRaises(app.ValidationError):
                    app.recommend(self.data)

    def test_journey_validation_rejects_bad_handoffs(self):
        actions = app.validate_input(self.data)
        for steps, budget in [(["first-listing", "seller-profile"], 25),
                              (["seller-profile", "seller-profile"], 25),
                              (["seller-profile", "first-listing"], 24),
                              (["seller-profile"], 25)]:
            with self.subTest(steps=steps, budget=budget):
                with self.assertRaises(app.ValidationError):
                    app.validate_journey(steps, actions, {"orientation"}, budget)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), app.recommend(self.data))
        self.assertEqual(run.stderr, "")
        self.assertEqual(len(run.stdout.splitlines()), 1)

    def test_cli_usage_and_file_errors(self):
        for args in [[], ["nonexistent-input.json"], ["."]]:
            with self.subTest(args=args):
                run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                      *args], capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(run.returncode, 2)
                self.assertEqual(json.loads(run.stdout)["status"], "error")
                self.assertEqual(run.stderr, "")

    def test_cli_malformed_and_invalid_json(self):
        for contents in ['{', '{"x": 1, "x": 2}', 'NaN', 'null', '[]',
                         json.dumps(dict(self.data, limit=False))]:
            with self.subTest(contents=contents):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=contents), redirect_stdout(output):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_encoding(self):
        output = io.StringIO()
        with patch.object(Path, "read_text",
                          side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")), \
                redirect_stdout(output):
            self.assertEqual(app.main(["synthetic-fixture.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
