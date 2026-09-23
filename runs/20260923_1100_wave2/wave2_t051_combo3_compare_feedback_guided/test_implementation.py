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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.input = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def output(self):
        return app.run_pipeline(self.input)["result"]["data"]

    def test_normalization(self):
        row = app.compare(self.input)["data"]["rows"][0]
        self.assertEqual(row["attributes"], {"price": 80, "weight": 500, "battery": 10})

    def test_preference_ranking(self):
        ranking = app.compare(self.input)["data"]["ranking"]
        self.assertEqual(ranking, [{"id": "aurora", "score": 0.8}, {"id": "beacon", "score": 0.2}])

    def test_ties_are_stable(self):
        self.input["products"][1]["attributes"] = copy.deepcopy(self.input["products"][0]["attributes"])
        self.input["products"].reverse()
        result = app.compare(self.input)["data"]
        self.assertEqual(result["selected_product_id"], "aurora")
        self.assertEqual([r["score"] for r in result["ranking"]], [1, 1])

    def test_dedup_and_traceability(self):
        feedback = self.output()["feedback"]["data"]
        self.assertEqual((feedback["source_count"], feedback["unique_count"], feedback["duplicate_count"]), (3, 2, 1))
        self.assertEqual(feedback["excerpts"][0]["source_ids"], ["r1", "r2"])
        self.assertEqual(feedback["excerpts"][0]["text"], self.input["feedback"][0]["text"])
        theme = next(t for t in feedback["themes"] if t["id"] == "battery")
        self.assertEqual(theme["excerpt_ids"], ["r1"])

    def test_cross_stage_winner_change(self):
        self.input["preferences"] = {"battery": {"weight": 1, "direction": "max"}}
        output = self.output()
        self.assertEqual(output["selected_product_id"], "beacon")
        self.assertEqual(output["feedback"]["data"]["excerpts"][0]["id"], "r4")
        charge = next(s for s in output["steps"] if s["id"] == "charge")
        self.assertEqual(charge["supporting_excerpt_ids"], ["r4"])
        self.assertIn("pair", {s["id"] for s in output["steps"]})

    def test_progress_and_states(self):
        output = self.output()
        self.assertEqual([s["state"] for s in output["steps"]], ["completed", "available", "blocked"])
        self.assertEqual(output["progress"], {"completed": 1, "total": 3, "percent": 33.33})
        self.input["onboarding"]["completed_steps"].append("pair")
        self.assertEqual(self.output()["steps"][-1]["state"], "available")

    def test_no_feedback(self):
        self.input["feedback"] = []
        output = self.output()
        self.assertEqual(output["feedback"]["data"]["themes"], [])
        self.assertEqual([s["id"] for s in output["steps"]], ["unpack"])
        self.assertEqual(output["progress"]["percent"], 100)

    def test_no_steps(self):
        self.input["onboarding"] = {"steps": [], "completed_steps": []}
        self.assertEqual(self.output()["progress"], {"completed": 0, "total": 0, "percent": 100})

    def test_general_theme(self):
        self.input["feedback"] = [{"id": "r1", "product_id": "aurora", "text": "Wonderful sound."}]
        self.assertEqual(self.output()["feedback"]["data"]["themes"][0]["id"], "general")

    def test_invalid_numbers_units(self):
        for bad in [True, -1, float("nan"), float("inf"), "80"]:
            with self.subTest(bad=bad):
                request = copy.deepcopy(self.input)
                request["products"][0]["attributes"]["price"]["value"] = bad
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)
        self.input["products"][0]["attributes"]["price"]["unit"] = "EUR"
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_missing_attributes(self):
        del self.input["products"][0]["attributes"]["weight"]
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_bad_references_duplicates(self):
        for mutation in ("feedback_product", "product_id", "prerequisite", "completed"):
            with self.subTest(mutation=mutation):
                request = copy.deepcopy(self.input)
                if mutation == "feedback_product":
                    request["feedback"][0]["product_id"] = "missing"
                elif mutation == "product_id":
                    request["products"][1]["id"] = "aurora"
                elif mutation == "prerequisite":
                    request["onboarding"]["steps"][0]["prerequisites"] = ["missing"]
                else:
                    request["onboarding"]["completed_steps"] = ["missing"]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_cycles(self):
        self.input["onboarding"]["steps"][0]["prerequisites"] = ["charge"]
        with self.assertRaisesRegex(app.ValidationError, "cyclic"):
            self.output()

    def test_completion_prerequisites(self):
        self.input["onboarding"]["completed_steps"] = ["charge"]
        with self.assertRaisesRegex(app.ValidationError, "prerequisites"):
            self.output()

    def test_inactive_completion(self):
        self.input["feedback"] = []
        self.input["onboarding"]["completed_steps"] = ["unpack", "pair"]
        with self.assertRaisesRegex(app.ValidationError, "inactive"):
            self.output()

    def test_handoff_tampering(self):
        comparison = app.compare(self.input)
        comparison["data"]["selected_product_id"] = "beacon"
        with self.assertRaises(app.ValidationError):
            app.analyze_feedback(self.input, comparison)
        feedback = app.analyze_feedback(self.input, app.compare(self.input))
        feedback["data"]["excerpts"][0]["text"] = "Invented quote."
        with self.assertRaisesRegex(app.ValidationError, "belong"):
            app.guide(self.input, feedback)

    def test_input_order_invariance(self):
        expected = self.output()
        for key in ("products", "feedback"):
            self.input[key].reverse()
        self.input["onboarding"]["steps"].reverse()
        self.assertEqual(self.output(), expected)

    def test_single_product(self):
        self.input["products"] = self.input["products"][:1]
        self.input["feedback"] = self.input["feedback"][:3]
        self.assertEqual(app.compare(self.input)["data"]["ranking"], [{"id": "aurora", "score": 1}])

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["a", "b"]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_malformed_validation_and_duplicate_errors(self):
        for contents in ("{", "{}", '{"a":1,"a":2}', '{"a":NaN}', "null", "[]"):
            with self.subTest(contents=contents), patch.object(Path, "read_text", return_value=contents):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_wrong_shape_inputs(self):
        for key, value in (("products", []), ("preferences", {}), ("schema_version", True),
                           ("synthetic", False), ("feedback", "bad")):
            request = copy.deepcopy(self.input)
            request[key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)


if __name__ == "__main__":
    unittest.main()
