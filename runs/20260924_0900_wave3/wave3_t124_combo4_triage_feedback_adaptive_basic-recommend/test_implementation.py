import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def stages(self):
        return app.run(self.data)["stages"]

    def test_full_pipeline(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), list(app.ORDER))
        app.validate_state(result, 4)

    def test_triage_priority_and_accountability(self):
        ticket = self.stages()["triage"]["tickets"][0]
        self.assertEqual((ticket["category"], ticket["priority"], ticket["owner"]),
                         ("technical", "urgent", "support-engineering"))

    def test_default_category(self):
        self.data["tickets"] = [{"id": "x", "text": "Hello there"}]
        self.assertEqual(self.stages()["triage"]["tickets"][0]["category"], "general")

    def test_word_boundaries(self):
        self.data["tickets"] = [{"id": "x", "text": "unbroken exports"}]
        self.assertEqual(self.stages()["triage"]["tickets"][0]["category"], "general")

    def test_configurable_owner_and_priority(self):
        self.data["config"]["categories"]["technical"]["owner"] = "export-specialist"
        self.data["config"]["categories"]["technical"]["priority"] = "urgent"
        result = self.stages()
        self.assertEqual(result["feedback"]["entries"][0]["owners"], ["customer-education"])
        ticket = result["triage"]["tickets"][1]
        self.assertEqual((ticket["owner"], ticket["priority"]), ("export-specialist", "urgent"))

    def test_category_tie_is_lexical(self):
        self.data["tickets"] = [{"id": "x", "text": "export tutorial"}]
        self.assertEqual(self.stages()["triage"]["tickets"][0]["category"], "education")

    def test_feedback_deduplicates_and_traces(self):
        result = self.stages()["feedback"]
        self.assertEqual(len(result["entries"]), 2)
        export = next(e for e in result["entries"] if "export" in e["themes"])
        self.assertEqual(export["ticket_ids"], ["t1", "t2"])
        self.assertEqual(export["priority"], "urgent")
        self.assertEqual(export["excerpts"][0]["text"], self.data["tickets"][0]["text"])
        theme = next(t for t in result["themes"] if t["name"] == "export")
        self.assertEqual(theme["unique_feedback_count"], 1)
        self.assertEqual(theme["feedback_ids"], [export["id"]])

    def test_feedback_fallback(self):
        self.data["config"]["themes"] = {}
        self.assertIn("technical", self.stages()["adaptive"]["interests"])

    def test_adaptive_prerequisites_and_explanations(self):
        plan = self.stages()["adaptive"]["plan"]
        self.assertEqual(plan[0]["step_id"], "basics")
        export = next(step for step in plan if step["step_id"] == "export-guide")
        self.assertIn("feedback theme: export", export["reasons"])
        self.assertEqual(export["prerequisites"], ["basics"])

    def test_experience_adaptation(self):
        self.data["tickets"] = []
        self.data["profile"]["preferences"] = []
        self.data["profile"]["experience"] = "expert"
        self.assertEqual(self.stages()["adaptive"]["plan"], [])
        self.data["profile"]["experience"] = "beginner"
        self.assertEqual(self.stages()["adaptive"]["plan"][0]["step_id"], "basics")

    def test_completed_steps_not_planned(self):
        self.data["profile"]["completed_steps"] = ["basics"]
        self.assertNotIn("basics", [s["step_id"] for s in self.stages()["adaptive"]["plan"]])

    def test_recommendation_personalized(self):
        product = self.stages()["recommend"]["products"][0]
        self.assertEqual(product["product_id"], "starter")
        self.assertEqual(product["score"], 3)
        self.assertTrue(product["feedback_ids"])

    def test_planned_is_not_completed(self):
        result = self.stages()
        self.assertIn("export-guide", [s["step_id"] for s in result["adaptive"]["plan"]])
        self.assertNotIn("export-kit", [p["product_id"] for p in result["recommend"]["products"]])
        self.data["profile"]["completed_steps"] = ["basics", "export-guide"]
        self.assertIn("export-kit", [p["product_id"] for p in self.stages()["recommend"]["products"]])

    def test_all_recommendation_filters(self):
        self.data["profile"]["excluded_products"] = ["starter"]
        self.data["catalog"][3]["available"] = False
        result = self.stages()["recommend"]
        self.assertEqual(result["products"], [])
        self.assertEqual(len(result["excluded"]), 4)

    def test_cross_stage_theme_propagation(self):
        self.data["tickets"] = [{"id": "new", "text": "analytics analytics"}]
        result = self.stages()
        self.assertEqual(result["feedback"]["themes"][0]["name"], "analytics")
        self.assertIn("analytics", result["adaptive"]["interests"])
        self.assertEqual(result["recommend"]["products"][0]["score"], 3)
        self.assertEqual(result["recommend"]["products"][0]["feedback_ids"], ["feedback-1"])

    def test_empty_input_collections(self):
        self.data["tickets"] = []
        self.data["catalog"] = []
        result = self.stages()
        self.assertEqual(result["feedback"], {"entries": [], "themes": []})
        self.assertEqual(result["recommend"]["products"], [])

    def test_determinism_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, before)

    def test_invalid_inputs(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(tickets="bad"),
            lambda d: d["tickets"].append(d["tickets"][0]),
            lambda d: d["tickets"][0].update(text="!!!"),
            lambda d: d["profile"].update(budget=float("nan")),
            lambda d: d["profile"].update(experience="wizard"),
            lambda d: d["profile"].update(completed_steps=["export-guide"]),
            lambda d: d["catalog"][0].update(available=1),
            lambda d: d["config"].update(recommendation_limit=False),
            lambda d: d["config"]["categories"]["technical"].update(owner=" "),
            lambda d: d["config"]["onboarding_steps"][0].update(prerequisites=["missing"]),
            lambda d: d.update(unexpected=1),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_cycle_rejected(self):
        self.data["config"]["onboarding_steps"][0]["prerequisites"] = ["export-guide"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            app.run(self.data)

    def test_stage_boundary_tampering_rejected(self):
        state = {"schema_version": 1, "status": "in_progress", "request": self.data, "stages": {}}
        state = app.advance(state, "triage")
        state["stages"]["triage"]["tickets"][0]["owner"] = "unaccountable"
        with self.assertRaisesRegex(app.ValidationError, "invalid triage"):
            app.advance(state, "feedback")

    def test_wrong_stage_order_rejected(self):
        state = {"schema_version": 1, "status": "in_progress", "request": self.data, "stages": {}}
        with self.assertRaises(app.ValidationError):
            app.advance(state, "adaptive")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "does-not-exist.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_malformed_and_invalid_json(self):
        for content in ("{", "null", '{"schema_version":1,"schema_version":1}',
                        json.dumps(dict(self.data, schema_version=99))):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
