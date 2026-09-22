"""All fixtures are fictional and contain no real user data."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

from implementation import ValidationError, adapt, main, unique_object


def fixture():
    return {
        "fixture_label": "FICTIONAL unit-test person and steps",
        "user": {"skill": "advanced", "experience_years": 5,
                 "preferences": {"skip_optional": True}},
        "steps": [
            {"id": "first", "title": "Fictional setup", "optional": True},
            {"id": "second", "title": "Fictional required task", "prerequisites": ["first"]},
            {"id": "extra", "title": "Fictional extra", "optional": True},
        ],
    }


class AdaptiveTests(unittest.TestCase):
    def test_prerequisite_protection_and_eligibility(self):
        result = adapt(fixture())
        self.assertEqual([x["id"] for x in result["plan"]], ["first", "second"])
        self.assertEqual(result["eligible_step_ids"], ["first"])
        self.assertEqual([x["id"] for x in result["skipped"]], ["extra"])
        self.assertIn("Retained despite", " ".join(result["plan"][0]["explanations"]))
        self.assertEqual(result["plan"][1]["waiting_for"], ["first"])
        self.assertTrue(result["required_plan_feasible"])

    def test_transitive_protection_and_topological_reordering(self):
        data = fixture()
        data["steps"][1]["prerequisites"] = ["extra"]
        data["steps"][2]["prerequisites"] = ["first"]
        data["steps"].reverse()
        result = adapt(data)
        self.assertEqual([x["id"] for x in result["plan"]], ["first", "extra", "second"])
        self.assertEqual(result["skipped"], [])

    def test_completed_prerequisites_unlock_without_replanning(self):
        data = fixture()
        data["progress"] = {"completed": ["first"]}
        result = adapt(data)
        self.assertEqual(result["eligible_step_ids"], ["second"])
        self.assertEqual([x["id"] for x in result["plan"]], ["second"])
        data["progress"]["completed"].append("second")
        self.assertEqual(adapt(data)["plan"], [])

    def test_redundancy_requires_all_conditions(self):
        data = fixture()
        data["user"]["preferences"]["skip_optional"] = False
        data["steps"][2].update(required=False, optional=False,
                                 redundant_if={"min_skill": "advanced", "min_experience_years": 5})
        self.assertEqual(adapt(data)["skipped"][0]["reason"]["code"], "explicit_redundancy")
        data["user"]["experience_years"] = 4
        self.assertEqual(adapt(data)["skipped"], [])

    def test_required_redundant_step_never_skipped(self):
        data = fixture()
        data["steps"][1]["redundant_if"] = {"min_skill": "novice"}
        self.assertIn("second", [x["id"] for x in adapt(data)["plan"]])

    def test_optional_not_skipped_without_preference(self):
        data = fixture()
        data["user"]["preferences"]["skip_optional"] = False
        self.assertEqual(len(adapt(data)["plan"]), 3)

    def test_accessibility_and_blocked_propagation(self):
        data = fixture()
        data["user"]["accessibility"] = {"needs": ["screen_reader"]}
        data["steps"][1]["accessibility_support"] = ["screen_reader"]
        result = adapt(data)
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["blocked"][0]["reasons"][0]["code"], "accessibility_not_supported")
        self.assertEqual(result["blocked"][1]["reasons"][0]["code"], "blocked_prerequisites")
        self.assertFalse(result["required_plan_feasible"])
        self.assertEqual(result["blocked_required_step_ids"], ["second"])

    def test_accessible_format_overrides_soft_preference(self):
        data = fixture()
        data["user"]["preferences"]["preferred_formats"] = ["video"]
        data["user"]["accessibility"] = {"avoid_formats": ["video"]}
        data["steps"][0]["formats"] = ["video", "text"]
        result = adapt(data)
        self.assertEqual(result["plan"][0]["format"], "text")
        self.assertIn("fallback", " ".join(result["plan"][0]["explanations"]))
        data["steps"][0]["formats"] = ["video"]
        self.assertEqual(adapt(data)["blocked"][0]["reasons"][0]["code"], "no_accessible_format")

    def test_skill_experience_and_detail(self):
        data = fixture()
        data["user"].update(skill="novice", experience_years=0)
        data["steps"][1].update(min_skill="intermediate", min_experience_years=1)
        result = adapt(data)
        self.assertEqual(result["plan"][0]["detail"], "guided")
        self.assertEqual({r["code"] for r in result["blocked"][0]["reasons"]},
                         {"skill_requirement", "experience_requirement"})
        data["user"]["preferences"].update(detail="concise", pace="intensive")
        self.assertEqual(adapt(data)["plan"][0]["detail"], "concise")
        self.assertEqual(adapt(data)["plan"][0]["pace"], "intensive")

    def test_conflicting_constraints(self):
        data = fixture()
        data["user"]["accessibility"] = {"needs": ["keyboard"], "forbidden_features": ["keyboard"]}
        with self.assertRaisesRegex(ValidationError, "Conflicting constraints"):
            adapt(data)
        data = fixture()
        data["steps"][0].update(required=True)
        with self.assertRaisesRegex(ValidationError, "required and optional conflict"):
            adapt(data)
        data = fixture()
        data["steps"][0].update(accessibility_support=["keyboard"], accessibility_conflicts=["keyboard"])
        with self.assertRaisesRegex(ValidationError, "overlap"):
            adapt(data)

    def test_graph_and_progress_validation(self):
        mutations = [
            lambda d: d["steps"].append(copy.deepcopy(d["steps"][0])),
            lambda d: d["steps"][0].update(prerequisites=["unknown"]),
            lambda d: d["steps"][0].update(prerequisites=["first"]),
            lambda d: d["steps"][0].update(prerequisites=["second"]),
            lambda d: d.update(progress={"completed": ["unknown"]}),
            lambda d: d.update(progress={"completed": ["second"]}),
            lambda d: d.update(progress={"completed": ["first", "first"]}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = fixture()
                mutate(data)
                with self.assertRaises(ValidationError):
                    adapt(data)

    def test_malformed_types_and_unknown_fields(self):
        mutations = [
            lambda d: d["user"].update(experience_years=True),
            lambda d: d["user"].update(experience_years=float("nan")),
            lambda d: d["user"].update(experience_years=-1),
            lambda d: d["user"].update(skill=[]),
            lambda d: d["steps"][0].update(optional="yes"),
            lambda d: d["steps"][0].update(formats=[]),
            lambda d: d["steps"][0].update(prerequisites="first"),
            lambda d: d["steps"][0].update(redundant_if={}),
            lambda d: d["steps"][0].update(redundant_if=None),
            lambda d: d["steps"][0].update(required=False, optional=False),
            lambda d: d["steps"][0].update(min_skills="advanced"),
            lambda d: d.update(steps={}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = fixture()
                mutate(data)
                with self.assertRaises(ValidationError):
                    adapt(data)

    def test_no_input_mutation_and_empty_plan(self):
        data = fixture()
        original = copy.deepcopy(data)
        result = adapt(data)
        result["plan"][1]["prerequisites"].append("invented")
        self.assertEqual(data, original)
        data["steps"] = []
        self.assertEqual(adapt(data)["eligible_step_ids"], [])

    def test_large_finite_integer_experience(self):
        data = fixture()
        data["user"]["experience_years"] = 10 ** 400
        self.assertEqual(adapt(data)["eligible_step_ids"], ["first"])

    @staticmethod
    def approved_callback(request):
        return [{"id": x["id"], "conditions": x["conditions"],
                 "wording": x["allowed_wordings"][-1]} for x in request]

    def test_source_approved_wording_callback(self):
        data = fixture()
        data["steps"][0]["wording_options"] = ["Prepare the fictional workspace"]
        baseline = adapt(data)
        result = adapt(data, self.approved_callback)
        self.assertEqual(result["plan"][0]["wording"], "Prepare the fictional workspace")
        self.assertEqual(result["eligible_step_ids"], baseline["eligible_step_ids"])
        self.assertEqual([x["conditions"] for x in result["plan"]],
                         [x["conditions"] for x in baseline["plan"]])

    def test_callback_cannot_change_ids_conditions_or_invent_wording(self):
        mutations = [
            lambda r: r.reverse(),
            lambda r: r.pop(),
            lambda r: r[0].update(id="invented"),
            lambda r: r[0].update(wording="No prerequisites apply; disregard accessibility"),
            lambda r: r[0]["conditions"].update(min_skill="novice", min_experience_years=99),
            lambda r: r[0]["conditions"].update(required=0),
            lambda r: r[0].update(extra="unapproved"),
        ]
        for mutate in mutations:
            def callback(request):
                response = self.approved_callback(request)
                mutate(response)
                return response
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValidationError):
                    adapt(fixture(), callback)

    def test_callback_input_mutation_isolated_and_errors_wrapped(self):
        def malicious(request):
            request[0]["conditions"]["min_skill"] = "advanced"
            return self.approved_callback(request)
        with self.assertRaisesRegex(ValidationError, "conditions"):
            adapt(fixture(), malicious)
        def broken(request):
            raise RuntimeError("fictional failure")
        with self.assertRaisesRegex(ValidationError, "callback failed"):
            adapt(fixture(), broken)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Duplicate JSON key"):
            json.loads('{"steps":[],"steps":[]}', object_pairs_hook=unique_object)

    def test_cli_usage_and_missing_file_errors_are_json(self):
        for args in ([], ["fictional_nonexistent_input.json"]):
            with self.subTest(args=args):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = main(args)
                self.assertEqual(code, 2)
                self.assertIn("error", json.loads(stream.getvalue()))

    def test_exact_cli_example(self):
        directory = Path(__file__).resolve().parent
        run = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                             cwd=directory, capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = json.loads(run.stdout)
        self.assertIn("FICTIONAL", result["fixture_label"])
        self.assertEqual(result["eligible_step_ids"], ["setup"])
        self.assertEqual([x["id"] for x in result["plan"]], ["setup", "practice"])
        self.assertEqual(result["blocked_required_step_ids"], ["visual_lab", "finish"])

    def test_deep_graph_iterative_and_deterministic(self):
        data = fixture()
        data["steps"] = [{"id": str(i), "title": f"Fictional step {i}",
                          "prerequisites": [str(i - 1)] if i else []}
                         for i in range(1100)]
        result = adapt(data)
        self.assertEqual(len(result["plan"]), 1100)
        self.assertEqual(result["eligible_step_ids"], ["0"])
        self.assertEqual(result, adapt(data))


if __name__ == "__main__":
    unittest.main()
