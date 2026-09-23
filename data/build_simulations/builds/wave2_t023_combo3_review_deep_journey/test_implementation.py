"""Standard-library synthetic fixtures; all filesystem access stays in this build."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def rows(self, result):
        return {r["requirement_id"]: r for r in result["review"]["requirements"]}

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["action_id"] for s in result["journey"]["two_step_journey"]],
                         ["inspect", "reconcile"])
        self.assertTrue(result["synthetic"])

    def test_review_coverage_conflict_and_missing(self):
        result = app.run_pipeline(self.data)
        rows = self.rows(result)
        self.assertEqual(rows["consent"]["status"], "satisfied")
        self.assertEqual(rows["retention"]["status"], "conflict")
        self.assertEqual(rows["training"]["status"], "missing")
        self.assertEqual(len(result["review"]["gaps"]), 2)
        self.assertIn("not certification", result["review"]["notice"])

    def test_repeated_evidence_not_multiple_sources(self):
        self.data["documents"][1]["evidence"].pop(0)
        duplicate = copy.deepcopy(self.data["documents"][0]["evidence"][0])
        duplicate["id"] = "e5"
        self.data["documents"][0]["evidence"].append(duplicate)
        row = self.rows(app.run_pipeline(self.data))["consent"]
        self.assertEqual(row["status"], "insufficient")
        self.assertEqual(row["supporting_documents"], ["policy"])

    def test_research_disagreement_traces(self):
        deep = app.run_pipeline(self.data)["deep"]
        self.assertEqual(deep["source_count"], 2)
        self.assertEqual(deep["disagreements"], [{
            "requirement_id": "retention", "supporting_evidence_ids": ["e2"],
            "contradicting_evidence_ids": ["e4"]}])
        self.assertEqual({q["requirement_id"] for q in deep["unresolved_questions"]},
                         {"retention", "training"})

    def test_journey_citations_and_questions_propagate(self):
        journey = app.run_pipeline(self.data)["journey"]
        for step in journey["two_step_journey"]:
            self.assertEqual(step["evidence_ids"], ["e2", "e4"])
            self.assertEqual(step["question_ids"], ["q:retention"])
            self.assertTrue(step["prerequisites_satisfied"])
        self.assertNotIn("reconcile", [s["action_id"] for s in journey["next_actions"]])

    def test_contradiction_removal_propagates(self):
        before = app.run_pipeline(self.data)
        self.data["documents"][1]["evidence"].pop()
        after = app.run_pipeline(self.data)
        self.assertEqual(self.rows(after)["retention"]["status"], "satisfied")
        self.assertFalse(after["deep"]["disagreements"])
        self.assertEqual(after["journey"]["two_step_journey"][0]["action_id"], "train")
        self.assertNotEqual(before["journey"], after["journey"])

    def test_unknown_and_adverse(self):
        self.data["documents"][0]["evidence"][0]["stance"] = "unknown"
        self.data["documents"][1]["evidence"][0]["stance"] = "unknown"
        self.data["documents"][0]["evidence"].pop()
        result = app.run_pipeline(self.data)
        self.assertEqual(self.rows(result)["consent"]["status"], "unknown")
        self.assertEqual(self.rows(result)["retention"]["status"], "contradicted")
        findings = {f["requirement_id"]: f for f in result["deep"]["findings"]}
        self.assertEqual(findings["retention"]["conclusion"], "adverse")

    def test_no_documents(self):
        self.data["documents"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(all(r["status"] == "missing" for r in self.rows(result).values()))
        self.assertEqual(result["deep"]["source_count"], 0)
        self.assertTrue(all(not f["citations"] for f in result["deep"]["findings"]))

    def test_no_actions(self):
        self.data["actions"] = []
        self.data["completed_actions"] = []
        journey = app.run_pipeline(self.data)["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual(journey["two_step_journey"], [])
        self.assertEqual(journey["unaddressed_requirement_ids"], ["retention", "training"])

    def test_one_action_is_not_a_two_step_journey(self):
        self.data["actions"] = self.data["actions"][:1]
        self.data["completed_actions"] = []
        journey = app.run_pipeline(self.data)["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual(len(journey["next_actions"]), 1)
        self.assertEqual(journey["two_step_journey"], [])

    def test_foundation_can_unlock_second_step(self):
        self.data["completed_actions"] = []
        steps = app.run_pipeline(self.data)["journey"]["two_step_journey"]
        self.assertEqual([s["action_id"] for s in steps], ["orient", "inspect"])

    def test_all_completed_is_blocked(self):
        self.data["completed_actions"] = [a["id"] for a in self.data["actions"]]
        journey = app.run_pipeline(self.data)["journey"]
        self.assertEqual(journey["status"], "blocked")
        self.assertEqual(journey["next_actions"], [])

    def test_order_independence_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        expected = app.run_pipeline(self.data)
        self.assertEqual(self.data, original)
        for key in ("documents", "requirements", "actions"):
            self.data[key].reverse()
        for document in self.data["documents"]:
            document["evidence"].reverse()
        self.assertEqual(app.run_pipeline(self.data), expected)

    def test_invalid_shapes_and_types(self):
        for field, value in [
            ("schema_version", "2"), ("synthetic", False), ("requirements", []),
            ("requirements", None), ("documents", {}), ("actions", "bad"),
            ("completed_actions", [42]), ("completed_actions", ["absent"])
        ]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                data[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        with self.assertRaises(app.ValidationError):
            app.run_pipeline([])

    def test_positive_integer_threshold_not_boolean(self):
        for value in (True, 0, -1, 1.2, "1"):
            with self.subTest(value=value):
                self.data["requirements"][0]["min_sources"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_invalid_evidence(self):
        for field, value in [
            ("quote", "Invented quote"), ("quote", ""), ("stance", "maybe"),
            ("requirement_id", "absent"), ("requirement_id", [])
        ]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["documents"][0]["evidence"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_ids_and_extra_fields(self):
        mutations = [
            lambda d: d["requirements"].append(copy.deepcopy(d["requirements"][0])),
            lambda d: d["documents"][1]["evidence"][0].update(id="e1"),
            lambda d: d.update(extra=True),
            lambda d: d["actions"][0]["requirement_ids"].extend(["consent", "consent"]),
        ]
        for mutate in mutations:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_invalid_prerequisites(self):
        for prerequisites in (["absent"], ["orient"], ["reconcile"]):
            with self.subTest(prerequisites=prerequisites):
                data = copy.deepcopy(self.data)
                data["actions"][0]["prerequisites"] = prerequisites
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_completed_action_prerequisite_closure(self):
        self.data["completed_actions"] = ["inspect"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_review_tamper_rejected_at_handoff(self):
        review = app.review_stage(self.data)
        review["requirements"][0]["status"] = "missing"
        with self.assertRaises(app.ValidationError):
            app.deep_stage(review, self.data)

    def test_research_fabricated_citation_rejected(self):
        review = app.review_stage(self.data)
        deep = app.deep_stage(review, self.data)
        deep["findings"][0]["citations"][0]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.journey_stage(deep, review, self.data)

    def test_invalid_journey_order_rejected(self):
        result = app.run_pipeline(self.data)
        result["journey"]["two_step_journey"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate("journey", result["journey"], {"deep": result["deep"], "input": self.data})

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), app.run_pipeline(self.data))
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ((), ("does_not_exist.json",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_existing_invalid_input(self):
        result = self.cli("build_manifest.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_duplicate_nonstandard_json(self):
        for content in ('{', '{"a": 1, "a": 2}', '{"a": NaN}', 'null', '[]'):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.StringIO(content)):
                    with redirect_stdout(output):
                        code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_and_file_errors(self):
        for error in (PermissionError("fixture denied"), UnicodeError("fixture encoding")):
            output = io.StringIO()
            with patch.object(Path, "open", side_effect=error):
                with redirect_stdout(output):
                    code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
