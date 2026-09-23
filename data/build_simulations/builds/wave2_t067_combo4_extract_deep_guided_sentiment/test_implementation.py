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
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def initial(self):
        return {"schema_version": 1, "status": "ok", "stage": "input",
                "input": copy.deepcopy(self.data)}

    def result(self):
        return app.run_pipeline(self.data)

    def test_end_to_end(self):
        out = self.result()
        self.assertEqual(out["stage"], "sentiment")
        app.validate_state(out, "sentiment")
        self.assertEqual(out["onboarding"]["progress"]["completed"], 1)

    def test_extracted_spans_point_to_original_unicode(self):
        self.data["documents"][0]["text"] = "🌍 café\n" + self.data["documents"][0]["text"]
        out = self.result()
        source = {d["id"]: d["text"] for d in self.data["documents"]}
        for doc in out["extraction"]["documents"]:
            for matches in doc["fields"].values():
                for evidence in matches:
                    span = evidence["span"]
                    self.assertEqual(source[evidence["document_id"]][span["start"]:span["end"]],
                                     evidence["value"])

    def test_missing_optional_field_is_reported(self):
        missing = self.result()["extraction"]["documents"][1]["missing"]
        self.assertEqual(missing, [{"field": "owner", "required": False}])

    def test_required_missing_blocks_consensus(self):
        self.data["documents"][1]["text"] = "Plan: Enterprise\n"
        out = self.result()
        region = next(t for t in out["research"]["topics"] if t["field"] == "region")
        self.assertEqual(region["status"], "consensus")
        self.assertFalse(region["usable"])
        self.assertEqual(out["onboarding"]["steps"][0]["status"], "blocked")

    def test_disagreement_propagates_to_blocked_priority(self):
        out = self.result()
        self.assertEqual(out["research"]["disagreements"][0]["values"], ["enterprise", "starter"])
        plan = out["onboarding"]["steps"][1]
        self.assertEqual(plan["blocked_fields"], ["plan"])
        self.assertEqual(plan["status"], "blocked")
        issue = next(i for i in out["insights"]["issues"] if i["step_id"] == "plan")
        self.assertEqual(issue["priority_components"]["onboarding"], 10)
        self.assertEqual(issue["priority"], 85)

    def test_consensus_resolution_propagates(self):
        self.data["documents"][1]["text"] = "Plan: starter\nRegion: eu\n"
        out = self.result()
        self.assertFalse(out["research"]["disagreements"])
        self.assertEqual(out["onboarding"]["steps"][1]["status"], "completed")
        self.assertEqual(out["onboarding"]["steps"][2]["status"], "ready")
        self.assertAlmostEqual(out["onboarding"]["progress"]["fraction"], 2 / 3)
        issue = next(i for i in out["insights"]["issues"] if i["step_id"] == "plan")
        self.assertEqual(issue["priority"], 75)

    def test_all_missing(self):
        for doc in self.data["documents"]:
            doc["text"] = ""
        out = self.result()
        self.assertTrue(all(t["status"] == "missing" for t in out["research"]["topics"]))
        self.assertEqual(out["onboarding"]["progress"]["fraction"], 0)
        self.assertEqual(len(out["research"]["unresolved_questions"]), 3)

    def test_multiple_matches_preserve_evidence(self):
        self.data["documents"][0]["text"] += "Plan: Starter\nPlan: Deluxe\n"
        out = self.result()
        self.assertEqual(len(out["extraction"]["documents"][0]["fields"]["plan"]), 3)
        starter = next(c for c in out["research"]["topics"][0]["claims"]
                       if c["normalized_value"] == "starter")
        self.assertEqual(len(starter["evidence"]), 2)
        self.assertEqual(len(starter["supporting_documents"]), 1)

    def test_empty_capture_is_missing(self):
        self.data["fields"][0]["pattern"] = "(?P<value>)"
        self.assertEqual(self.result()["research"]["topics"][0]["status"], "missing")

    def test_unmatched_optional_capture_is_missing(self):
        self.data["fields"][0]["pattern"] = "SYNTHETIC(?P<value>XYZ)?"
        self.assertEqual(self.result()["research"]["topics"][0]["status"], "missing")

    def test_dependency_order_is_topological(self):
        self.data["steps"].reverse()
        self.assertEqual([s["id"] for s in self.result()["onboarding"]["steps"]],
                         ["region", "plan", "launch"])

    def test_incomplete_prerequisite_blocks_downstream(self):
        self.data["steps"][0]["complete"] = False
        out = self.result()
        self.assertEqual(out["onboarding"]["steps"][0]["status"], "ready")
        self.assertEqual(out["onboarding"]["steps"][1]["blocked_prerequisites"], ["region"])

    def test_invalid_inputs(self):
        cases = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(documents=[]),
            lambda d: d.update(steps=[]),
            lambda d: d.update(extra=1),
            lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
            lambda d: d["fields"][0].update(required=1),
            lambda d: d["fields"][0].update(pattern="["),
            lambda d: d["fields"][0].update(pattern="(?P<value>a{99999999999999999999})"),
            lambda d: d["fields"][0].update(pattern="Plan: (.*)"),
            lambda d: d["steps"][0].update(complete="true"),
            lambda d: d["steps"][0].update(requires_fields=["unknown"]),
            lambda d: d["steps"][0].update(prerequisites=["unknown"]),
            lambda d: d["steps"][0].update(prerequisites=["launch"]),
            lambda d: d["steps"][0].update(prerequisites=["region"]),
            lambda d: d["feedback"][0].update(step_id="unknown"),
            lambda d: d["feedback"][0].update(severity="urgent"),
            lambda d: d["feedback"][0].update(severity=[]),
            lambda d: d["feedback"].append(copy.deepcopy(d["feedback"][0])),
            lambda d: d["documents"][0].update(text=None),
        ]
        for mutate in cases:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_input_is_not_mutated(self):
        before = copy.deepcopy(self.data)
        self.result()
        self.assertEqual(self.data, before)

    def test_deterministic(self):
        self.assertEqual(self.result(), self.result())

    def test_extraction_handoff_tampering_rejected(self):
        out = app.extract(self.initial())
        out["extraction"]["documents"][0]["fields"]["plan"][0]["span"]["start"] = 0
        with self.assertRaises(app.ValidationError):
            app.deep(out)

    def test_research_handoff_tampering_rejected(self):
        out = app.deep(app.extract(self.initial()))
        out["research"]["topics"][0]["usable"] = True
        with self.assertRaises(app.ValidationError):
            app.guided(out)

    def test_onboarding_handoff_tampering_rejected(self):
        out = app.guided(app.deep(app.extract(self.initial())))
        out["onboarding"]["feedback"][0]["step_status"] = "completed"
        with self.assertRaises(app.ValidationError):
            app.sentiment(out)

    def test_stage_order_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.guided(self.initial())

    def test_sentiment_negation(self):
        result = app.score_sentiment("not good never broken no helpful")
        self.assertEqual(result["score"], 0)
        self.assertTrue(all(c["negated"] for c in result["contributions"]))

    def test_negation_does_not_cross_punctuation(self):
        self.assertEqual(app.score_sentiment("not. good")["score"], 1)

    def test_unicode_prefix_preserves_negation_boundaries(self):
        self.assertEqual(app.score_sentiment("ß NOT GOOD")["score"], -1)

    def test_sentiment_clipping_and_neutral(self):
        self.assertEqual(app.score_sentiment("great " * 20)["score"], 5)
        self.assertEqual(app.score_sentiment("broken " * 20)["score"], -5)
        self.assertEqual(app.score_sentiment("")["label"], "neutral")
        self.assertEqual(app.score_sentiment("ordinary observation")["score"], 0)

    def test_severity_priority_cap(self):
        first = self.result()["insights"]["issues"][0]
        self.assertEqual(first["severity"], "critical")
        self.assertEqual(first["priority"], 100)
        self.assertEqual(first["rank"], 1)

    def test_ties_are_stable_by_id(self):
        feedback = self.data["feedback"][0]
        self.data["feedback"] = [dict(feedback, id="z"), dict(feedback, id="a")]
        self.assertEqual([i["id"] for i in self.result()["insights"]["issues"]], ["a", "z"])

    def test_no_feedback(self):
        self.data["feedback"] = []
        self.assertEqual(self.result()["insights"]["issues"], [])

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        result = self.cli("nonexistent-synthetic-input.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_argument_errors(self):
        for args in [(), ("example_input.json", "extra")]:
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_and_schema(self):
        # Inject file contents in memory so the deliverable directory stays at four files.
        for raw in ["{", "[]", '{"x":1,"x":2}', '{"value": NaN}', "null"]:
            with self.subTest(raw=raw), patch.object(Path, "read_text", return_value=raw):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_file_decoding_error(self):
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        with patch.object(Path, "read_text", side_effect=error):
            output = io.StringIO()
            with redirect_stdout(output):
                code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
