"""All fixtures are synthetic. No network or external dependencies."""
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


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_handoffs(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["journey"]["source_product_ids"], ["camera"])
        self.assertEqual(result["sentiment"]["source_action_ids"], ["setup", "shoot"])
        self.assertEqual({i["feedback_id"] for i in result["sentiment"]["issues"]}, {"f1", "f2", "f3"})
        self.assertEqual(result["sentiment"]["excluded_feedback_count"], 2)

    def test_synonym_search(self):
        self.data["query"] = "intro photographic"
        self.assertEqual(app.run(self.data)["semantic"]["hits"][0]["product_id"], "camera")

    def test_index_and_ranking(self):
        index = app.SearchIndex(self.data["products"])
        self.assertEqual(index.postings["camera"], {"camera"})
        self.assertGreater(index.scores("camera")["camera"], 0)

    def test_ties_deterministic(self):
        self.data["products"].append(dict(self.data["products"][0], id="a"))
        hits = app.run(self.data)["semantic"]["hits"]
        self.assertEqual([h["product_id"] for h in hits], ["a", "camera"])
        self.assertEqual(app.run(self.data), app.run(self.data))

    def test_injected_embedding(self):
        self.data["query"] = "unknownword"
        def embed(value):
            return [1.0, 0.0] if "Garden" not in value else [0.0, 1.0]
        result = app.run(self.data, embed)
        self.assertEqual(result["semantic"]["hits"][0]["embedding_score"], 1.0)
        self.assertEqual(result["journey"]["status"], "ready")

    def test_invalid_vectors(self):
        for value in ([], [0, 0], [True], ["x"], [float("nan")], [float("inf")], [1e308, 1e308, 1e308, 1e308]):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run(self.data, lambda _: value)

    def test_vector_dimension(self):
        with self.assertRaisesRegex(app.ValidationError, "dimension"):
            app.run(self.data, lambda text: [1] if text == self.data["query"] else [1, 0])

    def test_embedding_exception(self):
        def failing(_):
            raise RuntimeError("fixture error")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run(self.data, failing)

    def test_completed_prerequisite(self):
        self.data["profile"]["completed_actions"] = ["setup"]
        result = app.run(self.data)
        self.assertEqual([s["action_id"] for s in result["journey"]["steps"]], ["shoot", "edit"])
        self.assertNotIn("f1", [i["feedback_id"] for i in result["sentiment"]["issues"]])

    def test_personalized_order(self):
        self.data["actions"].append({"id": "aaa", "product_id": "camera", "title": "Other",
                                     "requires": [], "interest": "gardening"})
        self.assertEqual(app.run(self.data)["journey"]["next_actions"], ["setup", "aaa"])

    def test_no_match(self):
        self.data["query"] = "unmatched"
        result = app.run(self.data)
        self.assertEqual(result["semantic"]["hits"], [])
        self.assertEqual(result["journey"]["status"], "unavailable")
        self.assertEqual(result["sentiment"]["issues"], [])

    def test_one_step_unavailable(self):
        self.data["profile"]["completed_actions"] = ["setup", "shoot"]
        result = app.run(self.data)
        self.assertEqual(result["journey"]["next_actions"], ["edit"])
        self.assertEqual(result["journey"]["steps"], [])

    def test_cross_product_prerequisite_blocked(self):
        self.data["actions"][0]["requires"] = ["plant"]
        self.assertEqual(app.run(self.data)["journey"]["status"], "unavailable")

    def test_empty_catalog(self):
        self.data.update(products=[], actions=[], feedback=[])
        self.assertEqual(app.run(self.data)["journey"]["status"], "unavailable")

    def test_sentiment_explanation(self):
        score = app.score_sentiment("not bad helpful")
        self.assertEqual(score["score"], 1)
        self.assertTrue(score["evidence"][0]["negated"])
        self.assertEqual(app.score_sentiment("great broken")["label"], "neutral")
        self.assertEqual(app.score_sentiment("ordinary wording")["evidence"], [])

    def test_severity_dominates_sentiment(self):
        issues = app.run(self.data)["sentiment"]["issues"]
        self.assertEqual([i["feedback_id"] for i in issues], ["f2", "f1", "f3"])
        self.assertEqual(issues[0]["priority"], 400)
        self.assertEqual(issues[1]["priority"], 310)

    def test_invalid_shapes_and_types(self):
        for key, value in (("query", " "), ("limit", True), ("limit", 0),
                           ("products", {}), ("schema_version", True), ("feedback", None)):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.data)
                bad[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(bad)
        with self.assertRaises(app.ValidationError):
            app.run([])

    def test_duplicates_and_missing_refs(self):
        bad = copy.deepcopy(self.data)
        bad["products"].append(copy.deepcopy(bad["products"][0]))
        with self.assertRaises(app.ValidationError):
            app.run(bad)
        self.data["actions"][0]["requires"] = ["nonexistent"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_cycle(self):
        self.data["actions"][0]["requires"] = ["shoot"]
        with self.assertRaisesRegex(app.ValidationError, "cyclic"):
            app.run(self.data)

    def test_invalid_completed_history(self):
        self.data["profile"]["completed_actions"] = ["shoot"]
        with self.assertRaisesRegex(app.ValidationError, "history"):
            app.run(self.data)

    def test_feedback_reference_validation(self):
        self.data["feedback"][0]["product_id"] = "garden"
        with self.assertRaisesRegex(app.ValidationError, "mismatch"):
            app.run(self.data)

    def test_tampered_handoffs(self):
        result = app.run(self.data)
        search = copy.deepcopy(result["semantic"])
        search["hits"][0]["product_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.validate(self.data, "semantic", search)
        recommendation = copy.deepcopy(result["journey"])
        recommendation["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate(self.data, "journey", recommendation, result["semantic"])
        insights = copy.deepcopy(result["sentiment"])
        insights["issues"][0]["priority"] = 0
        with self.assertRaises(app.ValidationError):
            app.validate(self.data, "sentiment", insights,
                         {"semantic": result["semantic"], "journey": result["journey"]})

    def test_pipeline_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        app.run(self.data)
        self.assertEqual(before, self.data)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")],
                                 capture_output=True, text=True, cwd=HERE)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["nonexistent-fixture.json"]):
            process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                                     capture_output=True, text=True, cwd=HERE)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_schema(self):
        for value in ("{", "{}", '{"x":1,"x":2}', '{"x":NaN}', "null"):
            with self.subTest(value=value), patch.object(Path, "read_text", return_value=value):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(buffer.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
