"""All fixtures are synthetic; injected embeddings are fixed local functions."""

import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def test_complete_pipeline(self):
        output = self.run_data()
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["semantic"]["results"][0]["product_id"], "p1")
        self.assertEqual(output["schema_version"], 1)

    def test_normalized_deduplication(self):
        feedback = self.run_data()["feedback"]
        self.assertEqual((feedback["input_count"], feedback["unique_count"],
                          feedback["duplicate_count"]), (4, 3, 1))
        self.assertEqual(feedback["items"][0]["source_ids"], ["f1", "f2"])
        self.assertEqual(feedback["items"][0]["excerpt"], self.data["feedback"][0]["text"])

    def test_deduplication_preserves_customer_and_product_boundaries(self):
        new = copy.deepcopy(self.data["feedback"][0])
        new.update(id="f5", customer_id="another-synthetic-customer")
        self.data["feedback"].append(new)
        new = copy.deepcopy(new)
        new.update(id="f6", product_id="p2")
        self.data["feedback"].append(new)
        self.assertEqual(self.run_data()["feedback"]["unique_count"], 5)

    def test_themes_have_verbatim_traceable_evidence(self):
        feedback = self.run_data()["feedback"]
        originals = {row["id"]: row for row in self.data["feedback"]}
        self.assertIn("quality", feedback["themes"])
        for evidence in feedback["themes"].values():
            for row in evidence:
                self.assertEqual(row["excerpt"], originals[row["feedback_id"]]["text"])

    def test_negation_and_unknown_theme(self):
        self.assertEqual(app.sentiment(app.tokens("not comfortable")), -1)
        self.data["feedback"][0]["text"] = "A descriptive observation"
        self.assertEqual(self.run_data()["feedback"]["items"][0]["themes"], ["other"])

    def test_recency_half_life_and_purchase_weight(self):
        self.data["feedback"] = []
        self.data["products"][0]["tags"] = []
        self.data["products"][1]["tags"] = []
        self.data["interactions"][0]["timestamp"] = self.data["as_of"]
        ranking = {row["product_id"]: row for row in self.run_data()["behavior"]["ranking"]}
        self.assertEqual(ranking["p1"]["event_score"], 3)
        self.assertEqual(ranking["p2"]["event_score"], 0.5)
        self.assertEqual(ranking["p3"]["event_score"], 0)

    def test_tag_transfer_personalizes_unseen_product(self):
        self.data["interactions"] = self.data["interactions"][:1]
        ranking = {row["product_id"]: row for row in self.run_data()["behavior"]["ranking"]}
        self.assertGreater(ranking["p2"]["tag_score"], 0)
        self.assertEqual(ranking["p2"]["event_score"], 0)
        self.assertEqual(ranking["p3"]["tag_score"], 0)

    def test_cold_start_uses_global_events(self):
        self.data["request"]["customer_id"] = "synthetic-newcomer"
        behavior = self.run_data()["behavior"]
        self.assertTrue(behavior["cold_start"])
        row = next(row for row in behavior["ranking"] if row["product_id"] == "p3")
        self.assertEqual(row["interaction_ids"], ["e3"])
        self.assertEqual(row["event_score"], 3)

    def test_no_history_fallback_is_stable(self):
        self.data["feedback"] = []
        self.data["interactions"] = []
        self.data["request"]["query"] = ""
        output = self.run_data()
        self.assertTrue(output["behavior"]["cold_start"])
        self.assertEqual([row["product_id"] for row in output["semantic"]["results"]],
                         ["p1", "p2", "p3"])
        self.assertTrue(all(row["personalization"] == 0.5
                            for row in output["semantic"]["results"]))

    def test_feedback_to_behavior_to_semantic_propagates(self):
        self.data["interactions"] = []
        self.data["feedback"] = []
        self.data["request"]["query"] = "sneakers"
        baseline = self.run_data()
        self.data["feedback"] = [{
            "id": "new-feedback", "customer_id": "synthetic-z", "product_id": "p2",
            "timestamp": self.data["as_of"], "text": "Excellent durable quality",
        }]
        changed = self.run_data()
        self.assertEqual(baseline["semantic"]["results"][0]["product_id"], "p1")
        self.assertEqual(changed["semantic"]["results"][0]["product_id"], "p2")
        result = changed["semantic"]["results"][0]
        self.assertEqual(result["feedback_ids"], ["new-feedback"])
        self.assertEqual(result["themes"], ["quality"])
        self.assertEqual(result["personalization"], changed["behavior"]["ranking"][0]["normalized_score"])
        self.assertGreater(result["score"], baseline["semantic"]["results"][1]["score"])

    def test_synonym_index(self):
        self.data["request"]["query"] = "couch"
        result = self.run_data()["semantic"]
        self.assertEqual([row["product_id"] for row in result["results"]], ["p3"])
        self.assertIn("sofa", result["index"]["p3"])
        self.assertEqual(result["results"][0]["matched_terms"], ["sofa"])

    def test_no_match_does_not_recommend_unrelated_products(self):
        self.data["request"]["query"] = "submarines"
        self.assertEqual(self.run_data()["semantic"]["results"], [])

    def test_limit_and_personalization_disabled(self):
        self.data["request"].update(query="sneakers", limit=1)
        self.data["config"]["personalization_weight"] = 0
        rows = self.run_data()["semantic"]["results"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score"], rows[0]["relevance"])

    def test_injected_embedding_fixture(self):
        self.data["request"]["query"] = "relaxation"
        calls = []

        def embed(texts):
            calls.append(texts)
            return [[1, 0], [0, 1], [0, 1], [1, 0]]

        result = app.run_pipeline(self.data, embedding=embed)["semantic"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)
        self.assertTrue(result["embedding_used"])
        self.assertEqual(result["results"][0]["product_id"], "p3")
        self.assertEqual(result["results"][0]["embedding_score"], 1)

    def test_embedding_validation(self):
        invalid = [
            [], [[1, 0]] * 3, [[0, 0]] * 4, [[math.nan, 1]] * 4,
            [[math.inf, 1]] * 4, [[True, 1]] * 4, [[1, 0], [1], [1, 0], [1, 0]],
            ["vector"] * 4,
        ]
        for vectors in invalid:
            with self.subTest(vectors=vectors), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, embedding=lambda texts: vectors)

    def test_embedding_callable_failure(self):
        def fail(texts):
            raise RuntimeError("synthetic failure")
        with self.assertRaisesRegex(app.ValidationError, "embedding callable failed"):
            app.run_pipeline(self.data, embedding=fail)

    def test_huge_finite_embeddings_are_normalized(self):
        output = app.run_pipeline(self.data, embedding=lambda texts: [[1e308, 1e308]] * len(texts))
        self.assertTrue(output["semantic"]["embedding_used"])
        self.assertTrue(all(math.isfinite(row["score"]) for row in output["semantic"]["results"]))

    def test_embedding_skipped_for_discovery_and_empty_catalog(self):
        def fail(texts):
            self.fail("embedding must not be invoked")
        self.data["request"]["query"] = " "
        self.assertFalse(app.run_pipeline(self.data, embedding=fail)["semantic"]["embedding_used"])
        self.data.update(products=[], feedback=[], interactions=[])
        self.data["request"]["query"] = "shoes"
        self.assertEqual(app.run_pipeline(self.data, embedding=fail)["semantic"]["results"], [])

    def test_empty_catalog_and_feedback(self):
        self.data.update(products=[], feedback=[], interactions=[])
        output = self.run_data()
        self.assertEqual(output["feedback"]["themes"], {})
        self.assertEqual(output["behavior"]["ranking"], [])
        self.assertEqual(output["semantic"]["index"], {})

    def test_deterministic_and_nonmutating(self):
        before = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(self.data, before)
        for collection in ("products", "feedback", "interactions"):
            self.data[collection].reverse()
        self.assertEqual(first, self.run_data())

    def test_invalid_input_cases(self):
        cases = [
            ("schema_version", True), ("schema_version", 2), ("synthetic", False),
            ("as_of", "not-a-date"), ("as_of", "2026-09-23"), ("products", {}),
            ("feedback", None), ("interactions", "events"), ("request", {}),
            ("config", {"half_life_days": 0}), ("config", {"feedback_weight": math.inf}),
            ("config", {"personalization_weight": 2}), ("config", {"unknown": 1}),
        ]
        for key, value in cases:
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_duplicate_ids_unknown_references_future_events_and_action(self):
        cases = [("id", "e2"), ("product_id", "absent"), ("timestamp", "2030-01-01T00:00:00Z"),
                 ("action", "refund"), ("customer_id", ""), ("timestamp", "2026-01-01T00:00:00")]
        for key, value in cases:
            data = copy.deepcopy(self.data)
            data["interactions"][0][key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_invalid_limit_and_feedback_text(self):
        for limit in (True, 0, 101, 1.5):
            self.data["request"]["limit"] = limit
            with self.assertRaises(app.ValidationError):
                self.run_data()
        self.data["request"]["limit"] = 3
        for value in ("", "!!!", None):
            self.data["feedback"][0]["text"] = value
            with self.assertRaises(app.ValidationError):
                self.run_data()

    def test_feedback_handoff_rejects_tampering(self):
        feedback = self.run_data()["feedback"]
        feedback["items"][0]["excerpt"] = "fabricated quotation"
        with self.assertRaisesRegex(app.ValidationError, "verbatim"):
            app.rank_behavior(self.data, feedback)

    def test_behavior_handoff_rejects_tampering(self):
        output = self.run_data()
        output["behavior"]["ranking"][0]["feedback_ids"] = ["unknown"]
        with self.assertRaisesRegex(app.ValidationError, "handoff"):
            app.semantic_search(self.data, output["feedback"], output["behavior"])

    def test_invalid_stage_blocks_downstream(self):
        with patch.object(app, "analyze_feedback", return_value={}), \
                patch.object(app, "rank_behavior") as downstream:
            with self.assertRaises(app.ValidationError):
                self.run_data()
            downstream.assert_not_called()

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_and_argument_errors(self):
        for args in ([], ["not-present.json"], ["a", "b"]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     capture_output=True, text=True, cwd=ROOT)
            with self.subTest(args=args):
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_malformed_json_and_schema_errors(self):
        for document in ("{", "[]", "null", '{"schema_version": 1, "schema_version": 1}',
                         '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{}'):
            stream = io.StringIO()
            with self.subTest(document=document), patch("builtins.open", mock_open(read_data=document)), \
                    redirect_stdout(stream):
                code = app.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_unicode_error_is_json_error(self):
        stream = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("synthetic decoding failure")), \
                redirect_stdout(stream):
            code = app.main(["synthetic-invalid.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_timezone_normalization(self):
        baseline = self.run_data()
        self.data["as_of"] = "2026-09-23T11:00:00+02:00"
        self.data["interactions"][0]["timestamp"] = "2026-09-22T04:00:00-05:00"
        self.assertEqual(baseline, self.run_data())

    def test_utc_overflow_is_validation_error(self):
        for value in ("0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"):
            self.data["as_of"] = value
            stream = io.StringIO()
            with self.subTest(value=value), \
                    patch("builtins.open", mock_open(read_data=json.dumps(self.data))), \
                    redirect_stdout(stream):
                code = app.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
