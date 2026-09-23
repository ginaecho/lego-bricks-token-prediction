import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

from implementation import (
    ValidationError, Validator, behavior_stage, main, run_pipeline,
    semantic_stage, triage_stage,
)


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_triage_route_priority_accountability(self):
        result = triage_stage(self.data)["triage"]
        self.assertEqual((result["category"], result["priority"], result["accountable_team"]),
                         ("billing", "high", "payments"))
        self.assertEqual(result["matched_keywords"], ["refund", "payment"])

    def test_configurable_route(self):
        self.data["config"]["routes"] = [
            {"category": "audio", "keywords": ["wireless headphones"], "priority": "urgent",
             "team": "audio-owner"}]
        result = run_pipeline(self.data)
        self.assertEqual(result["behavior"]["accountable_team"], "audio-owner")
        self.assertEqual(result["behavior"]["priority"], "urgent")
        self.assertEqual(result["semantic"]["category"], "audio")

    def test_default_route_and_token_boundaries(self):
        self.data["ticket"]["text"] = "refunded paymentless"
        self.assertEqual(triage_stage(self.data)["triage"]["category"], "general")

    def test_route_tie_uses_config_order(self):
        self.data["ticket"]["text"] = "refund broken"
        self.assertEqual(triage_stage(self.data)["triage"]["category"], "billing")

    def test_index_ranking_and_category_boost(self):
        result = semantic_stage(triage_stage(self.data))
        self.assertEqual(result["semantic"]["candidates"][0]["product_id"], "p2")
        self.assertFalse(result["semantic"]["embedding_used"])

    def test_embedding_fixture(self):
        observed = []
        def embed(texts):
            observed.extend(texts)
            return [[1, 0], [1, 0], [0, 1], [0, 0]]
        result = run_pipeline(self.data, embed)
        self.assertEqual(len(observed), 4)
        self.assertEqual(observed[0], self.data["ticket"]["text"])
        self.assertTrue(result["semantic"]["embedding_used"])
        self.assertEqual(result["semantic"]["candidates"][0]["product_id"], "p1")

    def test_invalid_embeddings(self):
        cases = [None, [], [[1]] * 3, [[1], [1, 0], [1], [1]],
                 [[float("nan")]] * 4, [[True]] * 4, [[], [], [], []]]
        for vectors in cases:
            with self.subTest(vectors=vectors), self.assertRaises(ValidationError):
                run_pipeline(self.data, lambda texts: vectors)

    def test_embedding_exception_is_validation_error(self):
        def broken(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(ValidationError, "callable failed"):
            run_pipeline(self.data, broken)

    def test_cross_stage_propagation(self):
        result = run_pipeline(self.data)
        self.assertEqual(result["triage"]["query"], result["semantic"]["query"])
        candidates = {r["product_id"]: r for r in result["semantic"]["candidates"]}
        for row in result["behavior"]["recommendations"]:
            self.assertEqual(row["semantic_score"],
                             candidates[row["product_id"]]["semantic_score"])
        self.assertEqual(result["behavior"]["ticket_id"], self.data["ticket"]["id"])

    def test_tampered_triage_handoff_rejected(self):
        envelope = triage_stage(self.data)
        envelope["triage"]["query"] = "unvalidated query"
        with self.assertRaises(ValidationError):
            semantic_stage(envelope)

    def test_tampered_search_handoff_rejected(self):
        envelope = semantic_stage(triage_stage(self.data))
        envelope["semantic"]["candidates"][0]["product_id"] = "unknown"
        with self.assertRaises(ValidationError):
            behavior_stage(envelope)

    def test_purchase_outweighs_browse(self):
        self.data["events"] = [
            {"product_id": "p1", "action": "browse", "timestamp": self.data["as_of"]},
            {"product_id": "p3", "action": "purchase", "timestamp": self.data["as_of"]}]
        self.data["config"]["final_limit"] = 3
        rows = {r["product_id"]: r for r in run_pipeline(self.data)["behavior"]["recommendations"]}
        self.assertGreater(rows["p3"]["personalization_score"], rows["p1"]["personalization_score"])

    def test_recency_half_life(self):
        self.data["config"]["final_limit"] = 3
        self.data["events"] = [
            {"product_id": "p1", "action": "browse", "timestamp": self.data["as_of"]},
            {"product_id": "p3", "action": "browse", "timestamp": "2026-08-24T12:00:00Z"}]
        rows = {r["product_id"]: r for r in run_pipeline(self.data)["behavior"]["recommendations"]}
        self.assertAlmostEqual(rows["p1"]["personalization_score"], 0.25)
        self.assertAlmostEqual(rows["p3"]["personalization_score"], 1 / 6)

    def test_cold_start_popularity(self):
        self.data["events"] = []
        self.data["config"]["final_limit"] = 3
        result = run_pipeline(self.data)["behavior"]
        self.assertTrue(result["cold_start"])
        rows = {r["product_id"]: r for r in result["recommendations"]}
        self.assertAlmostEqual(rows["p1"]["personalization_score"], 0.08)

    def test_empty_catalog(self):
        self.data["catalog"], self.data["events"] = [], []
        result = run_pipeline(self.data)
        self.assertEqual(result["semantic"]["candidates"], [])
        self.assertEqual(result["behavior"]["recommendations"], [])

    def test_limits_do_not_reintroduce_excluded_products(self):
        self.data["config"].update(search_limit=1, final_limit=3)
        result = run_pipeline(self.data)
        self.assertEqual(len(result["behavior"]["recommendations"]), 1)
        self.assertEqual(result["behavior"]["recommendations"][0]["product_id"],
                         result["semantic"]["candidates"][0]["product_id"])

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(run_pipeline(self.data), run_pipeline(self.data))
        self.assertEqual(original, self.data)

    def test_equal_scores_sort_by_id(self):
        self.data["events"] = []
        self.data["ticket"]["text"] = "unmatched"
        self.data["config"]["cold_start_weight"] = 0
        self.data["catalog"].reverse()
        result = run_pipeline(self.data)
        self.assertEqual([r["product_id"] for r in result["semantic"]["candidates"]],
                         ["p1", "p2", "p3"])

    def test_invalid_numeric_config(self):
        for key, value in [("search_limit", True), ("final_limit", 0),
                           ("half_life_days", 0), ("behavior_weight", float("inf")),
                           ("cold_start_weight", -1), ("unknown_setting", 1)]:
            with self.subTest(key=key), self.assertRaises(ValidationError):
                data = copy.deepcopy(self.data)
                data["config"][key] = value
                run_pipeline(data)

    def test_invalid_events(self):
        for field, value in [("product_id", "missing"), ("action", "click"),
                             ("timestamp", "2027-01-01T00:00:00Z"),
                             ("timestamp", "2026-01-01T00:00:00"),
                             ("timestamp", "not a date")]:
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                data = copy.deepcopy(self.data)
                data["events"][0][field] = value
                run_pipeline(data)

    def test_invalid_schema_and_catalog(self):
        cases = [None, [], {}, {**self.data, "schema_version": True}]
        duplicate = copy.deepcopy(self.data)
        duplicate["catalog"].append(copy.deepcopy(duplicate["catalog"][0]))
        cases.append(duplicate)
        blank = copy.deepcopy(self.data)
        blank["ticket"]["text"] = "!!!"
        cases.append(blank)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                run_pipeline(data)

    def test_final_handoff_validation(self):
        result = run_pipeline(self.data)
        result["behavior"]["recommendations"][0]["score"] += 1
        with self.assertRaises(ValidationError):
            Validator.envelope(result, "behavior")

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        process = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in [(), ("missing-synthetic-input.json",),
                     ("example_input.json", "extra")]:
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for contents in ("{invalid", "null", '{"schema_version": 999}'):
            with self.subTest(contents=contents):
                with patch("builtins.open", mock_open(read_data=contents)), \
                     patch("builtins.print") as output:
                    self.assertEqual(main(["synthetic-invalid.json"]), 2)
                    self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")


if __name__ == "__main__":
    unittest.main()
