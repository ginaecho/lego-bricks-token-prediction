import contextlib
import copy
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

    def run_data(self):
        return app.run_pipeline(self.data)["stages"]

    def test_end_to_end(self):
        stages = self.run_data()
        self.assertEqual(list(stages), ["semantic", "web", "review", "behavior"])
        self.assertEqual(stages["behavior"]["candidates"][0]["product_id"], "pack-a")

    def test_search_ranking_and_index(self):
        found = self.run_data()["semantic"]["candidates"]
        self.assertEqual([c["product_id"] for c in found], ["pack-a", "pack-b"])
        self.assertEqual(found[0]["matched_tokens"], ["backpack", "recycled", "trail"])
        self.assertGreater(found[0]["semantic_score"], found[1]["semantic_score"])

    def test_limit_propagates(self):
        self.data["limit"] = 1
        for stage in self.run_data().values():
            self.assertEqual(len(stage["candidates"]), 1)

    def test_no_matches(self):
        self.data["query"] = "telescope"
        self.assertTrue(all(not s["candidates"] for s in self.run_data().values()))

    def test_empty_catalog(self):
        self.data.update(products=[], events=[])
        self.assertEqual(self.run_data()["behavior"]["candidates"], [])

    def test_injected_embedding(self):
        self.data["query"] = "cupboard"
        seen = []
        def embed(texts):
            seen.extend(texts)
            return [[1, 0], [0, 1], [0, 1], [1, 0]]
        output = app.run_pipeline(self.data, embed)["stages"]
        self.assertEqual(output["behavior"]["candidates"][0]["product_id"], "mug-c")
        self.assertEqual(len(seen), 4)

    def test_invalid_embeddings(self):
        for result in ([], [[1]] * 3, [[1], [1, 2], [1], [1]], [[float("nan")]] * 4,
                       [[True]] * 4, [[float("inf")]] * 4):
            with self.subTest(result=result), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, lambda texts: result)

    def test_embedding_exception(self):
        def fail(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(self.data, fail)

    def test_zero_embeddings(self):
        result = app.run_pipeline(self.data, lambda texts: [[0, 0] for _ in texts])
        self.assertEqual(result["status"], "ok")

    def test_provenance(self):
        web = self.run_data()["web"]["candidates"][0]
        docs = {d["url"]: d["text"] for d in self.data["documents"]}
        self.assertEqual(len(web["findings"]), 2)
        for finding in web["findings"]:
            self.assertEqual(finding["quote"], docs[finding["url"]][finding["start"]:finding["end"]])

    def test_missing_url_gap(self):
        candidate = self.run_data()["review"]["candidates"][1]
        self.assertEqual(candidate["missing_urls"], ["https://research.example/missing"])
        self.assertEqual(candidate["coverage"], 0)
        self.assertTrue(all(c["status"] == "gap" for c in candidate["checks"]))

    def test_minimum_distinct_sources(self):
        self.data["requirements"][0]["min_sources"] = 2
        self.data["documents"][0]["text"] += " This uses recycled fabric again."
        check = self.run_data()["review"]["candidates"][0]["checks"][0]
        self.assertEqual(check["status"], "gap")
        self.assertEqual(check["gap"]["sources_found"], 1)

    def test_empty_requirements(self):
        self.data["requirements"] = []
        candidate = self.run_data()["review"]["candidates"][0]
        self.assertEqual(candidate["coverage"], 1)
        self.assertEqual(candidate["checks"], [])

    def test_review_propagates_into_behavior(self):
        stages = self.run_data()
        reviewed = {c["product_id"]: c for c in stages["review"]["candidates"]}
        for candidate in stages["behavior"]["candidates"]:
            for key in ("checks", "coverage", "findings", "sources", "semantic_score"):
                self.assertEqual(candidate[key], reviewed[candidate["product_id"]][key])

    def test_review_coverage_changes_final_score(self):
        before = self.run_data()["behavior"]["candidates"][0]["final_score"]
        self.data["documents"][0]["text"] = "Synthetic unrelated source."
        after = next(c for c in self.run_data()["behavior"]["candidates"] if c["product_id"] == "pack-a")
        self.assertLess(after["final_score"], before)

    def test_cold_start(self):
        self.data["events"] = []
        for candidate in self.run_data()["behavior"]["candidates"]:
            self.assertEqual(candidate["personalization_mode"], "cold_start")
            self.assertEqual(candidate["behavior_score"], 0)

    def test_history_outside_candidates_is_cold(self):
        self.data["events"] = [{"product_id": "mug-c", "kind": "purchase", "timestamp": self.data["now"]}]
        self.assertTrue(all(c["personalization_mode"] == "cold_start"
                            for c in self.run_data()["behavior"]["candidates"]))

    def test_purchase_and_recency_weights(self):
        def score(kind, when):
            self.data["events"] = [{"product_id": "pack-a", "kind": kind, "timestamp": when}]
            return self.run_data()["behavior"]["candidates"][0]["behavior_score"]
        recent = score("browse", self.data["now"])
        older = score("browse", "2026-08-24T10:00:00Z")
        purchase = score("purchase", self.data["now"])
        self.assertGreater(recent, older)
        self.assertGreater(purchase, recent)
        self.assertAlmostEqual(older, 1 / 3, places=7)

    def test_bad_urls(self):
        for url in ("http://research.example/a", "https://research.example.evil/a",
                    "https://user@research.example/a", "file:///secret", "https://research.example:444/a",
                    "https://research.example/a#fragment", "https://research.example\\evil/a"):
            with self.subTest(url=url):
                data = copy.deepcopy(self.data)
                data["products"][0]["source_urls"] = [url]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_unallowlisted_document(self):
        self.data["documents"].append({"url": "https://evil.example/a", "text": "untrusted"})
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_invalid_inputs(self):
        for key, value in (("query", ""), ("query", "!!!"), ("limit", True), ("limit", 0),
                           ("half_life_days", 0), ("half_life_days", float("nan")),
                           ("synthetic", False), ("schema_version", 2), ("schema_version", True),
                           ("now", "2026-09-23"), ("products", None), ("events", "bad")):
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_product(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_duplicate_document(self):
        self.data["documents"].append(copy.deepcopy(self.data["documents"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_bad_events(self):
        for key, value in (("product_id", "absent"), ("kind", "click"), ("timestamp", "2027-01-01T00:00:00Z")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["events"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_tampered_handoff(self):
        stages = self.run_data()
        stages["web"]["candidates"][0]["findings"][0]["quote"] = "invented"
        with self.assertRaisesRegex(app.ValidationError, "quote mismatch"):
            app.document_review(stages["web"], self.data)

    def test_tampered_review(self):
        stages = self.run_data()
        stages["review"]["candidates"][1]["coverage"] = 1
        with self.assertRaisesRegex(app.ValidationError, "coverage"):
            app.personalize(stages["review"], self.data)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.document_review(self.run_data()["semantic"], self.data)

    def test_tampered_missing_url(self):
        stages = self.run_data()
        stages["web"]["candidates"][1]["missing_urls"] = []
        with self.assertRaisesRegex(app.ValidationError, "missing URL"):
            app.document_review(stages["web"], self.data)

    def test_tampered_gap(self):
        stages = self.run_data()
        stages["review"]["candidates"][1]["checks"][0]["gap"] = None
        with self.assertRaisesRegex(app.ValidationError, "review gap"):
            app.personalize(stages["review"], self.data)

    def test_tampered_search_tokens(self):
        stages = self.run_data()
        stages["semantic"]["candidates"][0]["matched_tokens"] = ["fabricated"]
        with self.assertRaisesRegex(app.ValidationError, "matched tokens"):
            app.web_research(stages["semantic"], self.data)

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(self.data, original)

    def test_tie_breaks_by_id(self):
        self.data["query"] = "backpack"
        self.data["requirements"] = []
        self.data["events"] = []
        self.data["products"].reverse()
        self.assertEqual([c["product_id"] for c in self.run_data()["behavior"]["candidates"]],
                         ["pack-a", "pack-b"])

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_file_error(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "missing-input.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage_error(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_json_and_validation_errors(self):
        for body in ("{", "null", '{"schema_version":1,"schema_version":1}', '{"x":NaN}',
                     '{"schema_version":1,"synthetic":true,"query":""}'):
            with self.subTest(body=body):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=body)), contextlib.redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
