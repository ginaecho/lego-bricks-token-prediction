import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.input = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def behavior(self):
        return app.behavioral(self.input)

    def reviewed(self):
        return app.review(self.behavior())

    def researched(self):
        return app.research(self.reviewed())

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, text=True, capture_output=True, check=False)

    def test_full_pipeline_and_order(self):
        out = app.run_pipeline(self.input)
        self.assertEqual(out["stage"], "semantic")
        self.assertEqual(list(out["results"]), ["behavior", "review", "normal", "semantic"])
        self.assertEqual(out["results"]["semantic"]["hits"][0]["product_id"], "bag")

    def test_no_input_mutation(self):
        before = copy.deepcopy(self.input)
        app.run_pipeline(self.input)
        self.assertEqual(self.input, before)

    def test_deterministic(self):
        self.assertEqual(app.run_pipeline(self.input), app.run_pipeline(self.input))

    def test_purchase_and_half_life(self):
        rows = self.behavior()["results"]["behavior"]["recommendations"]
        self.assertEqual(rows[0], {"product_id": "bag", "score": 3.0})
        self.assertEqual(rows[1], {"product_id": "coat", "score": 0.5})
        self.assertEqual(rows[2]["score"], 0)

    def test_cold_start_popularity(self):
        self.input["data"]["user_id"] = "new-user"
        result = self.behavior()["results"]["behavior"]
        self.assertTrue(result["cold_start"])
        self.assertEqual(result["recommendations"][0]["product_id"], "mug")

    def test_stable_ties(self):
        self.input["data"]["events"] = []
        for p in self.input["data"]["products"]:
            p["popularity"] = 1
        ids = [p["product_id"] for p in self.behavior()["results"]["behavior"]["recommendations"]]
        self.assertEqual(ids, ["bag", "coat", "mug"])

    def test_evidence_and_traceable_gaps(self):
        result = self.reviewed()["results"]["review"]
        self.assertIn("no certification", result["notice"])
        self.assertEqual(result["products"][0]["checks"][0]["status"], "evidence_found")
        self.assertEqual(result["products"][0]["gaps"], ["test-report"])
        self.assertEqual(result["products"][2]["gaps"], ["material", "test-report"])

    def test_keywords_must_share_passage(self):
        self.input["data"]["documents"][0]["text"] = "Recycled content. Fabric construction."
        row = self.reviewed()["results"]["review"]["products"][0]
        self.assertEqual(row["checks"][0]["status"], "gap")

    def test_research_exact_citations(self):
        docs = {d["id"]: d["text"] for d in self.input["data"]["documents"]}
        products = self.researched()["results"]["normal"]["products"]
        self.assertTrue(products[0]["findings"])
        for p in products:
            for finding in p["findings"]:
                c = finding["citation"]
                self.assertEqual(c["quote"], docs[c["document_id"]][c["start"]:c["end"]])
        self.assertEqual(products[0]["findings"][0]["requirement_ids"], ["material"])

    def test_selection_propagates_all_stages(self):
        self.input["data"]["top_k"] = 1
        out = app.run_pipeline(self.input)["results"]
        self.assertEqual([p["product_id"] for p in out["review"]["products"]], ["bag"])
        self.assertEqual([p["product_id"] for p in out["normal"]["products"]], ["bag"])
        self.assertEqual([p["product_id"] for p in out["semantic"]["hits"]], ["bag"])
        self.assertTrue(all(p == ["bag"] for p in out["semantic"]["index"].values()))

    def test_gaps_and_citations_survive_search(self):
        result = app.run_pipeline(self.input)["results"]
        contexts = {p["product_id"]: p for p in result["normal"]["products"]}
        for hit in result["semantic"]["hits"]:
            self.assertEqual(hit["gaps"], contexts[hit["product_id"]]["gaps"])
            self.assertEqual(hit["citations"], [f["citation"] for f in contexts[hit["product_id"]]["findings"]])

    def test_review_evidence_drives_research_without_query(self):
        self.input["data"]["query"] = ""
        result = self.researched()["results"]["normal"]
        self.assertTrue(any("material" in f["requirement_ids"] for f in result["products"][0]["findings"]))

    def test_research_text_drives_search_index(self):
        self.input["data"]["documents"][0]["text"] += " Backpack zirconium."
        out = app.run_pipeline(self.input)
        self.assertEqual(out["results"]["semantic"]["index"]["zirconium"], ["bag"])

    def test_empty_catalog(self):
        for key in ("products", "documents", "events"):
            self.input["data"][key] = []
        out = app.run_pipeline(self.input)["results"]
        self.assertTrue(out["behavior"]["cold_start"])
        self.assertEqual(out["semantic"]["hits"], [])

    def test_no_documents_no_requirements(self):
        self.input["data"]["documents"] = []
        self.input["data"]["requirements"] = []
        out = app.run_pipeline(self.input)["results"]
        self.assertEqual(out["normal"]["products"][0]["findings"], [])
        self.assertEqual(out["semantic"]["hits"][0]["gaps"], [])

    def test_empty_query(self):
        self.input["data"]["query"] = "  "
        hits = app.run_pipeline(self.input)["results"]["semantic"]["hits"]
        self.assertTrue(all(h["score"] == 0 for h in hits))
        self.assertEqual(hits[0]["product_id"], "bag")

    def test_embedding_fixture_changes_ranking(self):
        self.input["data"]["query"] = "unknownterm"
        calls = []
        def fixture(value):
            calls.append(value)
            return [1, 0] if value == "unknownterm" or "Mug" in value else [0, 1]
        hits = app.run_pipeline(self.input, embed=fixture)["results"]["semantic"]["hits"]
        self.assertEqual(hits[0]["product_id"], "mug")
        self.assertEqual(len(calls), 4)

    def test_invalid_embedding_vectors(self):
        for value in ([], [0, 0], [math.nan], [math.inf], [True], "vector", None):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.input, embed=lambda s: value)

    def test_embedding_dimensions(self):
        query = self.input["data"]["query"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.input, embed=lambda s: [1] if s == query else [1, 2])

    def test_embedding_exception(self):
        def fail(value):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(self.input, embed=fail)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.research(self.behavior())

    def test_corrupt_citation_rejected_at_handoff(self):
        envelope = self.reviewed()
        envelope["results"]["review"]["products"][0]["checks"][0]["evidence"][0]["quote"] = "invented"
        with self.assertRaisesRegex(app.ValidationError, "exact extract"):
            app.research(envelope)

    def test_corrupt_gap_rejected_at_handoff(self):
        envelope = self.researched()
        envelope["results"]["normal"]["products"][0]["gaps"] = []
        with self.assertRaisesRegex(app.ValidationError, "lost review gaps"):
            app.semantic_search(envelope)

    def test_corrupt_requirement_link_rejected_at_handoff(self):
        envelope = self.researched()
        envelope["results"]["normal"]["products"][0]["findings"][0]["requirement_ids"] = ["test-report"]
        with self.assertRaisesRegex(app.ValidationError, "evidence links mismatch"):
            app.semantic_search(envelope)

    def test_falsey_embedding_callable(self):
        class Fixture:
            def __bool__(self):
                return False

            def __call__(self, value):
                return [1, 0]
        out = app.run_pipeline(self.input, embed=Fixture())
        self.assertEqual(out["results"]["semantic"]["mode"], "lexical+embedding")

    def test_invalid_input_scalars(self):
        for key, value in (("half_life_days", 0), ("half_life_days", math.inf),
                           ("top_k", True), ("top_k", 0), ("query", None)):
            with self.subTest(key=key, value=value):
                item = copy.deepcopy(self.input)
                item["data"][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(item)

    def test_duplicate_ids(self):
        self.input["data"]["products"].append(copy.deepcopy(self.input["data"]["products"][0]))
        with self.assertRaisesRegex(app.ValidationError, "unique"):
            app.run_pipeline(self.input)

    def test_unknown_event_reference(self):
        self.input["data"]["events"][0]["product_id"] = "absent"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.input)

    def test_bad_timestamp(self):
        for value in ("2026-09-23", "not-a-time", "2030-01-01T00:00:00Z"):
            with self.subTest(value=value):
                self.input["data"]["events"][0]["timestamp"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.input)

    def test_invalid_schema(self):
        for key, value in (("schema_version", 2), ("schema_version", True),
                           ("fixture_label", "real data"), ("results", {"normal": {}})):
            with self.subTest(key=key):
                item = copy.deepcopy(self.input)
                item[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(item)

    def test_unicode_offsets(self):
        self.input["data"]["documents"][0]["text"] = "  Café recycled fabric 🌲.\nWaterproof backpack."
        out = self.researched()
        citations = out["results"]["normal"]["products"][0]["findings"]
        self.assertTrue(any("🌲" in f["citation"]["quote"] for f in citations))
        app.validate(out)

    def test_cli_success_one_object(self):
        result = self.run_cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["stage"], "semantic")

    def test_cli_missing_file(self):
        result = self.run_cli("does-not-exist.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_bad_arguments(self):
        for args in ((), ("example_input.json", "extra")):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json(self):
        result = self.run_cli("implementation.py")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_schema(self):
        result = self.run_cli("build_manifest.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_json_duplicate_and_nonstandard_values(self):
        for raw in ('{"a":1,"a":2}', '{"a": NaN}', '{"a": Infinity}'):
            with self.assertRaises(app.ValidationError):
                json.loads(raw, object_pairs_hook=app.unique_keys, parse_constant=app.reject_constant)


if __name__ == "__main__":
    unittest.main()
