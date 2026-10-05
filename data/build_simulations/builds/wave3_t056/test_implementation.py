import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_pipeline_and_catalog_truth(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["stage"], "web")
        self.assertEqual(result["items"][0]["sku"], "FABLE-T01")
        catalog = {p["sku"]: p for p in self.data["catalog"]}
        for item in result["items"]:
            for key in ("price", "stock", "currency"):
                self.assertEqual(item[key], catalog[item["sku"]][key])

    def test_semantic_relevance_and_ties(self):
        items = app.semantic_search(self.data)["items"]
        self.assertEqual([i["sku"] for i in items], ["FABLE-B01", "FABLE-B02", "FABLE-T01"])
        self.assertEqual([i["semantic_score"] for i in items], [1.0, 1.0, 0.5])

    def test_recency_purchase_ranking(self):
        self.data["basket"]["items"] = []
        self.data["customer"]["interests"] = []
        self.data["events"] = [
            dict(id="recent", customer_id=self.data["customer"]["id"], sku="FABLE-B01",
                 type="purchase", timestamp="2026-09-24T09:00:00Z"),
            dict(id="older", customer_id=self.data["customer"]["id"], sku="FABLE-B02",
                 type="purchase", timestamp="2026-08-25T09:00:00Z")]
        result = app.personalize(self.data, app.semantic_search(self.data))
        items = {i["sku"]: i for i in result["items"]}
        self.assertEqual(items["FABLE-B01"]["behavior_score"], 3.375)
        self.assertEqual(items["FABLE-B02"]["behavior_score"], 2.25)

    def test_cold_start_interests(self):
        self.data["events"] = []
        self.data["basket"]["items"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["personalization_mode"], "cold_start")
        tote = next(i for i in result["items"] if i["sku"] == "FABLE-T01")
        self.assertEqual(tote["behavior_score"], 0.2)

    def test_anonymous_cold_start_preserves_search(self):
        self.data["events"] = []
        self.data["basket"]["items"] = []
        self.data["customer"]["interests"] = []
        result = app.run_pipeline(self.data)
        self.assertTrue(all(i["behavior_score"] == 0 for i in result["items"]))

    def test_consent_required(self):
        for law in ("gdpr", "ccpa"):
            with self.subTest(law=law):
                data = copy.deepcopy(self.data)
                data["customer"]["consent"][law] = False
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_disabled_personalization_does_not_use_history(self):
        self.data["personalization"]["enabled"] = False
        self.data["customer"]["consent"] = {"gdpr": False, "ccpa": False}
        result = app.run_pipeline(self.data)
        self.assertEqual(result["personalization_mode"], "disabled")
        self.assertTrue(all(i["behavior_score"] == 0 for i in result["items"]))
        self.assertEqual(result["items"][0]["sku"], "FABLE-B01")

    def test_cross_stage_propagation_and_immutability(self):
        original = copy.deepcopy(self.data)
        semantic = app.semantic_search(self.data)
        saved_semantic = copy.deepcopy(semantic)
        behavior = app.personalize(self.data, semantic)
        saved_behavior = copy.deepcopy(behavior)
        web = app.research(self.data, behavior)
        self.assertEqual(semantic, saved_semantic)
        self.assertEqual(behavior, saved_behavior)
        self.assertEqual(self.data, original)
        self.assertEqual({i["sku"] for i in semantic["items"]}, {i["sku"] for i in web["items"]})
        self.assertEqual([i["sku"] for i in behavior["items"]], [i["sku"] for i in web["items"]])
        for b, w in zip(behavior["items"], web["items"]):
            self.assertEqual(b["score"], w["score"])
            self.assertEqual(b["semantic_score"], w["semantic_score"])
            self.assertEqual(b["behavior_score"], w["behavior_score"])

    def test_provenance_is_verbatim(self):
        result = app.run_pipeline(self.data)
        docs = {d["url"]: d for d in self.data["research"]["documents"]}
        for item in result["items"]:
            self.assertTrue(item["findings"])
            for finding in item["findings"]:
                self.assertEqual(finding["quote"], docs[finding["url"]]["text"])
                self.assertEqual(len(finding["content_sha256"]), 64)

    def test_no_documents_means_no_findings(self):
        self.data["research"]["documents"] = []
        self.assertTrue(all(not i["findings"] for i in app.run_pipeline(self.data)["items"]))

    def test_unrelated_document_not_retrieved(self):
        self.data["research"]["documents"][0]["title"] = "Synthetic material data"
        self.data["research"]["documents"][0]["text"] = "Steel body and screw cap."
        item = next(i for i in app.run_pipeline(self.data)["items"] if i["sku"] == "FABLE-B01")
        self.assertEqual(item["findings"], [])

    def test_allowlist_attacks_rejected(self):
        for url in ("http://specs.example.test/x", "https://specs.example.test.evil.test/x",
                    "https://user@specs.example.test/x", "https://specs.example.test:444/x",
                    "https://127.0.0.1/x", "https://specs.example.test/x#fragment",
                    "https://specs.example.test\\@evil.test/x"):
            with self.subTest(url=url):
                data = copy.deepcopy(self.data)
                data["research"]["documents"][0]["url"] = url
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_reviews_and_commercial_research_claims_rejected(self):
        for content in ("Customer reviews say excellent.", "Celebrity endorsement.",
                        "Price USD 2.", "Stock is 999.", "Costs $10."):
            with self.subTest(content=content):
                data = copy.deepcopy(self.data)
                data["research"]["documents"][0]["text"] = content
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_unknown_fields_cannot_inject_reviews(self):
        self.data["catalog"][0]["reviews"] = ["Great"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_catalog_invalid_values(self):
        for key, value in (("stock", -1), ("stock", True), ("price", float("nan")),
                           ("price", -1), ("price", 1.234), ("currency", "dollars")):
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["catalog"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_basket_stock_validation(self):
        self.data["basket"]["items"] = [{"sku": "FABLE-B02", "quantity": 1}]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_event_validation(self):
        for key, value in (("sku", "unknown"), ("customer_id", "other"), ("type", "like"),
                           ("timestamp", "2027-01-01T00:00:00Z"),
                           ("timestamp", "2026-01-01T00:00:00")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["events"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_catalog_and_events(self):
        for key in ("catalog", "events"):
            data = copy.deepcopy(self.data)
            data[key].append(copy.deepcopy(data[key][0]))
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_empty_search_and_limit(self):
        self.data["query"] = "unfindable"
        self.assertEqual(app.run_pipeline(self.data)["items"], [])
        self.data["query"] = "travel"
        self.data["limit"] = 1
        self.assertEqual(len(app.run_pipeline(self.data)["items"]), 1)

    def test_invalid_query_and_top_level(self):
        for query in ("", " ", "!!!", None):
            self.data["query"] = query
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data)
        with self.assertRaises(app.ValidationError):
            app.run_pipeline([])

    def test_embedding_injection(self):
        calls = []
        def embedding(texts):
            calls.append(texts)
            return [[1, 0], [0, 1], [1, 0], [0, 1]]
        result = app.run_pipeline(self.data, embedding)
        self.assertEqual(len(calls), 1)
        item = next(i for i in result["items"] if i["sku"] == "FABLE-B02")
        self.assertEqual(item["semantic_score"], 2.0)

    def test_invalid_embeddings(self):
        for vectors in ([], [[1]] * 3, [[0, 0]] * 4, [[float("nan")]] * 4,
                        [[True]] * 4, [[1], [1, 2], [1], [1]]):
            with self.subTest(vectors=vectors):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data, lambda texts: vectors)
        def broken(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data, broken)

    def test_tampered_handoffs_rejected(self):
        for key, value in (("price", 0), ("stock", 999), ("score", 100)):
            output = app.semantic_search(self.data)
            output["items"][0][key] = value
            with self.assertRaises(app.ValidationError):
                app.personalize(self.data, output)
        with self.assertRaises(app.ValidationError):
            app.research(self.data, app.semantic_search(self.data))
        output = app.semantic_search(self.data)
        self.data["query"] = "different"
        with self.assertRaises(app.ValidationError):
            app.personalize(self.data, output)

    def test_fabricated_finding_rejected(self):
        output = app.run_pipeline(self.data)
        output["items"][0]["findings"][0]["quote"] = "Invented statement."
        with self.assertRaises(app.ValidationError):
            app.validate_output(output, self.data, "web")

    def test_determinism(self):
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(HERE / "not-present.json")]):
            process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for content in ('{', '{"a":1,"a":2}', '{"value":NaN}', '[]', '{}'):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)):
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    self.assertEqual(app.main(["fixture.json"]), 2)
                    self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
