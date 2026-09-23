import contextlib
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_pipeline(self):
        return app.run_pipeline(self.request)["stages"]

    def test_integrated_order_and_success(self):
        stages = self.run_pipeline()
        self.assertEqual([stage["stage"] for stage in stages], ["interests", "faq", "semantic", "web"])
        self.assertTrue(all(stage["status"] == "ok" for stage in stages))

    def test_preference_ranking_and_grounded_explanation(self):
        interests = self.run_pipeline()[0]
        self.assertEqual(interests["product_ids"], ["p1", "p2"])
        self.assertEqual(interests["items"][0]["score"], 1)
        self.assertEqual(interests["items"][1]["score"], 0.6)
        self.assertEqual(interests["items"][0]["text"], "Matches your interests: camping (weight 3), solar (weight 2).")

    def test_exclusions_survive_all_handoffs(self):
        for stage in self.run_pipeline():
            self.assertFalse({"p3", "p4"} & set(stage["product_ids"]))
            for source in stage["evidence"]:
                self.assertFalse({"p3", "p4"} & set(source["product_ids"]))

    def test_faq_copies_answer_and_citation(self):
        faq = self.run_pipeline()[1]
        self.assertEqual(faq["items"][0]["text"], self.request["faqs"][0]["answer"])
        self.assertEqual(faq["items"][0]["citations"], ["faq:f1"])

    def test_faq_abstains_without_question_evidence(self):
        self.request["profile"]["question"] = "Quantum banana warranty?"
        stages = self.run_pipeline()
        self.assertEqual(stages[1]["status"], "abstained")
        self.assertTrue(stages[1]["reason"])
        self.assertEqual(stages[1]["query"], stages[0]["query"])
        self.assertEqual(stages[2]["query"], stages[1]["query"])

    def test_interest_scope_controls_faq_eligibility(self):
        self.request["profile"]["excluded_product_ids"].append("p1")
        stages = self.run_pipeline()
        self.assertNotIn("p1", stages[0]["product_ids"])
        self.assertNotIn("f1", [item["id"] for item in stages[1]["items"]])
        self.assertNotIn("https://research.example/lantern", [item["id"] for item in stages[3]["items"]])

    def test_faq_answer_propagates_to_semantic_and_web(self):
        stages = self.run_pipeline()
        self.assertEqual(stages[1]["query"], stages[0]["query"] + " " + self.request["faqs"][0]["answer"])
        self.assertEqual(stages[2]["query"], stages[1]["query"])
        self.assertEqual(stages[3]["query"], stages[2]["query"])
        self.assertEqual(stages[2]["items"][0]["id"], "p1")

    def test_semantic_scope_controls_web(self):
        self.request["settings"]["search_threshold"] = 0.8
        stages = self.run_pipeline()
        self.assertNotIn("p2", stages[2]["product_ids"])
        for item in stages[3]["items"]:
            self.assertTrue(set(item["product_ids"]) <= set(stages[2]["product_ids"]))

    def test_provenance_retained_at_every_handoff(self):
        stages = self.run_pipeline()
        for previous, current in zip(stages, stages[1:]):
            self.assertEqual(current["evidence"][:len(previous["evidence"])], previous["evidence"])
        web = stages[-1]
        source = {entry["id"]: entry for entry in web["evidence"]}
        for finding in web["items"]:
            evidence = source[finding["citations"][0]]
            self.assertEqual(finding["text"], evidence["text"])
            self.assertEqual(finding["id"], evidence["url"])

    def test_empty_interests_abstain_everywhere(self):
        self.request["profile"]["interests"] = []
        for stage in self.run_pipeline():
            self.assertEqual(stage["status"], "abstained")
            self.assertEqual(stage["product_ids"], [])

    def test_all_products_excluded(self):
        self.request["profile"]["excluded_tags"] = ["camping"]
        self.assertTrue(all(stage["status"] == "abstained" for stage in self.run_pipeline()))

    def test_no_faqs_still_searches_grounded_recommendations(self):
        self.request["faqs"] = []
        stages = self.run_pipeline()
        self.assertEqual(stages[1]["status"], "abstained")
        self.assertEqual(stages[2]["status"], "ok")

    def test_no_web_documents_abstains(self):
        self.request["web"]["documents"] = []
        self.assertEqual(self.run_pipeline()[-1]["status"], "abstained")

    def test_index_relevance_and_oov(self):
        index = app.SearchIndex([("a", "solar lantern"), ("b", "blanket")])
        self.assertGreater(index.scores("solar")["a"], index.scores("solar")["b"])
        self.assertEqual(index.scores("unknown"), {"a": 0, "b": 0})

    def test_tie_break_is_deterministic(self):
        self.assertEqual(app.ranked([("z", "solar"), ("a", "solar")], "solar", 0, 2),
                         [("a", 1.0), ("z", 1.0)])
        first = app.run_pipeline(self.request)
        self.assertEqual(first, app.run_pipeline(self.request))

    def test_injected_embeddings_receive_faq_enriched_query(self):
        calls = []

        def embedder(batch):
            calls.append(batch)
            return [[1.0, 0.0] for _ in batch]

        stages = app.run_pipeline(self.request, embedder)["stages"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-1], stages[1]["query"])
        self.assertEqual(stages[2]["status"], "ok")

    def test_invalid_embedding_outputs_rejected(self):
        invalid = [[], [[0, 0]] * 3, [[1], [1, 2], [1]], [[math.nan]] * 3,
                   [[True]] * 3, "not vectors"]
        for vectors in invalid:
            with self.subTest(vectors=vectors), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.request, lambda batch: vectors)

    def test_injected_embedding_changes_search_ranking(self):
        def fixture_embeddings(batch):
            self.assertEqual(len(batch), 3)
            return [[-1, 0], [1, 0], [1, 0]]
        baseline = self.run_pipeline()[2]
        injected = app.run_pipeline(self.request, fixture_embeddings)["stages"][2]
        self.assertEqual(baseline["items"][0]["id"], "p1")
        self.assertEqual(injected["items"][0]["id"], "p2")

    def test_empty_scope_does_not_call_embedder(self):
        self.request["profile"]["interests"] = []
        def forbidden(batch):
            self.fail("No embedding call should be made for an empty index")
        result = app.run_pipeline(self.request, forbidden)
        self.assertEqual(result["stages"][2]["status"], "abstained")

    def test_strict_search_threshold_propagates_empty_web_scope(self):
        self.request["settings"]["search_threshold"] = 1
        stages = self.run_pipeline()
        self.assertEqual(stages[2]["status"], "abstained")
        self.assertEqual(stages[3]["status"], "abstained")
        self.assertEqual(stages[3]["product_ids"], [])

    def test_mixed_source_with_excluded_product_is_not_exposed(self):
        self.request["faqs"][0]["product_ids"].append("p3")
        self.request["web"]["documents"][0]["product_ids"].append("p3")
        stages = self.run_pipeline()
        self.assertNotIn("f1", [entry["id"] for entry in stages[1]["items"]])
        self.assertNotIn("https://research.example/lantern",
                         [entry["id"] for entry in stages[3]["items"]])

    def test_empty_catalog_is_valid_and_abstains(self):
        self.request["products"] = []
        self.request["faqs"] = []
        self.request["web"]["documents"] = []
        self.request["profile"]["excluded_product_ids"] = []
        self.assertTrue(all(stage["status"] == "abstained" for stage in self.run_pipeline()))

    def test_embedding_exception_wrapped(self):
        def failed(batch):
            raise RuntimeError("fixture error")
        with self.assertRaisesRegex(app.ValidationError, "embedding callable failed"):
            app.run_pipeline(self.request, failed)

    def test_unallowlisted_and_ambiguous_urls_rejected(self):
        invalid = ["http://research.example/a", "https://evil.example/a",
                   "https://research.example.evil.example/a",
                   "https://evil.research.example/a", "https://user@research.example/a",
                   "https://research.example:444/a", "https://research.example/a#fragment",
                   "https://research.example\\@evil.example/a",
                   "https://research.example/\npath", "https://research.example:/a"]
        for url in invalid:
            request = copy.deepcopy(self.request)
            request["web"]["documents"][0]["url"] = url
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)

    def test_https_default_port_accepted(self):
        self.request["web"]["documents"][0]["url"] = "https://research.example:443/lantern"
        self.assertEqual(app.run_pipeline(self.request)["status"], "ok")

    def test_unknown_fields_and_missing_fields(self):
        for field, value in [("unexpected", 1), ("profile", None), ("schema_version", True)]:
            request = copy.deepcopy(self.request)
            request[field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)
        del self.request["profile"]
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_bad_numeric_settings_and_weights(self):
        for value in [True, -1, math.inf, math.nan, "1", 10 ** 400]:
            request = copy.deepcopy(self.request)
            request["profile"]["interests"][0]["weight"] = value
            with self.subTest(value=str(value)), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)
        self.request["settings"]["limit"] = 0
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_duplicate_ids_and_unknown_references(self):
        request = copy.deepcopy(self.request)
        request["products"].append(request["products"][0])
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(request)
        self.request["faqs"][0]["product_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            self.run_pipeline()

    def test_case_insensitive_preferences_and_exclusions(self):
        self.request["profile"]["interests"][0]["tag"] = " CAMPING "
        self.request["profile"]["excluded_tags"] = ["FUEL"]
        self.assertEqual(self.run_pipeline()[0]["product_ids"], ["p1", "p2"])

    def test_forged_upstream_grounding_is_rejected(self):
        interests = app.run_interests(app.validate_input(self.request))
        interests["items"][0]["text"] = "Unverified best product."
        with self.assertRaises(app.ValidationError):
            app.run_faq(self.request, interests)

    def test_forged_upstream_exclusion_is_rejected(self):
        interests = app.run_interests(app.validate_input(self.request))
        interests["product_ids"].append("p3")
        with self.assertRaises(app.ValidationError):
            app.run_faq(self.request, interests)

    def test_input_is_not_mutated(self):
        original = copy.deepcopy(self.request)
        self.run_pipeline()
        self.assertEqual(self.request, original)

    def test_cli_success_actual_subprocess(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], cwd=ROOT,
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_wrong_arguments(self):
        for arguments in [[], ["no-such-fixture.json"], ["one", "two"]]:
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + arguments,
                                     cwd=ROOT, capture_output=True, text=True, check=False)
            with self.subTest(arguments=arguments):
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_malformed_json_duplicate_keys_and_invalid_schema(self):
        for content in ["{", '{"x":1,"x":2}', '{"x": NaN}', '[]', '{}']:
            output = io.StringIO()
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)):
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
