import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.input = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def output(self):
        return app.run_pipeline(self.input)

    def test_full_pipeline_and_determinism(self):
        original = copy.deepcopy(self.input)
        result = self.output()
        self.assertEqual(result, self.output())
        self.assertEqual(self.input, original)
        self.assertEqual(result["stage"], "semantic")
        self.assertEqual(app.validate(result, "semantic"), result)

    def test_research_preserves_exact_provenance(self):
        result = app.research(self.input)
        for finding, source in zip(result["findings"], result["sources"]):
            self.assertEqual(finding["quote"], source["text"])
            self.assertEqual(finding["sha256"], source["sha256"])
            self.assertEqual(finding["span"], [0, len(source["text"])])
            self.assertEqual(finding["url"], source["url"])

    def test_injected_retriever_fixture(self):
        calls = []
        self.input["fixture_pages"] = {}
        def retrieve(url):
            calls.append(url)
            return "SYNTHETIC camping warranty two years."
        result = app.run_pipeline(self.input, retriever=retrieve)
        self.assertEqual(calls, [s["url"] for s in self.input["sources"]])
        self.assertEqual(result["sources"][0]["text"], "SYNTHETIC camping warranty two years.")

    def test_allowlist_before_retrieval(self):
        self.input["sources"][0]["url"] = "https://evil.example/solar"
        calls = []
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.input, retriever=lambda url: calls.append(url))
        self.assertEqual(calls, [])

    def test_unsafe_urls(self):
        for url in ("http://catalog.example/a", "https://catalog.example.evil/a",
                    "https://user@catalog.example/a", "https://catalog.example:444/a",
                    "https://catalog.example/a#fragment", "https://catalog.example\\evil/a",
                    "https://catalog.example/a\nb", "https://[broken"):
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.allowed_url(url, self.input["allowlisted_hosts"])

    def test_missing_fixture_and_bad_retriever(self):
        self.input["fixture_pages"] = {}
        with self.assertRaises(app.ValidationError):
            self.output()
        for value in (None, {}, "", " "):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.input, retriever=lambda url: value)

    def test_retriever_failure_is_controlled(self):
        def fail(url):
            raise RuntimeError("fixture unavailable")
        with self.assertRaisesRegex(app.ValidationError, "Retrieval failed"):
            app.run_pipeline(self.input, retriever=fail)

    def test_preference_ranking_and_grounding(self):
        result = self.output()
        solar, desk = result["recommendations"]
        self.assertEqual((solar["product_id"], solar["score"]), ("solar-lamp", 3))
        self.assertEqual((desk["product_id"], desk["score"]), ("desk-lamp", 1))
        self.assertEqual(solar["explanations"][0]["finding_ids"], ["finding:solar"])

    def test_exclusions_propagate_through_every_derived_stage(self):
        self.input["preferences"]["exclude_product_ids"] = ["desk-lamp"]
        result = self.output()
        for key in ("recommendations", "reviews"):
            self.assertEqual([item["product_id"] for item in result[key]], ["solar-lamp"])
        for key in ("documents", "hits"):
            self.assertEqual([item["product_id"] for item in result["search"][key]], ["solar-lamp"])
        self.assertNotIn("fuel-lamp", sum(result["search"]["index"].values(), []))

    def test_case_insensitive_tag_exclusion(self):
        self.input["preferences"]["exclude_tags"] = ["FUEL", "LIGHTING"]
        result = self.output()
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["reviews"], [])
        self.assertEqual(result["search"]["documents"], [])
        self.assertEqual(result["search"]["index"], {})
        self.assertEqual(result["search"]["hits"], [])

    def test_no_interests_stable_ties(self):
        self.input["preferences"]["interests"] = []
        result = self.output()
        self.assertEqual([r["product_id"] for r in result["recommendations"]], ["desk-lamp", "solar-lamp"])
        self.assertTrue(all(r["score"] == 0 and r["explanations"] == [] for r in result["recommendations"]))

    def test_limit_propagates(self):
        self.input["preferences"]["limit"] = 1
        result = self.output()
        self.assertEqual(len(result["reviews"]), 1)
        self.assertEqual(len(result["search"]["documents"]), 1)

    def test_review_support_gap_and_not_applicable(self):
        result = self.output()
        solar, desk = result["reviews"]
        self.assertEqual(solar["checks"][0]["status"], "keyword_evidence_found")
        self.assertEqual(solar["checks"][0]["evidence_finding_ids"], ["finding:solar"])
        self.assertEqual(solar["checks"][1]["status"], "gap")
        self.assertEqual(solar["checks"][1]["missing_keywords"], ["waterproof"])
        self.assertEqual(solar["checks"][1]["checked_finding_ids"], ["finding:solar"])
        self.assertEqual(desk["checks"][1]["status"], "not_applicable")
        self.assertIn("not certification", solar["notice"])
        self.assertEqual(solar["gap_count"], 1)

    def test_empty_requirements(self):
        self.input["requirements"] = []
        self.assertTrue(all(r["checks"] == [] and r["gap_count"] == 0 for r in self.output()["reviews"]))

    def test_search_index_and_review_linkage(self):
        result = self.output()
        self.assertEqual(result["search"]["index"]["camping"], ["solar-lamp"])
        hit = result["search"]["hits"][0]
        self.assertEqual(hit["product_id"], "solar-lamp")
        self.assertEqual(hit["gap_count"], 1)
        self.assertEqual(hit["finding_ids"], ["finding:solar"])

    def test_gap_words_not_indexed_as_claims(self):
        self.input["search"]["query"] = "waterproof"
        result = self.output()
        self.assertEqual(result["search"]["hits"], [])
        self.assertNotIn("waterproof", result["search"]["index"])

    def test_search_threshold_and_limit(self):
        self.input["search"]["limit"] = 1
        self.assertEqual(len(self.output()["search"]["hits"]), 1)
        self.input["search"]["min_score"] = 1
        self.assertEqual(self.output()["search"]["hits"], [])

    def test_embedding_fixture_interface(self):
        def embed(value):
            return [1, 0] if "camping" in value else [0, 1]
        result = app.run_pipeline(self.input, embedder=embed)
        self.assertEqual(result["search"]["mode"], "injected_embedding")
        self.assertEqual([h["product_id"] for h in result["search"]["hits"]], ["solar-lamp"])
        self.assertEqual(result["search"]["hits"][0]["score"], 1)

    def test_invalid_embedding_vectors(self):
        for value in ([], [True], [float("nan")], [float("inf")], ["x"], None):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.input, embedder=lambda text: value)

    def test_embedding_dimension_mismatch(self):
        vectors = iter([[1, 0], [1]])
        with self.assertRaisesRegex(app.ValidationError, "dimension"):
            app.run_pipeline(self.input, embedder=lambda text: next(vectors))

    def test_zero_and_negative_embeddings(self):
        result = app.run_pipeline(self.input, embedder=lambda text: [0, 0])
        self.assertEqual(result["search"]["hits"], [])
        self.assertEqual(app.cosine([1, 0], [-1, 0]), 0)

    def test_wrong_stage_and_tampered_provenance_rejected(self):
        web = app.research(self.input)
        with self.assertRaises(app.ValidationError):
            app.review(web)
        web["findings"][0]["quote"] = "invented claim"
        with self.assertRaises(app.ValidationError):
            app.recommend(web)

    def test_tampered_recommendation_and_review_rejected(self):
        interests = app.recommend(app.research(self.input))
        interests["recommendations"][0]["explanations"][0]["finding_ids"] = ["finding:desk"]
        with self.assertRaises(app.ValidationError):
            app.review(interests)
        reviewed = app.review(app.recommend(app.research(self.input)))
        reviewed["reviews"][0]["checks"][1]["status"] = "keyword_evidence_found"
        with self.assertRaises(app.ValidationError):
            app.semantic_search(reviewed)

    def test_multiple_sources_merge_without_losing_provenance(self):
        source = dict(self.input["sources"][0], id="solar-extra", url="https://catalog.example/extra")
        self.input["sources"].append(source)
        self.input["fixture_pages"][source["url"]] = "SYNTHETIC fixture: Waterproof housing."
        result = self.output()
        self.assertEqual(result["recommendations"][0]["finding_ids"], ["finding:solar", "finding:solar-extra"])
        self.assertEqual(result["reviews"][0]["gap_count"], 0)
        self.assertEqual(result["reviews"][0]["checks"][1]["evidence_finding_ids"], ["finding:solar-extra"])

    def test_conflicting_metadata_and_duplicate_identifiers(self):
        source = dict(self.input["sources"][0], id="other", url="https://catalog.example/other", title="Different")
        self.input["sources"].append(source)
        with self.assertRaises(app.ValidationError):
            self.output()
        self.input["sources"][-1] = self.input["sources"][0]
        with self.assertRaises(app.ValidationError):
            self.output()

    def test_invalid_requests(self):
        mutations = [
            lambda x: x.update(synthetic=False),
            lambda x: x.update(schema_version="2"),
            lambda x: x.update(unexpected=True),
            lambda x: x.update(sources=[]),
            lambda x: x["preferences"].update(limit=True),
            lambda x: x["preferences"].update(limit=0),
            lambda x: x["preferences"].update(interests=["!!!"]),
            lambda x: x["search"].update(query=" "),
            lambda x: x["search"].update(min_score=float("nan")),
            lambda x: x["requirements"][0].update(keywords=[]),
            lambda x: x["requirements"].append(copy.deepcopy(x["requirements"][0])),
        ]
        for mutation in mutations:
            value = copy.deepcopy(self.input)
            mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(app.ValidationError):
                app.run_pipeline(value)

    def test_cli_success_single_json_object(self):
        proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                               str(HERE / "example_input.json")], capture_output=True, text=True, cwd=HERE)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(HERE / "does-not-exist.json")]):
            proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")] + args,
                                  capture_output=True, text=True, cwd=HERE)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_malformed_duplicate_and_invalid_json(self):
        for raw in ("{", '{"a":1,"a":2}', "[]", '{"schema_version":"bad"}',
                    json.dumps(dict(self.input, synthetic=False))):
            with self.subTest(raw=raw), patch("builtins.open", mock_open(read_data=raw)):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
