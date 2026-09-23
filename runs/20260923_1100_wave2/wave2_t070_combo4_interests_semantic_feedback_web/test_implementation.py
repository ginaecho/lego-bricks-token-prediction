import contextlib
import copy
import hashlib
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
        with (ROOT / "example_input.json").open(encoding="utf-8") as stream:
            self.data = json.load(stream)

    def run_data(self):
        return app.run_pipeline(self.data)["stages"]

    def assert_invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_full_pipeline_order_and_determinism(self):
        original = copy.deepcopy(self.data)
        result = app.run_pipeline(self.data)
        self.assertEqual(list(result["stages"]), list(app.STAGES))
        self.assertEqual(result, app.run_pipeline(self.data))
        self.assertEqual(self.data, original)
        self.assertEqual(result["status"], "ok")

    def test_interest_ranking_and_grounding(self):
        recommendations = self.run_data()["interests"]["recommendations"]
        self.assertEqual([row["product_id"] for row in recommendations], ["pack-a", "pack-b"])
        self.assertEqual([row["score"] for row in recommendations], [1.0, 0.5])
        self.assertEqual(recommendations[0]["evidence"][0], {"interest": "hiking", "matched_tag": "hiking"})
        self.assertIn("durable", recommendations[0]["explanation"])

    def test_exclusions_propagate_all_stages(self):
        self.data["preferences"]["excluded_ids"] = ["pack-a"]
        stages = self.run_data()
        self.assertEqual([row["product_id"] for row in stages["semantic"]["matches"]], ["pack-b"])
        self.assertEqual([row["feedback_id"] for row in stages["feedback"]["items"]], ["f4"])
        for finding in stages["web"]["findings"]:
            self.assertEqual(finding["product_ids"], ["pack-b"])
            self.assertEqual(finding["feedback_ids"], ["f4"])

    def test_budget_boundary(self):
        self.data["preferences"]["max_price"] = 40
        self.assertEqual([row["product_id"] for row in self.run_data()["interests"]["recommendations"]], ["pack-b"])

    def test_case_insensitive_tag_exclusion(self):
        self.data["preferences"]["excluded_tags"] = ["HIKING"]
        self.assertEqual(self.run_data()["interests"]["recommendations"], [])

    def test_empty_interests_discover_eligible_catalog(self):
        self.data["preferences"]["interests"] = []
        rows = self.run_data()["interests"]["recommendations"]
        self.assertEqual([row["product_id"] for row in rows], ["mug-a", "pack-a", "pack-b"])
        self.assertTrue(all(row["score"] == 0 and not row["evidence"] for row in rows))

    def test_interest_limit_is_downstream_boundary(self):
        self.data["preferences"]["max_results"] = 1
        stages = self.run_data()
        self.assertEqual(stages["feedback"]["selected_product_ids"], ["pack-a"])
        self.assertNotIn("pack-b", {pid for ids in stages["semantic"]["index"].values() for pid in ids})

    def test_semantic_aliases_and_index(self):
        semantic = self.run_data()["semantic"]
        self.assertEqual(semantic["index"]["backpack"], ["pack-a", "pack-b"])
        self.assertEqual(semantic["matches"][0]["score"], 1.0)
        self.assertEqual(semantic["matches"][0]["matched_terms"], ["backpack", "durable", "waterproof"])
        self.assertEqual(semantic["matches"][0]["preference_evidence"],
                         self.run_data()["interests"]["recommendations"][0]["evidence"])

    def test_search_threshold_and_top_k(self):
        self.data["search"]["min_score"] = 0.5
        self.data["search"]["top_k"] = 1
        self.assertEqual(self.run_data()["feedback"]["selected_product_ids"], ["pack-a"])

    def test_no_search_match_no_feedback_or_findings(self):
        self.data["search"]["query"] = "telescope"
        stages = self.run_data()
        self.assertEqual(stages["semantic"]["matches"], [])
        self.assertEqual(stages["feedback"]["items"], [])
        self.assertEqual(stages["web"]["queries"], [])
        self.assertEqual(stages["web"]["findings"], [])

    def test_injected_embedding_fixture(self):
        calls = []

        def fixture_embedder(value):
            calls.append(value)
            return [1, 0] if "Summit" in value or value == self.data["search"]["query"] else [0, 1]

        semantic = app.run_pipeline(self.data, fixture_embedder)["stages"]["semantic"]
        self.assertTrue(semantic["embedding_used"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(semantic["matches"][0]["embedding_score"], 1.0)
        self.assertEqual(semantic["matches"][1]["embedding_score"], 0.0)

    def test_invalid_embedding_vectors(self):
        for bad in ([], [0, 0], [True, 1], [float("nan"), 1], [float("inf")], "vector"):
            with self.subTest(bad=bad), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, lambda value: bad)

    def test_embedding_dimension_and_exception_validation(self):
        count = 0

        def wrong_dimension(value):
            nonlocal count
            count += 1
            return [1] if count == 1 else [1, 1]

        with self.assertRaisesRegex(app.ValidationError, "dimension"):
            app.run_pipeline(self.data, wrong_dimension)
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(self.data, lambda value: 1 / 0)

    def test_very_small_nonzero_embedding_is_valid(self):
        semantic = app.run_pipeline(self.data, lambda value: [1e-300, 0])["stages"]["semantic"]
        self.assertEqual(semantic["matches"][0]["embedding_score"], 1.0)

    def test_feedback_deduplication_preserves_sources(self):
        feedback = self.run_data()["feedback"]
        self.assertEqual(feedback["duplicate_count"], 1)
        self.assertEqual(len(feedback["items"]), 3)
        self.assertEqual(feedback["items"][0]["source_feedback_ids"], ["f1", "f2"])
        self.assertEqual(feedback["items"][0]["rating"], 1)

    def test_same_text_for_different_products_not_duplicate(self):
        self.data["feedback"][3]["text"] = self.data["feedback"][0]["text"]
        feedback = self.run_data()["feedback"]
        self.assertEqual(feedback["duplicate_count"], 1)
        theme = next(row for row in feedback["themes"] if row["theme"] == "durability")
        self.assertEqual(theme["count"], 2)

    def test_feedback_themes_excerpts_and_sentiment(self):
        themes = {row["theme"]: row for row in self.run_data()["feedback"]["themes"]}
        self.assertEqual(themes["comfort"]["count"], 2)
        self.assertEqual(themes["durability"]["sentiment"]["negative"], 1)
        support = themes["durability"]["support"][0]
        self.assertEqual(support["quote"], self.data["feedback"][0]["text"])
        self.assertEqual(support["source_feedback_ids"], ["f1", "f2"])

    def test_other_theme_does_not_invent_findings(self):
        self.data["feedback"] = [{"id": "new", "product_id": "pack-a", "text": "Purple design.", "rating": 3}]
        stages = self.run_data()
        self.assertEqual(stages["feedback"]["themes"][0]["theme"], "other")
        self.assertEqual(stages["web"]["queries"][0]["feedback_ids"], ["new"])
        self.assertEqual(stages["web"]["findings"], [])

    def test_web_provenance_and_feedback_handoff(self):
        web = self.run_data()["web"]
        pages = {page["id"]: page for page in self.data["research"]["pages"]}
        for finding in web["findings"]:
            self.assertIn(finding["quote"], pages[finding["source_id"]]["text"])
            self.assertIn("research.example", finding["url"])
            self.assertTrue(finding["feedback_ids"])
            self.assertNotIn("f5", finding["feedback_ids"])
        durability = next(row for row in web["findings"] if row["theme"] == "durability")
        self.assertEqual(durability["feedback_ids"], ["f1", "f2"])
        self.assertEqual(web["sources"][0]["content_sha256"],
                         hashlib.sha256(pages["source-a"]["text"].encode("utf-8")).hexdigest())
        self.assertNotIn("#", web["sources"][0]["url"])
        self.assertEqual(web["rejected_sources"][0]["reason"], "host_not_allowlisted")

    def test_allowlist_rejects_url_tricks(self):
        urls = [
            "http://research.example/a", "https://research.example.evil.example/a",
            "https://sub.research.example/a", "https://research.example@evil.example/a",
            "https://user@research.example/a", "https://research.example:444/a",
            "https://127.0.0.1/a", "file:///local", "https://research.example\\@evil.example/a",
            "https://research.example/\npath", "https://[broken/a",
            "https://research.example:bad/a", "https://research.example./a",
        ]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.canonical_url(url, ["research.example"])

    def test_canonical_url_deduplication(self):
        duplicate = copy.deepcopy(self.data["research"]["pages"][0])
        duplicate.update(id="duplicate", url="https://RESEARCH.example:443/backpack-guide#other")
        self.data["research"]["pages"].append(duplicate)
        web = self.run_data()["web"]
        self.assertEqual(len(web["sources"]), 2)
        self.assertEqual(web["rejected_sources"][-1]["reason"], "duplicate_url")

    def test_empty_allowlist_and_empty_pages(self):
        self.data["research"]["allowed_hosts"] = []
        self.assertEqual(self.run_data()["web"]["findings"], [])
        self.data["research"]["pages"] = []
        self.assertEqual(self.run_data()["web"]["sources"], [])

    def test_unrelated_page_produces_no_finding(self):
        self.data["research"]["pages"] = [{
            "id": "unrelated", "url": "https://research.example/unrelated",
            "title": "Waterproof durable comfort price", "text": "Astronomy explores distant galaxies."
        }]
        self.assertEqual(self.run_data()["web"]["findings"], [])

    def test_empty_catalog(self):
        self.data["catalog"] = []
        self.data["feedback"] = []
        stages = self.run_data()
        self.assertEqual(stages["interests"]["recommendations"], [])
        self.assertEqual(stages["semantic"]["index"], {})
        self.assertEqual(stages["web"]["findings"], [])

    def test_invalid_shapes_and_unknown_fields(self):
        for key, bad in (("catalog", {}), ("preferences", []), ("feedback", None),
                         ("schema_version", "2.0"), ("fixture_label", "real")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = bad
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        self.data["unexpected"] = True
        self.assert_invalid()

    def test_invalid_numeric_fields(self):
        for bad in (True, -1, float("nan"), float("inf"), "75"):
            with self.subTest(bad=bad):
                self.data["catalog"][0]["price"] = bad
                self.assert_invalid()

    def test_invalid_query_ids_and_allowlist(self):
        cases = [
            ("search", "query", " "),
            ("search", "query", "!!!"),
            ("search", "top_k", False),
            ("search", "min_score", 1.1),
            ("research", "allowed_hosts", ["https://research.example"]),
            ("research", "allowed_hosts", ["127.0.0.1"]),
            ("preferences", "interests", ["hiking", "HIKING"]),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                data = copy.deepcopy(self.data)
                data[section][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_ids_and_unknown_feedback_product(self):
        self.data["catalog"][1]["id"] = "pack-a"
        self.assert_invalid()
        self.data["catalog"][1]["id"] = "pack-b"
        self.data["feedback"][0]["product_id"] = "missing"
        self.assert_invalid()

    def test_tampered_interest_handoff_rejected(self):
        interests = app.interests_stage(self.data)
        interests["recommendations"][0]["evidence"][0]["matched_tag"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.semantic_stage(self.data, interests)

    def test_reordered_or_incomplete_interest_handoff_rejected(self):
        for transform in (lambda rows: rows[::-1], lambda rows: rows[:1]):
            interests = app.interests_stage(self.data)
            interests["recommendations"] = transform(interests["recommendations"])
            with self.assertRaises(app.ValidationError):
                app.semantic_stage(self.data, interests)

    def test_tampered_semantic_handoff_rejected(self):
        interests = app.interests_stage(self.data)
        semantic = app.semantic_stage(self.data, interests)
        semantic["matches"][0]["product_id"] = "pack-c"
        with self.assertRaises(app.ValidationError):
            app.feedback_stage(self.data, semantic, interests)

    def test_tampered_feedback_handoff_rejected(self):
        stages = self.run_data()
        stages["feedback"]["themes"][0]["support"][0]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.web_stage(self.data, stages["feedback"], stages["semantic"], stages["interests"])

    def test_tampered_final_provenance_rejected(self):
        result = app.run_pipeline(self.data)
        result["stages"]["web"]["findings"][0]["url"] = "https://evil.example/"
        with self.assertRaises(app.ValidationError):
            app.validate("output", result, self.data)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        output = json.loads(process.stdout)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(list(output["stages"]), list(app.STAGES))
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_file_and_argument_errors(self):
        for args in ([], ["nonexistent-build-fixture.json"], ["example_input.json", "extra"], ["."]):
            with self.subTest(args=args):
                process = subprocess.run([sys.executable, "-B", "implementation.py", *args],
                                         capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_malformed_and_invalid_json_without_scratch_files(self):
        for payload in ('{', '{"x":1,"x":2}', '{"x":NaN}', '[]', '{}',
                        json.dumps({**self.data, "search": {"query": "x", "top_k": 0, "min_score": 0}})):
            with self.subTest(payload=payload[:30]):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-mocked-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_encoding(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("invalid UTF-8")), contextlib.redirect_stdout(output):
            code = app.main(["synthetic-mocked-input.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
