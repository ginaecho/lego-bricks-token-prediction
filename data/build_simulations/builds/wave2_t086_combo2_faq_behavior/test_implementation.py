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
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_grounded_faq(self):
        faq = app.faq_stage(self.request)["faq"]
        self.assertEqual(faq["status"], "answered")
        self.assertEqual(faq["answer"], self.request["knowledge_base"][0]["text"])
        self.assertEqual(faq["evidence"][0]["kb_id"], "care-jacket")
        self.assertEqual(faq["product_ids"], ["trail-jacket"])

    def test_explicit_abstention_and_no_faq_boost(self):
        self.request["query"] = "quantum teleportation"
        result = app.run(self.request)
        self.assertEqual(result["faq"]["status"], "abstained")
        self.assertIsNone(result["faq"]["answer"])
        self.assertTrue(result["faq"]["reason"])
        self.assertEqual(result["behavior"]["faq_status"], "abstained")
        self.assertEqual(result["behavior"]["faq_product_ids"], [])
        self.assertTrue(all(row["components"]["faq_score"] == 0
                            for row in result["behavior"]["recommendations"]))

    def test_cross_stage_boost_changes_ranking(self):
        self.request["events"] = []
        self.request["query"] = "unmatched"
        baseline = app.run(self.request)
        self.assertEqual(baseline["behavior"]["recommendations"][0]["product_id"], "ceramic-mug")
        self.request["query"] = "wash trail jacket"
        result = app.run(self.request)
        self.assertEqual(result["behavior"]["recommendations"][0]["product_id"], "trail-jacket")
        self.assertEqual(result["behavior"]["faq_product_ids"], result["faq"]["product_ids"])

    def test_recency_and_purchase_weight(self):
        ranked = app.run(self.request)["behavior"]["recommendations"]
        by_id = {row["product_id"]: row for row in ranked}
        self.assertAlmostEqual(by_id["trail-jacket"]["components"]["history_score"], 1.5)
        self.assertAlmostEqual(by_id["day-pack"]["components"]["history_score"], 2 ** (-1 / 30))
        self.request["events"][1]["type"] = "view"
        changed = app.run(self.request)["behavior"]["recommendations"]
        jacket = next(row for row in changed if row["product_id"] == "trail-jacket")
        self.assertAlmostEqual(jacket["components"]["history_score"], 0.5)

    def test_cold_start_and_deterministic_ties(self):
        self.request["events"] = []
        self.request["knowledge_base"] = []
        for product in self.request["products"]:
            product["popularity"] = 0
        behavior = app.run(self.request)["behavior"]
        self.assertEqual(behavior["mode"], "cold_start")
        self.assertEqual([row["product_id"] for row in behavior["recommendations"]],
                         ["ceramic-mug", "day-pack", "trail-jacket"])

    def test_empty_catalog_and_stopword_query(self):
        for key in ("events", "products", "knowledge_base"):
            self.request[key] = []
        self.request["query"] = "the and is"
        result = app.run(self.request)
        self.assertEqual(result["faq"]["status"], "abstained")
        self.assertEqual(result["behavior"]["recommendations"], [])

    def test_optional_callable_validated_and_isolated(self):
        original = copy.deepcopy(self.request)
        def answerer(evidence):
            answer = evidence[0]["text"]
            evidence[0]["text"] = "Tampered"
            return answer
        self.assertEqual(app.run(self.request, answerer)["faq"]["status"], "answered")
        self.assertEqual(self.request, original)
        with self.assertRaises(app.ValidationError):
            app.run(self.request, lambda evidence: "All products are free.")
        with self.assertRaises(app.ValidationError):
            app.run(self.request, lambda evidence: None)

    def test_abstention_never_calls_answerer(self):
        self.request["knowledge_base"] = []
        def forbidden(evidence):
            self.fail("Answerer must not run without evidence")
        self.assertEqual(app.run(self.request, forbidden)["faq"]["status"], "abstained")

    def test_tampered_handoff_rejected(self):
        valid = app.faq_stage(self.request)
        for target, key, replacement in [
            ("faq", "product_ids", ["day-pack"]),
            ("faq", "answer", "Invented policy"),
            ("faq", "evidence", []),
        ]:
            with self.subTest(key=key):
                handoff = copy.deepcopy(valid)
                handoff[target][key] = replacement
                with self.assertRaises(app.ValidationError):
                    app.behavior_stage(handoff)
        app.behavior_stage(valid)
        self.assertEqual(valid["stage"], "faq")
        self.assertNotIn("behavior", valid)

    def test_invalid_requests(self):
        mutations = [
            lambda r: r.update(query=" "),
            lambda r: r.update(schema_version=True),
            lambda r: r.update(fixture_label="real data"),
            lambda r: r.update(as_of="2026-09-23"),
            lambda r: r["events"][0].update(timestamp="2027-01-01T00:00:00Z"),
            lambda r: r["events"][0].update(product_id="missing"),
            lambda r: r["events"][0].update(type="click"),
            lambda r: r["events"].append(copy.deepcopy(r["events"][0])),
            lambda r: r["products"].append(copy.deepcopy(r["products"][0])),
            lambda r: r["knowledge_base"][0].update(product_ids=["missing"]),
            lambda r: r["settings"].update(half_life_days=0),
            lambda r: r["settings"].update(top_k=True),
            lambda r: r["settings"].update(min_match=0),
            lambda r: r["settings"].update(faq_boost=float("nan")),
            lambda r: r["products"][0].update(popularity=2),
            lambda r: r.update(unrecognized=1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                request = copy.deepcopy(self.request)
                mutate(request)
                with self.assertRaises(app.ValidationError):
                    app.run(request)

    def test_defaults_top_k_and_no_input_mutation(self):
        original = copy.deepcopy(self.request)
        app.run(self.request)
        self.assertEqual(self.request, original)
        del self.request["settings"]
        self.assertEqual(len(app.run(self.request)["behavior"]["recommendations"]), 3)
        self.request["settings"] = {"top_k": 1}
        self.assertEqual(len(app.run(self.request)["behavior"]["recommendations"]), 1)

    def test_timezone_equivalence_and_order_invariance(self):
        expected = app.run(self.request)["behavior"]
        self.request["as_of"] = "2026-09-23T14:00:00+02:00"
        self.request["events"].reverse()
        self.request["products"].reverse()
        self.request["knowledge_base"].reverse()
        self.assertEqual(app.run(self.request)["behavior"], expected)

    def test_retrieval_threshold_and_limit(self):
        self.request["query"] = "wash unknown"
        self.request["settings"]["min_match"] = 0.51
        self.assertEqual(app.run(self.request)["faq"]["status"], "abstained")
        self.request["settings"]["min_match"] = 0.5
        self.assertEqual(app.run(self.request)["faq"]["status"], "answered")
        entry = self.request["knowledge_base"][0]
        self.request["knowledge_base"] = [dict(entry, id=str(i)) for i in range(5)]
        self.assertEqual([item["kb_id"] for item in app.run(self.request)["faq"]["evidence"]],
                         ["0", "1", "2"])

    def test_final_validation_detects_ranking_tampering(self):
        result = app.run(self.request)
        result["behavior"]["recommendations"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.validate(result, "complete")

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, check=False, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout), app.run(self.request))

    def test_cli_file_and_usage_errors(self):
        for args in ([], ["does-not-exist.json"], ["."]):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                        capture_output=True, text=True, check=False, cwd=ROOT)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_json_and_validation_errors(self):
        for content in ("{bad", '{"schema_version":1,"schema_version":1}', "NaN", "[]", "{}"):
            with self.subTest(content=content):
                stream = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(stream):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
