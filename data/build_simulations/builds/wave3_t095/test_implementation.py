"""Synthetic fixture tests. No network, temporary files, or third-party packages."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def pipeline(self):
        return app.run_pipeline(self.request)

    def result(self, stage):
        return self.pipeline()["results"][stage]

    def test_integrated_stage_order(self):
        output = self.pipeline()
        self.assertEqual(output["completed_stages"], list(app.STAGES))
        self.assertEqual(output["status"], "ok")
        self.assertEqual(app.validate(output, "handoff", self.request, "deep"), output)

    def test_faq_is_exactly_grounded(self):
        result = self.result("faq")
        source = self.request["knowledge_base"][0]
        self.assertEqual(result["answer"], source["answer"])
        self.assertEqual(result["citations"], [{"source_id": source["id"], "quote": source["answer"]}])

    def test_faq_abstains_and_propagates(self):
        self.request["question"] = "Does this store repair telescopes?"
        output = self.pipeline()["results"]
        self.assertIsNone(output["faq"]["answer"])
        self.assertEqual(output["faq"]["citations"], [])
        self.assertEqual(output["adaptive"]["support_context"]["answer_status"], "abstained")
        self.assertIn("support_followup", [s["id"] for s in output["adaptive"]["steps"]])
        self.assertEqual(output["deep"]["unresolved_questions"][0]["question"], self.request["question"])

    def test_ambiguous_faq_abstains(self):
        self.request["question"] = "returns shipping"
        self.assertEqual(self.result("faq")["answer_status"], "abstained")

    def test_empty_knowledge_base_abstains(self):
        self.request["knowledge_base"] = []
        self.assertEqual(self.result("faq")["answer_status"], "abstained")

    def test_novice_prerequisites_and_explanations(self):
        steps = self.result("adaptive")["steps"]
        self.assertEqual(steps[0]["status"], "ready")
        self.assertTrue(all(s["status"] == "locked" for s in steps[1:]))
        prior = set()
        for step in steps:
            self.assertTrue(set(step["prerequisites"]) <= prior)
            self.assertTrue(step["explanation"])
            prior.add(step["id"])

    def test_expert_skips_known_basics(self):
        self.request["customer"]["experience"] = "expert"
        output = self.pipeline()["results"]
        self.assertEqual(output["adaptive"]["steps"][0]["status"], "satisfied_by_experience")
        self.assertEqual(output["basic:recommend"]["onboarding_next_steps"],
                         ["choose_budget", "policy_review"])

    def test_onboarding_explains_consent_gated_preferences(self):
        steps = self.result("adaptive")["steps"]
        self.assertIn("6500 cents", steps[1]["explanation"])
        self.assertIn("bags", steps[3]["explanation"])
        self.request["customer"]["consent"]["ccpa_personalization"] = False
        steps = self.result("adaptive")["steps"]
        self.assertNotIn("6500", steps[1]["explanation"])
        self.assertNotIn("bags", steps[3]["explanation"])

    def test_consented_discovery_uses_preferences_and_events(self):
        result = self.result("basic:recommend")
        self.assertEqual(result["products"][0]["sku"], "SYN-TOTE-101")
        self.assertEqual(result["products"][0]["score"], 14)
        self.assertEqual(result["products"][1]["score"], 11)

    def test_each_consent_is_required(self):
        for key in ("gdpr_personalization", "ccpa_personalization"):
            with self.subTest(key=key):
                request = copy.deepcopy(self.request)
                request["customer"]["consent"][key] = False
                output = app.run_pipeline(request)
                self.assertEqual(output["personalization_mode"], "generic")
                self.assertEqual(output["results"]["adaptive"]["interest_scores"], {})
                self.assertEqual(output["results"]["adaptive"]["experience_used"], "unspecified")
                self.assertTrue(all(p["score"] == 0 for p in output["results"]["basic:recommend"]["products"]))

    def test_no_consent_ignores_personal_data(self):
        self.request["customer"]["consent"]["gdpr_personalization"] = False
        baseline = self.pipeline()
        self.request["customer"]["preferences"] = {"categories": ["accessories"], "max_price_cents": 0}
        self.request["customer"]["experience"] = "expert"
        self.request["clickstream"] = []
        self.assertEqual(self.pipeline(), baseline)

    def test_budget_is_hard_limit(self):
        self.request["customer"]["preferences"]["max_price_cents"] = 2400
        self.assertEqual([p["sku"] for p in self.result("basic:recommend")["products"]], ["SYN-TOTE-101"])

    def test_zero_budget_produces_no_selection(self):
        self.request["customer"]["preferences"]["max_price_cents"] = 0
        output = self.pipeline()["results"]
        self.assertEqual(output["basic:recommend"]["selection_status"], "no_eligible_products")
        self.assertEqual(output["deep"]["researched_products"], [])
        self.assertEqual(output["deep"]["unresolved_questions"][0]["attribute"], "selection")

    def test_out_of_stock_and_basket_exclusion(self):
        skus = {p["sku"] for p in self.result("basic:recommend")["products"]}
        self.assertNotIn("SYN-SCARF-404", skus)
        self.assertNotIn("SYN-MUG-202", skus)

    def test_catalog_price_stock_and_basket_total(self):
        self.request["catalog"]["products"][0]["price_cents"] = 2397
        self.request["catalog"]["products"][0]["stock"] = 19
        output = self.pipeline()
        self.assertEqual(output["basket"]["total_cents"], 1500)
        lookup = {p["sku"]: p for p in self.request["catalog"]["products"]}
        for stage, field in (("basic:recommend", "products"), ("deep", "researched_products")):
            for product in output["results"][stage][field]:
                self.assertEqual(product["price_cents"], lookup[product["sku"]]["price_cents"])
                self.assertEqual(product["stock"], lookup[product["sku"]]["stock"])

    def test_faq_support_sources_flow_through_all_stages(self):
        output = self.pipeline()["results"]
        self.assertEqual(output["adaptive"]["support_context"], output["basic:recommend"]["support_context"])
        self.assertEqual(output["deep"]["support_sources"], ["synthetic-kb-returns"])

    def test_discovery_limits_research_scope(self):
        self.request["customer"]["preferences"]["max_price_cents"] = 2400
        output = self.pipeline()["results"]
        self.assertEqual([p["sku"] for p in output["deep"]["researched_products"]], ["SYN-TOTE-101"])
        self.assertNotIn("synthetic-doc-pack", output["deep"]["document_ids"])

    def test_multi_document_synthesis(self):
        finding = next(f for f in self.result("deep")["findings"]
                       if f["sku"] == "SYN-TOTE-101" and f["attribute"] == "material")
        self.assertEqual(finding["value"], "cotton canvas")
        self.assertEqual(finding["support"], "multiple_documents")
        self.assertEqual(len(finding["evidence"]), 2)

    def test_disagreement_preserves_both_sides(self):
        research = self.result("deep")
        conflict = research["disagreements"][0]
        self.assertEqual(conflict["attribute"], "durability")
        self.assertEqual({a["value"] for a in conflict["alternatives"]},
                         {"5 kg load limit", "8 kg load limit"})
        self.assertFalse(any(f["sku"] == "SYN-TOTE-101" and f["attribute"] == "durability"
                             for f in research["findings"]))
        self.assertTrue(any(q["sku"] == "SYN-TOTE-101" and q["attribute"] == "durability"
                            for q in research["unresolved_questions"]))

    def test_missing_evidence_is_unresolved_not_invented(self):
        self.request["documents"] = []
        research = self.result("deep")
        self.assertEqual(research["findings"], [])
        self.assertEqual(research["disagreements"], [])
        self.assertEqual(len(research["unresolved_questions"]), 6)

    def test_evidence_quotes_are_grounded(self):
        research = self.result("deep")
        texts = {d["id"]: d["text"] for d in self.request["documents"]}
        entries = list(research["findings"])
        for conflict in research["disagreements"]:
            entries.extend(conflict["alternatives"])
        for entry in entries:
            for evidence in entry["evidence"]:
                self.assertIn(evidence["quote"], texts[evidence["document_id"]])
                self.assertTrue(evidence["synthetic"])

    def test_unknown_quote_is_rejected(self):
        self.request["documents"][0]["claims"][0]["quote"] = "Invented statement."
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_claim_value_must_be_in_quote(self):
        self.request["documents"][0]["claims"][0]["value"] = "gold"
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_evidence_cannot_be_reassigned_to_another_sku_or_attribute(self):
        for field, value in (("sku", "SYN-PACK-303"), ("attribute", "care")):
            with self.subTest(field=field):
                request = copy.deepcopy(self.request)
                request["documents"][0]["claims"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_reviews_endorsements_and_ratings_not_supported(self):
        for kind in ("review", "endorsement"):
            with self.subTest(kind=kind):
                self.request["documents"][0]["kind"] = kind
                with self.assertRaises(app.ValidationError):
                    self.pipeline()
        self.request["documents"][0]["kind"] = "specification"
        self.request["catalog"]["products"][0]["rating"] = 5
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_invalid_money_stock_and_consent_types(self):
        for field, value in (("price_cents", -1), ("price_cents", 2.99),
                             ("price_cents", True), ("stock", -1), ("stock", False)):
            with self.subTest(field=field, value=value):
                request = copy.deepcopy(self.request)
                request["catalog"]["products"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)
        self.request["customer"]["consent"]["gdpr_personalization"] = "yes"
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_basket_oversell_and_customer_mismatch(self):
        self.request["basket"]["items"][0]["quantity"] = 9
        with self.assertRaises(app.ValidationError):
            self.pipeline()
        self.request["basket"]["items"][0]["quantity"] = 1
        self.request["basket"]["customer_id"] = "another-person"
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_duplicate_and_unknown_skus(self):
        request = copy.deepcopy(self.request)
        request["catalog"]["products"].append(copy.deepcopy(request["catalog"]["products"][0]))
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(request)
        self.request["clickstream"][0]["sku"] = "UNKNOWN"
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_invalid_clickstream_date_and_identity(self):
        for field, value in (("timestamp", "yesterday"), ("timestamp", "2026-09-24"),
                             ("customer_id", "another-person"), ("event", "endorsement")):
            with self.subTest(field=field):
                request = copy.deepcopy(self.request)
                request["clickstream"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_empty_catalog_is_valid(self):
        self.request["catalog"]["products"] = []
        self.request["basket"]["items"] = []
        self.request["clickstream"] = []
        self.request["documents"] = []
        self.assertEqual(self.result("basic:recommend")["selection_status"], "no_eligible_products")

    def test_missing_fields_and_non_synthetic_input(self):
        request = copy.deepcopy(self.request)
        del request["customer"]["consent"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(request)
        self.request["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            self.pipeline()

    def test_handoffs_reject_tampering(self):
        first = app.faq(self.request)
        first["results"]["faq"]["answer"] = "A fabricated policy"
        with self.assertRaises(app.ValidationError):
            app.adaptive(first, self.request)
        second = app.adaptive(app.faq(self.request), self.request)
        second["results"]["adaptive"]["categories"] = ["not-consented"]
        with self.assertRaises(app.ValidationError):
            app.recommend(second, self.request)
        third = app.recommend(app.adaptive(app.faq(self.request), self.request), self.request)
        third["results"]["basic:recommend"]["products"][0]["price_cents"] = 1
        with self.assertRaises(app.ValidationError):
            app.deep(third, self.request)

    def test_handoffs_reject_out_of_order_and_stale_consent(self):
        first = app.faq(self.request)
        with self.assertRaises(app.ValidationError):
            app.recommend(first, self.request)
        self.request["customer"]["consent"]["ccpa_personalization"] = False
        with self.assertRaises(app.ValidationError):
            app.adaptive(first, self.request)

    def test_determinism_and_no_input_mutation(self):
        saved = copy.deepcopy(self.request)
        first = self.pipeline()
        self.assertEqual(self.request, saved)
        self.assertEqual(first, self.pipeline())

    def test_intermediate_output_does_not_alias_input(self):
        second = app.adaptive(app.faq(self.request), self.request)
        second["results"]["adaptive"]["categories"].append("unexpected")
        self.assertEqual(self.request["customer"]["preferences"]["categories"], ["bags"])

    def test_cli_success_one_json_object(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, "")
        self.assertEqual(len(run.stdout.splitlines()), 1)
        self.assertEqual(json.loads(run.stdout), self.pipeline())

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "nonexistent-input.json")], [str(ROOT)]):
            with self.subTest(args=args):
                run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     cwd=ROOT, capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode, 2)
                self.assertEqual(json.loads(run.stdout)["status"], "error")
                self.assertEqual(run.stderr, "")

    def test_cli_invalid_inputs_without_writing_extra_files(self):
        invalids = [
            b"{", b"\xff", b"[]", b'{"schema_version":1,"schema_version":1}',
            b'{"value":NaN}', b"x" * (app.MAX_BYTES + 1),
            json.dumps({**self.request, "synthetic": False}).encode(),
        ]
        for raw in invalids:
            with self.subTest(size=len(raw)):
                output = io.StringIO()
                with mock.patch.object(Path, "open", return_value=io.BytesIO(raw)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["in-memory-fixture"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
