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

    def output(self):
        return app.run_pipeline(self.data)["data"]

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_full_pipeline(self):
        result = self.output()
        self.assertEqual(result["stage"], "research")
        self.assertIs(app.validate(result, "research"), result)

    def test_documents_reshape_and_exact_money(self):
        docs = app.documents(self.data)["documents"]
        self.assertEqual(docs["order_rows"][0]["line_total_cents"], 2500)
        self.assertEqual(docs["feedback_rows"][0]["text"], self.data["feedback"][0]["text"].strip())
        self.assertEqual(docs["offers"][0]["stock"], 8)

    def test_themes_and_provenance(self):
        result = self.output()
        themes = {t["theme"]: t for t in result["insights"]["themes"]}
        self.assertEqual(themes["delivery"]["feedback_ids"], ["fb-01"])
        self.assertEqual(themes["quality"]["count"], 1)
        self.assertEqual(result["insights"]["feedback_count"], 3)

    def test_metrics_distinguish_baskets_from_sales(self):
        metric = self.output()["insights"]["product_metrics"][0]
        self.assertEqual(metric["completed_units"], 2)
        self.assertEqual(metric["basket_adds"], 1)
        self.assertEqual(metric["purchase_events"], 0)

    def test_active_consent_only_and_no_out_of_stock_offers(self):
        people = self.output()["insights"]["personalization"]
        self.assertEqual([p["customer_id"] for p in people], ["persona-01"])
        self.assertEqual([o["sku"] for o in people[0]["offers"]], ["FICT-MUG-101"])

    def test_withdrawal_overrides_consent(self):
        self.data["customers"][0]["consent"]["withdrawn"] = True
        self.assertEqual(self.output()["insights"]["personalization"], [])

    def test_missing_consent_rejected(self):
        del self.data["customers"][0]["consent"]
        self.invalid()

    def test_truthy_consent_rejected(self):
        self.data["customers"][0]["consent"]["personalization"] = "yes"
        self.invalid()

    def test_research_evidence_and_insufficient_evidence(self):
        answers = self.output()["research"]["answers"]
        self.assertIn("theme:delivery", answers[0]["evidence_ids"])
        self.assertIn("source:src-01", answers[0]["evidence_ids"])
        self.assertEqual(answers[-1]["status"], "insufficient_evidence")
        self.assertEqual(answers[-1]["findings"], [])

    def test_findings_are_extractive_not_invented_endorsements(self):
        result = self.output()["research"]
        evidence = {e["id"]: e for e in result["evidence"]}
        for answer in result["answers"]:
            for finding in answer["findings"]:
                self.assertEqual(finding["text"], evidence[finding["evidence_id"]]["text"])

    def test_feedback_change_propagates_through_all_stages(self):
        self.data["feedback"][0]["text"] = "price expensive"
        result = self.output()
        self.assertEqual(result["documents"]["feedback_rows"][0]["text"], "price expensive")
        themes = {t["theme"]: t for t in result["insights"]["themes"]}
        self.assertEqual(themes["price"]["count"], 2)
        self.assertNotIn("theme:delivery", result["research"]["answers"][0]["evidence_ids"])
        self.assertIn("theme:price", result["research"]["answers"][2]["evidence_ids"])

    def test_documents_tampering_blocked_at_handoff(self):
        output = app.documents(self.data)
        output["documents"]["offers"][0]["stock"] = 99
        with self.assertRaises(app.ValidationError):
            app.insights(output)

    def test_insight_tampering_blocked_at_handoff(self):
        output = app.insights(app.documents(self.data))
        output["insights"]["themes"][0]["count"] += 100
        with self.assertRaises(app.ValidationError):
            app.research(output)

    def test_stage_order_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.research(app.documents(self.data))

    def test_derived_boolean_cannot_impersonate_integer(self):
        output = app.insights(app.documents(self.data))
        output["insights"]["themes"][0]["count"] = True
        with self.assertRaises(app.ValidationError):
            app.research(output)

    def test_research_tampering_rejected(self):
        output = self.output()
        output["research"]["answers"][0]["findings"][0]["text"] = "Invented endorsement"
        with self.assertRaises(app.ValidationError):
            app.validate(output, "research")

    def test_catalog_research_shows_authoritative_price_and_stock(self):
        findings = self.output()["research"]["answers"][2]["findings"]
        mug = next(f for f in findings if f["evidence_id"] == "catalog:FICT-MUG-101")
        self.assertIn("USD 1250 cents", mug["text"])
        self.assertIn("stock 8", mug["text"])

    def test_empty_collections(self):
        self.data["catalog"]["products"] = []
        for key in ("customers", "orders", "clickstream", "feedback", "sources"):
            self.data[key] = []
        output = self.output()
        self.assertEqual(output["insights"]["themes"], [])
        self.assertTrue(all(a["status"] == "insufficient_evidence" for a in output["research"]["answers"]))

    def test_unknown_theme_is_retained(self):
        self.data["feedback"][0]["text"] = "Purple sunflowers"
        themes = {t["theme"]: t for t in self.output()["insights"]["themes"]}
        self.assertEqual(themes["other"]["feedback_ids"], ["fb-01"])

    def test_duplicate_identifiers(self):
        self.data["clickstream"].append(copy.deepcopy(self.data["clickstream"][0]))
        self.invalid()

    def test_unknown_foreign_key(self):
        self.data["feedback"][0]["sku"] = "MISSING"
        self.invalid()

    def test_review_requires_purchase(self):
        self.data["feedback"][0]["order_id"] = None
        self.invalid()

    def test_review_cannot_claim_another_persons_purchase(self):
        self.data["feedback"][0]["customer_id"] = "persona-02"
        self.invalid()

    def test_review_cannot_use_unpurchased_basket(self):
        self.data["orders"][0]["status"] = "basket"
        self.invalid()

    def test_display_price_matches_catalog(self):
        self.data["clickstream"][0]["shown_price_cents"] = 1
        self.invalid()

    def test_display_stock_matches_catalog(self):
        self.data["clickstream"][0]["shown_stock"] = 9
        self.invalid()

    def test_source_catalog_claim_checked(self):
        self.data["sources"][1]["catalog_claims"][0]["stock"] = 100
        self.invalid()

    def test_order_price_matches_catalog(self):
        self.data["orders"][0]["items"][0]["unit_price_cents"] = 2
        self.invalid()

    def test_basket_cannot_exceed_stock(self):
        self.data["orders"][1]["items"][0]["quantity"] = 9
        self.invalid()

    def test_boolean_and_float_money_rejected(self):
        for bad in (True, 12.50, -1):
            with self.subTest(value=bad):
                self.data["catalog"]["products"][0]["price_cents"] = bad
                self.invalid()

    def test_malformed_timestamp(self):
        self.data["clickstream"][0]["timestamp"] = "yesterday"
        self.invalid()

    def test_naive_timestamp_rejected(self):
        self.data["clickstream"][0]["timestamp"] = "2026-09-24T09:00:00"
        self.invalid()

    def test_synthetic_label_required(self):
        self.data["synthetic"] = False
        self.invalid()

    def test_unknown_fields_rejected(self):
        self.data["extra"] = 1
        self.invalid()

    def test_input_not_mutated_and_repeatable(self):
        original = copy.deepcopy(self.data)
        first = self.output()
        self.assertEqual(self.data, original)
        self.assertEqual(first, self.output())

    def test_cli_success_single_json(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "does-not-exist.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main([])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        cases = ("{", '{"a":1,"a":2}', '{"a":NaN}', "[]", '{"schema_version":"2"}')
        for content in cases:
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                    code = app.main(["example_input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_industry_constraint(self):
        self.data["clickstream"][0]["shown_stock"] = -3
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data=json.dumps(self.data))), contextlib.redirect_stdout(output):
            code = app.main(["example_input.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
