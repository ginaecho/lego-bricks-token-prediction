import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def test_synthesis_and_disagreement(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["research"]["summary"],
                         {"finding_count": 3, "disagreement_count": 1, "unresolved_count": 2})
        finding = next(f for f in result["research"]["findings"] if f["topic"] == "durability")
        self.assertEqual(finding["state"], "disputed")
        self.assertEqual([c["claim_id"] for c in finding["citations"]], ["SYN-C1", "SYN-C3"])

    def test_catalog_and_basket_are_exact(self):
        result = app.run(self.data)
        expected = sorted(self.data["catalog"]["products"], key=lambda p: p["sku"])
        self.assertEqual(result["catalog_snapshot"]["products"], expected)
        self.assertEqual(result["basket"]["total_cents"], 4800)
        self.assertEqual(result["basket"]["lines"][0]["unit_price_cents"], 2400)

    def test_determinism_no_mutation_and_order_independence(self):
        original = copy.deepcopy(self.data)
        first = app.run(self.data)
        self.assertEqual(self.data, original)
        self.data["documents"].reverse()
        self.data["catalog"]["products"].reverse()
        self.assertEqual(first, app.run(self.data))

    def test_consent_required(self):
        self.data["customer"]["personalization_requested"] = True
        for consent in ({"gdpr": False, "ccpa": False},
                        {"gdpr": True, "ccpa": False}, {"gdpr": False, "ccpa": True}):
            with self.subTest(consent=consent):
                self.data["customer"]["consent"] = consent
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_personalization_with_consent(self):
        self.data["customer"]["personalization_requested"] = True
        self.data["customer"]["consent"] = {"gdpr": True, "ccpa": True}
        research = app.run(self.data)["research"]
        self.assertEqual(research["selected_skus"], ["SYN-TOTE-01"])
        self.assertEqual(research["out_of_scope_question_ids"], ["SYN-Q2"])

    def test_generic_ignores_behavior_for_selection(self):
        first = app.run(self.data)
        self.data["clickstream"] = []
        self.data["customer"]["persona"] = "Different synthetic persona"
        self.assertEqual(app.run(self.data), first)

    def test_empty_documents_and_basket(self):
        self.data["documents"] = []
        self.data["basket"]["items"] = []
        result = app.run(self.data)
        self.assertEqual(result["basket"]["total_cents"], 0)
        self.assertEqual(len(result["research"]["coverage_gaps"]), 2)
        self.assertTrue(all(f["state"] == "no_evidence" for f in result["research"]["findings"]))

    def test_same_source_is_not_independent(self):
        self.data["documents"][1]["source_id"] = self.data["documents"][0]["source_id"]
        finding = next(f for f in app.run(self.data)["research"]["findings"] if f["topic"] == "ease_of_use")
        self.assertEqual(finding["state"], "single_source")

    def test_uncertain_evidence_stays_unresolved(self):
        self.data["documents"][1]["claims"][1]["value"] = "uncertain"
        finding = next(f for f in app.run(self.data)["research"]["findings"] if f["topic"] == "ease_of_use")
        self.assertEqual(finding["state"], "inconclusive")

    def test_no_review_or_endorsement(self):
        for kind in ("review", "endorsement"):
            self.data["documents"][0]["claims"][0]["evidence_kind"] = kind
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_non_catalog_price_claim_rejected(self):
        self.data["documents"][0]["claims"][0]["topic"] = "price"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_stock_and_quantity_constraints(self):
        for sku, quantity in (("SYN-TOTE-01", 9), ("SYN-MUG-02", 1),
                              ("SYN-TOTE-01", 0), ("SYN-TOTE-01", True)):
            with self.subTest(sku=sku, quantity=quantity):
                self.data["basket"]["items"] = [{"sku": sku, "quantity": quantity}]
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_bad_schema_and_types(self):
        for data in ([], {}, dict(self.data, unexpected=True),
                     dict(self.data, synthetic=False), dict(self.data, schema_version=True)):
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaises(app.ValidationError):
                    app.run(data)
        for price in (-1, 1.2, True, "12"):
            self.data["catalog"]["products"][0]["price_cents"] = price
            with self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_unknown_sku_duplicate_ids_and_bad_timestamp(self):
        for mutation in ("sku", "duplicate", "timestamp", "customer"):
            data = copy.deepcopy(self.data)
            if mutation == "sku":
                data["documents"][0]["claims"][0]["sku"] = "UNKNOWN"
            elif mutation == "duplicate":
                data["documents"][1]["document_id"] = data["documents"][0]["document_id"]
            elif mutation == "timestamp":
                data["clickstream"][0]["timestamp"] = "2026-09-24"
            else:
                data["clickstream"][0]["customer_id"] = "SOMEONE-ELSE"
            with self.subTest(mutation=mutation), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_output_validation_rejects_fabrication_and_catalog_drift(self):
        for mutation in ("price", "stock", "evidence"):
            result = app.run(self.data)
            if mutation == "price":
                result["catalog_snapshot"]["products"][0]["price_cents"] += 1
            elif mutation == "stock":
                result["catalog_snapshot"]["products"][0]["stock"] += 1
            else:
                result["research"]["findings"][0]["citations"].append({"claim_id": "FAKE"})
            with self.subTest(mutation=mutation), self.assertRaises(app.ValidationError):
                app.validate(result, "output", self.data)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                 str(HERE / "example_input.json")],
                                capture_output=True, text=True, cwd=HERE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), app.run(self.data))
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], [str(HERE / "nonexistent.json")], ["a", "b"]):
            result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=HERE)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_malformed_duplicate_nonfinite_and_invalid_json(self):
        for content in ('{', '{"a":1,"a":2}', '{"a":NaN}', '[]', '{"a":Infinity}'):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", return_value=io.StringIO(content)), redirect_stdout(output):
                    self.assertEqual(app.main(["fixture.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
