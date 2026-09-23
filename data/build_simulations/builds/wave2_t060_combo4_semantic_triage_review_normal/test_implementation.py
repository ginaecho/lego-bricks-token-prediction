"""All fixtures are synthetic. No network, temporary directories, or providers."""

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


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def partial(self, count):
        envelope = {"schema_version": "1.0", "synthetic": True,
                    "status": "processing", "input": copy.deepcopy(self.data), "stages": {}}
        for stage in (app.semantic_stage, app.triage_stage, app.review_stage)[:count]:
            stage(envelope)
        return envelope

    def test_complete_pipeline_order_and_status(self):
        result = self.run_data()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), list(app.STAGES))
        app.validate_envelope(result, 4)

    def test_input_unchanged_and_deterministic(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(self.data, original)

    def test_search_relevance_and_provenance(self):
        hits = self.run_data()["stages"]["semantic"]["hits"]
        self.assertEqual([hit["product_id"] for hit in hits], ["product-aurora"])
        self.assertEqual(hits[0]["matched_tokens"], ["aurora", "charger", "travel"])
        self.assertEqual(hits[0]["source_ids"], ["manual-aurora"])

    def test_search_ties_and_limits(self):
        index = app.SearchIndex([("z", "same product"), ("a", "same product")])
        self.assertEqual(index.rank("same", 1)[0][0], "a")

    def test_injected_embedding_enables_synonym_match(self):
        self.data["query"] = "power adapter"
        def fixture_embedding(value):
            return [1, 0] if value == "power adapter" or "charger" in value else [0, 1]
        result = app.run_pipeline(self.data, embedding=fixture_embedding)
        self.assertEqual(result["stages"]["semantic"]["method"], "lexical+embedding")
        self.assertEqual(result["stages"]["triage"]["routes"][0]["product_ids"],
                         ["product-aurora"])

    def test_bad_embedding_values(self):
        for bad in ([], [True], [float("nan")], [float("inf")], ["x"], {}, None):
            with self.subTest(bad=repr(bad)), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, embedding=lambda value: bad)

    def test_embedding_dimension_mismatch(self):
        def fixture_embedding(value):
            return [1] if value == self.data["query"] else [1, 0]
        with self.assertRaisesRegex(app.ValidationError, "dimension"):
            app.run_pipeline(self.data, embedding=fixture_embedding)

    def test_embedding_failure_is_wrapped(self):
        def broken(value):
            raise RuntimeError("synthetic plugin failure")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(self.data, embedding=broken)

    def test_zero_and_large_finite_vectors(self):
        result = app.run_pipeline(self.data, embedding=lambda value: [0, 0])
        self.assertEqual(result["status"], "ok")
        result = app.run_pipeline(self.data, embedding=lambda value: [1e308, -1e308])
        self.assertEqual(result["status"], "ok")

    def test_triage_accountable_route(self):
        route = self.run_data()["stages"]["triage"]["routes"][0]
        self.assertEqual((route["category"], route["priority"], route["owner"]),
                         ("warranty", "high", "synthetic-service-desk"))
        self.assertIn("repair", route["matched_terms"])

    def test_search_context_changes_triage(self):
        self.data["tickets"][0].update(subject="SYNTHETIC help", body="Please advise.")
        route = self.run_data()["stages"]["triage"]["routes"][0]
        self.assertEqual(route["category"], "warranty")
        self.data["query"] = "nonexistent"
        route = self.run_data()["stages"]["triage"]["routes"][0]
        self.assertEqual(route["category"], "general")
        self.assertEqual(route["source_ids"], [])

    def test_route_tie_uses_configuration_order(self):
        self.data["query"] = "nonexistent"
        self.data["tickets"][0].update(subject="warranty delivery", body="Question")
        self.assertEqual(self.run_data()["stages"]["triage"]["routes"][0]["category"], "warranty")

    def test_review_supported_partial_missing(self):
        checks = self.run_data()["stages"]["review"]["checks"]
        self.assertEqual([c["status"] for c in checks], ["supported", "partial", "missing"])
        self.assertEqual(checks[1]["missing_terms"], ["30 days"])
        self.assertEqual(checks[2]["evidence"], [])
        self.assertIsNone(checks[0]["gap"])

    def test_unrelated_sources_cannot_close_gap(self):
        checks = self.run_data()["stages"]["review"]["checks"]
        self.assertEqual(checks[1]["status"], "partial")
        self.assertEqual(checks[2]["status"], "missing")
        for check in checks:
            for evidence in check["evidence"]:
                self.assertEqual(evidence["citation"]["source_id"], "manual-aurora")

    def test_review_requires_one_complete_passage(self):
        self.data["sources"][0]["passages"] = [
            {"id": "a", "text": "Warranty information."},
            {"id": "b", "text": "Repair information."}]
        self.assertEqual(self.run_data()["stages"]["review"]["checks"][0]["status"], "partial")

    def test_multiword_terms_and_word_boundaries(self):
        self.assertTrue(app.term_present("30 DAYS", "Within 30 days."))
        self.assertFalse(app.term_present("repair", "repairable"))
        self.assertFalse(app.term_present("30 days", "30 working days"))

    def test_category_scope_and_multiple_tickets(self):
        self.data["tickets"].append(
            {"id": "ticket-002", "subject": "shipping delivery", "body": "Delivery overdue"})
        result = self.run_data()
        checks = [c for c in result["stages"]["review"]["checks"]
                  if c["ticket_id"] == "ticket-002"]
        self.assertEqual([c["requirement_id"] for c in checks], ["requirement-recycling"])
        self.assertEqual(checks[0]["owner"], "synthetic-logistics-desk")

    def test_review_to_research_propagation(self):
        result = self.run_data()["stages"]
        for check, finding in zip(result["review"]["checks"], result["normal"]["findings"]):
            self.assertEqual(finding["review_status"], check["status"])
            self.assertEqual(finding["unresolved_terms"], check["missing_terms"])
            self.assertEqual(finding["owner"], check["owner"])
            self.assertEqual(finding["priority"], check["priority"])
        self.assertIn("30 days", result["normal"]["findings"][1]["query"])

    def test_research_exact_extracts_and_offsets(self):
        result = self.run_data()
        passages = app.passage_map(self.data)
        for finding in result["stages"]["normal"]["findings"]:
            for retrieved in finding["passages"]:
                c = retrieved["citation"]
                original = passages[(c["source_id"], c["passage_id"])]
                self.assertEqual(retrieved["finding"], original[c["start"]:c["end"]])
        first = result["stages"]["normal"]["findings"][0]["passages"][0]["citation"]
        self.assertGreater(first["start"], 0)
        self.assertEqual(first["quote"], "The warranty includes repair through the service desk.")

    def test_unicode_citations_are_character_offsets(self):
        self.data["sources"][0]["passages"][0]["text"] = "Résumé café. Warranty repair: naïve ✓."
        result = self.run_data()
        citation = result["stages"]["normal"]["findings"][0]["passages"][0]["citation"]
        self.assertEqual(citation["quote"], "Warranty repair: naïve ✓.")
        self.assertEqual(citation["start"], len("Résumé café. "))

    def test_newline_delimited_extracts_without_punctuation(self):
        self.data["sources"][0]["passages"][0]["text"] = "Warranty repair\nOther unrelated notes"
        result = self.run_data()
        citation = result["stages"]["normal"]["findings"][0]["passages"][0]["citation"]
        self.assertEqual(citation["quote"], "Warranty repair")
        self.assertEqual((citation["start"], citation["end"]), (0, 15))

    def test_research_limit(self):
        self.data["config"]["research_limit"] = 1
        findings = self.run_data()["stages"]["normal"]["findings"]
        self.assertEqual(len(findings[1]["passages"]), 1)
        self.assertEqual(findings[1]["passages"][0]["citation"]["passage_id"], "p2")

    def test_empty_collections(self):
        for key in ("products", "tickets", "requirements", "sources"):
            self.data[key] = []
        result = self.run_data()
        self.assertEqual(result["stages"]["semantic"]["hits"], [])
        self.assertEqual(result["stages"]["normal"]["findings"], [])

    def test_no_search_results_no_eligible_evidence(self):
        self.data["query"] = "nonexistent"
        result = self.run_data()["stages"]
        self.assertEqual(result["semantic"]["hits"], [])
        self.assertTrue(all(c["status"] == "missing" for c in result["review"]["checks"]))
        self.assertTrue(all(f["passages"] == [] for f in result["normal"]["findings"]))

    def test_invalid_inputs(self):
        changes = [
            ("schema_version", "2.0"), ("synthetic", False), ("query", ""),
            ("query", "..."), ("products", {}), ("tickets", [None]),
            ("requirements", None), ("config", []), ("query", 23)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_ids_and_unknown_references(self):
        self.data["products"].append(copy.deepcopy(self.data["products"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate id"):
            self.run_data()
        self.data["products"].pop()
        self.data["products"][0]["source_ids"] = ["unknown"]
        with self.assertRaisesRegex(app.ValidationError, "unknown product source"):
            self.run_data()

    def test_config_validation(self):
        for value in (True, 0, -1, 101, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                self.data["config"]["search_limit"] = value
                self.run_data()

    def test_invalid_route_owner_priority_and_terms(self):
        original = copy.deepcopy(self.data["config"]["routing_rules"][0])
        for field, value in (("owner", " "), ("priority", "panic"), ("terms", []),
                             ("terms", ["..."]), ("terms", ["Repair", "repair"])):
            self.data["config"]["routing_rules"][0] = dict(original, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(app.ValidationError):
                self.run_data()

    def test_unknown_fields_rejected(self):
        self.data["pretend_setting"] = True
        with self.assertRaisesRegex(app.ValidationError, "unknown fields"):
            self.run_data()

    def test_wrong_stage_order_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.review_stage(self.partial(1))

    def test_tampered_semantic_handoff_rejected(self):
        envelope = self.partial(1)
        envelope["stages"]["semantic"]["hits"][0]["source_ids"] = ["manual-breeze"]
        with self.assertRaises(app.ValidationError):
            app.triage_stage(envelope)

    def test_tampered_triage_handoff_rejected(self):
        envelope = self.partial(2)
        envelope["stages"]["triage"]["routes"][0]["owner"] = "wrong-owner"
        with self.assertRaises(app.ValidationError):
            app.review_stage(envelope)

    def test_tampered_review_handoff_rejected(self):
        envelope = self.partial(3)
        envelope["stages"]["review"]["checks"][0]["evidence"][0]["citation"]["quote"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.normal_stage(envelope)

    def test_tampered_research_rejected(self):
        envelope = self.run_data()
        envelope["stages"]["normal"]["findings"][0]["passages"][0]["finding"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.validate_envelope(envelope, 4)

    def test_citation_span_validator(self):
        for start, end, quote in ((-1, 2, "ab"), (0, 4, "abc"), (0, 2, "xx"), (True, 2, "b")):
            with self.subTest(start=start, end=end), self.assertRaises(app.ValidationError):
                app.validate_citation(
                    {"source_id": "s", "passage_id": "p", "start": start, "end": end, "quote": quote},
                    {("s", "p"): "abc"}, ["s"])

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_missing_file(self):
        result = self.cli("synthetic-file-does-not-exist.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_usage(self):
        for args in ((), ("example_input.json", "extra")):
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json_and_duplicate_keys(self):
        for raw in ('{broken', '{"x": 1, "x": 2}', '{"x": NaN}', '[]', '{}', 'null'):
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=raw)), redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_read_failure(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=PermissionError("synthetic denial")), redirect_stdout(output):
            code = app.main(["synthetic-denied.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
