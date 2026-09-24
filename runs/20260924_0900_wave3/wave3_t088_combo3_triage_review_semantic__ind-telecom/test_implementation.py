"""Offline synthetic fixtures; no generated files or third-party libraries."""

import contextlib
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def intake(self):
        return app.intake(self.payload)[0]

    def reviewed(self):
        return app.review(app.triage(self.intake()))

    def test_integrated_pipeline(self):
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["stage"], "semantic")
        self.assertEqual(len(output["records"]), 2)
        self.assertEqual(output["matches"][0]["ticket_id"], "TKT-SYN-001")
        self.assertIs(app.canonical_validate(output), output)

    def test_triage_network_priority_and_accountable_route(self):
        ticket = app.triage(self.intake())["records"][0]
        self.assertEqual(ticket["triage"], {
            "category": "network", "priority": 1,
            "team": "network_operations", "owner": "fault_desk"})

    def test_triage_billing(self):
        record = app.triage(self.intake())["records"][1]
        self.assertEqual(record["triage"]["category"], "billing")
        self.assertEqual(record["triage"]["priority"], 3)
        self.assertEqual(record["triage"]["owner"], "billing_desk")

    def test_custom_config_and_fallback(self):
        self.payload["config"] = copy.deepcopy(app.DEFAULT_CONFIG)
        self.payload["config"]["categories"][0]["owner"] = "regional_desk"
        self.payload["config"]["categories"][0]["priority"] = 3
        self.payload["config"]["urgent_keywords"] = []
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["records"][0]["triage"]["owner"], "regional_desk")
        self.assertEqual(output["records"][0]["triage"]["priority"], 3)
        self.payload["tickets"][0]["subject"] = "Synthetic question"
        self.payload["tickets"][0]["transcript"] = "SUBSCRIBER: Hello."
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["records"][0]["triage"]["category"], "general")

    def test_category_ties_follow_config_order(self):
        self.payload["tickets"][0]["subject"] = "outage bill"
        self.payload["tickets"][0]["transcript"] = "SUPPORT: Synthetic request."
        record = app.run_pipeline(self.payload)["records"][0]
        self.assertEqual(record["triage"]["category"], "network")

    def test_keyword_matching_uses_whole_words(self):
        self.payload["tickets"][0]["subject"] = "billing"
        self.payload["tickets"][0]["transcript"] = "SUBSCRIBER: Recharging."
        self.assertEqual(app.run_pipeline(self.payload)["records"][0]["triage"]["category"],
                         "general")

    def test_review_traceable_gap(self):
        record = self.reviewed()["records"][0]
        self.assertEqual(record["review"]["gaps"], [{
            "requirement": "diagnostics", "ticket_id": "TKT-SYN-001",
            "owner": "fault_desk", "reason": "missing_evidence"}])
        self.assertEqual(record["review"]["checks"][0]["source_ids"], ["DOC-SYN-001"])
        self.assertFalse(record["review"]["certification_claim"])

    def test_review_complete_is_not_certification(self):
        self.payload["documents"][0]["evidence"]["diagnostics"] = "Synthetic test evidence"
        record = app.run_pipeline(self.payload)["records"][0]
        self.assertEqual(record["review"]["status"], "evidence_present")
        self.assertEqual(record["review"]["gaps"], [])
        self.assertFalse(record["review"]["certification_claim"])

    def test_billing_reason_and_cdr_are_evidence(self):
        record = self.reviewed()["records"][1]
        checks = {row["requirement"]: row for row in record["review"]["checks"]}
        self.assertEqual(checks["usage_record"]["source_ids"], ["CDR-SYN-003"])
        self.assertEqual(checks["adjustment_reason"]["source_ids"], ["TKT-SYN-002"])
        self.assertEqual(record["review"]["status"], "evidence_present")

    def test_cross_stage_propagation_and_no_mutation(self):
        original = copy.deepcopy(self.payload)
        intake = self.intake()
        triage = app.triage(intake)
        reviewed = app.review(triage)
        output = app.semantic_search(reviewed, self.payload["query"])
        self.assertEqual(self.payload, original)
        self.assertNotIn("triage", intake["records"][0])
        self.assertNotIn("review", triage["records"][0])
        self.assertEqual(reviewed["matches"], [])
        self.assertEqual(output["records"][0]["triage"], triage["records"][0]["triage"])
        self.assertEqual(output["matches"][0]["gap_requirements"], ["diagnostics"])
        self.assertEqual(output["matches"][0]["review_status"], "gaps_found")

    def test_stage_order_is_enforced(self):
        with self.assertRaises(app.ValidationError):
            app.review(self.intake())
        with self.assertRaises(app.ValidationError):
            app.semantic_search(app.triage(self.intake()), self.payload["query"])
        with self.assertRaises(app.ValidationError):
            app.triage(self.reviewed())

    def test_tampered_handoffs_are_rejected(self):
        triage = app.triage(self.intake())
        triage["records"][0]["triage"]["owner"] = "wrong_owner"
        with self.assertRaises(app.ValidationError):
            app.review(triage)
        reviewed = self.reviewed()
        reviewed["records"][0]["review"]["gaps"] = []
        with self.assertRaises(app.ValidationError):
            app.semantic_search(reviewed, self.payload["query"])

    def test_search_synonym_without_literal_match(self):
        self.payload["query"]["text"] = "connectivity"
        result = app.run_pipeline(self.payload)
        self.assertEqual(result["matches"][0]["ticket_id"], "TKT-SYN-001")
        self.assertGreater(result["matches"][0]["score"], 0)

    def test_search_empty_result(self):
        self.payload["query"]["text"] = "zoological"
        self.assertEqual(app.run_pipeline(self.payload)["matches"], [])

    def test_search_residency_is_partitioned(self):
        self.payload["query"] = {"text": "invoice", "residency": "US", "limit": 1}
        matches = app.run_pipeline(self.payload)["matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["ticket_id"], "TKT-SYN-002")
        self.assertEqual(matches[0]["residency"], "US")

    def test_embedding_receives_only_residency_partition(self):
        calls = []

        def embed(content):
            calls.append(content)
            return [1.0, 0.25]

        output = app.run_pipeline(self.payload, embed)
        self.assertEqual(len(calls), 2)
        self.assertEqual(output["matches"][0]["mode"], "injected_embedding")
        self.assertAlmostEqual(output["matches"][0]["score"], 1)
        self.assertNotIn("billing", calls[1])
        for content in calls:
            self.assertNotIn("+999", content)
            self.assertNotIn("SUBSCRIBER", content)

    def test_invalid_embedding_vectors(self):
        invalid = ([], [0, 0], [float("nan")], [float("inf")], [True], ["bad"], "bad", None)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.payload, lambda content: value)

    def test_embedding_dimensions_and_exception(self):
        vectors = iter([[1, 0], [1]])
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.payload, lambda content: next(vectors))

        def broken(content):
            raise RuntimeError("private provider details")

        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(self.payload, broken)

    def test_tiny_embedding_vectors_are_normalized_safely(self):
        output = app.run_pipeline(self.payload, lambda content: [1e-300, 1e-300])
        self.assertAlmostEqual(output["matches"][0]["score"], 1.0)

    def test_handoff_booleans_are_not_numeric_aliases(self):
        triaged = app.triage(self.intake())
        triaged["records"][0]["triage"]["priority"] = True
        with self.assertRaises(app.ValidationError):
            app.review(triaged)
        reviewed = self.reviewed()
        reviewed["records"][0]["review"]["checks"][0]["satisfied"] = 1
        with self.assertRaises(app.ValidationError):
            app.semantic_search(reviewed, self.payload["query"])

    def test_deterministic_search_tie_and_limit(self):
        extra = copy.deepcopy(self.payload["tickets"][0])
        extra["id"] = "TKT-SYN-000"
        self.payload["tickets"].append(extra)
        self.payload["query"]["limit"] = 1
        result = app.run_pipeline(self.payload, lambda content: [1, 1])
        self.assertEqual(result["matches"][0]["ticket_id"], "TKT-SYN-000")
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(app.run_pipeline(self.payload), app.run_pipeline(self.payload))

    def test_no_subscriber_content_in_public_output(self):
        self.payload["tickets"][0]["transcript"] += " Name: PrivatePerson; secret@example.test."
        self.payload["documents"][0]["evidence"]["fault_description"] += " PRIVATE-DOCUMENT"
        serialized = json.dumps(app.run_pipeline(self.payload))
        for secret in ("+999000000001", "000000000000001", "PrivatePerson",
                       "secret@example.test", "PRIVATE-DOCUMENT", "Invented duplicate data"):
            self.assertNotIn(secret, serialized)
        self.assertNotIn('"transcript"', serialized)
        self.assertNotIn('"phone"', serialized)
        self.assertNotIn('"imei"', serialized)

    def test_privacy_policy_and_permission(self):
        mutations = [
            lambda p: p["accounts"][0].update(consent=False),
            lambda p: p["policy"].update(purpose="marketing"),
            lambda p: p["policy"].update(retention_days=366),
            lambda p: p["policy"].update(retention_days=True),
            lambda p: p.update(synthetic=False),
            lambda p: p["accounts"][0].update(phone="+12025550123"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                payload = copy.deepcopy(self.payload)
                mutate(payload)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)

    def test_residency_on_every_entity(self):
        for entity in ("accounts", "tickets", "documents"):
            with self.subTest(entity=entity):
                payload = copy.deepcopy(self.payload)
                payload[entity][0]["residency"] = "CA"
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)
        self.payload["cdr_csv"] = self.payload["cdr_csv"].replace(",EU,", ",CA,", 1)
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.payload)

    def test_allowed_but_cross_residency_relationship_denied(self):
        for entity in ("tickets", "documents"):
            with self.subTest(entity=entity):
                payload = copy.deepcopy(self.payload)
                payload[entity][0]["residency"] = "US"
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)
        self.payload["cdr_csv"] = self.payload["cdr_csv"].replace(",EU,", ",US,", 1)
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.payload)

    def test_adjustment_reason_required_even_at_zero(self):
        for reason in (None, "", "   ", 42):
            with self.subTest(reason=reason):
                payload = copy.deepcopy(self.payload)
                payload["tickets"][1]["billing_adjustment"] = {"amount": 0, "reason": reason}
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)
        del self.payload["tickets"][1]["billing_adjustment"]["reason"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.payload)

    def test_nonfinite_and_negative_usage_rejected(self):
        for value in ("nan", "inf", "-1", "1000001", "not-a-number"):
            with self.subTest(value=value):
                payload = copy.deepcopy(self.payload)
                payload["cdr_csv"] = payload["cdr_csv"].replace("184.75", value)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)

    def test_csv_structure_identity_and_duplicates(self):
        original = self.payload["cdr_csv"]
        invalid = [
            original.replace("call_seconds", "duration"),
            original + original.splitlines()[1] + "\n",
            original.replace("632,184.75", "632,184.75,extra"),
            original.replace("632,184.75", "632"),
            original.replace("+999000000001", "+999000000099", 1),
            original.replace("632,184.75", "1.5,184.75"),
            original.replace("632,184.75", '632,"unterminated'),
        ]
        for cdr in invalid:
            with self.subTest(csv=cdr):
                self.payload["cdr_csv"] = cdr
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.payload)

    def test_unknown_reference_and_duplicate_entities(self):
        for entity in ("accounts", "tickets", "documents"):
            with self.subTest(entity=entity):
                payload = copy.deepcopy(self.payload)
                payload[entity].append(copy.deepcopy(payload[entity][0]))
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)
        self.payload["documents"][0]["ticket_id"] = "TKT-UNKNOWN"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.payload)

    def test_blank_evidence_is_gap(self):
        self.payload["documents"][0]["evidence"]["fault_description"] = " \n "
        gaps = app.run_pipeline(self.payload)["records"][0]["review"]["gaps"]
        self.assertEqual({gap["requirement"] for gap in gaps}, {"fault_description", "diagnostics"})

    def test_randomized_synthetic_usage_aggregation(self):
        rng = random.Random(88)
        rows = []
        expected_seconds = 0
        expected_mb = 0
        for index in range(12):
            seconds, mb = rng.randrange(86401), rng.randrange(10000) / 100
            expected_seconds += seconds
            expected_mb += mb
            rows.append(f"CDR-RANDOM-{index},ACC-SYN-001,EU,+999000000001,000000000000001,{seconds},{mb}")
        self.payload["cdr_csv"] = ",".join(app.CDR_FIELDS) + "\n" + "\n".join(rows)
        usage = app.run_pipeline(self.payload)["records"][0]["usage"]
        self.assertEqual(usage["call_seconds"], expected_seconds)
        self.assertAlmostEqual(usage["data_mb"], expected_mb)
        self.assertEqual(len(usage["record_ids"]), 12)

    def test_header_only_cdr_and_empty_dataset(self):
        self.payload["cdr_csv"] = ",".join(app.CDR_FIELDS) + "\n"
        output = app.run_pipeline(self.payload)
        self.assertEqual(output["records"][0]["usage"]["record_ids"], [])
        self.assertEqual(output["records"][1]["review"]["gaps"][0]["requirement"], "usage_record")
        self.payload["accounts"] = []
        self.payload["tickets"] = []
        self.payload["documents"] = []
        self.assertEqual(app.run_pipeline(self.payload)["records"], [])

    def test_invalid_shapes_query_and_config(self):
        mutations = [
            lambda p: p.update(schema_version=True),
            lambda p: p.update(accounts={}),
            lambda p: p["query"].update(text=" "),
            lambda p: p["query"].update(text="12345"),
            lambda p: p["query"].update(limit=True),
            lambda p: p["query"].update(limit=0),
            lambda p: p["query"].update(residency="CA"),
            lambda p: p.update(unexpected="field"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                payload = copy.deepcopy(self.payload)
                mutate(payload)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)
        self.payload["config"] = copy.deepcopy(app.DEFAULT_CONFIG)
        self.payload["config"]["categories"][0]["owner"] = ""
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.payload)

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(json.loads(completed.stdout)["stage"], "semantic")

    def test_cli_errors_are_json_exit_two(self):
        for arguments in ([], ["does-not-exist.json"], ["implementation.py"],
                          ["example_input.json", "extra"]):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py")] + arguments,
                    cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stderr, "")
                self.assertEqual(len(completed.stdout.splitlines()), 1)
                self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_duplicate_json_and_invalid_payload(self):
        for content in ('{"synthetic": true, "synthetic": false}', "null", "[]",
                        '{"accounts":', json.dumps({"schema_version": 1})):
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
