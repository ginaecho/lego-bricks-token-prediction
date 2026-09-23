"""Offline synthetic fixtures; no filesystem writes, networks, or provider calls."""

import copy
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
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_all(self):
        return app.run_pipeline(self.payload)

    def cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        return result.returncode, json.loads(result.stdout)

    def test_example_runs_all_stages(self):
        output = self.run_all()
        self.assertEqual(output["stage"], "deep")
        self.assertEqual(output["status"], "ok")
        self.assertIs(app.validate(output, "deep"), output)

    def test_review_exact_citation_offsets(self):
        review = app.document_review(self.payload)["review"]
        self.assertEqual(len(review["checked_evidence"]), 4)
        sources = {doc["id"]: doc["text"] for doc in self.payload["data"]["documents"]}
        for evidence in review["checked_evidence"]:
            self.assertEqual(sources[evidence["document_id"]][
                evidence["quote_start"]:evidence["quote_end"]], evidence["quote"])
        self.assertIn("not certification", review["notice"])

    def test_review_traceable_gaps(self):
        gaps = app.document_review(self.payload)["review"]["gaps"]
        self.assertIn({"requirement_id": "r-warranty", "reason": "no_supporting_evidence",
                       "evidence_ids": []}, gaps)
        conflict = next(gap for gap in gaps if gap["reason"] == "contradictory_evidence")
        self.assertEqual(set(conflict["evidence_ids"]), {"e-guide-delivery", "e-report-delivery"})

    def test_recency_outweighs_old_purchase(self):
        ranking = self.run_all()["behavior"]["ranking"]
        self.assertEqual(ranking[0]["document_id"], "doc-report")
        self.assertEqual(ranking[0]["activity_score"], 1.0)
        self.assertEqual(ranking[1]["activity_score"], 0.75)

    def test_purchase_weight_and_half_life(self):
        self.payload["data"]["events"] = [
            {"id": "x", "document_id": "doc-guide", "kind": "purchase",
             "at": "2026-08-24T12:00:00Z"}]
        row = self.run_all()["behavior"]["ranking"][0]
        self.assertEqual(row["signals"][0]["age_days"], 30.0)
        self.assertEqual(row["activity_score"], 1.5)

    def test_cold_start_coverage_and_stable_ties(self):
        self.payload["data"]["events"] = []
        behavior = self.run_all()["behavior"]
        self.assertTrue(behavior["cold_start"])
        self.assertTrue(all(row["cold_start"] for row in behavior["ranking"]))
        self.assertEqual([row["document_id"] for row in behavior["ranking"]],
                         ["doc-guide", "doc-report", "doc-notes"])

    def test_per_document_cold_start(self):
        behavior = self.run_all()["behavior"]
        self.assertFalse(behavior["cold_start"])
        notes = next(row for row in behavior["ranking"] if row["document_id"] == "doc-notes")
        self.assertTrue(notes["cold_start"])
        self.assertEqual(notes["score"], 0.0)

    def test_gaps_propagate_into_behavior_and_deep(self):
        output = self.run_all()
        self.assertEqual(output["review"]["gaps"], output["behavior"]["review_gaps"])
        deep_gaps = [gap for finding in output["deep"]["findings"] for gap in finding["review_gaps"]]
        self.assertEqual(deep_gaps, output["review"]["gaps"])

    def test_behavior_order_propagates_to_deep_citations(self):
        output = self.run_all()
        order = [row["document_id"] for row in output["behavior"]["ranking"]]
        self.assertEqual(output["deep"]["reading_order"], order)
        delivery = output["deep"]["findings"][0]
        self.assertEqual(delivery["citations"][0]["document_id"], order[0])
        self.assertEqual(delivery["citations"][0]["personalization_score"],
                         output["behavior"]["ranking"][0]["score"])

    def test_disagreement_and_unresolved_questions(self):
        deep = self.run_all()["deep"]
        self.assertEqual(len(deep["disagreements"]), 1)
        self.assertEqual(deep["findings"][0]["conclusion"], "conflicting_evidence")
        self.assertEqual({item["requirement_id"] for item in deep["unresolved_questions"]},
                         {"r-delivery", "r-warranty"})
        reuse = deep["findings"][1]
        self.assertEqual(reuse["conclusion"], "supporting_evidence_only")
        self.assertEqual(len(reuse["source_document_ids"]), 2)

    def test_ranking_never_suppresses_counterevidence(self):
        self.payload["data"]["events"] = [
            {"id": str(i), "document_id": "doc-guide", "kind": "purchase",
             "at": self.payload["data"]["as_of"]} for i in range(25)]
        output = self.run_all()
        self.assertEqual(output["deep"]["reading_order"][0], "doc-guide")
        self.assertEqual(output["deep"]["findings"][0]["conclusion"], "conflicting_evidence")
        self.assertEqual(len(output["deep"]["findings"][0]["citations"]), 2)

    def test_no_documents_produces_gaps_not_crash(self):
        self.payload["data"]["documents"] = []
        self.payload["data"]["events"] = []
        output = self.run_all()
        self.assertEqual(output["behavior"]["ranking"], [])
        self.assertEqual(len(output["deep"]["unresolved_questions"]), 3)

    def test_contradicting_only_is_unresolved(self):
        self.payload["data"]["documents"][0]["evidence"] = []
        output = self.run_all()
        self.assertEqual(output["deep"]["findings"][0]["conclusion"], "contradicting_evidence_only")
        self.assertEqual(output["deep"]["disagreements"], [])

    def test_very_old_event_underflows_safely(self):
        self.payload["data"]["events"][0]["at"] = "0001-01-01T00:00:00Z"
        output = self.run_all()
        guide = next(row for row in output["behavior"]["ranking"] if row["document_id"] == "doc-guide")
        self.assertEqual(guide["activity_score"], 0.0)

    def test_timezone_equivalence(self):
        first = self.run_all()["behavior"]
        self.payload["data"]["as_of"] = "2026-09-23T14:00:00+02:00"
        self.assertEqual(first, self.run_all()["behavior"])

    def test_timestamp_utc_overflow_rejected(self):
        for value in ("0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"):
            with self.subTest(value=value):
                self.payload["data"]["as_of"] = value
                with self.assertRaises(app.ValidationError):
                    self.run_all()

    def test_deterministic_and_input_not_mutated(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(self.run_all(), self.run_all())
        self.assertEqual(original, self.payload)

    def test_invalid_quote(self):
        self.payload["data"]["documents"][0]["evidence"][0]["quote"] = "Absent quotation"
        with self.assertRaises(app.ValidationError):
            self.run_all()

    def test_unknown_references(self):
        for target, key in [
            (self.payload["data"]["events"][0], "document_id"),
            (self.payload["data"]["documents"][0]["evidence"][0], "requirement_id"),
        ]:
            old = target[key]
            target[key] = "unknown"
            with self.assertRaises(app.ValidationError):
                self.run_all()
            target[key] = old

    def test_duplicate_ids(self):
        for name in ("requirements", "documents", "events"):
            with self.subTest(name=name):
                payload = copy.deepcopy(self.payload)
                payload["data"][name].append(copy.deepcopy(payload["data"][name][0]))
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)

    def test_duplicate_evidence_ids(self):
        evidence = self.payload["data"]["documents"][0]["evidence"]
        evidence.append(copy.deepcopy(evidence[0]))
        with self.assertRaises(app.ValidationError):
            self.run_all()

    def test_future_naive_and_bad_timestamps(self):
        for value in ("2027-01-01T00:00:00Z", "2026-01-01T00:00:00", "not-a-date"):
            with self.subTest(value=value):
                self.payload["data"]["events"][0]["at"] = value
                with self.assertRaises(app.ValidationError):
                    self.run_all()

    def test_invalid_types_and_schema(self):
        for key, value in (("schema_version", True), ("schema_version", 2),
                           ("synthetic", False), ("stage", "deep"), ("data", [])):
            with self.subTest(key=key, value=value):
                payload = copy.deepcopy(self.payload)
                payload[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)
        self.payload["data"]["unknown"] = True
        with self.assertRaises(app.ValidationError):
            self.run_all()

    def test_empty_requirements_rejected(self):
        self.payload["data"]["requirements"] = []
        with self.assertRaises(app.ValidationError):
            self.run_all()

    def test_invalid_event_kind_and_stance(self):
        self.payload["data"]["events"][0]["kind"] = []
        with self.assertRaises(app.ValidationError):
            self.run_all()
        self.payload["data"]["events"][0]["kind"] = "browse"
        self.payload["data"]["documents"][0]["evidence"][0]["stance"] = "neutral"
        with self.assertRaises(app.ValidationError):
            self.run_all()

    def test_stages_cannot_be_skipped(self):
        for stage in (app.personalize, app.deep_research):
            with self.assertRaises(app.ValidationError):
                stage(self.payload)

    def test_tampered_review_handoff_rejected(self):
        output = app.document_review(self.payload)
        output["review"]["gaps"] = []
        with self.assertRaises(app.ValidationError):
            app.personalize(output)

    def test_tampered_behavior_handoff_rejected(self):
        output = app.personalize(app.document_review(self.payload))
        output["behavior"]["ranking"][0]["score"] = 999
        with self.assertRaises(app.ValidationError):
            app.deep_research(output)

    def test_tampered_deep_citation_rejected(self):
        output = self.run_all()
        output["deep"]["findings"][0]["citations"][0]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.validate(output, "deep")

    def test_cli_success(self):
        code, output = self.cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(output, self.run_all())

    def test_cli_missing_file(self):
        code, output = self.cli("nonexistent-synthetic-fixture.json")
        self.assertEqual(code, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_bad_arguments(self):
        for args in ((), ("example_input.json", "extra")):
            code, output = self.cli(*args)
            self.assertEqual(code, 2)
            self.assertEqual(output["status"], "error")

    def test_cli_malformed_and_duplicate_json(self):
        from contextlib import redirect_stdout
        from io import StringIO
        for content in ("{", '{"x": 1, "x": 2}', '{"x": NaN}', "null",
                        json.dumps({**self.payload, "synthetic": False})):
            with self.subTest(content=content[:30]):
                output = StringIO()
                with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(output):
                    code = app.main(["in-memory-synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
