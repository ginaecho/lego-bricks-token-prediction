"""Standard-library tests; all data and retrieval are synthetic and offline."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)["stages"]

    def reject(self):
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def cli(self, *args):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        return result.returncode, json.loads(result.stdout)

    def cli_text(self, text):
        path = ROOT / ("_test_input_" + uuid.uuid4().hex + ".json")
        try:
            with path.open("x", encoding="utf-8") as target:
                target.write(text)
            return self.cli(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_example_integrates_all_four_stages(self):
        stages = self.run_data()
        self.assertEqual(list(stages), ["behavior", "review", "guided", "web"])
        self.assertEqual(stages["review"]["summary"]["gap_count"], 2)
        self.assertEqual(stages["guided"]["summary"]["completed"], 1)
        self.assertEqual(stages["web"]["summary"]["finding_count"], 1)

    def test_recency_half_life(self):
        self.data["behavior"]["events"] = [
            {"id": "e", "candidate_id": "analytics", "kind": "browse", "at": "2026-09-16T10:00:00Z"}]
        self.assertEqual(self.run_data()["behavior"]["items"][0]["score"], 0.5)

    def test_purchase_has_triple_browse_weight(self):
        event = self.data["behavior"]["events"][0]
        self.data["behavior"]["events"] = [event]
        event["at"] = self.data["as_of"]
        self.assertEqual(self.run_data()["behavior"]["items"][0]["score"], 3)
        event["kind"] = "browse"
        self.assertEqual(self.run_data()["behavior"]["items"][0]["score"], 1)

    def test_cold_start_is_stable_id_order(self):
        self.data["behavior"]["events"] = []
        self.data["candidates"].reverse()
        result = self.run_data()["behavior"]
        self.assertTrue(result["summary"]["cold_start"])
        self.assertEqual([r["candidate_id"] for r in result["items"]], ["analytics", "storefront"])

    def test_same_category_transfers_quarter_weight(self):
        self.data["candidates"][1]["category"] = "software"
        event = self.data["behavior"]["events"][0]
        event["at"] = self.data["as_of"]
        self.data["behavior"]["events"] = [event]
        self.assertEqual([r["score"] for r in self.run_data()["behavior"]["items"]], [3, 0.75])

    def test_behavior_changes_review_guided_and_web_priority(self):
        self.data["onboarding"]["steps"][2]["depends_on"] = ["profile_step"]
        baseline = self.run_data()
        self.assertEqual(baseline["web"]["items"][0]["requirement_id"], "privacy")
        self.data["behavior"]["events"] = [
            {"id": "new", "candidate_id": "storefront", "kind": "purchase", "at": self.data["as_of"]}]
        result = self.run_data()
        self.assertEqual(result["behavior"]["summary"]["requirement_order"], ["identity", "payments", "privacy"])
        self.assertEqual(result["review"]["items"][1]["requirement_id"], "payments")
        self.assertEqual(result["guided"]["items"][1]["step_id"], "payments_step")
        self.assertEqual(result["web"]["items"][0]["requirement_id"], "payments")

    def test_review_gap_records_missing_terms_and_document(self):
        item = self.run_data()["review"]["items"][1]
        self.assertEqual(item["gap_id"], "gap:privacy")
        self.assertEqual(item["candidate_ids"], ["analytics"])
        self.assertEqual(item["evidence"], [{"document_id": "policy", "matched_terms": ["retention"],
                                           "missing_terms": ["deletion"]}])

    def test_review_checks_only_explicitly_linked_evidence(self):
        self.data["documents"][0]["text"] += " retention deletion"
        self.assertEqual(self.run_data()["review"]["items"][1]["status"], "gap")

    def test_review_terms_case_insensitive(self):
        self.data["documents"][1]["text"] = "RETENTION and DeLeTiOn"
        result = self.run_data()
        self.assertEqual(result["review"]["items"][1]["status"], "covered")
        self.assertEqual(result["web"]["items"], [])

    def test_review_requires_one_complete_document(self):
        self.data["documents"].append({"id": "second", "requirement_ids": ["privacy"], "text": "deletion"})
        self.assertEqual(self.run_data()["review"]["items"][1]["status"], "gap")

    def test_no_documents_exposes_all_gaps(self):
        self.data["documents"] = []
        self.data["onboarding"]["steps"][0]["completed"] = False
        result = self.run_data()
        self.assertEqual(result["review"]["summary"]["gap_count"], 3)
        self.assertEqual(result["guided"]["summary"]["blocked"], 2)
        self.assertEqual(result["web"]["items"][0]["requirement_id"], "identity")
        self.assertEqual(result["web"]["items"][0]["status"], "unavailable")

    def test_guided_prerequisites_and_progress(self):
        result = self.run_data()["guided"]
        self.assertEqual([r["state"] for r in result["items"]], ["completed", "ready", "blocked"])
        self.assertAlmostEqual(result["summary"]["completion_fraction"], 1 / 3)

    def test_gap_cannot_be_marked_complete(self):
        self.data["onboarding"]["steps"][1]["completed"] = True
        self.reject()

    def test_covered_step_cannot_skip_incomplete_prerequisite(self):
        self.data["documents"][1]["text"] = "retention deletion"
        self.data["onboarding"]["steps"][0]["completed"] = False
        self.data["onboarding"]["steps"][1]["completed"] = True
        self.reject()

    def test_review_resolution_and_completion_unlock_next_research(self):
        self.data["documents"][1]["text"] = "retention deletion"
        self.data["onboarding"]["steps"][1]["completed"] = True
        result = self.run_data()
        self.assertEqual(result["guided"]["summary"]["completed"], 2)
        self.assertEqual(result["web"]["summary"]["eligible_step_ids"], ["payments_step"])
        self.assertEqual(result["web"]["items"][0]["gap_id"], "gap:payments")

    def test_research_preserves_provenance_without_closing_gap(self):
        result = self.run_data()
        finding = result["web"]["items"][0]
        resource = self.data["research"]["resources"][0]
        self.assertEqual(finding["status"], "supported")
        self.assertEqual(finding["provenance"]["sha256"],
                         hashlib.sha256(resource["content"].encode("utf-8")).hexdigest())
        self.assertEqual(finding["provenance"]["retrieved_at"], resource["retrieved_at"])
        self.assertEqual(finding["provenance"]["url"], resource["url"])
        self.assertEqual(result["review"]["items"][1]["status"], "gap")

    def test_insufficient_research_is_not_supported(self):
        self.data["research"]["resources"][0]["content"] = "SYNTHETIC retention only"
        finding = self.run_data()["web"]["items"][0]
        self.assertEqual(finding["status"], "insufficient")
        self.assertEqual(finding["missing_terms"], ["deletion"])

    def test_missing_snapshot_reports_unavailable(self):
        self.data["research"]["resources"] = []
        item = self.run_data()["web"]["items"][0]
        self.assertEqual(item["status"], "unavailable")
        self.assertIsNone(item["provenance"])

    def test_no_source_urls_preserves_gap(self):
        self.data["requirements"][1]["source_urls"] = []
        item = self.run_data()["web"]["items"][0]
        self.assertEqual(item["status"], "no_sources")
        self.assertEqual(item["gap_id"], "gap:privacy")
        self.assertIsNone(item["url"])

    def test_all_completed_no_research_and_full_progress(self):
        self.data["documents"] += [
            {"id": "fullprivacy", "requirement_ids": ["privacy"], "text": "retention deletion"},
            {"id": "fullpayments", "requirement_ids": ["payments"], "text": "refund receipt"}]
        for step in self.data["onboarding"]["steps"]:
            step["completed"] = True
        result = self.run_data()
        self.assertEqual(result["guided"]["summary"]["completion_fraction"], 1)
        self.assertEqual(result["web"]["items"], [])

    def test_no_network_attempts(self):
        with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            self.run_data()

    def test_deterministic_and_does_not_mutate_input(self):
        original = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(first, self.run_data())
        self.assertEqual(self.data, original)

    def test_invalid_references(self):
        self.data["behavior"]["events"][0]["candidate_id"] = "missing"
        self.reject()

    def test_invalid_numeric_values(self):
        for value in (0, -1, True, "7", float("nan"), float("inf"), 10 ** 500):
            with self.subTest(value=str(value)[:30]):
                self.data["behavior"]["half_life_days"] = value
                self.reject()

    def test_invalid_event_time(self):
        for value in ("2027-01-01T00:00:00Z", "2026-01-01", "not-a-time"):
            with self.subTest(value=value):
                self.data["behavior"]["events"][0]["at"] = value
                self.reject()

    def test_timezone_equivalence(self):
        self.data["behavior"]["events"][0]["at"] = "2026-09-23T12:00:00+02:00"
        self.data["behavior"]["events"] = self.data["behavior"]["events"][:1]
        self.assertEqual(self.run_data()["behavior"]["items"][0]["score"], 3)

    def test_cycles_and_unknown_prerequisites(self):
        self.data["onboarding"]["steps"][0]["depends_on"] = ["payments_step"]
        self.reject()
        self.data["onboarding"]["steps"][0]["depends_on"] = ["missing"]
        self.reject()

    def test_duplicate_ids_and_missing_fields(self):
        self.data["candidates"].append(copy.deepcopy(self.data["candidates"][0]))
        self.reject()
        self.data["candidates"].pop()
        del self.data["candidates"][0]["title"]
        self.reject()

    def test_invalid_top_level_and_empty_arrays(self):
        for value in (None, [], {}, False):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(value)
        self.data["candidates"] = []
        self.reject()

    def test_url_allowlist_rejects_bypass_and_unsafe_schemes(self):
        urls = ["http://docs.example.test/privacy", "https://evil.test/privacy",
                "https://docs.example.test.evil.test/privacy",
                "https://docs.example.test@evil.test/privacy",
                "https://user:password@docs.example.test/privacy",
                "https://docs.example.test:444/privacy", "file:///private",
                "https://docs.example.test/privacy#fragment", "https://docs.example.test\\evil",
                "https://docs.example.test/\nprivacy", "https://[bad/privacy"]
        for url in urls:
            with self.subTest(url=url):
                self.data["requirements"][1]["source_urls"] = [url]
                self.reject()

    def test_unused_resources_still_must_be_allowlisted(self):
        self.data["research"]["resources"].append(
            {"url": "https://evil.test/x", "content": "SYNTHETIC", "retrieved_at": self.data["as_of"]})
        self.reject()

    def test_canonical_urls_match_snapshot(self):
        self.data["requirements"][1]["source_urls"] = ["https://DOCS.EXAMPLE.TEST:443/privacy"]
        self.assertEqual(self.run_data()["web"]["items"][0]["status"], "supported")

    def test_duplicate_canonical_resources_rejected(self):
        row = copy.deepcopy(self.data["research"]["resources"][0])
        row["url"] = "https://DOCS.EXAMPLE.TEST:443/privacy"
        self.data["research"]["resources"].append(row)
        self.reject()

    def test_future_snapshot_rejected(self):
        self.data["research"]["resources"][0]["retrieved_at"] = "2027-01-01T00:00:00Z"
        self.reject()

    def test_stage_boundary_rejects_tampered_handoffs(self):
        result = self.run_data()
        result["review"]["items"][1]["gap_id"] = "gap:other"
        with self.assertRaises(app.ValidationError):
            app.Schema.stage("review", result["review"], result["behavior"])
        result = self.run_data()
        result["guided"]["items"][2]["state"] = "ready"
        with self.assertRaises(app.ValidationError):
            app.Schema.stage("guided", result["guided"], result["review"])
        result = self.run_data()
        result["web"]["items"][0]["step_id"] = "payments_step"
        with self.assertRaises(app.ValidationError):
            app.Schema.stage("web", result["web"], result["guided"])

    def test_cli_success(self):
        code, result = self.cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), ["behavior", "review", "guided", "web"])

    def test_cli_missing_file_and_usage(self):
        for args in ((), ("_nonexistent_input.json",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                code, result = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_malformed_duplicate_nonfinite_and_schema_errors(self):
        for text in ('{', '{"x":1,"x":2}', '{"x":NaN}', '[]', '{"schema_version":1}'):
            with self.subTest(text=text):
                code, result = self.cli_text(text)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_nested_wrong_type_reports_json_error(self):
        self.data["behavior"]["events"][0]["candidate_id"] = []
        code, result = self.cli_text(json.dumps(self.data))
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "error")

    def test_cli_invalid_encoding(self):
        path = ROOT / ("_test_bytes_" + uuid.uuid4().hex + ".json")
        try:
            with path.open("xb") as target:
                target.write(b"\xff")
            code, result = self.cli(str(path))
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "error")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
