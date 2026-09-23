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

    def run_data(self):
        return app.run_pipeline(self.data)["results"]

    def rankings(self):
        return {r["item_id"]: r for r in self.run_data()["behavior"]["rankings"]}

    def test_full_envelope_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        output = app.run_pipeline(self.data)
        self.assertEqual(output["stage"], "behavior")
        self.assertEqual(list(output["results"]), list(app.STAGES[1:]))
        self.assertEqual(self.data, before)
        self.assertIs(app.validate(output, "behavior"), output)

    def test_triage_priority_and_accountable_owner(self):
        tickets = self.run_data()["triage"]["tickets"]
        self.assertEqual((tickets[0]["category"], tickets[0]["priority"], tickets[0]["owner"]),
                         ("billing", "high", "billing-team"))
        self.assertEqual(tickets[1]["category"], "shipping")

    def test_triage_default(self):
        ticket = self.run_data()["triage"]["tickets"][2]
        self.assertEqual((ticket["category"], ticket["owner"], ticket["priority"]),
                         ("general", "support-team", "low"))

    def test_triage_tie_uses_configuration_order(self):
        self.data["tickets"][0]["text"] = "refund delivery"
        self.assertEqual(self.run_data()["triage"]["tickets"][0]["category"], "billing")

    def test_triage_word_boundaries(self):
        self.data["tickets"][0]["text"] = "A fraudulent refunding"
        ticket = self.run_data()["triage"]["tickets"][0]
        self.assertEqual(ticket["category"], "general")
        self.assertFalse(ticket["urgent"])

    def test_web_provenance_and_handoff(self):
        result = self.run_data()
        finding = result["web"]["findings"][0]
        routed = result["triage"]["tickets"][0]
        for field in ("ticket_id", "category", "owner", "priority"):
            self.assertEqual(finding[field], routed[field])
        snapshot = self.data["web_config"]["snapshots"][0]
        self.assertEqual(finding["excerpt"], snapshot["content"][finding["start"]:finding["end"]])
        self.assertEqual(finding["retrieved_at"], snapshot["retrieved_at"])
        self.assertEqual(finding["url"], snapshot["url"])

    def test_web_allowlist_exact_match(self):
        self.data["tickets"][0]["urls"] = ["https://help.example.test.evil.test/refunds"]
        result = self.run_data()["web"]
        self.assertNotIn("t1", [f["ticket_id"] for f in result["findings"]])
        self.assertEqual(result["unresolved"][0]["reason"], "host_not_allowlisted")

    def test_web_missing_is_explicit(self):
        reasons = {x["reason"] for x in self.run_data()["web"]["unresolved"]}
        self.assertEqual(reasons, {"snapshot_not_found", "host_not_allowlisted"})

    def test_url_canonicalization_deduplicates(self):
        self.data["tickets"][0]["urls"].append("https://HELP.EXAMPLE.TEST:443/refunds")
        self.assertEqual(len(self.run_data()["web"]["findings"]), 2)

    def test_excerpt_limit(self):
        self.data["web_config"]["max_excerpt_chars"] = 12
        finding = self.run_data()["web"]["findings"][0]
        self.assertEqual(finding["end"], 12)
        self.assertEqual(len(finding["excerpt"]), 12)

    def test_feedback_dedup_and_original_excerpts(self):
        feedback = self.run_data()["feedback"]
        self.assertEqual(len(feedback["groups"]), 2)
        group = feedback["groups"][0]
        self.assertEqual(group["record_ids"], ["f1", "f2"])
        self.assertEqual(group["support"][1]["excerpt"], "REFUND updates are unclear!")
        self.assertEqual(feedback["themes"][0]["count"], 1)

    def test_web_context_supplies_feedback_theme(self):
        group = self.run_data()["feedback"]["groups"][1]
        self.assertEqual(group["themes"], ["shipping"])
        self.assertEqual(group["finding_ids"], ["finding-2"])
        self.data["tickets"][1]["urls"] = []
        group = self.run_data()["feedback"]["groups"][1]
        self.assertEqual(group["themes"], ["general"])

    def test_cross_ticket_duplicates_keep_all_provenance(self):
        self.data["feedback"][1]["ticket_id"] = "t2"
        group = self.run_data()["feedback"]["groups"][0]
        self.assertEqual(group["ticket_ids"], ["t1", "t2"])
        self.assertEqual(group["themes"], ["billing", "shipping"])
        self.assertEqual(group["finding_ids"], ["finding-1", "finding-2"])

    def test_behavior_recency_and_purchase_weight(self):
        ranks = self.rankings()
        self.assertAlmostEqual(ranks["billing-guide"]["event_score"], 0.5)
        self.assertAlmostEqual(ranks["delivery-guide"]["event_score"], 3.0)
        self.assertEqual(self.run_data()["behavior"]["rankings"][0]["item_id"], "delivery-guide")

    def test_behavior_ignores_other_users(self):
        self.assertEqual(self.rankings()["welcome-guide"]["event_score"], 0)
        self.data["behavior"]["events"] = [self.data["behavior"]["events"][2]]
        self.assertTrue(self.run_data()["behavior"]["cold_start"])

    def test_full_chain_provenance_and_routing_change(self):
        rank = self.rankings()["delivery-guide"]
        self.assertEqual(rank["feedback_group_ids"], ["f3"])
        self.assertEqual(rank["finding_ids"], ["finding-2"])
        self.data["tickets"][1]["text"] = "invoice problem"
        output = self.run_data()
        self.assertEqual(output["web"]["findings"][1]["owner"], "billing-team")
        self.assertEqual(output["feedback"]["groups"][1]["themes"], ["billing"])
        ranks = {r["item_id"]: r for r in output["behavior"]["rankings"]}
        self.assertEqual(ranks["delivery-guide"]["feedback_score"], 0)
        self.assertEqual(ranks["billing-guide"]["feedback_score"], 2)

    def test_feedback_count_does_not_inflate_for_duplicates(self):
        self.assertEqual(self.rankings()["billing-guide"]["feedback_score"], 1)
        self.data["feedback"].append({"id": "f4", "ticket_id": "t1", "text": "refund UPDATES ARE UNCLEAR"})
        self.assertEqual(self.rankings()["billing-guide"]["feedback_score"], 1)

    def test_cold_start_uses_feedback_and_popularity(self):
        self.data["behavior"]["events"] = []
        result = self.run_data()["behavior"]
        self.assertTrue(result["cold_start"])
        self.assertEqual(result["strategy"], "feedback_and_popularity")
        self.assertEqual(result["rankings"][0]["item_id"], "billing-guide")

    def test_no_feedback_popularity_fallback(self):
        self.data["feedback"] = []
        self.data["behavior"]["events"] = []
        self.assertEqual(self.run_data()["behavior"]["rankings"][0]["item_id"], "welcome-guide")

    def test_empty_collections(self):
        self.data["tickets"] = []
        self.data["feedback"] = []
        self.data["behavior"]["events"] = []
        self.data["behavior"]["catalog"] = []
        result = self.run_data()
        self.assertEqual(result["web"]["findings"], [])
        self.assertEqual(result["behavior"]["rankings"], [])
        self.assertTrue(result["behavior"]["cold_start"])

    def test_tie_break_and_limit(self):
        self.data["feedback"] = []
        self.data["behavior"]["events"] = []
        self.data["behavior"]["limit"] = 1
        for item in self.data["behavior"]["catalog"]:
            item["popularity"] = 0
        self.assertEqual([r["item_id"] for r in self.run_data()["behavior"]["rankings"]],
                         ["billing-guide"])

    def test_handoff_tampering_rejected(self):
        output = app.run_pipeline(self.data)
        for stage, path in [
            ("triage", ("tickets", 0, "owner")),
            ("web", ("findings", 0, "excerpt")),
            ("feedback", ("groups", 0, "finding_ids")),
            ("behavior", ("rankings", 0, "score")),
        ]:
            with self.subTest(stage=stage):
                broken = copy.deepcopy(output)
                a, b, c = path
                broken["results"][stage][a][b][c] = "fabricated"
                with self.assertRaises(app.ValidationError):
                    app.validate(broken, "behavior")

    def test_stage_skipping_rejected(self):
        envelope = {"schema_version": "1.0", "status": "ok", "stage": "input",
                    "data": self.data, "results": {}}
        with self.assertRaises(app.ValidationError):
            app.advance(envelope, "feedback")

    def test_invalid_schemas(self):
        cases = [
            lambda d: d.update(schema_version="2"),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(extra=True),
            lambda d: d["triage_config"].update(default_category="unknown"),
            lambda d: d["triage_config"]["categories"][0].update(owner=""),
            lambda d: d["feedback"][0].update(ticket_id="absent"),
            lambda d: d["behavior"]["events"][0].update(item_id="absent"),
            lambda d: d["behavior"]["events"][0].update(kind="click"),
            lambda d: d["behavior"].update(half_life_days=0),
            lambda d: d["behavior"].update(half_life_days=float("nan")),
            lambda d: d["behavior"].update(limit=True),
            lambda d: d["web_config"].update(max_excerpt_chars=0),
            lambda d: d["behavior"]["catalog"][0].update(popularity=-1),
            lambda d: d["behavior"]["catalog"][0].update(themes=["unknown"]),
            lambda d: d["feedback"][0].update(text="!!!"),
        ]
        for change in cases:
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_identifiers_and_snapshots(self):
        for collection in ("tickets", "feedback"):
            with self.subTest(collection=collection):
                data = copy.deepcopy(self.data)
                data[collection].append(copy.deepcopy(data[collection][0]))
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        self.data["web_config"]["snapshots"].append(
            copy.deepcopy(self.data["web_config"]["snapshots"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_unsafe_urls(self):
        for url in ("http://help.example.test/x", "https://u:p@help.example.test/x",
                    "https://help.example.test:8080/x", "file:///data",
                    "https://help.example.test:bad/x", "https://help.example.test/a b"):
            with self.subTest(url=url):
                self.data["tickets"][0]["urls"] = [url]
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_snapshot_requires_allowlist(self):
        self.data["web_config"]["snapshots"][0]["url"] = "https://evil.example.test/x"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_timezone_and_future_dates(self):
        for date in ("not-a-date", "2026-09-23T10:00:00", "2027-01-01T00:00:00Z"):
            with self.subTest(date=date):
                self.data["behavior"]["events"][0]["at"] = date
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_timezone_equivalence(self):
        self.data["behavior"]["events"][1]["at"] = "2026-09-23T12:00:00+02:00"
        self.assertEqual(self.rankings()["delivery-guide"]["event_score"], 3)

    def test_deterministic(self):
        self.assertEqual(app.run_pipeline(self.data), app.run_pipeline(self.data))

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["does-not-exist.json"], ["a", "b"]):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", "implementation.py"] + args,
                                        cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_malformed_json_and_validation(self):
        for payload in ("{", "[]", '{"x": NaN}', '{"x": 1, "x": 2}',
                        '{"x": Infinity}', '{"x": 1e999}', '{"schema_version": "bad"}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
