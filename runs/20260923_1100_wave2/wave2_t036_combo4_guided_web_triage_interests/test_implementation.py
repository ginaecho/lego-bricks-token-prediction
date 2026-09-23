"""All fixtures are synthetic. Tests never use networks or scratch files."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def reject(self, pattern):
        with self.assertRaisesRegex(app.ValidationError, pattern):
            self.run_data()

    def test_complete_pipeline(self):
        output = self.run_data()
        self.assertEqual(output["status"], "ok")
        self.assertEqual(list(output["stages"]), ["guided", "web", "triage", "interests"])
        self.assertEqual(output["stages"]["guided"]["percent"], 100)
        recommendations = output["stages"]["interests"]["recommendations"]
        self.assertEqual([item["id"] for item in recommendations],
                         ["garden_help", "garden_tracking"])
        self.assertEqual(recommendations[0]["score"], 84)

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(self.data, original)

    def test_prerequisites_topologically_ordered(self):
        self.data["guided"]["steps"].reverse()
        steps = self.run_data()["stages"]["guided"]["steps"]
        self.assertEqual([step["id"] for step in steps],
                         ["research_consent", "support_consent", "personalization_consent"])

    def test_incomplete_onboarding_tracks_blocked_steps(self):
        self.data["guided"]["responses"]["research_consent"] = False
        with self.assertRaises(app.ValidationError) as caught:
            self.run_data()
        progress = caught.exception.details
        self.assertEqual(progress["completed"], 0)
        self.assertEqual([s["status"] for s in progress["steps"]],
                         ["pending", "blocked", "blocked"])

    def test_incomplete_onboarding_tracks_partial_progress(self):
        del self.data["guided"]["responses"]["personalization_consent"]
        with self.assertRaises(app.ValidationError) as caught:
            self.run_data()
        self.assertEqual(caught.exception.details["percent"], 66.67)

    def test_cycle_rejected(self):
        self.data["guided"]["steps"][0]["requires"] = ["personalization_consent"]
        self.reject("cyclic")

    def test_unknown_prerequisite_rejected(self):
        self.data["guided"]["steps"][0]["requires"] = ["unknown"]
        self.reject("unknown or self")

    def test_boolean_responses_strict(self):
        self.data["guided"]["responses"]["research_consent"] = 1
        self.reject("booleans")

    def test_required_consent_steps(self):
        self.data["guided"]["steps"].pop()
        self.reject("consent")

    def test_url_security_boundaries(self):
        for url in [
            "http://help.example.test/a",
            "https://help.example.test.evil.test/a",
            "https://evil.test/a",
            "https://user:password@help.example.test/a",
            "https://help.example.test:80/a",
            "https://help.example.test/a#fragment",
            "https://help.example.test\\@evil.test/a",
            "https://help.example.test/\npath",
            "https://[broken/a",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(app.ValidationError):
                    app.allowed_url(url, ["help.example.test"])

    def test_https_default_port_allowed(self):
        self.assertEqual(app.allowed_url("https://help.example.test:443/a",
                                       ["help.example.test"]),
                         "https://help.example.test:443/a")

    def test_missing_fixture_retrieval_fails_closed(self):
        del self.data["web"]["documents"][self.data["web"]["urls"][0]]
        self.reject("exactly match")

    def test_provenance_digest_and_quotes(self):
        findings = self.run_data()["stages"]["web"]["findings"]
        for finding in findings:
            original = self.data["web"]["documents"][finding["url"]]["text"]
            self.assertEqual(finding["evidence"], original)
            self.assertEqual(finding["sha256"],
                             hashlib.sha256(original.encode("utf-8")).hexdigest())
            self.assertEqual(finding["retrieval"], "synthetic_fixture")

    def test_triage_routing_priority(self):
        tickets = self.run_data()["stages"]["triage"]["tickets"]
        self.assertEqual(tickets[0]["category"], "shipping")
        self.assertEqual(tickets[0]["priority"], "urgent")
        self.assertEqual(tickets[0]["route"]["owner"], "Synthetic Logistics Lead")
        self.assertEqual(tickets[1]["priority"], "normal")

    def test_research_content_drives_triage(self):
        ticket = self.data["triage"]["tickets"][0]
        ticket.update(subject="Question", body="Please help")
        routed = self.run_data()["stages"]["triage"]["tickets"][0]
        self.assertEqual(routed["category"], "shipping")
        self.assertEqual(routed["matched_keywords"], ["delivery"])
        self.assertEqual(routed["priority"], "high")

    def test_fallback_route(self):
        self.data["triage"]["rules"] = []
        self.data["triage"]["routes"] = {
            "general": self.data["triage"]["routes"]["general"]}
        self.data["triage"]["urgent_keywords"] = []
        ticket = self.run_data()["stages"]["triage"]["tickets"][0]
        self.assertEqual(ticket["category"], "general")
        self.assertEqual(ticket["priority"], "low")
        self.assertIsNone(ticket["rule_id"])

    def test_rule_tie_uses_configuration_order(self):
        first = self.data["triage"]["rules"][0]
        second = self.data["triage"]["rules"][1]
        second["keywords"] = first["keywords"][:]
        self.assertEqual(self.run_data()["stages"]["triage"]["tickets"][0]["rule_id"],
                         first["id"])

    def test_accountable_owner_required(self):
        self.data["triage"]["routes"]["shipping"]["owner"] = " "
        self.reject("nonempty")

    def test_missing_route_rejected(self):
        del self.data["triage"]["routes"]["shipping"]
        self.reject("accountable route")

    def test_unvalidated_ticket_source_rejected(self):
        self.data["triage"]["tickets"][0]["source_urls"] = ["https://help.example.test/new"]
        self.reject("unvalidated research")

    def test_invalid_priority_rejected(self):
        self.data["triage"]["rules"][0]["priority"] = []
        self.reject("invalid priority")

    def test_topic_and_item_exclusions_override_preferences(self):
        self.data["interests"]["excluded_item_ids"] = ["garden_help"]
        items = self.run_data()["stages"]["interests"]["recommendations"]
        self.assertEqual([i["id"] for i in items], ["garden_tracking"])

    def test_empty_preferences_yields_no_recommendations(self):
        self.data["interests"]["preferences"] = {}
        self.assertEqual(self.run_data()["stages"]["interests"]["recommendations"], [])

    def test_zero_limit(self):
        self.data["interests"]["limit"] = 0
        result = self.run_data()["stages"]["interests"]
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["eligible_count"], 2)

    def test_ungrounded_item_rejected(self):
        self.data["interests"]["items"][0]["topics"] = ["spaceflight"]
        self.reject("ungrounded")

    def test_recommendation_tie_breaks_by_id(self):
        self.data["interests"]["items"][0]["topics"] = ["gardening"]
        self.data["interests"]["items"].reverse()
        recommendations = self.run_data()["stages"]["interests"]["recommendations"]
        self.assertEqual([r["id"] for r in recommendations],
                         ["garden_help", "garden_tracking"])

    def test_explanations_reference_research_and_routing(self):
        stages = self.run_data()["stages"]
        explanation = stages["interests"]["recommendations"][0]["explanation"]
        self.assertEqual(explanation["ticket_ids"], ["T1"])
        self.assertEqual(explanation["finding_ids"], ["F1"])
        self.assertEqual(explanation["source_urls"], stages["triage"]["tickets"][0]["source_urls"])
        self.assertEqual(explanation["accountable_owners"], ["Synthetic Logistics Lead"])
        self.assertEqual(explanation["matched_preferences"],
                         {"gardening": 5, "sustainability": 3})

    def test_routing_change_propagates_to_recommendations(self):
        self.data["triage"]["rules"][0]["category"] = "billing"
        del self.data["triage"]["routes"]["shipping"]
        result = self.run_data()["stages"]["interests"]["recommendations"]
        self.assertEqual(result, [])

    def test_priority_change_propagates_to_ranking(self):
        self.data["triage"]["urgent_keywords"] = []
        self.data["triage"]["rules"][0]["priority"] = "low"
        self.assertEqual(self.run_data()["stages"]["interests"]["recommendations"][0]["score"], 81)

    def test_shared_schema_validation(self):
        for field, value in [("schema_version", "2.0"), ("fixture_label", "Real data"),
                             ("guided", []), ("web", None), ("triage", 4),
                             ("interests", "not an object")]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data[field] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_records_rejected(self):
        self.data["triage"]["tickets"].append(copy.deepcopy(self.data["triage"]["tickets"][0]))
        self.reject("duplicate id")

    def test_unknown_fields_rejected(self):
        self.data["web"]["live_provider"] = "forbidden"
        self.reject("unknown fields")

    def test_numeric_validation_rejects_boolean(self):
        self.data["interests"]["preferences"]["gardening"] = True
        self.reject("expected integer")

    def test_stages_require_predecessor(self):
        envelope = {"schema_version": "1.0", "fixture_label": "SYNTHETIC",
                    "status": "ok", "stages": {}}
        with self.assertRaisesRegex(app.ValidationError, "predecessor"):
            app.research(self.data["web"], envelope)

    def test_word_boundary_matching(self):
        self.assertEqual(app.hits("redelivery not deliveryman", ["delivery"]), [])
        self.assertEqual(app.hits("DELIVERY!", ["delivery"]), ["delivery"])

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"),
             str(HERE / "example_input.json")],
            cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        output = json.loads(completed.stdout)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(completed.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in [[], ["missing_synthetic_input.json"], ["implementation.py"],
                     ["example_input.json", "extra"]]:
            with self.subTest(args=args):
                completed = subprocess.run(
                    [sys.executable, "-B", str(HERE / "implementation.py"), *args],
                    cwd=HERE, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(json.loads(completed.stdout)["status"], "error")
                self.assertEqual(completed.stderr, "")

    def test_cli_malformed_json_and_validation_errors(self):
        invalid_values = [
            '{"schema_version":"1.0","schema_version":"1.0"}',
            '{"bad":NaN}', '{"bad":Infinity}', '{', '[]',
            json.dumps(dict(self.data, schema_version="9")),
        ]
        for value in invalid_values:
            with self.subTest(value=value[:60]):
                output = io.StringIO()
                with patch.object(Path, "read_text", return_value=value), redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_and_invalid_path_errors(self):
        for error in [UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad input"),
                      ValueError("embedded null byte")]:
            output = io.StringIO()
            with patch.object(Path, "read_text", side_effect=error), redirect_stdout(output):
                code = app.main(["fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
