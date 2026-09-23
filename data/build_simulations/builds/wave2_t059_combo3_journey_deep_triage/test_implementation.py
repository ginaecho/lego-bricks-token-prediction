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

    def test_full_pipeline(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["triage"]["ticket_count"], 2)

    def test_prerequisite_two_step(self):
        journey = app.recommend(self.data)
        self.assertEqual([s["action_id"] for s in journey["steps"]],
                         ["shipping-setup", "launch-review"])
        self.assertEqual(journey["steps"][1]["satisfied_by"], ["shipping-setup"])
        self.assertNotIn("launch-review", [a["action_id"] for a in journey["next_actions"]])

    def test_personalization(self):
        self.data["profile"]["interests"] = ["design"]
        self.assertEqual(app.recommend(self.data)["steps"][0]["action_id"], "theme")

    def test_consensus_disagreement_and_gap(self):
        findings = {f["topic"]: f for f in app.run(self.data)["research"]["findings"]}
        self.assertEqual(findings["refunds"]["status"], "consensus")
        self.assertEqual(findings["shipping"]["positions"], ["five days", "two days"])
        self.assertEqual(findings["shipping"]["status"], "disputed")
        self.assertEqual(findings["tax"]["status"], "unanswered")
        self.assertEqual(findings["tax"]["evidence"], [])
        self.assertTrue(findings["tax"]["unresolved_questions"])

    def test_single_source_needs_corroboration(self):
        self.data["documents"].pop()
        result = app.run(self.data)
        ticket = next(t for t in result["triage"]["tickets"] if t["topic"] == "shipping")
        self.assertEqual(ticket["kind"], "corroboration_needed")
        self.assertEqual(ticket["owner"], "synthetic-support-team")
        self.assertEqual(ticket["matched_rule"], "default")

    def test_accountable_configurable_routing(self):
        result = app.run(self.data)
        tickets = {t["topic"]: t for t in result["triage"]["tickets"]}
        self.assertEqual(tickets["shipping"]["priority"], "high")
        self.assertEqual(tickets["tax"]["priority"], "urgent")
        self.assertEqual(tickets["tax"]["owner"], "synthetic-tax-team")

    def test_first_matching_rule_wins(self):
        self.data["routing"]["rules"].insert(0, {
            "kind": "conflict", "topics": [],
            "route": {"category": "override", "priority": "low", "owner": "synthetic-owner"}})
        ticket = app.run(self.data)["triage"]["tickets"][0]
        self.assertEqual(ticket["category"], "override")
        self.assertEqual(ticket["matched_rule"], 0)

    def test_evidence_and_action_propagation(self):
        result = app.run(self.data)
        for ticket in result["triage"]["tickets"]:
            issue = next(i for i in result["research"]["issues"] if i["id"] == ticket["issue_id"])
            self.assertEqual(ticket["evidence"], issue["evidence"])
            self.assertEqual(ticket["journey_action_ids"], issue["journey_action_ids"])
            self.assertEqual(ticket["subject"], issue["question"])
        self.assertEqual(result["research"]["topics"], result["journey"]["research_topics"])

    def test_tampered_journey_rejected(self):
        journey = app.recommend(self.data)
        journey["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.research(self.data, journey)

    def test_tampered_research_rejected(self):
        journey = app.recommend(self.data)
        findings = app.research(self.data, journey)
        findings["issues"][0]["evidence"][0]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.triage(self.data, journey, findings)

    def test_tampered_routing_output_rejected(self):
        result = app.run(self.data)
        result["triage"]["tickets"][0]["owner"] = "unaccountable"
        with self.assertRaises(app.ValidationError):
            app.validate("triage", result["triage"], self.data,
                         result["journey"], result["research"])

    def test_empty_documents_creates_gaps(self):
        self.data["documents"] = []
        result = app.run(self.data)
        self.assertEqual(result["triage"]["ticket_count"], 3)
        self.assertTrue(all(t["kind"] == "evidence_gap" for t in result["triage"]["tickets"]))

    def test_all_consensus_no_tickets(self):
        for doc in self.data["documents"]:
            doc["claims"][0]["position"] = "two days"
            doc["text"] += " Synthetic tax guidance."
            doc["claims"].append({"topic": "tax", "position": "guidance",
                                  "quote": "Synthetic tax guidance."})
        self.assertEqual(app.run(self.data)["triage"], {"tickets": [], "ticket_count": 0})

    def test_impossible_journey(self):
        self.data["actions"] = [self.data["actions"][0]]
        with self.assertRaisesRegex(app.ValidationError, "two-step"):
            app.run(self.data)

    def test_completed_actions_not_recommended(self):
        self.data["profile"]["completed"].append("shipping-setup")
        journey = app.recommend(self.data)
        self.assertNotIn("shipping-setup", [s["action_id"] for s in journey["steps"]])

    def test_invalid_inputs(self):
        variants = []
        for key, value in (("synthetic", False), ("schema_version", True),
                           ("documents", None), ("actions", "invalid")):
            data = copy.deepcopy(self.data)
            data[key] = value
            variants.append(data)
        data = copy.deepcopy(self.data)
        data["routing"]["default"]["owner"] = " "
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["routing"]["default"]["priority"] = "critical"
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["documents"][0]["claims"][0]["quote"] = "not in source"
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["actions"][0]["requires"] = ["unknown"]
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["documents"].append(copy.deepcopy(data["documents"][0]))
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["extra"] = 1
        variants.append(data)
        for invalid in variants:
            with self.subTest(invalid=invalid), self.assertRaises(app.ValidationError):
                app.run(invalid)

    def test_cycle_rejected(self):
        self.data["actions"][0]["requires"] = ["launch-review"]
        with self.assertRaisesRegex(app.ValidationError, "cyclic"):
            app.run(self.data)

    def test_duplicate_action_rejected(self):
        self.data["actions"].append(copy.deepcopy(self.data["actions"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate"):
            app.run(self.data)

    def test_determinism_and_input_unchanged(self):
        snapshot = copy.deepcopy(self.data)
        first = app.run(self.data)
        self.assertEqual(self.data, snapshot)
        self.data["actions"].reverse()
        self.data["documents"].reverse()
        self.assertEqual(first, app.run(self.data))

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["a", "b"]):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                        capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_json_and_validation_errors(self):
        for source in ('{bad', '{"x": 1, "x": 2}', '{"x": NaN}', 'null',
                       '{"schema_version": true}', '[]'):
            with self.subTest(source=source), patch("builtins.open", mock_open(read_data=source)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
