"""All fixtures and identities in these tests are synthetic."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch
import contextlib
import io

import implementation as app


ROOT = Path(__file__).resolve().parent


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_routing_and_accountability(self):
        output = app.triage(self.payload)
        self.assertEqual(output["status"], "ok")
        self.assertEqual([a["team"] for a in output["assignments"]],
                         ["technical", "finance", "support"])
        self.assertEqual(output["assignments"][0]["priority"], "urgent")
        self.assertEqual(output["assignments"][0]["accountable_owner"], "synthetic-tech-oncall")
        self.assertEqual(output["summary"]["total"], 3)
        self.assertEqual(sum(output["summary"]["by_priority"].values()), 3)

    def test_first_match_wins(self):
        self.payload["tickets"][0]["body"] = "refund and outage"
        self.assertEqual(app.triage(self.payload)["assignments"][0]["rule_id"], "service-outage")

    def test_channel_filter_and_empty_body(self):
        self.payload["tickets"][1].update(channel="web", body="")
        output = app.triage(self.payload)["assignments"][1]
        self.assertEqual(output["source"], "default")
        self.assertIsNone(output["rule_id"])

    def test_empty_tickets(self):
        self.payload["tickets"] = []
        output = app.triage(self.payload)
        self.assertEqual(output["assignments"], [])
        self.assertEqual(output["summary"]["total"], 0)

    def test_configurable_decision(self):
        self.payload["config"]["rules"][0]["decision"]["priority"] = "high"
        self.assertEqual(app.triage(self.payload)["assignments"][0]["priority"], "high")

    def test_deterministic_no_mutation(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(app.triage(self.payload), app.triage(self.payload))
        self.assertEqual(original, self.payload)

    def test_injected_callable_fixture(self):
        calls = []
        def fixture(ticket):
            calls.append(ticket["id"])
            ticket["subject"] = "mutation is isolated"
            return {"category": "general", "priority": "high", "team": "support"}
        output = app.triage(self.payload, fixture)
        self.assertEqual(calls, ["synthetic-003"])
        self.assertEqual(output["assignments"][2]["source"], "injected_callable")
        self.assertEqual(output["assignments"][2]["priority"], "high")
        self.assertNotEqual(self.payload["tickets"][2]["subject"], "mutation is isolated")

    def test_injected_callable_invalid_or_failed(self):
        for result in (None, [], {"category": "general", "priority": "critical", "team": "support"},
                       {"category": "general", "priority": "low", "team": "missing"}):
            with self.subTest(result=result), self.assertRaises(app.ValidationError):
                app.triage(self.payload, lambda ticket: result)
        def broken(ticket):
            raise RuntimeError("synthetic failure")
        with self.assertRaisesRegex(app.ValidationError, "classifier failed"):
            app.triage(self.payload, broken)

    def test_invalid_schema_cases(self):
        mutations = [
            lambda p: p.update(schema_version=True),
            lambda p: p.update(extra="unexpected"),
            lambda p: p.update(tickets={}),
            lambda p: p["tickets"].append(copy.deepcopy(p["tickets"][0])),
            lambda p: p["tickets"][0].update(subject=" "),
            lambda p: p["tickets"][0].update(body=None),
            lambda p: p["config"]["default"].update(team="unowned"),
            lambda p: p["config"]["default"].update(priority=[]),
            lambda p: p["config"]["teams"].update(support=" "),
            lambda p: p["config"]["rules"][0].update(keywords=[]),
            lambda p: p["config"]["rules"].append(copy.deepcopy(p["config"]["rules"][0])),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                payload = copy.deepcopy(self.payload)
                mutate(payload)
                with self.assertRaises(app.ValidationError):
                    app.triage(payload)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), app.triage(self.payload))
        self.assertEqual(proc.stderr, "")

    def test_cli_argument_and_file_errors(self):
        for args in ([], ["nonexistent-synthetic-input.json"], ["one", "two"]):
            with self.subTest(args=args):
                proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_cli_bad_json_and_validation_errors(self):
        for raw in ("{", "[]", '{"schema_version":1,"schema_version":1}', '{"x":NaN}'):
            with self.subTest(raw=raw):
                stdout = io.StringIO()
                with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(stdout):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_file_encoding_error(self):
        stdout = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("synthetic invalid encoding")):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(app.main(["synthetic-invalid.json"]), 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
