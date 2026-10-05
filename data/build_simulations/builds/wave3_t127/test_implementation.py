import contextlib
import copy
import datetime
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def request(entity, content, fmt="json", names=None):
    names = names or list(app.FIELDS[entity])
    return {
        "synthetic": True,
        "schema": [{"entity": entity, "name": name, "required": True} for name in names],
        "documents": [{"id": "synthetic-test", "entity": entity, "format": fmt,
                       "content": json.dumps(content) if fmt == "json" else content}],
    }


class ExtractionTests(unittest.TestCase):
    def test_example_all_entities(self):
        data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        output = app.process(data)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["records"]), 3)
        serialized = json.dumps(output)
        for private in ("Mira", "Imaginary", "SYNTHETICVIN", "raw_document"):
            self.assertNotIn(private, serialized)
        self.assertEqual(output, app.process(copy.deepcopy(data)))

    def test_json_nested_escaped_source_span(self):
        data = request("policy", {"ACORD": {"property_region": 'North "A" & West'}},
                       names=["property_region"])
        field = app.process(data)["records"][0]["fields"]["property_region"]
        span = field["source"]
        raw = data["documents"][0]["content"][span["start"]:span["end"]]
        self.assertEqual(json.loads(raw), field["value"])

    def test_xml_escaped_span(self):
        text = "<ACORD><property_region>North &amp; West</property_region></ACORD>"
        result = app.process(request("policy", text, "xml", ["property_region"]))
        field = result["records"][0]["fields"]["property_region"]
        self.assertEqual(field["value"], "North & West")
        self.assertEqual(text[field["source"]["start"]:field["source"]["end"]], "North &amp; West")

    def test_text_span_unicode_and_crlf(self):
        text = "SYNTHETIC café\r\nclaim_id: SYN-C1\r\nloss_amount: 12.50\r\n"
        result = app.process(request("claim", text, "text", ["claim_id", "loss_amount"]))
        field = result["records"][0]["fields"]["loss_amount"]
        self.assertEqual(text[field["source"]["start"]:field["source"]["end"]], "12.50")
        self.assertEqual(field["value"], 12.5)

    def test_required_optional_and_null_missing(self):
        data = request("policy", {"policy_id": None}, names=["policy_id", "effective_date"])
        data["schema"][1]["required"] = False
        output = app.process(data)
        self.assertEqual(output["status"], "incomplete")
        self.assertEqual(len(output["records"][0]["missing_fields"]), 2)
        data["schema"][0]["required"] = False
        self.assertEqual(app.process(data)["status"], "ok")

    def test_duplicate_fields_all_formats(self):
        cases = [
            ("json", '{"policy_id":"A","policy_id":"B"}'),
            ("json", '{"a":{"policy_id":"A"},"b":{"policy_id":"B"}}'),
            ("xml", "<ACORD><policy_id>A</policy_id><policy_id>B</policy_id></ACORD>"),
            ("text", "policy_id: A\npolicy_id: B"),
        ]
        for fmt, content in cases:
            with self.subTest(fmt=fmt, content=content):
                data = request("policy", {}, names=["policy_id"])
                data["documents"][0].update(format=fmt, content=content)
                with self.assertRaises(app.ValidationError):
                    app.process(data)

    def test_claim_stated_reason_cannot_be_hidden_by_schema(self):
        for decision, reason in [("denied", None), ("approved", "Because of age"),
                                 ("denied", app.REASONS["approved"]), ("unknown", "anything")]:
            with self.subTest(decision=decision, reason=reason):
                with self.assertRaises(app.ValidationError):
                    app.process(request("claim", {"claim_id": "SYN-C1", "decision": decision,
                                                  "decision_reason": reason}, names=["claim_id"]))
        for decision, reason in app.REASONS.items():
            self.assertEqual(app.process(request("claim", {"decision": decision,
                "decision_reason": reason}, names=["decision"]))["status"], "ok")

    def test_underwriting_disclosure_and_protected_factors(self):
        for used, disclosed in [(["roof_age"], []), (["religion"], ["religion"]),
                                (["roof_age"], None), (["roof_age", "roof_age"], ["roof_age"])]:
            with self.subTest(used=used, disclosed=disclosed):
                with self.assertRaises(app.ValidationError):
                    app.process(request("underwriting_submission",
                        {"submission_id": "SYN-U1", "risk_factors": used,
                         "disclosed_factors": disclosed}, names=["submission_id"]))

    def test_minimization_rejects_unapproved_schema(self):
        with self.assertRaises(app.ValidationError):
            app.process(request("policy", {"policyholder_name": "Invented Person"},
                                names=["policyholder_name"]))

    def test_bad_shape(self):
        for data in [None, [], {}, {"synthetic": False, "schema": [], "documents": []}]:
            with self.subTest(data=data):
                with self.assertRaises(app.ValidationError):
                    app.process(data)
        data = request("policy", {"policy_id": "P"}, names=["policy_id"])
        data["schema"][0]["required"] = "yes"
        with self.assertRaises(app.ValidationError):
            app.process(data)

    def test_invalid_amounts_and_dates(self):
        for amount in [True, -1, "NaN", {}, 1e20]:
            with self.subTest(amount=amount):
                with self.assertRaises(app.ValidationError):
                    app.process(request("claim", {"loss_amount": amount}, names=["loss_amount"]))
        for date in ["2026-02-29", "2026-6-1", "yesterday"]:
            with self.subTest(date=date):
                with self.assertRaises(app.ValidationError):
                    app.process(request("claim", {"loss_date": date}, names=["loss_date"]))

    def test_xml_entity_rejected(self):
        content = '<!DOCTYPE x [<!ENTITY p "private">]><policy_id>&p;</policy_id>'
        with self.assertRaises(app.ValidationError):
            app.process(request("policy", content, "xml", ["policy_id"]))

    def test_seeded_randomized_synthetic_losses(self):
        rng = random.Random(127)
        for index in range(25):
            amount = round(rng.uniform(10, 50000), 2)
            date = datetime.date(2025, 1, 1) + datetime.timedelta(days=rng.randrange(600))
            data = request("claim", {"claim_id": f"SYN-C{index}", "loss_amount": amount,
                "loss_date": date.isoformat()}, names=["claim_id", "loss_amount", "loss_date"])
            output = app.process(data)
            self.assertEqual(output["status"], "ok")
            self.assertEqual(output["records"][0]["fields"]["loss_amount"]["value"], amount)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "nonexistent.json")]]:
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_malformed_input_no_echo(self):
        for content in ['{"secret":"DO-NOT-ECHO",', '{"synthetic":false}', '[]']:
            output = io.StringIO()
            with patch("builtins.open", return_value=io.StringIO(content)):
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
            self.assertNotIn("DO-NOT-ECHO", output.getvalue())

    def test_blank_xml_missing(self):
        result = app.process(request("policy", "<ACORD><policy_id/></ACORD>", "xml", ["policy_id"]))
        self.assertEqual(result["status"], "incomplete")

    def test_nested_xml_cannot_hide_decision(self):
        text = "<ACORD><claim_id>SYN-C1</claim_id><decision><value>denied</value></decision></ACORD>"
        with self.assertRaises(app.ValidationError):
            app.process(request("claim", text, "xml", ["claim_id"]))


if __name__ == "__main__":
    unittest.main()
