import contextlib
import copy
import io
import json
import pathlib
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


HERE = pathlib.Path(__file__).resolve().parent


def document(entity="claim"):
    data = {
        "entity": entity, "reference": "SYN-TEST-001",
        "summary": "SYNTHETIC: invented storm damage",
        "policyholder": "Invented Mira Example",
        "vin": "SYNTHETICVIN000001",
        "property_address": "999 Fiction Lane, Imaginary City",
    }
    if entity == "claim":
        data.update(loss_amount=1000, loss_date="2026-04-03")
    if entity == "underwriting_submission":
        data["underwriting_factors"] = [
            {"name": "roof_age_years", "value": 10, "disclosed": True}
        ]
    return {"schema_version": 1, "synthetic": True, "format": "acord_json",
            "payload": {"ACORD": {"Submission": data}}}


def submission(doc):
    return doc["payload"]["ACORD"]["Submission"]


class TriageTests(unittest.TestCase):
    def test_each_entity_routes_to_accountable_owner(self):
        for entity in app.ENTITIES:
            with self.subTest(entity=entity):
                output = app.process(document(entity))
                self.assertEqual(output["ticket"]["entity"], entity)
                self.assertTrue(output["triage"]["route"]["owner"])
                self.assertTrue(output["triage"]["reasons"])
                self.assertEqual(output["triage"]["coverage_decision"], "not_made")

    def test_configuration(self):
        doc = document()
        doc["config"] = {
            "high_loss_amount": 500,
            "categories": {"claim": "storm_claim"},
            "routes": {"claim": {"team": "storm_team", "owner": "storm_duty_adjuster"}},
        }
        triage = app.process(doc)["triage"]
        self.assertEqual(triage["category"], "storm_claim")
        self.assertEqual(triage["priority"], "high")
        self.assertEqual(triage["route"]["owner"], "storm_duty_adjuster")

    def test_threshold_boundary_and_zero(self):
        for amount, priority in ((0, "normal"), (24999.99, "normal"), (25000, "high")):
            doc = document()
            submission(doc)["loss_amount"] = amount
            self.assertEqual(app.process(doc)["triage"]["priority"], priority)

    def test_urgent_policy(self):
        doc = document("policy")
        submission(doc)["urgency"] = "urgent"
        self.assertEqual(app.process(doc)["triage"]["priority"], "high")

    def test_personal_data_does_not_affect_decisions_or_escape(self):
        one = document()
        two = copy.deepcopy(one)
        submission(two).update(policyholder="Different invented person",
                               property_address="Other invented address", vin="OTHERFAKEVIN",
                               summary="Sensitive invented content")
        self.assertEqual(app.process(one), app.process(two))
        serialized = json.dumps(app.process(one))
        for value in ("Invented Mira", "999 Fiction", "SYNTHETICVIN", "invented storm"):
            self.assertNotIn(value, serialized)

    def test_claim_form_equivalence(self):
        doc = document()
        expected = app.process(doc)
        doc["payload"] = "\n".join(f"{key}: {value}" for key, value in submission(doc).items())
        doc["format"] = "claim_form"
        self.assertEqual(app.process(doc), expected)

    def test_xml_equivalence(self):
        doc = document()
        expected = app.process(doc)
        doc["payload"] = "<ACORD><Submission>" + "".join(
            f"<{key}>{value}</{key}>" for key, value in submission(doc).items()
        ) + "</Submission></ACORD>"
        doc["format"] = "acord_xml"
        self.assertEqual(app.process(doc), expected)

    def test_xml_underwriting_factors(self):
        doc = document("underwriting_submission")
        doc["format"] = "acord_xml"
        doc["payload"] = (
            "<ACORD><Submission><entity>underwriting_submission</entity>"
            "<reference>SYN-TEST-001</reference><summary>SYNTHETIC</summary>"
            "<underwriting_factors><Factor><name>roof_age_years</name>"
            "<value>10</value><disclosed>true</disclosed></Factor>"
            "</underwriting_factors></Submission></ACORD>"
        )
        self.assertEqual(app.process(doc)["ticket"]["underwriting_factors"][0]["value"], 10)

    def test_hidden_sensitive_duplicate_factors_rejected(self):
        for factors in (
            [], [{"name": "roof_age_years", "value": 2, "disclosed": False}],
            [{"name": "ethnicity", "value": 1, "disclosed": True}],
            [{"name": "roof_age_years", "value": 2, "disclosed": True}] * 2,
        ):
            doc = document("underwriting_submission")
            submission(doc)["underwriting_factors"] = factors
            with self.subTest(factors=factors), self.assertRaises(app.ValidationError):
                app.process(doc)

    def test_invalid_loss_amounts(self):
        for amount in (-1, True, "100", None, float("nan"), float("inf"), 1e100, 10**400):
            doc = document()
            submission(doc)["loss_amount"] = amount
            with self.subTest(amount=amount), self.assertRaises(app.ValidationError):
                app.process(doc)

    def test_invalid_dates(self):
        for date in ("2026-02-30", "2026-2-01", "", 123):
            doc = document()
            submission(doc)["loss_date"] = date
            with self.subTest(date=date), self.assertRaises(app.ValidationError):
                app.process(doc)

    def test_invalid_shapes(self):
        samples = [None, [], {}, dict(document(), synthetic=False),
                   dict(document(), format=[]), dict(document(), config={"unknown": 1})]
        for sample in samples:
            with self.subTest(sample=sample), self.assertRaises(app.ValidationError):
                app.process(sample)

    def test_missing_owner_and_unknown_fields(self):
        doc = document()
        doc["config"] = {"routes": {"claim": {"team": "claims"}}}
        with self.assertRaises(app.ValidationError):
            app.process(doc)
        doc = document()
        submission(doc)["hidden_factor"] = "disallowed"
        with self.assertRaises(app.ValidationError):
            app.process(doc)

    def test_malformed_adapters(self):
        for fmt, payload in (
            ("acord_xml", "<ACORD>"),
            ("acord_xml", '<!DOCTYPE ACORD [<!ENTITY e "x">]><ACORD/>'),
            ("claim_form", "entity: claim\nentity: claim"),
            ("claim_form", "missing delimiter"),
        ):
            doc = document()
            doc.update(format=fmt, payload=payload)
            with self.subTest(fmt=fmt, payload=payload), self.assertRaises(app.ValidationError):
                app.process(doc)

    def test_cli_example(self):
        result = subprocess.run(
            [sys.executable, "-B", str(HERE / "implementation.py"), str(HERE / "example_input.json")],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_argument(self):
        for args in ([], [str(HERE / "nonexistent.json")]):
            result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                                    text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for content in ("{", '{"a": 1, "a": 2}', '{"value": NaN}', "[]",
                        '{"value": ' + "9" * 5000 + "}",
                        json.dumps(dict(document(), synthetic=False))):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                code = app.main(["in_memory_fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_determinism_and_no_mutation(self):
        doc = document("underwriting_submission")
        before = copy.deepcopy(doc)
        self.assertEqual(app.process(doc), app.process(doc))
        self.assertEqual(doc, before)


if __name__ == "__main__":
    unittest.main()
