"""Synthetic fixtures only; tests never create files or access networks."""

import contextlib
import copy
import datetime
import hashlib
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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def raw(self):
        return self.data["ticket"]["submission"]["ACORD"]["Claim"]

    def use_entity(self, kind, raw, fmt="acord_json"):
        self.data["ticket"].update(entity_type=kind, format=fmt,
                                   submission={"ACORD": {app.KINDS[kind]: raw}})
        self.data["documents"][0]["body"] = kind + "|SYNTHETIC relevant routing guidance."

    def test_integrated_claim_and_provenance(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["triage"]["category"], "claim_review")
        self.assertEqual(result["triage"]["priority"], "high")
        self.assertEqual(len(result["findings"]), 2)
        provenance = result["findings"][0]["provenance"]
        self.assertEqual(provenance["line_number"], 1)
        self.assertEqual(provenance["content_sha256"], hashlib.sha256(
            self.data["documents"][0]["body"].encode()).hexdigest())
        self.assertEqual(provenance["url"], self.data["ticket"]["source_urls"][0])
        app.validate(result, "triage")

    def test_cross_stage_exact_propagation_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        research = app.research(self.data)
        before = copy.deepcopy(research)
        result = app.triage(research, self.data["config"])
        for key in ("ticket_id", "findings", "source_urls", "entity"):
            self.assertEqual(result[key], research[key])
        self.assertEqual(self.data, original)
        self.assertEqual(research, before)
        ids = [x["finding_id"] for x in research["findings"]]
        self.assertEqual(result["triage"]["reasons"][0]["evidence_ids"], ids)

    def test_policy_json(self):
        self.use_entity("policy", {"Reference": "POL-SYN001"})
        result = app.run_pipeline(self.data)
        self.assertEqual(result["triage"]["category"], "policy_service")
        self.assertEqual(result["triage"]["priority"], "normal")

    def test_acord_xml_claim(self):
        self.data["ticket"].update(format="acord_xml", submission=(
            "<ACORD><Claim><Reference>CLM-SYN001</Reference>"
            "<PolicyReference>POL-SYN001</PolicyReference><LossAmount>17.50</LossAmount>"
            "<LossDate>2024-02-29</LossDate></Claim></ACORD>"))
        result = app.run_pipeline(self.data)
        self.assertEqual(result["entity"]["loss_amount"], 17.5)
        self.assertEqual(result["triage"]["priority"], "normal")

    def test_claim_form_text(self):
        self.data["ticket"].update(format="claim_form_text", submission=(
            "Reference: CLM-SYN002\nPolicyReference: POL-SYN002\n"
            "LossAmount: 50000\nLossDate: 2025-06-01\n"
            "PolicyholderName: Invented Rowan Example\nDescription: SYNTHETIC loss: storm"))
        self.assertEqual(app.run_pipeline(self.data)["triage"]["priority"], "high")

    def test_underwriting_disclosure_json(self):
        self.use_entity("underwriting_submission", {
            "Reference": "UW-SYN001", "UnderwritingFactors": {"occupancy": "residential", "building_age": 22},
            "DisclosedFactors": ["occupancy", "building_age"]})
        result = app.run_pipeline(self.data)
        self.assertEqual(result["triage"]["category"], "underwriting_review")
        self.assertEqual(result["entity"]["disclosed_factors"], ["building_age", "occupancy"])
        self.assertIn("occupancy", result["triage"]["reasons"][1]["detail"])

    def test_underwriting_xml(self):
        self.use_entity("underwriting_submission", {})
        self.data["ticket"].update(format="acord_xml", submission=(
            "<ACORD><UnderwritingSubmission><Reference>UW-SYN002</Reference>"
            "<UnderwritingFactors><Factor name=\"building_age\">31</Factor></UnderwritingFactors>"
            "<DisclosedFactors><Factor>building_age</Factor></DisclosedFactors>"
            "</UnderwritingSubmission></ACORD>"))
        result = app.run_pipeline(self.data)
        self.assertEqual(result["entity"]["underwriting_factors"]["building_age"], 31)

    def test_hidden_or_protected_underwriting_factors_rejected(self):
        for values, disclosed in [({"occupancy": "commercial"}, []),
                                  ({"ethnicity": "invented"}, ["ethnicity"]),
                                  ({"building_age": True}, ["building_age"]),
                                  ({"building_age": 5}, ["building_age", "building_age"])]:
            with self.subTest(values=values):
                self.use_entity("underwriting_submission", {
                    "Reference": "UW-SYN003", "UnderwritingFactors": values, "DisclosedFactors": disclosed})
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_minimizes_private_data_in_entity_and_findings(self):
        private = [self.raw()[key] for key in app.PRIVATE_FIELDS]
        self.data["documents"][0]["body"] += "\nclaim|" + "; ".join(private)
        result = app.run_pipeline(self.data)
        serialized = json.dumps(result)
        for value in private:
            self.assertNotIn(value, serialized)
        self.assertNotIn(self.raw()["Description"], serialized)
        self.assertTrue(result["findings"][-1]["provenance"]["redacted"])
        self.assertFalse(result["privacy"]["raw_policyholder_data_retained"])

    def test_fair_claim_routing_invariant(self):
        initial = app.run_pipeline(self.data)
        self.raw()["PolicyholderName"] = "Another Invented Person"
        self.raw()["PropertyAddress"] = "999 Fabricated Plaza, Fictional City"
        changed = app.run_pipeline(self.data)
        self.assertEqual(initial["triage"], changed["triage"])
        self.assertTrue(changed["triage"]["human_review_required"])
        self.assertEqual(changed["triage"]["decision"],
                         "route_only_no_coverage_or_eligibility_decision")
        self.assertTrue(all(r["detail"] for r in changed["triage"]["reasons"]))

    def test_configurable_threshold_and_accountable_route(self):
        self.data["config"]["high_loss_amount"] = 80000
        self.data["config"]["routes"]["claim_review"] = {"team": "special-claims", "owner": "duty-two"}
        result = app.run_pipeline(self.data)
        self.assertEqual(result["triage"]["priority"], "normal")
        self.assertEqual(result["triage"]["route"]["owner"], "duty-two")
        self.data["config"]["routes"]["claim_review"]["owner"] = ""
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_configurable_category_and_missing_category_route(self):
        config = self.data["config"]
        config["categories"]["claim"] = "property_loss_review"
        config["routes"]["property_loss_review"] = config["routes"].pop("claim_review")
        self.assertEqual(app.run_pipeline(self.data)["triage"]["category"], "property_loss_review")
        del config["routes"]["property_loss_review"]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_allowlist_blocks_url_bypasses(self):
        for bad in ["http://insurance.example/claims", "https://insurance.example.evil.test/claims",
                    "https://insurance.example@evil.test/claims", "https://insurance.example:443/claims",
                    "https://insurance.example/claims?policyholder=private",
                    "https://insurance.example/claims#fragment", "https://insurance.example/%2e%2e"]:
            with self.subTest(url=bad):
                self.data["documents"][0]["url"] = bad
                self.data["ticket"]["source_urls"] = [bad]
                with self.assertRaises(app.ValidationError):
                    app.research(self.data)

    def test_retrieval_failure_redirect_and_content_type(self):
        for status in (301, 404, 500):
            with self.subTest(status=status):
                self.data["documents"][0]["status"] = status
                with self.assertRaises(app.ValidationError):
                    app.research(self.data)
        self.data["documents"][0]["status"] = 200
        self.data["documents"][0]["content_type"] = "text/html"
        with self.assertRaises(app.ValidationError):
            app.research(self.data)

    def test_missing_duplicate_or_unrelated_evidence(self):
        for body in ("policy|SYNTHETIC policy only.", "unstructured", "claim|"):
            with self.subTest(body=body):
                self.data["documents"][0]["body"] = body
                with self.assertRaises(app.ValidationError):
                    app.research(self.data)
        self.data["documents"] *= 2
        with self.assertRaises(app.ValidationError):
            app.research(self.data)

    def test_handoff_tampering_rejected(self):
        research = app.research(self.data)
        for mutation in ("hash", "entity", "url", "duplicate", "private"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(research)
                if mutation == "hash":
                    bad["findings"][0]["provenance"]["content_sha256"] = "invalid"
                elif mutation == "entity":
                    bad["findings"][0]["entity_type"] = "policy"
                elif mutation == "url":
                    bad["findings"][0]["provenance"]["url"] = "https://elsewhere.example/"
                elif mutation == "duplicate":
                    bad["findings"].append(bad["findings"][0])
                else:
                    bad["entity"]["PolicyholderName"] = "Invented Name"
                with self.assertRaises(app.ValidationError):
                    app.triage(bad, self.data["config"])

    def test_invalid_amounts_dates_and_unknown_fields(self):
        original = copy.deepcopy(self.raw())
        for key, value in [("LossAmount", -1), ("LossAmount", True), ("LossAmount", "NaN"),
                           ("LossAmount", "Infinity"), ("LossAmount", 0.001),
                           ("LossDate", "2025-02-29"), ("LossDate", "20250719"),
                           ("SecretRiskScore", 90)]:
            with self.subTest(key=key, value=value):
                self.data["ticket"]["submission"]["ACORD"]["Claim"] = copy.deepcopy(original)
                self.raw()[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_xml_entities_duplicates_and_claim_form_duplicates(self):
        self.data["ticket"]["format"] = "acord_xml"
        for payload in ['<!DOCTYPE ACORD [<!ENTITY x "bad">]><ACORD/>',
                        "<ACORD><Claim>", "<ACORD><Claim><Reference>CLM-A</Reference>"
                        "<Reference>CLM-B</Reference></Claim></ACORD>"]:
            self.data["ticket"]["submission"] = payload
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data)
        self.data["ticket"].update(format="claim_form_text",
                                   submission="Reference: CLM-A\nReference: CLM-B")
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_seeded_synthetic_loss_fixtures(self):
        rng = random.Random(2403)
        for _ in range(20):
            loss = rng.randint(1, 9999999) / 100
            day = datetime.date(2024, 1, 1) + datetime.timedelta(days=rng.randrange(600))
            self.raw().update(LossAmount=loss, LossDate=day.isoformat())
            result = app.run_pipeline(self.data)
            self.assertEqual(result["triage"]["priority"], "high" if loss >= 50000 else "normal")

    def test_cli_subprocess_success_single_json(self):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                    str(ROOT / "example_input.json")],
                                   capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")

    def test_cli_missing_file_and_usage(self):
        for arguments in [[], [str(ROOT / "nonexistent.json")]]:
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                                       capture_output=True, text=True, cwd=ROOT, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stderr, "")
            self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_malformed_json_duplicate_keys_and_invalid_input(self):
        for payload in [b"{", b'{"x":1,"x":2}', b'{"x":NaN}', b"[]", b"\xff",
                        json.dumps({**self.data, "synthetic_fixture": False}).encode()]:
            with self.subTest(payload=payload[:40]):
                output = io.StringIO()
                with patch("builtins.open", return_value=io.BytesIO(payload)), contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-in-memory.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_output_validation_rejects_unreasoned_or_final_decisions(self):
        result = app.run_pipeline(self.data)
        result["triage"]["reasons"] = []
        with self.assertRaises(app.ValidationError):
            app.validate(result, "triage")
        result = app.run_pipeline(self.data)
        result["triage"]["decision"] = "deny_claim"
        with self.assertRaises(app.ValidationError):
            app.validate(result, "triage")


if __name__ == "__main__":
    unittest.main()
