"""Standard-library tests; all insurance identities and events are synthetic."""

import copy
from datetime import date, timedelta
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        with (HERE / "example_input.json").open(encoding="utf-8") as handle:
            self.data = json.load(handle)

    def run_data(self):
        return app.run_pipeline(self.data)

    def policy(self, index=0):
        return self.data["research"]["documents"][0]["content"]["ACORD"]["Policy"][index]

    def bad(self):
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_full_pipeline_all_three_entities_and_provenance(self):
        result = self.run_data()
        self.assertEqual(result["stage"], "semantic")
        self.assertEqual(result["status"], "ok")
        self.assertEqual({r["entity_type"] for r in result["findings"]},
                         {"policy", "claim", "underwriting_submission"})
        self.assertEqual(len(result["sources"]), 3)
        self.assertTrue(all(len(s["sha256"]) == 64 for s in result["sources"]))
        self.assertTrue(result["guided"]["ready"])
        self.assertEqual(result["behavior"]["rankings"][0]["policy_id"], "POL-HOME")
        self.assertEqual(result["semantic"]["results"][0]["entity_id"], "POL-HOME")

    def test_no_input_mutation_and_deterministic_output(self):
        original = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(original, self.data)
        self.assertEqual(first, self.run_data())

    def test_minimization_removes_personal_data_everywhere(self):
        output = json.dumps(self.run_data())
        for forbidden in ("Invented Mira", "Invented Jules", "mira@example", "Lantern Lane",
                          "Comet Road", "SYNTHVIN"):
            self.assertNotIn(forbidden, output)
        self.assertIn("HOLDER-001", output)

    def test_allowlist_exact_host_and_transport(self):
        for url in ("http://fixtures.insurance.example/policies",
                    "https://fixtures.insurance.example.attacker.invalid/policies",
                    "https://user@fixtures.insurance.example/policies",
                    "https://fixtures.insurance.example:443/policies",
                    "https://fixtures.insurance.example/policies?email=private",
                    "https://fixtures.insurance.example/policies#fragment",
                    "https://fixtures.insurance.example/%2e%2e/private"):
            with self.subTest(url=url):
                self.data["research"]["documents"][0]["url"] = url
                self.bad()

    def test_allowlist_validation_precedes_injected_retrieval(self):
        self.data["research"]["documents"][0]["url"] = "https://blocked.example/x"
        calls = []
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data, retriever=lambda url: calls.append(url))
        self.assertEqual(calls, [])

    def test_injected_offline_retrieval_and_hash_change(self):
        fixtures = {d["url"]: copy.deepcopy(d["content"]) for d in self.data["research"]["documents"]}
        url = self.data["research"]["documents"][0]["url"]
        fixtures[url]["ACORD"]["Policy"][0]["Title"] = "Synthetic revised flood policy"
        calls = []

        def retrieve(source_url):
            calls.append(source_url)
            return fixtures[source_url]

        result = app.run_pipeline(self.data, retriever=retrieve)
        self.assertEqual(calls, [d["url"] for d in self.data["research"]["documents"]])
        self.assertNotEqual(result["sources"][0]["sha256"], self.run_data()["sources"][0]["sha256"])
        self.assertIn("revised", result["findings"][0]["title"])

    def test_injected_retrieval_invalid_and_exception(self):
        for retriever in (lambda url: [], lambda url: 1 / 0):
            with self.subTest(retriever=retriever), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, retriever=retriever)

    def test_json_claim_and_submission_equivalence(self):
        original = self.run_data()
        docs = self.data["research"]["documents"]
        docs[1]["format"] = "acord-json"
        docs[1]["content"] = {"ACORD": {"UnderwritingSubmission": [{
            "Id": "SUB-HOME", "PolicyId": "POL-HOME",
            "Policyholder": {"Ref": "HOLDER-001"},
            "UnderwritingFactors": {"property_type": "synthetic detached residence",
                                    "construction_year": "2007"},
            "DisclosedFactors": ["property_type", "construction_year"]}]}}
        docs[2]["format"] = "acord-json"
        claim = next(r for r in original["findings"] if r["entity_type"] == "claim")
        docs[2]["content"] = {"ACORD": {"Claim": [{
            "Id": claim["id"], "PolicyId": claim["policy_id"], "LossAmount": claim["loss_amount"],
            "LossDate": claim["loss_date"], "Peril": claim["peril"]}]}}
        result = self.run_data()
        for before, after in zip(original["findings"], result["findings"]):
            self.assertEqual({k: v for k, v in before.items() if k != "provenance"},
                             {k: v for k, v in after.items() if k != "provenance"})

    def test_xml_policy_and_claim_numeric_fields(self):
        self.data["research"]["documents"] = [{
            "url": "https://fixtures.insurance.example/xml",
            "format": "acord-xml",
            "content": "<ACORD><Policy><Id>POL-HOME</Id><Title>Synthetic home</Title>"
                       "<Description>Synthetic flood policy</Description>"
                       "<Policyholder><Ref>HOLDER-001</Ref></Policyholder>"
                       "<CoverageLimit>250000</CoverageLimit>"
                       "<CoveredPerils><Peril>flood</Peril></CoveredPerils>"
                       "<EffectiveDate>2026-01-01</EffectiveDate><ExpiryDate>2026-12-31</ExpiryDate>"
                       "</Policy><Claim><Id>CLM-HOME</Id><PolicyId>POL-HOME</PolicyId>"
                       "<LossAmount>152.25</LossAmount><LossDate>2026-03-01</LossDate>"
                       "<Peril>flood</Peril></Claim></ACORD>"}]
        self.data["guided"]["acknowledged_factors"] = []
        self.data["behavior"]["events"] = []
        result = self.run_data()
        claim = next(r for r in result["findings"] if r["entity_type"] == "claim")
        self.assertEqual(claim["loss_amount"], 152.25)

    def test_malformed_and_unsafe_xml(self):
        for xml in ("<ACORD>", '<!DOCTYPE ACORD [<!ENTITY x "a">]><ACORD/>',
                    "<ACORD><Unknown/></ACORD>", '<ACORD secret="yes"/>',
                    "<ACORD><Claim><Id>A</Id><Id>B</Id></Claim></ACORD>"):
            with self.subTest(xml=xml):
                self.data["research"]["documents"][1]["content"] = xml
                self.bad()

    def test_unknown_fields_and_claim_form_duplicates(self):
        self.policy()["HiddenRisk"] = 10
        self.bad()
        del self.policy()["HiddenRisk"]
        self.data["research"]["documents"][2]["content"] += "\nClaim-ID: CLM-OTHER"
        self.bad()

    def test_duplicate_sources_and_entity_identifiers(self):
        self.data["research"]["documents"].append(copy.deepcopy(self.data["research"]["documents"][0]))
        self.bad()
        self.data["research"]["documents"].pop()
        self.policy(1)["Id"] = self.policy()["Id"]
        self.bad()

    def test_unknown_policy_references_rejected(self):
        self.data["research"]["documents"][2]["content"] = (
            self.data["research"]["documents"][2]["content"].replace("POL-HOME", "POL-MISSING"))
        self.bad()

    def test_claim_reasons_are_uniform_and_nonfinal(self):
        result = self.run_data()
        claim = next(r for r in result["findings"] if r["entity_type"] == "claim")
        self.assertEqual(claim["assessment"]["decision"], "eligible_for_review")
        self.assertTrue(claim["assessment"]["human_review_required"])
        self.assertTrue(claim["assessment"]["reasons"])
        self.policy()["CoverageLimit"] = 10
        self.policy()["CoveredPerils"] = ["fire"]
        self.policy()["EffectiveDate"] = "2026-09-01"
        self.data["guided"]["preferred_perils"] = []
        claim = next(r for r in self.run_data()["findings"] if r["entity_type"] == "claim")
        self.assertEqual(claim["assessment"]["decision"], "manual_review")
        self.assertEqual(len(claim["assessment"]["reasons"]), 3)

    def test_claim_decision_independent_of_identity_and_activity(self):
        before = next(r["assessment"] for r in self.run_data()["findings"] if r["entity_type"] == "claim")
        self.policy()["Policyholder"]["Name"] = "Invented Different Person"
        self.data["behavior"]["events"] = []
        self.data["guided"]["preferred_perils"] = ["collision"]
        after = next(r["assessment"] for r in self.run_data()["findings"] if r["entity_type"] == "claim")
        self.assertEqual(before, after)

    def test_hidden_or_sensitive_underwriting_factor_rejected(self):
        original = self.data["research"]["documents"][1]["content"]
        for replacement in (
            original.replace("<Factor>construction_year</Factor>", ""),
            original.replace("construction_year", "ethnicity")):
            self.data["research"]["documents"][1]["content"] = replacement
            self.bad()

    def test_sensitive_holder_field_and_holder_mismatch(self):
        self.policy()["Policyholder"]["HealthDiagnosis"] = "not permitted"
        self.bad()
        del self.policy()["Policyholder"]["HealthDiagnosis"]
        self.data["research"]["documents"][1]["content"] = (
            self.data["research"]["documents"][1]["content"].replace("HOLDER-001", "HOLDER-099"))
        self.bad()

    def test_seeded_random_synthetic_claim_fixtures(self):
        rng = random.Random(119)
        for _ in range(12):
            amount = round(rng.uniform(100, 300000), 2)
            loss_date = date(2026, 1, 1) + timedelta(days=rng.randrange(240))
            self.data["research"]["documents"][2]["content"] = (
                f"Claim-ID: CLM-HOME\nPolicy-ID: POL-HOME\nLoss-Amount: {amount}\n"
                f"Loss-Date: {loss_date.isoformat()}\nPeril: flood")
            result = self.run_data()
            claim = next(r for r in result["findings"] if r["entity_type"] == "claim")
            self.assertEqual(claim["loss_amount"], amount)
            self.assertEqual(claim["loss_date"], loss_date.isoformat())
            self.assertEqual(claim["assessment"]["decision"],
                             "manual_review" if amount > 250000 else "eligible_for_review")

    def test_invalid_amounts_and_dates(self):
        for amount in (0, -1, True, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(amount=str(amount)[:20]):
                self.policy()["CoverageLimit"] = amount
                self.bad()
        self.policy()["CoverageLimit"] = 250000
        self.data["research"]["documents"][2]["content"] = (
            self.data["research"]["documents"][2]["content"].replace("2026-07-19", "2026-02-30"))
        self.bad()

    def test_future_loss_date_rejected(self):
        self.data["research"]["documents"][2]["content"] = (
            self.data["research"]["documents"][2]["content"].replace("2026-07-19", "2026-12-01"))
        self.bad()

    def test_guided_prerequisites(self):
        for steps in (["review_disclosures"], ["review_minimization", "choose_preferences"],
                      ["review_minimization", "review_minimization"]):
            self.data["guided"]["completed_steps"] = steps
            self.bad()

    def test_guided_requires_privacy_and_disclosure_acknowledgment(self):
        self.data["guided"]["acknowledge_minimization"] = False
        self.bad()
        self.data["guided"]["acknowledge_minimization"] = True
        self.data["guided"]["acknowledged_factors"] = ["property_type"]
        self.bad()

    def test_partial_progress_blocks_downstream(self):
        for count in range(3):
            self.data["guided"]["completed_steps"] = list(app.STEPS[:count])
            result = self.run_data()
            self.assertEqual(result["guided"]["progress"], count / 3)
            self.assertEqual(result["guided"]["next_step"], app.STEPS[count])
            self.assertEqual(result["behavior"]["mode"], "blocked")
            self.assertEqual(result["behavior"]["events_used"], 0)
            self.assertEqual(result["semantic"]["results"], [])
            self.assertEqual(result["semantic"]["index"], [])

    def test_unknown_preferences_rejected(self):
        self.data["guided"]["preferred_perils"] = ["unavailable-peril"]
        self.bad()

    def test_cold_start_deterministic_ties(self):
        self.data["behavior"]["events"] = []
        self.data["guided"]["preferred_perils"] = []
        result = self.run_data()
        self.assertEqual(result["behavior"]["mode"], "cold_start")
        self.assertEqual([r["policy_id"] for r in result["behavior"]["rankings"]],
                         ["POL-AUTO", "POL-HOME"])

    def test_recency_half_life_and_purchase_weight(self):
        self.data["guided"]["preferred_perils"] = []
        self.data["behavior"]["events"] = [
            {"entity_id": "POL-AUTO", "action": "purchase", "timestamp": "2026-08-25T12:00:00Z"},
            {"entity_id": "POL-HOME", "action": "browse", "timestamp": "2026-09-24T12:00:00Z"}]
        result = self.run_data()
        scores = {r["policy_id"]: r["score"] for r in result["behavior"]["rankings"]}
        self.assertEqual(scores["POL-AUTO"], 1.5)
        self.assertEqual(scores["POL-HOME"], 1)

    def test_cross_stage_event_linkage_and_search_prior(self):
        self.data["guided"]["preferred_perils"] = []
        self.data["behavior"]["events"] = [
            {"entity_id": "SUB-HOME", "action": "purchase", "timestamp": self.data["as_of"]}]
        result = self.run_data()
        self.assertEqual(result["behavior"]["rankings"][0]["score"], 3)
        for match in result["semantic"]["results"]:
            if match["policy_id"] == "POL-HOME":
                self.assertEqual(match["behavior_score"], 1)
                self.assertAlmostEqual(match["score"], match["relevance"] + 0.1, places=8)
        self.assertEqual(result["guided"]["disclosed_factors"],
                         ["construction_year", "property_type"])

    def test_invalid_events_and_half_life(self):
        event = self.data["behavior"]["events"][0]
        for key, value in (("entity_id", "POL-MISSING"), ("action", "delete"),
                           ("timestamp", "2027-01-01T00:00:00Z"),
                           ("timestamp", "2026-01-01T00:00:00")):
            original = event[key]
            event[key] = value
            self.bad()
            event[key] = original
        self.data["behavior"]["half_life_days"] = 0
        self.bad()

    def test_no_user_profile_fields_in_events(self):
        self.data["behavior"]["events"][0]["policyholder_email"] = "private@example.invalid"
        self.bad()

    def test_semantic_synonym_index_and_no_match(self):
        self.data["semantic"]["query"] = "car"
        result = self.run_data()
        self.assertEqual(result["semantic"]["results"][0]["entity_id"], "POL-AUTO")
        self.assertEqual(len(result["semantic"]["index"]), 4)
        self.data["semantic"]["query"] = "unfindablexyz"
        self.assertEqual(self.run_data()["semantic"]["results"], [])

    def test_search_top_k_and_invalid_query(self):
        self.data["semantic"]["top_k"] = 1
        self.assertEqual(len(self.run_data()["semantic"]["results"]), 1)
        for value in (0, 101, True, 1.5):
            self.data["semantic"]["top_k"] = value
            self.bad()
        self.data["semantic"]["top_k"] = 5
        for value in (" ", "!!!", None, ["flood"]):
            self.data["semantic"]["query"] = value
            self.bad()

    def test_injected_embedding_fixture(self):
        def embed(value):
            words = app.tokens(value)
            return [1.0 if "flood" in words or "home" in words else 0.0,
                    1.0 if "auto" in words or "collision" in words else 0.0, 0.1]
        result = app.run_pipeline(self.data, embedder=embed)
        self.assertEqual(result["semantic"]["mode"], "injected_embedding")
        self.assertEqual(result["semantic"]["results"][0]["entity_id"], "POL-HOME")
        self.assertEqual(result, app.run_pipeline(self.data, embedder=embed))

    def test_invalid_embedding_outputs_and_failure(self):
        for function in (lambda value: [], lambda value: [0, 0],
                         lambda value: [float("nan")], lambda value: [True],
                         lambda value: "bad", lambda value: 1 / 0,
                         lambda value: [1, 2] if value == self.data["semantic"]["query"] else [1]):
            with self.subTest(function=function), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, embedder=function)

    def test_blocked_search_does_not_call_embedding(self):
        self.data["guided"]["completed_steps"] = []
        calls = []
        app.run_pipeline(self.data, embedder=lambda value: calls.append(value))
        self.assertEqual(calls, [])

    def test_handoff_tampering_and_wrong_order_rejected(self):
        research = app.research_stage(self.data)
        with self.assertRaises(app.ValidationError):
            app.behavior_stage(research, self.data["behavior"])
        research["findings"][0]["Name"] = "should not propagate"
        with self.assertRaises(app.ValidationError):
            app.guided_stage(research, self.data["guided"])

    def test_provenance_and_claim_assessment_tampering_rejected(self):
        for field in ("provenance", "assessment"):
            result = app.research_stage(self.data)
            claim = next(r for r in result["findings"] if r["entity_type"] == "claim")
            if field == "provenance":
                claim["provenance"]["sha256"] = "0" * 64
            else:
                claim["assessment"]["decision"] = "denied"
                claim["assessment"]["reasons"] = []
            with self.assertRaises(app.ValidationError):
                app.guided_stage(result, self.data["guided"])

    def test_invalid_shared_schema_empty_documents_and_nonsynthetic(self):
        for key, value in (("schema_version", "9.0"), ("synthetic", False)):
            original = self.data[key]
            self.data[key] = value
            self.bad()
            self.data[key] = original
        self.data["research"]["documents"] = []
        self.bad()

    def test_cli_success_exactly_one_json_object(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                 str(HERE / "example_input.json")],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_missing_file_invalid_json_and_usage(self):
        for arguments in ([], [str(HERE / "does-not-exist.json")],
                          [str(HERE / "implementation.py")]):
            result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *arguments],
                                    cwd=HERE, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stderr, "")
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_validation_and_duplicate_json_without_extra_files(self):
        for payload in ('{"schema_version": "wrong"}', '{"x":1,"x":2}', '{"x": NaN}'):
            output = io.StringIO()
            with patch("builtins.open", return_value=io.StringIO(payload)), redirect_stdout(output):
                code = app.main(["in-memory-fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_excessive_integer_is_json_error(self):
        payload = '{"schema_version": ' + "9" * 5000 + "}"
        output = io.StringIO()
        with patch("builtins.open", return_value=io.StringIO(payload)), redirect_stdout(output):
            code = app.main(["in-memory-fixture.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
