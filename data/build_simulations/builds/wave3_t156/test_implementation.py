import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone

from implementation import ValidationError, run

ROOT = Path(__file__).resolve().parent


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def invalid(self, data=None):
        with self.assertRaises(ValidationError):
            run(self.data if data is None else data)

    def test_normal_and_deterministic(self):
        result = run(self.data)
        self.assertEqual(result, run(copy.deepcopy(self.data)))
        self.assertEqual(result["recommendations"][0]["id"], "SYN-POLICY-PROPERTY")
        self.assertEqual(len(result["recommendations"]), 4)

    def test_half_life(self):
        self.data["events"] = [
            {"id": "SYN-E1", "item_id": "SYN-POLICY-AUTO", "action": "browse", "at": "2026-08-25T12:00:00Z"}
        ]
        result = run(self.data)["recommendations"]
        self.assertAlmostEqual(result[0]["score"], 0.6)
        self.assertAlmostEqual(result[1]["score"], 0.1)

    def test_purchase_weight(self):
        event = self.data["events"][0]
        self.data["events"] = [event]
        browse = run(self.data)["recommendations"][0]["score"]
        event["action"] = "purchase"
        self.assertAlmostEqual(run(self.data)["recommendations"][0]["score"], 3 * browse)

    def test_cold_start_and_limit(self):
        self.data["events"] = []
        self.data["limit"] = 2
        result = run(self.data)
        self.assertTrue(result["cold_start"])
        self.assertEqual(len(result["recommendations"]), 2)
        self.assertEqual(result["recommendations"][0]["id"], "SYN-CLAIM-001")

    def test_empty_catalog(self):
        self.data["items"] = []
        self.data["events"] = []
        self.assertEqual(run(self.data)["recommendations"], [])

    def test_privacy_rejects_identifiers_from_synthetic_source(self):
        # Clearly synthetic upstream fixtures are deliberately excluded from model inputs.
        fixture = {"name": "Invented Mira Example", "vin": "SYNVIN00000000001",
                   "address": "999 Imaginary Avenue, Fictionville"}
        for key, value in fixture.items():
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["items"][1]["source"]["payload"]["ACORD"][key] = value
                self.invalid(data)

    def test_underwriting_disclosure(self):
        factor = self.data["items"][3]["source"]["payload"]["ACORD"]["factors"][0]
        factor["disclosed"] = False
        self.invalid()
        factor["disclosed"] = True
        factor["name"] = "ethnicity"
        self.invalid()

    def test_factors_visible(self):
        result = run(self.data)
        submission = next(item for item in result["recommendations"] if item["entity"] == "underwriting_submission")
        self.assertTrue(all(f["disclosed"] for f in submission["attributes"]["factors"]))

    def test_fair_claim_routing_independent_of_behavior(self):
        original = run(self.data)["claim_handling"]
        self.data["events"].append({"id": "SYN-E3", "item_id": "SYN-CLAIM-001",
                                    "action": "browse", "at": self.data["as_of"]})
        self.data["limit"] = 1
        self.assertEqual(run(self.data)["claim_handling"], original)
        self.data["items"][2]["source"]["payload"] = self.data["items"][2]["source"]["payload"].replace("true", "false")
        claim = run(self.data)["claim_handling"][0]
        self.assertEqual(claim["decision"], "request_information")
        self.assertIn("missing", claim["reason"])

    def test_seeded_randomized_synthetic_losses(self):
        rng = random.Random(156)
        for index in range(20):
            amount = round(rng.uniform(100, 100000), 2)
            date = datetime(2026, 9, 24, tzinfo=timezone.utc) - timedelta(days=rng.randint(1, 365))
            self.data["items"][2]["source"]["payload"] = (
                f"line: auto\nloss_amount: {amount}\nloss_date: {date.isoformat()}\ndocuments_complete: true"
            )
            self.assertEqual(run(self.data)["claim_handling"][0]["decision"], "ready_for_human_review")

    def test_future_and_unknown_events(self):
        self.data["events"][0]["at"] = "2027-01-01T00:00:00Z"
        self.invalid()
        self.data["events"][0]["at"] = self.data["as_of"]
        self.data["events"][0]["item_id"] = "SYN-UNKNOWN"
        self.invalid()

    def test_duplicates_and_purchase_constraints(self):
        self.data["events"].append(copy.deepcopy(self.data["events"][0]))
        self.invalid()
        self.data["events"].pop()
        self.data["events"][1]["item_id"] = "SYN-CLAIM-001"
        self.invalid()

    def test_bad_numbers_types_and_timezone(self):
        for field, value in [("half_life_days", 0), ("half_life_days", float("nan")),
                             ("half_life_days", True), ("limit", False), ("limit", 0),
                             ("as_of", "2026-09-24"), ("synthetic", False), ("events", {})]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                data[field] = value
                self.invalid(data)

    def test_xml_security_and_duplicate_fields(self):
        for payload in ["<!DOCTYPE ACORD><ACORD/>", "<ACORD>", "<ACORD><line>auto</line><line>auto</line></ACORD>"]:
            self.data["items"][0]["source"]["payload"] = payload
            self.invalid()

    def test_duplicate_claim_text(self):
        self.data["items"][2]["source"]["payload"] += "\nline: property"
        self.invalid()

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_error_paths_and_invalid_input(self):
        for args in [[], ["missing.json"], ["test_implementation.py"], ["build_manifest.json"]]:
            with self.subTest(args=args):
                proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                      capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main()
