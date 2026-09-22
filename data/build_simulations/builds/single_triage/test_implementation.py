"""Explicitly labeled synthetic fixtures; no production data or external services."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, triage


def fixture_document():
    """SYNTHETIC FIXTURE: configurable helpdesk categories and routing."""
    return {
        "config": {
            "categories": {"billing": "finance", "technical": "support", "other": None},
            "owners": ["finance", "support", "oncall"],
            "rules": [
                {"id": "invoice", "category": "billing", "keywords": ["invoice", "refund"]},
                {"id": "outage", "category": "technical", "keywords": ["offline"],
                 "severity": ["high", "critical"], "urgency": ["high"],
                 "owner": "oncall", "priority": "P0"},
            ],
        },
        "tickets": [{"id": "T1", "subject": "INVOICE issue"}],
    }


class TriageFixtureTests(unittest.TestCase):
    def test_keyword_route_and_explanation(self):
        result = triage(fixture_document())["results"][0]
        self.assertEqual((result["category"], result["owner"], result["status"]),
                         ("billing", "finance", "assigned"))
        self.assertEqual(result["matched_rules"][0]["conditions"], {"keywords": ["invoice"]})
        self.assertEqual(result["priority"], "P2")

    def test_and_conditions_and_any_keyword(self):
        document = fixture_document()
        document["tickets"] = [{"id": "T1", "subject": "", "body": "offline",
                                "severity": "high", "urgency": "low"}]
        self.assertEqual(triage(document)["results"][0]["status"], "unassigned")
        document["tickets"][0]["urgency"] = "high"
        result = triage(document)["results"][0]
        self.assertEqual((result["owner"], result["priority"]), ("oncall", "P0"))
        self.assertEqual(result["matched_rules"][0]["conditions"],
                         {"keywords": ["offline"], "severity": "high", "urgency": "high"})
        document["tickets"][0]["subject"] = "refund"
        self.assertEqual(triage(document)["results"][0]["status"], "ambiguous")

    def test_cross_category_ambiguity(self):
        document = fixture_document()
        document["tickets"][0].update(subject="invoice offline", severity="high", urgency="high")
        result = triage(document)["results"][0]
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["category"])
        self.assertIsNone(result["owner"])
        self.assertEqual(result["candidate_categories"], ["billing", "technical"])

    def test_same_category_owner_conflict(self):
        document = fixture_document()
        document["config"]["rules"].append(
            {"id": "override", "category": "billing", "owner": "oncall", "keywords": ["invoice"]})
        result = triage(document)["results"][0]
        self.assertEqual((result["category"], result["reason"]), ("billing", "conflicting_owners"))
        self.assertIsNone(result["owner"])

    def test_null_owner_conflicts_with_explicit_owner(self):
        document = fixture_document()
        document["config"]["categories"]["billing"] = None
        document["config"]["rules"].append(
            {"id": "override", "category": "billing", "owner": "oncall", "keywords": ["invoice"]})
        self.assertEqual(triage(document)["results"][0]["reason"], "conflicting_owners")

    def test_unmatched_and_ownerless_are_explicit(self):
        document = fixture_document()
        document["tickets"][0]["subject"] = "hello"
        result = triage(document)["results"][0]
        self.assertEqual((result["status"], result["reason"]), ("unassigned", "no_matching_rule"))
        document["config"]["rules"] = [{"id": "other", "category": "other", "keywords": ["hello"]}]
        result = triage(document)["results"][0]
        self.assertEqual((result["category"], result["reason"]), ("other", "category_has_no_owner"))

    def test_severity_only_urgency_only_and_priority_escalation(self):
        document = fixture_document()
        document["config"]["rules"] = [
            {"id": "severity", "category": "billing", "severity": ["low"], "priority": "P1"},
            {"id": "urgency", "category": "billing", "urgency": ["low"], "priority": "P3"},
        ]
        document["tickets"][0].update(severity="low", urgency="low")
        result = triage(document)["results"][0]
        self.assertEqual(result["priority"], "P1")
        self.assertEqual(result["priority_explanation"]["baseline"], "P3")
        self.assertEqual([m["rule_id"] for m in result["matched_rules"]], ["severity", "urgency"])

    def test_baseline_priority_matrix(self):
        document = fixture_document()
        document["config"]["rules"] = []
        expected = {
            "low": ["P3", "P2", "P1"], "medium": ["P2", "P2", "P1"],
            "high": ["P1", "P1", "P0"], "critical": ["P0", "P0", "P0"],
        }
        for severity, priorities in expected.items():
            for urgency, priority in zip(("low", "medium", "high"), priorities):
                with self.subTest(severity=severity, urgency=urgency):
                    document["tickets"][0].update(severity=severity, urgency=urgency)
                    self.assertEqual(triage(document)["results"][0]["priority"], priority)

    def test_stability_and_no_mutation(self):
        document = fixture_document()
        document["tickets"] += [{"id": "A", "subject": "refund"}, {"id": "Z", "subject": "hello"}]
        before = copy.deepcopy(document)
        first = triage(document)
        self.assertEqual(first, triage(document))
        self.assertEqual([t["id"] for t in first["results"]], ["T1", "A", "Z"])
        self.assertEqual(document, before)

    def test_duplicate_ticket_and_rule_ids(self):
        for key in ("tickets", "rules"):
            with self.subTest(key=key):
                document = fixture_document()
                entries = document["tickets"] if key == "tickets" else document["config"]["rules"]
                entries.append(copy.deepcopy(entries[0]))
                with self.assertRaisesRegex(ValidationError, "duplicate"):
                    triage(document)

    def test_invalid_rules(self):
        invalid = [
            {"category": "missing"}, {"owner": "missing"}, {"priority": "urgent"},
            {"keywords": []}, {"keywords": [4]}, {"severity": ["urgent"]},
            {"urgency": ["critical"]}, {"keywords": ["invoice", "invoice"]},
            {"id": ""}, {"unexpected": True}, {"owner": None},
        ]
        for patch in invalid:
            with self.subTest(patch=patch):
                document = fixture_document()
                document["config"]["rules"][0].update(patch)
                with self.assertRaises(ValidationError):
                    triage(document)
        document = fixture_document()
        del document["config"]["rules"][0]["keywords"]
        with self.assertRaises(ValidationError):
            triage(document)

    def test_invalid_shapes_and_tickets(self):
        for invalid in (None, [], {}, {"config": {}, "tickets": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                triage(invalid)
        for patch in ({"severity": []}, {"urgency": "now"}, {"body": None},
                      {"subject": 5}, {"id": " "}, {"unknown": 1}):
            with self.subTest(patch=patch):
                document = fixture_document()
                document["tickets"][0].update(patch)
                with self.assertRaises(ValidationError):
                    triage(document)
        for categories, owners in (({}, ["finance"]), ({"billing": "missing"}, ["finance"]),
                                   ({"billing": None}, []), ({"billing": []}, ["finance"]),
                                   ({"billing": "finance"}, ["finance", "finance"])):
            document = fixture_document()
            document["config"].update(categories=categories, owners=owners)
            with self.assertRaises(ValidationError):
                triage(document)

    def test_classifier_fallback_and_copy_isolation(self):
        document = fixture_document()
        document["tickets"][0]["subject"] = "hello"
        before = copy.deepcopy(document)

        def fixture_classifier(ticket):
            ticket["subject"] = "mutated"
            return {"category": "technical", "owner": "oncall", "confidence": 0.8}

        result = triage(document, fixture_classifier)["results"][0]
        self.assertEqual((result["category"], result["owner"], result["confidence"]),
                         ("technical", "oncall", 0.8))
        self.assertEqual(result["source"], "classifier")
        self.assertEqual(document, before)
        for category, owner in (("billing", "finance"), ("other", None)):
            result = triage(document, lambda t: {"category": category, "confidence": 1})["results"][0]
            self.assertEqual(result["owner"], owner)

    def test_classifier_not_called_for_matches_or_ambiguity(self):
        document = fixture_document()

        def forbidden(ticket):
            self.fail("Classifier must not run when a rule matched")

        triage(document, forbidden)
        document["tickets"][0].update(subject="invoice offline", severity="high", urgency="high")
        triage(document, forbidden)

    def test_invalid_classifier_results(self):
        document = fixture_document()
        document["tickets"][0]["subject"] = "hello"
        invalid = [
            None, {}, {"category": "unknown", "confidence": 1},
            {"category": "billing", "owner": "unknown", "confidence": 1},
            {"category": "billing", "owner": None, "confidence": 1},
            {"category": "billing", "confidence": 1, "priority": "P0"},
        ] + [{"category": "billing", "confidence": value}
             for value in (True, "0.5", None, -0.1, 1.1, float("nan"), float("inf"), 10 ** 400)]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValidationError):
                triage(document, lambda ticket: result)
        with self.assertRaises(ValidationError):
            triage(document, "not callable")

        def failing(ticket):
            raise RuntimeError("fixture failure")

        with self.assertRaisesRegex(ValidationError, "classifier failed"):
            triage(document, failing)

    def test_entire_batch_validated_before_callback(self):
        document = fixture_document()
        document["tickets"] = [{"id": "same", "subject": "hello"}, {"id": "same", "subject": "world"}]
        calls = []
        with self.assertRaises(ValidationError):
            triage(document, lambda ticket: calls.append(ticket))
        self.assertEqual(calls, [])

    def test_empty_batch(self):
        document = fixture_document()
        document["tickets"] = []
        self.assertEqual(triage(document), {"results": []})

    def cli(self, payload=None, *args):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("implementation.py")), *args],
                              input=payload, text=True, capture_output=True, check=False)

    def test_cli_stdin_and_file(self):
        result = self.cli(json.dumps(fixture_document()))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), triage(fixture_document()))
        example = Path(__file__).with_name("example_input.json")
        result = self.cli(None, str(example))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["results"]), 4)

    def test_cli_rejects_invalid_json_duplicate_keys_and_nonfinite(self):
        for payload in ("{", '{"config": {}, "config": {}, "tickets": []}',
                        '{"config": NaN, "tickets": []}', "null"):
            with self.subTest(payload=payload):
                result = self.cli(payload)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error", json.loads(result.stderr))

    def test_cli_missing_file(self):
        result = self.cli(None, str(Path(__file__).with_name("fixture_missing_file.json")))
        self.assertEqual(result.returncode, 2)
        self.assertIn("error", json.loads(result.stderr))


if __name__ == "__main__":
    unittest.main()
