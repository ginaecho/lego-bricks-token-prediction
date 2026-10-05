import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_and_deterministic(self):
        result = app.run(self.data)
        self.assertEqual(result, app.run(self.data))
        self.assertEqual(result["review_queue"][0], "syn-auth-1")
        self.assertEqual(result["records"][0]["analysis"]["sentiment"]["score"], 3)

    def test_negation_and_boundaries(self):
        self.assertEqual(app.score("not happy")["score"], -2)
        self.assertEqual(app.score("not. happy")["score"], 2)
        self.assertEqual(app.score("not happy but helpful")["score"], 0)
        self.assertEqual(app.score("not the patient is happy")["score"], 2)

    def test_neutral(self):
        self.assertEqual(app.score("synthetic clinical note")["label"], "neutral")

    def test_severity_dominates(self):
        self.data["records"][0]["impact"] = "critical"
        self.data["records"][0]["text"] = "happy"
        self.data["records"][2]["text"] = "denied " * 20
        result = app.run(self.data)
        self.assertEqual(result["review_queue"][0], "syn-patient-1")
        self.assertEqual(result["records"][2]["analysis"]["priority_score"], 29)

    def test_no_mutation_and_complete_audit(self):
        original = copy.deepcopy(self.data)
        result = app.run(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(len(result["audit_trail"]), len(original["records"]))
        for before, after, event in zip(original["records"], result["records"], result["audit_trail"]):
            replay = copy.deepcopy(before)
            for change in event["changes"]:
                self.assertEqual(replay.get(change["path"]), change["before"])
                replay[change["path"]] = change["after"]
            self.assertEqual(replay, after)

    def test_human_review_no_clinical_decision(self):
        result = app.run(self.data)
        for before, after in zip(self.data["records"], result["records"]):
            self.assertTrue(after["analysis"]["human_review_required"])
            self.assertIsNone(after["analysis"]["clinical_decision"])
            for field in ("diagnoses", "labs", "status"):
                if field in before:
                    self.assertEqual(before[field], after[field])

    def test_identifier_fields_rejected(self):
        for key in ("name", "birthDate", "identifier", "address", "telecom"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["records"][0][key] = "prohibited"
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_identifier_text_rejected(self):
        for text in ("Patient Jane Doe", "MRN 123456", "2000-01-01", "test@example.org",
                     "555-123-4567", "https://example.org"):
            with self.subTest(text=text):
                self.data["records"][0]["text"] = text
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_bad_shared_schema(self):
        for value in (None, [], {}, {"schema_version": "1.0", "synthetic": True, "records": []}):
            with self.subTest(value=value):
                with self.assertRaises(app.ValidationError):
                    app.run(value)

    def test_bad_record_types(self):
        for key, value in (("resourceType", []), ("impact", {}), ("version", True),
                           ("synthetic", False), ("text", ""), ("diagnoses", ["unknown"])):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["records"][0][key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_links_duplicates_and_lab_limits(self):
        for mode in ("link", "duplicate", "lab"):
            data = copy.deepcopy(self.data)
            if mode == "link":
                data["records"][1]["patientRef"] = "syn-missing-1"
            elif mode == "duplicate":
                data["records"][1]["id"] = data["records"][0]["id"]
            else:
                data["records"][0]["labs"][0]["value"] = float("nan")
            with self.assertRaises(app.ValidationError):
                app.run(data)

    def test_queue_ties(self):
        for record in self.data["records"]:
            record["impact"] = "low"
            record["text"] = "synthetic"
        self.assertEqual(app.run(self.data)["review_queue"],
                         sorted(r["id"] for r in self.data["records"]))

    def test_cli_success(self):
        p = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                            str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["status"], "ok")
        self.assertEqual(p.stderr, "")

    def test_cli_errors(self):
        for args in ([], ["nonexistent-file.json"], [str(ROOT / "implementation.py")],
                     [str(ROOT / "build_manifest.json")]):
            with self.subTest(args=args):
                p = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                   cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
                self.assertEqual(json.loads(p.stdout)["status"], "error")
                self.assertEqual(p.stderr, "")

    def test_duplicate_json_keys(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"synthetic":true,"synthetic":false}', object_pairs_hook=app.unique_object)


if __name__ == "__main__":
    unittest.main()
