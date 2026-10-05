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


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_example_ranking_and_handoffs(self):
        result = app.compare(self.payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ranking"][0]["option_id"], "SYN-OPT-A")
        self.assertAlmostEqual(result["ranking"][0]["score"], 0.9)
        self.assertEqual(len(result["comparison"]["rows"]), 4)
        self.assertEqual(result["options"][0]["benefits_application"]["case_number"], "SYN-APP-001")
        self.assertEqual(result["options"][0]["policy_document"]["id"], "SYN-POL-A")

    def test_normalization(self):
        result = app.compare(self.payload)
        a, b = result["options"]
        self.assertEqual(a["normalized_attributes"],
                         {"turnaround": 2.0, "fee": 0.0, "accessibility": 100.0, "online": False})
        self.assertEqual(b["normalized_attributes"]["fee"], 5.0)
        self.assertEqual(b["normalized_attributes"]["turnaround"], 7.0)

    def test_preference_changes_ranking(self):
        self.payload["preferences"] = {"turnaround": 0, "fee": 0, "accessibility": 0, "online": 1}
        self.assertEqual(app.compare(self.payload)["ranking"][0]["option_id"], "SYN-OPT-B")

    def test_ties_and_input_order(self):
        self.payload["options"][1]["attributes"] = copy.deepcopy(self.payload["options"][0]["attributes"])
        first = app.compare(self.payload)
        self.payload["options"].reverse()
        self.assertEqual(first, app.compare(self.payload))
        self.assertEqual(first["ranking"][0]["option_id"], "SYN-OPT-A")

    def test_missing_values_and_no_redistribution(self):
        for option in self.payload["options"]:
            option["attributes"]["turnaround"] = None
        result = app.compare(self.payload)
        self.assertIsNone(result["comparison"]["rows"][0]["range"]["minimum"])
        for option in result["options"]:
            self.assertEqual(option["explanation"]["turnaround"]["weighted_points"], 0)
            self.assertEqual(option["explanation"]["turnaround"]["weight"], 0.4)
            self.assertIn("turnaround", option["missing_attributes"])

    def test_single_option(self):
        self.payload["options"] = self.payload["options"][:1]
        result = app.compare(self.payload)
        self.assertEqual(len(result["ranking"]), 1)
        self.assertAlmostEqual(result["ranking"][0]["score"], 0.9)

    def test_all_missing(self):
        for option in self.payload["options"]:
            option["attributes"] = dict.fromkeys(app.ATTRIBUTES)
        result = app.compare(self.payload)
        self.assertTrue(all(row["score"] == 0 for row in result["ranking"]))

    def test_invalid_weights(self):
        for value in [-1, True, float("nan"), float("inf"), "10", 101, 10 ** 400]:
            with self.subTest(value=value):
                data = copy.deepcopy(self.payload)
                data["preferences"]["fee"] = value
                with self.assertRaises(app.ValidationError):
                    app.compare(data)
        self.payload["preferences"] = dict.fromkeys(app.ATTRIBUTES, 0)
        with self.assertRaises(app.ValidationError):
            app.compare(self.payload)

    def test_invalid_units_types_ranges(self):
        for name, invalid in [
            ("fee", {"value": 1, "unit": "EUR"}),
            ("turnaround", {"value": -1, "unit": "days"}),
            ("accessibility", {"value": 6, "unit": "rating5"}),
            ("online", {"value": 1, "unit": "boolean"}),
            ("fee", {"value": True, "unit": "USD"}),
            ("turnaround", {"value": 1, "unit": []}),
        ]:
            with self.subTest(name=name, invalid=invalid):
                data = copy.deepcopy(self.payload)
                data["options"][0]["attributes"][name] = invalid
                with self.assertRaises(app.ValidationError):
                    app.compare(data)

    def test_schema_empty_duplicate_and_bad_links(self):
        mutations = [
            lambda d: d.update(options=[]),
            lambda d: d.update(extra="unsupported"),
            lambda d: d["options"][1].update(id="SYN-OPT-A"),
            lambda d: d["options"][1]["benefits_application"].update(case_number="SYN-APP-001"),
            lambda d: d["options"][0]["benefits_application"].update(request_case_number="SYN-REQ-X"),
            lambda d: d["options"][0]["attributes"].pop("fee"),
            lambda d: d.update(schema_version="2.0"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.payload)
                mutate(data)
                with self.assertRaises(app.ValidationError):
                    app.compare(data)

    def test_privacy_constraints(self):
        mutations = [
            lambda d: d.update(synthetic=False),
            lambda d: d["citizen_service_request"].update(address="123 Actual Street"),
            lambda d: d["citizen_service_request"].update(name="Invented Person"),
            lambda d: d["options"][0].update(label="contact@example.invalid"),
            lambda d: d["options"][0]["policy_document"].update(text="SYNTHETIC POLICY: SSN 123-45-6789"),
            lambda d: d["options"][0]["policy_document"].update(text="Unlabeled rule"),
        ]
        for mutate in mutations:
            data = copy.deepcopy(self.payload)
            mutate(data)
            with self.assertRaises(app.ValidationError):
                app.compare(data)

    def test_privacy_minimization_and_plain_language(self):
        result = app.compare(self.payload)
        rendered = json.dumps(result)
        self.assertNotIn("SYN-PERSONA-001", rendered)
        self.assertNotIn("NOT-A-REAL-ADDRESS", rendered)
        self.assertNotIn("Invented housing help rule", rendered)
        self.assertIn("not a decision about benefits", result["summary"])
        self.assertIn("No Privacy Act", result["review"]["limitations"])

    def test_explainable_recomputable_scores(self):
        result = app.compare(self.payload)
        for option in result["options"]:
            contributions = option["explanation"].values()
            self.assertAlmostEqual(sum(item["weighted_points"] for item in contributions), option["score"])
            self.assertAlmostEqual(sum(item["weight"] for item in contributions), 1)
            for item in contributions:
                self.assertAlmostEqual(item["points"] * item["weight"], item["weighted_points"])
                self.assertTrue(item["explanation"])

    def test_deterministic_without_input_mutation(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(app.compare(self.payload), app.compare(self.payload))
        self.assertEqual(original, self.payload)

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for arguments in [[], [str(ROOT / "missing-input.json")]]:
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                capture_output=True, text=True, cwd=ROOT, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_invalid_json_and_validation_errors(self):
        invalid_payload = copy.deepcopy(self.payload)
        invalid_payload["citizen_service_request"]["address"] = "PRIVATE REJECTED DATA"
        for content in [b"{", b"\xff", b'{"synthetic":NaN}', b'{"a":1,"a":2}',
                        b"[]" , b"x" * (app.MAX_BYTES + 1),
                        json.dumps(invalid_payload).encode("utf-8")]:
            with self.subTest(content_prefix=content[:20]):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")
                self.assertNotIn("PRIVATE REJECTED DATA", output.getvalue())


if __name__ == "__main__":
    unittest.main()
