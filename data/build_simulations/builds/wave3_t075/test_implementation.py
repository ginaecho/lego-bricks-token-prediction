import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).parent


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_benefits_and_privacy(self):
        result = app.extract(self.data)
        self.assertEqual(result["status"], "ok")
        output = json.dumps(result)
        self.assertNotIn("Juniper", output)
        self.assertNotIn("Imaginary", output)
        self.assertEqual(result["fields"][1]["value"], "[REDACTED]")
        self.assertFalse(result["review"]["decision_made"])
        self.assertIn("person must review", result["review"]["explanation"])

    def test_citizen_request(self):
        self.data["document"]["entity"] = "citizen_service_request"
        self.assertEqual(app.extract(self.data)["entity"], "citizen_service_request")

    def test_json_spans_unicode_and_escaped_values(self):
        self.data["document"]["content"]["citizen_name"] = 'Synthetic Éloise "Demo"'
        text, _ = app.source_index(self.data["document"])
        self.assertEqual(text, json.dumps(self.data["document"]["content"],
                                         sort_keys=True, indent=2, ensure_ascii=False))
        for field in app.extract(self.data)["fields"]:
            if field["source_span"]:
                span = field["source_span"]
                value = json.loads(text[span["start"]:span["end"]])
                self.assertEqual(value, self.data["document"]["content"][field["name"]])

    def policy(self):
        return {
            "schema_version": 1, "synthetic": True, "purpose": "policy_review",
            "document": {"id": "SYN-POL-01", "entity": "policy_document",
                         "format": "policy_regulation_text",
                         "content": "SYNTHETIC policy — not a real regulation.\r\nPolicy ID: SYN-POL-01\r\nDays: 0  \r\nActive: false\r\n"},
            "fields": [
                {"name": "policy_id", "source": "Policy ID", "type": "string", "required": True, "pii": False},
                {"name": "deadline_days", "source": "Days", "type": "integer", "required": True, "pii": False},
                {"name": "is_active", "source": "Active", "type": "boolean", "required": True, "pii": False},
            ],
        }

    def test_policy_types_and_spans(self):
        data = self.policy()
        fields = app.extract(data)["fields"]
        self.assertEqual([f["value"] for f in fields], ["SYN-POL-01", 0, False])
        span = fields[1]["source_span"]
        self.assertEqual(data["document"]["content"][span["start"]:span["end"]], "0")

    def test_required_and_optional_missing(self):
        del self.data["document"]["content"]["case_number"]
        result = app.extract(self.data)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_fields"], [
            {"name": "case_number", "required": True},
            {"name": "supporting_document", "required": False},
        ])

    def test_null_blank_zero_and_false(self):
        self.data["document"]["content"].update(citizen_name=None, address="  ", eligibility_age=0)
        fields = app.extract(self.data)["fields"]
        self.assertEqual(fields[1]["status"], "missing")
        self.assertEqual(fields[2]["status"], "missing")
        self.assertEqual(fields[4]["status"], "redacted")
        self.assertIs(fields[5]["value"], False)

    def test_pii_cannot_be_declared_public(self):
        for name in ("citizen_name", "case_number"):
            data = copy.deepcopy(self.data)
            data["fields"][1].update(name=name, pii=False)
            if name == "case_number":
                data["fields"].pop(0)
            with self.assertRaises(app.ValidationError):
                app.extract(data)

    def test_public_free_text_is_rejected(self):
        self.data["document"]["content"]["case_number"] = "Contact synthetic@example.invalid"
        with self.assertRaises(app.ValidationError):
            app.extract(self.data)

    def test_invalid_schema(self):
        for key, value in (("synthetic", False), ("schema_version", True),
                           ("purpose", "advertising"), ("fields", []),
                           ("extra", "unexpected")):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.extract(data)

    def test_wrong_type_boolean_not_integer(self):
        self.data["document"]["content"]["eligibility_age"] = True
        with self.assertRaises(app.ValidationError):
            app.extract(self.data)

    def test_duplicate_policy_label(self):
        data = self.policy()
        data["document"]["content"] += "days: 9\n"
        with self.assertRaises(app.ValidationError):
            app.extract(data)

    def test_duplicate_json_keys_and_nonfinite_numbers(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(app.ValidationError):
                app.load_json(text)

    def test_entity_format_purpose_consistency(self):
        self.data["document"]["entity"] = "policy_document"
        with self.assertRaises(app.ValidationError):
            app.extract(self.data)

    def test_cli_success(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "absent.json")], ["a", "b"]):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_invalid_content_and_no_error_pii_leak(self):
        for text in ('{"synthetic":', '{"secret":"Fabricated Persona Juniper Example"}',
                     '{"x":NaN}', '[]'):
            output = io.StringIO()
            with patch.object(Path, "read_text", return_value=text), contextlib.redirect_stdout(output):
                code = app.main([str(ROOT / "example_input.json")])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")
            self.assertNotIn("Juniper", output.getvalue())

    def test_deterministic(self):
        self.assertEqual(app.extract(self.data), app.extract(copy.deepcopy(self.data)))


if __name__ == "__main__":
    unittest.main()
