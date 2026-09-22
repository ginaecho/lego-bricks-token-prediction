"""All fixtures in this suite are SYNTHETIC, not measured customer data."""

import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as impl


def synthetic(text="Value: 12", field_type="integer", pattern=r"Value: (?P<value>[^\n]+)"):
    return {
        "fixture_label": "SYNTHETIC unit fixture",
        "schema": {"fields": [{"id": "value", "type": field_type, "required": True, "pattern": pattern}]},
        "documents": [{"id": "synthetic_document", "text": text}],
    }


class SyntheticExtractionTests(unittest.TestCase):
    def test_example_types_conflicts_missing_and_rejection(self):
        payload = json.loads(Path(__file__).with_name("example_input.json").read_text(encoding="utf-8"))
        output = impl.extract(payload)
        fields = {field["id"]: field for field in output["fields"]}
        self.assertEqual(output["conflicts"], ["quantity"])
        self.assertEqual(output["missing_required"], ["purchase_order"])
        self.assertEqual([entry["value"] for entry in fields["quantity"]["values"]], [3, 4])
        self.assertEqual(fields["amount"]["values"][0]["value"], 12.5)
        self.assertIs(fields["paid"]["values"][0]["value"], False)
        self.assertEqual(fields["date"]["values"][0]["value"], "2026-09-21")
        self.assertEqual(len(fields["invoice"]["values"][0]["sources"]), 2)
        self.assertEqual(len(output["rejected_candidates"]), 1)
        documents = {doc["id"]: doc["text"] for doc in payload["documents"]}
        for field in output["fields"]:
            for value in field["values"]:
                for source in value["sources"]:
                    self.assertEqual(documents[source["document_id"]][source["start"]:source["end"]], source["text"])

    def test_unicode_offsets_and_no_input_mutation(self):
        payload = synthetic("🧱 café\nValue: 12\nValue: 12")
        original = copy.deepcopy(payload)
        result = impl.extract(payload)["fields"][0]
        self.assertEqual(result["status"], "resolved")
        self.assertEqual([source["start"] for source in result["values"][0]["sources"]], [14, 24])
        self.assertEqual(payload, original)

    def test_missing_optional_and_required(self):
        payload = synthetic("")
        payload["schema"]["fields"].append({"id": "optional", "type": "string", "required": False, "pattern": "(?P<value>absent)"})
        result = impl.extract(payload)
        self.assertEqual(result["missing_required"], ["value"])
        self.assertTrue(all(field["values"] == [] for field in result["fields"]))

    def test_invalid_text_is_not_coerced_or_invented(self):
        for text, kind in [("12x", "integer"), ("NaN", "number"), ("1e999", "number"), ("yes", "boolean"), ("2026-02-30", "date")]:
            with self.subTest(synthetic_value=text):
                result = impl.extract(synthetic("Value: " + text, kind))
                self.assertEqual(result["missing_required"], ["value"])
                self.assertEqual(len(result["rejected_candidates"]), 1)

    def test_valid_type_conversions(self):
        for text, kind, expected in [("-004", "integer", -4), ("1.25e2", "number", 125.0), ("TRUE", "boolean", True), ("2024-02-29", "date", "2024-02-29"), ("  exact  ", "string", "  exact  ")]:
            with self.subTest(synthetic_value=text):
                result = impl.extract(synthetic("Value: " + text, kind))
                actual = result["fields"][0]["values"][0]["value"]
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), type(expected))

    def test_duplicate_ids_and_bad_schema(self):
        invalid = []
        item = synthetic()
        item["schema"]["fields"] *= 2
        invalid.append(item)
        item = synthetic()
        item["documents"] *= 2
        invalid.append(item)
        for key, value in [("type", "money"), ("type", []), ("required", 1), ("id", ""), ("pattern", 12), ("pattern", ""), ("pattern", "x" * 513)]:
            item = synthetic()
            item["schema"]["fields"][0][key] = value
            invalid.append(item)
        for payload in [None, {}, {"schema": [], "documents": []}]:
            invalid.append(payload)
        for payload in invalid:
            with self.subTest(synthetic_invalid=payload):
                with self.assertRaises(impl.ExtractionError):
                    impl.extract(payload)

    def test_invalid_regex_or_capture_contract(self):
        for pattern in ["[", "(abc)", "(?P<other>abc)", "(?P<value>a)(b)"]:
            with self.subTest(synthetic_pattern=pattern):
                with self.assertRaises(impl.ExtractionError):
                    impl.extract(synthetic(pattern=pattern))

    def test_input_limits(self):
        payloads = []
        item = synthetic("x" * (impl.MAX_DOCUMENT_CHARS + 1))
        payloads.append(item)
        item = synthetic()
        item["documents"] = [{"id": str(i), "text": ""} for i in range(17)]
        payloads.append(item)
        item = synthetic()
        item["documents"] = [{"id": str(i), "text": "x" * impl.MAX_DOCUMENT_CHARS} for i in range(5)]
        payloads.append(item)
        item = synthetic()
        item["schema"]["fields"] = [dict(item["schema"]["fields"][0], id=str(i)) for i in range(33)]
        payloads.append(item)
        for payload in payloads:
            with self.assertRaises(impl.ExtractionError):
                impl.extract(payload)

    def test_candidate_limit(self):
        with self.assertRaisesRegex(impl.ExtractionError, "candidate limit"):
            impl.extract(synthetic("a" * (impl.MAX_CANDIDATES + 1), "string", "(?P<value>a)"))

    def test_regex_timeout(self):
        with self.assertRaisesRegex(impl.ExtractionError, "time limit"):
            impl.extract(synthetic("a" * 20000 + "!", "string", "(?P<value>(?:a+)+)$"))

    def test_empty_and_oversized_capture(self):
        for text, pattern in [("", "(?P<value>)"), ("a" * 1025, "(?P<value>a+)")]:
            result = impl.extract(synthetic(text, "string", pattern))
            self.assertEqual(result["missing_required"], ["value"])
            self.assertEqual(len(result["rejected_candidates"]), 1)

    def test_callback_exact_grounding(self):
        payload = synthetic("12", pattern="Never: (?P<value>[0-9]+)")
        def callback(document, fields):
            document["text"] = "mutated copy"
            fields[0]["id"] = "mutated copy"
            return [{"field_id": "value", "start": 0, "end": 2, "value": 12}]
        result = impl.extract(payload, callback)
        self.assertEqual(result["fields"][0]["values"][0]["value"], 12)
        self.assertEqual(result["fields"][0]["values"][0]["sources"][0]["method"], "callback")
        self.assertEqual(payload["documents"][0]["text"], "12")
        self.assertEqual(payload["schema"]["fields"][0]["id"], "value")

    def test_callback_rejects_unknown_ungrounded_and_bad_types(self):
        good = {"field_id": "value", "start": 0, "end": 2, "value": 12}
        bad_items = [
            dict(good, field_id="invented"), dict(good, value=13),
            dict(good, value="12"), dict(good, value=12.0),
            dict(good, value=True), dict(good, start=True),
            dict(good, start=-1), dict(good, end=3), dict(good, end=0),
            dict(good, extra="not permitted"), dict(good, field_id=[]),
        ]
        payload = synthetic("12", pattern="Never: (?P<value>[0-9]+)")
        for item in bad_items:
            with self.subTest(synthetic_callback=item):
                with self.assertRaises(impl.ExtractionError):
                    impl.extract(payload, lambda doc, fields: [item])
        with self.assertRaises(impl.ExtractionError):
            impl.extract(payload, lambda doc, fields: {"value": 12})
        with self.assertRaises(impl.ExtractionError):
            impl.extract(payload, lambda doc, fields: [good] * (impl.MAX_CANDIDATES + 1))
        with self.assertRaises(impl.ExtractionError):
            impl.extract(payload, "not callable")

    def test_callback_supplements_without_hiding_conflicts(self):
        payload = synthetic("Value: 12\n13")
        result = impl.extract(payload, lambda doc, fields: [{"field_id": "value", "start": 10, "end": 12, "value": 13}])
        self.assertEqual(result["conflicts"], ["value"])

    def test_strict_json_and_cli_errors(self):
        for raw in [b'{"schema":1,"schema":2}', b'{"value":NaN}', b'\xff', b'{}', b"x" * (impl.MAX_INPUT_BYTES + 1)]:
            with self.subTest(synthetic_json=raw[:50]):
                with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                    with redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(impl.main(["synthetic.json"]), 2)
                    self.assertIn("error", json.loads(output.getvalue()))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(impl.main([]), 2)
        self.assertIn("Usage", json.loads(output.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
