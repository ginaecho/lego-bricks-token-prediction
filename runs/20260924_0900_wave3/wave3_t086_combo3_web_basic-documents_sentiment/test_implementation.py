import copy
import io
import json
import pathlib
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = pathlib.Path(__file__).parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def first_page(self):
        return self.data["fixture_pages"][self.data["urls"][0]]

    def test_integrated_pipeline(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["documents"]["records"]), 3)
        self.assertEqual(result["sentiment"]["issues"][0]["priority"], "P0")

    def test_retrieval_provenance(self):
        source = app.research(self.data)["findings"][0]
        self.assertEqual(source["url"], self.data["urls"][0])
        self.assertEqual(source["sha256"], app.digest(self.first_page()["body"]))

    def test_cross_stage_propagation(self):
        result = app.run(self.data)
        source = result["research"]["findings"][0]
        record = result["documents"]["records"][0]
        issue = result["sentiment"]["issues"][0]
        self.assertEqual(record["provenance"]["sha256"], source["sha256"])
        for key, value in record.items():
            self.assertEqual(issue[key], value)

    def test_allowlist_rejects_subdomain(self):
        self.data["allowed_hosts"] = ["example"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_unsafe_urls(self):
        for url in ["http://feedback.example/a", "https://user@feedback.example/a",
                    "https://feedback.example:80/a", "https://feedback.example/a#x",
                    "https://feedback.example.evil/a", "https://feedback.example/ a"]:
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.check_url(url, ["feedback.example"])

    def test_missing_fixture(self):
        self.data["fixture_pages"].pop(self.data["urls"][0])
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_duplicate_urls(self):
        self.data["urls"].append(self.data["urls"][0])
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_synthetic_label_required(self):
        self.data["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_extract_csv_quoted_text(self):
        rows = app.extract({"media_type": "text/csv",
                            "body": 'id,message\n1,"bad, very slow"\n'})
        self.assertEqual(rows[0]["message"], "bad, very slow")

    def test_invalid_csv(self):
        for body in ["id,id\n1,2", "id,text\n1,2,3", 'id,text\n1,"unterminated']:
            with self.subTest(body=body), self.assertRaises((app.ValidationError, app.csv.Error)):
                app.extract({"media_type": "text/csv", "body": body})

    def test_invalid_json_document(self):
        self.first_page()["body"] = '{"not":"an array"}'
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_mapping_and_whitespace(self):
        self.first_page()["body"] = '[{"id":" X ","message":" good ","urgency":" HIGH "}]'
        row = app.run(self.data)["documents"]["records"][0]
        self.assertEqual((row["issue_id"], row["text"], row["severity"]), ("X", "good", "high"))
        self.assertEqual(row["customer"], "")

    def test_required_field(self):
        self.first_page()["body"] = '[{"id":"X"}]'
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_duplicate_ids_across_sources(self):
        self.first_page()["body"] = '[{"id":"SYN-003","message":"good"}]'
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_severity_and_type(self):
        for field, value in [("urgency", "urgent"), ("message", 42)]:
            row = {"id": "X", "message": "good", field: value}
            self.first_page()["body"] = json.dumps([row])
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run(self.data)

    def test_empty_documents(self):
        self.first_page()["body"] = "[]"
        second = self.data["fixture_pages"][self.data["urls"][1]]
        second["body"] = "id,message\n"
        self.assertEqual(app.run(self.data)["sentiment"]["issues"], [])

    def test_sentiment_and_evidence(self):
        result = app.score("not bad. excellent", "low")
        self.assertEqual(result["raw_score"], 3)
        self.assertTrue(result["evidence"][0]["negated"])
        self.assertEqual(result["normalized_score"], 1.0)

    def test_negation_stops_at_sentence(self):
        self.assertEqual(app.score("not good. bad", "low")["raw_score"], -2)

    def test_neutral_and_mixed(self):
        self.assertEqual(app.score("plain statement", "low")["sentiment"], "neutral")
        self.assertEqual(app.score("good bad", "low")["raw_score"], 0)

    def test_severity_overrides_positive_sentiment(self):
        result = app.score("excellent outage", "low")
        self.assertEqual((result["sentiment"], result["priority"]), ("positive", "P1"))
        self.assertEqual(app.score("good", "critical")["priority"], "P0")

    def test_tampered_handoffs(self):
        found = app.research(self.data)
        found["findings"][0]["body"] += " "
        with self.assertRaises(app.ValidationError):
            app.documents(found)
        document = app.documents(app.research(self.data), self.data["document_rules"])
        document["records"][0]["provenance"]["url"] = "https://evil.example"
        with self.assertRaises(app.ValidationError):
            app.sentiment(document)

    def test_tampered_sentiment(self):
        result = app.run(self.data)["sentiment"]
        result["issues"][0]["raw_score"] = 900
        with self.assertRaises(app.ValidationError):
            app.validate("sentiment", result)

    def test_determinism(self):
        self.assertEqual(app.run(self.data), app.run(copy.deepcopy(self.data)))

    def test_duplicate_json_keys_and_nonfinite(self):
        for raw in ['{"a":1,"a":2}', '{"a":NaN}']:
            with self.assertRaises(app.ValidationError):
                app.load_json(raw)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_usage_errors(self):
        for arguments in [[], [str(ROOT / "missing.json")]]:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                     *arguments], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for raw in ["{bad", "[]", '{"schema_version":"wrong"}']:
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=raw)), redirect_stdout(output):
                code = app.main(["virtual-input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
