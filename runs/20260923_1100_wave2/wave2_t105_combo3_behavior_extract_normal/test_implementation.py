"""All fixtures are synthetic; subprocesses are local Python CLI calls only."""

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


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_full_pipeline(self):
        result = self.run_data()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["behavior"]["ranking"][0]["document_id"], "synthetic-solar-kit")
        self.assertEqual(len(result["extraction"]["documents"]), 2)
        self.assertTrue(result["research"]["queries"][0]["findings"])

    def test_purchase_weight(self):
        self.data["events"] = [
            {"document_id": "synthetic-lamp", "kind": "browse", "at": self.data["as_of"]},
            {"document_id": "synthetic-solar-kit", "kind": "purchase", "at": self.data["as_of"]}]
        self.assertEqual(self.run_data()["behavior"]["ranking"][0]["score"], 3.75)

    def test_exact_half_life_decay(self):
        self.data["events"] = [{"document_id": "synthetic-solar-kit", "kind": "browse",
                                "at": "2026-09-16T12:00:00Z"}]
        self.assertEqual(self.run_data()["behavior"]["ranking"][0]["score"], 0.625)

    def test_recent_browse_beats_old_purchase(self):
        self.data["events"] = [
            {"document_id": "synthetic-lamp", "kind": "browse", "at": self.data["as_of"]},
            {"document_id": "synthetic-solar-kit", "kind": "purchase",
             "at": "2025-01-01T12:00:00Z"}]
        self.assertEqual(self.run_data()["behavior"]["ranking"][0]["document_id"], "synthetic-lamp")

    def test_cold_start_popularity(self):
        self.data["events"] = []
        result = self.run_data()["behavior"]
        self.assertTrue(result["cold_start"])
        self.assertEqual(result["ranking"][0]["document_id"], "synthetic-lamp")

    def test_underflow_history_uses_cold_start(self):
        self.data["behavior"]["half_life_days"] = 0.000002
        self.assertTrue(self.run_data()["behavior"]["cold_start"])

    def test_ties_deterministic_and_no_input_mutation(self):
        self.data["events"] = []
        for doc in self.data["documents"]:
            doc["popularity"] = 1
        original = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(first, self.run_data())
        self.assertEqual(self.data, original)
        self.assertEqual(first["behavior"]["ranking"][0]["document_id"], "synthetic-battery")

    def test_empty_catalog(self):
        self.data["documents"] = []
        self.data["events"] = []
        result = self.run_data()
        self.assertEqual(result["extraction"]["documents"], [])
        self.assertTrue(all(not item["findings"] for item in result["research"]["queries"]))

    def test_typed_extraction_and_offsets(self):
        fields = self.run_data()["extraction"]["documents"][0]["fields"]
        self.assertEqual(fields["price"]["value"], 120)
        self.assertIs(type(fields["price"]["value"]), int)
        self.assertEqual(fields["available"]["value"], "2026-10-01")
        for field in fields.values():
            span = field["source"]
            self.assertEqual(self.data["documents"][0]["text"][span["start"]:span["end"]],
                             span["quote"])

    def test_optional_missing_does_not_mark_incomplete(self):
        item = self.run_data()["extraction"]["documents"][1]
        self.assertTrue(item["complete"])
        self.assertEqual(item["missing_fields"],
                         [{"name": "available", "reason": "not_found", "required": False}])

    def test_missing_required_field(self):
        self.data["documents"][0]["text"] = "Synthetic fixture without fields."
        item = self.run_data()["extraction"]["documents"][0]
        self.assertFalse(item["complete"])
        self.assertEqual(len(item["missing_fields"]), 4)

    def test_bad_date_is_reported_missing(self):
        self.data["documents"][0]["text"] = "Available: 2026-02-30"
        missing = self.run_data()["extraction"]["documents"][0]["missing_fields"]
        self.assertEqual(missing[-1]["reason"], "invalid_value")

    def test_bad_integer_is_reported_missing(self):
        self.data["extraction_schema"][1]["pattern"] = r"Price: (?P<value>[^\r\n]+)"
        self.data["documents"][0]["text"] = "Price: twelve"
        item = self.run_data()["extraction"]["documents"][0]
        self.assertEqual(item["missing_fields"][1]["reason"], "invalid_value")

    def test_empty_capture_is_reported(self):
        self.data["documents"][0]["text"] = "Product:    "
        item = self.run_data()["extraction"]["documents"][0]
        self.assertEqual(item["missing_fields"][0]["reason"], "empty_value")

    def test_unicode_and_whitespace_offsets(self):
        self.data["documents"][0]["text"] = "Synthetic 😀\nProduct:   Café solar   \nPrice: 12 credits"
        field = self.run_data()["extraction"]["documents"][0]["fields"]["product"]
        self.assertEqual(field["value"], "Café solar")
        self.assertEqual(field["source"]["start"],
                         self.data["documents"][0]["text"].index("Café"))

    def test_first_match_policy(self):
        self.data["documents"][0]["text"] = "Price: 8 credits\nPrice: 9 credits"
        self.assertEqual(self.run_data()["extraction"]["documents"][0]["fields"]["price"]["value"], 8)

    def test_ranking_handoff_controls_extraction_and_research(self):
        self.data["behavior"]["top_k"] = 1
        result = self.run_data()
        self.assertEqual([item["document_id"] for item in result["extraction"]["documents"]],
                         ["synthetic-solar-kit"])
        for query in result["research"]["queries"]:
            for finding in query["findings"]:
                self.assertEqual(finding["citation"]["document_id"], "synthetic-solar-kit")

    def test_extraction_handoff_controls_research(self):
        self.data["extraction_schema"] = [self.data["extraction_schema"][1]]
        result = self.run_data()
        self.assertEqual(result["research"]["queries"][0]["findings"], [])
        self.assertTrue(result["research"]["queries"][2]["findings"])
        self.assertEqual(result["research"]["queries"][2]["findings"][0]["evidence_fields"], ["price"])

    def test_exact_research_citations(self):
        docs = {doc["id"]: doc["text"] for doc in self.data["documents"]}
        for query in self.run_data()["research"]["queries"]:
            for finding in query["findings"]:
                citation = finding["citation"]
                self.assertEqual(finding["text"], citation["quote"])
                self.assertEqual(citation["quote"],
                                 docs[citation["document_id"]][citation["start"]:citation["end"]])

    def test_no_match_and_punctuation_query(self):
        self.data["research"]["queries"] = ["unobtainium", "???"]
        self.assertTrue(all(not query["findings"] for query in self.run_data()["research"]["queries"]))

    def test_result_limit(self):
        self.data["research"]["max_findings"] = 1
        self.assertEqual(len(self.run_data()["research"]["queries"][1]["findings"]), 1)

    def test_reject_invalid_input_variants(self):
        variations = [
            ("schema_version", True), ("synthetic", False), ("events", {}),
            ("documents", None), ("as_of", "2026-09-23T12:00:00"),
            ("extraction_schema", []), ("research", {"queries": [""], "max_findings": 1})]
        for key, value in variations:
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.data)
                candidate[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(candidate)

    def test_invalid_numeric_options(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.data["behavior"]["half_life_days"] = value
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_unknown_event_and_future_event(self):
        for changes in ({"document_id": "unknown"}, {"at": "2030-01-01T00:00:00Z"},
                        {"kind": "click"}):
            candidate = copy.deepcopy(self.data)
            candidate["events"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(app.ValidationError):
                app.run_pipeline(candidate)

    def test_duplicate_document_and_field(self):
        for key in ("documents", "extraction_schema"):
            candidate = copy.deepcopy(self.data)
            candidate[key].append(copy.deepcopy(candidate[key][0]))
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.run_pipeline(candidate)

    def test_invalid_regex_or_capture(self):
        for pattern in ("[", r"Price: (\d+)"):
            self.data["extraction_schema"][0]["pattern"] = pattern
            with self.assertRaises(app.ValidationError):
                self.run_data()

    def test_tampered_behavior_handoff_rejected(self):
        ranking = app.personalize(self.data)
        ranking["ranking"][0]["document_id"] = "nonexistent"
        with self.assertRaises(app.ValidationError):
            app.extract(self.data, ranking)

    def test_tampered_extraction_handoff_rejected(self):
        ranking = app.personalize(self.data)
        extraction = app.extract(self.data, ranking)
        extraction["documents"][0]["fields"]["price"]["source"]["quote"] = "999"
        with self.assertRaises(app.ValidationError):
            app.research(self.data, ranking, extraction)

    def test_tampered_finding_rejected(self):
        result = self.run_data()
        result["research"]["queries"][0]["findings"][0]["text"] = "Invented answer"
        with self.assertRaises(app.ValidationError):
            app.Validator.research(result["research"], result["extraction"],
                                   result["behavior"], self.data)

    def test_cli_success(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(json.loads(process.stdout), self.run_data())
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        process = self.cli("nonexistent-synthetic-input.json")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")
        self.assertEqual(process.stderr, "")

    def test_cli_usage(self):
        process = self.cli()
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_schema_existing_file(self):
        process = self.cli("build_manifest.json")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_json_decode_and_encoding_errors(self):
        for text in ("{", '{"x": NaN}', '{"x": 1, "x": 2}', "[]"):
            with self.subTest(text=text), patch.object(app.Path, "read_text", return_value=text):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(app.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")
        with patch.object(app.Path, "read_text",
                          side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
