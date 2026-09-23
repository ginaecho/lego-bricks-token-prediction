import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_ranking_and_grounded_explanation(self):
        items = app.recommend(self.data)
        self.assertEqual([x["id"] for x in items], ["solar-garden", "garden"])
        self.assertEqual(items[0]["score"], 5)
        self.assertEqual(items[0]["explanation"],
                         "Matched interests: gardening (weight 3), solar (weight 2)")

    def test_exclusions_override_high_scores(self):
        self.data["preferences"]["interests"]["restricted"] = 1000
        ids = [x["id"] for x in app.recommend(self.data)]
        self.assertNotIn("restricted", ids)
        self.assertNotIn("blocked", ids)

    def test_cross_stage_order_limit_and_identity(self):
        self.data["preferences"]["limit"] = 1
        output = app.run_pipeline(self.data)
        self.assertEqual([x["document_id"] for x in output["extractions"]], ["solar-garden"])
        self.assertEqual(output["extractions"][0]["rank"], 1)
        self.assertEqual(output["fixture_label"], self.data["fixture_label"])

    def test_tie_break_is_id_not_input_order(self):
        self.data["preferences"]["interests"] = {"gardening": 1}
        self.assertEqual([x["id"] for x in app.recommend(self.data)], ["garden", "solar-garden"])
        self.data["documents"].reverse()
        self.assertEqual([x["id"] for x in app.recommend(self.data)], ["garden", "solar-garden"])

    def test_typed_values_and_spans(self):
        output = app.run_pipeline(self.data)
        docs = {d["id"]: d for d in self.data["documents"]}
        for result in output["extractions"]:
            for field in result["fields"]:
                span = field["source_span"]
                if span:
                    self.assertEqual(docs[result["document_id"]]["text"][span["start"]:span["end"]],
                                     span["text"])
        self.assertEqual(output["extractions"][0]["fields"][1]["value"], 8)

    def test_missing_fields_and_required_subset(self):
        self.data["documents"][0]["text"] = "Organizer:   \n"
        result = app.run_pipeline(self.data)["extractions"][1]
        self.assertEqual(result["missing_fields"], ["organizer", "seats", "location"])
        self.assertEqual(result["missing_required_fields"], ["organizer", "seats"])
        self.assertTrue(all(f["value"] is None and f["source_span"] is None for f in result["fields"]))

    def test_unicode_crlf_whitespace_and_first_match(self):
        self.data["documents"][0]["text"] = "  Organizer:  Café 🌱  \r\nSeats: -2\r\nSeats: 99"
        result = app.run_pipeline(self.data)["extractions"][1]
        self.assertEqual(result["fields"][0]["value"], "Café 🌱")
        self.assertEqual(result["fields"][1]["value"], -2)
        span = result["fields"][0]["source_span"]
        self.assertEqual(self.data["documents"][0]["text"][span["start"]:span["end"]], "Café 🌱")

    def test_no_matches_and_zero_limit(self):
        for preferences in ({"interests": {}}, {"limit": 0}):
            data = copy.deepcopy(self.data)
            data["preferences"].update(preferences)
            result = app.run_pipeline(data)
            self.assertEqual(result["recommendations"], [])
            self.assertEqual(result["extractions"], [])

    def test_empty_catalog(self):
        self.data["documents"] = []
        self.assertEqual(app.run_pipeline(self.data)["extractions"], [])

    def test_invalid_inputs(self):
        for weight in (True, 0, -1, float("nan"), float("inf"), "3"):
            with self.subTest(weight=weight):
                data = copy.deepcopy(self.data)
                data["preferences"]["interests"]["solar"] = weight
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)
        for data in ([], {}, {"schema_version": 1}):
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_duplicate_documents_and_schema(self):
        data = copy.deepcopy(self.data)
        data["documents"].append(data["documents"][0])
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(data)
        self.data["extraction_schema"].append(self.data["extraction_schema"][0])
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_rejects_forged_handoff(self):
        items = app.recommend(self.data)
        for change in ("score", "explanation", "id"):
            forged = copy.deepcopy(items)
            forged[0][change] = 99 if change == "score" else "blocked"
            with self.assertRaises(app.ValidationError):
                app.extract(self.data, forged)

    def test_invalid_integer_only_selected_documents(self):
        self.data["documents"][2]["text"] = "Seats: twelve"
        app.run_pipeline(self.data)
        self.data["documents"][0]["text"] = "Seats: twelve"
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_exact_literal_labels(self):
        self.data["extraction_schema"][0]["label"] = "Org(.)"
        self.data["documents"][0]["text"] = "OrgXYZ: wrong\nOrg(.): correct"
        self.assertEqual(app.run_pipeline(self.data)["extractions"][1]["fields"][0]["value"],
                         "correct")

    def test_final_output_validation(self):
        result = app.run_pipeline(self.data)
        result["extractions"][0]["fields"][0]["source_span"]["start"] = 999
        with self.assertRaises(app.ValidationError):
            app.validate("output", result, self.data)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "not-present.json")]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_validation(self):
        for content in ("{broken", '{"a":1,"a":2}', '[]', '{"schema_version":true}'):
            with patch("builtins.open", mock_open(read_data=content)), redirect_stdout(io.StringIO()) as out:
                self.assertEqual(app.main(["fixture.json"]), 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
