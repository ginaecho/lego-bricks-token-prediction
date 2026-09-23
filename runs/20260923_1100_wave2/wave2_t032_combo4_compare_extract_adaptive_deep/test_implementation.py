"""Standard-library tests; all fixtures are synthetic and no files are created."""

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
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def extraction(self):
        return app.extract(app.compare(self.request))

    def onboarding(self):
        return app.adaptive(self.extraction())

    def result(self):
        return app.run_pipeline(self.request)["results"]

    def test_integrated_example(self):
        output = app.run_pipeline(self.request)
        self.assertEqual(output["stage"], "deep")
        self.assertEqual(output["status"], "ok")
        self.assertEqual(list(output["results"]), list(app.RESULT_NAMES))
        self.assertTrue(output["synthetic"])
        app.validate_envelope(output, "deep")

    def test_attribute_aliases_and_units(self):
        row = app.compare(self.request)["results"]["comparison"]["rows"][0]
        self.assertEqual(row["attributes"], {
            "price_usd": 120, "weight_g": 700, "battery_hours": 8})
        product = copy.deepcopy(self.request["products"][0])
        product["attributes"]["mass"] = {"value": 1, "unit": "lb"}
        self.assertAlmostEqual(app.normalize(product)["weight_g"], 453.59237)

    def test_preference_ranking_and_contributions(self):
        comparison = self.result()["comparison"]
        self.assertEqual(comparison["selected_product_id"], "atlas")
        self.assertEqual(comparison["ranking"][0]["product_id"], "atlas")
        for row in comparison["ranking"]:
            self.assertAlmostEqual(row["score"], sum(
                value for value in row["contributions"].values() if value is not None))

    def test_missing_attribute_penalty(self):
        comparison = self.result()["comparison"]
        row = next(row for row in comparison["rows"] if row["product_id"] == "cedar")
        rank = next(row for row in comparison["ranking"] if row["product_id"] == "cedar")
        self.assertEqual(row["missing_attributes"], ["battery_hours"])
        self.assertEqual(rank["contributions"]["battery_hours"], 0)

    def test_ties_break_by_id_not_input_order(self):
        self.request["products"][1]["attributes"] = copy.deepcopy(
            self.request["products"][0]["attributes"])
        self.request["products"].reverse()
        comparison = app.compare(self.request)["results"]["comparison"]
        self.assertEqual(comparison["ranking"][0]["product_id"], "atlas")
        self.assertEqual(comparison["ranking"][1]["product_id"], "boreal")

    def test_compare_to_extract_selection_propagates(self):
        self.request["comparison"]["criteria"] = [
            {"attribute": "price_usd", "direction": "min", "weight": 1}]
        results = self.result()
        self.assertEqual(results["comparison"]["selected_product_id"], "cedar")
        self.assertEqual(results["extraction"]["document_ids"], [])
        for stage in ("extraction", "onboarding", "research"):
            self.assertEqual(results[stage]["product_id"], "cedar")
        self.assertTrue(all(answer["status"] == "unresolved"
                            for answer in results["research"]["answers"]))

    def test_extraction_typed_values_aliases_and_spans(self):
        extraction = self.extraction()["results"]["extraction"]
        fields = {field["name"]: field for field in extraction["fields"]}
        self.assertTrue(fields["offline"]["value"])
        self.assertEqual(fields["setup_mode"]["value"], "USB")
        self.assertEqual(len(fields["setup_mode"]["sources"]), 2)
        self.assertEqual(fields["warranty_months"]["value"], 24)
        documents = {document["id"]: document["text"] for document in self.request["documents"]}
        for field in fields.values():
            for source in field["sources"]:
                span = source["span"]
                self.assertEqual(documents[source["document_id"]][span["start"]:span["end"]],
                                 source["raw"])

    def test_missing_required_and_conflicting_fields(self):
        extraction = self.extraction()["results"]["extraction"]
        self.assertEqual(extraction["missing_fields"], ["activation_code"])
        self.assertEqual(extraction["missing_required_fields"], ["activation_code"])
        self.assertEqual(extraction["conflicting_fields"], ["warranty_months"])

    def test_invalid_extracted_value_reported_not_guessed(self):
        self.request["documents"][0]["text"] = "Warranty months: many\nOffline capable: perhaps\n"
        self.request["documents"][1]["text"] = ""
        extraction = self.extraction()["results"]["extraction"]
        self.assertEqual(len(extraction["issues"]), 2)
        self.assertIn("warranty_months", extraction["missing_required_fields"])
        self.assertIn("offline", extraction["missing_fields"])

    def test_crlf_whitespace_and_empty_values(self):
        self.request["documents"][0]["text"] = (
            "SYNTHETIC\r\n  Setup mode:   USB  \r\nActivation code:   \r\n"
            "Warranty months: 2.4e1\r\n")
        self.request["documents"][1]["text"] = ""
        extraction = self.extraction()["results"]["extraction"]
        self.assertEqual(extraction["fields"][0]["value"], "USB")
        self.assertEqual(extraction["fields"][1]["value"], 24)
        self.assertEqual(extraction["issues"][0]["raw"], "")
        app.validate_envelope(self.extraction(), "extract")

    def test_schema_labels_are_literal_and_line_anchored(self):
        self.request["extraction_schema"]["fields"][0]["labels"] = ["Setup (mode)"]
        self.request["documents"][0]["text"] = (
            "SYNTHETIC: Setup (mode): ignored\nSetup (mode): cable\n")
        self.request["documents"][1]["text"] = ""
        fields = self.extraction()["results"]["extraction"]["fields"]
        self.assertEqual(fields[0]["value"], "cable")
        self.assertEqual(len(fields[0]["sources"]), 1)

    def test_global_documents_are_included_but_competitor_is_excluded(self):
        self.request["documents"].append({
            "id": "global", "product_ids": [], "text": "SYNTHETIC\nActivation code: DEMO-ONLY\n"})
        extraction = self.extraction()["results"]["extraction"]
        self.assertEqual(extraction["document_ids"], ["atlas-guide", "atlas-listing", "global"])
        self.assertEqual(extraction["missing_fields"], [])

    def test_adaptive_prerequisites_experience_and_preferences(self):
        onboarding = self.onboarding()["results"]["onboarding"]
        tasks = {task["id"]: task for task in onboarding["tasks"]}
        self.assertEqual(tasks["connect"]["state"], "ready")
        self.assertEqual(tasks["scan"]["state"], "waiting")
        self.assertEqual(tasks["register"]["state"], "blocked")
        self.assertEqual(tasks["automate"]["state"], "blocked")
        self.assertEqual(tasks["connect"]["mode"], "visual")
        self.assertIn("one step at a time", tasks["connect"]["explanation"])
        self.assertTrue(any("expert" in reason for reason in tasks["automate"]["reasons"]))
        self.assertEqual(onboarding["research_focus_fields"], ["activation_code", "warranty_months"])

    def test_completed_prerequisites_unlock_next_task(self):
        self.request["onboarding"]["completed"] = ["connect"]
        onboarding = self.onboarding()["results"]["onboarding"]
        self.assertEqual(onboarding["completed_task_ids"], ["connect"])
        self.assertIn("scan", onboarding["ready_task_ids"])

    def test_expert_and_preference_changes_propagate(self):
        self.request["onboarding"].update(
            experience="expert", preferences=["hands_on"], completed=["connect", "scan"])
        tasks = {task["id"]: task for task in self.onboarding()["results"]["onboarding"]["tasks"]}
        self.assertEqual(tasks["automate"]["state"], "ready")
        self.assertEqual(tasks["connect"]["mode"], "hands_on")
        self.assertIn("concise verification", tasks["automate"]["explanation"])

    def test_blocked_prerequisite_propagates(self):
        self.request["onboarding"]["tasks"][0]["requires_fields"] = ["activation_code"]
        tasks = {task["id"]: task for task in self.onboarding()["results"]["onboarding"]["tasks"]}
        self.assertEqual(tasks["scan"]["state"], "blocked")
        self.assertTrue(any("Blocked prerequisites: connect" in reason
                            for reason in tasks["scan"]["reasons"]))

    def test_deep_agreement_disagreement_and_unresolved(self):
        research = self.result()["research"]
        answers = {answer["question_id"]: answer for answer in research["answers"]}
        self.assertEqual(answers["connection"]["status"], "supported")
        self.assertEqual(answers["connection"]["distinct_document_count"], 2)
        self.assertEqual(answers["warranty"]["status"], "disputed")
        self.assertEqual([group["value"] for group in answers["warranty"]["groups"]], [24, 12])
        self.assertEqual(answers["activation"]["status"], "unresolved")
        self.assertEqual(research["disagreements"], ["warranty"])
        self.assertEqual(research["unresolved_questions"], ["activation", "warranty"])
        self.assertEqual([answer["question_id"] for answer in research["answers"]][:2],
                         ["activation", "warranty"])

    def test_single_document_is_not_claimed_as_corroboration(self):
        self.request["documents"] = self.request["documents"][:1]
        self.request["documents"][0]["text"] += "Setup mode: USB\n"
        research = self.result()["research"]
        answer = next(answer for answer in research["answers"]
                      if answer["question_id"] == "connection")
        self.assertEqual(answer["distinct_document_count"], 1)
        self.assertEqual(len(answer["groups"][0]["citations"]), 2)
        self.assertIn("Single-source", answer["summary"])

    def test_contradiction_inside_single_document(self):
        self.request["documents"] = self.request["documents"][:1]
        self.request["documents"][0]["text"] += "Warranty months: 12\n"
        answer = next(answer for answer in self.result()["research"]["answers"]
                      if answer["question_id"] == "warranty")
        self.assertEqual(answer["status"], "disputed")
        self.assertEqual(answer["distinct_document_count"], 1)

    def test_stages_do_not_mutate_handoffs_or_input(self):
        original = copy.deepcopy(self.request)
        comparison = app.compare(self.request)
        snapshot = copy.deepcopy(comparison)
        app.extract(comparison)
        self.assertEqual(comparison, snapshot)
        first = app.run_pipeline(self.request)
        self.assertEqual(first, app.run_pipeline(self.request))
        self.assertEqual(self.request, original)

    def test_empty_documents_and_tasks_are_valid(self):
        self.request["documents"] = []
        self.request["onboarding"]["tasks"] = []
        results = self.result()
        self.assertEqual(results["onboarding"]["tasks"], [])
        self.assertEqual(len(results["research"]["unresolved_questions"]), 4)

    def test_all_missing_comparison_criterion_is_zero(self):
        for product in self.request["products"]:
            product["attributes"] = {"price": {"value": 10, "unit": "USD"}}
        self.request["comparison"]["criteria"] = [
            {"attribute": "battery_hours", "direction": "max", "weight": 1}]
        ranking = app.compare(self.request)["results"]["comparison"]["ranking"]
        self.assertTrue(all(row["score"] == 0 for row in ranking))
        self.assertEqual(ranking[0]["product_id"], "atlas")

    def test_invalid_structures_and_numeric_values(self):
        cases = [
            ("schema_version", True), ("synthetic", False), ("products", []),
            ("documents", None), ("unknown", "extra"),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                request = copy.deepcopy(self.request)
                request[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)
        for value in (True, -1, float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(value_type=type(value).__name__):
                request = copy.deepcopy(self.request)
                request["products"][0]["attributes"]["Price"]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_duplicate_ids_aliases_and_unsupported_units(self):
        for change in ("id", "alias", "unit"):
            with self.subTest(change=change):
                request = copy.deepcopy(self.request)
                if change == "id":
                    request["products"][1]["id"] = "atlas"
                elif change == "alias":
                    request["products"][0]["attributes"]["cost"] = {"value": 1, "unit": "USD"}
                else:
                    request["products"][0]["attributes"]["Price"]["unit"] = "EUR"
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_cycles_and_unknown_references(self):
        for change in ("cycle", "task", "field", "document", "question", "completion"):
            with self.subTest(change=change):
                request = copy.deepcopy(self.request)
                if change == "cycle":
                    request["onboarding"]["tasks"][0]["prerequisites"] = ["scan"]
                elif change == "task":
                    request["onboarding"]["tasks"][0]["prerequisites"] = ["absent"]
                elif change == "field":
                    request["onboarding"]["tasks"][0]["requires_fields"] = ["absent"]
                elif change == "document":
                    request["documents"][0]["product_ids"] = ["absent"]
                elif change == "question":
                    request["research"]["questions"][0]["field"] = "absent"
                else:
                    request["onboarding"]["completed"] = ["scan"]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(request)

    def test_duplicate_labels_and_bad_weights(self):
        self.request["extraction_schema"]["fields"][1]["labels"] = [" setup MODE "]
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.request)
        self.setUp()
        for weight in (0, -1, True, float("inf")):
            self.request["comparison"]["criteria"][0]["weight"] = weight
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.request)

    def test_large_finite_weights_do_not_overflow(self):
        for criterion in self.request["comparison"]["criteria"]:
            criterion["weight"] = 1e308
        app.run_pipeline(self.request)

    def test_wrong_stage_and_tampered_handoffs_are_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.adaptive(app.compare(self.request))
        comparison = app.compare(self.request)
        comparison["results"]["comparison"]["selected_document_ids"].append("boreal-guide")
        with self.assertRaises(app.ValidationError):
            app.extract(comparison)
        extraction = self.extraction()
        extraction["results"]["extraction"]["fields"][0]["sources"][0]["span"]["end"] += 1
        with self.assertRaises(app.ValidationError):
            app.adaptive(extraction)
        onboarding = self.onboarding()
        onboarding["results"]["onboarding"]["unresolved_fields"] = []
        with self.assertRaises(app.ValidationError):
            app.deep(onboarding)

    def cli(self, *args):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.stderr, "")
        return process, json.loads(process.stdout)

    def test_cli_success_one_json_object(self):
        process, output = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0)
        self.assertEqual(output["stage"], "deep")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        process, output = self.cli(str(ROOT / "nonexistent-input.json"))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_invalid_json(self):
        process, output = self.cli(str(ROOT / "implementation.py"))
        self.assertEqual(process.returncode, 2)
        self.assertEqual(output["status"], "error")

    def test_cli_argument_error(self):
        for args in ([], ["one", "two"]):
            process, output = self.cli(*args)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(output["status"], "error")

    def test_cli_schema_error_and_strict_json_using_injected_file(self):
        payloads = ['{}', '{"schema_version":1,"schema_version":1}', '{"value": NaN}',
                    '[]', '{"value": Infinity}']
        for payload in payloads:
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.StringIO(payload)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-input.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
