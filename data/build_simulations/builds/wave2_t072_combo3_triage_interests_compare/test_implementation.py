"""Standard-library tests; every catalog and ticket is explicitly synthetic."""

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


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def test_example_pipeline_and_accountability(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["triage"]["category"], "audio")
        self.assertEqual(result["triage"]["priority"], "urgent")
        for stage in ("interests", "comparison"):
            self.assertEqual(result[stage]["category"], "audio")
            self.assertEqual(result[stage]["priority"], "urgent")
            self.assertEqual(result[stage]["accountable_route"], result["triage"]["route"])

    def test_fallback_and_default_priority(self):
        self.data["ticket"].update(subject="Question", body="Can you help me?")
        result = app.run_pipeline(self.data)
        self.assertTrue(result["triage"]["used_fallback"])
        self.assertEqual(result["triage"]["category"], "general")
        self.assertEqual(result["triage"]["priority"], "normal")
        self.assertEqual(result["comparison"]["rows"], [])

    def test_category_ties_use_configuration_order(self):
        self.data["ticket"].update(subject="HEADPHONES invoice", body="Question")
        result = app.triage_stage(self.data)
        self.assertEqual(result["triage"]["category"], "audio")
        self.assertEqual(result["triage"]["category_evidence"], ["headphones"])

    def test_word_boundaries_do_not_match_substrings(self):
        self.data["ticket"].update(subject="Unbroken invoice", body="Nonurgent request")
        result = app.triage_stage(self.data)
        self.assertEqual(result["triage"]["priority"], "normal")
        self.assertEqual(result["triage"]["category"], "billing")

    def test_priority_rules_cannot_lower_default(self):
        self.data["triage_policy"]["default_priority"] = "urgent"
        self.data["ticket"]["body"] = "Broken item."
        self.assertEqual(app.triage_stage(self.data)["triage"]["priority"], "urgent")

    def test_interest_ranking_grounded_explanations(self):
        result = app.run_pipeline(self.data)["interests"]
        first, second = result["recommendations"]
        self.assertEqual(first["product_id"], "SYN-P1")
        self.assertEqual(first["interest_score"], 1.0)
        self.assertAlmostEqual(second["interest_score"], 0.4)
        self.assertEqual([hit["tag"] for hit in first["matched_interests"]], ["travel", "wireless"])
        self.assertIn("Catalog tag 'wireless'", second["explanation"][0])

    def test_exclusions_and_category_enforced_before_limit(self):
        self.data["preferences"]["exclusions"]["tags"] = ["LEATHER"]
        result = app.run_pipeline(self.data)
        self.assertEqual([x["product_id"] for x in result["interests"]["excluded"]], ["SYN-P3", "SYN-P4"])
        self.assertEqual(result["interests"]["category_ineligible"], ["SYN-P5"])
        self.assertEqual({x["product_id"] for x in result["comparison"]["rows"]}, {"SYN-P1", "SYN-P2"})

    def test_unit_normalization_and_comparison_ranking(self):
        result = app.run_pipeline(self.data)["comparison"]
        rows = {row["product_id"]: row for row in result["rows"]}
        self.assertEqual(rows["SYN-P1"]["normalized_attributes"]["weight"], 200.0)
        self.assertEqual(rows["SYN-P2"]["normalized_attributes"]["battery_life"], 30.0)
        self.assertEqual(result["rows"][0]["product_id"], "SYN-P2")
        self.assertAlmostEqual(rows["SYN-P2"]["preference_score"], 4 / 6)
        self.assertAlmostEqual(rows["SYN-P1"]["preference_score"], 2 / 6)

    def test_shortlist_is_only_comparison_source(self):
        self.data["preferences"]["limit"] = 1
        result = app.run_pipeline(self.data)
        self.assertEqual([row["product_id"] for row in result["comparison"]["rows"]], ["SYN-P1"])
        self.assertEqual(result["interests"]["eligible_count"], 2)

    def test_route_category_changes_propagate(self):
        self.data["ticket"].update(subject="Refund", body="Invoice question")
        result = app.run_pipeline(self.data)
        self.assertEqual(result["comparison"]["rows"][0]["product_id"], "SYN-P5")
        self.assertEqual(result["comparison"]["accountable_route"]["owner"], "synthetic-billing-oncall")

    def test_missing_values_are_explicit_and_penalized(self):
        self.data["catalog"][0]["attributes"] = {}
        result = app.run_pipeline(self.data)["comparison"]
        row = next(row for row in result["rows"] if row["product_id"] == "SYN-P1")
        self.assertTrue(all(value is None for value in row["normalized_attributes"].values()))
        self.assertEqual(row["preference_score"], 0.0)
        self.assertTrue(all(e["utility"] == 0 for e in row["preference_evidence"]))

    def test_equal_values_and_tie_breaks(self):
        self.data["catalog"][1]["attributes"] = copy.deepcopy(self.data["catalog"][0]["attributes"])
        self.data["catalog"][1]["tags"] = ["travel", "wireless"]
        result = app.run_pipeline(self.data)["comparison"]["rows"]
        self.assertEqual([row["product_id"] for row in result], ["SYN-P1", "SYN-P2"])
        self.assertTrue(all(row["preference_score"] == 1 for row in result))

    def test_empty_preferences_use_stable_fallback(self):
        self.data["preferences"]["interests"] = []
        self.data["preferences"]["attributes"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["comparison"]["ranking_basis"], "interest_score")
        self.assertEqual(result["comparison"]["rows"][0]["product_id"], "SYN-P1")
        self.assertEqual(result["comparison"]["rows"][0]["preference_score"], 0.0)

    def test_empty_catalog_and_all_excluded(self):
        for empty_catalog in (True, False):
            with self.subTest(empty_catalog=empty_catalog):
                data = copy.deepcopy(self.data)
                if empty_catalog:
                    data["catalog"] = []
                else:
                    data["preferences"]["exclusions"]["product_ids"] = [p["id"] for p in data["catalog"]]
                self.assertEqual(app.run_pipeline(data)["comparison"]["rows"], [])

    def test_wildcard_support_category(self):
        self.data["catalog"][0]["support_categories"] = ["*"]
        self.data["ticket"].update(subject="Question", body="Help please")
        self.assertEqual(app.run_pipeline(self.data)["comparison"]["rows"][0]["product_id"], "SYN-P1")

    def test_no_mutation_and_determinism(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(self.data, original)
        self.data["catalog"].reverse()
        self.assertEqual(app.run_pipeline(self.data), first)
        triage = app.triage_stage(self.data)
        before = copy.deepcopy(triage)
        app.interests_stage(self.data, triage)
        self.assertEqual(triage, before)

    def test_tampered_triage_rejected(self):
        state = app.triage_stage(self.data)
        state["triage"]["route"]["owner"] = "forged-owner"
        with self.assertRaises(app.ValidationError):
            app.interests_stage(self.data, state)

    def test_tampered_interests_rejected(self):
        for field, value in (("product_id", "SYN-P3"), ("interest_score", 999),
                             ("explanation", ["Invented claim"])):
            with self.subTest(field=field):
                state = app.interests_stage(self.data, app.triage_stage(self.data))
                state["interests"]["recommendations"][0][field] = value
                with self.assertRaises(app.ValidationError):
                    app.comparison_stage(self.data, state)

    def test_handoff_requires_correct_ticket_and_stage(self):
        state = app.triage_stage(self.data)
        with self.assertRaises(app.ValidationError):
            app.comparison_stage(self.data, state)
        state["ticket_id"] = "OTHER"
        with self.assertRaises(app.ValidationError):
            app.interests_stage(self.data, state)

    def test_stale_handoff_rejected_after_preference_change(self):
        state = app.interests_stage(self.data, app.triage_stage(self.data))
        self.data["preferences"]["exclusions"]["product_ids"].append("SYN-P1")
        with self.assertRaises(app.ValidationError):
            app.comparison_stage(self.data, state)

    def test_strict_schema_rejections(self):
        changes = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(data_label="real"),
            lambda d: d.update(unknown=1),
            lambda d: d["ticket"].update(body=""),
            lambda d: d["preferences"].update(limit=True),
            lambda d: d["preferences"].update(limit=0),
            lambda d: d["preferences"].update(limit=51),
            lambda d: d["triage_policy"].update(fallback_category="unknown"),
            lambda d: d["triage_policy"]["categories"][0]["route"].update(owner=""),
            lambda d: d["catalog"].append(copy.deepcopy(d["catalog"][0])),
            lambda d: d["catalog"][0].update(support_categories=["unknown"]),
            lambda d: d["catalog"][0]["attributes"]["price"].update(unit="EUR"),
            lambda d: d["preferences"]["attributes"][0].update(direction="up"),
            lambda d: d["preferences"]["interests"].append({"tag": "TRAVEL", "weight": 1}),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_numbers_rejected(self):
        for value in (True, -1, float("inf"), float("nan"), "100", 10 ** 400):
            with self.subTest(value=str(value)[:20]):
                data = copy.deepcopy(self.data)
                data["catalog"][0]["attributes"]["weight"]["value"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_normalized_overflow_rejected(self):
        self.data["catalog"][0]["attributes"]["weight"]["value"] = 1e308
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.data)

    def test_large_valid_weights_do_not_overflow(self):
        for item in self.data["preferences"]["interests"] + self.data["preferences"]["attributes"]:
            item["weight"] = 1e308
        result = app.run_pipeline(self.data)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["interests"]["recommendations"][0]["interest_score"], 1.0)

    def test_no_comparison_preferences_uses_interest_order(self):
        self.data["preferences"]["attributes"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["comparison"]["rows"][0]["product_id"], "SYN-P1")


class CliTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                              cwd=HERE, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        result = self.invoke(str(HERE / "example_input.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_usage_and_missing_file(self):
        for args in ((), ("example_input.json", "extra"), ("does-not-exist.json",)):
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_cli_bad_json_and_invalid_input(self):
        for payload in ('{', '{}', '{"x": 1, "x": 2}', '{"x": NaN}', 'null', '[]'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_file_encoding_error(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("invalid UTF-8")), contextlib.redirect_stdout(output):
            code = app.main(["synthetic-fixture.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
