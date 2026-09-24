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


class PersonalizationTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_ranking_and_exact_recency(self):
        result = app.personalize(self.data)
        self.assertEqual(result["mode"], "personalized")
        self.assertEqual([r["item_id"] for r in result["recommendations"]],
                         ["item-0003", "item-0002", "item-0001"])
        self.assertEqual(result["recommendations"][0]["score"], 0.8)
        self.assertEqual(result["recommendations"][0]["explanation"]["topic_interest"], 0.8)

    def test_purchase_is_twice_browse(self):
        self.data["events"][0]["at"] = self.data["as_of"]
        result = app.personalize(self.data)
        housing = result["recommendations"][0]["explanation"]["topic_interest"]
        self.assertAlmostEqual(housing, 2 / 3)

    def test_empty_history_cold_start(self):
        self.data["events"] = []
        result = app.personalize(self.data)
        self.assertEqual(result["mode"], "cold_start")
        self.assertEqual(result["recommendations"][0]["score"], 0.8)

    def test_consent_blocks_personalization(self):
        self.data["consent"] = False
        result = app.personalize(self.data)
        self.assertEqual(result["reason"], "consent_not_given")
        self.assertEqual(result["audit"]["events_used"], 0)
        self.assertEqual(result["recommendations"][1]["item_id"], "item-0001")

    def test_ancient_history_and_empty_catalog(self):
        for event in self.data["events"]:
            event["at"] = "1900-01-01T00:00:00Z"
        self.assertEqual(app.personalize(self.data)["mode"], "cold_start")
        self.data["items"] = []
        self.data["events"] = []
        self.assertEqual(app.personalize(self.data)["recommendations"], [])

    def test_recency_not_normalized_away(self):
        self.data["events"] = [self.data["events"][0]]
        self.data["events"][0]["at"] = "2026-07-26T12:00:00Z"
        result = app.personalize(self.data)
        transport = next(r for r in result["recommendations"] if r["item_id"] == "item-0001")
        self.assertEqual(transport["explanation"]["topic_interest"], 0.25)

    def test_stable_order_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.personalize(self.data)
        self.assertEqual(original, self.data)
        self.data["items"].reverse()
        self.data["events"].reverse()
        self.assertEqual(first, app.personalize(self.data))
        self.data["events"] = []
        for item in self.data["items"]:
            item["public_priority"] = 50
        self.assertEqual(app.personalize(self.data)["recommendations"][0]["item_id"], "item-0001")

    def test_invalid_schema_values(self):
        for key, value in [("limit", True), ("limit", 0), ("consent", "yes"),
                           ("items", {}), ("events", None), ("synthetic", False),
                           ("schema_version", True), ("as_of", "2026-01-01")]:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.personalize(data)

    def test_invalid_entity_form_and_events(self):
        mutations = [
            lambda d: d["items"][0].update(entity=[]),
            lambda d: d["items"][0].update(public_priority=101),
            lambda d: d["items"][0]["form"].update(case_number="real-case"),
            lambda d: d["events"][0].update(item_id="item-9999"),
            lambda d: d["events"][0].update(action="approve"),
            lambda d: d["events"][0].update(at="2027-01-01T00:00:00Z"),
            lambda d: d["items"].append(copy.deepcopy(d["items"][0])),
            lambda d: d["events"].append(copy.deepcopy(d["events"][0])),
        ]
        for mutate in mutations:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(app.ValidationError):
                app.personalize(data)

    def test_pii_protection_and_safe_output(self):
        output = json.dumps(app.personalize(self.data))
        for value in ["persona-0001", "CASE-SYN", "NOT A REAL ADDRESS", "imaginary bus"]:
            self.assertNotIn(value, output)
        for value in ["fabricated@example.invalid", "123-45-6789", "555-010-9999"]:
            data = copy.deepcopy(self.data)
            data["items"][2]["policy_text"] = value
            with self.assertRaises(app.ValidationError):
                app.personalize(data)
        self.data["persona"]["name"] = "Fabricated Name"
        with self.assertRaises(app.ValidationError):
            app.personalize(self.data)

    def test_plain_language_and_explainability(self):
        result = app.personalize(self.data)
        self.assertIn("does not decide", result["audit"]["notice"])
        for row in result["recommendations"]:
            e = row["explanation"]
            self.assertLess(len(e["plain_language"].split()), 22)
            expected = e["public_weight"] * e["public_priority"] + e["interest_weight"] * e["topic_interest"]
            self.assertAlmostEqual(expected, row["score"], places=9)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in [[], ["missing-input.json"], ["a", "b"]]:
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_invalid_json_and_validation_without_files(self):
        for payload in ["{", '{"a": 1, "a": 2}', '{"a": NaN}', "[]",
                        json.dumps(dict(self.data, limit=-1))]:
            with patch("builtins.open", mock_open(read_data=payload)):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
