import copy
import io
import json
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def rejected(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_integrated_example(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["recommendations"]["rows"][0]["offer_id"], "OFFER_WORK")

    def test_attribute_normalization(self):
        rows = {r["offer_id"]: r for r in app.compare(self.data)["rows"]}
        self.assertEqual(rows["OFFER_WORK"]["data_mb"], 25000)
        self.assertEqual(rows["OFFER_BASIC"]["monthly_price_cents"], 2000)
        self.data["offers"][1].update(data_allowance=25000, data_unit="MB")
        self.assertEqual(rows, {r["offer_id"]: r for r in app.compare(self.data)["rows"]})

    def test_cross_stage_propagation_and_no_mutation(self):
        comparison = app.compare(self.data)
        original = copy.deepcopy(comparison)
        result = app.recommend(comparison)
        self.assertEqual(comparison, original)
        self.assertEqual(comparison["context"], result["context"])
        sources = {r["offer_id"]: r for r in comparison["rows"]}
        for row in result["rows"]:
            for key in ("data_mb", "monthly_price_cents", "preference_score", "residency_tag"):
                self.assertEqual(row[key], sources[row["offer_id"]][key])

    def test_preference_ranking_reacts_to_usage(self):
        before = app.compare(self.data)
        self.data["call_detail_records_csv"] = self.data["call_detail_records_csv"].replace("6174", "26174")
        after = app.compare(self.data)
        self.assertGreater(after["context"]["observed_data_mb"], before["context"]["observed_data_mb"])
        self.assertLess(after["rows"][0]["preference_score"], before["rows"][0]["preference_score"])

    def test_interest_bonus(self):
        comparison = app.compare(self.data)
        result = app.recommend(comparison)
        work = next(r for r in result["rows"] if r["offer_id"] == "OFFER_WORK")
        self.assertEqual(work["recommendation_score"], work["preference_score"] + 400)
        self.assertIn("Matched interests: work", work["explanations"])

    def test_exclusion_enforcement(self):
        result = app.run(self.data)
        self.assertIn("OFFER_TRAVEL", [r["offer_id"] for r in result["comparison"]["rows"]])
        self.assertNotIn("OFFER_TRAVEL", [r["offer_id"] for r in result["recommendations"]["rows"]])

    def test_all_excluded(self):
        self.data["subscriber"]["preferences"]["excluded_offer_ids"] = [x["offer_id"] for x in self.data["offers"]]
        self.assertEqual(app.run(self.data)["recommendations"]["rows"], [])

    def test_no_interest_and_empty_usage(self):
        self.data["subscriber"]["preferences"]["interests"] = []
        self.data["call_detail_records_csv"] = ",".join(app.CSV_FIELDS) + "\n"
        result = app.run(self.data)
        self.assertEqual(result["comparison"]["context"]["observed_data_mb"], 0)
        for row in result["recommendations"]["rows"]:
            self.assertEqual(row["recommendation_score"], row["preference_score"])

    def test_deterministic_randomized_synthetic_usage(self):
        rng = random.Random(4040)
        lines = [",".join(app.CSV_FIELDS)]
        volumes = []
        for i in range(8):
            volume = rng.randint(10, 5000)
            volumes.append(volume)
            lines.append(f"CDR_{i},SUB_SYNTH_01,EU,+1-202-555-0101,SYNTH-IMEI-000001,{rng.randint(0, 90)},{volume}")
        self.data["call_detail_records_csv"] = "\n".join(lines)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(app.compare(self.data)["context"]["observed_data_mb"], sum(volumes))

    def test_tie_breaking(self):
        prototype = self.data["offers"][0]
        self.data["offers"] = [dict(prototype, offer_id="OFFER_Z"), dict(prototype, offer_id="OFFER_A")]
        self.assertEqual([r["offer_id"] for r in app.compare(self.data)["rows"]], ["OFFER_A", "OFFER_Z"])

    def test_consent_and_purpose(self):
        for key, value in (("consent_personalization", False), ("purpose", "advertising")):
            with self.subTest(key=key):
                saved = self.data["subscriber"][key]
                self.data["subscriber"][key] = value
                self.rejected()
                self.data["subscriber"][key] = saved

    def test_private_fields_never_emitted(self):
        output = json.dumps(app.run(self.data))
        for sensitive in ("+1-202-555-0101", "SYNTH-IMEI-000001", "SUB_SYNTH_01",
                          "My fictional device", "goodwill credit"):
            self.assertNotIn(sensitive, output)

    def test_each_record_residency_enforced(self):
        for name in ("subscriber", "offers", "network_fault_tickets",
                     "support_chat_transcripts", "billing_adjustments"):
            with self.subTest(name=name):
                data = copy.deepcopy(self.data)
                row = data[name] if name == "subscriber" else data[name][0]
                row["residency_tag"] = "US"
                with self.assertRaises(app.ValidationError):
                    app.run(data)
        self.data["call_detail_records_csv"] = self.data["call_detail_records_csv"].replace(",EU,", ",US,")
        self.rejected()

    def test_adjustment_requires_reason(self):
        self.data["billing_adjustments"][0]["reason"] = " "
        self.rejected()

    def test_invalid_numbers(self):
        for value in (-1, True, "NaN", "Infinity", "no"):
            with self.subTest(value=value):
                self.data["offers"][0]["monthly_price"] = value
                self.rejected()

    def test_invalid_schema_and_duplicate_ids(self):
        self.data["offers"].append(copy.deepcopy(self.data["offers"][0]))
        self.rejected()
        for value in (None, [], {}, {"schema_version": 2}):
            with self.assertRaises(app.ValidationError):
                app.run(value)

    def test_invalid_csv_and_foreign_subscriber(self):
        original = self.data["call_detail_records_csv"]
        for value in ("wrong,headers\n", original.replace("SUB_SYNTH_01", "SUB_OTHER"),
                      original.replace("83,6174", "83,6174,extra"),
                      original.replace("SYNTH-IMEI-000001", "123456789012345")):
            self.data["call_detail_records_csv"] = value
            self.rejected()

    def test_chat_requires_known_fault(self):
        for value in ("FAULT_UNKNOWN", [], {}, None):
            self.data["support_chat_transcripts"][0]["ticket_id"] = value
            self.rejected()

    def test_handoff_requires_normalized_numeric_fields(self):
        for value in ("3500", True, [], None):
            comparison = app.compare(self.data)
            comparison["preferences"]["budget_cents"] = value
            with self.assertRaises(app.ValidationError):
                app.recommend(comparison)

    def test_handoff_rejects_private_extra_fields(self):
        comparison = app.compare(self.data)
        comparison["context"]["phone_number"] = "+1-202-555-0101"
        with self.assertRaises(app.ValidationError):
            app.recommend(comparison)

    def test_free_plan_and_top_limit(self):
        self.data["offers"][0]["monthly_price"] = 0
        self.data["subscriber"]["preferences"]["top_n"] = 1
        self.assertEqual(len(app.run(self.data)["recommendations"]["rows"]), 1)

    def test_handoff_rejects_tampering(self):
        comparison = app.compare(self.data)
        comparison["rows"][0]["preference_score"] = 0
        with self.assertRaises(app.ValidationError):
            app.recommend(comparison)
        comparison = app.compare(self.data)
        comparison["context"]["residency_tag"] = "US"
        with self.assertRaises(app.ValidationError):
            app.recommend(comparison)

    def test_explanations_must_be_grounded(self):
        result = app.recommend(app.compare(self.data))
        result["rows"][0]["explanations"] = ["Guaranteed perfect network"]
        with self.assertRaises(app.ValidationError):
            app.validate(result, "recommendations")

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], [str(HERE / "missing.json")]):
            process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_validation_errors(self):
        for content in ("{", "null", '{"schema_version": NaN}',
                        json.dumps(dict(self.data, synthetic_fixture=False))):
            with patch("builtins.open", return_value=io.StringIO(content)):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
