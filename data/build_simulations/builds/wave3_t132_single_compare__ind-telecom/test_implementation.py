import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

from implementation import ValidationError, compare, main


ROOT = Path(__file__).resolve().parent


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_comparison(self):
        output = compare(self.data)
        self.assertEqual(output["recommended_product_id"], "PLAN-PLUS")
        self.assertEqual(output["usage_summary"]["voice"], 202)
        self.assertAlmostEqual(output["usage_summary"]["data"], 14.37)
        basic = next(x for x in output["comparison"] if x["product_id"] == "PLAN-BASIC")
        self.assertEqual(basic["monthly_price_usd"], 18)
        self.assertEqual(basic["monthly_voice_minutes"], 300)
        self.assertIn("insufficient_data", basic["preference_gaps"])
        self.assertEqual(output["comparison"][0]["score"], 99.9)

    def test_deterministic_without_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(compare(self.data), compare(self.data))
        self.assertEqual(self.data, original)

    def test_privacy_and_minimization(self):
        output = json.dumps(compare(self.data))
        for secret in ("SYNTH-SUB-1", "+1-202-555-0107", "000000000000001", "goodwill", "no signal"):
            self.assertNotIn(secret, output)
        for key in ("consent", "do_not_sell"):
            data = copy.deepcopy(self.data)
            data["privacy"][key] = False
            with self.assertRaises(ValidationError):
                compare(data)

    def test_every_entity_residency(self):
        for key in ("subscriber", "fault_tickets", "support_chats", "products"):
            data = copy.deepcopy(self.data)
            item = data[key] if key == "subscriber" else data[key][0]
            item["residency"] = "US"
            with self.subTest(key=key), self.assertRaises(ValidationError):
                compare(data)
        self.data["usage_csv"] = self.data["usage_csv"].replace(",EU,", ",US,")
        with self.assertRaises(ValidationError):
            compare(self.data)

    def test_billing_reason_required_even_zero(self):
        self.data["fault_tickets"][0]["billing_adjustment_cents"] = 0
        for reason in (None, "", "   "):
            self.data["fault_tickets"][0]["billing_reason"] = reason
            with self.assertRaises(ValidationError):
                compare(self.data)

    def test_empty_usage_zero_needs_and_free_plan(self):
        self.data["usage_csv"] = self.data["usage_csv"].splitlines()[0] + "\n"
        self.data["preferences"]["minimum_data_gb"] = 0
        self.data["preferences"]["minimum_voice_minutes"] = 0
        self.data["preferences"]["max_monthly_price_usd"] = 0
        self.data["products"][0]["monthly_price"]["value"] = 0
        self.assertEqual(compare(self.data)["recommended_product_id"], "PLAN-BASIC")

    def test_invalid_units_and_numbers(self):
        for value in (-1, True, float("nan"), float("inf"), "30"):
            data = copy.deepcopy(self.data)
            data["products"][0]["monthly_price"]["value"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                compare(data)
        self.data["products"][0]["monthly_price"]["unit"] = "EUR"
        with self.assertRaises(ValidationError):
            compare(self.data)

    def test_csv_validation(self):
        original = self.data["usage_csv"]
        bad_inputs = [original.replace("voice_minutes", "minutes"),
                      original.replace(",83,", ",NaN,"),
                      original.replace("2026-09,119", "2026-10,119"),
                      original + original.splitlines()[1] + "\n",
                      original.replace("CDR-1,SYNTH-SUB-1", "CDR-1,OTHER"),
                      original.replace(",6247", ",6247,extra")]
        for csv_text in bad_inputs:
            self.data["usage_csv"] = csv_text
            with self.subTest(csv=csv_text), self.assertRaises(ValidationError):
                compare(self.data)

    def test_weights_empty_candidates_and_unknown_fields(self):
        for mutation in ("weights", "products", "unknown"):
            data = copy.deepcopy(self.data)
            if mutation == "weights":
                data["preferences"]["weights"] = dict.fromkeys(("price", "data", "voice", "reliability"), 0)
            elif mutation == "products":
                data["products"] = []
            else:
                data["unexpected"] = "ignored?"
            with self.assertRaises(ValidationError):
                compare(data)

    def test_stable_tie_break(self):
        product = self.data["products"][0]
        other = copy.deepcopy(product)
        other["id"] = "PLAN-AAAA"
        self.data["products"] = [product, other]
        self.assertEqual(compare(self.data)["recommended_product_id"], "PLAN-AAAA")

    def test_seeded_random_synthetic_usage(self):
        rng = random.Random(132)
        header = self.data["usage_csv"].splitlines()[0]
        volumes = [rng.randint(100, 10000) for _ in range(10)]
        self.data["usage_csv"] = header + "\n" + "\n".join(
            f"CDR-{i},SYNTH-SUB-1,EU,true,2026-09,{rng.randint(0, 100)},{v}"
            for i, v in enumerate(volumes))
        self.assertAlmostEqual(compare(self.data)["usage_summary"]["data"], sum(volumes) / 1000)

    def test_synthetic_labels_and_transcript_structure(self):
        self.data["synthetic"] = False
        with self.assertRaises(ValidationError):
            compare(self.data)
        self.data["synthetic"] = True
        self.data["support_chats"][0]["messages"][0]["speaker"] = "unknown"
        with self.assertRaises(ValidationError):
            compare(self.data)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        import io
        invalid = ["{", '{"x": 1, "x": 2}', '{"x": NaN}', "null", "[]"]
        data = copy.deepcopy(self.data)
        data["fault_tickets"][0]["billing_reason"] = ""
        invalid.append(json.dumps(data))
        for content in invalid:
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)), patch("sys.stdout", output):
                self.assertEqual(main(["input.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
