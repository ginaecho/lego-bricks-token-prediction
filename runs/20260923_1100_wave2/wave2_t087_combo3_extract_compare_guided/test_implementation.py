import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_full_pipeline(self):
        output = app.run(self.data)
        self.assertEqual(output["comparison"]["selected_id"], "synthetic-b")
        self.assertEqual(output["guided"]["selected_id"], "synthetic-b")
        self.assertEqual(output["guided"]["next_step"], "configure")

    def test_source_spans(self):
        self.data["documents"][0]["text"] = "Name:  Café  \r\nPrice: $10\r\nStorage: 1 TB"
        row = app.extract(self.data)["extraction"][0]
        for record in row["fields"].values():
            if record:
                start, end = record["span"]
                self.assertEqual(self.data["documents"][0]["text"][start:end], record["raw"])
        self.assertEqual(row["fields"]["name"]["value"], "Café")

    def test_normalized_units(self):
        fields = app.extract(self.data)["extraction"][1]["fields"]
        self.assertEqual(fields["storage"]["value"], 1000)
        self.assertEqual(fields["warranty"]["value"], 24)

    def test_missing_required_excluded(self):
        self.data["documents"][0]["text"] = "Name: Demo\nWarranty: 2 years"
        output = app.run(self.data)
        self.assertEqual(output["extraction"][0]["missing_fields"], ["price", "storage"])
        self.assertFalse(output["comparison"]["side_by_side"][0]["eligible"])

    def test_missing_optional_allowed(self):
        self.data["documents"][0]["text"] = "Name: Demo\nPrice: 0\nStorage: 0 GB"
        row = app.extract(self.data)["extraction"][0]
        self.assertTrue(row["complete"])
        self.assertEqual(row["missing_fields"], ["warranty"])

    def test_empty_document(self):
        self.data["documents"][0]["text"] = ""
        self.assertEqual(len(app.extract(self.data)["extraction"][0]["missing_fields"]), 4)

    def test_duplicate_field(self):
        self.data["documents"][0]["text"] += "Price: $100\n"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_units(self):
        self.data["documents"][0]["text"] = "Storage: 1 XB"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_budget_propagation(self):
        self.data["preferences"]["max_price"] = 500
        output = app.run(self.data)
        self.assertEqual(output["guided"]["selected_id"], "synthetic-a")
        self.assertIn("insufficient_selected_storage", output["guided"]["steps"][2]["blockers"])

    def test_preference_changes_selection(self):
        self.data["preferences"]["weights"] = {"price": 1, "storage": 0}
        self.assertEqual(app.run(self.data)["guided"]["selected_id"], "synthetic-a")

    def test_tie_break_by_id(self):
        self.data["documents"][1]["text"] = self.data["documents"][0]["text"]
        self.data["documents"].reverse()
        self.assertEqual(app.run(self.data)["comparison"]["selected_id"], "synthetic-a")

    def test_no_eligible_products(self):
        self.data["preferences"]["max_price"] = 1
        self.data["onboarding"]["completed"] = []
        output = app.run(self.data)
        self.assertEqual(output["guided"]["reason"], "no_eligible_product")
        self.assertEqual(output["comparison"]["ranking"], [])

    def test_no_selection_rejects_completion(self):
        self.data["preferences"]["max_price"] = 1
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_prerequisite_validation(self):
        self.data["onboarding"]["completed"] = ["configure"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_evidence_validation(self):
        self.data["onboarding"]["answers"]["purchase_confirmed"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_complete_progress(self):
        self.data["onboarding"]["completed"] = list(app.STEPS)
        self.data["onboarding"]["answers"]["transfer_confirmed"] = True
        guided = app.run(self.data)["guided"]
        self.assertEqual(guided["progress"], 1)
        self.assertEqual(guided["status"], "complete")
        self.assertIsNone(guided["next_step"])

    def test_capacity_blocks_completion(self):
        self.data["onboarding"]["completed"] = list(app.STEPS)
        self.data["onboarding"]["answers"]["transfer_confirmed"] = True
        self.data["onboarding"]["transfer_gb"] = 1001
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_invalid_request_variants(self):
        for key, value in (("documents", []), ("synthetic", False),
                           ("schema_version", "2"), ("onboarding", None)):
            with self.subTest(key=key):
                request = copy.deepcopy(self.data)
                request[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run(request)

    def test_invalid_weights(self):
        for weights in ({"price": 0, "storage": 0}, {"price": True, "storage": 1},
                        {"price": -1, "storage": 1}, {"price": float("nan"), "storage": 1}):
            with self.subTest(weights=weights):
                self.data["preferences"]["weights"] = weights
                with self.assertRaises(app.ValidationError):
                    app.run(self.data)

    def test_duplicate_ids(self):
        self.data["documents"][1]["id"] = self.data["documents"][0]["id"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_handoff_tampering(self):
        extracted = app.extract(self.data)
        extracted["extraction"][0]["fields"]["price"]["value"] = 1
        with self.assertRaises(app.ValidationError):
            app.compare(extracted)
        compared = app.compare(app.extract(self.data))
        compared["comparison"]["selected_id"] = "synthetic-a"
        with self.assertRaises(app.ValidationError):
            app.guide(compared)

    def test_input_not_mutated(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, before)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_cli_success(self):
        proc = self.cli("example_input.json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_missing_file(self):
        proc = self.cli("does-not-exist.json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_missing_argument(self):
        proc = self.cli()
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_invalid_json(self):
        proc = self.cli("implementation.py")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_cli_invalid_schema(self):
        proc = self.cli("build_manifest.json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")

    def test_strict_json(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"x": 1, "x": 2}', object_pairs_hook=app.unique_keys)
        with self.assertRaises(app.ValidationError):
            json.loads('{"x": NaN}', parse_constant=app.reject_constant)


if __name__ == "__main__":
    unittest.main()
