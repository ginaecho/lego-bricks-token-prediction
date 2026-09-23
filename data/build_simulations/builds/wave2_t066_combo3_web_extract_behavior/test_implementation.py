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

    def run_data(self):
        return app.run_pipeline(self.data)

    def body(self, value):
        self.data["fixtures"]["https://shop.example/products/red"]["text"] = value

    def test_integrated_example_and_recency(self):
        result = self.run_data()
        ranked = result["behavior"]["recommendations"]
        self.assertEqual([r["id"] for r in ranked], ["red", "blue", "book"])
        self.assertEqual([r["score"] for r in ranked], [6, 3, 1])
        self.assertFalse(result["behavior"]["cold_start"])

    def test_redirect_and_provenance_propagation(self):
        result = self.run_data()
        doc = result["research"]["documents"][0]
        record = result["extraction"]["records"][0]
        recommendation = result["behavior"]["recommendations"][0]
        self.assertEqual(len(doc["retrieval_chain"]), 2)
        self.assertEqual(record["source"]["sha256"], app.digest(doc["text"]))
        self.assertEqual(record["source"], recommendation["source"])

    def test_source_spans_and_unicode(self):
        self.body("ID: red\r\nTitle:   Café bricks  \r\nCategory: toys\r\nPrice: 12.50\r\n")
        result = self.run_data()
        doc = result["research"]["documents"][0]
        field = result["extraction"]["records"][0]["fields"]["title"]
        self.assertEqual(field["value"], "Café bricks")
        self.assertEqual(doc["text"][slice(*field["span"])], "Café bricks")

    def test_missing_optional_field(self):
        record = self.run_data()["extraction"]["records"][1]
        self.assertEqual(record["missing_fields"], ["stock"])
        self.assertEqual(record["missing_required"], [])
        self.assertIsNone(record["fields"]["stock"]["span"])

    def test_missing_required_excludes_and_ignores_history(self):
        self.body("ID: red\nTitle: Red\nCategory: toys\nPrice: \n")
        result = self.run_data()
        self.assertEqual(result["behavior"]["excluded"][0]["missing_required"], ["price"])
        self.assertEqual(result["behavior"]["events_ignored"], 1)
        self.assertNotIn("red", [r["id"] for r in result["behavior"]["recommendations"]])

    def test_cold_start_deterministic(self):
        self.data["history"] = []
        result = self.run_data()
        self.assertTrue(result["behavior"]["cold_start"])
        self.assertEqual([r["id"] for r in result["behavior"]["recommendations"]], ["blue", "book", "red"])
        self.assertEqual(result, self.run_data())

    def test_unknown_history_cold_start(self):
        self.data["history"] = [{"product_id": "unknown", "action": "view", "at": self.data["now"]}]
        behavior = self.run_data()["behavior"]
        self.assertTrue(behavior["cold_start"])
        self.assertEqual(behavior["events_ignored"], 1)

    def test_limit(self):
        self.data["ranking"]["limit"] = 1
        self.assertEqual(len(self.run_data()["behavior"]["recommendations"]), 1)

    def test_no_eligible_candidates(self):
        for key, fixture in self.data["fixtures"].items():
            if "text" in fixture:
                fixture["text"] = "Synthetic listing without fields"
        behavior = self.run_data()["behavior"]
        self.assertEqual(behavior["recommendations"], [])
        self.assertEqual(len(behavior["excluded"]), 3)

    def test_allowlist_blocks_bypasses(self):
        for url in ("http://shop.example/red", "https://shop.example.evil/red",
                    "https://shop.example@evil.example/red", "https://shop.example:444/red",
                    "https://shop.example/red#fragment", "https://shop.example\\evil/red"):
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.canonical_url(url, ["shop.example"])

    def test_redirect_to_unlisted_host(self):
        self.data["fixtures"]["https://shop.example/red"]["redirect"] = "https://evil.example/x"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_redirect_cycle(self):
        self.data["fixtures"]["https://shop.example/red"]["redirect"] = "https://shop.example/red"
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            self.run_data()

    def test_missing_fixture(self):
        del self.data["fixtures"]["https://shop.example/blue"]
        with self.assertRaisesRegex(app.ValidationError, "no offline fixture"):
            self.run_data()

    def test_invalid_field_types(self):
        for price in ("free", "NaN", "Infinity", "-1"):
            with self.subTest(price=price):
                self.body("ID: red\nTitle: Red\nCategory: toys\nPrice: " + price)
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_ambiguous_label_rejected(self):
        self.body("ID: red\nID: other\nTitle: Red\nCategory: toys\nPrice: 1\n")
        with self.assertRaisesRegex(app.ValidationError, "ambiguous"):
            self.run_data()

    def test_duplicate_ids_rejected(self):
        self.data["fixtures"]["https://shop.example/blue"]["text"] = (
            "ID: red\nTitle: Other\nCategory: toys\nPrice: 1")
        with self.assertRaisesRegex(app.ValidationError, "duplicate extracted"):
            self.run_data()

    def test_schema_rejects_duplicate_and_missing_core(self):
        self.data["fields"].append(copy.deepcopy(self.data["fields"][0]))
        with self.assertRaises(app.ValidationError):
            self.run_data()
        self.data["fields"].pop()
        self.data["fields"][0]["required"] = False
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_version_and_reserved_schema_names(self):
        self.data["schema_version"] = True
        with self.assertRaises(app.ValidationError):
            self.run_data()
        self.data["schema_version"] = 1
        self.data["fields"][-1]["name"] = "source"
        with self.assertRaisesRegex(app.ValidationError, "reserved"):
            self.run_data()

    def test_canonical_duplicate_url(self):
        self.assertEqual(app.canonical_url("https://SHOP.EXAMPLE:443/red", ["shop.example"]),
                         "https://shop.example/red")
        self.data["urls"].append("https://SHOP.EXAMPLE:443/red")
        with self.assertRaisesRegex(app.ValidationError, "duplicate requested"):
            self.run_data()

    def test_bad_history_and_ranking(self):
        baseline = copy.deepcopy(self.data)
        for change in ("future", "naive", "action", "half_life", "boolean_limit"):
            self.data = copy.deepcopy(baseline)
            if change == "future":
                self.data["history"][0]["at"] = "2027-01-01T00:00:00Z"
            elif change == "naive":
                self.data["history"][0]["at"] = "2026-09-01T00:00:00"
            elif change == "action":
                self.data["history"][0]["action"] = "click"
            elif change == "half_life":
                self.data["ranking"]["half_life_days"] = 0
            else:
                self.data["ranking"]["limit"] = True
            with self.subTest(change=change), self.assertRaises(app.ValidationError):
                self.run_data()

    def test_tampered_handoff_rejected(self):
        bundle = app.research(self.data)
        bundle["documents"][0]["text"] += "tampered"
        with self.assertRaisesRegex(app.ValidationError, "digest"):
            app.extract(bundle)
        bundle = app.extract(app.research(self.data))
        bundle["records"][0]["fields"]["price"]["value"] = 999
        with self.assertRaisesRegex(app.ValidationError, "source span"):
            app.personalize(bundle)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")
        self.assertEqual(len(run.stdout.splitlines()), 1)

    def test_cli_file_error_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_invalid_json(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "implementation.py")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["status"], "error")

    def test_cli_invalid_schema(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "build_manifest.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["status"], "error")


if __name__ == "__main__":
    unittest.main()
