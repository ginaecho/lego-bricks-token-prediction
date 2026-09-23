import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_handoffs(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["behavior"]["document_ids"], ["solar-guide", "tent-guide"])
        self.assertEqual([r["document_id"] for r in result["extraction"]["records"]],
                         result["behavior"]["document_ids"])
        self.assertTrue(all(f["citation"]["document_id"] != "lamp-guide"
                            for f in result["research"]["findings"]))

    def test_recency_changes_ranking(self):
        self.data["events"] = [{"item_id": "lamp", "kind": "browse", "age_days": 0},
                               {"item_id": "solar-kit", "kind": "purchase", "age_days": 100}]
        self.assertEqual(app.personalize(self.data)["behavior"]["ranked_items"][0]["item_id"], "lamp")

    def test_purchase_weight(self):
        self.data["events"] = [{"item_id": "lamp", "kind": "browse", "age_days": 0},
                               {"item_id": "solar-kit", "kind": "purchase", "age_days": 0}]
        self.assertEqual(app.personalize(self.data)["behavior"]["ranked_items"][0]["item_id"], "solar-kit")

    def test_cold_start(self):
        self.data["events"] = []
        result = app.personalize(self.data)["behavior"]
        self.assertEqual(result["mode"], "cold_start")
        self.assertEqual(result["ranked_items"][0]["item_id"], "lamp")

    def test_underflow_cold_start(self):
        for event in self.data["events"]:
            event["age_days"] = 1e300
        self.assertEqual(app.personalize(self.data)["behavior"]["mode"], "cold_start")

    def test_tie_break_deterministic(self):
        self.data["events"] = []
        for item in self.data["items"]:
            item["popularity"] = 0
        first = app.run(self.data)
        self.data["items"].reverse()
        self.assertEqual(first["behavior"], app.run(self.data)["behavior"])
        self.assertEqual(first["behavior"]["ranked_items"][0]["item_id"], "lamp")

    def test_source_spans_and_missing(self):
        result = app.run(self.data)
        field = result["extraction"]["records"][0]["fields"]["price_usd"]
        self.assertEqual(field["value"], 120)
        cite = field["citation"]
        self.assertEqual(self.data["documents"][0]["text"][cite["start"]:cite["end"]], "120")
        tent = result["extraction"]["records"][1]
        self.assertEqual(tent["missing_fields"], ["warranty_months"])
        self.assertEqual(tent["missing_required_fields"], [])

    def test_required_missing(self):
        self.data["fields"][1]["required"] = True
        result = app.run(self.data)
        self.assertEqual(result["extraction"]["records"][1]["missing_required_fields"],
                         ["warranty_months"])

    def test_conversion_error(self):
        self.data["fields"][0]["pattern"] = r"Price: (\w+) USD"
        self.data["documents"][0]["text"] = "Price: unknown USD."
        record = app.run(self.data)["extraction"]["records"][0]
        self.assertIn("price_usd", record["missing_fields"])
        self.assertEqual(record["errors"][0]["reason"], "not a finite number")

    def test_string_unicode_and_first_match(self):
        self.data["documents"][0]["text"] = "🌞 Name: Café. Name: Other."
        self.data["fields"] = [{"name": "name", "pattern": r"Name: (\w+)",
                               "type": "string", "required": True}]
        self.data["query"] = "café"
        result = app.run(self.data)
        field = result["extraction"]["records"][0]["fields"]["name"]
        self.assertEqual(field["value"], "Café")
        self.assertEqual(field["citation"]["start"], 8)
        finding = result["research"]["findings"][0]
        self.assertEqual(finding["text"], "🌞 Name: Café.")
        self.assertEqual(finding["extracted_fields"], ["name"])

    def test_exact_research_citations(self):
        result = app.run(self.data)
        docs = {d["id"]: d["text"] for d in self.data["documents"]}
        for finding in result["research"]["findings"]:
            cite = finding["citation"]
            self.assertEqual(finding["text"], docs[cite["document_id"]][cite["start"]:cite["end"]])
        self.assertTrue(any("warranty_months" in f["extracted_fields"]
                            for f in result["research"]["findings"]))

    def test_no_results(self):
        self.data["query"] = "xyzunmatched"
        self.assertTrue(app.run(self.data)["research"]["no_results"])
        self.data["query"] = "!!!"
        self.assertEqual(app.run(self.data)["research"]["findings"], [])

    def test_newline_passages_and_empty_document(self):
        self.data["documents"][0]["text"] = "  Solar power\nOutdoor travel\n\n"
        self.data["documents"][2]["text"] = ""
        self.data["query"] = "solar"
        result = app.run(self.data)
        self.assertEqual(result["research"]["findings"][0]["text"], "Solar power")
        self.assertEqual(result["research"]["findings"][0]["citation"]["start"], 2)
        self.assertEqual(result["extraction"]["records"][1]["missing_fields"],
                         ["price_usd", "warranty_months"])

    def test_empty_catalog(self):
        self.data["items"], self.data["documents"], self.data["events"] = [], [], []
        result = app.run(self.data)
        self.assertEqual(result["extraction"]["records"], [])
        self.assertEqual(result["research"]["findings"], [])

    def test_shared_document_deduplication(self):
        self.data["items"][2]["document_ids"] = ["solar-guide"]
        self.assertEqual(len(app.run(self.data)["extraction"]["records"]), 1)

    def test_reject_invalid_input(self):
        for path, value in [(("config", "half_life_days"), 0),
                            (("config", "max_items"), True),
                            (("config", "max_findings"), -1),
                            (("schema_version",), "2"),
                            (("query",), ""),
                            (("synthetic",), False)]:
            data = copy.deepcopy(self.data)
            target = data
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_reject_unknown_references_duplicates_and_regex(self):
        variants = []
        bad = copy.deepcopy(self.data)
        bad["events"][0]["item_id"] = "unknown"
        variants.append(bad)
        bad = copy.deepcopy(self.data)
        bad["documents"].append(bad["documents"][0])
        variants.append(bad)
        for pattern in ("(", "no group", "(two)(groups)"):
            bad = copy.deepcopy(self.data)
            bad["fields"][0]["pattern"] = pattern
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(app.ValidationError):
                app.run(bad)

    def test_tampered_behavior_handoff(self):
        result = app.personalize(self.data)
        result["behavior"]["document_ids"].append("lamp-guide")
        with self.assertRaises(app.ValidationError):
            app.extract(result)

    def test_tampered_extraction_handoff(self):
        result = app.extract(app.personalize(self.data))
        result["extraction"]["records"][0]["fields"]["price_usd"]["citation"]["quote"] = "fake"
        with self.assertRaises(app.ValidationError):
            app.research(result)

    def test_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(original, self.data)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              capture_output=True, text=True, cwd=ROOT)

    def test_cli_success(self):
        process = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(process.returncode, 0, process.stdout)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [(), ("does-not-exist.json",), ("a", "b")]:
            with self.subTest(args=args):
                process = self.cli(*args)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_bad_json_and_validation(self):
        for payload in ('{', '{"a": NaN}', '{"a": 1, "a": 2}', '{}', 'null'):
            with self.subTest(payload=payload):
                with patch("builtins.open", mock_open(read_data=payload)), patch("builtins.print") as printer:
                    self.assertEqual(app.main(["fixture.json"]), 2)
                    self.assertEqual(json.loads(printer.call_args.args[0])["status"], "error")


if __name__ == "__main__":
    unittest.main()
