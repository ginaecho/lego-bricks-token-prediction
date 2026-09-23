"""All fixtures are synthetic. Tests perform no network or provider access."""

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
        self.config = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_config(self):
        return app.run_pipeline(self.config)

    def test_extraction_typed_fields_and_spans(self):
        output = app.extract(self.config)
        price = output["products"][0]["fields"]["price"]
        self.assertEqual(price["value"], 80)
        self.assertEqual(price["unit"], "usd")
        a, b = price["span"]
        self.assertEqual(self.config["documents"][0]["text"][a:b], "80 USD")

    def test_missing_required_fields(self):
        product = app.extract(self.config)["products"][0]
        self.assertEqual(product["missing_fields"], ["mass"])
        self.assertEqual(product["missing_required"], ["mass"])

    def test_retrieval_preserves_provenance(self):
        evidence = self.run_config()["research"]["products"][0]["findings"]["mass"][0]
        self.assertEqual(evidence["source"]["url"], self.config["pages"][0]["url"])
        self.assertEqual(evidence["source"]["title"], self.config["pages"][0]["title"])
        a, b = evidence["span"]
        self.assertEqual(self.config["pages"][0]["text"][a:b], evidence["raw"])

    def test_cross_stage_fill_normalize_and_rank(self):
        output = self.run_config()
        product = output["comparison"]["products"][0]
        self.assertEqual(product["missing_required"], [])
        self.assertEqual(product["attributes"]["mass"]["value"], 1.2)
        self.assertEqual(product["attributes"]["mass"]["unit"], "kg")
        self.assertEqual(product["attributes"]["mass"]["evidence"],
                         output["research"]["products"][0]["findings"]["mass"][0])
        self.assertEqual(output["comparison"]["ranking"][0]["product_id"], "alpha")
        self.assertAlmostEqual(product["score"], 4 / 6, places=11)

    def test_document_precedence_and_conflicts(self):
        price = self.run_config()["comparison"]["products"][0]["attributes"]["price"]
        self.assertEqual(price["value"], 80)
        self.assertTrue(price["conflict"])
        self.assertEqual(price["alternatives"][0]["value"], 85)

    def test_equivalent_units_do_not_conflict(self):
        mass = self.run_config()["comparison"]["products"][1]["attributes"]["mass"]
        self.assertFalse(mass["conflict"])
        self.assertEqual(mass["value"], 0.8)

    def test_exact_allowlist_blocks_subdomain_and_suffix(self):
        for host in ("sub.catalog.example", "catalog.example.evil"):
            config = copy.deepcopy(self.config)
            url = "https://" + host + "/alpha"
            config["documents"][0]["research_urls"] = [url]
            config["pages"][0]["url"] = url
            output = app.run_pipeline(config)
            self.assertEqual(output["research"]["products"][0]["retrievals"][0]["status"], "blocked")
            self.assertIsNone(output["comparison"]["products"][0]["attributes"]["mass"]["value"])

    def test_unavailable_fixture_is_reported(self):
        self.config["pages"] = []
        result = self.run_config()
        self.assertEqual(result["research"]["products"][0]["retrievals"][0]["status"], "unavailable")
        self.assertEqual(result["comparison"]["products"][0]["missing_required"], ["mass"])

    def test_empty_documents_and_pages(self):
        for document in self.config["documents"]:
            document["text"] = ""
            document["research_urls"] = []
        self.config["pages"] = []
        result = self.run_config()["comparison"]
        self.assertEqual([p["score"] for p in result["products"]], [0, 0])
        self.assertEqual(result["ranking"][0]["product_id"], "alpha")

    def test_case_and_whitespace_normalization(self):
        self.config["documents"][0]["text"] = "  nAmE:   Alpha   Pack  \r\nPrice: 80 usd\r\nColor: BLUE  "
        product = self.run_config()["comparison"]["products"][0]
        self.assertEqual(product["attributes"]["name"]["value"], "alpha pack")
        self.assertEqual(product["score_components"]["color"], 1)

    def test_invalid_numeric_and_unit_values(self):
        for raw in ("NaN", "Infinity", "12 lb", "not a number", "1e999 usd"):
            with self.subTest(raw=raw):
                config = copy.deepcopy(self.config)
                config["documents"][0]["text"] = "Price: " + raw
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(config)

    def test_schema_validation(self):
        for key, value in (("fields", []), ("documents", []), ("synthetic", False),
                           ("preferences", []), ("allowed_hosts", ["Catalog.Example"]),
                           ("schema_version", True)):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(config)

    def test_invalid_urls(self):
        for url in ("http://catalog.example/a", "https://user:pass@catalog.example/a",
                    "https://catalog.example:444/a", "https://catalog.example/a#fragment",
                    "https://catalog.example:bad/a", "https://catalog.example/a b"):
            with self.subTest(url=url):
                config = copy.deepcopy(self.config)
                config["documents"][0]["research_urls"] = [url]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(config)

    def test_invalid_preferences(self):
        for change in ({"weight": True}, {"weight": -1}, {"field": "unknown"},
                       {"field": "name"}, {"direction": "nearest"}):
            config = copy.deepcopy(self.config)
            config["preferences"][0].update(change)
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(config)

    def test_duplicate_ids_labels_pages(self):
        for key in ("documents", "fields", "pages"):
            config = copy.deepcopy(self.config)
            config[key].append(copy.deepcopy(config[key][0]))
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(config)

    def test_extraction_handoff_rejects_tampering(self):
        output = app.extract(self.config)
        output["products"][0]["fields"]["price"]["span"] = [0, 6]
        with self.assertRaises(app.ValidationError):
            app.research(output, self.config)
        output = app.extract(self.config)
        output["products"][0]["research_urls"] = []
        with self.assertRaises(app.ValidationError):
            app.research(output, self.config)

    def test_research_handoff_rejects_tampering(self):
        output = app.research(app.extract(self.config), self.config)
        output["products"][0]["findings"]["mass"][0]["value"] = 1
        with self.assertRaises(app.ValidationError):
            app.compare(output, self.config)

    def test_input_not_mutated_and_deterministic(self):
        before = copy.deepcopy(self.config)
        first = self.run_config()
        self.assertEqual(self.config, before)
        self.assertEqual(first, self.run_config())

    def test_max_preference_and_equal_values(self):
        self.config["preferences"] = [{"field": "price", "direction": "max", "weight": 1}]
        self.assertEqual(self.run_config()["comparison"]["ranking"][0]["product_id"], "beta")
        self.config["documents"][1]["text"] = "Name: Beta\nPrice: 80 usd"
        result = self.run_config()["comparison"]
        self.assertEqual([p["score"] for p in result["products"]], [1, 1])
        self.assertEqual(result["ranking"][0]["product_id"], "alpha")

    def test_multiple_pages_first_page_wins_with_alternatives(self):
        url = "https://catalog.example/alpha-extra"
        self.config["documents"][0]["research_urls"].append(url)
        self.config["pages"].append({"url": url, "title": "Synthetic second page", "text": "Mass: 1500 g"})
        mass = self.run_config()["comparison"]["products"][0]["attributes"]["mass"]
        self.assertEqual(mass["value"], 1.2)
        self.assertTrue(mass["conflict"])
        self.assertEqual(mass["alternatives"][0]["source"]["url"], url)

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["one", "two"]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                     capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_invalid_shapes(self):
        for contents in ('{', '{"a":1,"a":2}', '{"x":NaN}', '[]', '{}', 'null'):
            with self.subTest(contents=contents):
                stdout = io.StringIO()
                with patch("builtins.open", mock_open(read_data=contents)), redirect_stdout(stdout):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_cli_encoding_failure(self):
        stdout = io.StringIO()
        with patch("builtins.open", side_effect=UnicodeError("Synthetic decode error")), redirect_stdout(stdout):
            self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_cli_oversized_integer_is_json_error(self):
        stdout = io.StringIO()
        with patch("builtins.open", mock_open(read_data='{"schema_version":' + "9" * 5000 + '}')):
            with redirect_stdout(stdout):
                self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
