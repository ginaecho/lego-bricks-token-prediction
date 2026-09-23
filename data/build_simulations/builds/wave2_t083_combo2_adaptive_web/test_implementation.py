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


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_novice_prerequisites_and_explanations(self):
        plan = app.adaptive_onboarding(self.data)
        self.assertEqual([s["id"] for s in plan["steps"]], list(app.STEPS))
        self.assertEqual(plan["steps"][1]["prerequisites"], ["basics"])
        self.assertTrue(all("Prerequisites:" in s["explanation"] for s in plan["steps"]))

    def test_experience_credits(self):
        self.data["profile"]["experience"] = "expert"
        plan = app.adaptive_onboarding(self.data)
        self.assertEqual(plan["credited"], ["basics", "source_evaluation"])
        self.assertEqual([s["id"] for s in plan["steps"]], ["research"])

    def test_intermediate_and_completed(self):
        self.data["profile"].update(experience="intermediate", completed=["source_evaluation"])
        self.assertEqual(len(app.adaptive_onboarding(self.data)["steps"]), 1)
        self.data["profile"].update(experience="novice", completed=list(app.STEPS))
        self.assertEqual(app.adaptive_onboarding(self.data)["steps"], [])

    def test_invalid_completed_prerequisite(self):
        self.data["profile"]["completed"] = ["research"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_preference_propagates_to_retrieval(self):
        detailed = app.run(self.data)
        self.data["profile"]["preference"] = "brief"
        brief = app.run(self.data)
        self.assertEqual(len(detailed["research"]["findings"]), 2)
        self.assertEqual(len(brief["research"]["findings"]), 1)
        self.assertNotIn("Prerequisites:", brief["onboarding"]["steps"][0]["explanation"])

    def test_query_and_provenance_handoff(self):
        self.data["profile"]["topic"] = "  sorting  "
        result = app.run(self.data)
        self.assertEqual(result["research"]["query"], "sorting")
        finding = result["research"]["findings"][0]
        document = self.data["research"]["fixture_documents"][finding["url"]]
        self.assertEqual(document["text"][finding["start"]:finding["end"]], finding["quote"])
        self.assertEqual(finding["fixture_id"], document["fixture_id"])
        self.assertEqual(finding["title"], document["title"])
        self.assertTrue(finding["synthetic"])

    def test_tampered_handoff_rejected(self):
        plan = app.adaptive_onboarding(self.data)
        plan["steps"].pop(0)
        with self.assertRaises(app.ValidationError):
            app.web_research(plan)

    def test_tampered_provenance_rejected(self):
        result = app.run(self.data)
        result["research"]["findings"][0]["quote"] = "invented evidence"
        with self.assertRaises(app.ValidationError):
            app.validate("output", result)

    def test_no_matches_is_success(self):
        self.data["profile"]["topic"] = "astronomy"
        result = app.run(self.data)
        self.assertEqual(result["research"]["findings"], [])
        self.assertEqual(result["research"]["unmatched_sources"],
                         self.data["research"]["sources"])

    def test_allowlist_url_boundaries(self):
        for url in ("http://research.example/a", "https://evil.example/a",
                    "https://research.example.evil/a", "https://user@research.example/a",
                    "https://research.example:443/a", "https://research.example/a#b",
                    "https://research.example\\@evil.example/a", "https://research.example/\na"):
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.canonical_url(url, ["research.example"])
        self.assertEqual(app.canonical_url("https://RESEARCH.EXAMPLE", ["research.example"]),
                         "https://research.example/")

    def test_missing_fixture_duplicate_and_bad_limit(self):
        for change in ("missing", "duplicate", "limit"):
            data = copy.deepcopy(self.data)
            if change == "missing":
                data["research"]["fixture_documents"].clear()
            elif change == "duplicate":
                data["research"]["sources"].append(data["research"]["sources"][0])
            else:
                data["research"]["max_results"] = True
            with self.subTest(change=change), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_limit_and_determinism_no_input_mutation(self):
        self.data["research"]["max_results"] = 1
        before = copy.deepcopy(self.data)
        first = app.run(self.data)
        self.assertEqual(first, app.run(self.data))
        self.assertEqual(self.data, before)
        self.assertEqual(len(first["research"]["findings"]), 1)
        self.assertEqual(first["research"]["findings"][0]["score"], 2)

    def test_invalid_schema_and_empty_query(self):
        for value in (None, [], {}, {"schema_version": 2}):
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.run(value)
        self.data["profile"]["topic"] = "!!!"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_whitespace_offsets(self):
        url = self.data["research"]["sources"][0]
        self.data["research"]["fixture_documents"][url]["text"] = "  Brick recycling! \n brick."
        result = app.run(self.data)
        for finding in result["research"]["findings"]:
            source = self.data["research"]["fixture_documents"][finding["url"]]["text"]
            self.assertEqual(source[finding["start"]:finding["end"]], finding["quote"])

    def test_cli_success(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_newline_delimited_passages(self):
        url = self.data["research"]["sources"][0]
        self.data["research"]["fixture_documents"][url]["text"] = "brick recycling\nunrelated"
        findings = app.run(self.data)["research"]["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["quote"], "brick recycling")

    def test_cli_file_and_usage_errors(self):
        for args in ([], ["nonexistent-input.json"]):
            process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                      *args], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_malformed_json_and_validation_errors(self):
        for raw in ("{", '{"x":1,"x":2}', '{"schema_version":1}', "null", "[]"):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(output):
                status = app.main(["fixture.json"])
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
