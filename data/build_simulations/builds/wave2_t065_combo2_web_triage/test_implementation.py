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
        self.request = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_research_preserves_full_content_and_provenance(self):
        finding = app.research(self.request)["findings"][0]
        self.assertEqual(finding["content"], self.request["pages"][finding["url"]])
        self.assertEqual(finding["sha256"], app.digest(finding["content"]))
        self.assertEqual(finding["source_id"], "status")

    def test_cross_stage_evidence_drives_routing(self):
        result = app.run_pipeline(self.request)
        ticket = result["tickets"][0]
        self.assertEqual(ticket["route"]["category"], "incident")
        self.assertEqual(ticket["route"]["priority"], 1)
        self.assertEqual(ticket["route"]["owner"], "synthetic-oncall")
        self.assertEqual(ticket["citations"][0]["sha256"],
                         result["research"]["findings"][0]["sha256"])

    def test_unreferenced_research_does_not_leak(self):
        result = app.run_pipeline(self.request)
        self.assertEqual(result["tickets"][1]["route"]["category"], "billing")
        self.assertEqual(result["tickets"][2]["route"]["category"], "general")
        self.assertEqual(result["tickets"][2]["citations"], [])

    def test_priority_wins_and_config_order_breaks_ties(self):
        self.request["tickets"][0]["body"] = "REFUND"
        self.request["routing"]["rules"].reverse()
        self.assertEqual(app.run_pipeline(self.request)["tickets"][0]["route"]["category"],
                         "incident")
        self.request["routing"]["rules"][0]["priority"] = 1
        self.assertEqual(app.run_pipeline(self.request)["tickets"][0]["route"]["category"],
                         "billing")

    def test_empty_batch(self):
        self.request["tickets"] = []
        self.request["sources"] = []
        self.request["pages"] = {}
        result = app.run_pipeline(self.request)
        self.assertEqual(result["tickets"], [])
        self.assertEqual(result["research"]["findings"], [])

    def test_disallowed_urls(self):
        for url in ["http://support.example.test/x", "https://evil.test/x",
                    "https://support.example.test.evil.test/x",
                    "https://user:pass@support.example.test/x",
                    "https://support.example.test:8080/x",
                    "https://support.example.test/x#fragment",
                    "https://support.example.test\\@evil.test/x",
                    "https://support.example.test/\nx"]:
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.allowed_url(url, self.request["allowed_hosts"])

    def test_missing_fixture(self):
        self.request["pages"] = {}
        with self.assertRaisesRegex(app.ValidationError, "offline retrieval"):
            app.run_pipeline(self.request)

    def test_unknown_reference(self):
        self.request["tickets"][0]["source_ids"] = ["unknown"]
        with self.assertRaisesRegex(app.ValidationError, "Unknown source"):
            app.run_pipeline(self.request)

    def test_duplicate_identifiers(self):
        for key in ["sources", "tickets"]:
            request = copy.deepcopy(self.request)
            request[key].append(copy.deepcopy(request[key][0]))
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)

    def test_invalid_priority_and_accountability(self):
        for key, value in [("priority", True), ("priority", 0), ("priority", 5),
                           ("priority", "1"), ("owner", "")]:
            request = copy.deepcopy(self.request)
            request["routing"]["rules"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)

    def test_tampered_handoff_rejected(self):
        for key, value in [("content", "forged"), ("sha256", "bad"),
                           ("url", "https://evil.test/x"), ("source_id", "bad")]:
            researched = app.research(self.request)
            researched["findings"][0][key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.triage(self.request, researched)

    def test_tampered_final_route_rejected(self):
        output = app.run_pipeline(self.request)
        output["tickets"][0]["route"]["owner"] = "unaccountable"
        with self.assertRaises(app.ValidationError):
            app.validate("output", output, self.request)

    def test_shared_schema_invalid_inputs(self):
        for request in [[], {}, dict(self.request, synthetic=False),
                        dict(self.request, schema_version=True),
                        dict(self.request, unknown="value")]:
            with self.subTest(request=request), self.assertRaises(app.ValidationError):
                app.run_pipeline(request)

    def test_cli_success(self):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                    str(ROOT / "example_input.json")],
                                   capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), app.run_pipeline(self.request))
        self.assertEqual(completed.stderr, "")

    def test_cli_missing_file_and_usage(self):
        for args in [[], [str(ROOT / "does-not-exist.json")]]:
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")]
                                       + args, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for content in ["{", '{"schema_version":1,"schema_version":1}', "{}",
                        '["not an input object"]']:
            stream = io.StringIO()
            with patch("builtins.open", mock_open(read_data=content)):
                with contextlib.redirect_stdout(stream):
                    code = app.main(["synthetic-invalid.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stream.getvalue())["status"], "error")

    def test_determinism(self):
        self.assertEqual(app.run_pipeline(self.request), app.run_pipeline(self.request))


if __name__ == "__main__":
    unittest.main()
