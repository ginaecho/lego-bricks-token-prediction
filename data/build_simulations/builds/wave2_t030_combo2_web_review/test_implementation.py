import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent
URL = "https://research.example/policy"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text("utf-8"))

    def run_example(self):
        return app.run_pipeline(self.payload)

    def test_retrieval_preserves_redirect_provenance_and_hash(self):
        source = self.run_example()["research"]["sources"][0]
        self.assertEqual(source["final_url"], URL)
        self.assertEqual(source["redirect_chain"],
                         ["https://research.example/old-policy", URL])
        self.assertEqual(source["sha256"],
                         hashlib.sha256(source["text"].encode()).hexdigest())

    def test_findings_quote_original_source(self):
        research = self.run_example()["research"]
        self.assertEqual(len(research["findings"]), 2)
        for finding in research["findings"]:
            source = next(s for s in research["sources"]
                          if s["id"] == finding["source_id"])
            self.assertEqual(source["text"][finding["start"]:finding["end"]],
                             finding["quote"])
            self.assertEqual(source["sha256"], finding["source_sha256"])

    def test_review_traces_supported_checks_and_gaps(self):
        output = self.run_example()
        checks = output["review"]["checks"]
        self.assertEqual(checks[0]["status"], "textually_supported")
        self.assertEqual(checks[0]["finding_ids"], ["finding-1"])
        self.assertEqual(checks[2]["status"], "gap")
        self.assertEqual(output["review"]["gap_count"], 2)
        self.assertEqual({g["kind"] for g in checks[2]["gaps"]},
                         {"document", "evidence"})
        self.assertEqual(checks[2]["gaps"][0]["checked_source_ids"],
                         ["src-1", "src-2"])
        self.assertIn("not certification", output["review"]["notice"])

    def test_source_changes_propagate_to_review(self):
        self.payload["request"]["fixtures"][URL]["text"] = "SYNTHETIC: no evidence."
        output = self.run_example()
        self.assertEqual(output["research"]["findings"], [])
        self.assertEqual(output["review"]["gap_count"], 4)
        self.assertEqual(output["review"]["checks"][0]["gaps"][0]["kind"], "evidence")

    def test_document_gap_despite_source_evidence(self):
        self.payload["request"]["document"]["text"] = ""
        check = self.run_example()["review"]["checks"][0]
        self.assertEqual(check["finding_ids"], ["finding-1"])
        self.assertEqual(check["gaps"][0]["kind"], "document")

    def test_retrieval_failures_are_traceable(self):
        del self.payload["request"]["fixtures"][URL]
        sources = self.run_example()["research"]["sources"]
        self.assertEqual(sources[0]["error"], "fixture_missing")
        self.assertEqual(sources[1]["error"], "http_503")

    def test_redirect_loop_is_bounded(self):
        self.payload["request"]["fixtures"][URL] = {
            "status_code": 301, "location": "https://research.example/old-policy"}
        output = self.run_example()
        self.assertEqual(output["research"]["sources"][0]["error"], "redirect_loop")
        self.assertEqual(output["research"]["findings"], [])

    def test_redirect_limit(self):
        fixtures = self.payload["request"]["fixtures"]
        for index in range(7):
            fixtures[f"https://research.example/hop{index}"] = {
                "status_code": 302,
                "location": f"https://research.example/hop{index + 1}"}
        self.payload["request"]["urls"] = ["https://research.example/hop0"]
        self.assertEqual(self.run_example()["research"]["sources"][0]["error"],
                         "redirect_limit")

    def test_url_allowlist_rejects_bypasses_and_redirects(self):
        for url in ["http://research.example/page", "https://evil.example/page",
                    "https://research.example.evil.example/page",
                    "https://user@research.example/page",
                    "https://research.example:8443/page",
                    "https://research.example/page#fragment",
                    "https://research.example\\@evil.example/page",
                    "https://research.example/\npage"]:
            with self.subTest(url=url):
                payload = copy.deepcopy(self.payload)
                payload["request"]["urls"] = [url]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)
        self.payload["request"]["fixtures"][URL] = {
            "status_code": 302, "location": "https://evil.example/page"}
        with self.assertRaises(app.ValidationError):
            self.run_example()

    def test_empty_source_and_literal_matching(self):
        self.payload["request"]["fixtures"][URL]["text"] = ""
        self.assertEqual(self.run_example()["research"]["findings"], [])
        self.payload["request"]["fixtures"][URL]["text"] = "30 DAYS"
        self.assertEqual(self.run_example()["research"]["findings"], [])

    def test_schema_rejects_invalid_and_duplicate_inputs(self):
        cases = []
        for key, value in [("schema_version", True), ("synthetic", False),
                           ("research", {}), ("unknown", 1)]:
            payload = copy.deepcopy(self.payload)
            payload[key] = value
            cases.append(payload)
        duplicate = copy.deepcopy(self.payload)
        duplicate["request"]["requirements"].append(
            copy.deepcopy(duplicate["request"]["requirements"][0]))
        cases.append(duplicate)
        empty = copy.deepcopy(self.payload)
        empty["request"]["requirements"][0]["evidence_terms"] = []
        cases.append(empty)
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(payload)

    def test_provenance_and_review_tampering_rejected(self):
        output = self.run_example()
        output["research"]["findings"][0]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.validate(output, "ok")
        output = self.run_example()
        output["review"]["gap_count"] = 0
        with self.assertRaises(app.ValidationError):
            app.validate(output, "ok")
        output = self.run_example()
        output["research"]["findings"][0]["start"] = False
        with self.assertRaises(app.ValidationError):
            app.validate(output, "ok")

    def test_handoff_validated_before_review(self):
        with patch.object(app, "research", side_effect=[
                {"sources": [], "findings": []}, {"sources": ["different"]}]), \
                patch.object(app, "review") as review:
            with self.assertRaises(app.ValidationError):
                self.run_example()
            review.assert_not_called()

    def test_deterministic_without_input_mutation(self):
        before = copy.deepcopy(self.payload)
        self.assertEqual(self.run_example(), self.run_example())
        self.assertEqual(self.payload, before)

    def test_all_checks_pass_still_not_certification(self):
        self.payload["request"]["requirements"].pop()
        output = self.run_example()
        self.assertEqual(output["review"]["conclusion"], "textual_checks_passed")
        self.assertEqual(output["review"]["gap_count"], 0)
        self.assertIn("not certification", output["review"]["notice"])

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_error_and_usage(self):
        for arguments in [[], [str(ROOT / "does-not-exist.json")]]:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py")] + arguments,
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(result.stderr, "")

    def test_cli_invalid_json_schema_and_encoding(self):
        for raw in ["{", '{"x":1,"x":2}', '{"x":NaN}', "[]",
                    '{"schema_version":true}']:
            with self.subTest(raw=raw), \
                    patch("builtins.open", mock_open(read_data=raw)), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "error")
        with patch("builtins.open", side_effect=UnicodeError("bad encoding")), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
