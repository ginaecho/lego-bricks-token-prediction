import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        self.url = self.payload["research"]["urls"][0]

    def run_valid(self):
        return app.run_pipeline(self.payload)

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            self.run_valid()

    def replace_records(self, records):
        self.payload["research"]["fixtures"][self.url]["body"] = json.dumps(records)

    def test_integrated_example(self):
        result = self.run_valid()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["documents"]["rows"][0]["data"]["email"], "demo@synthetic.example")
        self.assertEqual([s["status"] for s in result["guided"]["steps"]],
                         ["completed", "ready", "blocked"])
        self.assertEqual(result["guided"]["progress"]["completed"], 1)

    def test_provenance_preserved_across_both_handoffs(self):
        result = self.run_valid()
        for finding, row, evidence in zip(result["research"]["findings"], result["documents"]["rows"],
                                           result["guided"]["steps"][0]["evidence"]):
            self.assertEqual(finding["id"], row["id"])
            self.assertEqual(row["id"], evidence["record_id"])
            self.assertEqual(finding["provenance"], row["provenance"])
            self.assertEqual(row["provenance"], evidence["provenance"])
        digest = hashlib.sha256(self.payload["research"]["fixtures"][self.url]["body"].encode()).hexdigest()
        self.assertEqual(result["research"]["sources"][0]["sha256"], digest)

    def test_deterministic_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.payload)
        self.assertEqual(self.run_valid(), self.run_valid())
        self.assertEqual(before, self.payload)

    def test_exact_host_allowlist_blocks_subdomain_and_credentials(self):
        for url in ["https://catalog.synthetic.example.evil.example/setup",
                    "http://catalog.synthetic.example/setup",
                    "https://user@catalog.synthetic.example/setup",
                    "https://catalog.synthetic.example:444/setup",
                    "https://catalog.synthetic.example/setup#secret",
                    "https://catalog.synthetic.example\\evil/setup"]:
            with self.subTest(url=url), self.assertRaises(app.ValidationError):
                app.validate_url(url, ["catalog.synthetic.example"])

    def test_missing_fixture(self):
        self.payload["research"]["fixtures"] = {}
        self.invalid()

    def test_duplicate_urls(self):
        self.payload["research"]["urls"] *= 2
        self.invalid()

    def test_bad_media_and_retrieved_shape(self):
        fixture = self.payload["research"]["fixtures"][self.url]
        fixture["content_type"] = "text/html"
        self.invalid()
        fixture["content_type"] = "application/json"
        for body in ["{}", "[]", "[1]", '[{"nested":{}}]', '[{"v":NaN}]', '[{"v":1,"v":2}]']:
            with self.subTest(body=body):
                fixture["body"] = body
                self.invalid()

    def test_plain_text_extraction(self):
        self.payload["research"]["fixtures"][self.url] = {
            "content_type": "text/plain",
            "body": "organization=Synthetic shop\ncontact=test@synthetic.example\nseats=2\napproved=true\n"}
        result = self.run_valid()
        self.assertEqual(result["documents"]["rows"][0]["data"]["seats"], 2)

    def test_malformed_plain_text(self):
        self.payload["research"]["fixtures"][self.url] = {
            "content_type": "text/plain", "body": "name=a\nname=b"}
        self.invalid()

    def test_general_mapping_defaults_and_optional_null(self):
        self.payload["documents"]["fields"] += [
            {"name": "optional", "source": "missing", "type": "string", "required": False},
            {"name": "budget", "source": "budget", "type": "number", "default": "12.5"}]
        data = self.run_valid()["documents"]["rows"][0]["data"]
        self.assertIsNone(data["optional"])
        self.assertEqual(data["budget"], 12.5)
        self.assertEqual(data["region"], "synthetic")

    def test_required_missing(self):
        self.payload["documents"]["fields"][0]["source"] = "absent"
        self.invalid()

    def test_invalid_integer_and_nonfinite_number(self):
        spec = self.payload["documents"]["fields"][2]
        for value in [True, 1.2, "1.2", "NaN"]:
            with self.subTest(value=value), self.assertRaises(app.ValidationError):
                app.convert(value, spec)
        with self.assertRaises(app.ValidationError):
            app.convert("inf", {"name": "value", "type": "number"})

    def test_document_checks_stop_pipeline(self):
        self.payload["documents"]["checks"].append({"field": "seats", "op": "max", "value": 1})
        with mock.patch.object(app, "guided") as guided:
            self.invalid()
            guided.assert_not_called()

    def test_unique_check_rejects_duplicates(self):
        records = json.loads(self.payload["research"]["fixtures"][self.url]["body"])
        records[1]["contact"] = records[0]["contact"]
        self.replace_records(records)
        self.invalid()

    def test_invalid_check_and_field_configuration(self):
        self.payload["documents"]["checks"].append({"field": "absent", "op": "nonempty"})
        self.invalid()
        self.payload["documents"]["checks"].pop()
        self.payload["documents"]["fields"][2]["transforms"] = ["strip"]
        self.invalid()

    def test_invalid_regex(self):
        self.payload["documents"]["checks"][1]["value"] = "["
        self.invalid()

    def test_onboarding_data_blocker_propagates(self):
        records = json.loads(self.payload["research"]["fixtures"][self.url]["body"])
        records[0]["approved"] = "false"
        self.replace_records(records)
        steps = self.run_valid()["guided"]["steps"]
        self.assertEqual(steps[1]["status"], "blocked")
        self.assertIn("data:approved/equals", steps[1]["blockers"])
        self.assertIn("prerequisite:allocate", steps[2]["blockers"])

    def test_progress_and_unordered_prerequisites(self):
        self.payload["guided"]["completed_steps"] = ["launch", "allocate", "verify-contact"]
        self.payload["guided"]["steps"].reverse()
        result = self.run_valid()["guided"]
        self.assertEqual(result["progress"], {"completed": 3, "total": 3, "fraction": 1})
        self.assertTrue(all(step["status"] == "completed" for step in result["steps"]))

    def test_cannot_complete_blocked_step(self):
        self.payload["guided"]["completed_steps"] = ["allocate"]
        self.invalid()

    def test_unknown_prerequisite_cycle_and_duplicate(self):
        original = copy.deepcopy(self.payload["guided"])
        self.payload["guided"]["steps"][0]["requires"] = ["missing"]
        self.invalid()
        self.payload["guided"] = copy.deepcopy(original)
        self.payload["guided"]["steps"][0]["requires"] = ["launch"]
        self.invalid()
        self.payload["guided"] = copy.deepcopy(original)
        self.payload["guided"]["steps"].append(copy.deepcopy(original["steps"][0]))
        self.invalid()

    def test_shared_schema_rejects_corrupted_handoff(self):
        result = self.run_valid()
        result["research"]["findings"][0]["provenance"]["sha256"] = "bad"
        with self.assertRaises(app.ValidationError):
            app.documents(self.payload["documents"], result["research"])
        result["documents"]["rows"][0]["data"]["extra"] = "invalid"
        with self.assertRaises(app.ValidationError):
            app.guided(self.payload["guided"], result["documents"])

    def test_schema_and_synthetic_marker(self):
        for key, value in [("schema_version", True), ("schema_version", 2), ("synthetic", False)]:
            with self.subTest(key=key, value=value):
                candidate = copy.deepcopy(self.payload)
                candidate[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(candidate)

    def test_cli_success_single_json(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                  str(ROOT / "example_input.json")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file_and_arguments(self):
        for args in [[], [str(ROOT / "not_present.json")], ["a", "b"]]:
            with self.subTest(args=args):
                process = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                         capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stderr, "")
                self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_json_input_without_extra_files(self):
        for body in ['{"schema_version":', '{"schema_version":1}', '{"a":1,"a":2}']:
            with self.subTest(body=body), mock.patch.object(Path, "read_text", return_value=body):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_invalid_encoding(self):
        with mock.patch.object(Path, "read_text",
                               side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(app.main(["synthetic-invalid.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
