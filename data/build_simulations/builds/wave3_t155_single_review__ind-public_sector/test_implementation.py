import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import implementation as app

ROOT = Path(__file__).resolve().parent


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_and_traceable_gap(self):
        result = app.review(self.data)
        self.assertEqual(result["summary"], {"documents_reviewed": 3, "gaps_found": 1})
        check = result["reviews"][1]["checks"][0]
        self.assertEqual(check["requirement_id"], "R-002")
        self.assertEqual(check["missing_fields"], ["income_statement"])
        self.assertEqual(check["policy_reference"]["requirement_path"], "/policy/requirements/1")

    def test_complete_application(self):
        doc = self.data["documents"][1]
        doc["form_fields"]["income_statement"] = "Synthetic statement attached."
        doc["evidence"] = [{"id": "E-003", "requirement_id": "R-002",
                            "text": "INCOME   STATEMENT received"}]
        self.assertEqual(app.review(self.data)["summary"]["gaps_found"], 0)

    def test_no_evidence_is_gap(self):
        self.data["documents"][0]["evidence"] = []
        self.assertEqual(app.review(self.data)["reviews"][0]["checks"][0]["status"], "gap")

    def test_partial_evidence_does_not_pass(self):
        self.data["documents"][0]["evidence"][0]["text"] = "Request waiting."
        self.assertEqual(app.review(self.data)["summary"]["gaps_found"], 2)

    def test_no_pii_or_source_text_output(self):
        raw = json.dumps(app.review(self.data))
        for doc in self.data["documents"]:
            self.assertNotIn(doc["case_number"], raw)
            if doc["citizen"]:
                for value in doc["citizen"].values():
                    self.assertNotIn(value, raw)
            for item in doc["evidence"]:
                self.assertNotIn(item["text"], raw)
        self.assertNotIn(self.data["policy"]["text"], raw)

    def test_nonsynthetic_rejected(self):
        self.data["synthetic_fixture"] = False
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_unknown_fields_rejected(self):
        self.data["documents"][0]["form_fields"]["social_security_number"] = "fake"
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_duplicate_and_unknown_identifiers(self):
        for mutate in (
            lambda d: d["documents"][1].update(id="D-001"),
            lambda d: d["documents"][0]["evidence"][0].update(requirement_id="R-999"),
            lambda d: d["documents"][0].update(id="private@example.invalid"),
        ):
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.subTest(data=data["documents"][0]["id"]):
                with self.assertRaises(app.ValidationError):
                    app.review(data)

    def test_policy_grounding_required(self):
        self.data["policy"]["requirements"][0]["source_phrase"] = "Not in policy"
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_uncovered_document_not_supported(self):
        self.data["policy"]["requirements"] = self.data["policy"]["requirements"][1:]
        self.data["documents"][0]["evidence"] = []
        self.assertEqual(app.review(self.data)["reviews"][0]["status"], "not_assessed")

    def test_bad_types_and_empty_inputs(self):
        for value in (None, [], {}, {"schema_version": True}):
            with self.subTest(value=value):
                with self.assertRaises(app.ValidationError):
                    app.review(value)
        self.data["documents"] = []
        with self.assertRaises(app.ValidationError):
            app.review(self.data)

    def test_deterministic_and_input_unchanged(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.review(self.data), app.review(self.data))
        self.assertEqual(original, self.data)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["a", "b"]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_malformed_duplicate_nonfinite_and_oversized_json(self):
        for payload in (b'{"x":', b'{"x":1,"x":2}', b'{"x":NaN}',
                        b"\xff", b" " * (app.MAX_BYTES + 1)):
            with patch.object(Path, "open", return_value=io.BytesIO(payload)):
                out = io.StringIO()
                with redirect_stdout(out):
                    status = app.main(["synthetic.json"])
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")

    def test_plain_language_and_no_certification(self):
        result = app.review(self.data)
        self.assertIn("not a legal or compliance certification", result["limitations"][0])
        self.assertEqual(result["reviews"][0]["checks"][0]["explanation"],
                         "The form fields are present. One evidence item has all required phrases.")


if __name__ == "__main__":
    unittest.main()
