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
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_and_traceability(self):
        result = app.review(self.payload)
        self.assertEqual(result["summary"], {
            "requirements": 3, "lexically_supported": 1, "gaps": 2,
        })
        self.assertEqual(result["requirement_results"][1]["evidence_traces"][1], {
            "evidence_id": "SYN-E-3", "source": "synthetic-training-plan",
            "locator": "row 2", "qualifies": False, "missing_terms": ["completed"],
        })
        self.assertEqual(result["gaps"][1]["reason"], "missing_evidence")
        self.assertIn("not certification", result["disclaimer"])

    def test_all_supported(self):
        self.payload["requirements"] = self.payload["requirements"][:1]
        self.payload["evidence"] = self.payload["evidence"][:1]
        self.assertEqual(app.review(self.payload)["gaps"], [])

    def test_empty_evidence(self):
        self.payload["evidence"] = []
        result = app.review(self.payload)
        self.assertEqual(result["summary"]["gaps"], 3)
        self.assertEqual(result["gaps"][1]["additional_qualifying_evidence_needed"], 2)

    def test_deterministic_and_no_mutation(self):
        original = copy.deepcopy(self.payload)
        self.assertEqual(app.review(self.payload), app.review(self.payload))
        self.assertEqual(self.payload, original)

    def test_terms_must_cooccur_per_entry(self):
        self.payload["evidence"][0]["text"] = "owner"
        extra = copy.deepcopy(self.payload["evidence"][0])
        extra.update(id="extra", text="annual")
        self.payload["evidence"].append(extra)
        self.assertEqual(app.review(self.payload)["requirement_results"][0]["status"], "gap")

    def test_invalid_types_and_structure(self):
        for key, value in [("schema_version", 1), ("requirements", []),
                           ("document", None), ("evidence", "bad")]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.payload)
                bad[key] = value
                with self.assertRaises(app.ValidationError):
                    app.review(bad)
        self.payload["requirements"][0]["min_evidence"] = True
        with self.assertRaises(app.ValidationError):
            app.review(self.payload)

    def test_duplicate_ids_and_unknown_reference(self):
        for collection in ("requirements", "evidence"):
            with self.subTest(collection=collection):
                bad = copy.deepcopy(self.payload)
                bad[collection].append(copy.deepcopy(bad[collection][0]))
                with self.assertRaises(app.ValidationError):
                    app.review(bad)
        self.payload["evidence"][0]["requirement_id"] = "unknown"
        with self.assertRaises(app.ValidationError):
            app.review(self.payload)

    def test_duplicate_terms_and_unknown_fields(self):
        self.payload["requirements"][0]["required_terms"] = ["Owner", " owner "]
        with self.assertRaises(app.ValidationError):
            app.review(self.payload)
        self.payload["requirements"][0]["required_terms"] = ["owner"]
        self.payload["unexpected"] = True
        with self.assertRaises(app.ValidationError):
            app.review(self.payload)

    def test_unicode_casefold(self):
        self.payload["requirements"][0]["required_terms"] = ["STRASSE"]
        self.payload["evidence"][0]["text"] = "Straße"
        self.assertEqual(app.review(self.payload)["requirement_results"][0]["status"],
                         "lexically_supported")

    def test_cli_success(self):
        run = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"),
             str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stderr, "")
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(len(run.stdout.splitlines()), 1)

    def test_cli_file_and_usage_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["a", "b"]):
            with self.subTest(args=args):
                run = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(run.returncode, 2)
                self.assertEqual(json.loads(run.stdout)["status"], "error")
                self.assertEqual(run.stderr, "")

    def test_cli_invalid_json_and_validation(self):
        for raw in (b"{", b"[]", b'{"a":1,"a":2}', b'{"x":NaN}', b"\xff",
                    b"x" * 5_000_001):
            with self.subTest(raw=raw[:30]):
                output = io.StringIO()
                with patch.object(Path, "read_bytes", return_value=raw), redirect_stdout(output):
                    code = app.main(["synthetic.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
