import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_multi_document_disagreement_and_gaps(self):
        result = impl.run(self.payload)
        self.assertEqual(result["status"], "ok")
        questions = result["question_results"]
        self.assertEqual([q["finding"] for q in questions],
                         ["disputed", "uncertain_only", "no_evidence"])
        self.assertTrue(questions[0]["disagreements"][0]["cross_document"])
        self.assertEqual(questions[0]["disagreements"][0]["supporting_evidence_ids"], ["e1"])
        self.assertEqual(questions[0]["distinct_document_count"], 2)
        self.assertTrue(all(q["unresolved_questions"] for q in questions))

    def test_empty_documents(self):
        self.payload["documents"] = []
        result = impl.run(self.payload)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(all(q["finding"] == "no_evidence" for q in result["question_results"]))

    def test_support_and_contradiction_only(self):
        for stance, expected in [("supports", "support_only"), ("contradicts", "contradiction_only")]:
            with self.subTest(stance=stance):
                p = copy.deepcopy(self.payload)
                p["documents"] = p["documents"][:1]
                p["documents"][0]["evidence"][0]["stance"] = stance
                q = impl.run(p)["question_results"][0]
                self.assertEqual(q["finding"], expected)
                self.assertEqual(len(q["unresolved_questions"]), 1)

    def test_internal_disagreement(self):
        doc = self.payload["documents"][0]
        item = dict(doc["evidence"][0], id="e4", stance="contradicts")
        doc["evidence"].append(item)
        self.payload["documents"] = [doc]
        conflict = impl.run(self.payload)["question_results"][0]["disagreements"][0]
        self.assertEqual(conflict["internally_conflicted_document_ids"], ["pilot"])
        self.assertFalse(conflict["cross_document"])

    def test_duplicate_excerpts_do_not_inflate_document_count(self):
        self.payload["documents"][0]["evidence"].append(
            dict(self.payload["documents"][0]["evidence"][0], id="e4"))
        q = impl.run(self.payload)["question_results"][0]
        self.assertEqual(q["document_ids_by_stance"]["supports"], ["pilot"])
        self.assertEqual(q["distinct_document_count"], 2)

    def test_determinism_and_no_mutation(self):
        original = copy.deepcopy(self.payload)
        first = impl.run(self.payload)
        self.assertEqual(original, self.payload)
        self.payload["documents"].reverse()
        for doc in self.payload["documents"]:
            doc["evidence"].reverse()
        self.assertEqual(first, impl.run(self.payload))

    def test_schema_errors(self):
        mutations = [
            lambda p: p.update(schema_version=True),
            lambda p: p.update(synthetic=False),
            lambda p: p.update(questions=[]),
            lambda p: p.update(extra=1),
            lambda p: p["questions"].append(dict(p["questions"][0])),
            lambda p: p["documents"].append(copy.deepcopy(p["documents"][0])),
            lambda p: p["documents"][0]["evidence"].append(dict(p["documents"][0]["evidence"][0])),
            lambda p: p["documents"][0]["evidence"][0].update(question_id="unknown"),
            lambda p: p["documents"][0]["evidence"][0].update(stance="likely"),
            lambda p: p["documents"][0]["evidence"][0].update(excerpt="Invented quotation."),
            lambda p: p["documents"][0]["evidence"][0].update(question_id=[]),
            lambda p: p["questions"][0].update(text=" "),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                p = copy.deepcopy(self.payload)
                mutation(p)
                self.assertEqual(impl.run(p)["status"], "error")
        for payload in [None, [], "text", 1]:
            self.assertEqual(impl.run(payload)["status"], "error")

    def test_injected_fixture_and_isolation(self):
        original = copy.deepcopy(self.payload)

        def fixture(question):
            ids = [e["id"] for e in question["evidence"]]
            question["evidence"].clear()
            return {"summary": "SYNTHETIC fixture summary; see supplied evidence.", "evidence_ids": ids}

        result = impl.run(self.payload, fixture)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["question_results"][0]["evidence"]), 2)
        self.assertEqual(original, self.payload)
        self.assertEqual(result["question_results"][0]["optional_summary"]["verification"], "structure_only")

    def test_invalid_injected_outputs(self):
        for response in [None, {"summary": "x", "evidence_ids": ["invented"]},
                         {"summary": "", "evidence_ids": []},
                         {"summary": "x", "evidence_ids": ["e1", "e1", "e3"]}]:
            with self.subTest(response=response):
                self.assertEqual(impl.run(self.payload, lambda q: response)["status"], "error")

        def broken(question):
            raise RuntimeError("Synthetic callable failure")

        self.assertEqual(impl.run(self.payload, broken)["status"], "error")
        self.assertEqual(impl.run(self.payload, 3)["status"], "error")

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, text=True, capture_output=True, check=False)

    def test_cli_success(self):
        proc = self.cli("example_input.json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), impl.run(self.payload))
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_usage_and_missing_file(self):
        for args in [(), ("does-not-exist.json",), ("example_input.json", "extra")]:
            with self.subTest(args=args):
                proc = self.cli(*args)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_schema_without_writing_files(self):
        for content in ["{", '{"a":1,"a":2}', '{"a":NaN}', "[]", "{}"]:
            with self.subTest(content=content):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=content)), contextlib.redirect_stdout(output):
                    code = impl.main(["synthetic-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_decode_error(self):
        output = io.StringIO()
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        with patch("builtins.open", side_effect=error), contextlib.redirect_stdout(output):
            code = impl.main(["synthetic-fixture.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["error"]["type"], "UnicodeDecodeError")


if __name__ == "__main__":
    unittest.main()
