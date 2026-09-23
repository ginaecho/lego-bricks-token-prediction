"""All documents and questions here are synthetic fixtures."""

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


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_grounded_normal_answer(self):
        result = app.run(self.data)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], self.data["knowledge_base"][0]["content"])
        self.assertEqual(result["answer"], result["citations"][0]["quote"])
        self.assertEqual(result["citations"][0]["id"], "synthetic-returns")

    def test_unrelated_abstains(self):
        self.data["question"] = "Do you provide cryptocurrency financing?"
        result = app.run(self.data)
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])

    def test_empty_kb(self):
        self.data["knowledge_base"] = []
        self.assertEqual(app.run(self.data)["reason"], "insufficient_evidence")

    def test_no_searchable_terms(self):
        self.data["question"] = "What is it???"
        self.assertEqual(app.run(self.data)["reason"], "no_searchable_terms")

    def test_ambiguous_evidence(self):
        other = dict(self.data["knowledge_base"][0], id="synthetic-conflict",
                     content="SYNTHETIC FIXTURE: The returns policy forbids all returns.")
        self.data["knowledge_base"].append(other)
        self.assertEqual(app.run(self.data)["reason"], "ambiguous_evidence")

    def test_identical_ties_are_stable(self):
        self.data["knowledge_base"].append(dict(self.data["knowledge_base"][0], id="aaa"))
        first = app.run(self.data)
        self.data["knowledge_base"].reverse()
        self.assertEqual(first, app.run(self.data))
        self.assertEqual(first["citations"][0]["id"], "aaa")

    def test_title_alone_not_evidence(self):
        self.data["knowledge_base"][0]["content"] = "SYNTHETIC FIXTURE: Welcome!"
        self.assertEqual(app.run(self.data)["status"], "abstained")

    def test_exact_threshold(self):
        self.data["question"] = "returns policy warranty"
        self.data["min_coverage"] = 2 / 3
        self.assertEqual(app.run(self.data)["status"], "answered")
        self.data["min_coverage"] = 0.667
        self.assertEqual(app.run(self.data)["status"], "abstained")

    def test_invalid_input_matrix(self):
        cases = [None, [], {}, dict(self.data, extra=1)]
        for field, values in (
            ("schema_version", [True, 2, "1"]),
            ("question", ["", " ", 12, "x" * 2001]),
            ("knowledge_base", [{}, [None], [{"id": "x"}]]),
            ("min_coverage", [True, 0, -1, 2, 10 ** 400, "0.6", float("nan"), float("inf")]),
        ):
            cases.extend(dict(self.data, **{field: value}) for value in values)
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(app.run(case)["status"], "error")

    def test_invalid_documents(self):
        for field, value in (("id", " "), ("title", ""), ("content", None), ("id", " x")):
            data = copy.deepcopy(self.data)
            data["knowledge_base"][0][field] = value
            self.assertEqual(app.run(data)["status"], "error")
        self.data["knowledge_base"].append(self.data["knowledge_base"][0])
        self.assertEqual(app.run(self.data)["status"], "error")

    def test_deterministic_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(self.data, original)

    def cli(self, *args):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.stderr, "")
        return process.returncode, json.loads(process.stdout)

    def test_cli_example(self):
        code, result = self.cli("example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "answered")

    def test_cli_missing_file_and_usage(self):
        for args in [(), ("nonexistent-synthetic-fixture.json",), ("a", "b")]:
            with self.subTest(args=args):
                code, result = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for text in ["{", "[]", '{"schema_version":1,"schema_version":1}', '{"x":NaN}']:
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = app.main(["synthetic-mocked.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_encoding_error(self):
        with patch.object(Path, "read_text", side_effect=UnicodeError("synthetic invalid encoding")):
            with redirect_stdout(io.StringIO()) as output:
                code = app.main(["synthetic-mocked.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
