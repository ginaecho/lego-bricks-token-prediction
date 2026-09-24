import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_conflict_and_attribution(self):
        result = app.research(self.data)
        self.assertEqual(result["status"], "ok")
        brief = result["briefs"][0]
        self.assertEqual(brief["assessment"], "mixed")
        self.assertEqual(brief["source_count"], 3)
        self.assertEqual(brief["independent_origin_count"], 2)
        self.assertEqual(brief["directional_weights"]["support"], 0.8)
        sources = {s["id"]: s["text"] for s in self.data["sources"]}
        for quote in brief["evidence"]:
            self.assertIn(quote["quote"], sources[quote["source_id"]])

    def test_unannotated_is_informational(self):
        self.assertEqual(app.research(self.data)["briefs"][1]["assessment"], "informational")

    def test_empty_sources(self):
        self.data["sources"] = []
        brief = app.research(self.data)["briefs"][0]
        self.assertEqual(brief["assessment"], "insufficient_evidence")
        self.assertEqual(brief["evidence"], [])

    def test_unrelated_sources(self):
        for source in self.data["sources"]:
            source["text"] = "Synthetic geology observations."
        self.assertEqual(app.research(self.data)["briefs"][0]["source_count"], 0)

    def test_zero_reliability(self):
        for source in self.data["sources"]:
            source["reliability"] = 0
        brief = app.research(self.data)["briefs"][0]
        self.assertTrue(brief["evidence"])
        self.assertEqual(brief["assessment"], "insufficient_evidence")

    def test_support_and_oppose(self):
        for position in ("support", "oppose"):
            with self.subTest(position=position):
                for source in self.data["sources"]:
                    source["positions"] = {"q1": position}
                self.assertEqual(app.research(self.data)["briefs"][0]["assessment"], position)

    def test_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.research(self.data), app.research(self.data))
        self.assertEqual(self.data, before)

    def test_invalid_quality(self):
        for quality in (True, "0.5", -1, 2, 10 ** 1000, float("nan"), float("inf"), None):
            with self.subTest(quality=quality):
                self.data["sources"][0]["reliability"] = quality
                with self.assertRaises(app.ValidationError):
                    app.research(self.data)

    def test_invalid_schema(self):
        mutations = [
            lambda d: d.update(schema_version="2"),
            lambda d: d.update(questions=[]),
            lambda d: d.update(unexpected=True),
            lambda d: d["questions"].append(d["questions"][0]),
            lambda d: d["sources"].append(d["sources"][0]),
            lambda d: d["sources"][0].update(positions={"unknown": "support"}),
            lambda d: d["sources"][0].update(positions={"q1": []}),
            lambda d: d["questions"][0].update(keywords=[]),
            lambda d: d["questions"][0].update(keywords=["the"]),
            lambda d: d["sources"][0].update(text=" "),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                mutation(data)
                with self.assertRaises(app.ValidationError):
                    app.research(data)

    def test_passage_cap_and_case_matching(self):
        self.data["sources"] = [self.data["sources"][0]]
        self.data["sources"][0]["text"] = "LIBRARY evening HOURS. " * 8
        brief = app.research(self.data)["briefs"][0]
        self.assertEqual(len(brief["evidence"]), 3)
        self.assertEqual(brief["directional_weights"]["support"], 0.8)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_usage_and_file_error(self):
        for args in ([], [str(ROOT / "does-not-exist.json")],
                     [str(ROOT / "example_input.json"), "extra"]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_without_scratch_files(self):
        for raw in (b"{bad", b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff",
                    b"[]", b"x" * (app.MAX_BYTES + 1)):
            with self.subTest(raw=raw[:40]):
                stream = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                    with contextlib.redirect_stdout(stream):
                        code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
