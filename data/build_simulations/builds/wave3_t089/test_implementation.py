import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, synthesize


ROOT = Path(__file__).parent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args, stdin=None):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              input=stdin, capture_output=True, text=True, cwd=ROOT)

    def test_conflict_and_retrieval(self):
        result = synthesize(self.data)
        self.assertTrue(result["synthetic_fixture"])
        self.assertEqual(result["results"][0]["direction"], "mixed")
        self.assertEqual(result["results"][1]["direction"], "insufficient")
        self.assertTrue(result["results"][1]["evidence"])
        for finding in result["results"]:
            for evidence in finding["evidence"]:
                content = next(s["content"] for s in self.data["sources"]
                               if s["id"] == evidence["source_id"])
                start = evidence["start_offset"]
                self.assertEqual(content[start:start + len(evidence["quote"])], evidence["quote"])

    def test_corroboration(self):
        self.data["sources"][1]["claims"][0]["stance"] = "supports"
        result = synthesize(self.data)["results"][0]
        self.assertEqual(result["direction"], "supports")
        self.assertEqual(result["evidence_strength"], "corroborated")

    def test_empty_sources_and_no_overlap(self):
        self.data["sources"] = []
        self.assertEqual(synthesize(self.data)["results"][0]["evidence"], [])
        self.data["sources"] = [{"id": "s", "title": "SYNTHETIC", "content": "Zebras graze."}]
        self.assertEqual(synthesize(self.data)["results"][0]["counts"]["matched_passages"], 0)

    def test_quality_threshold(self):
        self.data["sources"][1]["reliability"] = 0.1
        result = synthesize(self.data)["results"][0]
        self.assertEqual(result["direction"], "insufficient")
        self.assertEqual(result["counts"]["opposing_origins"], 0)

    def test_duplicate_text_not_corroboration(self):
        source = copy.deepcopy(self.data["sources"][0])
        source["id"] = "duplicate"
        source["origin"] = "other"
        self.data["sources"] = [self.data["sources"][0], source]
        self.assertEqual(synthesize(self.data)["results"][0]["direction"], "insufficient")

    def test_same_origin_not_corroboration(self):
        self.data["sources"][1]["origin"] = self.data["sources"][0]["origin"]
        self.data["sources"][1]["claims"][0]["stance"] = "supports"
        self.assertEqual(synthesize(self.data)["results"][0]["counts"]["supporting_origins"], 1)

    def test_cap_does_not_hide_conflict_from_conclusion(self):
        self.data["max_evidence"] = 1
        result = synthesize(self.data)["results"][0]
        self.assertEqual(result["direction"], "mixed")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["counts"]["matched_passages"], 2)

    def test_transitive_origin_deduplication(self):
        duplicate = copy.deepcopy(self.data["sources"][0])
        duplicate.update(id="third", origin=self.data["sources"][1]["origin"])
        self.data["sources"][1]["claims"][0]["stance"] = "supports"
        self.data["sources"].append(duplicate)
        result = synthesize(self.data)["results"][0]
        self.assertEqual(result["counts"]["supporting_origins"], 1)
        self.assertEqual(result["direction"], "insufficient")

    def test_determinism_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(synthesize(self.data), synthesize(self.data))
        self.assertEqual(original, self.data)

    def test_validation(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(min_sources=True),
            lambda d: d.update(unknown="x"),
            lambda d: d.update(questions=[]),
            lambda d: d["sources"][0].update(reliability=float("nan")),
            lambda d: d["sources"][0].update(reliability=True),
            lambda d: d["sources"][0].update(published="2026-02-30"),
            lambda d: d["sources"][0]["claims"][0].update(quote="not in source"),
            lambda d: d["sources"][0]["claims"][0].update(question_id="missing"),
            lambda d: d["sources"][0]["claims"][0].update(stance=[]),
            lambda d: d["questions"].append(copy.deepcopy(d["questions"][0])),
            lambda d: d["sources"].append(copy.deepcopy(d["sources"][0])),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(ValidationError):
                    synthesize(data)

    def test_cli_success(self):
        completed = self.cli("example_input.json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(completed.stderr, "")

    def test_cli_errors(self):
        for args, stdin in [((), None), (("does-not-exist.json",), None),
                            (("-",), "{"), (("-",), '{"schema_version":1,"schema_version":1}'),
                            (("-",), '{"x":NaN}'), (("-",), "[]")]:
            with self.subTest(args=args, stdin=stdin):
                completed = self.cli(*args, stdin=stdin)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(json.loads(completed.stdout)["status"], "error")
                self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
