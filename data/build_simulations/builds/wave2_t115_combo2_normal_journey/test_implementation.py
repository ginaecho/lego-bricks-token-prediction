import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, journey_stage, research_stage, run_pipeline, validate


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *arguments):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *arguments],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_exact_citations(self):
        findings = research_stage(self.data)["findings"]
        self.assertEqual(len(findings), 2)
        for finding in findings:
            citation = finding["citation"]
            source = next(s for s in self.data["sources"] if s["id"] == citation["source_id"])
            passage = next(p for p in source["passages"] if p["id"] == citation["passage_id"])
            self.assertEqual(finding["quote"], passage["text"][citation["start"]:citation["end"]])
            self.assertEqual(finding["score"], 2)

    def test_two_step_prerequisites(self):
        journey = run_pipeline(self.data)["journey"]
        self.assertEqual(journey["status"], "ready")
        self.assertEqual([s["action_id"] for s in journey["steps"]], ["sort", "compost"])
        self.assertEqual(journey["steps"][0]["skills_before"], [])
        self.assertEqual(journey["steps"][1]["skills_before"], ["sorting"])
        self.assertEqual([s["action_id"] for s in journey["next_actions"]], ["sort"])
        self.assertEqual(journey["blocked_now"][0]["missing_prerequisites"], ["sorting"])

    def test_evidence_handoff(self):
        result = run_pipeline(self.data)
        evidence = {f["id"]: f for f in result["research"]["findings"]}
        for step in result["journey"]["steps"]:
            self.assertTrue(step["finding_ids"])
            for finding_id in step["finding_ids"]:
                self.assertIn(step["action_id"], evidence[finding_id]["action_ids"])
        self.assertNotIn("paint", [s["action_id"] for s in result["journey"]["next_actions"]])

    def test_retrieval_limit_propagates(self):
        self.data["limit"] = 1
        result = run_pipeline(self.data)
        self.assertEqual(len(result["research"]["findings"]), 1)
        self.assertEqual(result["journey"]["status"], "no_valid_two_step")
        self.assertEqual(result["journey"]["steps"], [])

    def test_no_matches(self):
        self.data["query"] = "astronomy"
        result = run_pipeline(self.data)
        self.assertEqual(result["research"]["findings"], [])
        self.assertEqual(result["journey"]["status"], "insufficient_evidence")

    def test_empty_catalog(self):
        self.data["sources"] = []
        self.data["actions"] = []
        self.assertEqual(run_pipeline(self.data)["journey"]["steps"], [])

    def test_cyclic_prerequisites_blocked(self):
        self.data["actions"][0]["requires"] = ["composting"]
        journey = run_pipeline(self.data)["journey"]
        self.assertEqual(journey["status"], "no_valid_two_step")
        self.assertEqual(journey["next_actions"], [])

    def test_profile_and_interests(self):
        self.data["profile"]["skills"] = ["sorting"]
        self.data["actions"][0]["tags"] = ["other"]
        journey = run_pipeline(self.data)["journey"]
        self.assertEqual(journey["next_actions"][0]["action_id"], "compost")
        self.assertEqual(journey["next_actions"][0]["score"], 3)

    def test_tampered_handoff_rejected(self):
        for mutation in ("quote", "action_ids", "citation"):
            research = research_stage(self.data)
            if mutation == "quote":
                research["findings"][0]["quote"] = "invented"
            elif mutation == "action_ids":
                research["findings"][0]["action_ids"] = ["paint"]
            else:
                research["findings"][0]["citation"]["end"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                journey_stage(self.data, research)

    def test_tampered_journey_rejected(self):
        result = run_pipeline(self.data)
        result["journey"]["steps"].reverse()
        with self.assertRaises(ValidationError):
            validate(result, "output", self.data)

    def test_invalid_input(self):
        for key, value in (("limit", True), ("limit", 0), ("query", " !!! "),
                           ("synthetic", False), ("sources", {}), ("profile", None),
                           ("schema_version", True)):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                run_pipeline(data)

    def test_duplicate_identifiers_and_unknown_references(self):
        for duplicate in (True, False):
            data = copy.deepcopy(self.data)
            if duplicate:
                data["actions"].append(copy.deepcopy(data["actions"][0]))
            else:
                data["sources"][0]["passages"][0]["action_ids"] = ["missing"]
            with self.subTest(duplicate=duplicate), self.assertRaises(ValidationError):
                run_pipeline(data)

    def test_unicode_offsets_and_casefold(self):
        self.data["query"] = "CAFÉ"
        self.data["sources"][0]["passages"][0]["text"] = "A café grows 🌱."
        finding = research_stage(self.data)["findings"][0]
        self.assertEqual(finding["quote"], "A café grows 🌱.")
        self.assertEqual(finding["citation"]["end"], len("A café grows 🌱."))

    def test_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(run_pipeline(self.data), run_pipeline(self.data))
        self.assertEqual(self.data, before)

    def test_cli_success(self):
        process = self.cli("example_input.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["journey"]["status"], "ready")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_usage_and_missing_file(self):
        for arguments in ((), ("absent-input.json",), ("example_input.json", "extra")):
            process = self.cli(*arguments)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "error")
            self.assertEqual(process.stderr, "")

    def test_cli_bad_json_and_schema(self):
        path = ROOT / "_invalid_fixture.json"
        try:
            for content in (b"{", b"{}", b'{"a":1,"a":2}', b'{"x":NaN}', b"\xff"):
                path.write_bytes(content)
                process = self.cli(str(path))
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
