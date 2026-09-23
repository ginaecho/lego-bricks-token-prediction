import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import contextlib
import io

import implementation as impl


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_integrated_success_and_determinism(self):
        result = impl.run_pipeline(self.data)
        self.assertEqual(result, impl.run_pipeline(copy.deepcopy(self.data)))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["web"]["status"], "complete")

    def test_faq_exact_grounding(self):
        result = impl.run_pipeline(self.data)["faq"]
        self.assertEqual(result["answer"], self.data["knowledge_base"][0]["text"])
        self.assertEqual(result["citations"][0]["source_id"], "faq-calibration")

    def test_abstention_propagates(self):
        self.data["question"] = "Quantum banana insurance?"
        result = impl.run_pipeline(self.data)
        self.assertEqual(result["faq"]["status"], "abstained")
        self.assertIsNone(result["faq"]["answer"])
        self.assertEqual(result["journey"]["reason"], "faq_abstained")
        self.assertEqual(result["web"]["status"], "blocked")
        self.assertEqual(result["web"]["findings"], [])

    def test_stopword_only_question_abstains(self):
        self.data["question"] = "How do I?"
        self.assertEqual(impl.run_pipeline(self.data)["faq"]["status"], "abstained")

    def test_two_steps_respect_prerequisites(self):
        steps = impl.run_pipeline(self.data)["journey"]["steps"]
        self.assertEqual([s["action_id"] for s in steps], ["align", "focus"])
        self.assertEqual(steps[1]["prerequisites"], ["align"])

    def test_completed_actions_not_recommended(self):
        self.data["completed_actions"] = ["align"]
        result = impl.run_pipeline(self.data)
        self.assertEqual(result["journey"]["status"], "blocked")
        self.assertEqual(result["journey"]["steps"], [])

    def test_completed_prerequisite_unlocks_two_steps(self):
        self.data["completed_actions"] = ["align"]
        extra = copy.deepcopy(self.data["actions"][1])
        extra.update(id="observe", title="Synthetic observation", prerequisites=["focus"])
        self.data["actions"].append(extra)
        steps = impl.run_pipeline(self.data)["journey"]["steps"]
        self.assertEqual([s["action_id"] for s in steps], ["focus", "observe"])

    def test_cycle_rejected(self):
        self.data["actions"][0]["prerequisites"] = ["focus"]
        with self.assertRaisesRegex(impl.ValidationError, "cyclic"):
            impl.run_pipeline(self.data)

    def test_unknown_prerequisite_rejected(self):
        self.data["actions"][1]["prerequisites"] = ["missing"]
        with self.assertRaisesRegex(impl.ValidationError, "unknown prerequisite"):
            impl.run_pipeline(self.data)

    def test_incompatible_topics_block_journey(self):
        self.data["actions"][1]["required_topics"] = ["billing"]
        self.assertEqual(impl.run_pipeline(self.data)["journey"]["status"], "blocked")

    def test_web_provenance_cross_stage(self):
        result = impl.run_pipeline(self.data)
        for finding, step in zip(result["web"]["findings"], result["journey"]["steps"]):
            self.assertEqual(finding["action_id"], step["action_id"])
            self.assertEqual(finding["research_terms"], step["research_terms"])
            self.assertEqual(finding["faq_source_ids"], result["journey"]["faq_source_ids"])
            page = next(p for p in self.data["web"]["pages"] if p["id"] == finding["source_id"])
            self.assertEqual(finding["url"], page["url"])
            self.assertEqual(finding["quote"], page["text"])
            self.assertEqual(finding["sha256"], hashlib.sha256(page["text"].encode()).hexdigest())

    def test_allowlist_rejects_url_tricks(self):
        for url in ("http://guides.example/a", "https://guides.example.evil.example/a",
                    "https://sub.guides.example/a", "https://user@guides.example/a",
                    "https://guides.example:444/a", "https://guides.example/a#fragment",
                    "https://guides.example\\@evil.example/a", "https://guides.example/\na",
                    "file:///guides.example/a", "https://[broken"):
            with self.subTest(url=url):
                data = copy.deepcopy(self.data)
                data["web"]["pages"][0]["url"] = url
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(data)

    def test_explicit_default_https_port_allowed(self):
        self.data["web"]["pages"][0]["url"] = "https://guides.example:443/mount"
        self.assertEqual(impl.run_pipeline(self.data)["web"]["status"], "complete")

    def test_web_no_evidence_abstains(self):
        self.data["web"]["pages"] = []
        result = impl.run_pipeline(self.data)["web"]
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["unfulfilled_action_ids"], ["align", "focus"])

    def test_web_partial_evidence(self):
        self.data["web"]["pages"] = self.data["web"]["pages"][:1]
        result = impl.run_pipeline(self.data)["web"]
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unfulfilled_action_ids"], ["focus"])

    def test_tampered_faq_rejected_before_journey(self):
        faq = impl.answer_faq(self.data)
        faq["answer"] = "An invented answer"
        with self.assertRaises(impl.ValidationError):
            impl.recommend_journey(self.data, faq)

    def test_tampered_journey_rejected_before_web(self):
        faq = impl.answer_faq(self.data)
        journey = impl.recommend_journey(self.data, faq)
        journey["steps"].reverse()
        with self.assertRaises(impl.ValidationError):
            impl.research_web(self.data, faq, journey)

    def test_tampered_web_provenance_rejected(self):
        result = impl.run_pipeline(self.data)
        result["web"]["findings"][0]["url"] = "https://guides.example/invented"
        with self.assertRaises(impl.ValidationError):
            impl.Validator.output(result, self.data)

    def test_invalid_input_shapes(self):
        for value in (None, [], {}, {"schema_version": 1}):
            with self.subTest(value=value), self.assertRaises(impl.ValidationError):
                impl.run_pipeline(value)
        for key, value in (("schema_version", True), ("question", ""),
                           ("actions", {}), ("completed_actions", ["missing"]),
                           ("fixture_label", "real")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(data)

    def test_duplicates_and_unknown_fields_rejected(self):
        for key in ("knowledge_base", "actions"):
            data = copy.deepcopy(self.data)
            data[key].append(copy.deepcopy(data[key][0]))
            with self.assertRaises(impl.ValidationError):
                impl.run_pipeline(data)
        self.data["unexpected"] = True
        with self.assertRaises(impl.ValidationError):
            impl.run_pipeline(self.data)

    def test_empty_knowledge_base(self):
        self.data["knowledge_base"] = []
        self.data["actions"] = []
        self.assertEqual(impl.run_pipeline(self.data)["faq"]["status"], "abstained")

    def test_tie_breaking_is_source_id_order(self):
        doc = copy.deepcopy(self.data["knowledge_base"][0])
        doc["id"] = "aaa"
        self.data["knowledge_base"].append(doc)
        result = impl.run_pipeline(self.data)
        self.assertEqual([c["source_id"] for c in result["faq"]["citations"]], ["aaa", "faq-calibration"])

    def test_cli_success_one_json(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")

    def test_cli_missing_file_and_usage(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_json_and_encoding_errors(self):
        # Mock file content instead of creating additional delivered files.
        for content in ('{', '{"x":1,"x":2}', 'NaN', '[]', '{"schema_version":true}'):
            with self.subTest(content=content), patch.object(Path, "read_text", return_value=content):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = impl.main([str(ROOT / "example_input.json")])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")
        with patch.object(Path, "read_text", side_effect=UnicodeError("bad encoding")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(impl.main([str(ROOT / "example_input.json")]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
