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


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_to(self, stage):
        previous = None
        for name in app.STAGES:
            previous = app.advance(self.config, name, previous)
            if name == stage:
                return previous

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run_pipeline(self.config)

    def test_complete_pipeline(self):
        result = app.run_pipeline(self.config)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["data"]), list(app.STAGES))
        self.assertEqual([step["action_id"] for step in result["data"]["journey"]["steps"]],
                         ["a-setup", "b-export"])

    def test_review_evidence_and_gaps(self):
        review = self.run_to("review")["data"]["review"]
        self.assertEqual(review["gap_ids"], ["r-export", "r-retention"])
        evidence = review["checks"][0]["evidence"][0]
        excerpt = evidence["excerpt"]
        self.assertEqual(excerpt["text"], self.config["documents"][0]["text"][excerpt["start"]:excerpt["end"]])
        self.assertIn("not certification", review["notice"])

    def test_empty_documents_all_gaps(self):
        self.config["documents"] = []
        review = self.run_to("review")["data"]["review"]
        self.assertEqual(len(review["gap_ids"]), 3)

    def test_review_to_web_scope_propagates(self):
        self.config["documents"][0]["text"] += " The export instructions are here."
        result = app.run_pipeline(self.config)["data"]
        self.assertEqual(result["web"]["requested_requirement_ids"], ["r-retention"])
        self.assertEqual([f["requirement_id"] for f in result["web"]["findings"]], ["r-retention"])
        self.assertEqual([t["requirement_id"] for t in result["feedback"]["themes"]], ["r-retention"])
        self.assertEqual(result["journey"]["steps"][0]["action_id"], "c-retention")

    def test_web_preserves_provenance(self):
        finding = self.run_to("web")["data"]["web"]["findings"][0]
        source = self.config["sources"][0]
        for key in ("url", "title", "retrieved_at"):
            self.assertEqual(finding[key], source[key])
        excerpt = finding["excerpt"]
        self.assertEqual(excerpt["text"], source["body"][excerpt["start"]:excerpt["end"]])
        self.assertEqual(finding["source_id"], source["id"])

    def test_missing_snapshot_stays_unresolved(self):
        self.config["sources"] = []
        result = app.run_pipeline(self.config)["data"]
        self.assertEqual(result["web"]["unresolved_gap_ids"], ["r-export", "r-retention"])
        self.assertEqual(result["feedback"]["themes"][0]["research_status"], "unresolved")
        self.assertEqual(result["journey"]["steps"][0]["finding_ids"], [])

    def test_nonmatching_snapshot_no_finding(self):
        self.config["sources"][0]["body"] = "Synthetic unrelated guidance."
        web = self.run_to("web")["data"]["web"]
        self.assertEqual(web["unresolved_gap_ids"], ["r-export"])

    def test_web_findings_do_not_close_review_gaps(self):
        data = app.run_pipeline(self.config)["data"]
        self.assertEqual(len(data["review"]["gap_ids"]), 2)
        self.assertEqual(data["web"]["unresolved_gap_ids"], [])

    def test_feedback_deduplicates_and_preserves_originals(self):
        data = self.run_to("feedback")["data"]["feedback"]
        self.assertEqual(data["duplicate_count"], 1)
        self.assertEqual(data["groups"][0]["original_ids"], ["fb-1", "fb-2"])
        self.assertEqual(data["groups"][0]["excerpts"][1]["text"], self.config["feedback"][1]["text"])
        self.assertEqual(data["themes"][0]["unique_feedback_count"], 2)

    def test_duplicates_merge_requirement_tags(self):
        self.config["feedback"][1]["requirement_ids"] = ["r-retention"]
        data = self.run_to("feedback")["data"]["feedback"]
        self.assertEqual(data["groups"][0]["requirement_ids"], ["r-export", "r-retention"])
        self.assertEqual(data["themes"][1]["unique_feedback_count"], 2)

    def test_feedback_to_journey_support_chain(self):
        data = app.run_pipeline(self.config)["data"]
        step = data["journey"]["steps"][0]
        self.assertEqual(step["score"], 2)
        self.assertEqual(step["finding_ids"], ["finding-1"])
        self.assertEqual(step["supporting_group_ids"], ["feedback-group-1", "feedback-group-2"])
        self.assertEqual(data["journey"]["steps"][1]["satisfied_by"], ["a-setup"])

    def test_feedback_change_changes_journey(self):
        self.config["feedback"] = [self.config["feedback"][-1]]
        step = app.run_pipeline(self.config)["data"]["journey"]["steps"][0]
        self.assertEqual(step["action_id"], "c-retention")

    def test_empty_feedback_uses_explicit_fallback(self):
        self.config["feedback"] = []
        data = app.run_pipeline(self.config)["data"]
        self.assertEqual(data["feedback"]["themes"], [])
        self.assertEqual(data["journey"]["steps"][0]["score"], 0)
        self.assertIn("fallback", data["journey"]["steps"][0]["rationale"])

    def test_completed_prerequisites_not_recommended(self):
        self.config["completed_actions"] = ["a-setup"]
        steps = app.run_pipeline(self.config)["data"]["journey"]["steps"]
        self.assertEqual([s["action_id"] for s in steps], ["b-export", "c-retention"])

    def test_impossible_two_step_journey_rejected(self):
        self.config["completed_actions"] = ["a-setup", "b-export"]
        self.invalid()

    def test_cycle_rejected(self):
        self.config["actions"][0]["prerequisites"] = ["b-export"]
        self.invalid()

    def test_unknown_prerequisite_rejected(self):
        self.config["actions"][0]["prerequisites"] = ["unknown"]
        self.invalid()

    def test_inconsistent_completed_actions_rejected(self):
        self.config["completed_actions"] = ["b-export"]
        self.invalid()

    def test_unknown_requirement_rejected(self):
        self.config["feedback"][0]["requirement_ids"] = ["unknown"]
        self.invalid()

    def test_duplicate_record_id_rejected(self):
        self.config["documents"].append(copy.deepcopy(self.config["documents"][0]))
        self.invalid()

    def test_invalid_schema_types_and_fields(self):
        for key, value in (("schema_version", True), ("synthetic", False),
                           ("feedback", {}), ("unexpected", [])):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(config)

    def test_disallowed_urls(self):
        bad_urls = [
            "http://guides.example.org/export", "https://evil.example/export",
            "https://guides.example.org.evil.example/export",
            "https://sub.guides.example.org/export",
            "https://user:pass@guides.example.org/export",
            "https://guides.example.org:444/export",
            "https://guides.example.org:bad/export",
            "https://guides.example.org/export#fragment",
            "https://guides.example.org\\@evil.example/export",
            "https://guides.example.org/\nexport",
        ]
        for url in bad_urls:
            with self.subTest(url=url):
                config = copy.deepcopy(self.config)
                config["sources"][0]["url"] = url
                with self.assertRaises(app.ValidationError):
                    app.validate_input(config)

    def test_duplicate_canonical_urls_rejected(self):
        self.config["sources"][1]["url"] = "https://guides.example.org:443/export"
        self.invalid()

    def test_bad_timestamp_rejected(self):
        for stamp in ("not-a-date", "2026-09-23T10:00:00"):
            with self.subTest(stamp=stamp):
                self.config["sources"][0]["retrieved_at"] = stamp
                self.invalid()

    def test_tampered_handoff_rejected(self):
        web = self.run_to("web")
        web["data"]["web"]["findings"][0]["excerpt"]["text"] = "Invented evidence"
        with self.assertRaises(app.ValidationError):
            app.advance(self.config, "feedback", web)

    def test_wrong_stage_rejected(self):
        review = self.run_to("review")
        with self.assertRaises(app.ValidationError):
            app.advance(self.config, "journey", review)

    def test_tampered_journey_rejected(self):
        final = app.run_pipeline(self.config)
        final["data"]["journey"]["steps"].reverse()
        with self.assertRaises(app.ValidationError):
            app.validate_envelope(final, "journey", self.config)

    def test_bool_cannot_replace_numeric_count(self):
        feedback = self.run_to("feedback")
        feedback["data"]["feedback"]["duplicate_count"] = True
        with self.assertRaises(app.ValidationError):
            app.advance(self.config, "journey", feedback)

    def test_deterministic_and_no_input_mutation(self):
        before = copy.deepcopy(self.config)
        self.assertEqual(app.run_pipeline(self.config), app.run_pipeline(self.config))
        self.assertEqual(self.config, before)

    def test_cli_success_one_json(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "example_input.json")],
                                 capture_output=True, text=True, cwd=HERE)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")

    def test_cli_missing_file(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                                  str(HERE / "nonexistent.json")],
                                 capture_output=True, text=True, cwd=HERE)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_bad_usage(self):
        process = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py")],
                                 capture_output=True, text=True, cwd=HERE)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["status"], "error")

    def test_cli_invalid_input_without_extra_files(self):
        payloads = (b"{", b'{"x":1,"x":2}', b'{"value":NaN}', b"\xff",
                    json.dumps({"schema_version": 1}).encode(),
                    b" " * (app.MAX_INPUT_BYTES + 1))
        for payload in payloads:
            with self.subTest(prefix=payload[:30]):
                output = io.StringIO()
                with patch.object(Path, "open", return_value=io.BytesIO(payload)):
                    with contextlib.redirect_stdout(output):
                        code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_all_requirements_evidenced(self):
        self.config["documents"][0]["text"] += " Export and retention guidance."
        data = app.run_pipeline(self.config)["data"]
        self.assertEqual(data["review"]["gap_ids"], [])
        self.assertEqual(data["web"]["findings"], [])
        self.assertEqual(data["feedback"]["themes"], [])
        self.assertEqual(len(data["journey"]["steps"]), 2)


if __name__ == "__main__":
    unittest.main()
