"""Synthetic fixtures only; all file access is confined to this build directory."""

import copy
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
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)["stages"]

    def test_full_pipeline_and_order(self):
        stages = self.run_data()
        self.assertEqual(list(stages), ["interests", "adaptive", "web", "review"])
        self.assertEqual(len(stages["web"]["findings"]), 3)
        self.assertEqual([c["status"] for c in stages["review"]["checks"]],
                         ["evidence_present", "evidence_present", "gap"])

    def test_weighted_rank_and_explanations(self):
        recs = self.run_data()["interests"]
        self.assertEqual([(r["id"], r["score"]) for r in recs],
                         [("privacy-kit", 5), ("analytics-kit", 3)])
        self.assertIn("privacy (weight 5)", recs[0]["explanation"])

    def test_ties_use_stable_item_id(self):
        self.data["profile"]["interests"]["privacy"] = 3
        self.assertEqual(self.run_data()["interests"][0]["id"], "analytics-kit")

    def test_topic_exclusion_overrides_weight(self):
        self.data["profile"]["interests"]["advertising"] = 10
        self.assertNotIn("ad-kit", [r["id"] for r in self.run_data()["interests"]])

    def test_item_exclusion_propagates_through_all_stages(self):
        self.data["profile"]["excluded_items"] = ["privacy-kit"]
        stages = self.run_data()
        self.assertEqual([r["id"] for r in stages["interests"]], ["analytics-kit"])
        self.assertNotIn("privacy", [s["id"] for s in stages["adaptive"]["steps"]])
        self.assertNotIn("privacy", [f["step_id"] for f in stages["web"]["findings"]])
        self.assertEqual(stages["review"]["checks"][1]["status"], "out_of_scope")

    def test_limit_zero_yields_empty_research(self):
        self.data["profile"]["limit"] = 0
        stages = self.run_data()
        self.assertEqual(stages["interests"], [])
        self.assertEqual(stages["adaptive"]["steps"], [])
        self.assertEqual(stages["web"]["findings"], [])
        self.assertTrue(all(c["status"] == "out_of_scope" for c in stages["review"]["checks"]))

    def test_empty_interests(self):
        self.data["profile"]["interests"] = {}
        self.assertEqual(self.run_data()["interests"], [])

    def test_prerequisites_order_and_lineage(self):
        steps = self.run_data()["adaptive"]["steps"]
        self.assertEqual([s["id"] for s in steps], ["basics", "privacy", "analytics"])
        self.assertEqual(steps[0]["recommendation_ids"], ["privacy-kit", "analytics-kit"])
        self.assertIn("prerequisite", steps[0]["reason"])

    def test_completed_prerequisite_not_repeated(self):
        self.data["profile"]["completed_steps"] = ["basics"]
        stages = self.run_data()
        self.assertEqual([s["id"] for s in stages["adaptive"]["steps"]], ["privacy", "analytics"])
        self.assertEqual(stages["review"]["checks"][0]["status"], "out_of_scope")

    def test_excluded_prerequisite_blocks_roots_without_leaks(self):
        self.data["profile"]["excluded_topics"].append("foundations")
        stages = self.run_data()
        self.assertEqual(stages["adaptive"]["steps"], [])
        self.assertEqual(len(stages["adaptive"]["blocked"]), 2)
        self.assertIn("basics", stages["adaptive"]["blocked"][0]["reason"])
        self.assertEqual(stages["web"]["findings"], [])

    def test_experience_skips_optional_intro_but_preserves_prerequisite(self):
        self.data["profile"]["experience"] = "experienced"
        self.data["steps"].append({"id": "intro", "topic": "privacy", "format": "text",
                                   "audience": "beginner", "prerequisites": [],
                                   "queries": ["orientation"]})
        steps = self.run_data()["adaptive"]["steps"]
        self.assertNotIn("intro", [s["id"] for s in steps])
        self.assertIn("basics", [s["id"] for s in steps])
        self.assertTrue(all("experience=experienced" in s["reason"] for s in steps))

    def test_format_preference_orders_relevant_roots(self):
        self.data["profile"]["preferred_format"] = "video"
        self.data["steps"].append({"id": "privacy-video", "topic": "privacy", "format": "video",
                                   "audience": "all", "prerequisites": ["basics"],
                                   "queries": ["retention schedule"]})
        ids = [s["id"] for s in self.run_data()["adaptive"]["steps"]]
        self.assertLess(ids.index("privacy-video"), ids.index("privacy"))

    def test_missing_prerequisite_rejected(self):
        self.data["steps"][0]["prerequisites"] = ["missing"]
        with self.assertRaisesRegex(app.ValidationError, "Unknown prerequisite"):
            self.run_data()

    def test_cycle_rejected(self):
        self.data["steps"][0]["prerequisites"] = ["privacy"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            self.run_data()

    def test_exact_host_allowlist(self):
        for url in ("http://research.example.test/a", "https://research.example.test.evil.test/a",
                    "https://sub.research.example.test/a", "https://user@research.example.test/a",
                    "https://research.example.test:444/a", "https://research.example.test/a#x",
                    "https://research.example.test\\evil/a", "https://research.example.test/\na",
                    "file:///data", "https://research.example.test:bad/a"):
            with self.subTest(url=url):
                self.data["sources"][0]["url"] = url
                with self.assertRaises(app.ValidationError):
                    self.run_data()

    def test_https_port_443_accepted(self):
        self.data["sources"][0]["url"] = "https://research.example.test:443/a"
        self.assertTrue(self.run_data()["web"]["findings"])

    def test_retrieval_offsets_and_source_provenance(self):
        findings = self.run_data()["web"]["findings"]
        source = self.data["sources"][0]
        for finding in findings:
            self.assertEqual(finding["source_id"], source["id"])
            self.assertEqual(finding["url"], source["url"])
            self.assertEqual(finding["quote"], source["body"][finding["start"]:finding["end"]])

    def test_unmatched_query_is_traceable(self):
        self.data["steps"][1]["queries"] = ["absent synthetic phrase"]
        stages = self.run_data()
        self.assertEqual(stages["web"]["unmatched"], [{"step_id": "privacy", "query": "absent synthetic phrase"}])
        self.assertEqual(stages["review"]["checks"][1]["status"], "gap")
        self.assertIn("research", stages["review"]["checks"][1]["gaps"][0])

    def test_no_sources_reports_gaps(self):
        self.data["sources"] = []
        stages = self.run_data()
        self.assertEqual(len(stages["web"]["unmatched"]), 3)
        self.assertTrue(all(c["status"] == "gap" for c in stages["review"]["checks"]))

    def test_document_evidence_offsets(self):
        check = self.run_data()["review"]["checks"][1]
        evidence = check["document_evidence"][0]
        self.assertEqual(evidence["document_id"], "policy")
        self.assertEqual(evidence["quote"], self.data["documents"][0]["body"][evidence["start"]:evidence["end"]])
        self.assertTrue(check["finding_ids"])

    def test_unassigned_document_cannot_fill_gap(self):
        self.data["documents"].append({"id": "other", "title": "SYNTHETIC unrelated draft", "body": "aggregate metrics"})
        check = self.run_data()["review"]["checks"][2]
        self.assertEqual(check["status"], "gap")
        self.assertEqual(check["document_evidence"], [])
        self.assertIn("assigned document", check["gaps"][0])

    def test_case_insensitive_literal_matching(self):
        self.data["requirements"][0]["phrase"] = "DATA INVENTORY"
        self.assertEqual(self.run_data()["review"]["checks"][0]["status"], "evidence_present")
        self.assertFalse(app.contains(".*", "data inventory"))

    def test_no_certification_claim(self):
        review = self.run_data()["review"]
        self.assertIn("not certification", review["disclaimer"])
        self.assertNotIn("certified", json.dumps(review))

    def test_input_not_mutated_and_deterministic(self):
        before = copy.deepcopy(self.data)
        first = self.run_data()
        self.assertEqual(first, self.run_data())
        self.assertEqual(self.data, before)

    def test_duplicate_id_rejected(self):
        self.data["catalog"].append(copy.deepcopy(self.data["catalog"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate id"):
            self.run_data()

    def test_invalid_weights_and_limits(self):
        for weight in (True, 0, 11, 1.5, "5"):
            with self.subTest(weight=weight):
                self.data["profile"]["interests"]["privacy"] = weight
                with self.assertRaises(app.ValidationError):
                    self.run_data()
        self.data["profile"]["interests"]["privacy"] = 5
        for limit in (-1, True, 101):
            self.data["profile"]["limit"] = limit
            with self.assertRaises(app.ValidationError):
                self.run_data()

    def test_schema_and_synthetic_label_required(self):
        for key, value in (("schema_version", True), ("fixture_label", "REAL"), ("documents", {})):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_unknown_fields_rejected(self):
        self.data["extra"] = "untrusted"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_unknown_requirement_reference_rejected(self):
        self.data["requirements"][0]["document_ids"] = ["unknown"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_tampered_finding_rejected_at_boundary(self):
        stages = self.run_data()
        stages["web"]["findings"][0]["quote"] = "invented evidence"
        with self.assertRaisesRegex(app.ValidationError, "Evidence does not match"):
            app.validate(self.data, stages)

    def test_tampered_recommendation_rejected_before_next_stage(self):
        original = app.interests_stage

        def tamper(data):
            rows = original(data)
            rows[0]["score"] = 900
            return rows

        with patch.object(app, "interests_stage", side_effect=tamper), patch.object(app, "adaptive_stage") as next_stage:
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data)
            next_stage.assert_not_called()

    def test_tampered_review_status_rejected(self):
        stages = self.run_data()
        stages["review"]["checks"][2]["status"] = "evidence_present"
        with self.assertRaisesRegex(app.ValidationError, "Incorrect review status"):
            app.validate(self.data, stages)

    def test_tampered_onboarding_lineage_rejected(self):
        stages = self.run_data()
        stages["adaptive"]["steps"][1]["recommendation_ids"] = ["analytics-kit"]
        with self.assertRaisesRegex(app.ValidationError, "Ungrounded onboarding lineage"):
            app.validate(self.data, stages)

    def test_cli_success_single_json(self):
        proc = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"),
                               str(HERE / "example_input.json")], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], ["does-not-exist.json"], ["example_input.json", "extra"], ["."]):
            proc = subprocess.run([sys.executable, "-B", "implementation.py", *args],
                                  cwd=HERE, capture_output=True, text=True)
            with self.subTest(args=args):
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for content in ("{bad", "null", '{"schema_version":1,"schema_version":1}', '{"schema_version":1}',
                        json.dumps({**self.data, "schema_version": 2})):
            with self.subTest(content=content[:50]), patch.object(Path, "open", return_value=io.StringIO(content)), patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(app.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")

    def test_cli_decode_failure(self):
        with patch.object(Path, "open", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")), patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(app.main(["synthetic.json"]), 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
