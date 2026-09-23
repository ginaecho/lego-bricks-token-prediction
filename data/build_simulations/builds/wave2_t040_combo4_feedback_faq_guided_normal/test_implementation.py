import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return app.run_pipeline(self.data)

    def test_stage_order(self):
        self.assertEqual([s["stage"] for s in self.run_data()["stages"]],
                         ["feedback", "faq", "guided", "normal"])

    def test_deduplication_preserves_original(self):
        rows = self.run_data()["stages"][0]["records"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["details"]["feedback_ids"], ["f1", "f2"])
        self.assertEqual(rows[0]["text"], self.data["feedback"][0]["text"])

    def test_theme_support(self):
        feedback, faq, _, _ = self.run_data()["stages"]
        self.assertEqual(feedback["records"][0]["details"]["theme_ids"], ["inventory"])
        self.assertEqual(faq["records"][0]["upstream_ids"], ["f1"])

    def test_grounded_answer(self):
        row = self.run_data()["stages"][1]["records"][0]
        self.assertEqual(row["state"], "answered")
        self.assertEqual(row["text"], row["citations"][0]["quote"])
        self.assertEqual(row["citations"][0]["collection"], "knowledge_base")

    def test_explicit_faq_abstention(self):
        row = self.run_data()["stages"][1]["records"][2]
        self.assertEqual(row["state"], "abstained")
        self.assertEqual(row["text"], "")
        self.assertEqual(row["citations"], [])
        self.assertEqual(row["details"]["reason"], "no_matching_knowledge_passage")

    def test_prerequisites_and_progress(self):
        result = self.run_data()
        states = {r["id"]: r["state"] for r in result["stages"][2]["records"]}
        self.assertEqual(states, {"connect": "completed", "notify": "ready",
                                  "teleport": "blocked", "audit": "blocked"})
        self.assertEqual(result["progress"], {
            "total": 4, "completed": 1, "ready": 1, "blocked": 2,
            "fraction_complete": 0.25})

    def test_exact_research_citations(self):
        for row in self.run_data()["stages"][3]["records"]:
            for ref in row["citations"]:
                app.validate_citation(ref, self.data, "research_sources")
                self.assertEqual(row["text"], ref["quote"])

    def test_full_provenance_chain(self):
        feedback, faq, guided, normal = self.run_data()["stages"]
        current = normal["records"][0]
        for stage in (guided, faq, feedback):
            upstream = {r["id"]: r for r in stage["records"]}
            current = upstream[current["upstream_ids"][0]]
        self.assertEqual(current["details"]["feedback_ids"], ["f1", "f2"])

    def test_abstention_propagates(self):
        self.data["knowledge_base"] = []
        self.data["completed_step_ids"] = []
        result = self.run_data()
        self.assertTrue(all(r["state"] == "blocked"
                            for r in result["stages"][2]["records"]))
        self.assertTrue(all(r["details"]["reason"] == "setup_blocked"
                            for r in result["stages"][3]["records"]))

    def test_no_research_match(self):
        self.data["research_sources"] = []
        row = self.run_data()["stages"][3]["records"][0]
        self.assertEqual(row["state"], "abstained")
        self.assertEqual(row["details"]["reason"], "no_matching_research_passage")

    def test_empty_collections(self):
        for name in ("feedback", "themes", "knowledge_base", "setup_steps",
                     "completed_step_ids", "research_sources"):
            self.data[name] = []
        result = self.run_data()
        self.assertTrue(all(s["records"] == [] for s in result["stages"]))
        self.assertEqual(result["progress"]["fraction_complete"], 1.0)

    def test_unclassified_feedback(self):
        self.data["feedback"].append({"id": "other", "text": "Wonderful colors!"})
        row = self.run_data()["stages"][0]["records"][-1]
        self.assertEqual(row["details"]["theme_ids"], [])

    def test_duplicate_id_rejected(self):
        self.data["feedback"][1]["id"] = "f1"
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_missing_field_rejected(self):
        del self.data["themes"][0]["question"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_wrong_types_rejected(self):
        for value in (None, [], True, "1"):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["schema_version"] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_unknown_prerequisite_rejected(self):
        self.data["setup_steps"][0]["prerequisites"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_cycle_rejected(self):
        self.data["setup_steps"][0]["prerequisites"] = ["notify"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_invalid_completion_rejected(self):
        self.data["completed_step_ids"] = ["notify"]
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_unsupported_completion_rejected(self):
        self.data["completed_step_ids"].append("teleport")
        with self.assertRaises(app.ValidationError):
            self.run_data()

    def test_invalid_citation_rejected(self):
        feedback = app.analyze_feedback(self.data)
        feedback["records"][0]["citations"][0]["quote"] = "fabricated"
        with self.assertRaises(app.ValidationError):
            app.answer_faq(self.data, feedback)

    def test_unknown_upstream_rejected(self):
        feedback = app.analyze_feedback(self.data)
        faq = app.answer_faq(self.data, feedback)
        faq["records"][0]["upstream_ids"] = ["missing"]
        with self.assertRaises(app.ValidationError):
            app.guide_setup(self.data, faq, feedback)

    def test_forged_ready_state_rejected(self):
        _, faq, guided, _ = self.run_data()["stages"]
        next(r for r in guided["records"] if r["id"] == "audit")["state"] = "ready"
        with self.assertRaises(app.ValidationError):
            app.research(self.data, guided, faq)

    def test_changed_handoff_query_rejected(self):
        _, faq, guided, _ = self.run_data()["stages"]
        guided["records"][0]["details"]["research_query"] = "unrelated"
        with self.assertRaises(app.ValidationError):
            app.research(self.data, guided, faq)

    def test_citation_offset_rejected(self):
        feedback = app.analyze_feedback(self.data)
        feedback["records"][0]["citations"][0]["end"] = 999
        with self.assertRaises(app.ValidationError):
            app.answer_faq(self.data, feedback)

    def test_completion_unlocks_downstream_step(self):
        self.data["completed_step_ids"].append("notify")
        result = self.run_data()
        guided = {r["id"]: r for r in result["stages"][2]["records"]}
        normal = {r["id"]: r for r in result["stages"][3]["records"]}
        self.assertEqual(guided["audit"]["state"], "ready")
        self.assertEqual(normal["audit"]["state"], "found")
        self.assertEqual(result["progress"]["fraction_complete"], 0.5)

    def test_missing_feedback_blocks_setup(self):
        self.data["feedback"] = []
        self.data["completed_step_ids"] = []
        result = self.run_data()
        self.assertEqual(result["stages"][1]["records"], [])
        self.assertTrue(all(r["state"] == "blocked"
                            for r in result["stages"][2]["records"]))

    def test_unicode_excerpt_offsets(self):
        self.data["knowledge_base"][0]["passages"][0]["text"] = "Inventory café: 你好."
        result = self.run_data()
        ref = result["stages"][1]["records"][0]["citations"][0]
        self.assertEqual(ref["end"], len("Inventory café: 你好."))
        app.validate_citation(ref, self.data, "knowledge_base")

    def test_deterministic_and_nonmutating(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(self.data, original)

    def test_retrieval_ties_use_identifiers(self):
        sources = [{"id": source, "title": "Synthetic", "passages": [
            {"id": "p", "text": "inventory"}]} for source in ("z", "a")]
        ref = app.retrieve("inventory", sources, "knowledge_base")
        self.assertEqual(ref["source_id"], "a")

    def test_word_boundaries(self):
        self.data["themes"][0]["keywords"] = ["invent"]
        self.data["completed_step_ids"] = []
        row = self.run_data()["stages"][0]["records"][0]
        self.assertEqual(row["details"]["theme_ids"], [])

    def test_steps_topologically_ordered(self):
        self.data["setup_steps"].reverse()
        rows = self.run_data()["stages"][2]["records"]
        positions = {r["id"]: i for i, r in enumerate(rows)}
        self.assertLess(positions["connect"], positions["notify"])
        self.assertLess(positions["notify"], positions["audit"])

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               *args], cwd=ROOT, capture_output=True, text=True)

    def test_cli_success(self):
        result = self.cli("example_input.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_file(self):
        result = self.cli("nonexistent-input.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_usage_error(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_schema(self):
        result = self.cli("build_manifest.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_malformed_json(self):
        result = self.cli("implementation.py")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"x": 1, "x": 2}', object_pairs_hook=app.unique_object)

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"x": NaN}', parse_constant=app.reject_constant)


if __name__ == "__main__":
    unittest.main()
