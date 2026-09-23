"""Standard-library tests; every fixture is synthetic and no scratch files are used."""

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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def clear_progress(self):
        self.data["guided"]["progress"] = {"completed_step_ids": [], "responses": {}}

    def test_exact_source_citations(self):
        result = app.run_pipeline(self.data)
        for finding in result["research"]["findings"]:
            citation = finding["citation"]
            source = next(s for s in self.data["research"]["sources"] if s["id"] == citation["source_id"])
            passage = next(p for p in source["passages"] if p["id"] == citation["passage_id"])
            self.assertEqual(finding["quote"], passage["text"][citation["start"]:citation["end"]])
            self.assertEqual(citation["source_title"], source["title"])

    def test_ranking_and_limit(self):
        self.data["research"]["max_findings"] = 1
        result = app.run_research(self.data)
        self.assertEqual(result["retrieved_count"], 1)
        self.assertEqual(result["findings"][0]["citation"]["passage_id"], "catalog")
        self.assertEqual(result["findings"][0]["score"], 3)

    def test_ties_use_identifiers_not_input_order(self):
        first = app.run_research(self.data)
        self.data["research"]["sources"][0]["passages"].reverse()
        self.assertEqual(first, app.run_research(self.data))

    def test_normal_to_guided_handoff(self):
        result = app.run_pipeline(self.data)
        valid_ids = {finding["id"] for finding in result["research"]["findings"]}
        for step in result["guided"]["steps"]:
            self.assertTrue(step["evidence_finding_ids"])
            self.assertLessEqual(set(step["evidence_finding_ids"]), valid_ids)
        self.assertEqual([s["status"] for s in result["guided"]["steps"]],
                         ["completed", "awaiting_input", "blocked"])
        self.assertEqual(result["guided"]["next_step_ids"], ["create-catalog"])
        self.assertEqual(result["guided"]["progress"]["completion_percent"], 33.33)

    def test_progress_replay_unlocks_next_step(self):
        progress = self.data["guided"]["progress"]
        progress["responses"]["create-catalog"] = {"catalog_title": "Synthetic products"}
        self.assertEqual(app.run_pipeline(self.data)["guided"]["steps"][1]["status"], "ready")
        progress["completed_step_ids"].append("create-catalog")
        result = app.run_pipeline(self.data)
        self.assertEqual(result["guided"]["steps"][2]["status"], "awaiting_input")
        self.assertEqual(result["guided"]["progress"]["completed_count"], 2)

    def test_complete_progress(self):
        self.data["guided"]["progress"] = {
            "completed_step_ids": ["verify-catalog", "create-account", "create-catalog"],
            "responses": {
                "create-account": {"display_name": "Synthetic shop"},
                "create-catalog": {"catalog_title": "Synthetic catalog"},
                "verify-catalog": {"reviewer_name": "Synthetic reviewer"},
            },
        }
        result = app.run_pipeline(self.data)
        self.assertEqual(result["guided"]["progress"]["completion_percent"], 100.0)
        self.assertEqual(result["guided"]["next_step_ids"], [])

    def test_empty_sources_block_guided(self):
        self.clear_progress()
        self.data["research"]["sources"] = []
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["findings"], [])
        self.assertTrue(all(step["status"] == "blocked" for step in result["guided"]["steps"]))

    def test_no_matches_do_not_invent_evidence(self):
        self.clear_progress()
        self.data["research"]["keywords"] = ["unicorn"]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["retrieved_count"], 0)
        self.assertEqual(result["guided"]["next_step_ids"], [])

    def test_evidence_is_only_from_retrieved_passages(self):
        self.clear_progress()
        self.data["research"]["keywords"] = ["display"]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["research"]["retrieved_count"], 1)
        review = result["guided"]["steps"][2]
        self.assertEqual(review["evidence_finding_ids"], [])
        self.assertEqual(review["missing_evidence_keywords"], ["verification"])

    def test_all_evidence_keywords_required(self):
        self.clear_progress()
        self.data["guided"]["steps"][0]["evidence_keywords"] = ["account", "absent"]
        result = app.run_pipeline(self.data)
        self.assertEqual(result["guided"]["steps"][0]["status"], "blocked")
        self.assertEqual(result["guided"]["steps"][0]["missing_evidence_keywords"], ["absent"])

    def test_tampered_handoff_rejected(self):
        for mutation in ("quote", "source", "offset", "score", "dropped"):
            with self.subTest(mutation=mutation):
                research = app.run_research(self.data)
                if mutation == "quote":
                    research["findings"][0]["quote"] = "Fabricated account instruction"
                elif mutation == "source":
                    research["findings"][0]["citation"]["source_id"] = "unknown"
                elif mutation == "offset":
                    research["findings"][0]["citation"]["end"] = True
                elif mutation == "score":
                    research["findings"][0]["score"] += 1
                else:
                    research["findings"].pop()
                with self.assertRaises(app.ValidationError):
                    app.run_guided(self.data, research)

    def test_dependency_cycles_and_unknown_references(self):
        for prerequisite in ("verify-catalog", "nonexistent", "create-account"):
            with self.subTest(prerequisite=prerequisite):
                data = copy.deepcopy(self.data)
                data["guided"]["steps"][0]["prerequisites"] = [prerequisite]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_completed_progress(self):
        for mutation in ("missing_response", "missing_prerequisite", "missing_evidence", "unknown"):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                if mutation == "missing_response":
                    data["guided"]["progress"]["responses"] = {}
                elif mutation == "missing_prerequisite":
                    data["guided"]["progress"]["completed_step_ids"] = ["create-catalog"]
                    data["guided"]["progress"]["responses"]["create-catalog"] = {"catalog_title": "Synthetic"}
                elif mutation == "missing_evidence":
                    data["research"]["keywords"] = ["unicorn"]
                else:
                    data["guided"]["progress"]["completed_step_ids"] = ["unknown"]
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_duplicate_identifiers_and_keywords(self):
        for mutation in ("source", "passage", "step", "keyword"):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                if mutation == "source":
                    data["research"]["sources"].append(copy.deepcopy(data["research"]["sources"][0]))
                elif mutation == "passage":
                    passages = data["research"]["sources"][0]["passages"]
                    passages.append(copy.deepcopy(passages[0]))
                elif mutation == "step":
                    data["guided"]["steps"].append(copy.deepcopy(data["guided"]["steps"][0]))
                else:
                    data["research"]["keywords"].append("ACCOUNT")
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_types_and_unknown_fields(self):
        for path, value in (
            (("schema_version",), True),
            (("research", "max_findings"), 0),
            (("research", "max_findings"), True),
            (("research", "sources"), {}),
            (("guided", "steps"), []),
            (("fixture_label",), "Not marked synthetic"),
            (("extra",), "unexpected"),
        ):
            with self.subTest(path=path):
                data = copy.deepcopy(self.data)
                target = data
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_response_validation(self):
        for response in ({"display_name": " "}, {"display_name": 7}, {"password": "synthetic"}):
            with self.subTest(response=response):
                self.data["guided"]["progress"]["responses"]["create-account"] = response
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(self.data)

    def test_unicode_citations_and_literal_keywords(self):
        self.clear_progress()
        self.data["research"]["keywords"] = ["café", "a+b"]
        passage = self.data["research"]["sources"][0]["passages"][0]
        passage["text"] = "🧱 CAFÉ offers a+b account setup."
        result = app.run_research(self.data)
        self.assertEqual(result["findings"][0]["score"], 2)
        self.assertEqual(result["findings"][0]["citation"]["end"], len(passage["text"]))
        self.assertFalse(app.matches("account", "accounting"))
        self.assertFalse(app.matches("a+b", "aaab"))

    def test_deterministic_and_does_not_mutate_input(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(first, app.run_pipeline(self.data))
        self.assertEqual(self.data, original)

    def test_cli_success_subprocess(self):
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ok")
        self.assertEqual(process.stderr, "")
        self.assertEqual(len(process.stdout.splitlines()), 1)

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["one", "two"]):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(process.returncode, 2)
                self.assertEqual(json.loads(process.stdout)["status"], "error")
                self.assertEqual(process.stderr, "")

    def test_cli_malformed_json_and_validation_errors(self):
        for text in ('{', '{"a": 1, "a": 2}', '{"a": NaN}', '[]', '{}',
                     json.dumps({**self.data, "schema_version": False})):
            with self.subTest(text=text[:30]):
                output = io.StringIO()
                with patch.object(app.Path, "read_text", return_value=text), redirect_stdout(output):
                    self.assertEqual(app.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_file_decoding_error(self):
        output = io.StringIO()
        with patch.object(app.Path, "read_text", side_effect=UnicodeError("synthetic decoding error")):
            with redirect_stdout(output):
                self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
