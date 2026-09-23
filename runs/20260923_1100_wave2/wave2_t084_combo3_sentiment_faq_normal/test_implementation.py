"""Synthetic fixtures only; tests use no network and create no scratch files."""

import copy
import io
import json
import pathlib
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


HERE = pathlib.Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def one_feedback(self, text, severity="low"):
        self.data["feedback"] = [{"id": "synthetic-test", "text": text, "severity": severity}]
        return app.sentiment_stage(self.data)["results"]["sentiment"][0]

    def test_positive_explanation(self):
        record = self.one_feedback("great helpful thanks")
        self.assertEqual((record["score"], record["sentiment"], record["priority"]), (4, "positive", "P3"))
        self.assertEqual(sum(e["contribution"] for e in record["evidence"]), 4)

    def test_negative_explanation(self):
        record = self.one_feedback("late failed terrible")
        self.assertEqual((record["score"], record["priority"]), (-5, "P1"))

    def test_negation(self):
        record = self.one_feedback("not good never bad")
        self.assertEqual(record["score"], 0)
        self.assertTrue(all(e["negated"] for e in record["evidence"]))

    def test_neutral_unknown_tokens(self):
        self.assertEqual(self.one_feedback("quasar resonance")["sentiment"], "neutral")

    def test_critical_overrides_positive_sentiment(self):
        self.assertEqual(self.one_feedback("excellent", "critical")["priority"], "P0")

    def test_priority_order_propagation(self):
        output = app.run_pipeline(self.data)
        results = output["results"]
        expected = [(r["feedback_id"], r["priority"]) for r in results["sentiment"]]
        self.assertEqual(expected[0], ("synthetic-f2", "P0"))
        for stage in ("faq", "normal"):
            self.assertEqual([(r["feedback_id"], r["priority"]) for r in results[stage]], expected)

    def test_faq_grounding(self):
        output = app.run_pipeline(self.data)
        answer = next(r for r in output["results"]["faq"] if r["feedback_id"] == "synthetic-f1")
        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["answer"], answer["evidence"][0]["citation"]["quote"])

    def test_explicit_abstention(self):
        results = app.run_pipeline(self.data)["results"]
        for stage in ("faq", "normal"):
            row = next(r for r in results[stage] if r["feedback_id"] == "synthetic-f4")
            self.assertEqual(row["status"], "abstained")
            self.assertTrue(row["reason"])
        answer = next(r for r in results["faq"] if r["feedback_id"] == "synthetic-f4")
        self.assertIsNone(answer["answer"])

    def test_faq_to_research_handoff(self):
        results = app.run_pipeline(self.data)["results"]
        for answer, research in zip(results["faq"], results["normal"]):
            self.assertEqual(research["query"], answer["research_query"])
            self.assertEqual(research["faq_status"], answer["status"])
            if answer["answer"]:
                self.assertIn(answer["answer"], research["query"])

    def test_research_can_find_after_faq_abstention(self):
        self.data["knowledge_base"] = []
        output = app.run_pipeline(self.data)
        self.assertTrue(all(a["status"] == "abstained" for a in output["results"]["faq"]))
        self.assertEqual(output["results"]["normal"][0]["status"], "found")

    def test_exact_research_citations(self):
        output = app.run_pipeline(self.data)
        sources = {s["id"]: s for s in self.data["research_sources"]}
        count = 0
        for research in output["results"]["normal"]:
            for finding in research["findings"]:
                citation = finding["citation"]
                source = sources[citation["source_id"]]
                self.assertEqual(source["text"][citation["start"]:citation["end"]], finding["text"])
                self.assertEqual(finding["text"], citation["quote"])
                count += 1
        self.assertGreater(count, 0)

    def test_unicode_offsets(self):
        source = {"id": "synthetic-unicode", "title": "Synthetic Unicode", "text": "  Café parcel tracking.\n  Café tracking late!"}
        hits = app.retrieve("café tracking late", [source], 3)
        self.assertEqual(len(hits), 2)
        for hit in hits:
            c = hit["citation"]
            self.assertEqual(source["text"][c["start"]:c["end"]], c["quote"])

    def test_retrieval_minimum_evidence(self):
        self.assertEqual(app.retrieve("tracking", self.data["knowledge_base"], 1), [])
        self.assertEqual(app.retrieve("the and", self.data["knowledge_base"], 1), [])

    def test_unpunctuated_newline_passages(self):
        source = {"id": "synthetic-lines", "title": "Synthetic lines",
                  "text": "  parcel tracking\nbattery support\n"}
        quotes = [citation["quote"] for citation in app.passages(source)]
        self.assertEqual(quotes, ["parcel tracking", "battery support"])
        self.assertEqual(len(app.retrieve("parcel tracking", [source], 1)), 1)

    def test_retrieval_ties_are_stable(self):
        sources = [{"id": identifier, "title": "Synthetic tie", "text": "parcel tracking."}
                   for identifier in ("b", "a")]
        self.assertEqual(app.retrieve("parcel tracking", sources, 1)[0]["citation"]["source_id"], "a")

    def test_empty_collections(self):
        for name in ("feedback", "knowledge_base", "research_sources"):
            self.data[name] = []
        self.assertEqual(app.run_pipeline(self.data)["results"], {"sentiment": [], "faq": [], "normal": []})

    def test_no_sources_abstains(self):
        self.data["knowledge_base"] = []
        self.data["research_sources"] = []
        result = app.run_pipeline(self.data)["results"]
        self.assertTrue(all(r["status"] == "abstained" for r in result["faq"] + result["normal"]))

    def test_input_not_mutated_and_deterministic(self):
        original = copy.deepcopy(self.data)
        first = app.run_pipeline(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(first, app.run_pipeline(self.data))

    def test_invalid_input_variants(self):
        variants = []
        for key, value in (("schema_version", True), ("synthetic", False), ("stage", []),
                           ("feedback", "wrong"), ("results", []), ("status", "error")):
            bad = copy.deepcopy(self.data)
            bad[key] = value
            variants.append(bad)
        bad = copy.deepcopy(self.data)
        bad["feedback"][0]["severity"] = "urgent"
        variants.append(bad)
        bad = copy.deepcopy(self.data)
        bad["feedback"][0]["text"] = "   "
        variants.append(bad)
        bad = copy.deepcopy(self.data)
        bad["feedback"].append(copy.deepcopy(bad["feedback"][0]))
        variants.append(bad)
        bad = copy.deepcopy(self.data)
        bad["unknown"] = 1
        variants.extend([bad, [], None])
        for bad in variants:
            with self.subTest(bad=bad):
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(bad)

    def test_tampered_sentiment_rejected_before_faq(self):
        output = app.sentiment_stage(self.data)
        output["results"]["sentiment"][0]["priority"] = "P3"
        with self.assertRaises(app.ValidationError):
            app.faq_stage(output)

    def test_tampered_faq_rejected_before_research(self):
        output = app.faq_stage(app.sentiment_stage(self.data))
        output["results"]["faq"][0]["evidence"][0]["citation"]["start"] += 1
        with self.assertRaises(app.ValidationError):
            app.normal_stage(output)

    def test_wrong_stage_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.normal_stage(self.data)

    def test_final_tampering_rejected(self):
        output = app.run_pipeline(self.data)
        output["results"]["normal"][0]["findings"][0]["text"] = "Invented conclusion"
        with self.assertRaises(app.ValidationError):
            app.validate(output)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                              cwd=HERE, capture_output=True, text=True, check=False)

    def test_cli_success(self):
        completed = self.cli("example_input.json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertEqual(output["stage"], "normal")
        self.assertEqual(output["status"], "ok")
        self.assertEqual(completed.stderr, "")
        app.validate(output, "normal")

    def test_cli_missing_file_and_usage(self):
        for args in ((), ("missing-synthetic.json",), ("example_input.json", "extra")):
            completed = self.cli(*args)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_invalid_json_and_schema(self):
        for payload in ("{", "null", '{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(payload=payload), patch("builtins.open", mock_open(read_data=payload)):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = app.main(["synthetic-invalid.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stream.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
