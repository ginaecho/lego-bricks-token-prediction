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
        self.source = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args, content=None):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            input=content, text=True, capture_output=True, cwd=ROOT, check=False)

    def test_lexicon_explanation(self):
        score = app.score_sentiment("Great but broken.")
        self.assertEqual(score["score"], 0)
        self.assertEqual(score["label"], "neutral")
        self.assertEqual(sum(x["contribution"] for x in score["contributions"]), 0)

    def test_negation_and_boundary(self):
        self.assertEqual(app.score_sentiment("not good")["score"], -1)
        self.assertEqual(app.score_sentiment("not broken")["score"], 2)
        self.assertEqual(app.score_sentiment("not. good")["score"], 1)
        self.assertEqual(app.score_sentiment("not one two three good")["score"], 1)
        self.assertEqual(app.score_sentiment("not very good")["score"], -1)

    def test_unknown_words_neutral(self):
        self.assertEqual(app.score_sentiment("ordinary table 你好")["contributions"], [])

    def test_severity_dominates_sentiment(self):
        self.source["issues"] = [
            {"id": "low", "text": "unsafe " * 100, "severity": "low", "document_ids": []},
            {"id": "medium", "text": "excellent", "severity": "medium", "document_ids": []},
        ]
        records = app.sentiment_stage(self.source)["issues"]
        self.assertEqual([r["issue_id"] for r in records], ["medium", "low"])
        self.assertEqual(records[1]["priority_score"], 99)

    def test_tie_breaking(self):
        issue = self.source["issues"][0]
        issue["id"] = "z"
        self.source["issues"] = [issue, dict(issue, id="a")]
        self.assertEqual([x["issue_id"] for x in app.sentiment_stage(self.source)["issues"]], ["a", "z"])

    def test_disagreement_and_grounding(self):
        result = app.run_pipeline(self.source)
        finding = result["research"]["findings"][0]
        topic = next(t for t in finding["topics"] if t["topic"] == "checkout instability")
        self.assertTrue(topic["disagreement"])
        self.assertEqual(topic["conclusion"], "disputed")
        self.assertEqual(topic["source_count"], 2)
        docs = {d["id"]: d["text"] for d in self.source["documents"]}
        for evidence in topic["evidence"]:
            self.assertIn(evidence["quote"], docs[evidence["document_id"]])

    def test_uncertainty_and_single_source(self):
        topics = app.run_pipeline(self.source)["research"]["findings"][0]["topics"]
        topic = next(t for t in topics if t["topic"] == "payment loss")
        self.assertEqual(topic["conclusion"], "inconclusive")
        self.assertEqual(topic["stance_source_counts"]["uncertain"], 1)
        questions = app.run_pipeline(self.source)["research"]["findings"][0]["unresolved_questions"]
        self.assertTrue(any("uncertain" in q for q in questions))
        self.assertTrue(any("independent" in q for q in questions))

    def test_cross_stage_propagation(self):
        result = app.run_pipeline(self.source)
        for before, after in zip(result["sentiment"]["issues"], result["research"]["findings"]):
            for field in ("issue_id", "severity", "priority_score", "rank", "document_ids"):
                self.assertEqual(before[field], after[field])
            self.assertEqual(before["sentiment"]["score"], after["sentiment_score"])
        self.source["issues"][1]["severity"] = "critical"
        self.source["issues"][1]["text"] = "unsafe " * 20
        self.assertEqual(app.run_pipeline(self.source)["research"]["findings"][0]["issue_id"], "help")

    def test_empty_input_and_missing_evidence(self):
        self.assertEqual(app.run_pipeline(dict(self.source, issues=[], documents=[]))["research"]["findings"], [])
        finding = next(f for f in app.run_pipeline(self.source)["research"]["findings"] if f["issue_id"] == "shipping")
        self.assertEqual(finding["topics"], [])
        self.assertTrue(finding["unresolved_questions"])

    def test_document_without_claims(self):
        for doc in self.source["documents"]:
            doc["claims"] = []
        finding = app.run_pipeline(self.source)["research"]["findings"][0]
        self.assertEqual(finding["topics"], [])
        self.assertEqual(len(finding["unresolved_questions"]), 3)

    def test_document_counts_not_claim_counts(self):
        doc = self.source["documents"][0]
        doc["claims"].append({"topic": "checkout instability", "stance": "supports", "quote": "Checkout crashes"})
        topics = app.run_pipeline(self.source)["research"]["findings"][0]["topics"]
        self.assertEqual(topics[0]["stance_source_counts"]["supports"], 1)

    def test_handoff_tampering_rejected(self):
        handoff = app.sentiment_stage(self.source)
        for key, value in [("priority_score", 0), ("document_ids", []),
                           ("issue_id", "invented"), ("rank", True), ("rank", 1.0)]:
            mutated = copy.deepcopy(handoff)
            mutated["issues"][0][key] = value
            with self.assertRaises(app.ValidationError):
                app.research_stage(self.source, mutated)

    def test_research_tampering_rejected(self):
        sentiment = app.sentiment_stage(self.source)
        result = app.research_stage(self.source, sentiment)
        result["findings"][0]["topics"][0]["evidence"][0]["quote"] = "invented"
        with self.assertRaises(app.ValidationError):
            app.validate(result, "research", (self.source, sentiment))

    def test_invalid_inputs(self):
        variants = [
            dict(self.source, synthetic=False), dict(self.source, schema_version="2"),
            dict(self.source, issues={}), dict(self.source, extra=True), [],
        ]
        for field, value in [("severity", "urgent"), ("severity", []), ("text", " "),
                             ("document_ids", ["absent"]), ("document_ids", [3]),
                             ("document_ids", ["synthetic-lab", "synthetic-lab"])]:
            item = copy.deepcopy(self.source)
            item["issues"][0][field] = value
            variants.append(item)
        duplicate = copy.deepcopy(self.source)
        duplicate["issues"].append(copy.deepcopy(duplicate["issues"][0]))
        variants.append(duplicate)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(app.ValidationError):
                app.run_pipeline(variant)

    def test_invalid_claims(self):
        for field, value in [("quote", "fabricated"), ("stance", "maybe"), ("topic", "UNCANONICAL")]:
            source = copy.deepcopy(self.source)
            source["documents"][0]["claims"][0][field] = value
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(source)

    def test_opposing_only_evidence(self):
        self.source["issues"][0]["document_ids"] = ["synthetic-field"]
        topics = app.run_pipeline(self.source)["research"]["findings"][0]["topics"]
        self.assertEqual(topics[0]["conclusion"], "opposed")
        self.assertFalse(topics[0]["disagreement"])

    def test_duplicate_documents_and_claims(self):
        duplicate_doc = copy.deepcopy(self.source)
        duplicate_doc["documents"].append(copy.deepcopy(duplicate_doc["documents"][0]))
        duplicate_claim = copy.deepcopy(self.source)
        claims = duplicate_claim["documents"][0]["claims"]
        claims.append(copy.deepcopy(claims[0]))
        for source in (duplicate_doc, duplicate_claim):
            with self.assertRaises(app.ValidationError):
                app.run_pipeline(source)

    def test_no_mutation_and_determinism(self):
        original = copy.deepcopy(self.source)
        self.assertEqual(app.run_pipeline(self.source), app.run_pipeline(self.source))
        self.assertEqual(self.source, original)

    def test_cli_success(self):
        proc = self.cli("example_input.json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), app.run_pipeline(self.source))
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)

    def test_cli_errors(self):
        for args, content in [
            ((), None), (("nonexistent-input.json",), None),
            (("-",), "{"), (("-",), '{"a":1,"a":2}'),
            (("-",), "NaN"), (("-",), '{"synthetic":false}'),
            (("-",), json.dumps(dict(self.source, issues=2))),
        ]:
            with self.subTest(args=args, content=content):
                proc = self.cli(*args, content=content)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")
                self.assertEqual(proc.stderr, "")
                self.assertEqual(len(proc.stdout.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
