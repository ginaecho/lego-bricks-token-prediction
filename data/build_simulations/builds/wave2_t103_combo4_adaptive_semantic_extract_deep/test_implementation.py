import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def stages(self):
        return app.run_pipeline(self.data)["stages"]

    def test_full_pipeline(self):
        result = app.run_pipeline(self.data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["stages"]), ["adaptive", "semantic", "extract", "deep"])

    def test_novice_prerequisites_order(self):
        onboarding = app.adaptive(self.data)
        self.assertEqual(onboarding["steps"][:2], ["search_basics", "source_verification"])
        self.assertTrue(onboarding["ready"])

    def test_missing_prerequisite_blocks_search(self):
        self.data["profile"]["completed_prerequisites"] = []
        onboarding = app.adaptive(self.data)
        self.assertFalse(onboarding["ready"])
        with self.assertRaisesRegex(app.ValidationError, "prerequisites unmet"):
            app.run_pipeline(self.data)

    def test_expert_skips_basics(self):
        self.data["profile"]["experience"] = "expert"
        self.data["profile"]["completed_prerequisites"] = ["source_verification"]
        self.assertEqual(self.stages()["adaptive"]["required_prerequisites"], ["source_verification"])

    def test_explanation_preference(self):
        long = app.adaptive(self.data)["explanation"]
        self.data["profile"]["explanation_detail"] = "brief"
        self.assertLess(len(app.adaptive(self.data)["explanation"]), len(long))

    def test_preference_handoff_and_weighting(self):
        baseline = self.stages()["semantic"]
        self.data["profile"]["focus_terms"] = ["catalog"]
        changed = self.stages()["semantic"]
        self.assertEqual(changed["focus_terms"], ["catalog"])
        self.assertEqual(changed["results"][0]["document_id"], "catalog")
        self.assertNotEqual(baseline["results"], changed["results"])

    def test_index_ranking_and_tie_break(self):
        search = self.stages()["semantic"]
        self.assertEqual(search["index"]["solar"], ["catalog", "lab"])
        self.assertEqual([r["document_id"] for r in search["results"]], ["catalog", "lab"])
        self.assertEqual(search["results"][0]["score"], search["results"][1]["score"])

    def test_top_k_propagates_to_research(self):
        self.data["options"]["top_k"] = 1
        stages = self.stages()
        self.assertEqual(len(stages["extract"]["documents"]), 1)
        self.assertEqual(stages["deep"]["documents_considered"], ["catalog"])
        self.assertEqual(stages["deep"]["disagreements"], [])

    def test_source_spans_and_types(self):
        stages = self.stages()
        sources = {d["id"]: d["text"] for d in self.data["documents"]}
        for doc in stages["extract"]["documents"]:
            for entries in doc["fields"].values():
                for entry in entries:
                    start, end = entry["span"]
                    self.assertEqual(sources[doc["document_id"]][start:end], entry["raw"])
        fields = stages["extract"]["documents"][0]["fields"]
        self.assertEqual(fields["battery_hours"][0]["value"], 12.0)
        self.assertIs(fields["waterproof"][0]["value"], True)

    def test_missing_fields_reported(self):
        docs = self.stages()["extract"]["documents"]
        self.assertEqual(docs[1]["missing_fields"], ["price_usd"])

    def test_invalid_values_remain_auditable(self):
        self.data["documents"][0]["text"] = self.data["documents"][0]["text"].replace("Battery hours: 12", "Battery hours: twelve")
        stages = self.stages()
        doc = stages["extract"]["documents"][0]
        self.assertIn("battery_hours", doc["missing_fields"])
        self.assertEqual(doc["invalid_fields"][0]["raw"], "twelve")
        self.assertTrue(any("invalid battery_hours" in q for q in stages["deep"]["unresolved_questions"]))

    def test_repeated_values_and_internal_disagreement(self):
        self.data["options"]["top_k"] = 1
        self.data["documents"][0]["text"] += "Runtime hours: 10\nBattery hours: 12\n"
        research = self.stages()["deep"]
        conflict = next(d for d in research["disagreements"] if d["field"] == "battery_hours")
        twelve = next(v for v in conflict["variants"] if v["value"] == 12)
        self.assertEqual(len(twelve["evidence"]), 2)
        self.assertEqual(twelve["supporting_documents"], ["catalog"])

    def test_disagreement_without_arbitrary_winner(self):
        research = self.stages()["deep"]
        conflicts = {d["field"] for d in research["disagreements"]}
        self.assertEqual(conflicts, {"battery_hours", "waterproof", "reviewed"})
        product = next(f for f in research["findings"] if f["field"] == "product")
        self.assertEqual(product["state"], "supported")
        self.assertEqual(len(product["variants"][0]["supporting_documents"]), 2)

    def test_no_results(self):
        self.data["query"] = "zyxwvu"
        self.data["profile"]["focus_terms"] = []
        stages = self.stages()
        self.assertEqual(stages["semantic"]["results"], [])
        self.assertEqual(stages["deep"]["documents_considered"], [])
        self.assertTrue(stages["deep"]["unresolved_questions"])

    def test_empty_corpus(self):
        self.data["documents"] = []
        self.assertEqual(self.stages()["extract"]["documents"], [])

    def test_embedding_injection(self):
        calls = []
        def embedding(texts):
            calls.append(texts)
            return [[1, 0], [0, 1], [1, 0], [-1, 0]]
        stages = app.run_pipeline(self.data, embedding)["stages"]
        self.assertEqual(stages["semantic"]["mode"], "hybrid")
        self.assertEqual(stages["semantic"]["results"][0]["document_id"], "lab")
        self.assertIn("battery portable", calls[0][0])
        self.assertEqual(len(calls), 1)

    def test_invalid_embeddings(self):
        for vectors in ([], [[1]] * 3, [[0, 0]] * 4, [[float("nan")]] * 4,
                        [[True]] * 4, [[1], [1, 2], [1], [1]], "bad"):
            with self.subTest(vectors=vectors), self.assertRaises(app.ValidationError):
                app.run_pipeline(self.data, lambda texts: vectors)

    def test_embedding_exception(self):
        def broken(texts):
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run_pipeline(self.data, broken)

    def test_invalid_inputs(self):
        cases = [
            ("schema_version", "2"), ("synthetic", False), ("query", "..."),
            ("documents", {}), ("fields", []), ("profile", []),
            ("options", {"top_k": True}), ("options", {"min_score": float("inf")}),
            ("options", {"unknown": 1}),
        ]
        for key, value in cases:
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(app.ValidationError):
                app.run_pipeline(data)

    def test_duplicate_ids_and_aliases(self):
        self.data["documents"].append(copy.deepcopy(self.data["documents"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate document"):
            self.stages()
        self.data["documents"].pop()
        self.data["fields"][1]["aliases"].append("product")
        with self.assertRaisesRegex(app.ValidationError, "duplicate alias"):
            self.stages()

    def test_tampered_handoffs_rejected(self):
        onboarding = app.adaptive(self.data)
        search = app.semantic(self.data, onboarding)
        search["results"][0]["document_id"] = "nonexistent"
        with self.assertRaises(app.ValidationError):
            app.extract(self.data, search, onboarding)
        search = app.semantic(self.data, onboarding)
        extracted = app.extract(self.data, search, onboarding)
        extracted["documents"][0]["fields"]["product"][0]["span"] = [0, 4]
        with self.assertRaises(app.ValidationError):
            app.deep(self.data, extracted, search, onboarding)

    def test_unicode_crlf_whitespace_spans(self):
        self.data["documents"][0]["text"] = "Solar lantern battery portable café\r\n  Product:  Lümä  \r\nBattery hours: 12\r\nPrice USD: 4\r\n"
        doc = self.stages()["extract"]["documents"][0]
        entry = doc["fields"]["product"][0]
        self.assertEqual(entry["raw"], "Lümä")
        start, end = entry["span"]
        self.assertEqual(self.data["documents"][0]["text"][start:end], "Lümä")

    def test_malformed_and_type_confused_handoffs(self):
        onboarding = app.adaptive(self.data)
        search = app.semantic(self.data, onboarding)
        extracted = app.extract(self.data, search, onboarding)
        extracted["documents"].append(None)
        with self.assertRaises(app.ValidationError):
            app.deep(self.data, extracted, search, onboarding)
        extracted = app.extract(self.data, search, onboarding)
        extracted["documents"][0]["fields"]["waterproof"][0]["value"] = 1
        with self.assertRaisesRegex(app.ValidationError, "typed evidence"):
            app.deep(self.data, extracted, search, onboarding)

    def test_conversion_edge_cases(self):
        for raw, kind in [("NaN", "number"), ("1e999", "number"), ("2026-02-30", "date"),
                          ("20260901", "date"), ("maybe", "boolean"), ("", "string")]:
            with self.subTest(raw=raw), self.assertRaises(app.ValidationError):
                app.convert(raw, kind)
        self.assertEqual(app.convert("-1.2e2", "number"), -120)

    def test_deterministic_and_nonmutating(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(self.stages(), self.stages())
        self.assertEqual(before, self.data)

    def test_cli_success(self):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                    str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)

    def test_cli_missing_file_and_usage(self):
        for args in ([], ["does-not-exist.json"], ["one", "two"]):
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                       capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")

    def test_cli_invalid_json_and_schema(self):
        for payload in ("{", '{"x": NaN}', '{"x":1,"x":2}', "[]",
                        json.dumps({**self.data, "query": ""})):
            output = io.StringIO()
            with patch.object(Path, "open", return_value=io.StringIO(payload)), contextlib.redirect_stdout(output):
                code = app.main(["fixture.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
