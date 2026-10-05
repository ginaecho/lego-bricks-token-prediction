import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def test_multidocument_synthesis_and_disagreement(self):
        research = app.synthesize(self.data)["research"]
        findings = {f["topic"]: f for f in research["findings"]}
        self.assertEqual(findings["form"]["source_count"], 2)
        self.assertAlmostEqual(findings["form"]["support_weight"], 1.5)
        self.assertEqual(findings["sound"]["assessment"], "disputed")
        self.assertEqual(len(research["disagreements"]), 1)
        self.assertEqual(research["disagreements"][0]["opposing_document_ids"],
                         ["synthetic-field"])
        self.assertEqual({q["reason"] for q in research["unresolved_questions"]},
                         {"disputed", "no_evidence"})

    def test_personalized_ranking_and_cross_stage_provenance(self):
        output = app.run(self.data)
        top = output["recommendations"][0]
        self.assertEqual(top["product_id"], "synthetic-a")
        quiet = next(r for r in top["reasons"] if r["topic"] == "sound")
        self.assertEqual(quiet["assessment"], "disputed")
        self.assertEqual(quiet["document_ids"], ["synthetic-field", "synthetic-lab"])
        self.assertEqual({q["reason"] for q in top["unresolved_questions"]},
                         {"disputed", "no_evidence"})
        self.assertAlmostEqual(top["score"], (1 + .8 * .75 + .3 * .75) / 2.1)
        self.assertIs(app.validate(output, "output"), output)

    def test_evidence_changes_recommendation_score(self):
        before = app.run(self.data)["recommendations"][0]["score"]
        self.data["documents"][1]["claims"][0]["stance"] = "support"
        after = app.run(self.data)
        self.assertGreater(after["recommendations"][0]["score"], before)
        self.assertEqual(after["research"]["disagreements"], [])

    def test_tampered_handoff_and_output_rejected(self):
        state = app.synthesize(self.data)
        state["research"]["findings"][0]["support_weight"] = 100
        with self.assertRaises(app.ValidationError):
            app.recommend(state)
        output = app.run(self.data)
        output["recommendations"][0]["score"] = 1
        with self.assertRaises(app.ValidationError):
            app.validate(output, "output")
        with self.assertRaises(app.ValidationError):
            app.recommend(self.data)

    def test_immutability_and_determinism(self):
        original = copy.deepcopy(self.data)
        first = app.run(self.data)
        self.assertEqual(original, self.data)
        self.assertEqual(first, app.run(self.data))
        self.data["products"].reverse()
        self.data["documents"].reverse()
        second = app.run(self.data)
        self.assertEqual(first["research"], second["research"])
        self.assertEqual(first["recommendations"], second["recommendations"])

    def test_no_documents_preserves_uncertainty(self):
        self.data["documents"] = []
        output = app.run(self.data)
        self.assertEqual(output["research"]["findings"], [])
        self.assertEqual(len(output["research"]["unresolved_questions"]), 3)
        self.assertEqual(output["recommendations"][0]["score"], .75)
        self.assertTrue(all(r["assessment"] == "no_evidence"
                            for r in output["recommendations"][0]["reasons"]))

    def test_empty_catalog_and_all_excluded(self):
        self.data["products"] = []
        self.assertEqual(app.run(self.data)["recommendations"], [])
        self.setUp()
        self.data["profile"]["exclude_product_ids"] = [p["id"] for p in self.data["products"]]
        self.assertEqual(app.run(self.data)["recommendations"], [])

    def test_cold_start_and_stable_ties(self):
        self.data["profile"]["preferences"] = []
        self.data["documents"] = []
        for product in self.data["products"]:
            product["attributes"] = {}
        self.data["products"].reverse()
        output = app.run(self.data)
        self.assertEqual([r["product_id"] for r in output["recommendations"]],
                         ["synthetic-a", "synthetic-b", "synthetic-c"])
        self.assertTrue(all(r["mode"] == "cold_start" and r["score"] == 0
                            for r in output["recommendations"]))
        self.setUp()
        self.data["profile"]["preferences"] = []
        self.assertGreater(app.run(self.data)["recommendations"][0]["score"], 0)

    def test_exclusion_and_limit(self):
        self.data["profile"]["exclude_product_ids"] = ["synthetic-a"]
        self.data["limit"] = 1
        output = app.run(self.data)
        self.assertEqual(len(output["recommendations"]), 1)
        self.assertEqual(output["recommendations"][0]["product_id"], "synthetic-c")

    def test_zero_weight_source_is_unresolved(self):
        for doc in self.data["documents"]:
            doc["reliability"] = 0
        output = app.run(self.data)
        self.assertTrue(all(f["assessment"] == "unresolved"
                            for f in output["research"]["findings"]))
        self.assertEqual(output["research"]["disagreements"], [])

    def test_invalid_types_ranges_fields_and_duplicates(self):
        mutations = [
            lambda d: d.update(limit=True),
            lambda d: d.update(schema_version=True),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(extra=1),
            lambda d: d.update(documents={}),
            lambda d: d["documents"][0].update(reliability=float("nan")),
            lambda d: d["documents"][0].update(reliability=10 ** 1000),
            lambda d: d["documents"][0].update(reliability=1.1),
            lambda d: d["documents"][0]["claims"][0].update(confidence=True),
            lambda d: d["documents"][0]["claims"][0].update(stance="maybe"),
            lambda d: d["documents"][0]["claims"][0].update(quote="invented citation"),
            lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["documents"][0]["claims"].append(copy.deepcopy(d["documents"][0]["claims"][0])),
            lambda d: d["profile"]["preferences"][0].update(weight=0),
            lambda d: d["profile"]["preferences"].append(copy.deepcopy(d["profile"]["preferences"][0])),
            lambda d: d["profile"].update(exclude_product_ids=["unknown"]),
            lambda d: d["products"][0].update(attributes={"form": ""}),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                mutation(data)
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def cli(self, *args):
        result = subprocess.run([sys.executable, "-B", str(HERE / "implementation.py"), *args],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.stderr, "")
        return result.returncode, json.loads(result.stdout)

    def test_cli_success(self):
        code, payload = self.cli(str(HERE / "example_input.json"))
        self.assertEqual(code, 0)
        self.assertEqual(payload, app.run(self.data))

    def test_cli_file_error_and_usage(self):
        for args in ((), ("missing-fixture.json",), ("example_input.json", "extra")):
            with self.subTest(args=args):
                code, payload = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(payload["status"], "error")

    def test_cli_bad_json_and_validation_without_extra_files(self):
        oversized_number = copy.deepcopy(self.data)
        oversized_number["documents"][0]["reliability"] = 10 ** 1000
        for contents in ("{", "null", '{"x":1,"x":2}', '{"x":NaN}',
                         '{"schema_version": 1}', '[]', json.dumps(oversized_number)):
            with self.subTest(contents=contents):
                out = io.StringIO()
                with patch("builtins.open", mock_open(read_data=contents)):
                    with contextlib.redirect_stdout(out):
                        code = app.main(["synthetic-memory-fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
