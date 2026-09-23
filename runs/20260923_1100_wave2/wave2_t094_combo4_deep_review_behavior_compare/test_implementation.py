"""Standard-library tests; all input data is explicitly synthetic."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_data(self):
        return impl.run_pipeline(self.data)["stages"]

    def row(self, stages, stage, pid):
        return next(row for row in stages[stage]["records"] if row["product_id"] == pid)

    def finding(self, stages, pid, rid):
        return next(row for row in self.row(stages, "research", pid)["findings"]
                    if row["requirement_id"] == rid)

    def test_integrated_example_ranking(self):
        stages = self.run_data()
        self.assertEqual(list(stages), list(impl.STAGES))
        self.assertEqual([row["product_id"] for row in stages["compare"]["records"]], ["b", "a"])
        self.assertAlmostEqual(self.row(stages, "compare", "b")["score"], 0.8)

    def test_multi_document_synthesis(self):
        finding = self.finding(self.run_data(), "a", "outdoor")
        self.assertEqual(finding["state"], "supported")
        self.assertEqual([e["document_id"] for e in finding["evidence"]], ["manual", "report"])
        self.assertIsNone(finding["unresolved_question"])

    def test_disagreement_preserves_opposing_evidence(self):
        finding = self.finding(self.run_data(), "c", "outdoor")
        self.assertEqual(finding["state"], "disputed")
        self.assertEqual({e["stance"] for e in finding["evidence"]}, {"supports", "opposes"})
        self.assertIn("outdoor", finding["unresolved_question"])

    def test_uncertainty_does_not_become_support(self):
        self.data["documents"][1]["claims"][0]["stance"] = "uncertain"
        stages = self.run_data()
        self.assertEqual(self.finding(stages, "a", "outdoor")["state"], "unresolved")
        self.assertFalse(self.row(stages, "review", "a")["eligible"])

    def test_opposition_without_support(self):
        self.data["documents"][0]["claims"][3]["stance"] = "opposes"
        stages = self.run_data()
        self.assertEqual(self.finding(stages, "c", "outdoor")["state"], "contradicted")

    def test_missing_optional_evidence_is_traceable(self):
        stages = self.run_data()
        self.assertEqual(self.finding(stages, "b", "repair")["state"], "missing")
        self.assertTrue(self.row(stages, "review", "b")["eligible"])
        for stage, key in (("review", "gaps"), ("behavior", "review_gaps"),
                           ("compare", "review_gaps")):
            self.assertEqual(self.row(stages, stage, "b")[key], ["repair"])

    def test_review_is_not_certification(self):
        self.assertIn("not certification", self.run_data()["review"]["warnings"][0])

    def test_no_documents_no_candidates(self):
        self.data["documents"] = []
        stages = self.run_data()
        self.assertEqual(stages["compare"]["records"], [])
        self.assertTrue(all(not row["eligible"] for row in stages["review"]["records"]))

    def test_purchase_and_half_life(self):
        stages = self.run_data()
        self.assertEqual(self.row(stages, "behavior", "b")["raw_signal"], 1.5)
        self.assertEqual(self.row(stages, "behavior", "a")["raw_signal"], 1.0)
        self.assertAlmostEqual(self.row(stages, "behavior", "a")["score"], 2 / 3)

    def test_history_cannot_override_review_exclusion(self):
        stages = self.run_data()
        excluded = self.row(stages, "behavior", "c")
        self.assertEqual(excluded["mode"], "excluded")
        self.assertEqual(excluded["score"], 0)
        self.assertNotIn("c", [row["product_id"] for row in stages["compare"]["records"]])

    def test_cold_start_uses_category(self):
        self.data["events"] = []
        stages = self.run_data()
        self.assertEqual(self.row(stages, "behavior", "a")["mode"], "cold_start")
        self.assertEqual(self.row(stages, "behavior", "a")["score"], 1)
        self.assertEqual(self.row(stages, "behavior", "b")["score"], 0)

    def test_excluded_only_history_is_cold_start(self):
        self.data["events"] = [self.data["events"][2]]
        self.assertEqual(self.row(self.run_data(), "behavior", "a")["mode"], "cold_start")

    def test_neutral_cold_start_ties_use_id(self):
        self.data["events"] = []
        self.data["preferences"]["preferred_categories"] = []
        self.data["preferences"]["weights"] = {"price": 0, "weight": 0, "runtime": 0, "behavior": 1}
        rows = self.run_data()["compare"]["records"]
        self.assertEqual([row["product_id"] for row in rows], ["a", "b"])
        self.assertEqual([row["rank"] for row in rows], [1, 2])

    def test_attribute_unit_normalization(self):
        row = self.row(self.run_data(), "compare", "a")
        self.assertEqual(row["attributes"], {"price": 60.0, "weight": 300.0, "runtime": 8.0})
        self.assertEqual(row["units"], {"price": "USD", "weight": "g", "runtime": "h"})

    def test_missing_attribute_has_zero_utility(self):
        self.data["products"][0]["attributes"]["runtime"] = None
        self.assertEqual(self.row(self.run_data(), "compare", "a")["utilities"]["runtime"], 0)

    def test_equal_known_attributes_have_equal_utility(self):
        self.data["products"][1]["attributes"]["price"]["value"] = 60
        rows = self.run_data()["compare"]["records"]
        self.assertEqual([row["utilities"]["price"] for row in rows], [1.0, 1.0])

    def test_budget_filter_and_missing_price(self):
        self.data["preferences"]["max_price_usd"] = 50
        self.assertEqual([row["product_id"] for row in self.run_data()["compare"]["records"]], ["b"])
        self.data["products"][1]["attributes"]["price"] = None
        self.assertEqual(self.run_data()["compare"]["records"], [])

    def test_no_budget_allows_missing_price(self):
        self.data["preferences"]["max_price_usd"] = None
        self.data["products"][0]["attributes"]["price"] = None
        self.assertEqual(self.row(self.run_data(), "compare", "a")["utilities"]["price"], 0)

    def test_provenance_survives_all_stages(self):
        stages = self.run_data()
        original = self.finding(stages, "a", "outdoor")["evidence"]
        for stage, key in (("review", "checks"), ("behavior", "review_checks"),
                           ("compare", "review_checks")):
            self.assertEqual(self.row(stages, stage, "a")[key][0]["evidence"], original)

    def test_new_conflict_propagates_to_final_candidates(self):
        self.data["documents"][1]["claims"][0]["stance"] = "opposes"
        stages = self.run_data()
        self.assertFalse(self.row(stages, "review", "a")["eligible"])
        self.assertEqual(self.row(stages, "behavior", "a")["mode"], "excluded")
        self.assertEqual([row["product_id"] for row in stages["compare"]["records"]], ["b"])

    def test_input_not_mutated_and_deterministic(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(self.data, original)

    def test_validation_rejects_forged_citation(self):
        stages = self.run_data()
        stages["research"]["records"][0]["findings"][0]["evidence"][0]["quote"] = "invented"
        with self.assertRaises(impl.ValidationError):
            impl.validate_artifact(stages["research"], "research", self.data)

    def test_validation_rejects_lost_gap_and_invalid_score(self):
        stages = self.run_data()
        stages["behavior"]["records"][1]["review_gaps"] = []
        with self.assertRaises(impl.ValidationError):
            impl.validate_artifact(stages["behavior"], "behavior", self.data, stages["review"])
        stages = self.run_data()
        stages["compare"]["records"][0]["score"] = float("nan")
        with self.assertRaises(impl.ValidationError):
            impl.validate_artifact(stages["compare"], "compare", self.data, stages["behavior"])

    def test_pipeline_validates_before_next_stage(self):
        # Altering the builder result, without changing the trusted validator's
        # implementation, proves that the next stage is never called.
        real_validate = impl.validate_artifact

        def validate(value, stage, data, previous=None):
            if stage == "review":
                value["records"][0]["eligible"] = False
            return real_validate(value, stage, data, previous)

        with patch.object(impl, "validate_artifact", side_effect=validate):
            with patch.object(impl, "behavior") as downstream:
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(self.data)
                downstream.assert_not_called()

    def test_invalid_inputs(self):
        mutations = [
            lambda d: d.update(schema_version="2"),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(products=[]),
            lambda d: d.update(requirements=[]),
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["events"][0].update(product_id="unknown"),
            lambda d: d["events"][0].update(type="click"),
            lambda d: d["events"][0].update(at="2027-01-01T00:00:00Z"),
            lambda d: d["documents"][0]["claims"][0].update(requirement_id="unknown"),
            lambda d: d["documents"][0]["claims"][0].update(quote=" "),
            lambda d: d["products"][0]["attributes"]["price"].update(unit="EUR"),
            lambda d: d["products"][0]["attributes"]["weight"].update(value=-1),
            lambda d: d["products"][0]["attributes"]["runtime"].update(value=True),
            lambda d: d["products"][0]["attributes"]["runtime"].update(value=float("inf")),
            lambda d: d["preferences"].update(half_life_days=0),
            lambda d: d["preferences"].update(weights=dict.fromkeys(("price", "weight", "runtime", "behavior"), 0)),
            lambda d: d.update(as_of="2026-09-23"),
            lambda d: d["requirements"][0].update(mandatory="true"),
            lambda d: d.update(unexpected=1),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                data = copy.deepcopy(self.data)
                mutation(data)
                with self.assertRaises(impl.ValidationError):
                    impl.run_pipeline(data)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_success_one_json_object(self):
        result = self.cli(str(ROOT / "example_input.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_cli_missing_file(self):
        result = self.cli(str(ROOT / "does-not-exist.json"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_argument(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_invalid_json_and_validation_errors(self):
        for content in ("{", '{"x":1,"x":2}', '{"x":NaN}', "[]",
                        json.dumps({**self.data, "synthetic": False})):
            with self.subTest(content=content[:30]):
                with patch("builtins.open", mock_open(read_data=content)):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = impl.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_non_utf8_file_error(self):
        with patch("builtins.open", side_effect=UnicodeError("invalid UTF-8")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = impl.main(["fixture.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
