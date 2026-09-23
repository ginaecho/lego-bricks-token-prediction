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


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normalization_and_side_by_side(self):
        comparison = app.compare(self.data)["comparison"]
        self.assertAlmostEqual(comparison["side_by_side"][0]["values"]["a"], 0.8)
        self.assertEqual(comparison["normalized_products"][0]["attributes"]["material"], "aluminum")

    def test_weighted_ranking(self):
        ranked = app.run(self.data)["comparison"]["ranking"]
        self.assertEqual([r["product_id"] for r in ranked], ["a", "b"])
        self.assertEqual([r["score"] for r in ranked], [0.75, 0.25])

    def test_review_trace_and_gaps(self):
        review = app.run(self.data)["review"]
        self.assertEqual(review["gap_count"], 2)
        portable = review["checks"][0]
        self.assertEqual(portable["status"], "met")
        self.assertEqual(portable["evidence"][0]["document_id"], "synthetic-sheet")
        self.assertEqual(review["checks"][-1]["gaps"], ["missing_supporting_evidence"])
        self.assertIn("no certification", review["notice"])

    def test_preference_change_propagates(self):
        self.data["preferences"][1]["weight"] = 10
        result = app.run(self.data)
        self.assertEqual(result["comparison"]["winner_id"], "b")
        first = result["review"]["checks"][0]
        self.assertEqual(first["product_id"], "b")
        self.assertEqual(first["comparison_rank"], 1)
        self.assertIn("requirement_not_met", first["gaps"])
        self.assertIn("missing_supporting_evidence", first["gaps"])

    def test_handoff_tampering_rejected(self):
        result = app.compare(self.data)
        result["comparison"]["normalized_products"][0]["attributes"]["weight"] = 0
        with self.assertRaises(app.ValidationError):
            app.review(result)

    def test_review_output_tampering_rejected(self):
        result = app.run(self.data)
        result["review"]["gap_count"] = 0
        with self.assertRaises(app.ValidationError):
            app.validate_envelope(result, "review")

    def test_missing_attribute(self):
        del self.data["products"][0]["attributes"]["weight"]
        self.data["requirements"][0]["target"] = "all"
        result = app.run(self.data)
        a = next(r for r in result["comparison"]["ranking"] if r["product_id"] == "a")
        self.assertEqual(a["contributions"]["weight"], 0)
        self.assertIn("missing_attribute", result["review"]["checks"][0]["gaps"])

    def test_tie_is_stable(self):
        self.data["products"][1]["attributes"] = copy.deepcopy(self.data["products"][0]["attributes"])
        self.data["products"].reverse()
        ranking = app.run(self.data)["comparison"]["ranking"]
        self.assertEqual([r["product_id"] for r in ranking], ["a", "b"])
        self.assertEqual(ranking[0]["score"], 1)

    def test_conflicting_evidence(self):
        self.data["evidence"][0]["asserted"]["value"] = 900
        gaps = app.run(self.data)["review"]["checks"][0]["gaps"]
        self.assertEqual(gaps, ["conflicting_evidence", "missing_supporting_evidence"])

    def test_invalid_inputs(self):
        changes = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(products=[]),
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["products"][0]["attributes"]["weight"].update(value=True),
            lambda d: d["products"][0]["attributes"]["weight"].update(value=float("nan")),
            lambda d: d["products"][0]["attributes"]["weight"].update(unit="m"),
            lambda d: d["preferences"][0].update(weight=0),
            lambda d: d["requirements"][0].update(attribute="unknown"),
            lambda d: d["evidence"][0].update(document_id="unknown"),
            lambda d: d["evidence"][0].update(quote="not in document"),
        ]
        for change in changes:
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(app.ValidationError):
                    app.run(data)

    def test_max_preference_and_min_requirement(self):
        self.data["preferences"][0]["direction"] = "max"
        self.data["requirements"][0]["operator"] = "min"
        self.data["requirements"][0]["evidence_required"] = False
        result = app.run(self.data)
        self.assertEqual(result["comparison"]["winner_id"], "b")
        self.assertEqual(result["review"]["checks"][0]["status"], "met")

    def test_single_product_and_empty_evidence(self):
        self.data["products"] = self.data["products"][:1]
        self.data["documents"] = []
        self.data["evidence"] = []
        result = app.run(self.data)
        self.assertEqual(result["comparison"]["ranking"][0]["score"], 1)
        self.assertEqual(result["review"]["gap_count"], 2)

    def test_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.run(self.data), app.run(self.data))
        self.assertEqual(before, self.data)

    def test_large_finite_numbers(self):
        self.data["products"][0]["attributes"]["price"]["value"] = -1e308
        self.data["products"][1]["attributes"]["price"]["value"] = 1e308
        result = app.run(self.data)
        self.assertEqual(result["comparison"]["ranking"][0]["score"], 1)
        json.dumps(result, allow_nan=False)

    def test_cli_success(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_file_and_argument_errors(self):
        for args in ([], [str(ROOT / "does-not-exist.json")]):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                    capture_output=True, text=True, cwd=ROOT, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_cli_json_and_validation_errors(self):
        for raw in ('{', '{"a": 1, "a": 2}', '{"schema_version": 99}', 'null'):
            with self.subTest(raw=raw):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=raw)), contextlib.redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
