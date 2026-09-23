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

    def test_two_step_journey_and_available_actions(self):
        journey = self.run_data()["journey"]
        self.assertEqual(["seller-profile", "publish-listing"],
                         [s["action_id"] for s in journey["steps"]])
        self.assertEqual(["seller-profile", "buyer-tour"], journey["next_actions"])

    def test_preference_changes_journey_order(self):
        self.data["profile"]["interests"] = ["buying"]
        self.assertEqual("buyer-tour", self.run_data()["journey"]["steps"][0]["action_id"])

    def test_beginner_onboarding(self):
        steps = self.run_data()["adaptive"]["steps"]
        self.assertEqual("guided", steps[0]["depth"])
        self.assertEqual("exercise", steps[0]["format"])
        self.assertEqual(3, len(steps[0]["instructions"]))
        self.assertIn("beginner", steps[0]["explanation"])

    def test_experienced_preferred_format(self):
        self.data["profile"]["experience"] = "experienced"
        self.data["profile"]["preferred_formats"] = ["video"]
        step = self.run_data()["adaptive"]["steps"][0]
        self.assertEqual(("concise", "video"), (step["depth"], step["format"]))
        self.assertEqual(2, len(step["instructions"]))

    def test_planning_does_not_complete_prerequisites(self):
        result = self.run_data()
        first, second = result["adaptive"]["steps"]
        self.assertEqual("ready_now", first["readiness"])
        self.assertEqual("conditional", second["readiness"])
        self.assertEqual(["seller-profile"], second["requires_completion"])
        self.assertEqual(["orientation"], result["profile"]["completed_actions"])

    def test_review_found_and_gap_trace(self):
        result = self.run_data()
        review = result["review"]
        self.assertEqual("gaps_found", review["state"])
        gap = review["gaps"][0]
        self.assertEqual("returns-policy", gap["requirement_id"])
        self.assertEqual(["publish-listing"], gap["action_ids"])
        self.assertEqual(["onboarding-2"], gap["onboarding_step_ids"])
        self.assertEqual(["journey-2"], gap["journey_step_ids"])
        self.assertEqual("insufficient_terms", gap["reason"])
        found = next(f for f in review["findings"] if f["requirement_id"] == "seller-identity")
        self.assertEqual("evidence_found", found["result"])
        self.assertEqual(2, len(found["onboarding_step_ids"]))
        self.assertEqual("fixture:profile#paragraph-1", found["observations"][0]["locator"])
        self.assertIn("not certification", review["disclaimer"])

    def test_requirements_propagate_without_duplication(self):
        result = self.run_data()
        for journey, adaptive in zip(result["journey"]["steps"], result["adaptive"]["steps"]):
            self.assertEqual(journey["requirement_ids"], adaptive["requirement_ids"])
            self.assertEqual(journey["step_id"], adaptive["source_journey_step_id"])
        self.assertEqual(2, len(result["review"]["findings"]))

    def test_missing_evidence(self):
        self.data["documents"] = []
        self.assertTrue(all(g["reason"] == "missing_evidence"
                            for g in self.run_data()["review"]["gaps"]))

    def test_case_insensitive_whole_word_matching(self):
        self.data["documents"][1]["text"] = "RETURNS and REFUND."
        self.assertEqual("evidence_found", self.run_data()["review"]["state"])
        self.data["documents"][1]["text"] = "Returns refunded"
        self.assertEqual("gaps_found", self.run_data()["review"]["state"])

    def test_terms_must_occur_in_one_assigned_document(self):
        self.data["documents"].append({
            "id": "other-policy", "locator": "fixture:other",
            "requirement_ids": ["returns-policy"], "text": "refund"})
        self.assertEqual("gaps_found", self.run_data()["review"]["state"])
        self.data["documents"][-1]["text"] = "returns refund"
        self.assertEqual("evidence_found", self.run_data()["review"]["state"])

    def test_unassigned_text_is_not_evidence(self):
        self.data["documents"][0]["text"] += " returns refund"
        self.assertEqual("gaps_found", self.run_data()["review"]["state"])

    def test_no_two_step_path_is_explicit(self):
        self.data["profile"]["completed_actions"] = [
            "orientation", "seller-profile", "publish-listing"]
        result = self.run_data()
        self.assertEqual(["buyer-tour"], result["journey"]["next_actions"])
        self.assertEqual("no_two_step_path", result["journey"]["state"])
        self.assertEqual("blocked_no_journey", result["adaptive"]["state"])
        self.assertEqual("not_applicable", result["review"]["state"])

    def test_empty_catalog(self):
        self.data.update(actions=[], requirements=[], documents=[])
        self.data["profile"]["completed_actions"] = []
        self.assertEqual([], self.run_data()["journey"]["steps"])

    def test_empty_interests_has_deterministic_fallback(self):
        self.data["profile"]["interests"] = []
        self.assertEqual(self.run_data(), self.run_data())
        self.assertEqual(2, len(self.run_data()["journey"]["steps"]))

    def test_input_not_mutated(self):
        before = copy.deepcopy(self.data)
        result = self.run_data()
        result["profile"]["interests"].append("new")
        self.assertEqual(before, self.data)

    def test_catalog_order_independent(self):
        expected = self.run_data()
        self.data["actions"].reverse()
        self.data["requirements"].reverse()
        self.data["documents"].reverse()
        actual = self.run_data()
        for stage in ("journey", "adaptive", "review"):
            self.assertEqual(expected[stage], actual[stage])

    def test_unknown_reference(self):
        self.data["actions"][1]["prerequisites"] = ["absent"]
        with self.assertRaisesRegex(app.ValidationError, "unknown reference"):
            self.run_data()

    def test_cycle(self):
        self.data["actions"][0]["prerequisites"] = ["publish-listing"]
        with self.assertRaisesRegex(app.ValidationError, "cycle"):
            self.run_data()

    def test_inconsistent_completion(self):
        self.data["profile"]["completed_actions"] = ["publish-listing"]
        with self.assertRaisesRegex(app.ValidationError, "missing completed"):
            self.run_data()

    def test_duplicate_id(self):
        self.data["actions"].append(copy.deepcopy(self.data["actions"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate id"):
            self.run_data()

    def test_invalid_schema_and_values(self):
        invalid = [
            ("schema_version", True), ("schema_version", 2),
            ("fixture_label", "Real data"), ("actions", {}),
            ("profile", None), ("documents", [None]),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_invalid_formats_and_terms(self):
        for change in ("empty_formats", "bad_format", "bad_term", "duplicate_term"):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                if change == "empty_formats":
                    data["profile"]["preferred_formats"] = []
                elif change == "bad_format":
                    data["profile"]["preferred_formats"] = ["hologram"]
                else:
                    data["requirements"][0]["required_terms"] = (
                        ["two words"] if change == "bad_term" else ["seller", "SELLER"])
                with self.assertRaises(app.ValidationError):
                    app.run_pipeline(data)

    def test_tampered_journey_rejected_before_onboarding(self):
        handoff = app.advance(self.data, "journey")
        handoff["journey"]["steps"][0]["action_id"] = "publish-listing"
        with self.assertRaisesRegex(app.ValidationError, "journey"):
            app.advance(handoff, "adaptive")

    def test_tampered_onboarding_rejected_before_review(self):
        handoff = app.advance(app.advance(self.data, "journey"), "adaptive")
        handoff["adaptive"]["review_requirement_ids"] = []
        with self.assertRaisesRegex(app.ValidationError, "adaptive"):
            app.advance(handoff, "review")

    def test_stale_profile_invalidates_handoff(self):
        handoff = app.advance(app.advance(self.data, "journey"), "adaptive")
        handoff["profile"]["experience"] = "experienced"
        with self.assertRaisesRegex(app.ValidationError, "adaptive"):
            app.advance(handoff, "review")

    def test_stage_skipping_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.advance(self.data, "review")

    def test_final_output_validated(self):
        result = self.run_data()
        self.assertIs(result, app.validate(result, "review"))
        result["review"]["gaps"] = []
        with self.assertRaisesRegex(app.ValidationError, "review"):
            app.validate(result, "review")


class CLITests(unittest.TestCase):
    def cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual("", result.stderr)
        self.assertEqual(1, len(result.stdout.splitlines()))
        return result.returncode, json.loads(result.stdout)

    def test_success(self):
        code, output = self.cli("example_input.json")
        self.assertEqual(0, code)
        self.assertEqual("ok", output["status"])
        self.assertEqual("gaps_found", output["review"]["state"])

    def test_missing_file(self):
        code, output = self.cli("nonexistent-input.json")
        self.assertEqual(2, code)
        self.assertEqual("error", output["status"])

    def test_usage(self):
        for args in ((), ("example_input.json", "extra")):
            with self.subTest(args=args):
                code, output = self.cli(*args)
                self.assertEqual(2, code)
                self.assertEqual("error", output["status"])

    def test_invalid_file_contents(self):
        path = ROOT / "_invalid_fixture.json"
        try:
            for contents in (
                b"{broken", b'{"schema_version": 1, "schema_version": 2}',
                b'{"schema_version": NaN}', b"null", b"\xff", b"[]",
            ):
                with self.subTest(contents=contents):
                    path.write_bytes(contents)
                    code, output = self.cli(str(path))
                    self.assertEqual(2, code)
                    self.assertEqual("error", output["status"])
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
