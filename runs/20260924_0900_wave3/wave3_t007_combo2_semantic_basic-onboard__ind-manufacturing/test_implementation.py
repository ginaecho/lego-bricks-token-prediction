import copy
import csv
import io
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def change_reading(self, **changes):
        reader = csv.DictReader(io.StringIO(self.data["telemetry_csv"]))
        rows = list(reader)
        rows[0].update(changes)
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=app.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        self.data["telemetry_csv"] = out.getvalue()

    def test_integrated_handoff(self):
        output = app.pipeline(self.data)
        hit = output["search"]["hits"][0]["record"]
        self.assertEqual(output["next_step"]["record_id"], hit["record_id"])
        self.assertEqual(output["next_step"]["work_order_id"], hit["work_order_id"])
        self.assertEqual(output["customer"], self.data["customer"])
        self.assertIn(self.data["customer"]["goal"], output["next_step"]["instruction"])

    def test_synonyms_and_index(self):
        self.data["query"] = "shaking"
        result = app.pipeline(self.data)
        self.assertTrue(result["search"]["hits"])
        self.assertTrue(any(h["record"]["kind"] == "sensor" for h in result["search"]["hits"]))
        self.assertTrue(all("vibration" in app.terms(h["record"]["text"])
                            for h in result["search"]["hits"]))

    def test_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(app.pipeline(self.data), app.pipeline(self.data))
        self.assertEqual(before, self.data)

    def test_personalized_roles(self):
        for role, action in (("planner", "review_work_order_schedule"),
                             ("quality_engineer", "review_inspection_evidence"),
                             ("operator", "review_work_order_and_safety_checklist")):
            with self.subTest(role=role):
                self.data["customer"]["role"] = role
                output = app.pipeline(self.data)
                self.assertEqual(output["next_step"]["action"], action)
                self.assertEqual(output["next_step"]["guidance_level"], "guided")
        self.data["customer"]["experience"] = "experienced"
        self.assertEqual(app.pipeline(self.data)["next_step"]["guidance_level"], "concise")

    def test_no_results_and_punctuation(self):
        for query in ("unfindablexyz", "!!!"):
            self.data["query"] = query
            result = app.pipeline(self.data)
            self.assertEqual(result["search"]["hits"], [])
            self.assertEqual(result["next_step"]["action"], "refine_search")

    def test_safety_escalation_not_hidden_by_search(self):
        self.change_reading(value="95")
        self.data["query"] = "unfindablexyz"
        result = app.pipeline(self.data)
        self.assertEqual(result["search"]["hits"], [])
        self.assertEqual(result["next_step"]["action"], "contact_human_safety_supervisor")
        self.assertEqual(result["human_escalations"][0]["route"], "human_safety_supervisor")
        self.assertEqual(result["human_escalations"][0]["evidence"]["value"], 95.0)

    def test_quality_traceability_and_failure(self):
        inspection = self.data["quality_inspections"][0]
        inspection.update(value=25.08, decision="fail", safety_critical=True)
        result = app.pipeline(self.data)
        decision = result["search"]["quality_decisions"][0]
        for key in ("inspector", "timestamp", "rationale", "work_order_id"):
            self.assertEqual(decision[key], inspection[key])
        self.assertEqual(decision["evidence"], inspection)
        self.assertTrue(result["next_step"]["requires_human"])

    def test_quality_requires_actor_and_consistent_decision(self):
        for field, bad in (("inspector", ""), ("rationale", ""), ("decision", "fail"),
                           ("timestamp", "yesterday"), ("unit", "in"), ("lower", 26)):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["quality_inspections"][0][field] = bad
                with self.assertRaises(app.ValidationError):
                    app.pipeline(data)

    def test_units_tolerances_and_finite_values(self):
        for changes in ({"unit": "F"}, {"lower": "90", "upper": "80"},
                        {"value": "nan"}, {"value": "inf"}, {"value": "-274"},
                        {"safety_critical": "yes"}, {"value": "not-a-number"}):
            with self.subTest(changes=changes):
                self.setUp()
                self.change_reading(**changes)
                with self.assertRaises(app.ValidationError):
                    app.pipeline(self.data)

    def test_tolerance_boundaries(self):
        for value in ("20", "80"):
            self.change_reading(value=value)
            self.assertFalse(app.pipeline(self.data)["human_escalations"])

    def test_bad_references_and_duplicates(self):
        for changes in ({"asset_id": "FAKE"}, {"work_order_id": "MISSING"},
                        {"reading_id": "SYN-R002"}):
            self.setUp()
            self.change_reading(**changes)
            with self.assertRaises(app.ValidationError):
                app.pipeline(self.data)
        self.setUp()
        self.data["work_orders"].append(copy.deepcopy(self.data["work_orders"][0]))
        with self.assertRaises(app.ValidationError):
            app.pipeline(self.data)

    def test_csv_shape_and_time_order(self):
        original = self.data["telemetry_csv"]
        for bad in ("wrong,header\n1,2\n", original.replace("64.83,C", "64.83,C,extra"),
                    original.replace("2026-09-24T08:00:00Z", "2026-09-24T09:00:00Z")):
            self.data["telemetry_csv"] = bad
            with self.assertRaises(app.ValidationError):
                app.pipeline(self.data)

    def test_top_limit_and_ties(self):
        self.data["query"] = "vibration"
        self.data["limit"] = 2
        hits = app.pipeline(self.data)["search"]["hits"]
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits, sorted(hits, key=lambda h: (-h["score"], h["record"]["record_id"])))

    def test_injected_embedding(self):
        calls = []

        def embedding(texts):
            calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

        self.data["query"] = "unfindablexyz"
        result = app.pipeline(self.data, embedding)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "unfindablexyz")
        self.assertEqual(result["search"]["hits"][0]["score"], 0.35)

    def test_invalid_embedding_contract(self):
        for callback in (lambda xs: [], lambda xs: [[0, 0]] * len(xs),
                         lambda xs: [[float("nan")]] * len(xs),
                         lambda xs: [[1]] + [[1, 2]] * (len(xs) - 1),
                         lambda xs: [[True]] * len(xs),
                         lambda xs: 1 / 0):
            with self.subTest(callback=callback):
                with self.assertRaises(app.ValidationError):
                    app.pipeline(self.data, callback)

    def test_handoff_tampering_rejected(self):
        source = app.validate(self.data)
        found = app.search(source)
        found["hits"][0]["record"]["work_order_id"] = "FORGED"
        with self.assertRaises(app.ValidationError):
            app.validate(found, "search", source)
        found = app.search(source)
        final = app.onboard(found)
        final["next_step"]["work_order_id"] = "FORGED"
        with self.assertRaises(app.ValidationError):
            app.validate(final, "onboarding", found)

    def test_schema_invalid(self):
        for key, value in (("schema_version", 9), ("schema_version", True),
                           ("synthetic", False), ("query", ""),
                           ("limit", True), ("limit", 0), ("work_orders", []),
                           ("customer", None), ("quality_inspections", {})):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key), self.assertRaises(app.ValidationError):
                app.pipeline(data)

    def test_seeded_physically_plausible_telemetry(self):
        rng = random.Random(42)
        expected = ([round(rng.gauss(65, 1.2), 2) for _ in range(2)] +
                    [round(rng.gauss(2.4, 0.15), 2) for _ in range(2)])
        rows = list(csv.DictReader(io.StringIO(self.data["telemetry_csv"])))
        self.assertEqual([float(row["value"]) for row in rows], expected)
        for _ in range(20):
            self.change_reading(value=str(round(rng.gauss(65, 1.2), 2)))
            self.assertEqual(app.pipeline(self.data)["human_escalations"], [])

    def test_empty_optional_evidence(self):
        self.data["telemetry_csv"] = ",".join(app.CSV_FIELDS) + "\n"
        self.data["quality_inspections"] = []
        result = app.pipeline(self.data)
        self.assertEqual(result["search"]["quality_decisions"], [])
        self.assertEqual(result["search"]["hits"][0]["record"]["kind"], "work_order")

    def test_huge_measurement_and_lost_alert(self):
        self.data["quality_inspections"][0]["value"] = 10 ** 400
        with self.assertRaises(app.ValidationError):
            app.pipeline(self.data)
        self.setUp()
        self.change_reading(value="95")
        source = app.validate(self.data)
        found = app.search(source)
        found["alerts"] = []
        with self.assertRaises(app.ValidationError):
            app.validate(found, "search", source)

    def test_cli_success(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                                 str(ROOT / "example_input.json")],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_errors(self):
        # Reuse deliverables as invalid-schema/invalid-JSON inputs; create no scratch files.
        for args in ([], ["missing.json"], ["build_manifest.json"], ["implementation.py"]):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                    capture_output=True, text=True, cwd=ROOT)
            with self.subTest(args=args):
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
