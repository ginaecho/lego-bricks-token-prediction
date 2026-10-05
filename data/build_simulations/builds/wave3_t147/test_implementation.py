import copy
import json
import pathlib
import subprocess
import sys
import unittest

import implementation as app


ROOT = pathlib.Path(__file__).resolve().parent


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def telemetry(self, mutation):
        candidate = self.data["candidates"][0]
        value = json.loads(candidate["telemetry_json"])
        mutation(value)
        candidate["telemetry_json"] = json.dumps(value)

    def test_normalization_and_ranking(self):
        result = app.compare(self.data)
        self.assertEqual(result["ranking"][0]["name"], self.data["candidates"][0]["name"])
        attrs = result["products"][0]["attributes"]
        self.assertAlmostEqual(attrs["consumption_kwh"], 3.9)
        self.assertAlmostEqual(attrs["emissions_kgCO2e"], 1.0)
        self.assertEqual(len(result["side_by_side"]), 5)
        self.assertEqual(result, app.compare(copy.deepcopy(self.data)))

    def test_cost_preference(self):
        self.data["preferences"] = {"monthly_cost_usd": 1}
        self.assertIn("Economy", app.compare(self.data)["ranking"][0]["name"])

    def test_protected_identifiers(self):
        output = json.dumps(app.compare(self.data))
        self.assertNotIn("FICT-", output)
        self.assertNotIn(self.data["protection_key"], output)
        self.assertIn("asset_", output)

    def test_public_text_leak_rejected(self):
        self.data["candidates"][1]["name"] = "Package FICT-SUB-ALPHA"
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)

    def test_priority_safety(self):
        report = self.data["candidates"][1]["outage_reports"][0]
        report["life_safety"] = True
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)
        report["priority"] = "emergency"
        self.assertEqual(app.compare(self.data)["products"][1]["attributes"]["outage_risk"], 3)

    def test_declared_rules(self):
        self.data["safety_rules"]["life_safety"] = "routine"
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)

    def test_emissions_source_required(self):
        self.telemetry(lambda v: v["samples"][0]["emissions"].pop("source"))
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)

    def test_emissions_units_required(self):
        self.telemetry(lambda v: v["samples"][0]["emissions"].update(unit="tons"))
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)

    def test_invalid_preferences(self):
        for prefs in ({"unknown": 1}, {"monthly_cost_usd": 0}, {"monthly_cost_usd": True},
                      {"monthly_cost_usd": -1}, {"monthly_cost_usd": float("nan")}):
            with self.subTest(prefs=prefs), self.assertRaises(app.ValidationError):
                self.data["preferences"] = prefs
                app.compare(self.data)

    def test_empty_search(self):
        self.data["query"] = "No such package"
        result = app.compare(self.data)
        self.assertEqual(result["ranking"], [])
        self.assertEqual(result["products"], [])

    def test_single_search(self):
        self.data["query"] = "EFFICIENCY"
        result = app.compare(self.data)
        self.assertEqual(len(result["ranking"]), 1)
        self.assertEqual(result["ranking"][0]["score"], 1)

    def test_equal_scores_alphabetical(self):
        first = self.data["candidates"][0]
        second = copy.deepcopy(first)
        first["name"], second["name"] = "Zulu", "Alpha"
        self.data["candidates"] = [first, second]
        self.assertEqual([r["name"] for r in app.compare(self.data)["ranking"]], ["Alpha", "Zulu"])

    def test_duplicate_meter_interval(self):
        c = self.data["candidates"][0]
        c["meter_csv"] += c["meter_csv"].splitlines()[1] + "\n"
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)

    def test_observation_window_mismatch(self):
        c = self.data["candidates"][1]
        c["meter_csv"] = c["meter_csv"].replace("2026-01-15", "2026-01-16")
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)

    def test_bad_topology(self):
        self.data["candidates"][0]["topology"][0]["to"] = "FICT-MISSING"
        with self.assertRaises(app.ValidationError):
            app.compare(self.data)

    def test_strict_json(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(app.ValidationError):
                app.strict_json(raw)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                               str(ROOT / "example_input.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_errors(self):
        for args in ([], ["not-a-real-input.json"], [str(ROOT / "test_implementation.py")]):
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), *args],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main()
