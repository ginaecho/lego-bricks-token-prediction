import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def test_normal_accountable_delivery(self):
        result = app.triage(self.data)
        self.assertEqual(result["category"], "delivery")
        self.assertEqual(result["priority"], "high")
        self.assertEqual(result["route"]["owner"], "synthetic-dispatch-lead")
        self.assertEqual(result["products"][0]["price"], "14.50")

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        self.assertEqual(app.triage(self.data), app.triage(self.data))
        self.assertEqual(self.data, original)

    def test_first_rule_wins_and_urgent_overrides(self):
        self.data["ticket"]["body"] = "Refund needed; UNSAFE packaging"
        result = app.triage(self.data)
        self.assertEqual(result["category"], "delivery")
        self.assertEqual(result["priority"], "urgent")

    def test_fallback_empty_rules(self):
        self.data["config"]["rules"] = []
        self.data["clickstream"] = []
        self.data["ticket"]["skus"] = []
        result = app.triage(self.data)
        self.assertEqual(result["category"], "general")
        self.assertEqual(result["products"], [])

    def test_configurable_routing(self):
        self.data["config"]["rules"][0]["owner"] = "synthetic-replacement-owner"
        self.assertEqual(app.triage(self.data)["route"]["owner"], "synthetic-replacement-owner")

    def test_each_consent_required(self):
        for key in self.data["customer"]["consent"]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["customer"]["consent"][key] = False
                with self.assertRaises(app.ValidationError):
                    app.triage(data)

    def test_no_personalization_without_request(self):
        self.data["customer"]["consent"] = {"gdpr_personalization": False, "ccpa_personalization": False}
        self.data["ticket"]["personalize"] = False
        self.assertEqual(app.triage(self.data)["personalization"], {"applied": False, "recent_skus": []})

    def test_no_endorsements(self):
        self.data["ticket"]["request_endorsement"] = True
        with self.assertRaises(app.ValidationError):
            app.triage(self.data)

    def test_catalog_consistency(self):
        for field, bad in (("shown_price", "14.51"), ("shown_stock", 99)):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["order"]["items"][0][field] = bad
                with self.assertRaises(app.ValidationError):
                    app.triage(data)

    def test_zero_stock_shown_honestly(self):
        self.data["ticket"]["skus"] = ["FICTION-TOTE-202"]
        self.assertEqual(app.triage(self.data)["products"][0]["stock"], 0)

    def test_invalid_values(self):
        mutations = [
            lambda d: d["catalog"]["products"][0].update(stock=True),
            lambda d: d["catalog"]["products"][0].update(price="NaN"),
            lambda d: d["order"]["items"][0].update(quantity=0),
            lambda d: d["order"].update(customer_id="other"),
            lambda d: d["ticket"].update(skus=["missing"]),
            lambda d: d["customer"]["consent"].update(gdpr_personalization="yes"),
            lambda d: d["catalog"]["products"].append(d["catalog"]["products"][0].copy()),
            lambda d: d["config"]["fallback"].update(owner=""),
            lambda d: d.update(synthetic=False),
            lambda d: d.update(extra="unknown"),
            lambda d: d["clickstream"][1].update(sequence=1),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                data = copy.deepcopy(self.data)
                mutation(data)
                with self.assertRaises(app.ValidationError):
                    app.triage(data)

    def test_injected_callable_validated(self):
        seen = []
        def classifier(ticket):
            seen.append(ticket)
            return "returns"
        self.assertEqual(app.triage(self.data, classifier)["category"], "returns")
        self.assertEqual(set(seen[0]), {"subject", "body"})
        for invalid in ("invented", {"review": "amazing"}, None):
            with self.subTest(invalid=invalid), self.assertRaises(app.ValidationError):
                app.triage(self.data, lambda _: invalid)

    def test_callable_failure_is_validation_error(self):
        def broken(_):
            raise RuntimeError("broken fixture")
        with self.assertRaises(app.ValidationError):
            app.triage(self.data, broken)

    def test_output_constraint_validation(self):
        for field, value in (("products", []), ("reviews", ["invented review"]),
                             ("endorsements", ["fake endorsement"]),
                             ("route", {"team": "unknown", "owner": "unknown"})):
            with self.subTest(field=field):
                result = app.triage(self.data)
                result[field] = value
                with self.assertRaises(app.ValidationError):
                    app.validate(self.data, result)

    def test_cli_success(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"),
                              str(ROOT / "example_input.json")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")
        self.assertEqual(run.stderr, "")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], [str(ROOT / "does-not-exist.json")], ["a", "b"]):
            run = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["status"], "error")
            self.assertEqual(run.stderr, "")

    def test_cli_malformed_or_invalid_json(self):
        for payload in ('{', '{"a": 1, "a": 2}', '[]', '{"synthetic": false}'):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with patch("builtins.open", mock_open(read_data=payload)), redirect_stdout(output):
                    code = app.main(["fixture.json"])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
