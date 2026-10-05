import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def result(self):
        return app.run_pipeline(self.source)["result"]

    def invalid(self):
        with self.assertRaises((app.ValidationError, TypeError)):
            app.run_pipeline(self.source)

    def test_example_all_stages(self):
        result = self.result()
        self.assertEqual(result["stage"], "recommend")
        self.assertEqual(result["support"]["action"], "review_billing")
        self.assertEqual(result["onboarding"]["next_step"], "review_usage")
        self.assertEqual(result["recommendations"]["items"][0]["product_id"], "plan_saver")

    def test_handoffs(self):
        result = self.result()
        self.assertEqual(result["support"]["action"], result["onboarding"]["source_support_action"])
        self.assertEqual(result["onboarding"]["next_step"], result["recommendations"]["source_onboarding_step"])

    def test_network_fault_defers_discovery(self):
        ticket = self.source["tickets"][0]
        ticket.update(category="network", billing_adjustment=None)
        result = self.result()
        self.assertTrue(result["support"]["blocked"])
        self.assertEqual(result["onboarding"]["next_step"], "await_network")
        self.assertEqual(result["recommendations"]["reason"], "deferred_network_fault")
        self.assertEqual(result["recommendations"]["items"], [])

    def test_resolved_network_not_blocking(self):
        self.source["tickets"][0].update(category="network", status="resolved", billing_adjustment=None)
        self.assertFalse(self.result()["support"]["blocked"])

    def test_general_support_and_activation(self):
        self.source["tickets"] = []
        self.source["chat"] = []
        self.source["subscriber"].update(goal="activate", completed_steps=[])
        result = self.result()
        self.assertEqual(result["support"]["action"], "continue_onboarding")
        self.assertEqual(result["onboarding"]["next_step"], "activate_sim")
        self.assertEqual(result["recommendations"]["reason"], "finish_activation_first")

    def test_network_chat_without_ticket(self):
        self.source["tickets"] = []
        self.source["chat"][0]["text"] = "My signal is poor"
        self.assertEqual(self.result()["support"]["action"], "check_connection")

    def test_completed_onboarding(self):
        self.source["subscriber"]["completed_steps"] = ["activate_sim", "check_connection", "review_usage"]
        self.assertEqual(self.result()["onboarding"]["next_step"], "complete")

    def test_more_data_discovery(self):
        self.source["subscriber"]["goal"] = "more_data"
        self.assertEqual(self.result()["recommendations"]["items"][0]["product_id"], "plan_large")

    def test_opt_out(self):
        self.source["subscriber"]["discovery_consent"] = False
        self.assertEqual(self.result()["recommendations"]["items"], [])
        self.assertEqual(self.result()["recommendations"]["reason"], "discovery_consent_not_granted")

    def test_processing_consent_required(self):
        self.source["subscriber"]["processing_consent"] = False
        self.invalid()

    def test_no_sensitive_data_in_output(self):
        self.source["chat"][0]["text"] += " PRIVATECHAT secret@example.invalid"
        output = json.dumps(self.result())
        for sensitive in ("PRIVATECHAT", "secret@example.invalid", self.source["subscriber"]["phone"],
                          self.source["subscriber"]["imei"], "duplicate activation charge", "2026-09-01"):
            self.assertNotIn(sensitive, output)

    def test_residency_all_entities(self):
        for location in ("subscriber", "tickets", "chat", "catalog", "adjustment", "csv"):
            with self.subTest(location=location):
                original = copy.deepcopy(self.source)
                if location == "csv":
                    self.source["usage_csv"] = self.source["usage_csv"].replace(",EU,", ",US,", 1)
                elif location == "adjustment":
                    self.source["tickets"][0]["billing_adjustment"]["residency"] = "US"
                elif location == "subscriber":
                    self.source[location]["residency"] = "US"
                else:
                    self.source[location][0]["residency"] = "US"
                self.invalid()
                self.source = original

    def test_us_residency_propagates(self):
        self.source = json.loads(json.dumps(self.source).replace("EU", "US"))
        result = self.result()
        for name in ("subscriber", "usage_summary", "facts", "support", "onboarding", "recommendations"):
            self.assertEqual(result[name]["residency"], "US")
        self.assertEqual(result["recommendations"]["items"][0]["residency"], "US")

    def test_adjustment_reason_mandatory(self):
        self.source["tickets"][0]["billing_adjustment"]["reason"] = " "
        self.invalid()

    def test_adjustment_never_applied(self):
        self.assertFalse(self.result()["support"]["billing_adjustments_applied"])

    def test_synthetic_label_required(self):
        self.source["synthetic"] = False
        self.invalid()

    def test_subscriber_mismatch(self):
        self.source["tickets"][0]["subscriber_id"] = "sub_other"
        self.invalid()

    def test_duplicate_cdr(self):
        self.source["usage_csv"] += self.source["usage_csv"].splitlines()[1] + "\n"
        self.invalid()

    def test_malformed_cdr(self):
        self.source["usage_csv"] += "cdr_broken,sub_demo85\n"
        self.invalid()

    def test_malformed_csv_header_cli(self):
        self.source["usage_csv"] = '"unterminated CSV header'
        with patch("builtins.open", mock_open(read_data=json.dumps(self.source))), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(app.main(["input.json"]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_usage_timestamp_requires_timezone(self):
        self.source["usage_csv"] = self.source["usage_csv"].replace("10:00:00Z", "10:00:00")
        self.invalid()

    def test_cli_existing_invalid_schema_file(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "build_manifest.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")
        self.assertEqual(proc.stderr, "")

    def test_invalid_numeric_usage(self):
        for bad in ("-1", "NaN", "Infinity", "oops"):
            with self.subTest(bad=bad):
                original = self.source["usage_csv"]
                self.source["usage_csv"] = original.replace(",18,", "," + bad + ",")
                self.invalid()
                self.source["usage_csv"] = original

    def test_seeded_randomized_usage(self):
        rng = random.Random(85)
        volumes = [rng.randint(100, 1500) for _ in range(8)]
        self.source["usage_csv"] = self.source["usage_csv"].splitlines()[0] + "\n" + "".join(
            f"cdr_r{i},sub_demo85,EU,2026-09-01T10:00:00Z,{rng.randint(0, 60)},{volume}\n"
            for i, volume in enumerate(volumes))
        self.assertEqual(self.result()["usage_summary"]["data_gb"], round(sum(volumes) / 1024, 3))

    def test_empty_usage(self):
        self.source["usage_csv"] = self.source["usage_csv"].splitlines()[0] + "\n"
        self.assertEqual(self.result()["usage_summary"]["record_count"], 0)

    def test_no_suitable_plan(self):
        self.source["catalog"] = self.source["catalog"][:1]
        self.assertEqual(self.result()["recommendations"]["reason"], "no_suitable_alternative")

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.source)
        self.assertEqual(self.result(), self.result())
        self.assertEqual(self.source, original)

    def test_tampered_handoff_rejected(self):
        state = app.support(app.prepare(self.source))
        state["support"]["answer"] = "Your refund is approved"
        with self.assertRaises(app.ValidationError):
            app.onboard(state)

    def test_out_of_order_rejected(self):
        with self.assertRaises(app.ValidationError):
            app.recommend(app.prepare(self.source))

    def test_output_residency_tampering_rejected(self):
        state = app.onboard(app.support(app.prepare(self.source)))
        state["onboarding"]["residency"] = "US"
        with self.assertRaises(app.ValidationError):
            app.recommend(state)

    def test_cli_success(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")
        self.assertEqual(proc.stderr, "")

    def test_cli_missing_file(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "missing.json")],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "error")
        self.assertEqual(proc.stderr, "")

    def test_cli_bad_json_and_schema(self):
        for content in ("{", "[]", '{"synthetic": true, "synthetic": false}', '{"schema_version": "9"}'):
            with self.subTest(content=content), patch("builtins.open", mock_open(read_data=content)), \
                    patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(app.main(["input.json"]), 2)
                self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_cli_wrong_argument_count(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(app.main([]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_unknown_fields_rejected(self):
        self.source["subscriber"]["email"] = "synthetic@example.invalid"
        self.invalid()

    def test_invalid_boolean_cost(self):
        self.source["catalog"][0]["monthly_cost"] = True
        self.invalid()


if __name__ == "__main__":
    unittest.main()
