import contextlib
import copy
import io
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import implementation as app


HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / "example_input.json").read_text(encoding="utf-8"))

    def search(self):
        return app.semantic_search(app.parse_input(self.data))

    def documents(self):
        return app.automate_documents(app.recommend_journey(self.search()))

    def test_integrated_pipeline_and_determinism(self):
        original = copy.deepcopy(self.data)
        result = app.run(self.data)
        self.assertEqual(result, app.run(self.data))
        self.assertEqual(self.data, original)
        self.assertEqual(result["stage"], "feedback")
        self.assertEqual(len(result["journey"]), 2)
        self.assertEqual(len(result["documents"]), 2)

    def test_synonym_index_and_rank_order(self):
        self.data["query"] = "connectivity"
        result = self.search()
        self.assertTrue(result["search"])
        self.assertIn("TICKET-101", [x["record_id"] for x in result["search"]])
        self.assertEqual(result["search"], sorted(
            result["search"], key=lambda x: (-x["score"], x["record_id"])))

    def test_injected_embedding_is_local_and_receives_redacted_text(self):
        calls = []
        def fake(texts):
            calls.append(texts)
            return [[1.0, 0.0] for _ in texts]
        result = app.run(self.data, fake)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("+1-202-555-0101", json.dumps(calls))
        self.assertTrue(all(h["score"] == 1.0 for h in result["search"]))

    def test_invalid_embeddings_rejected(self):
        for output in (None, [], [[1]], [[float("nan")]] * 8,
                       [[True]] * 8, [[1, 2], [1]]):
            with self.subTest(output=output), self.assertRaises(app.ValidationError):
                app.run(self.data, lambda texts: output)

    def test_embedding_exception_is_validation_error(self):
        def broken(texts):
            raise RuntimeError("provider unavailable")
        with self.assertRaisesRegex(app.ValidationError, "injected embedding failed"):
            app.run(self.data, broken)

    def test_zero_embeddings_give_no_supported_journey(self):
        with self.assertRaisesRegex(app.ValidationError, "no relevant"):
            app.run(self.data, lambda texts: [[0, 0] for _ in texts])

    def test_unmatched_query(self):
        self.data["query"] = "qzxw"
        self.assertEqual(self.search()["search"], [])
        with self.assertRaisesRegex(app.ValidationError, "no relevant"):
            app.run(self.data)

    def test_prerequisite_aware_two_steps(self):
        self.data["query"] = "connectivity"
        result = app.recommend_journey(self.search())
        done = set()
        for step in result["journey"]:
            self.assertLessEqual(set(step["prerequisites"]), done)
            done.add(step["id"])
        self.assertEqual(result["journey"][0]["id"], "diagnose_network")
        self.assertEqual(result["journey"][1]["id"], "request_network_repair")

    def test_completed_actions_not_repeated(self):
        self.data["query"] = "connectivity"
        self.data["completed_actions"] = ["diagnose_network"]
        result = app.run(self.data)
        self.assertEqual(result["journey"][0]["id"], "request_network_repair")
        self.assertNotIn("diagnose_network", [x["id"] for x in result["journey"]])

    def test_invalid_completed_prerequisites(self):
        self.data["completed_actions"] = ["request_network_repair"]
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_exhausted_journey(self):
        self.data["completed_actions"] = list(app.ACTIONS)
        with self.assertRaisesRegex(app.ValidationError, "not enough"):
            app.run(self.data)

    def test_document_extraction_and_adjustment_reason(self):
        result = self.documents()
        values = result["documents"][0]["extracted"]
        rows = [r for r in result["records"] if r["kind"] == "usage"]
        self.assertEqual(values["call_seconds"], sum(r["values"]["call_seconds"] for r in rows))
        self.assertEqual(values["data_mb"], sum(r["values"]["data_mb"] for r in rows))
        self.assertEqual(values["ticket_ids"], ["TICKET-101"])
        self.assertEqual(values["adjustments"][0]["amount_cents"], -450)
        self.assertTrue(values["adjustments"][0]["reason"])

    def test_cross_stage_source_propagation(self):
        result = app.run(self.data)
        hits = [h["record_id"] for h in result["search"]]
        for step, document in zip(result["journey"], result["documents"]):
            self.assertEqual(step["source_ids"], hits)
            self.assertEqual(document["source_ids"], step["source_ids"])
            self.assertEqual(document["action_id"], step["id"])
        for theme in result["feedback"]:
            for evidence in theme["evidence"]:
                self.assertLessEqual(set(evidence["source_ids"]), set(hits))
                self.assertEqual(evidence["document_ids"], ["DOC-1", "DOC-2"])

    def test_feedback_deduplication_and_themes(self):
        result = app.run(self.data)
        network = next(x for x in result["feedback"] if x["theme"] == "network")
        self.assertEqual(network["count"], 1)
        self.assertEqual(network["evidence"][0]["source_ids"], ["CHAT-101", "CHAT-102"])
        self.assertIn("billing", [x["theme"] for x in result["feedback"]])

    def test_whitespace_duplicate_keeps_traceable_excerpt(self):
        self.data["records"][3]["text"] = self.data["records"][2]["text"].replace(" ", "  ")
        result = app.run(self.data)
        network = next(x for x in result["feedback"] if x["theme"] == "network")
        self.assertEqual(network["count"], 1)
        self.assertEqual(len(network["evidence"][0]["source_ids"]), 2)

    def test_no_chat_produces_empty_feedback(self):
        self.data["records"] = [r for r in self.data["records"] if r["kind"] != "chat"]
        self.assertEqual(app.run(self.data)["feedback"], [])

    def test_document_scope_limits_feedback(self):
        self.data["query"] = "incorrect"
        result = app.run(self.data)
        self.assertEqual([h["record_id"] for h in result["search"]], ["CHAT-103"])
        for theme in result["feedback"]:
            self.assertEqual(theme["evidence"][0]["source_ids"], ["CHAT-103"])

    def test_pii_redacted_from_entire_output(self):
        self.data["records"][2]["text"] += " Device SYN-IMEI-00000001 and 490154203237518."
        output = json.dumps(app.run(self.data))
        for secret in ("+1-202-555-0101", "demo@example.invalid",
                       "SYN-IMEI-00000001", "490154203237518"):
            self.assertNotIn(secret, output)
        self.assertIn("[EMAIL]", output)

    def test_every_input_record_respects_residency(self):
        for index in range(len(self.data["records"])):
            data = copy.deepcopy(self.data)
            data["records"][index]["residency"] = "US"
            with self.subTest(index=index), self.assertRaisesRegex(app.ValidationError, "residency"):
                app.run(data)
        self.data["cdr_csv"] = self.data["cdr_csv"].replace(",EU,", ",US,")
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_output_residency_propagates(self):
        result = app.run(self.data)
        for field in ("records", "search", "journey", "documents", "feedback"):
            self.assertTrue(all(r["residency"] == "EU" for r in result[field]))
        for theme in result["feedback"]:
            self.assertTrue(all(r["residency"] == "EU" for r in theme["evidence"]))

    def test_consent_and_purpose_required(self):
        for field, value in (("consent", False), ("purpose", "advertising")):
            data = copy.deepcopy(self.data)
            data["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_billing_adjustment_requires_reason(self):
        self.data["records"][-1]["reason"] = "  "
        with self.assertRaisesRegex(app.ValidationError, "reason"):
            app.run(self.data)

    def test_real_phone_or_imei_rejected(self):
        for field, value in (("phone", "+1-415-123-4567"), ("imei", "490154203237518")):
            data = copy.deepcopy(self.data)
            data["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_duplicate_ids_and_orphan_account_rejected(self):
        data = copy.deepcopy(self.data)
        data["records"].append(copy.deepcopy(data["records"][1]))
        with self.assertRaises(app.ValidationError):
            app.run(data)
        self.data["records"][1]["account_id"] = "ACC-UNKNOWN"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_account_isolation(self):
        subscriber = copy.deepcopy(self.data["records"][0])
        subscriber["id"] = subscriber["account_id"] = "ACC-OTHER"
        chat = copy.deepcopy(self.data["records"][2])
        chat.update(id="CHAT-OTHER", account_id="ACC-OTHER")
        self.data["records"].extend([subscriber, chat])
        result = app.run(self.data)
        self.assertNotIn("ACC-OTHER", json.dumps(result))
        self.assertNotIn("CHAT-OTHER", json.dumps(result))

    def test_csv_header_and_numeric_errors(self):
        for csv_text in ("bad,header\n1,2\n",
                         self.data["cdr_csv"].replace(",2926,", ",-1,"),
                         self.data["cdr_csv"] + "broken,row\n"):
            data = copy.deepcopy(self.data)
            data["cdr_csv"] = csv_text
            with self.subTest(csv_text=csv_text), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_zero_usage_and_seeded_random_usage(self):
        rng = random.Random(103)
        for seconds, mb in [(0, 0), (rng.randint(1, 4000), rng.randint(1, 2000))]:
            data = copy.deepcopy(self.data)
            data["cdr_csv"] = data["cdr_csv"].splitlines()[0] + (
                f"\nCDR-ONLY,ACC-DEMO1,EU,true,{seconds},{mb},"
                "+1-202-555-0101,SYN-IMEI-00000001\n")
            values = app.run(data)["documents"][0]["extracted"]
            self.assertEqual((values["call_seconds"], values["data_mb"]), (seconds, mb))

    def test_invalid_input_shapes(self):
        for data in (None, [], {}, {"schema_version": 99}):
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run(data)
        self.data["records"][0]["name"] = "Not permitted"
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_synthetic_label_required(self):
        self.data["records"][1]["synthetic"] = False
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_wrong_stage_rejected(self):
        with self.assertRaisesRegex(app.ValidationError, "wrong pipeline stage"):
            app.automate_documents(app.parse_input(self.data))

    def test_tampered_search_reference_rejected(self):
        env = self.search()
        env["search"][0]["record_id"] = "MISSING"
        with self.assertRaises(app.ValidationError):
            app.recommend_journey(env)

    def test_tampered_prerequisite_rejected(self):
        env = app.recommend_journey(self.search())
        env["journey"][0]["prerequisites"] = ["inspect_bill"]
        with self.assertRaises(app.ValidationError):
            app.automate_documents(env)

    def test_tampered_document_totals_rejected(self):
        env = self.documents()
        env["documents"][0]["extracted"]["data_mb"] += 1
        with self.assertRaises(app.ValidationError):
            app.analyze_feedback(env)

    def test_tampered_safe_usage_rejected(self):
        env = self.search()
        next(r for r in env["records"] if r["kind"] == "usage")["values"]["data_mb"] = -1
        with self.assertRaises(app.ValidationError):
            app.recommend_journey(env)

    def test_tampered_feedback_excerpt_rejected(self):
        env = app.run(self.data)
        env["feedback"][0]["evidence"][0]["excerpt"] = "invented evidence"
        with self.assertRaises(app.ValidationError):
            app.validate_envelope(env, "feedback")

    def test_tampered_feedback_theme_rejected(self):
        env = app.run(self.data)
        env["feedback"][0]["theme"] = "other"
        with self.assertRaises(app.ValidationError):
            app.validate_envelope(env, "feedback")

    def test_fixture_usage_is_seeded_random(self):
        records = app.parse_input(self.data)["records"]
        actual = [(r["values"]["call_seconds"], r["values"]["data_mb"])
                  for r in records if r["kind"] == "usage"]
        rng = random.Random(103)
        self.assertEqual(actual, [(rng.randint(1, 4000), rng.randint(1, 2000)) for _ in range(2)])

    def test_cli_success_single_json(self):
        proc = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                              cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        self.assertEqual(json.loads(proc.stdout)["status"], "ok")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], ["does-not-exist.json"]):
            proc = subprocess.run([sys.executable, "-B", "implementation.py"] + args,
                                  cwd=HERE, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "error")
            self.assertEqual(proc.stderr, "")

    def test_cli_malformed_json_and_invalid_data(self):
        for contents in ("{broken", "null", json.dumps({**self.data, "synthetic": False})):
            output = io.StringIO()
            with patch("builtins.open", mock_open(read_data=contents)), contextlib.redirect_stdout(output):
                code = app.main(["example_input.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
