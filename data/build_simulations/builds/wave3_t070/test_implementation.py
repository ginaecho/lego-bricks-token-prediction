"""All fixture data and injected embeddings in this suite are synthetic."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import implementation as app


ROOT = Path(__file__).resolve().parent


def fixture():
    return {
        "schema_version": 1, "fixture_label": "SYNTHETIC unit test feedback",
        "feedback": [
            {"id": "a", "customer_id": "synthetic-1", "source": "chat",
             "text": "Payment failed. Please refund."},
            {"id": "b", "customer_id": "synthetic-2", "source": "survey",
             "text": "Login blocked by outage."},
            {"id": "c", "customer_id": "synthetic-1", "source": "review",
             "text": "Love the great payment feature."},
        ],
        "queries": [{"text": "reimbursement", "limit": 2}],
    }


class PipelineTests(unittest.TestCase):
    def test_insights_evidence_and_customer_deduplication(self):
        state = app.customer_insights(app.prepare(fixture()))
        billing = next(t for t in state["insights"]["themes"] if t["id"] == "billing")
        self.assertEqual(billing["count"], 2)
        self.assertEqual(billing["feedback_ids"], ["a", "c"])
        self.assertEqual(billing["customer_ids"], ["synthetic-1"])
        self.assertEqual(billing["sentiment_counts"], {"positive": 1, "negative": 1, "neutral": 0})
        self.assertTrue(billing["action"])

    def test_general_purpose_emerging_theme(self):
        data = fixture()
        data["feedback"] = [
            {"id": "a", "customer_id": "synthetic-1", "source": "survey", "text": "Export spreadsheet"},
            {"id": "b", "customer_id": "synthetic-2", "source": "survey", "text": "Export report"},
        ]
        theme = app.run(data)["insights"]["themes"][0]
        self.assertEqual(theme["id"], "inferred:export")
        self.assertEqual(theme["count"], 2)

    def test_punctuation_only_feedback_gets_fallback(self):
        data = fixture()
        data["feedback"][0]["text"] = "!!!"
        result = app.run(data)
        self.assertIn("inferred:other", result["triage"]["tickets"][0]["theme_ids"])

    def test_emerging_themes_exclude_sentiment_words(self):
        data = fixture()
        data["feedback"][0]["text"] = "Love the excellent export tool."
        self.assertEqual(app.run(data)["triage"]["tickets"][0]["theme_ids"], ["inferred:export"])

    def test_multitheme_and_category_precedence(self):
        data = fixture()
        data["feedback"][0]["text"] = "Login and payment failed"
        ticket = app.run(data)["triage"]["tickets"][0]
        self.assertEqual(ticket["theme_ids"], ["billing", "access"])
        self.assertEqual(ticket["category"], "account")
        self.assertEqual(len(ticket["actions"]), 2)

    def test_priority_escalation_and_accountability(self):
        tickets = app.run(fixture())["triage"]["tickets"]
        self.assertEqual(tickets[0]["priority"], "high")
        self.assertEqual(tickets[1]["priority"], "urgent")
        self.assertEqual(tickets[1]["team"], "Support")
        self.assertEqual(tickets[1]["owner"], "account-duty")

    def test_custom_rules_flow_through_all_stages(self):
        data = fixture()
        data["config"] = {
            "theme_rules": [{"id": "money", "label": "Money", "keywords": ["payment"],
                             "action": "Investigate invoice failures"}],
            "category_rules": [{"category": "invoicing", "theme_ids": ["money"], "keywords": [],
                                "priority": "low", "team": "Billing squad", "owner": "synthetic-lead"}],
            "priority_rules": [{"keywords": ["failed"], "priority": "urgent"}],
        }
        data["queries"] = [{"text": "invoice failures", "limit": 5}]
        result = app.run(data)
        ticket = result["triage"]["tickets"][0]
        self.assertEqual(ticket["theme_ids"], ["money"])
        self.assertEqual(ticket["priority"], "urgent")
        self.assertEqual(ticket["actions"], ["Investigate invoice failures"])
        hit = result["semantic"]["results"][0]["hits"][0]
        self.assertEqual(hit["owner"], "synthetic-lead")
        self.assertEqual(hit["category"], "invoicing")
        self.assertEqual(hit["theme_ids"], ["money"])

    def test_default_route_for_unmatched_theme(self):
        data = fixture()
        data["feedback"][0]["text"] = "Calendar export feature"
        data["config"] = {"default_route": {
            "category": "unclassified", "priority": "low", "team": "Intake", "owner": "synthetic-duty"}}
        ticket = app.run(data)["triage"]["tickets"][0]
        self.assertEqual(ticket["routing_reason"], "default_route")
        self.assertEqual(ticket["owner"], "synthetic-duty")

    def test_keyword_routing_and_phrase_matching(self):
        data = fixture()
        data["feedback"][0]["text"] = "Data export failed"
        data["config"] = {"category_rules": [{
            "category": "exports", "theme_ids": [], "keywords": ["data export"],
            "priority": "normal", "team": "Data", "owner": "synthetic-data"}]}
        self.assertEqual(app.run(data)["triage"]["tickets"][0]["category"], "exports")
        self.assertFalse(app.matches("refunding", ["refund"], app.DEFAULT_CONFIG))

    def test_search_synonyms_and_index(self):
        result = app.run(fixture())
        search = result["semantic"]
        self.assertEqual(search["mode"], "lexical_synonyms")
        self.assertEqual(search["results"][0]["hits"][0]["ticket_id"], "ticket:a")
        self.assertIn("ticket:a", search["index"]["postings"]["refund"])
        self.assertEqual(len(search["index"]["documents"]), 3)

    def test_no_matches_and_limits(self):
        data = fixture()
        data["queries"] = [{"text": "zzzzunknown", "limit": 5},
                           {"text": "payment", "limit": 1}]
        results = app.run(data)["semantic"]["results"]
        self.assertEqual(results[0]["hits"], [])
        self.assertEqual(len(results[1]["hits"]), 1)

    def test_search_ties_have_stable_id_order(self):
        data = fixture()
        data["feedback"][1]["text"] = data["feedback"][0]["text"]
        data["feedback"][2]["text"] = data["feedback"][0]["text"]
        data["feedback"].reverse()
        data["queries"] = [{"text": "payment", "limit": 3}]
        hits = app.run(data)["semantic"]["results"][0]["hits"]
        self.assertEqual([h["ticket_id"] for h in hits], ["ticket:a", "ticket:b", "ticket:c"])

    def test_optional_embedding_can_find_nonlexical_match(self):
        data = fixture()
        data["queries"] = [{"text": "financial restitution", "limit": 3}]

        def synthetic_embedding(value):
            return [1.0, 0.0] if "financial" in value or "Payment failed" in value else [0.0, 1.0]

        self.assertEqual(app.run(data)["semantic"]["results"][0]["hits"], [])
        result = app.run(data, embedder=synthetic_embedding)
        self.assertEqual(result["semantic"]["mode"], "hybrid_embeddings")
        self.assertEqual(result["semantic"]["results"][0]["hits"][0]["ticket_id"], "ticket:a")

    def test_embedding_contract_rejections(self):
        for vector in ([], [0, 0], [True, 1], [float("nan")], [float("inf")],
                       [10 ** 500], "vector"):
            with self.subTest(vector=vector), self.assertRaises(app.ValidationError):
                app.run(fixture(), embedder=lambda value: vector)
        with self.assertRaises(app.ValidationError):
            app.run(fixture(), embedder="not callable")

    def test_embedding_dimension_and_exception(self):
        vectors = iter([[1, 0], [1]])
        with self.assertRaisesRegex(app.ValidationError, "dimensions"):
            app.run(fixture(), embedder=lambda value: next(vectors))
        with self.assertRaisesRegex(app.ValidationError, "callable failed"):
            app.run(fixture(), embedder=lambda value: 1 / 0)

    def test_empty_feedback_and_queries(self):
        data = fixture()
        data["feedback"] = []
        result = app.run(data)
        self.assertEqual(result["insights"]["themes"], [])
        self.assertEqual(result["triage"]["tickets"], [])
        self.assertEqual(result["semantic"]["index"], {"documents": [], "postings": {}})
        self.assertEqual(result["semantic"]["results"][0]["hits"], [])
        data.pop("queries")
        self.assertEqual(app.run(data)["semantic"]["results"], [])

    def test_determinism_and_input_not_mutated(self):
        data = fixture()
        before = copy.deepcopy(data)
        self.assertEqual(app.run(data), app.run(data))
        self.assertEqual(data, before)

    def test_invalid_input_schema(self):
        variants = [
            None, [], {}, {**fixture(), "extra": 2}, {**fixture(), "schema_version": True},
            {**fixture(), "feedback": "bad"}, {**fixture(), "queries": {}},
            {**fixture(), "fixture_label": ""}, {**fixture(), "config": {"unknown": True}},
        ]
        for data in variants:
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_invalid_feedback_and_query(self):
        for bad in ("", None, 3, [], {}):
            data = fixture()
            data["feedback"][0]["text"] = bad
            with self.subTest(bad=bad), self.assertRaises(app.ValidationError):
                app.run(data)
        data = fixture()
        data["feedback"].append(copy.deepcopy(data["feedback"][0]))
        with self.assertRaisesRegex(app.ValidationError, "duplicate"):
            app.run(data)
        for query in ({"text": "x", "limit": True}, {"text": "", "limit": 1},
                      {"text": "x", "limit": 0}, {"text": "x", "limit": 101}):
            data = fixture()
            data["queries"] = [query]
            with self.subTest(query=query), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_invalid_config_and_route(self):
        for config in (
            {"category_rules": [{"category": "x", "theme_ids": ["unknown"], "keywords": [],
                                 "priority": "normal", "team": "X", "owner": "Y"}]},
            {"priority_rules": [{"keywords": ["x"], "priority": "critical"}]},
            {"default_route": {"category": "x", "priority": "normal", "team": "X", "owner": ""}},
            {"synonyms": {"a": ["same"], "b": ["same"]}},
            {"theme_rules": None},
        ):
            data = fixture()
            data["config"] = config
            with self.subTest(config=config), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_insights_boundary_rejects_tampering(self):
        state = app.customer_insights(app.prepare(fixture()))
        state["insights"]["themes"][0]["feedback_ids"].append("missing")
        with self.assertRaises(app.ValidationError):
            app.triage_tickets(state)
        state = app.customer_insights(app.prepare(fixture()))
        state["insights"]["signals"].pop()
        with self.assertRaises(app.ValidationError):
            app.triage_tickets(state)

    def test_triage_boundary_rejects_tampering(self):
        state = app.triage_tickets(app.customer_insights(app.prepare(fixture())))
        state["triage"]["tickets"][0]["text"] = "changed"
        with self.assertRaises(app.ValidationError):
            app.semantic_search(state)
        with self.assertRaises(app.ValidationError):
            app.semantic_search(app.prepare(fixture()))

    def test_search_boundary_rejects_tampering(self):
        result = app.run(fixture())
        result["semantic"]["results"][0]["hits"][0]["owner"] = "wrong"
        with self.assertRaises(app.ValidationError):
            app.validate(result, "semantic")

    def test_cli_real_example(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), str(ROOT / "example_input.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "ok")
        app.validate(result, "semantic")

    def test_cli_missing_file_and_arguments(self):
        for args in ([], ["does-not-exist.json"], ["."], ["one", "two"]):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "implementation.py")] + args,
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["status"], "error")
            self.assertEqual(completed.stderr, "")

    def test_cli_malformed_validation_and_read_errors_without_extra_files(self):
        for contents in ("{", "null", '{"schema_version":NaN}', '{"a":1,"a":2}', "[]"):
            with patch.object(Path, "read_text", return_value=contents), redirect_stdout(io.StringIO()) as out:
                code = app.main(["synthetic.json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "error")
        with patch.object(Path, "read_text", side_effect=PermissionError("synthetic denial")), \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(out.getvalue())["status"], "error")

    def test_cli_numeric_parser_limits_are_json_errors(self):
        with patch.object(Path, "read_text", side_effect=ValueError("integer string too long")), \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(app.main(["synthetic.json"]), 2)
        self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
