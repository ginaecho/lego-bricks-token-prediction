"""All inputs and callbacks are explicitly fictional; no provider requests."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from implementation import ValidationError, load_json, run_pipeline


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def run_input(self, **kwargs):
        return run_pipeline(self.raw, **kwargs)

    def assert_invalid(self):
        with self.assertRaises(ValidationError):
            self.run_input()

    def test_complete_data_handoff(self):
        result = self.run_input()
        self.assertEqual(result["research"]["method"], "explicit_claim_consolidation")
        hit = result["search"]["hits"][0]
        self.assertEqual(hit["provenance"][0]["claim_id"], "claim-access")
        self.assertEqual(hit["provenance"][0]["statement_id"], "statement-access")
        self.assertIn(hit["finding_id"], [f["id"] for f in result["research"]["findings"]])
        self.assertIn(hit["id"], result["triage"]["evidence_ids"])
        self.assertEqual(result["triage"]["owner"], "FICTIONAL Identity Desk")
        self.assertEqual(result["onboarding"]["priority"], result["triage"]["priority"])
        steps = result["onboarding"]["steps"]
        self.assertEqual([s["task_id"] for s in steps],
                         ["setup", "identity-guide", "login-practice"])
        self.assertEqual([s["status"] for s in steps], ["ready", "waiting", "waiting"])
        self.assertEqual(result["search"]["method"], "lexical_tfidf_cosine")

    def test_evidence_change_changes_route_and_plan(self):
        before = self.run_input()
        statement = self.raw["sources"][0]["statements"][0]
        statement["category_id"] = "accounts"
        statement["value"] = "accounts support"
        statement["text"] = "FICTIONAL: portal login password reset belongs to accounts support."
        self.raw["claims"][0].update(category_id="accounts", value="accounts support")
        after = self.run_input()
        self.assertEqual(before["triage"]["category_id"], "identity")
        self.assertEqual(after["triage"]["category_id"], "accounts")
        self.assertEqual(after["triage"]["owner"], "FICTIONAL Accounts Desk")
        self.assertEqual(after["onboarding"]["pace"], "self_paced")
        self.assertEqual([s["task_id"] for s in after["onboarding"]["steps"]],
                         ["setup", "invoice-practice"])

    def test_question_change_changes_route(self):
        self.raw["ticket"]["question"] = "invoice export receipt"
        result = self.run_input()
        self.assertEqual(result["triage"]["category_id"], "accounts")
        self.assertEqual(result["onboarding"]["steps"][-1]["task_id"], "invoice-practice")

    def test_profile_change_changes_plan_not_route(self):
        before = self.run_input()
        self.raw["profile"].update(experience="expert", preferred_format="text",
                                   completed_task_ids=["setup"])
        after = self.run_input()
        self.assertEqual(before["triage"], after["triage"])
        self.assertEqual(after["onboarding"]["completed_task_ids"], ["setup"])
        self.assertEqual(after["onboarding"]["steps"][0]["task_id"], "identity-guide")
        self.assertEqual(after["onboarding"]["steps"][0]["status"], "ready")
        self.assertTrue(all(s["detail"] == "concise_checklist" and s["format"] == "text"
                            for s in after["onboarding"]["steps"]))

    def test_priority_and_configured_task_change_reach_onboarding(self):
        before = self.run_input()
        self.raw["categories"][0].update(priority="low", task_ids=["setup"],
                                          owner="FICTIONAL Backup Desk")
        after = self.run_input()
        self.assertEqual(after["triage"]["owner"], "FICTIONAL Backup Desk")
        self.assertEqual(after["triage"]["priority"], "normal")
        self.assertNotEqual(before["onboarding"]["pace"], after["onboarding"]["pace"])
        self.assertEqual([s["task_id"] for s in after["onboarding"]["steps"]], ["setup"])
        self.raw["ticket"]["urgency"] = "critical"
        self.assertEqual(self.run_input()["onboarding"]["pace"], "expedited")

    def add_disagreement(self, same_category=False):
        statement = copy.deepcopy(self.raw["sources"][0]["statements"][0])
        statement.update(id="statement-conflict", value="different explicit instruction",
                         category_id="identity" if same_category else "accounts",
                         text="FICTIONAL: an alternative explicit instruction is recorded.")
        self.raw["sources"].append({"id": "fictional-dissent", "title": "FICTIONAL dissent",
                                    "statements": [statement]})
        claim = copy.deepcopy(self.raw["claims"][0])
        claim.update(id="claim-conflict", source_id="fictional-dissent",
                     statement_id=statement["id"], value=statement["value"],
                     category_id=statement["category_id"])
        self.raw["claims"].append(claim)

    def test_disagreement_preserved_beyond_search_cutoff(self):
        self.add_disagreement()
        self.raw["search"]["top_k"] = 1
        result = self.run_input()
        finding = next(f for f in result["research"]["findings"] if f["disagreement"])
        self.assertEqual(len(finding["alternatives"]), 2)
        self.assertEqual(len(result["search"]["hits"]), 1)
        self.assertEqual(result["triage"]["candidate_category_ids"], ["accounts", "identity"])
        self.assertEqual(result["triage"]["status"], "needs_review")
        self.assertIsNone(result["triage"]["owner"])
        self.assertEqual(result["triage"]["conflicting_finding_ids"], [finding["id"]])
        self.assertEqual(result["onboarding"]["status"], "needs_review")
        self.assertTrue(all(s["status"] == "blocked" for s in result["onboarding"]["steps"]))

    def test_same_category_disagreement_requires_review(self):
        self.add_disagreement(same_category=True)
        self.assertEqual(self.run_input()["triage"]["status"], "needs_review")

    def test_duplicate_claim_provenance_does_not_inflate_search(self):
        before = self.run_input()
        claim = copy.deepcopy(self.raw["claims"][0])
        claim["id"] = "claim-independent-copy"
        self.raw["claims"].append(claim)
        after = self.run_input()
        self.assertEqual(before["triage"]["category_scores"], after["triage"]["category_scores"])
        self.assertEqual(len(after["search"]["hits"][0]["provenance"]), 2)

    def test_ambiguity_ties_not_hidden_by_top_k(self):
        self.raw["search"]["top_k"] = 1
        result = self.run_input(embed=lambda texts: [[1, 0] for _ in texts])
        self.assertEqual(len(result["search"]["hits"]), 2)
        self.assertEqual(result["triage"]["status"], "needs_review")

    def test_ambiguity_margin_not_hidden_by_top_k(self):
        self.raw["search"]["top_k"] = 1
        vectors = [[1, 0], [1, 0], [0.95, (1 - 0.95 ** 2) ** 0.5]]
        result = self.run_input(embed=lambda _: vectors)
        self.assertEqual(len(result["search"]["hits"]), 2)
        self.assertEqual(result["triage"]["status"], "needs_review")
        self.raw["search"]["ambiguity_margin"] = 0.01
        result = self.run_input(embed=lambda _: vectors)
        self.assertEqual(len(result["search"]["hits"]), 1)
        self.assertEqual(result["triage"]["status"], "routed")

    def test_injected_embedding_actually_changes_route(self):
        calls = []

        def fixture(texts):
            calls.append(texts)
            return [[1, 0]] + [[1, 0] if "invoice" in t else [0, 1] for t in texts[1:]]

        result = self.run_input(embed=fixture)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], self.raw["ticket"]["question"])
        self.assertEqual(result["search"]["method"], "injected_embedding_cosine")
        self.assertEqual(result["triage"]["category_id"], "accounts")
        self.assertEqual(result["onboarding"]["steps"][-1]["task_id"], "invoice-practice")

    def test_zero_vectors_no_nan_no_route(self):
        result = self.run_input(embed=lambda texts: [[0, 0] for _ in texts])
        self.assertEqual(result["search"]["hits"], [])
        self.assertEqual(result["triage"]["status"], "needs_evidence")
        self.assertEqual(result["onboarding"]["steps"], [])
        json.dumps(result, allow_nan=False)

    def test_zero_query_or_document_vectors(self):
        for vectors in ([[0, 0], [1, 0], [1, 0]], [[1, 0], [0, 0], [0, 0]]):
            with self.subTest(vectors=vectors):
                self.assertEqual(self.run_input(embed=lambda _: vectors)["search"]["hits"], [])

    def test_negative_cosine_is_not_evidence(self):
        self.assertEqual(self.run_input(
            embed=lambda texts: [[1, 0]] + [[-1, 0] for _ in texts[1:]]
        )["triage"]["status"], "needs_evidence")

    def test_large_and_small_embedding_vectors_are_stable(self):
        for scale in (1e308, 1e-308):
            result = self.run_input(embed=lambda texts: [[scale, scale] for _ in texts])
            self.assertEqual(result["triage"]["status"], "needs_review")
            json.dumps(result, allow_nan=False)

    def test_invalid_embeddings(self):
        fixtures = (None, [], [[1]], [[1], [1, 0], [1]],
                    [[True], [1], [1]], [[float("nan")]] * 3,
                    [[float("inf")]] * 3, [["1"]] * 3, [[], [], []],
                    [[10 ** 1000]] * 3)
        for vectors in fixtures:
            with self.subTest(vectors=str(vectors)[:50]):
                with self.assertRaises(ValidationError):
                    self.run_input(embed=lambda _: vectors)

    def test_no_lexical_evidence(self):
        for question in ("zzzxxyy", "!!!"):
            self.raw["ticket"]["question"] = question
            result = self.run_input()
            self.assertEqual(result["triage"]["status"], "needs_evidence")
            self.assertIsNone(result["triage"]["owner"])
            self.assertEqual(result["onboarding"]["steps"], [])

    def test_empty_claims_skip_embed(self):
        self.raw["claims"] = []

        def forbidden(_):
            self.fail("Embedding must not be called without documents")

        self.assertEqual(self.run_input(embed=forbidden)["onboarding"]["status"], "needs_evidence")

    def test_sources_without_claims_are_not_indexed(self):
        self.raw["claims"] = self.raw["claims"][1:]
        self.assertEqual(self.run_input()["triage"]["status"], "needs_evidence")

    def test_blocked_setup_propagates(self):
        self.raw["profile"]["available_resources"] = []
        result = self.run_input()
        steps = result["onboarding"]["steps"]
        self.assertEqual(result["onboarding"]["status"], "blocked")
        self.assertEqual(steps[0]["blocked_by"], ["missing_resource:training-workspace"])
        self.assertIn("blocked_prerequisite:setup", steps[1]["blocked_by"])
        self.assertIn("blocked_prerequisite:identity-guide", steps[2]["blocked_by"])
        self.assertTrue(all(s["status"] == "blocked" for s in steps))

    def test_user_block_and_unblocked_sibling(self):
        self.raw["profile"]["blocked_task_ids"] = ["identity-guide"]
        result = self.run_input()
        self.assertEqual(result["onboarding"]["steps"][0]["status"], "ready")
        self.assertIn("user_reported_block", result["onboarding"]["steps"][1]["blocked_by"])
        self.assertEqual(result["onboarding"]["steps"][2]["status"], "blocked")

    def test_expert_does_not_skip_prerequisites(self):
        self.raw["profile"]["experience"] = "expert"
        self.assertEqual(self.run_input()["onboarding"]["steps"][0]["task_id"], "setup")

    def test_all_completed(self):
        self.raw["profile"]["completed_task_ids"] = ["setup", "identity-guide", "login-practice"]
        result = self.run_input()
        self.assertEqual(result["onboarding"]["status"], "complete")
        self.assertEqual(result["onboarding"]["steps"], [])

    def test_topological_order_with_shared_prerequisite(self):
        self.raw["categories"][0]["task_ids"].append("invoice-practice")
        result = self.run_input()
        ordered = set()
        for step in result["onboarding"]["steps"]:
            self.assertTrue(set(step["prerequisites"]) <= ordered)
            self.assertNotIn(step["task_id"], ordered)
            ordered.add(step["task_id"])
        self.assertEqual(len(ordered), 4)

    def test_unknown_and_duplicate_ids(self):
        original = copy.deepcopy(self.raw)
        for collection in ("sources", "claims", "categories", "tasks"):
            with self.subTest(collection=collection):
                self.raw = copy.deepcopy(original)
                self.raw[collection].append(copy.deepcopy(self.raw[collection][0]))
                self.assert_invalid()
        self.raw = copy.deepcopy(original)
        self.raw["sources"][0]["statements"].append(
            copy.deepcopy(self.raw["sources"][0]["statements"][0]))
        self.assert_invalid()
        changes = (
            lambda r: r["claims"][0].update(source_id="unknown"),
            lambda r: r["claims"][0].update(statement_id="unknown"),
            lambda r: r["sources"][0]["statements"][0].update(category_id="unknown"),
            lambda r: r["tasks"][0].update(prerequisites=["unknown"]),
            lambda r: r["categories"][0].update(task_ids=["unknown"]),
            lambda r: r["profile"].update(completed_task_ids=["unknown"]),
            lambda r: r["profile"].update(blocked_task_ids=["unknown"]),
            lambda r: r["categories"][0].update(task_ids=["setup", "setup"]),
            lambda r: r["profile"].update(available_resources=["a", "a"]),
        )
        for change in changes:
            self.raw = copy.deepcopy(original)
            change(self.raw)
            self.assert_invalid()

    def test_unsupported_claims(self):
        for field in ("subject", "predicate", "value", "category_id"):
            before = copy.deepcopy(self.raw)
            self.raw["claims"][0][field] = "unsupported fictional assertion"
            self.assert_invalid()
            self.raw = before
        self.raw["sources"].append({"id": "empty", "title": "FICTIONAL empty", "statements": []})
        self.raw["claims"][0]["source_id"] = "empty"
        self.assert_invalid()

    def test_cycles_including_unused_tasks(self):
        self.raw["tasks"][0]["prerequisites"] = ["login-practice"]
        self.assert_invalid()
        self.raw["tasks"][0]["prerequisites"] = []
        self.raw["tasks"].append({"id": "unused-cycle", "title": "FICTIONAL cycle",
                                  "prerequisites": ["unused-cycle"], "required_resources": []})
        self.assert_invalid()

    def test_completed_profile_integrity(self):
        self.raw["profile"]["completed_task_ids"] = ["login-practice"]
        self.assert_invalid()
        self.raw["profile"].update(completed_task_ids=["setup"], blocked_task_ids=["setup"])
        self.assert_invalid()

    def test_malformed_schema(self):
        original = copy.deepcopy(self.raw)
        for raw in (None, [], {}, "not an object"):
            with self.assertRaises(ValidationError):
                run_pipeline(raw)
        changes = (
            lambda r: r.update(schema_version=True),
            lambda r: r.update(sources={}),
            lambda r: r.update(unexpected=True),
            lambda r: r.update(profile=[]),
            lambda r: r["ticket"].update(question=" "),
            lambda r: r["ticket"].update(urgency=[]),
            lambda r: r["categories"][0].update(owner=""),
            lambda r: r["tasks"][0].update(prerequisites="setup"),
            lambda r: r["search"].update(top_k=True),
            lambda r: r["search"].update(top_k=0),
            lambda r: r["search"].update(ambiguity_margin=float("nan")),
            lambda r: r["search"].update(ambiguity_margin=-0.1),
            lambda r: r["search"].update(ambiguity_margin=1.1),
            lambda r: r["profile"].update(experience="genius"),
            lambda r: r["profile"].update(preferred_format="telepathy"),
        )
        for change in changes:
            self.raw = copy.deepcopy(original)
            change(self.raw)
            self.assert_invalid()

    def test_strict_json(self):
        for serialized in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{bad'):
            with self.assertRaises(ValueError):
                load_json(serialized)

    def test_grounded_wording(self):
        def fixture(context):
            evidence_id = next(iter(context["evidence"]))
            return {"evidence_ids": [evidence_id], "text": context["evidence"][evidence_id]}

        before = self.run_input()
        after = self.run_input(wording=fixture)
        self.assertEqual(before["triage"], after["triage"])
        self.assertEqual(before["onboarding"], after["onboarding"])
        self.assertEqual(after["wording"]["text"], after["search"]["hits"][0]["text"])

    def test_unsupported_wording_is_rejected(self):
        callbacks = (
            lambda c: {"evidence_ids": list(c["evidence"]), "text": "FICTIONAL made-up guarantee."},
            lambda c: {"evidence_ids": ["unknown"], "text": ""},
            lambda c: {"evidence_ids": list(c["evidence"]) * 2, "text": ""},
            lambda c: "FICTIONAL invented plain response",
        )
        for callback in callbacks:
            with self.assertRaises(ValidationError):
                self.run_input(wording=callback)

    def test_no_evidence_wording_cannot_invent(self):
        self.raw["claims"] = []
        result = self.run_input(wording=lambda c: {"evidence_ids": [], "text": ""})
        self.assertEqual(result["wording"]["text"], "")
        with self.assertRaises(ValidationError):
            self.run_input(wording=lambda c: {"evidence_ids": [], "text": "FICTIONAL invention"})

    def test_determinism_and_no_input_mutation(self):
        original = copy.deepcopy(self.raw)
        first = self.run_input()
        self.assertEqual(first, self.run_input())
        self.assertEqual(original, self.raw)
        self.raw["claims"].reverse()
        self.raw["tasks"].reverse()
        self.assertEqual(first, self.run_input())

    def test_cli_success_json(self):
        proc = subprocess.run([sys.executable, "-B", "implementation.py", "example_input.json"],
                              cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(json.loads(proc.stdout)["triage"]["status"], "routed")

    def test_cli_error_json(self):
        proc = subprocess.run([sys.executable, "-B", "implementation.py",
                               "fictional-nonexistent-input.json"],
                              cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["error"]["type"], "validation_error")
        self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main()
