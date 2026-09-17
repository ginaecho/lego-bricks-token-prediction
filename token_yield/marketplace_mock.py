"""Explicit MOCKED protocol fixtures. Never evidence of LLM quality or real token usage.

The production runtime never imports this module. Only --mock-agents and tests do.
"""

from __future__ import annotations

import json
import time

from .foundry_dispatch import DispatchResult, ResponseCall, TokenUsage
from .marketplace_agent_contracts import ATOMS, ROLES, source_documents


class MockProvider:
    """Rule-based public-output fixture exercising the real pipeline without networking."""

    def __init__(self, delay: float = 0):
        self.calls = []
        self.delay = delay

    def __call__(self, prompt: str, *, target: str, output_cap: int) -> DispatchResult:
        time.sleep(self.delay)
        payload = json.loads(prompt)
        self.calls.append(payload)
        task = payload["task"]
        docs = payload.get("documents", source_documents("train-0", 0))
        evidence = [{"document_id": docs[0]["id"], "quote": docs[0]["text"][:100]}]
        catalog = payload.get("catalog", [])
        selected = [{"id": "extract", "quantity": 1}, {"id": "review", "quantity": 1}]
        selected += [{"id": b["id"], "quantity": 1} for b in catalog if b["id"].startswith("novel_")]
        atoms = dict(zip(ATOMS, (2, 1, 0, 1, 1, 1, 1)))
        if task == "novelty":
            term = payload["custom_function"]
            # Fixture-only synonym normalization, not a live semantic classifier.
            normalized = term.casefold().replace("register", "ledger")
            match = next((b for b in catalog if b["name"].casefold() == normalized), None)
            answer = {"decision": "reuse" if match else "establish",
                      "reuse_id": match["id"] if match else "", "new_name": "" if match else term,
                      "rationale": "MOCK rule: normalize register/ledger and compare names. "
                      "This fixture is not semantic evidence."}
        elif task == "decompose":
            answer = {"atoms": atoms, "rationale": "MOCK bounded document transformation."}
        elif task in ("contract_review", "contract_reconcile"):
            answer = {"atoms": atoms, "rationale": "MOCK review of supplied peer counts.",
                      "agreed": True, "dissent": []}
        elif task == "propose":
            answer = {"summary": "MOCK independent " + payload["role"], "bricks": selected,
                      "evidence": evidence, "limitations": ["Fictional fixture."]}
        elif task == "discuss":
            answer = {"summary": "MOCK peer revision", "bricks": selected, "evidence": evidence,
                      "agreed": True, "dissent": [],
                      "critiques": [{"role": role, "critique": "MOCK source-only scope review."} for role in ROLES]}
        elif task == "adjudicate":
            gap = ("Handoff obligation mapping" if "handoff" in payload["customer_request"].lower()
                   and not any("handoff" in b["name"].lower() for b in catalog) else "")
            answer = {"summary": "MOCK reconciled project", "bricks": selected, "evidence": evidence,
                      "agreed": True, "dissent": [], "unsupported": [], "proposed_new_function": gap,
                      "decisions": [{"id": b["id"], "decision": "include",
                                     "rationale": "MOCK source-bound task."} for b in selected]}
        elif task == "features":
            answer = {"builder": "atoms_context_v1", "rationale": "MOCK selects finite byte descriptors."}
        elif task == "fit":
            best = min(payload["candidate_metrics"], key=lambda m: m["input_mae"] + m["output_mae"])
            answer = {"alpha": best["alpha"], "rationale": "MOCK chooses lowest supplied training CV error."}
        elif task == "metrics":
            answer = {"accepted": True, "summary": "MOCK accepts protocol fixture, not model quality.",
                      "limitations": ["Synthetic token labels; not production measurements."]}
        elif task == "workload":
            answer = {"answer": "MOCK fixture response using the first supplied source.",
                      "evidence": evidence, "limitations": ["Not an LLM execution."]}
            if payload.get("steps"):
                answer["atom_results"] = [{"step_id": step["id"], "result": "MOCK " + step["operation"] +
                                          ": " + docs[0]["text"][:40]} for step in payload["steps"]]
        else:
            raise ValueError("Unknown mocked task")
        text = json.dumps(answer)
        inputs, outputs = len(prompt.encode()) // 4 + 20, len(text.encode()) // 4 + 1
        usage = TokenUsage(inputs, outputs, 0, 0, inputs + outputs)
        call = ResponseCall(target, f"mock-{len(self.calls)}", "completed", "gpt-5.4",
                            1, usage, "mock-request", "mock-response")
        return DispatchResult(text, call.response_id, "completed", "gpt-5.4", 1,
                              usage, (call,), ())
