"""MOCKED-provider tests; no paid Foundry calls or synthetic production labels.

Run pytest with a repository-local ``--basetemp`` to keep artifacts in the worktree.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from token_yield import marketplace_agents as engine
from token_yield.foundry_dispatch import DispatchResult, FoundryDispatchError, ResponseCall, TokenUsage
from token_yield.marketplace_agent_contracts import (
    FEATURE_BUILDERS, ROLES, contracts, numeric_features, source_documents,
    schema_from_example, strict_json, validate_message,
)
from token_yield.marketplace_source_fixtures import fixture_documents


class MockProvider:
    """Deterministic stand-in for protocol testing, explicitly NOT measured evidence."""

    def __init__(self, *, disagree: bool = False, accepted: bool = True):
        self.calls = []
        self.disagree = disagree
        self.accepted = accepted

    def __call__(self, prompt: str, *, target: str, output_cap: int) -> DispatchResult:
        payload = json.loads(prompt)
        self.calls.append({"payload": payload, "target": target, "output_cap": output_cap})
        task = payload["task"]
        docs = payload.get("documents", source_documents("train-0", 0))
        evidence = [{"document_id": docs[0]["id"], "quote": docs[0]["text"]}]
        selected = [{"id": "extract", "quantity": 2}, {"id": "review", "quantity": 1}]
        for brick in payload.get("catalog", []):
            if brick["id"].startswith("novel_"):
                selected.append({"id": brick["id"], "quantity": 1})
        if task == "propose":
            answer = {"summary": f"Independent {payload['role']} conclusions",
                      "bricks": selected, "evidence": evidence, "limitations": ["Proxy only."]}
        elif task == "discuss":
            answer = {"summary": "Revision after all peer proposals", "bricks": selected,
                      "evidence": evidence, "agreed": not self.disagree,
                      "critiques": [{"role": r, "critique": f"Reviewed {r} proposal"} for r in ROLES],
                      "dissent": ["Insufficient evidence"] if self.disagree else []}
        elif task == "adjudicate":
            answer = {"summary": "Explicit reconciliation", "bricks": selected, "evidence": evidence,
                      "agreed": True, "dissent": [], "unsupported": [], "proposed_new_function": "",
                      "decisions": [{"id": b["id"], "decision": "include",
                                     "rationale": "Scoped source task"} for b in selected]}
        elif task == "novelty":
            answer = {"decision": "establish", "reuse_id": "", "new_name": payload["custom_function"],
                      "rationale": "Genuinely new capability not covered by the catalog."}
        elif task == "decompose":
            answer = {"atoms": {"extract": 2, "classify": 1, "score": 0, "plan": 1,
                                "retrieve": 1, "verify": 1, "write": 1},
                      "rationale": "Bounded decomposition over the fixed atom vocabulary."}
        elif task in ("contract_review", "contract_reconcile"):
            answer = {"atoms": {"extract": 2, "classify": 1, "score": 0, "plan": 1,
                                "retrieve": 1, "verify": 1, "write": 1},
                      "rationale": "Reviewed all peer contract proposals.", "agreed": True, "dissent": []}
        elif task == "features":
            answer = {"builder": "atoms_context_v1", "rationale": "Use fixed source-size descriptors."}
        elif task == "fit":
            answer = {"alpha": 1.0, "rationale": "Bounded regularization."}
        elif task == "metrics":
            answer = {"accepted": self.accepted, "summary": "Only a small pilot.", "limitations": []}
        elif task == "workload":
            answer = {"answer": "Owner and missing restoration evidence extracted from fictional sources.",
                      "evidence": evidence, "limitations": ["Not a certification."]}
            if payload.get("steps"):
                answer["atom_results"] = [{"step_id": s["id"], "result": "Mock source-only result"}
                                          for s in payload["steps"]]
        else:
            raise AssertionError(task)
        text = json.dumps(answer)
        # These counts exist solely to test provider handling, not to train a live model.
        inputs = len(prompt.encode("utf-8")) // 4 + 20
        outputs = 120 if task == "workload" else 300
        usage = TokenUsage(inputs, outputs, 0, 0, inputs + outputs)
        call = ResponseCall(target, f"mock-{len(self.calls)}", "completed", engine.MODEL_ID,
                            1, usage, "request", "response")
        return DispatchResult(text, call.response_id, "completed", engine.MODEL_ID, 1,
                              usage, (call,), ())


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({
        "endpoint": engine.APPROVED_ENDPOINT, "deployment": engine.MODEL_ID,
        "expected_response_model": engine.MODEL_ID, "deployment_version": "2026-03-05",
        "deployment_sku": "GlobalStandard", "reasoning_effort": "none", "text_verbosity": "low",
        "pricing": {"input_per_million": 2.5, "cached_input_per_million": .25, "output_per_million": 15.0},
        "cap_usd": 49.8832, "authorization": {"old": "NOT the new approval"},
    }), encoding="utf-8")
    return path


@pytest.fixture
def request_data():
    return {"description": "Review the fictional document requirements and identify evidence gaps.",
            "model_id": "gpt", "runtime": "foundry", "new_function": "",
            "runs_per_month": 10, "execution_mode": "automatic"}


def run(runtime, request, path, run_id="test-run", **kwargs):
    events, stages = [], []
    result = runtime.run_pipeline(request, path, run_id=run_id, on_event=events.append,
                                  before_stage=stages.append, **kwargs)
    return result, events, stages


def approve_establishment(pending, *, actor="OFFLINE TEST approver"):
    return {"decision": "approve", "contract_hash": pending["contract_hash"],
            "actor": actor, "timestamp": "2026-09-17T00:00:00+00:00",
            "reason": "Offline test approval for exact contract hash."}


def reject_establishment(pending, *, actor="OFFLINE TEST rejecter"):
    return {"decision": "reject", "contract_hash": pending["contract_hash"],
            "actor": actor, "timestamp": "2026-09-17T00:00:00+00:00",
            "reason": "Offline test rejection for exact contract hash."}


def test_independent_proposals_actual_peer_discussion_and_separate_usage(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    result, events, stages = run(runtime, request_data, tmp_path / "run")
    assert tuple(stages) == engine.AGENT_STAGES
    proposals = [c["payload"] for c in provider.calls if c["payload"]["task"] == "propose"]
    assert [p["role"] for p in proposals] == list(ROLES)
    assert all("all_proposals" not in p for p in proposals)
    discussions = [c["payload"] for c in provider.calls if c["payload"]["task"] == "discuss"]
    assert len(discussions) == 3
    for payload in discussions:
        assert {p["role"] for p in payload["all_proposals"]} == set(ROLES)
        assert {p["proposal"]["summary"] for p in payload["all_proposals"]} == {
            f"Independent {role} conclusions" for role in ROLES
        }
    assert result["source"] == "mocked-test-provider"
    assert result["model_id"] == "gpt-5.4"
    assert len(result["agents"]) == result["orchestration"]["calls"] == 10
    assert result["workload"]["calls"] == 96
    assert len(result["training"]["rows"]) == 96
    assert all(row["measurement"]["kind"] == "workload" for row in result["training"]["rows"])
    assert result["usage_ledger"]["response_calls"] == 106
    assert result["before"]["supported"] is False and result["before"]["total"] is None
    assert result["before"]["input_tokens"] is None
    assert result["before"]["output_tokens"] is None
    assert result["before"]["total_tokens"] is None
    assert result["after"]["supported"] is True
    forecast = result["after"]
    assert forecast["input_tokens"] == forecast["input"]
    assert forecast["output_tokens"] == forecast["output"]
    assert forecast["total_tokens"] == forecast["total"]
    assert forecast["total"] == pytest.approx(forecast["input"] + forecast["output"])
    assert forecast["usd_per_month"] == pytest.approx(forecast["usd_per_run"] * 10)
    assert forecast["input"] == pytest.approx(sum(
        f["input_tokens"] * b["quantity"] for f, b in zip(forecast["per_brick"], result["bricks"])
    ))
    assert result["training"]["pilot_published"] is True
    assert result["training"]["production_promoted"] is False
    artifact_root = tmp_path / "run" / "test-run"
    assert json.loads((artifact_root / "result.json").read_text())["model_id"] == engine.MODEL_ID
    assert json.loads((artifact_root / "result.json").read_text())["training"]["pilot_published"] is True
    assert [json.loads(line) for line in (artifact_root / "events.jsonl").read_text().splitlines()] == events
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    for message in result["agents"] + result["measurements"]:
        saved = json.loads((artifact_root / message["evidence"]).read_text())
        assert saved["output"] and saved["prompt"] and saved["status"] == "validated"
    assert runtime.public_status()["cap_usd"] == 25
    assert runtime.public_status()["spend_usd"] > 0


def test_catalog_covers_original_variants_with_finite_atoms(config, tmp_path):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    items = runtime.catalog()["items"]
    assert {b["id"] for b in items} == {
        "interests", "behavior", "journey", "normal", "web", "deep", "faq", "triage",
        "semantic", "compare", "feedback", "sentiment", "extract", "review", "guided", "adaptive",
    }
    assert all(not b["supported"] and b["total_tokens"] is None for b in items)
    assert all(b["model_id"] == engine.MODEL_ID and "reference" in b["scope"].lower() for b in items)
    for brick in contracts():
        vector = numeric_features(brick, source_documents("train-0", 0))
        assert set(vector) == set(FEATURE_BUILDERS["atoms_context_v1"])
        assert all(value >= 0 for value in vector.values())


@pytest.mark.parametrize("text", [
    '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}', "[]", '```json\n{}\n```',
])
def test_strict_json_rejects_invalid_numbers_duplicates_and_markdown(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_citation_and_schema_validation():
    docs = source_documents("train-0", 0)
    valid = {"answer": "Draft", "evidence": [{"document_id": docs[0]["id"], "quote": docs[0]["text"]}],
             "limitations": []}
    validate_message("workload", valid, docs, contracts())
    for evidence in ([{"document_id": "invented", "quote": docs[0]["text"]}],
                     [{"document_id": docs[0]["id"], "quote": "invented quote"}], []):
        with pytest.raises(ValueError):
            validate_message("workload", {**valid, "evidence": evidence}, docs, contracts())
    with pytest.raises(ValueError):
        validate_message("workload", {**valid, "surprise": 1}, docs, contracts())
    with pytest.raises(ValueError):
        validate_message("features", {"builder": "__import__('os')", "rationale": "bad"}, docs, contracts())
    with pytest.raises(ValueError):
        validate_message("fit", {"alpha": True, "rationale": "bad"}, docs, contracts())


def test_unknown_usage_reservation_survives_restart_and_blocks(tmp_path, config, request_data):
    def timeout(*args, **kwargs):
        state = json.loads((tmp_path / "state" / "budget.json").read_text())
        assert state["budget"]["active_reserved_usd"] > 0
        raise TimeoutError("mocked uncertain transport")
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=timeout)
    with pytest.raises(TimeoutError):
        run(runtime, request_data, tmp_path / "run")
    status = runtime.public_status()
    assert status["reserved_usd"] > 0 and status["enabled"] is False
    restarted = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    assert restarted.public_status()["reserved_usd"] == status["reserved_usd"]
    with pytest.raises(RuntimeError, match="audit"):
        run(restarted, request_data, tmp_path / "other", run_id="other")
    assert not (tmp_path / "state" / "current.json").exists()


def test_budget_rejects_before_dispatch_and_ignores_old_approval(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, cap_usd=.01, dispatch=provider)
    with pytest.raises(RuntimeError, match="hard budget"):
        run(runtime, request_data, tmp_path / "run")
    assert not provider.calls
    assert runtime.public_status()["spend_usd"] == 0
    with pytest.raises(RuntimeError, match="mutation"):
        engine.AgentRuntime(tmp_path / "state", config, cap_usd=25, dispatch=provider)
    for cap in (26, float("nan"), float("inf"), -1, True):
        with pytest.raises(ValueError):
            engine.AgentRuntime(tmp_path / "other", config, cap_usd=cap, dispatch=provider)


def test_cancel_before_every_dispatch_and_release_unattempted_reservation(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    with pytest.raises(engine.AgentCancelled):
        run(runtime, request_data, tmp_path / "run", check_cancel=lambda: len(provider.calls) >= 2)
    assert len(provider.calls) == 2
    assert runtime.public_status()["reserved_usd"] == 0
    second = engine.AgentRuntime(tmp_path / "state2", config, dispatch=MockProvider())
    with pytest.raises(engine.AgentCancelled):
        run(second, request_data, tmp_path / "run2", check_cancel=lambda:
            second.public_status()["reserved_usd"] > 0)
    assert second.public_status()["reserved_usd"] == 0
    assert second.public_status()["enabled"] is True


@pytest.mark.parametrize("mode", ["output", "input", "reasoning", "model", "incomplete"])
def test_invalid_telemetry_keeps_reservation_and_halts(tmp_path, config, request_data, mode):
    provider = MockProvider()
    def invalid(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        if mode == "model":
            return replace(result, model="unapproved")
        if mode == "incomplete":
            return replace(result, status="incomplete")
        usage = result.usage
        if mode == "output":
            usage = replace(usage, output_tokens=output_cap + 1)
        elif mode == "input":
            usage = replace(usage, input_tokens=10**9)
        else:
            usage = replace(usage, reasoning_tokens=output_cap + 1)
        return replace(result, usage=usage)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=invalid)
    with pytest.raises(ValueError):
        run(runtime, request_data, tmp_path / "run")
    assert runtime.public_status()["reserved_usd"] > 0
    assert runtime.public_status()["enabled"] is False
    assert len(provider.calls) == 1
    restarted = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    with pytest.raises(RuntimeError, match="audit"):
        run(restarted, request_data, tmp_path / "retry", run_id="retry")
    assert len(provider.calls) == 1
    evidence = next((tmp_path / "run" / "test-run" / "agent-artifacts").glob("*.json"))
    assert json.loads(evidence.read_text())["output"]


def test_citation_matches_across_whitespace_but_not_paraphrase():
    docs = source_documents("train-0", 0)
    words = docs[0]["text"].split()
    assert len(words) >= 3
    reflowed = f"{words[0]}   {words[1]}\n\t{words[2]}"  # identical words, different whitespace
    base = {"answer": "ok", "limitations": ["source only"]}
    validate_message("workload", {**base, "evidence": [
        {"document_id": docs[0]["id"], "quote": reflowed}]}, docs, contracts())
    with pytest.raises(ValueError, match="exact supplied document span"):
        validate_message("workload", {**base, "evidence": [
            {"document_id": docs[0]["id"],
             "quote": "fabricated span absent from every supplied source zzz"}]}, docs, contracts())


def test_citation_allows_letter_case_only_for_exact_contiguous_span():
    docs = fixture_documents("archive-exceptions-v2", "train-2", 2)
    base = {"answer": "ok", "limitations": ["source only"]}
    validate_message("workload", {**base, "evidence": [
        {"document_id": docs[0]["id"],
         "quote": "Retain master media for 12 calendar months after project completion."},
        {"document_id": docs[0]["id"],
         "quote": "Retain invoice records for 24 calendar months after invoice closure."}]},
        docs, contracts())
    with pytest.raises(ValueError, match="exact supplied document span"):
        validate_message("workload", {**base, "evidence": [
            {"document_id": docs[0]["id"],
             "quote": "Retain master media after project completion for 12 calendar months."}]},
            docs, contracts())
    with pytest.raises(ValueError, match="exact supplied document span"):
        validate_message("workload", {**base, "evidence": [
            {"document_id": docs[0]["id"],
             "quote": "Retain invoice records for 24 calendar months after invoice closure. "
                      "INV-30 is an invoice closed 2024-05-15, owner Chen."}]},
            docs, contracts())


def _reject_workload_citation(result):
    answer = json.loads(result.output)
    answer["evidence"] = [{"document_id": answer["evidence"][0]["document_id"],
                           "quote": "fabricated span absent from every supplied source zzz"}]
    return replace(result, output=json.dumps(answer))


INSTRUCTION_TEXT_CITATION = (
    "For the source-only capability 'Retention exception ledger', execute each ordered atom step once "
    "on the supplied documents. Later steps may use earlier results. Return one result per step, even "
    "if evidence is missing; do not invent source facts."
)


def _replace_first_citation(result, quote):
    answer = json.loads(result.output)
    answer["evidence"] = [{"document_id": answer["evidence"][0]["document_id"], "quote": quote}]
    return replace(result, output=json.dumps(answer))


def _mutate_json_output(result, mutate):
    answer = json.loads(result.output)
    mutate(answer)
    return replace(result, output=json.dumps(answer))


def _stage_calls(provider, task):
    return [c for c in provider.calls if c["payload"]["task"] == task]


def _retry_events(events):
    return [e for e in events if e["message"] ==
            "Bounded stage retry authorized after content/citation rejection."]


def test_settled_workload_content_failure_retries_then_never_halts(tmp_path, config, request_data):
    provider = MockProvider()
    def always_bad(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        return _reject_workload_citation(result) if json.loads(prompt)["task"] == "workload" else result
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=always_bad)
    with pytest.raises(engine.ContentContractError):
        run(runtime, request_data, tmp_path / "run")
    status = runtime.public_status()
    assert status["reserved_usd"] == 0        # the call settled; nothing is retained
    assert status["halted"] is None           # a settled content failure never halts the campaign
    assert status["spend_usd"] > 0            # rejected generations were honestly paid for
    assert status["enabled"] is True          # the campaign can still run
    assert not (tmp_path / "state" / "current.json").exists()
    workload_calls = [c for c in provider.calls if c["payload"]["task"] == "workload"]
    assert len(workload_calls) == engine.WORKLOAD_ATTEMPTS  # bounded retry on one job, then fail


def test_bounded_retry_recovers_a_transient_citation_failure(tmp_path, config, request_data):
    provider = MockProvider()
    seen = {"n": 0}
    def flaky(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        if json.loads(prompt)["task"] == "workload" and seen["n"] < 1:
            seen["n"] += 1
            return _reject_workload_citation(result)
        return result
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=flaky)
    result, events, _ = run(runtime, request_data, tmp_path / "run")
    assert result["training"]["pilot_published"] is True
    assert runtime.public_status()["halted"] is None
    assert result["workload"]["calls"] == 96   # only conforming measurements become rows
    workload_calls = [c for c in provider.calls if c["payload"]["task"] == "workload"]
    assert len(workload_calls) == 97           # 96 accepted + 1 rejected retry, each paid
    assert result["usage_ledger"]["response_calls"] == 107
    retry_events = _retry_events(events)
    assert retry_events and retry_events[0]["data"]["retry_policy"] == (
        engine.CONTENT_RETRY_POLICY["policy"])
    assert retry_events[0]["data"]["error_category"] == "source_citation"


@pytest.mark.parametrize("stage,expected_calls", [
    ("propose", 4),
    ("discuss", 4),
    ("adjudicate", 2),
    ("workload", 97),
])
def test_bounded_retry_recovers_transient_bad_citation_at_each_cited_stage(
        tmp_path, config, request_data, stage, expected_calls):
    provider = MockProvider()
    seen = {"bad": False}
    def flaky(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        if json.loads(prompt)["task"] == stage and not seen["bad"]:
            seen["bad"] = True
            return _replace_first_citation(result, INSTRUCTION_TEXT_CITATION)
        return result
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=flaky)
    result, events, _ = run(runtime, request_data, tmp_path / "run")
    assert result["training"]["pilot_published"] is True
    assert len(_stage_calls(provider, stage)) == expected_calls
    assert any(e["data"]["operation"] == stage
               and e["data"]["retry_policy"] == engine.CONTENT_RETRY_POLICY["policy"]
               and e["data"]["error_category"] == "source_citation"
               for e in _retry_events(events))


@pytest.mark.parametrize("mutation", [
    lambda answer: answer.update(evidence="not-an-array"),
    lambda answer: answer.update(evidence=[]),
    lambda answer: answer.pop("summary"),
])
def test_schema_failure_fails_closed_without_stage_retry(
        tmp_path, config, request_data, mutation):
    provider = MockProvider()
    def malformed_schema(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        return (_mutate_json_output(result, mutation)
                if json.loads(prompt)["task"] == "propose" else result)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=malformed_schema)
    events = []
    with pytest.raises(engine.ContentContractError):
        runtime.run_pipeline(request_data, tmp_path / "run", run_id="test-run",
                             on_event=events.append, before_stage=lambda stage: None)
    assert len(_stage_calls(provider, "propose")) == 1
    assert _retry_events(events) == []
    state = json.loads((tmp_path / "state" / "budget.json").read_text())
    assert state["calls"] == 1 and len(state["budget"]["settled_requests"]) == 1
    assert state["budget"]["active_reserved_usd"] == 0 and state["halted"] is None


@pytest.mark.parametrize("mutation", [
    lambda answer: (
        answer.update(evidence=[{"document_id": answer["evidence"][0]["document_id"],
                                 "quote": INSTRUCTION_TEXT_CITATION}]),
        answer.update(limitations="not-array"),
    ),
    lambda answer: answer.update(evidence=[
        {"document_id": answer["evidence"][0]["document_id"], "quote": INSTRUCTION_TEXT_CITATION},
        {"document_id": answer["evidence"][0]["document_id"], "quote": 27},
    ]),
    lambda answer: answer.update(evidence=[
        {"document_id": answer["evidence"][0]["document_id"], "quote": 27},
        {"document_id": answer["evidence"][0]["document_id"], "quote": INSTRUCTION_TEXT_CITATION},
    ]),
])
def test_mixed_schema_and_citation_errors_fail_closed_before_retry(
        tmp_path, config, request_data, mutation):
    provider = MockProvider()
    def mixed_error(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        return (_mutate_json_output(result, mutation)
                if json.loads(prompt)["task"] == "propose" else result)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=mixed_error)
    events = []
    with pytest.raises(engine.ContentContractError) as excinfo:
        runtime.run_pipeline(request_data, tmp_path / "run", run_id="test-run",
                             on_event=events.append, before_stage=lambda stage: None)
    assert excinfo.value.category == "schema" and excinfo.value.retryable is False
    assert len(_stage_calls(provider, "propose")) == 1
    assert _retry_events(events) == []
    state = json.loads((tmp_path / "state" / "budget.json").read_text())
    assert state["calls"] == 1 and len(state["budget"]["settled_requests"]) == 1
    assert state["budget"]["active_reserved_usd"] == 0 and state["halted"] is None


def test_paraphrase_citation_retries_then_recovers(tmp_path, config, request_data):
    provider = MockProvider()
    seen = {"bad": False}
    def paraphrase_once(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        if json.loads(prompt)["task"] == "propose" and not seen["bad"]:
            seen["bad"] = True
            return _replace_first_citation(
                result, "Retain master media after project completion for 12 calendar months.")
        return result
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=paraphrase_once)
    result, events, _ = run(runtime, request_data, tmp_path / "run")
    assert result["training"]["pilot_published"] is True
    assert len(_stage_calls(provider, "propose")) == 4
    assert any(e["data"]["operation"] == "propose" for e in _retry_events(events))


def test_case_drift_citation_needs_no_retry(tmp_path, config, request_data):
    provider = MockProvider()
    def case_drift(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        if json.loads(prompt)["task"] == "propose":
            answer = json.loads(result.output)
            return _replace_first_citation(result, answer["evidence"][0]["quote"].lower())
        return result
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=case_drift)
    result, events, _ = run(runtime, request_data, tmp_path / "run")
    assert result["training"]["pilot_published"] is True
    assert len(_stage_calls(provider, "propose")) == 3
    assert _retry_events(events) == []


@pytest.mark.parametrize("stage", ["propose", "discuss", "adjudicate", "workload"])
def test_persistent_instruction_text_citation_fails_after_stage_retry_bound(
        tmp_path, config, request_data, stage):
    provider = MockProvider()
    def always_bad(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        return (_replace_first_citation(result, INSTRUCTION_TEXT_CITATION)
                if json.loads(prompt)["task"] == stage else result)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=always_bad)
    with pytest.raises(engine.ContentContractError, match="exact supplied document span"):
        run(runtime, request_data, tmp_path / "run")
    assert runtime.public_status()["halted"] is None
    assert runtime.public_status()["reserved_usd"] == 0
    assert len(_stage_calls(provider, stage)) == engine.CONTENT_CONTRACT_ATTEMPTS


def test_f70d_instruction_text_propose_failure_retries_and_recovers(
        tmp_path, config, request_data):
    provider = MockProvider()
    seen = {"bad": False}
    def flaky_propose(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        if json.loads(prompt)["task"] == "propose" and not seen["bad"]:
            seen["bad"] = True
            return _replace_first_citation(result, INSTRUCTION_TEXT_CITATION)
        return result
    request = {**request_data, "new_function": "Retention exception ledger"}
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=flaky_propose)
    result, events, _ = run(runtime, request, tmp_path / "run",
                            establishment_decision=approve_establishment)
    assert result["training"]["pilot_published"] is True
    assert len(_stage_calls(provider, "propose")) == 4
    assert any(e["data"]["operation"] == "propose" for e in _retry_events(events))


def test_f70d_instruction_text_propose_failure_fails_after_bound(
        tmp_path, config, request_data):
    provider = MockProvider()
    def always_bad_propose(prompt, *, target, output_cap):
        result = provider(prompt, target=target, output_cap=output_cap)
        return (_replace_first_citation(result, INSTRUCTION_TEXT_CITATION)
                if json.loads(prompt)["task"] == "propose" else result)
    request = {**request_data, "new_function": "Retention exception ledger"}
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=always_bad_propose)
    with pytest.raises(engine.ContentContractError, match="exact supplied document span"):
        run(runtime, request, tmp_path / "run", establishment_decision=approve_establishment)
    assert len(_stage_calls(provider, "propose")) == engine.CONTENT_CONTRACT_ATTEMPTS


def test_incomplete_response_with_known_usage_settles_then_retries(tmp_path, config, request_data, monkeypatch):
    provider = MockProvider()
    truncated = {"done": False}
    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        result = provider(payload["input"], target="mock", output_cap=payload["max_output_tokens"])
        if json.loads(payload["input"])["task"] == "workload" and not truncated["done"]:
            truncated["done"] = True
            result = replace(result, status="incomplete")  # truncated at the cap; usage still reported
        return raw_response(result)
    monkeypatch.setattr(engine, "_default_transport", transport)
    runtime = engine.AgentRuntime(tmp_path / "state", config, token_provider=lambda: "mock-token")
    result, events, _ = run(runtime, request_data, tmp_path / "runs")
    status = runtime.public_status()
    assert status["halted"] is None            # known usage is settled; never an unknown-telemetry halt
    assert status["reserved_usd"] == 0         # nothing is left reserved
    assert result["training"]["pilot_published"] is True
    assert result["workload"]["calls"] == 96   # the truncated output never becomes a training row
    assert result["usage_ledger"]["response_calls"] == 107   # truncated call was paid for, then retried
    assert any(e["data"]["error_category"] == "response_protocol" for e in _retry_events(events))


def test_holdout_not_seen_in_selection_and_not_fitted(tmp_path, config, request_data, monkeypatch):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    actual_fit = runtime._fit
    fitted_groups = []
    def guarded(rows, names, target, alpha):
        assert all(row["split"] == "train" for row in rows)
        fitted_groups.extend(row["group"] for row in rows)
        return actual_fit(rows, names, target, alpha)
    monkeypatch.setattr(runtime, "_fit", guarded)
    result, _, _ = run(runtime, request_data, tmp_path / "run")
    assert fitted_groups and all(group.startswith("train-") for group in fitted_groups)
    for call in provider.calls:
        if call["payload"]["task"] in ("features", "fit"):
            assert "holdout-" not in json.dumps(call["payload"])
            assert "test_count" not in call["payload"]
    split = json.loads((tmp_path / "run" / "test-run" / "agent-artifacts" / "frozen-split.json").read_text())
    assert len([j for j in split["jobs"] if j["split"] == "holdout"]) == 32
    assert result["training"]["train_count"] == 64
    assert result["training"]["test_count"] == 32


def test_novel_measurement_reuses_train_only_and_restart_model_persists(tmp_path, config, request_data):
    first_provider = MockProvider()
    first = engine.AgentRuntime(tmp_path / "state", config, dispatch=first_provider)
    result, _, _ = run(first, request_data, tmp_path / "first", run_id="first")
    second_provider = MockProvider()
    second = engine.AgentRuntime(tmp_path / "state", config, dispatch=second_provider)
    assert second.public_status()["version"] == result["training"]["version"]
    new_request = {**request_data, "new_function": "Policy document parsing"}
    newer, _, _ = run(second, new_request, tmp_path / "second", run_id="second",
                      establishment_decision=approve_establishment)
    assert newer["before"]["supported"] is False
    assert newer["after"]["supported"] is True
    assert newer["training"]["reused_train_count"] == 64
    assert newer["workload"]["calls"] == 38
    assert newer["training"]["new_holdout_count"] == 34
    assert newer["training"]["reused_holdout_count"] == 0
    assert len(newer["catalog"]) == 17
    assert newer["training"]["version"] != result["training"]["version"]
    novel_rows = [r for r in newer["training"]["rows"] if r["brick_id"].startswith("novel_")]
    assert len(novel_rows) == 6 and all(r["run_id"] == "second" for r in novel_rows)
    assert all("first" not in r["group"] for r in newer["training"]["rows"] if r["split"] == "holdout")


def test_novelty_reuse_when_agent_matches_existing_brick(tmp_path, config, request_data):
    provider = MockProvider()
    def reuse(prompt, **kwargs):
        result = provider(prompt, **kwargs)
        if json.loads(prompt)["task"] == "novelty":
            answer = {"decision": "reuse", "reuse_id": "review", "new_name": "",
                      "rationale": "Already covered by the document review brick."}
            return replace(result, output=json.dumps(answer))
        return result
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=reuse)
    result, events, _ = run(runtime, {**request_data, "new_function": "Documentation gap checker"},
                            tmp_path / "run")
    assert result["requested_custom"] == "review"       # mapped to the existing brick
    assert len(result["catalog"]) == 16                 # nothing new established
    assert not any(c["payload"]["task"] == "decompose" for c in provider.calls)
    assert result["capability_reviews"][0]["outcome"] == "reused"


def test_novelty_establishes_agent_decomposed_brick_and_persists_for_reuse(tmp_path, config, request_data):
    request = {**request_data, "new_function": "Compliance evidence mapper"}
    first_provider = MockProvider()
    first = engine.AgentRuntime(tmp_path / "state", config, dispatch=first_provider)
    result, events, _ = run(first, request, tmp_path / "first", run_id="first",
                            establishment_decision=approve_establishment)
    novel = next(b for b in result["catalog"] if b["novel"])
    assert result["requested_custom"] == novel["id"]
    assert novel["atoms"] == {"extract": 2, "classify": 1, "score": 0, "plan": 1,
                              "retrieve": 1, "verify": 1, "write": 1}   # agent-chosen decomposition
    assert novel["supported"] is True and novel["input_tokens"] is not None
    assert len(result["catalog"]) == 17 and result["training"]["pilot_published"] is True
    assert any(c["payload"]["task"] == "decompose" for c in first_provider.calls)
    assert result["capability_reviews"][0]["outcome"] == "established"
    # The established brick persists; the same request reuses its measured train rows next run.
    second = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    assert any(b["novel"] and b["supported"] for b in second.catalog()["items"])
    newer, _, _ = run(second, request, tmp_path / "second", run_id="second")
    assert len(newer["catalog"]) == 17
    assert newer["training"]["reused_train_count"] == 68   # 64 standard + 4 established-brick rows
    assert newer["training"]["new_holdout_count"] == 34


def test_human_establishment_gate_requires_callback_before_measurement(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    with pytest.raises(RuntimeError, match="human establishment approval callback required"):
        run(runtime, {**request_data, "new_function": "Governed retention ledger"}, tmp_path / "run")
    assert not any(c["payload"]["task"] == "workload" for c in provider.calls)
    assert not (tmp_path / "state" / "current.json").exists()


def test_human_establishment_approval_establishes_measures_and_publishes_new_brick(
        tmp_path, config, request_data):
    provider = MockProvider()
    approvals = []
    def approve(pending):
        approvals.append(pending)
        return approve_establishment(pending, actor="Ada Approver")
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    result, events, _ = run(runtime, {**request_data, "new_function": "Governed retention ledger"},
                            tmp_path / "run", establishment_decision=approve)
    novel = next(b for b in result["catalog"] if b["novel"])
    assert result["requested_custom"] == novel["id"]
    assert result["capability_reviews"][0]["outcome"] == "established"
    assert result["capability_reviews"][0]["human_establishment_decision"]["actor"] == "Ada Approver"
    assert approvals[0]["contract_hash"] == novel["contract_hash"]
    assert len([r for r in result["training"]["rows"] if r["brick_id"] == novel["id"]]) == 6
    assert novel["supported"] is True and result["training"]["pilot_published"] is True
    assert any(e["message"] == "Human approved new capability establishment." for e in events)


def test_human_establishment_rejection_uses_existing_proxy_bricks(
        tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    result, events, _ = run(runtime, {**request_data, "new_function": "Governed retention ledger"},
                            tmp_path / "run", establishment_decision=reject_establishment)
    assert result["requested_custom"] is None
    assert not any(b["novel"] for b in result["catalog"])
    assert result["capability_reviews"][0]["outcome"] == "human_rejected"
    assert all(not row["brick_id"].startswith("novel_") for row in result["training"]["rows"])
    assert any(e["message"] == "Human rejected new capability establishment." for e in events)


def test_human_establishment_preserves_resolved_and_unresolved_notes(
        tmp_path, config, request_data):
    provider = MockProvider()
    captured = []
    def dissenting(prompt, **kwargs):
        result = provider(prompt, **kwargs)
        payload = json.loads(prompt)
        if payload["task"] == "contract_review":
            answer = json.loads(result.output)
            answer["dissent"] = ["Scoring remains out of scope and should be score zero."]
            return replace(result, output=json.dumps(answer))
        return result
    def approve(pending):
        captured.append(pending)
        return approve_establishment(pending)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=dissenting)
    result, _, _ = run(runtime, {**request_data, "new_function": "Governed retention ledger"},
                       tmp_path / "run", establishment_decision=approve)
    notes = captured[0]["notes"]
    assert [note["note"] for note in notes[:3]] == [
        "Scoring remains out of scope and should be score zero."] * 3
    assert {note["classification"] for note in notes[:3]} == {"resolved_advisory"}
    assert result["capability_reviews"][0]["discussion"][0]["dissent"] == [
        "Scoring remains out of scope and should be score zero."]


@pytest.mark.parametrize("note", [
    "Required source evidence is missing; retention periods cannot be grounded.",
    "Scoring is required; zero scoring makes this contract infeasible.",
    "Scoring is excluded, but the downstream interface cannot accept the required records.",
])
def test_human_establishment_classifies_substantive_agreed_true_notes_as_blockers(
        tmp_path, config, request_data, note):
    provider = MockProvider()
    captured = []
    def dissenting(prompt, **kwargs):
        result = provider(prompt, **kwargs)
        payload = json.loads(prompt)
        if payload["task"] == "contract_review":
            answer = json.loads(result.output)
            answer["dissent"] = [note]
            return replace(result, output=json.dumps(answer))
        return result
    def approve(pending):
        captured.append(pending)
        return approve_establishment(pending)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=dissenting)
    run(runtime, {**request_data, "new_function": "Governed retention ledger"},
        tmp_path / "run", establishment_decision=approve)
    assert captured[0]["unresolved_substantive"]
    assert {entry["note"] for entry in captured[0]["unresolved_substantive"]} == {note}
    assert {entry["classification"] for entry in captured[0]["notes"][:3]} == {
        "unresolved_substantive"}


def test_human_establishment_surfaces_substantive_objection_and_rejects_stale_hash(
        tmp_path, config, request_data):
    provider = MockProvider()
    captured = []
    def objecting(prompt, **kwargs):
        result = provider(prompt, **kwargs)
        payload = json.loads(prompt)
        if payload["task"] == "contract_review" and payload["role"] == "skeptical_reviewer":
            answer = json.loads(result.output)
            answer.update(agreed=False, dissent=["Interface feasibility remains unresolved."])
            return replace(result, output=json.dumps(answer))
        return result
    def stale(pending):
        captured.append(pending)
        decision = approve_establishment(pending)
        decision["contract_hash"] = "0" * 64
        return decision
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=objecting)
    with pytest.raises(ValueError, match="exact contract hash"):
        run(runtime, {**request_data, "new_function": "Governed retention ledger"},
            tmp_path / "run", establishment_decision=stale)
    assert captured[0]["unresolved_substantive"][0]["note"] == "Interface feasibility remains unresolved."
    assert captured[0]["unresolved_substantive"][0]["classification"] == "unresolved_substantive"


def test_server_establishment_gate_waits_and_rejects_stale_contract_hash(
        tmp_path, config, request_data):
    from examples.marketplace_demo_server import RunStore, StageConflictError

    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    store = RunStore(tmp_path / "runs", agent_runtime=runtime, stage_timeout=15)
    run_id = store.submit({**request_data, "new_function": "Governed retention ledger"})
    deadline = time.monotonic() + 10
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = store.get(run_id)
        if snapshot and snapshot["status"] == "waiting":
            break
        time.sleep(0.05)
    assert snapshot["next_stage"] == "human_establishment_approval"
    pending = snapshot["establishment_request"]
    assert pending["contract_hash"]
    assert not any(c["payload"]["task"] == "workload" for c in provider.calls)
    with pytest.raises(ValueError, match="exact contract_hash"):
        store.decide_establishment(run_id, {
            "decision": "approve", "contract_hash": "0" * 64, "actor": "Mallory"})
    store.decide_establishment(run_id, {
        "decision": "approve", "contract_hash": pending["contract_hash"], "actor": "Ada"})
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        snapshot = store.get(run_id)
        if snapshot and snapshot["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert snapshot["status"] == "completed"
    assert snapshot["result"]["requested_custom"].startswith("novel_")


def test_deterministic_guardrail_vetoes_exact_duplicate_establish(tmp_path, config, request_data):
    provider = MockProvider()   # tries to establish new_name == custom_function
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    result, events, _ = run(runtime, {**request_data, "new_function": "Deep research"}, tmp_path / "run")
    assert result["requested_custom"] == "deep"          # exact normalized name, not lexical overlap
    assert len(result["catalog"]) == 16                  # establish blocked
    assert not any(c["payload"]["task"] == "decompose" for c in provider.calls)
    assert result["capability_reviews"][0]["rationale"] == "Exact normalized name guardrail."


@pytest.mark.parametrize("external,disagree,accepted,publishes", [
    (True, False, True, True), (False, True, True, True), (False, False, False, False)])
def test_project_scope_blocks_forecast_but_catalog_publishes_on_accepted_metrics(
    tmp_path, config, request_data, external, disagree, accepted, publishes
):
    provider = MockProvider(disagree=disagree, accepted=accepted)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    if external:
        request_data = {**request_data, "description": "Perform live web research for current security regulations."}
    result, _, _ = run(runtime, request_data, tmp_path / "run")
    # The whole-project forecast stays blocked on any project-scope or measurement problem.
    assert result["after"]["supported"] is False
    assert result["after"]["total"] is None and result["after"]["usd_per_run"] is None
    # Catalog/model publication depends only on measured-pilot integrity (metric acceptance),
    # not on a single custom project's scope or decomposition dissent.
    assert (tmp_path / "state" / "current.json").exists() is publishes
    assert result["training"]["pilot_published"] is publishes
    if publishes:
        published = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
        supported = [b for b in published.catalog()["items"] if b["supported"]]
        assert supported, "measured standard bricks should publish supported forecasts"
        assert all(b["pilot_version"] == result["training"]["version"] for b in supported)
    else:
        assert not (tmp_path / "state" / "current.json").exists()
    if disagree:
        assert result["dissent"]
        assert all(m["public_output"]["dissent"] for m in result["agents"] if m["kind"] == "discuss")


def test_concurrent_ownership_config_mutation_and_repeat_ids_fail_closed(tmp_path, config, request_data):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    with engine._exclusive(runtime.state_dir):
        with pytest.raises(RuntimeError, match="owner"):
            engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    altered = json.loads(config.read_text())
    altered["endpoint"] = "https://unapproved.example/openai/v1"
    config.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="changed"):
        run(runtime, request_data, tmp_path / "run")
    with pytest.raises(ValueError, match="approved"):
        engine.AgentRuntime(tmp_path / "other-state", config, dispatch=MockProvider())


def test_catalog_feature_bounds_suppress_extrapolation(tmp_path, config, request_data):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    run(runtime, request_data, tmp_path / "run")
    model = runtime._latest()
    model["feature_bounds"] = [[0, 0] for _ in model["feature_names"]]
    prediction = runtime._forecast_brick(contracts()[0], model)
    assert prediction["supported"] is False and prediction["total_tokens"] is None
    assert "support" in prediction["reason"]


def test_real_adapter_preflight_is_mocked_no_network(tmp_path, config, request_data, monkeypatch):
    observed = []
    def transport(url, headers, body, timeout):
        observed.append(json.loads(body))
        state = json.loads((tmp_path / "state" / "budget.json").read_text())
        assert state["budget"]["active_reserved_usd"] > 0
        assert observed[-1]["max_output_tokens"] == engine.AGENT_OUTPUT_CAP
        raise TimeoutError("MOCKED transport; deliberately no network")
    monkeypatch.setattr(engine, "_default_transport", transport)
    runtime = engine.AgentRuntime(tmp_path / "state", config, token_provider=lambda: "mock-secret")
    with pytest.raises(FoundryDispatchError, match="Responses API request failed"):
        run(runtime, request_data, tmp_path / "run")
    assert len(observed) == 1
    assert runtime.public_status()["reserved_usd"] > 0
    assert "mock-secret" not in json.dumps(runtime.public_status())


def test_root_artifact_contract_and_request_before_first_callback(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config,
                                  approval_id="marketplace-agent-pilot-usd25-v1", dispatch=provider)
    root = tmp_path / "runs"
    stages = []
    def gate(stage):
        stages.append(stage)
        artifact_root = root / "server-run"
        assert json.loads((artifact_root / "request.json").read_text()) == request_data
        (artifact_root / "run.json").write_text(json.dumps({"stage": stage}))
        raise engine.AgentCancelled("MOCKED stage gate stops before first call")
    with pytest.raises(engine.AgentCancelled):
        runtime.run_pipeline(request_data, root, run_id="server-run",
                             on_event=lambda event: None, before_stage=gate)
    assert stages == ["novelty"] and not provider.calls
    assert not (root / "request.json").exists()
    assert runtime.public_status()["approval_id"] == "marketplace-agent-pilot-usd25-v1"


def test_close_is_idempotent_read_only_and_stops_new_runs(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    budget_file = tmp_path / "state" / "budget.json"
    original = budget_file.read_bytes()
    runtime.close()
    runtime.close()
    assert runtime.public_status()["closed"] and not runtime.public_status()["enabled"]
    assert len(runtime.catalog()["items"]) == 16
    assert all(item["feature_id"] for item in runtime.catalog()["items"])
    with pytest.raises(RuntimeError, match="closed"):
        run(runtime, request_data, tmp_path / "runs")
    assert budget_file.read_bytes() == original and not provider.calls


def test_status_and_catalog_are_read_only_during_dispatch_and_close_stops_next_call(
    tmp_path, config, request_data
):
    provider = MockProvider()
    entered, release = threading.Event(), threading.Event()
    errors = []
    def blocked(prompt, **kwargs):
        entered.set()
        assert release.wait(10), "test did not release mocked provider"
        return provider(prompt, **kwargs)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=blocked)
    def worker():
        try:
            run(runtime, request_data, tmp_path / "runs")
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(10)
        budget_file = tmp_path / "state" / "budget.json"
        original = budget_file.read_bytes()
        for _ in range(3):
            assert runtime.public_status()["reserved_usd"] > 0
            assert len(runtime.catalog()["items"]) == 16
        assert budget_file.read_bytes() == original
        runtime.close()
    finally:
        release.set()
        thread.join(timeout=15)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], engine.AgentCancelled)
    assert len(provider.calls) == 1
    assert runtime.public_status()["reserved_usd"] == 0


def test_duplicate_or_unselected_include_decisions_rejected():
    docs = source_documents("train-0", 0)
    value = {"summary": "Decision", "bricks": [{"id": "extract", "quantity": 1}],
             "evidence": [{"document_id": docs[0]["id"], "quote": docs[0]["text"]}],
             "agreed": True, "dissent": [], "unsupported": [], "proposed_new_function": "",
             "decisions": [{"id": "extract", "decision": "include", "rationale": "Evidence task"}]}
    validate_message("adjudicate", value, docs, contracts())
    for extra in (
        {"id": "extract", "decision": "exclude", "rationale": "Conflicting decision"},
        {"id": "review", "decision": "include", "rationale": "Missing from selection"},
    ):
        with pytest.raises(ValueError):
            validate_message("adjudicate", {**value, "decisions": value["decisions"] + [extra]},
                             docs, contracts())


@pytest.mark.parametrize("conflict", ["different_revisions", "pending_review"])
def test_agreement_flags_cannot_hide_unresolved_decisions(tmp_path, config, request_data, conflict):
    provider = MockProvider()
    def disputed(prompt, **kwargs):
        result = provider(prompt, **kwargs)
        payload, answer = json.loads(prompt), json.loads(result.output)
        if (conflict == "different_revisions" and payload["task"] == "discuss"
                and payload["role"] == "architect"):
            answer["bricks"][0]["quantity"] = 3
        if conflict == "pending_review" and payload["task"] == "adjudicate":
            answer["decisions"].append({"id": "normal", "decision": "review",
                                        "rationale": "Scope not resolved"})
        return replace(result, output=json.dumps(answer))
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=disputed)
    result, _, _ = run(runtime, request_data, tmp_path / "runs")
    # Fabricated agreement cannot make the whole-project forecast supported when
    # peer revisions differ or an orchestrator decision is still pending review.
    assert not result["after"]["supported"]
    assert result["after"]["total"] is None
    # The independently measured brick catalog still publishes on accepted metrics;
    # the unresolved decision only blocks the custom project's whole forecast.
    assert result["training"]["pilot_published"] is True
    assert (tmp_path / "state" / "current.json").exists()


def test_measured_rate_separates_cache_and_reasoning_safety(tmp_path, config, request_data):
    provider = MockProvider()
    def usage_details(prompt, **kwargs):
        result = provider(prompt, **kwargs)
        usage = replace(result.usage, cached_tokens=100, reasoning_tokens=20)
        return replace(result, usage=usage,
                       response_calls=(replace(result.response_calls[0], usage=usage),))
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=usage_details)
    with pytest.raises(engine.AgentCancelled):
        run(runtime, request_data, tmp_path / "runs", check_cancel=lambda: len(provider.calls) == 1)
    evidence = json.loads(next((tmp_path / "runs" / "test-run" / "agent-artifacts").glob("*.json")).read_text())
    usage = evidence["usage"]
    assert evidence["rated_usd"] == pytest.approx(
        ((usage["input_tokens"] - 100) * 2.5 + 100 * .25 + usage["output_tokens"] * 15) / 1_000_000
    )
    assert evidence["safety_usd"] == pytest.approx(
        (usage["input_tokens"] * 20 + (usage["output_tokens"] + 20) * 200) / 1_000_000
    )


def test_repeat_ids_do_not_overwrite_artifacts_or_repeat_paid_calls(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    with pytest.raises(engine.AgentCancelled):
        run(runtime, request_data, tmp_path / "runs", check_cancel=lambda: len(provider.calls) == 1)
    with pytest.raises(RuntimeError, match="already attempted"):
        run(runtime, request_data, tmp_path / "runs")
    assert len(provider.calls) == 1


def test_missing_budget_cannot_reset_existing_agent_state(tmp_path, config):
    state = tmp_path / "state"
    state.mkdir()
    (state / "current.json").write_text('{"version":"existing-evidence"}')
    with pytest.raises(RuntimeError, match="missing durable budget"):
        engine.AgentRuntime(state, config, dispatch=MockProvider())
    assert not (state / "budget.json").exists()


def test_real_dispatcher_uses_measured_usage_and_no_network(tmp_path, config, request_data, monkeypatch):
    calls = []
    docs = source_documents("train-0", 0)
    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        assert payload["model"] == engine.MODEL_ID
        calls.append(payload)
        public = {"summary": "Measured public statement", "bricks": [{"id": "extract", "quantity": 1}],
                  "evidence": [{"document_id": docs[0]["id"], "quote": docs[0]["text"]}],
                  "limitations": ["Fictional source-only proxy"]}
        return 200, json.dumps({
            "id": "mock-response", "status": "completed", "model": engine.MODEL_ID,
            "usage": {"input_tokens": 321, "output_tokens": 123, "total_tokens": 444,
                      "input_tokens_details": {"cached_tokens": 20},
                      "output_tokens_details": {"reasoning_tokens": 0}},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(public)}]}],
        }).encode()
    monkeypatch.setattr(engine, "_default_transport", transport)
    runtime = engine.AgentRuntime(tmp_path / "state", config, token_provider=lambda: "mock-secret")
    with pytest.raises(engine.AgentCancelled):
        run(runtime, request_data, tmp_path / "runs", check_cancel=lambda: bool(calls))
    evidence = json.loads(next((tmp_path / "runs" / "test-run" / "agent-artifacts").glob("*.json")).read_text())
    assert evidence["source"] == "measured-foundry"
    assert evidence["status"] == "validated"
    assert evidence["usage"]["input_tokens"] == 321
    assert evidence["usage"]["output_tokens"] == 123
    assert evidence["usage"]["cached_tokens"] == 20
    assert len(calls) == 1 and runtime.public_status()["reserved_usd"] == 0


def assert_strict_schema(schema):
    if schema["type"] == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for child in schema["properties"].values():
            assert_strict_schema(child)
    elif schema["type"] == "array":
        assert_strict_schema(schema["items"])
    else:
        assert schema["type"] in {"string", "number", "integer", "boolean"}


def raw_response(result):
    usage = result.usage
    return 200, json.dumps({
        "id": result.response_id, "status": result.status, "model": result.model,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                  "total_tokens": usage.total_tokens,
                  "input_tokens_details": {"cached_tokens": usage.cached_tokens},
                  "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens}},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": result.output}]}],
    }).encode()


def test_all_actual_transport_calls_use_strict_schemas_and_full_body_reservations(
    tmp_path, config, request_data, monkeypatch
):
    provider, seen = MockProvider(), []
    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        contract = json.loads(payload["input"])
        fmt = payload["text"]["format"]
        assert payload["store"] is False and fmt["type"] == "json_schema" and fmt["strict"] is True
        assert fmt["name"] == "marketplace_" + contract["task"]
        assert fmt["schema"] == schema_from_example(contract["output_contract"])
        assert_strict_schema(fmt["schema"])
        state = json.loads((tmp_path / "state" / "budget.json").read_text())
        call_id, reserved = next(iter(state["budget"]["active_reservations"].items()))
        evidence = json.loads((tmp_path / "runs" / "test-run" / "agent-artifacts" / f"{call_id}.json").read_text())
        assert evidence["input_bound"] == len(body) + 4096
        assert evidence["request_payload"] == payload
        assert evidence["request_sha256"] == hashlib.sha256(body).hexdigest()
        assert reserved == pytest.approx(engine.SafetyRateCard().price(
            len(body) + 4096, payload["max_output_tokens"] * 2))
        seen.append(contract["task"])
        return raw_response(provider(payload["input"], target="mock", output_cap=payload["max_output_tokens"]))
    monkeypatch.setattr(engine, "_default_transport", transport)
    runtime = engine.AgentRuntime(tmp_path / "state", config, token_provider=lambda: "mock-token")
    result, _, _ = run(runtime, request_data, tmp_path / "runs")
    assert len(seen) == 106 and seen.count("workload") == 96
    assert set(seen) == {"propose", "discuss", "adjudicate", "features", "workload", "fit", "metrics"}
    assert result["training"]["source"] == "measured-foundry"
    assert result["after"]["source"] == "measured-foundry"
    assert all(item["source"] == "measured-foundry" for item in result["after"]["per_brick"])
    assert runtime.catalog()["source"] == "measured-foundry"
    assert all(item["source"] == "measured-foundry" for item in runtime.catalog()["items"])
    assert result["orchestration"]["calls"] == 10 and result["workload"]["calls"] == 96


def test_schema_helper_bounds_and_empty_text_arrays():
    schema = schema_from_example({"dissent": [], "bricks": [{"id": "catalog ID", "quantity": 1}],
                                  "agreed": True, "alpha": 1.0})
    assert_strict_schema(schema)
    assert schema["properties"]["dissent"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["alpha"]["type"] == "number"
    assert schema["properties"]["agreed"]["type"] == "boolean"
    for invalid in ([], {"x": None}, {"x": [1, "mixed"]}, {"x": float("inf")}, {"x": ["a"] * 21}):
        with pytest.raises(ValueError):
            schema_from_example(invalid)
    nested = {"x": "value"}
    for _ in range(6):
        nested = {"x": nested}
    with pytest.raises(ValueError, match="complexity"):
        schema_from_example(nested)


@pytest.mark.parametrize("mode", ["oversize_schema", "changed_after_reservation", "cancel_after_auth"])
def test_full_payload_guard_and_actual_predispatch_cancellation(
    tmp_path, config, request_data, monkeypatch, mode
):
    actual = engine.StructuredDispatcher.initial_payload
    builds, authenticated, sends = [], [], []
    def initial(self, prompt):
        payload = actual(self, prompt)
        builds.append(payload)
        if mode == "oversize_schema":
            payload["text"]["format"]["schema"]["description"] = "X" * engine.MAX_INPUT_BOUND
        elif mode == "changed_after_reservation" and len(builds) > 1:
            payload["max_output_tokens"] += 1
        return payload
    def token():
        authenticated.append(True)
        return "mock-token"
    def transport(*args):
        sends.append(args)
        raise AssertionError("preflight must reject without network")
    monkeypatch.setattr(engine.StructuredDispatcher, "initial_payload", initial)
    monkeypatch.setattr(engine, "_default_transport", transport)
    runtime = engine.AgentRuntime(tmp_path / "state", config, token_provider=token)
    with pytest.raises((ValueError, RuntimeError)):
        run(runtime, request_data, tmp_path / "runs",
            check_cancel=lambda: mode == "cancel_after_auth" and bool(authenticated))
    assert not sends
    state = json.loads((tmp_path / "state" / "budget.json").read_text())
    assert state["budget"]["settled_safety_usd"] == state["budget"]["active_reserved_usd"] == 0
    assert state["halted"] is None
    if mode == "oversize_schema":
        assert state["calls"] == 0 and not authenticated


@pytest.fixture
def settled_schema_failure(tmp_path, config, request_data, monkeypatch, request):
    provider = MockProvider()
    legacy = getattr(request, "param", "strict") == "legacy"
    if legacy:
        monkeypatch.setattr(engine.StructuredDispatcher, "initial_payload",
                            engine.FoundryDispatcher.initial_payload)
    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        result = provider(payload["input"], target="mock", output_cap=payload["max_output_tokens"])
        output = json.loads(result.output)
        output["catalog ID"] = "review"
        del output["bricks"]
        return raw_response(replace(result, output=json.dumps(output)))
    monkeypatch.setattr(engine, "_default_transport", transport)
    runtime = engine.AgentRuntime(tmp_path / "state", config, token_provider=lambda: "mock-token")
    with pytest.raises(ValueError, match="expected JSON fields"):
        run(runtime, request_data, tmp_path / "runs")
    path = next((tmp_path / "runs" / "test-run" / "agent-artifacts").glob("*.json"))
    evidence = json.loads(path.read_text())
    # A settled content failure no longer halts the campaign. Manual recovery still exists
    # to clear a halt persisted by an earlier engine build, so reconstruct that legacy
    # halted-with-settled-call state (spend recorded, no reservation retained) directly.
    evidence["status"] = "failed"
    evidence["error_type"] = "ValueError"
    if legacy:
        for key in ("transport_protocol", "request_payload", "request_sha256"):
            evidence.pop(key)
    engine._write(path, evidence)
    state_path = tmp_path / "state" / "budget.json"
    state = json.loads(state_path.read_text())
    state["halted"] = ("Dispatch/telemetry/contract failure; no automatic retry. "
                       "Unknown telemetry retains full reservation. Manual audit required.")
    if legacy:
        for key in ("halted_call_id", "halted_run_id", "halted_evidence_sha256"):
            state.pop(key, None)
    else:
        state.update(halted_call_id=evidence["id"], halted_run_id="test-run",
                     halted_evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    engine._write(state_path, state)
    kwargs = {"run_id": "test-run", "call_id": evidence["id"],
              "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "approval_id": runtime.approval_id,
              "reason": "Operator explicitly approved recovery after strict transport protocol repair."}
    return runtime, provider, path, kwargs


@pytest.mark.parametrize("settled_schema_failure", ["strict", "legacy"], indirect=True)
def test_manual_recovery_preserves_spend_evidence_failed_run_and_immutable_audit(
    settled_schema_failure, tmp_path, request_data
):
    runtime, provider, path, kwargs = settled_schema_failure
    original = path.read_bytes()
    before = json.loads((tmp_path / "state" / "budget.json").read_text())
    assert before["halted"] and before["budget"]["active_reserved_usd"] == 0
    outcome = runtime.recover_settled_failure(**kwargs)
    after = json.loads((tmp_path / "state" / "budget.json").read_text())
    assert after["budget"] == before["budget"] and after["calls"] == before["calls"] == 1
    assert after["halted"] is None
    assert outcome["recovered"] is True and outcome["original_run_retry_permitted"] is False
    audit_path = Path(outcome["audit"])
    original_audit = audit_path.read_bytes()
    audit = json.loads(original_audit)
    assert audit["budget_before"] == before and audit["evidence"] == json.loads(original)
    assert path.read_bytes() == original and len(provider.calls) == 1
    with pytest.raises(RuntimeError, match="already attempted"):
        run(runtime, request_data, tmp_path / "runs")
    with pytest.raises(RuntimeError, match="halted"):
        runtime.recover_settled_failure(**kwargs)
    assert audit_path.read_bytes() == original_audit and len(provider.calls) == 1


@pytest.mark.parametrize("tamper", [
    "reviewed_hash", "settlement", "reserved", "identity", "usage", "request_hash", "approval", "call_id",
])
def test_manual_recovery_rejects_unverifiable_failure(settled_schema_failure, tmp_path, tamper):
    runtime, provider, path, kwargs = settled_schema_failure
    state_path = tmp_path / "state" / "budget.json"
    state, evidence = json.loads(state_path.read_text()), json.loads(path.read_text())
    if tamper == "reviewed_hash":
        kwargs["evidence_sha256"] = "0" * 64
    elif tamper == "approval":
        kwargs["approval_id"] = "different-approval"
    elif tamper == "call_id":
        kwargs["call_id"] = "0" * 32
    elif tamper == "settlement":
        state["budget"]["settled_requests"][kwargs["call_id"]] += .01
        state["budget"]["settled_safety_usd"] += .01
        state["budget"]["remaining_to_stop_usd"] -= .01
    elif tamper == "reserved":
        state["budget"]["active_reservations"]["uncertain"] = 1
        state["budget"]["active_reserved_usd"] = 1
        state["budget"]["remaining_to_stop_usd"] -= 1
    else:
        if tamper == "identity":
            evidence["observed_status"] = "incomplete"
        elif tamper == "usage":
            evidence["usage"]["input_tokens"] += 1
        else:
            evidence["response_calls"][0]["request_sha256"] = "0" * 64
        engine._write(path, evidence)
        kwargs["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        state["halted_evidence_sha256"] = kwargs["evidence_sha256"]
    engine._write(state_path, state)
    before = state_path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        runtime.recover_settled_failure(**kwargs)
    assert state_path.read_bytes() == before
    assert len(provider.calls) == 1 and not (tmp_path / "state" / "recoveries").exists()


def test_transport_protocol_invalidates_old_model_without_budget_reset(tmp_path, config):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    original = (tmp_path / "state" / "budget.json").read_bytes()
    legacy_compatibility = engine.fingerprint({
        "pin": runtime._pin, "schema": engine.SCHEMA_VERSION,
        "builders": engine.FEATURE_BUILDERS, "prompt_protocol": engine.CONTRACT_VERSION,
    })
    assert legacy_compatibility != runtime._compatibility
    version = "a" * 32
    model = {"version": version, "compatibility": legacy_compatibility}
    engine._write(tmp_path / "state" / "versions" / f"{version}.json", model)
    engine._write(tmp_path / "state" / "current.json",
                  {"version": version, "sha256": engine.fingerprint(model)})
    restarted = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    assert restarted._latest() is None
    assert all(not item["supported"] for item in restarted.catalog()["items"])
    assert (tmp_path / "state" / "budget.json").read_bytes() == original
