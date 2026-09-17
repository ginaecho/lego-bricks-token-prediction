"""No-network feedback-loop, ordered-contract and campaign safety tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from test_marketplace_agents import config, request_data, run  # noqa: F401
from token_yield import marketplace_agents as engine
from token_yield.marketplace_agent_contracts import (
    ATOMS, ROLES, custom_contract, numeric_features, schema_from_example, source_documents,
    validate_message, workload_prompt,
)
from token_yield.marketplace_mock import MockProvider
from token_yield.marketplace_scenarios import SCENARIOS


@pytest.mark.parametrize("source", ["mocked-test-provider", "measured-foundry", None])
def test_stored_forecast_provenance_is_not_inferred_from_runtime(
        tmp_path, config, request_data, monkeypatch, source):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    result, _, _ = run(runtime, request_data, tmp_path / "runs")
    assert result["after"]["source"] == "mocked-test-provider"
    assert all(item["source"] == "mocked-test-provider" for item in result["after"]["per_brick"])
    model = runtime._latest()
    assert model is not None
    # Exercise metadata response paths, not paid inference or relabeling persisted evidence.
    model = {**model, "source": source}
    monkeypatch.setattr(runtime, "_latest", lambda: model)
    expected = source or "unknown"
    catalog = runtime.catalog()
    assert catalog["source"] == expected
    assert all(item["source"] == expected for item in catalog["items"])
    assert all(item["forecast_mode"] == "reference-context" for item in catalog["items"])
    assert any(item["supported"] for item in catalog["items"])
    forecast = runtime._prediction(result["bricks"], model, 10, [])
    assert forecast["source"] == expected
    assert all(item["source"] == expected for item in forecast["per_brick"])
    if source != "measured-foundry":
        assert all("measured" not in item["reason"] for item in catalog["items"])
    empty = runtime._forecast_brick(engine.contracts()[0], None)
    assert empty["source"] == "unknown" and empty["supported"] is False
    unsupported = runtime._prediction(result["bricks"], model, 10, ["Unsupported scope"])
    assert unsupported["source"] == expected and unsupported["supported"] is False


def test_three_original_scenarios_persistent_loop(tmp_path, config, request_data):
    provider = MockProvider()
    state = tmp_path / "state"
    results = []
    for index, scenario in enumerate(SCENARIOS):
        runtime = engine.AgentRuntime(state, config, dispatch=provider,
                                      campaign={**campaign(), "execution_enabled": True})
        result, events, stages = run(runtime, {
            **request_data, "description": scenario["description"], "new_function": scenario["new_function"],
        }, tmp_path / "runs", run_id=f"scenario-{index}")
        results.append(result)
        assert stages == list(engine.AGENT_STAGES)
        assert result["source"] == "mocked-test-provider"
        assert result["scenario"] == scenario
        assert result["training"]["pilot_published"] is True
        assert result["composition"]["measured_combinations"] is False
        assert result["training"]["reused_holdout_count"] == 0
        assert result["usage_ledger"]["response_calls"] == (
            result["orchestration"]["calls"] + result["workload"]["calls"])
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert any(e["data"].get("state") == "running" for e in events)
        assert any(e["data"].get("features") for e in events)
        assert events[-1]["data"]["published"] is True
        persisted = json.loads((tmp_path / "runs" / f"scenario-{index}" / "result.json").read_text())
        assert persisted["training"]["version"] == result["training"]["version"]
    first, similar, inferred = results
    assert first["capability_reviews"][0]["outcome"] == "established"
    assert similar["capability_reviews"][0]["outcome"] == "reused"
    assert similar["requested_custom"] == first["requested_custom"]
    assert similar["training"]["reused_train_count"] == 68
    assert inferred["request"]["new_function"] == ""
    assert inferred["capability_reviews"][0]["origin"] == "description"
    assert inferred["capability_reviews"][0]["outcome"] == "established"
    assert inferred["requested_custom"] in {brick["id"] for brick in inferred["bricks"]}
    assert inferred["training"]["reused_train_count"] == 68
    assert inferred["training"]["new_holdout_count"] == 36
    assert len({r["training"]["version"] for r in results}) == 3
    for result in (first, inferred):
        contract_calls = [m for m in result["agents"] if m["kind"] == "decompose"]
        reviews = [m for m in result["agents"] if m["kind"] == "contract_review"]
        assert {m["role"] for m in contract_calls} == set(ROLES)
        assert {m["role"] for m in reviews} == set(ROLES)
        assert sum(m["kind"] == "contract_reconcile" for m in result["agents"]) == 1
    assert all("all_proposals" not in c for c in provider.calls if c["task"] == "decompose")
    assert all(len(c["all_proposals"]) == 3 for c in provider.calls if c["task"] == "contract_review")
    budget = json.loads((state / "budget.json").read_text())
    assert not budget["budget"]["active_reservations"]
    assert budget["budget"]["settled_safety_usd"] == pytest.approx(sum(
        r["orchestration"]["safety_usd"] + r["workload"]["safety_usd"] for r in results))


def test_atom_counts_change_executed_contract_and_require_ordered_results():
    atoms = dict.fromkeys(ATOMS, 0)
    atoms["extract"] = 1
    a = custom_contract("Fictional ledger", atoms)
    atoms["verify"] = 2
    b = custom_contract("Fictional ledger", atoms)
    docs = source_documents("train-0", 0)
    assert a["contract_hash"] != b["contract_hash"]
    assert len(a["steps"]) == 1 and len(b["steps"]) == 3
    assert workload_prompt(a, docs) != workload_prompt(b, docs)
    assert numeric_features(b, docs)["verify"] == 2
    provider = MockProvider()
    output = json.loads(provider(workload_prompt(b, docs), target="workload", output_cap=1536).output)
    validate_message("workload", output, docs, [b])
    output["atom_results"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        validate_message("workload", output, docs, [b])


def _mock_contract_stage(tmp_path, config, request_data, provider):
    events = []
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)

    def stop_before_project(stage):
        if stage == "propose":
            raise engine.AgentCancelled("Contract-only test: no later stages")

    with pytest.raises(engine.AgentCancelled, match="Contract-only test"):
        runtime.run_pipeline({
            **request_data, "description": SCENARIOS[0]["description"],
            "new_function": SCENARIOS[0]["new_function"],
        }, tmp_path / "runs", run_id="contract-only", on_event=events.append,
            before_stage=stop_before_project)
    return next(event["data"] for event in reversed(events) if event["data"].get("outcome"))


def test_contract_prompts_distinguish_format_from_final_decisions(tmp_path, config, request_data):
    provider = MockProvider()
    review = _mock_contract_stage(tmp_path, config, request_data, provider)
    assert review["outcome"] == "established"
    contract_calls = [p for p in provider.calls if p["task"] != "novelty"]
    assert len(contract_calls) == 7
    for payload in contract_calls:
        note = payload.get("output_contract_semantics", "")
        assert "Formatting example only" in note
        assert "not proposed atom counts" in note
        assert "not requested votes" in note
        assert set(payload["output_contract"]["atoms"].values()) == {1}
        schema = schema_from_example(payload["output_contract"])
        atom_properties = schema["properties"]["atoms"]["properties"]
        assert all(value == {"type": "integer"} for value in atom_properties.values())
        if payload["task"] != "decompose":
            assert schema["properties"]["agreed"] == {"type": "boolean"}
            assert "final" in payload["instruction"]
            assert "unresolved" in payload["instruction"]
    reviews = [p for p in contract_calls if p["task"] == "contract_review"]
    assert all("rationale" in p["instruction"] and "discarded" in p["instruction"] for p in reviews)
    reconciliation = contract_calls[-1]
    assert reconciliation["task"] == "contract_reconcile"
    assert "carry forward" in reconciliation["instruction"]
    assert len(reconciliation["all_reviews"]) == 3


@pytest.mark.parametrize("role", [*ROLES, "orchestrator"])
@pytest.mark.parametrize("veto", ["false_vote", "dissent", "different_counts", "unexplained_false"])
def test_final_contract_objections_survive_identical_format_and_history(
        tmp_path, config, request_data, role, veto):
    provider = MockProvider()
    final_atoms = {"extract": 3, "classify": 2, "score": 1, "plan": 2,
                   "retrieve": 2, "verify": 2, "write": 2}
    objection = "Unresolved source-only verification boundary."

    def respond(prompt, **kwargs):
        response = provider(prompt, **kwargs)
        payload = json.loads(prompt)
        if payload["task"] in ("contract_review", "contract_reconcile"):
            value = json.loads(response.output)
            value.update(atoms=dict(final_atoms), agreed=True, dissent=[])
            if payload.get("role", "orchestrator") == role:
                if veto == "false_vote":
                    value.update(agreed=False, dissent=[objection])
                elif veto == "unexplained_false":
                    value["agreed"] = False
                elif veto == "dissent":
                    value["dissent"] = [objection]
                else:
                    value["atoms"]["extract"] = 2
            response = replace(response, output=json.dumps(value))
        return response

    if veto == "unexplained_false":
        with pytest.raises(engine.ContentContractError, match="disagreement requires dissent"):
            _mock_contract_stage(tmp_path, config, request_data, respond)
        events = (tmp_path / "runs" / "contract-only" / "events.jsonl").read_text()
        assert "Reconciled contract established" not in events
        return

    review = _mock_contract_stage(tmp_path, config, request_data, respond)
    assert review["outcome"] == "review"
    assert "contract" not in review
    retained = (review["reconciliation"] if role == "orchestrator"
                else next(r for r in review["discussion"] if r["role"] == role))
    if veto == "false_vote":
        assert retained["agreed"] is False and retained["dissent"] == [objection]
    elif veto == "dissent":
        assert retained["agreed"] is True and retained["dissent"] == [objection]
    else:
        assert retained["atoms"]["extract"] == 2
    assert len(provider.calls) == 8


def test_final_consensus_keeps_superseded_independent_proposals(tmp_path, config, request_data):
    provider = MockProvider()
    final_atoms = {"extract": 3, "classify": 2, "score": 1, "plan": 2,
                   "retrieve": 2, "verify": 2, "write": 2}

    def respond(prompt, **kwargs):
        response = provider(prompt, **kwargs)
        payload = json.loads(prompt)
        value = json.loads(response.output)
        if payload["task"] == "decompose":
            value["atoms"]["extract"] = ROLES.index(payload["role"]) + 1
        elif payload["task"] in ("contract_review", "contract_reconcile"):
            value.update(atoms=final_atoms, agreed=True, dissent=[],
                         rationale="MOCK: earlier alternatives resolved by the final allocation.")
        return replace(response, output=json.dumps(value))

    review = _mock_contract_stage(tmp_path, config, request_data, respond)
    assert review["outcome"] == "established"
    assert review["contract"]["atoms"] == final_atoms
    for payload in provider.calls:
        if payload["task"] == "contract_review":
            assert [p["proposal"]["atoms"]["extract"] for p in payload["all_proposals"]] == [1, 2, 3]
        elif payload["task"] == "contract_reconcile":
            assert all(r["review"]["atoms"] == final_atoms for r in payload["all_reviews"])
    artifacts = list((tmp_path / "runs" / "contract-only" / "agent-artifacts").glob("*.json"))
    proposals = [json.loads(path.read_text()) for path in artifacts]
    assert sorted(p["public_output"]["atoms"]["extract"]
                  for p in proposals if p["kind"] == "decompose") == [1, 2, 3]


@pytest.mark.parametrize("mode", ["dissent", "mismatched_counts"])
def test_contract_disagreement_never_establishes_brick(tmp_path, config, request_data, mode):
    provider = MockProvider()
    def disagree(prompt, **kwargs):
        response = provider(prompt, **kwargs)
        payload = json.loads(prompt)
        if payload["task"] == "contract_review" and payload["role"] == "skeptical_reviewer":
            value = json.loads(response.output)
            if mode == "dissent":
                value.update(agreed=False, dissent=["Missing evidence for this contract"])
            else:
                value["atoms"]["extract"] = 3
            response = replace(response, output=json.dumps(value))
        return response
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=disagree)
    result, _, _ = run(runtime, {**request_data, "new_function": "Novel ledger"}, tmp_path / "runs")
    assert result["capability_reviews"][0]["outcome"] == "review"
    assert result["requested_custom"] is None
    assert len(result["catalog"]) == 16
    assert result["after"]["supported"] is False
    assert result["after"]["total"] is None


def test_lexical_overlap_does_not_force_semantic_reuse(tmp_path, config, request_data):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    result, _, _ = run(runtime, {**request_data, "new_function": "Deep research ledger"}, tmp_path / "runs")
    review = result["capability_reviews"][0]
    assert review["matches"][0]["score"] >= .5 and not review["matches"][0]["exact"]
    assert review["outcome"] == "established"


def campaign():
    return json.loads((Path(__file__).resolve().parents[1] / "examples" /
                       "marketplace-feedback-campaign.json").read_text())


def test_prepared_campaign_disabled_and_legacy_cap_unchanged(tmp_path, config):
    with pytest.raises(ValueError, match="execution_enabled"):
        engine.AgentRuntime(tmp_path / "new", config, campaign=campaign(), dispatch=MockProvider())
    assert not (tmp_path / "new").exists()
    with pytest.raises(ValueError, match="US\\$25"):
        engine.AgentRuntime(tmp_path / "legacy", config, cap_usd=50, dispatch=MockProvider())


@pytest.mark.parametrize("field,value", [("cap_usd", 101), ("stop_usd", 97), ("stop_usd", 0),
                                        ("cap_usd", float("nan")), ("cap_usd", True),
                                        ("scenario_ids", ["unapproved"])])
def test_campaign_rejects_unsafe_configuration(tmp_path, config, field, value):
    with pytest.raises(ValueError):
        engine.AgentRuntime(tmp_path / "state", config,
                            campaign={**campaign(), "execution_enabled": True, field: value},
                            dispatch=MockProvider())


def test_campaign_new_identity_restricts_scope_and_cannot_reset_old_ledger(tmp_path, config, request_data):
    provider = MockProvider()
    enabled = {**campaign(), "execution_enabled": True}
    runtime = engine.AgentRuntime(tmp_path / "new", config, campaign=enabled, dispatch=provider)
    assert runtime.cap_usd == 100 and runtime.stop_usd == 96
    with pytest.raises(ValueError, match="three unchanged"):
        run(runtime, request_data, tmp_path / "runs")
    assert provider.calls == []
    old = engine.AgentRuntime(tmp_path / "old", config, dispatch=provider)
    before = (old.state_dir / "budget.json").read_bytes()
    with pytest.raises(RuntimeError, match="mutation"):
        engine.AgentRuntime(old.state_dir, config, campaign=enabled, dispatch=provider)
    assert (old.state_dir / "budget.json").read_bytes() == before
    with pytest.raises(RuntimeError, match="mutation"):
        engine.AgentRuntime(runtime.state_dir, config, campaign={**enabled, "stop_usd": 95}, dispatch=provider)


def test_cancel_at_contract_review_stops_without_publication(tmp_path, config, request_data):
    provider = MockProvider()
    def cancelled():
        return any(c["task"] == "contract_review" for c in provider.calls)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider)
    with pytest.raises(engine.AgentCancelled):
        run(runtime, {**request_data, "new_function": "Novel ledger"}, tmp_path / "runs",
            check_cancel=cancelled)
    assert not (runtime.state_dir / "current.json").exists()
    assert not any(c["task"] == "workload" for c in provider.calls)
    budget = json.loads((runtime.state_dir / "budget.json").read_text())
    assert not budget["budget"]["active_reservations"]


def test_new_50_campaign_retains_unknown_telemetry_and_blocks_restart(tmp_path, config, request_data):
    enabled = {**campaign(), "execution_enabled": True}
    scenario = SCENARIOS[0]
    request = {**request_data, "description": scenario["description"], "new_function": scenario["new_function"]}
    def timeout(*args, **kwargs):
        raise TimeoutError("Unknown MOCK transport outcome")
    runtime = engine.AgentRuntime(tmp_path / "state", config, campaign=enabled, dispatch=timeout)
    with pytest.raises(TimeoutError):
        run(runtime, request, tmp_path / "runs")
    status = runtime.public_status()
    assert status["cap_usd"] == 100 and status["stop_usd"] == 96
    assert status["reserved_usd"] > 0 and status["enabled"] is False
    provider = MockProvider()
    restarted = engine.AgentRuntime(runtime.state_dir, config, campaign=enabled, dispatch=provider)
    assert restarted.public_status()["reserved_usd"] == status["reserved_usd"]
    with pytest.raises(RuntimeError, match="audit"):
        run(restarted, request, tmp_path / "runs", run_id="blocked")
    assert provider.calls == []
    assert not (runtime.state_dir / "current.json").exists()
