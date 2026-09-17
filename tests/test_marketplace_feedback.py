"""No-network feedback-loop, ordered-contract and campaign safety tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from test_marketplace_agents import config, request_data, run  # noqa: F401
from token_yield import marketplace_agents as engine
from token_yield.marketplace_agent_contracts import (
    ATOMS, ROLES, custom_contract, numeric_features, source_documents, validate_message, workload_prompt,
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


@pytest.mark.parametrize("field,value", [("cap_usd", 51), ("stop_usd", 50), ("stop_usd", 0),
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
    assert runtime.cap_usd == 50 and runtime.stop_usd == 48
    with pytest.raises(ValueError, match="three unchanged"):
        run(runtime, request_data, tmp_path / "runs")
    assert provider.calls == []
    old = engine.AgentRuntime(tmp_path / "old", config, dispatch=provider)
    before = (old.state_dir / "budget.json").read_bytes()
    with pytest.raises(RuntimeError, match="mutation"):
        engine.AgentRuntime(old.state_dir, config, campaign=enabled, dispatch=provider)
    assert (old.state_dir / "budget.json").read_bytes() == before
    with pytest.raises(RuntimeError, match="mutation"):
        engine.AgentRuntime(runtime.state_dir, config, campaign={**enabled, "stop_usd": 47}, dispatch=provider)


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
    assert status["cap_usd"] == 50 and status["stop_usd"] == 48
    assert status["reserved_usd"] > 0 and status["enabled"] is False
    provider = MockProvider()
    restarted = engine.AgentRuntime(runtime.state_dir, config, campaign=enabled, dispatch=provider)
    assert restarted.public_status()["reserved_usd"] == status["reserved_usd"]
    with pytest.raises(RuntimeError, match="audit"):
        run(restarted, request, tmp_path / "runs", run_id="blocked")
    assert provider.calls == []
    assert not (runtime.state_dir / "current.json").exists()
