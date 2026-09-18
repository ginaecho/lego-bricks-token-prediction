"""Offline integration: real local fits and protocol fixtures, never paid calls."""

import json
import random
import itertools
from dataclasses import replace

import pytest

from test_marketplace_agents import MockProvider, approve_establishment, config, request_data, run  # noqa: F401
from token_yield import marketplace_agents as engine
from token_yield.measurement_policy import ROUNDS
from token_yield.measurement_policy import MeasurementPolicy
from token_yield.marketplace_agent_contracts import contracts, custom_contract, FEATURE_BUILDERS
from token_yield.marketplace_measurement import AdaptiveMeasurements, select_actions


def test_real_execution_rewards_frozen_holdouts_and_restart(tmp_path, config, request_data):
    provider = MockProvider()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=provider, measurement_policy=True)
    result, events, _ = run(runtime, request_data, tmp_path / "runs")
    report = result["measurement_policy"]
    assert report["rounds"] == ROUNDS and report["policy"]["updates"] == ROUNDS
    assert report["policy"]["source"] == "mocked-test-provider"
    assert report["policy"]["cost_basis"] == "simulated-rate-card"
    assert len(report["actions"]) == 3
    assert result["composition"]["measured_combinations"] is False
    assert result["training"]["pilot_published"]
    root = tmp_path / "runs" / "test-run" / "agent-artifacts"
    calibration = json.loads((root / "measurement-policy" / "calibration.json").read_text())["rows"]
    train = [r for r in result["training"]["rows"] if r["split"] == "train"]
    holdout = [r for r in result["training"]["rows"] if r["split"] == "holdout"]
    assert {r["group"] for r in train} == {"train-0", "train-1", "train-2"}
    assert all(r["group"].endswith("-3") for r in calibration)
    assert all(r["group"].endswith(("-4", "-5")) for r in holdout)
    frozen = json.loads((root / "final-parameters-before-holdouts.json").read_text())
    candidate = json.loads((root / "candidate-model.json").read_text())
    assert frozen["models"] == candidate["models"]
    assert set(frozen["train_ids"]).isdisjoint(r["id"] for r in calibration + holdout)
    last_feedback = max(i for i, e in enumerate(events) if e["message"] == "Calibration reward recorded.")
    first_holdout = min(i for i, e in enumerate(events) if e["data"].get("split") == "holdout")
    assert last_feedback < first_holdout
    compound = next(key for key in report["actions"] if key.startswith("ordered_"))
    compound_rows = [r for r in train + holdout + calibration if r["brick_id"] == compound]
    assert compound_rows
    for row in compound_rows:
        calls = row["measurements"]
        assert len(calls) == 2
        assert [c["role"] for c in calls] == [b["id"] for b in report["actions"][compound]]
        assert json.loads(calls[0]["prompt"])["documents"] == json.loads(calls[1]["prompt"])["documents"]
        assert row["input_tokens"] == sum(c["usage"]["input_tokens"] for c in calls)
        assert row["rated_usd"] == sum(c["rated_usd"] for c in calls)
        assert all(c["source"] == "mocked-test-provider" and c["status"] == "validated" for c in calls)
    assert result["workload"]["calls"] == len(result["measurements"])
    restored = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider(), measurement_policy=True)
    assert restored.public_status()["version"] == result["training"]["version"]
    second, _, _ = run(restored, request_data, tmp_path / "runs", run_id="second")
    assert second["measurement_policy"]["policy"]["updates"] == 2 * ROUNDS
    assert second["training"]["reused_train_count"] > 0
    assert second["training"]["reused_holdout_count"] == 0
    default = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider())
    assert default.public_status()["version"] is None


def test_novel_last_is_a_selectable_action(tmp_path, config):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider(), measurement_policy=True)
    existing = contracts()[:2]
    novel = custom_contract("Retention exception ledger", existing[0]["atoms"])
    adapter = AdaptiveMeasurements(
        runtime=runtime, bricks=[*existing, novel], names=list(FEATURE_BUILDERS["atoms_context_v1"]),
        source="mocked-test-provider", run_id="novel-last", run_dir=tmp_path / "runs",
        call=lambda *a, **k: pytest.fail("No provider needed for action-availability regression"),
        event=lambda *a: None, cancel=lambda: None, write=engine._write, read=engine._read)
    assert novel["id"] in adapter.actions
    assert novel["id"] in [b["id"] for b in adapter.actions[adapter.compound_id]]


def test_action_order_bound_and_policy_identity(tmp_path, config):
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider(), measurement_policy=True)
    existing = contracts()[:3]
    novel = custom_contract("Retention exception ledger", existing[0]["atoms"])
    def build(bricks, required=None):
        return AdaptiveMeasurements(
            runtime=runtime, bricks=bricks, requested_id=required,
            names=list(FEATURE_BUILDERS["atoms_context_v1"]), source="mocked-test-provider",
            run_id="availability", run_dir=tmp_path / "runs",
            call=lambda *a, **k: pytest.fail("No dispatch in action-schema test"), event=lambda *a: None,
            cancel=lambda: None, write=engine._write, read=engine._read)
    first = build([*existing, novel], novel["id"])
    for permutation in itertools.permutations([*existing, novel]):
        other = build(list(permutation), novel["id"])
        assert other.policy.path == first.policy.path
        assert other.actions == first.actions
        assert novel["id"] in other.spec["selection"]["required"]
        assert len(other.spec["selection"]["excluded"]) == 2
        assert len(other.actions) == 3
    # Description-detected then reused capability can have a different admission
    # rationale without losing scores for the identical executable actions.
    assert build([novel, *existing]).policy.path == first.policy.path
    changed = custom_contract("Different requested ledger", existing[0]["atoms"])
    assert build([*existing, changed], changed["id"]).policy.path != first.policy.path
    mutated = {**novel, "contract_hash": engine.fingerprint({"changed": novel["contract_hash"]})}
    assert build([*existing, mutated], novel["id"]).policy.path != first.policy.path
    with pytest.raises(ValueError, match="not in the supported"):
        select_actions(existing, novel["id"])
    with pytest.raises(ValueError, match="more than two required"):
        select_actions([custom_contract(f"Ledger {i}", novel["atoms"]) for i in range(3)])
    # Explicit reuse of an ordinary contract is also mandatory.
    singles, _ = select_actions([*existing, novel], existing[-1]["id"])
    assert existing[-1]["id"] in [b["id"] for b in singles]


@pytest.mark.parametrize("mode", ["explicit", "description"])
def test_new_action_executions_rewards_and_reordered_resume(tmp_path, config, request_data, mode):
    from token_yield.marketplace_mock import MockProvider as ProtocolFixture
    provider = ProtocolFixture()
    reorder = False
    def dispatch(prompt, **kwargs):
        result = provider(prompt, **kwargs)
        value = json.loads(result.output)
        if reorder and "bricks" in value:
            value["bricks"].reverse()
            return replace(result, output=json.dumps(value))
        return result
    request = {**request_data, "new_function": "Retention exception ledger"} if mode == "explicit" else {
        **request_data, "description": "Summarize source-only handoff obligations and evidence gaps."}
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=dispatch, measurement_policy=True)
    first, events, _ = run(runtime, request, tmp_path / "runs", run_id=mode + "-new",
                           establishment_decision=approve_establishment)
    novel = first["requested_custom"]
    assert novel and first["bricks"][-1]["id"] == novel
    report = first["measurement_policy"]
    assert novel in report["actions"]
    compound = next(key for key in report["actions"] if key.startswith("ordered_"))
    assert novel in [b["id"] for b in report["actions"][compound]]
    choices = [e["data"]["action"] for e in events if e["message"] == "Measurement policy decision."]
    assert novel in choices and compound in choices
    assert report["policy"]["scores"][novel]["n"] >= 1
    assert report["policy"]["scores"][compound]["n"] >= 1
    assert report["policy"]["updates"] == ROUNDS
    calibration_path = (tmp_path / "runs" / (mode + "-new") / "agent-artifacts" /
                        "measurement-policy" / "calibration.json")
    calibration = json.loads(calibration_path.read_text())["rows"]
    assert {row["brick_id"] for row in calibration} == set(report["actions"])
    assert all(call["status"] == "validated" and call["source"] == "mocked-test-provider"
               for row in calibration for call in row["measurements"])
    assert first["composition"]["measured_combinations"] is False
    assert report["selection"]["excluded"]
    reorder = True
    restarted = engine.AgentRuntime(tmp_path / "state", config, dispatch=dispatch, measurement_policy=True)
    second, _, _ = run(restarted, request, tmp_path / "runs", run_id=mode + "-resume")
    assert second["bricks"][0]["id"] == novel
    assert second["measurement_policy"]["policy"]["compatibility"] == report["policy"]["compatibility"]
    assert second["measurement_policy"]["policy"]["updates"] == 2 * ROUNDS
    assert second["measurement_policy"]["actions"] == report["actions"]
    assert second["training"]["reused_train_count"] > 0


def test_dissent_and_paid_mode_do_not_enter_bandit(tmp_path, config, request_data):
    with pytest.raises(ValueError, match="offline mock-only"):
        engine.AgentRuntime(tmp_path / "paid", config, measurement_policy=True)
    assert not (tmp_path / "paid").exists()
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=MockProvider(disagree=True),
                                  measurement_policy=True)
    with pytest.raises(ValueError, match="resolved scope and dissent"):
        run(runtime, request_data, tmp_path / "runs")
    assert not (tmp_path / "state" / "measurement-policy").exists()


@pytest.mark.parametrize("failure", ["cancel", "provider", "budget"])
def test_selected_measurement_failure_never_rewards(tmp_path, config, request_data, failure):
    provider = MockProvider()
    decision_seen, events = False, []
    def dispatch(*args, **kwargs):
        if decision_seen and failure == "provider":
            raise RuntimeError("unknown provider telemetry")
        return provider(*args, **kwargs)
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=dispatch, measurement_policy=True)
    def event(record):
        nonlocal decision_seen
        events.append(record)
        if record["message"] == "Measurement policy decision.":
            decision_seen = True
            if failure == "budget":
                # A finite dispatch ceiling, no ledger reset or mutation.
                monkeypatch.setattr(engine, "MAX_CALLS_PER_RUN", len(provider.calls))
    from pytest import MonkeyPatch
    with MonkeyPatch.context() as monkeypatch:
        with pytest.raises(RuntimeError):
            runtime.run_pipeline(request_data, tmp_path / "runs", run_id="stopped",
                                 on_event=event, before_stage=lambda _: None,
                                 check_cancel=lambda: decision_seen and failure == "cancel")
    from token_yield.measurement_policy import MeasurementPolicy
    import sqlite3
    db_path = next((tmp_path / "state" / "measurement-policy").glob("*.sqlite"))
    with sqlite3.connect(db_path) as db:
        config_row = json.loads(db.execute("SELECT body FROM events WHERE kind='config'").fetchone()[0])
    p = MeasurementPolicy(db_path, compatibility=config_row["compatibility"],
                          source=config_row["source"], actions=config_row["actions"])
    assert p.snapshot()["updates"] == 0 and p.snapshot()["pending"] == []
    assert any(e["message"] == "Measurement policy stopped; reward unavailable." for e in events)
    assert not (tmp_path / "state" / "current.json").exists()
    assert not any(e["message"] == "Calibration reward recorded." for e in events)
    if failure == "provider":
        assert runtime.public_status()["halted"] and runtime.public_status()["reserved_usd"] > 0


def test_compound_second_step_content_failure_is_paid_in_fixture_but_never_rewarded(
        tmp_path, config, request_data, monkeypatch):
    provider = MockProvider()
    selected, first_step = False, False
    choose = MeasurementPolicy.choose
    # Select the compound for this failure test, not to prescribe model outcomes.
    monkeypatch.setattr(MeasurementPolicy, "choose",
                        lambda self, key, context: choose(self, key, context, rng=random.Random(2)))
    def dispatch(*args, **kwargs):
        result = provider(*args, **kwargs)
        if selected and first_step:
            body = json.loads(result.output)
            body["evidence"] = [{"document_id": "outside", "quote": "fabricated"}]
            return replace(result, output=json.dumps(body))
        return result
    events = []
    def event(record):
        nonlocal selected, first_step
        events.append(record)
        if record["message"] == "Measurement policy decision.":
            assert record["data"]["action"].startswith("ordered_")
            selected = True
        if selected and record["message"] == "Ordered measurement step validated.":
            first_step = True
    runtime = engine.AgentRuntime(tmp_path / "state", config, dispatch=dispatch, measurement_policy=True)
    with pytest.raises(engine.ContentContractError):
        runtime.run_pipeline(request_data, tmp_path / "runs", run_id="bad-second",
                             on_event=event, before_stage=lambda _: None)
    status = runtime.public_status()
    assert status["spend_usd"] > 0 and status["reserved_usd"] == 0 and not status["halted"]
    assert not any(e["message"] == "Calibration reward recorded." for e in events)
    evidence = [json.loads(p.read_text()) for p in
                (tmp_path / "runs" / "bad-second" / "agent-artifacts").glob("*.json")]
    rejected = [e for e in evidence if e.get("status") == "content_rejected"]
    assert len(rejected) == 1 and rejected[0]["rated_usd"] > 0
