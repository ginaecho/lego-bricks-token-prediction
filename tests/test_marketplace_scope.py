"""Offline authorization engineering; synthetic ledger fixtures, zero dispatch."""

import copy
import json

import pytest

from test_marketplace_agents import config, request_data, run  # noqa: F401
from test_marketplace_feedback import campaign
from token_yield import marketplace_agents as engine
from token_yield.marketplace_scope import prepare_scope, validate_scope, FIXTURE_ID
from token_yield.marketplace_source_fixtures import fixture_scenario


@pytest.fixture
def funding(tmp_path, config):
    approval = {**campaign(), "execution_enabled": True}
    runtime = engine.AgentRuntime(tmp_path / "existing", config, campaign=approval,
                                  token_provider=lambda: pytest.fail("No authentication permitted"))
    ledger = runtime.state_dir / "budget.json"
    state = json.loads(ledger.read_text())
    # Synthetic historical accounting fixture, never the user's ledger.
    budget = engine.HardBudget(50, 48)
    budget._settled = {f"fixture-history-{i}": 1.49220 / 22 for i in range(22)}
    state.update(budget=budget.snapshot(), calls=22)
    ledger.write_text(json.dumps(state), encoding="utf-8")
    return approval, state, runtime.state_dir


def reviewed(scope):
    value = copy.deepcopy(scope)
    value["review"] = {"verdict": "PASS", "reference": "OFFLINE TEST review fixture"}
    value["run_authorization"] = {"approved": True, "reference": "OFFLINE TEST user consent fixture"}
    return value


def test_preparation_disabled_and_two_independent_gates(funding):
    approval, state, _ = funding
    before = copy.deepcopy(state)
    scope = prepare_scope(approval, state)
    assert scope["run_authorization"]["approved"] is False
    assert scope["review"]["verdict"] == "PENDING"
    assert scope["measurement_policy_enabled"] is False
    assert state == before
    with pytest.raises(ValueError, match="independent"):
        validate_scope(scope, approval, state)
    scope["review"] = reviewed(scope)["review"]
    with pytest.raises(ValueError, match="subsequent explicit"):
        validate_scope(scope, approval, state)
    validate_scope(reviewed(scope), approval, state)


@pytest.mark.parametrize("field,value", [
    ("cap_usd", 51), ("stop_usd", 49), ("funding", "new-campaign"),
    ("source_fixture", "archive-exceptions"), ("source_fingerprint", "tampered"),
    ("funding_pin", "another-state"), ("measurement_policy_enabled", True),
])
def test_scope_cannot_broaden_or_relabel(funding, field, value):
    approval, state, _ = funding
    scope = reviewed(prepare_scope(approval, state))
    scope[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        validate_scope(scope, approval, state)


@pytest.mark.parametrize("change", ["clear", "alter", "reserve", "halt", "stop", "calls"])
def test_historical_spend_unknowns_and_stop_are_not_overridden(funding, change):
    approval, state, _ = funding
    scope = reviewed(prepare_scope(approval, state))
    updated = copy.deepcopy(state)
    if change == "clear":
        updated["budget"]["settled_requests"] = {}
    elif change == "alter":
        updated["budget"]["settled_requests"]["fixture-history-0"] += .1
        updated["budget"]["settled_safety_usd"] += .1
    elif change == "reserve":
        updated["budget"]["active_reservations"] = {"unknown": 1}
        updated["budget"]["active_reserved_usd"] = 1
    elif change == "halt":
        updated["halted"] = "unknown telemetry"
    elif change == "stop":
        updated["budget"]["settled_safety_usd"] = 48
    else:
        updated["calls"] = 0
    with pytest.raises((ValueError, RuntimeError)):
        validate_scope(scope, approval, updated)


def test_ready_scope_reuses_exact_ledger_and_binds_v2_without_calls(
        tmp_path, funding, config, request_data):
    approval, state, directory = funding
    scope = reviewed(prepare_scope(approval, state))
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(json.dumps(scope))
    before = (directory / "budget.json").read_bytes()
    runtime = engine.AgentRuntime(directory, config, campaign=approval, source_fixture=FIXTURE_ID,
                                  scope_file=scope_file,
                                  token_provider=lambda: pytest.fail("No authentication permitted"))
    assert runtime._pin == state["pin"]
    assert runtime.public_status()["source"] == "measured-foundry"
    assert runtime.public_status()["version"] is None
    assert runtime._documents("train-0", 0)[0]["id"].startswith(FIXTURE_ID)
    assert (directory / "budget.json").read_bytes() == before
    scenario = fixture_scenario(FIXTURE_ID)
    request = {**request_data, "description": scenario["description"], "new_function": scenario["new_function"]}
    with pytest.raises(engine.AgentCancelled):
        run(runtime, request, tmp_path / "runs", run_id="valid-but-no-dispatch", check_cancel=lambda: True)
    assert (directory / "budget.json").read_bytes() == before
    assert len(list((directory / "scope-authorizations").glob("*.json"))) == 1
    marker = json.loads((directory / "run-ids" / "valid-but-no-dispatch.json").read_text())
    assert marker["scope_authorization_sha256"] == engine.fingerprint(scope)
    with pytest.raises(ValueError, match="exact versioned"):
        run(runtime, request_data, tmp_path / "runs", run_id="wrong-brief")
    assert (directory / "budget.json").read_bytes() == before
    scope["run_authorization"]["approved"] = False
    scope_file.write_text(json.dumps(scope))
    with pytest.raises(RuntimeError, match="authorization changed"):
        run(runtime, request, tmp_path / "runs", run_id="revoked")


def test_no_new_ledger_no_pending_scope_and_no_scope_in_mock(
        tmp_path, funding, config):
    approval, state, directory = funding
    scope_file = tmp_path / "pending.json"
    scope_file.write_text(json.dumps(prepare_scope(approval, state)))
    with pytest.raises(RuntimeError, match="new funding"):
        engine.AgentRuntime(tmp_path / "new-state", config, campaign=approval,
                            source_fixture=FIXTURE_ID, scope_file=scope_file)
    assert not (tmp_path / "new-state").exists()
    before = (directory / "budget.json").read_bytes()
    with pytest.raises(ValueError, match="independent"):
        engine.AgentRuntime(directory, config, campaign=approval, source_fixture=FIXTURE_ID,
                            scope_file=scope_file)
    assert (directory / "budget.json").read_bytes() == before
    with pytest.raises(ValueError, match="requires real"):
        engine.AgentRuntime(directory, config, campaign=approval, source_fixture=FIXTURE_ID,
                            scope_file=scope_file, dispatch=lambda **kwargs: pytest.fail("no dispatch"))


def test_later_settlements_preserve_baseline_without_reset(funding):
    approval, state, _ = funding
    scope = reviewed(prepare_scope(approval, state))
    later = copy.deepcopy(state)
    later["budget"]["settled_requests"]["future-fixture"] = .02
    later["budget"]["settled_safety_usd"] += .02
    later["calls"] += 1
    validate_scope(scope, approval, later)
    assert scope["baseline"] == prepare_scope(approval, state)["baseline"]


def test_cli_pending_scope_never_opens_server_or_authenticates(
        tmp_path, funding, config, monkeypatch, capsys):
    from examples import marketplace_demo_server as server
    approval, state, directory = funding
    scope_file, campaign_file = tmp_path / "scope.json", tmp_path / "campaign.json"
    scope_file.write_text(json.dumps(prepare_scope(approval, state)))
    campaign_file.write_text(json.dumps(approval))
    monkeypatch.setattr(server, "make_server", lambda *a, **kw: pytest.fail("No live server"))
    monkeypatch.setattr(engine, "acquire_entra_token", lambda: pytest.fail("No authentication"))
    monkeypatch.setattr("sys.argv", ["server", "--enable-foundry", "--source-fixture", FIXTURE_ID,
                                    "--scope-file", str(scope_file), "--campaign-file", str(campaign_file),
                                    "--agent-state-dir", str(directory), "--agent-config", str(config)])
    before = (directory / "budget.json").read_bytes()
    assert server.main() == 1
    assert "independent scope review" in capsys.readouterr().err
    assert (directory / "budget.json").read_bytes() == before
