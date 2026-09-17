"""Offline authorization engineering; synthetic ledger fixtures, zero dispatch."""

import copy
import json
import os
import shutil
import subprocess
import sys

import pytest

from test_marketplace_agents import config, request_data, run  # noqa: F401
from test_marketplace_feedback import campaign
from token_yield import marketplace_agents as engine
from token_yield.marketplace_scope import (
    prepare_scope, validate_scope, canonical_state_dir, funding_store_identity, FIXTURE_ID,
)
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


def test_scope_rejects_identical_copied_funding_store(tmp_path, funding, config):
    approval, state, original = funding
    scope = reviewed(prepare_scope(approval, state, original))
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(json.dumps(scope))
    clone = tmp_path / "cloned-funding"
    shutil.copytree(original, clone)
    original_before = (original / "budget.json").read_bytes()
    with engine._exclusive(original):
        with pytest.raises((ValueError, RuntimeError), match="funding store"):
            engine.AgentRuntime(clone, config, campaign=approval, source_fixture=FIXTURE_ID,
                                scope_file=scope_file,
                                token_provider=lambda: pytest.fail("No authentication permitted"))
    assert (original / "budget.json").read_bytes() == original_before
    assert (clone / "budget.json").read_bytes() == original_before
    assert not (clone / "run-ids").exists()


def test_run_guard_rejects_rebound_store_before_authorized_marker(tmp_path, funding, config, request_data):
    approval, state, directory = funding
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(json.dumps(reviewed(prepare_scope(approval, state, directory))))
    runtime = engine.AgentRuntime(directory, config, campaign=approval, source_fixture=FIXTURE_ID,
                                  scope_file=scope_file,
                                  token_provider=lambda: pytest.fail("No authentication permitted"))
    clone = tmp_path / "clone"
    shutil.copytree(directory, clone)
    before = (directory / "budget.json").read_bytes()
    runtime.state_dir = clone
    with pytest.raises(ValueError, match="funding store"):
        run(runtime, request_data, tmp_path / "runs", run_id="rebound", check_cancel=lambda: True)
    assert not (clone / "run-ids" / "rebound.json").exists()
    assert not (clone / "scope-authorizations").exists()
    assert (clone / "budget.json").read_bytes() == before
    assert (directory / "budget.json").read_bytes() == before


def test_moved_replaced_and_legacy_stores_require_new_review(tmp_path, funding):
    approval, state, directory = funding
    scope = reviewed(prepare_scope(approval, state, directory))
    old = copy.deepcopy(scope)
    old.pop("funding_store")
    old.update(version=1, scope_id="archive-atelier-v2-existing50-v1")
    with pytest.raises(ValueError, match="versioned"):
        validate_scope(old, approval, state, directory)
    moved = tmp_path / "moved"
    directory.rename(moved)
    with pytest.raises(ValueError, match="funding store"):
        validate_scope(scope, approval, state, moved)
    shutil.copytree(moved, directory)
    with pytest.raises(ValueError, match="funding store"):
        validate_scope(scope, approval, state, directory)


def test_canonical_alias_resume_and_cross_process_lock(tmp_path, funding, config, request_data):
    approval, state, directory = funding
    scope = reviewed(prepare_scope(approval, state, directory))
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(json.dumps(scope))
    (directory / "nested").mkdir()
    aliases = [directory, directory / "nested" / ".."]
    if os.name == "nt":
        aliases.append(type(directory)(str(directory).upper()))
    link = tmp_path / "directory-alias"
    if os.name == "nt":
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                        f"$ErrorActionPreference='Stop'; New-Item -ItemType Junction "
                        f"-Path {quote(link)} -Target {quote(directory)} | Out-Null"],
                       check=True, capture_output=True, text=True, timeout=30)
    else:
        link.symlink_to(directory, target_is_directory=True)
    try:
        aliases.append(link)
        before = (directory / "budget.json").read_bytes()
        scenario = fixture_scenario(FIXTURE_ID)
        request = {**request_data, "description": scenario["description"], "new_function": scenario["new_function"]}
        for index, alias in enumerate(aliases):
            assert funding_store_identity(alias) == scope["funding_store"]
            assert prepare_scope(approval, state, alias) == prepare_scope(approval, state, directory)
            runtime = engine.AgentRuntime(alias, config, campaign=approval, source_fixture=FIXTURE_ID,
                                          scope_file=scope_file,
                                          token_provider=lambda: pytest.fail("No authentication permitted"))
            assert runtime.state_dir == canonical_state_dir(directory)
            with pytest.raises(engine.AgentCancelled):
                run(runtime, request, tmp_path / "runs", run_id=f"alias-{index}", check_cancel=lambda: True)
        with engine._exclusive(directory):
            with pytest.raises(RuntimeError, match="active owner"):
                run(runtime, request, tmp_path / "runs", run_id="concurrent", check_cancel=lambda: True)
            script = """
import json, sys
from pathlib import Path
from token_yield.marketplace_agents import AgentRuntime
from token_yield.marketplace_scope import FIXTURE_ID
def no_auth():
    raise AssertionError("No authentication permitted")
try:
    AgentRuntime(Path(sys.argv[1]), Path(sys.argv[2]), campaign=json.loads(sys.argv[4]),
                 source_fixture=FIXTURE_ID, scope_file=Path(sys.argv[3]), token_provider=no_auth)
except RuntimeError as exc:
    print(exc)
    sys.exit(0 if "owned by another process" in str(exc) else 2)
sys.exit(3)
"""
            child = subprocess.run([sys.executable, "-c", script, str(link), str(config),
                                    str(scope_file), json.dumps(approval)],
                                   capture_output=True, text=True, timeout=30)
            assert child.returncode == 0, child.stdout + child.stderr
            assert "owned by another process" in child.stdout
        assert (directory / "budget.json").read_bytes() == before
        assert len(list((directory / "run-ids").glob("*.json"))) == len(aliases)
        assert not (directory / "run-ids" / "concurrent.json").exists()
    finally:
        if os.name == "nt":
            link.rmdir()
        else:
            link.unlink()


def test_preparation_disabled_and_two_independent_gates(funding):
    approval, state, directory = funding
    before = copy.deepcopy(state)
    scope = prepare_scope(approval, state, directory)
    assert scope["run_authorization"]["approved"] is False
    assert scope["review"]["verdict"] == "PENDING"
    assert scope["measurement_policy_enabled"] is False
    assert state == before
    with pytest.raises(ValueError, match="independent"):
        validate_scope(scope, approval, state, directory)
    scope["review"] = reviewed(scope)["review"]
    with pytest.raises(ValueError, match="subsequent explicit"):
        validate_scope(scope, approval, state, directory)
    validate_scope(reviewed(scope), approval, state, directory)


@pytest.mark.parametrize("field,value", [
    ("cap_usd", 51), ("stop_usd", 49), ("funding", "new-campaign"),
    ("source_fixture", "archive-exceptions"), ("source_fingerprint", "tampered"),
    ("funding_pin", "another-state"), ("measurement_policy_enabled", True),
])
def test_scope_cannot_broaden_or_relabel(funding, field, value):
    approval, state, directory = funding
    scope = reviewed(prepare_scope(approval, state, directory))
    scope[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        validate_scope(scope, approval, state, directory)


@pytest.mark.parametrize("change", ["clear", "alter", "reserve", "halt", "stop", "calls"])
def test_historical_spend_unknowns_and_stop_are_not_overridden(funding, change):
    approval, state, directory = funding
    scope = reviewed(prepare_scope(approval, state, directory))
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
        validate_scope(scope, approval, updated, directory)


def test_ready_scope_reuses_exact_ledger_and_binds_v2_without_calls(
        tmp_path, funding, config, request_data):
    approval, state, directory = funding
    scope = reviewed(prepare_scope(approval, state, directory))
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
    scope_file.write_text(json.dumps(prepare_scope(approval, state, directory)))
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
    approval, state, directory = funding
    scope = reviewed(prepare_scope(approval, state, directory))
    later = copy.deepcopy(state)
    later["budget"]["settled_requests"]["future-fixture"] = .02
    later["budget"]["settled_safety_usd"] += .02
    later["calls"] += 1
    validate_scope(scope, approval, later, directory)
    assert scope["baseline"] == prepare_scope(approval, state, directory)["baseline"]


def test_cli_pending_scope_never_opens_server_or_authenticates(
        tmp_path, funding, config, monkeypatch, capsys):
    from examples import marketplace_demo_server as server
    approval, state, directory = funding
    scope_file, campaign_file = tmp_path / "scope.json", tmp_path / "campaign.json"
    scope_file.write_text(json.dumps(prepare_scope(approval, state, directory)))
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
