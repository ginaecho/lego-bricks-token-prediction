"""Campaign integration tests with synthetic transport, never paid inference."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from token_yield.customer_pilot import (
    execute_campaign, load_config, prepare_campaign, summarize_batching,
)
from token_yield.foundry_count import FoundryCountError
from token_yield.foundry_dispatch import FoundryDispatchError
from token_yield import customer_pilot


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "customer_requests"


@pytest.fixture(autouse=True)
def fake_deployment_probe(monkeypatch):
    metadata = {
        "model": {"name": "gpt-5.4", "version": "2026-03-05"},
        "sku": "GlobalStandard", "state": "Succeeded",
        "versionUpgradeOption": "OnceNewDefaultVersionAvailable",
    }
    monkeypatch.setattr(customer_pilot, "_deployment_probe", lambda: copy.deepcopy(metadata))
    return metadata


def fake_output(project, operations):
    artifacts = {}
    for operation in operations:
        if operation == "report":
            artifacts[operation] = {
                "source_url": project["source_url"],
                "requirement_ids": [r["id"] for r in project["requirements"]],
                "exclusions": project["exclusions"], "delivery_status": "not_executed",
                "summary": "A proposed public-request scope needing evidence and human review.",
            }
            continue
        values = []
        for requirement in project["requirements"]:
            row = {"requirement_id": requirement["id"]}
            if operation == "extract":
                row["evidence"] = requirement["text"]
            elif operation == "classify":
                row.update(brick=requirement["brick"], owner=requirement["owner"])
            else:
                row.update(owner=requirement["owner"], prerequisite_status="unverified",
                           review_required=True,
                           proposed_artifact="Proposed scoping artifact pending human review.")
            values.append(row)
        artifacts[operation] = values
    return json.dumps(artifacts)


class FakeTransport:
    def __init__(self, campaign, run_dir, failure=None):
        self.campaign = campaign
        self.run_dir = run_dir
        self.failure = failure
        self.calls = 0

    def __call__(self, url, headers, body, timeout):
        call = self.campaign["calls"][self.calls]
        self.calls += 1
        budget = json.loads((self.run_dir / "budget.json").read_text())
        assert call["call_id"] in budget["active_reservations"]
        assert url.endswith("/responses")
        payload = json.loads(body)
        assert payload["input"] == call["prompt"]
        assert not payload.get("tools")
        if call["split"] == "holdout":
            frozen = json.loads((self.run_dir / "predictions.json").read_text())
            assert len(frozen["predictions"]) == 20
            models = json.loads((self.run_dir / "models.json").read_text())
            assert "handmade-arcade-strategic-plan" not in json.dumps(
                models["training_project_ids"])
        if self.failure == "http":
            return 429, b'{"error":{"code":"RateLimitReached"}}'
        project = next(p for p in self.campaign["catalog"]["projects"]
                       if p["id"] == call["project_id"])
        text = fake_output(project, call["operations"])
        inp, out = len(payload["input"]) // 4, len(text) // 4
        response = {
            "id": f"fake-{self.calls}", "status": "completed",
            "model": "gpt-5.4", "output_text": text, "output": [],
            "usage": {"input_tokens": inp, "output_tokens": out,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0},
                      "total_tokens": inp + out},
        }
        if self.failure == "usage":
            response.pop("usage")
        if self.failure == "incomplete":
            response["status"] = "incomplete"
        if self.failure == "identity":
            response["model"] = "wrong-model"
        if self.failure and self.failure.startswith("missing:"):
            path = self.failure.removeprefix("missing:").split(".")
            node = response["usage"]
            for key in path[:-1]:
                node = node[key]
            node.pop(path[-1])
        if self.failure == "over-budget":
            response["usage"]["input_tokens"] = 3_000_000
            response["usage"]["total_tokens"] = 3_000_000 + out
        return 200, json.dumps(response).encode()


def test_prepare_is_offline_deterministic_and_work_preserving():
    campaign = prepare_campaign(EXPERIMENT)
    assert campaign == prepare_campaign(EXPERIMENT)
    assert len(campaign["calls"]) == 60
    assert [c["split"] for c in campaign["calls"]] == ["train"] * 40 + ["holdout"] * 20
    assert len({c["call_id"] for c in campaign["calls"]}) == 60
    assert len(campaign["plans"]) == 6
    assert all(len(p["candidates"]) == 8 for p in campaign["plans"])


def test_fake_end_to_end_persists_models_predictions_and_safe_optimization(tmp_path):
    campaign = prepare_campaign(EXPERIMENT)
    run_dir = tmp_path / "fake-run"
    transport = FakeTransport(campaign, run_dir)

    def count_probe(payload):
        assert payload == {
            "model": campaign["config"]["deployment"], "input": campaign["calls"][0]["prompt"],
        }
        return SimpleNamespace(input_tokens=100)

    result = execute_campaign(
        campaign, run_dir, token_provider=lambda: "FAKE-SECRET",
        transport=transport, count_probe=count_probe,
    )
    assert transport.calls == 60
    assert result["contract_passed_calls"] == 60
    assert result["total_rated_cost_usd"] > 0
    assert result["budget"]["active_reserved_usd"] == 0
    assert result["budget"]["settled_safety_usd"] < 50
    assert not result["batching"]["quality_noninferiority_established"]
    assert all(p["production_recommendation"] is None for p in result["batching"]["projects"])
    for path in run_dir.glob("*.json"):
        assert "FAKE-SECRET" not in path.read_text()
    with pytest.raises(FileExistsError):
        execute_campaign(campaign, run_dir, token_provider=lambda: pytest.fail("must not retry"))


@pytest.mark.parametrize("failure", ["http", "usage", "incomplete", "identity"])
def test_failed_attempt_is_saved_and_never_automatically_retried(tmp_path, failure):
    campaign = prepare_campaign(EXPERIMENT)
    run_dir = tmp_path / failure
    transport = FakeTransport(campaign, run_dir, failure)
    with pytest.raises((FoundryDispatchError, RuntimeError)):
        execute_campaign(campaign, run_dir, token_provider=lambda: "fake",
                         transport=transport, count_probe=lambda _: SimpleNamespace(input_tokens=1))
    assert transport.calls == 1
    rows = json.loads((run_dir / "records.json").read_text())
    budget = json.loads((run_dir / "budget.json").read_text())
    assert len(rows) == 1
    assert rows[0]["status"] != "completed"
    if failure in ("incomplete", "identity"):
        assert rows[0]["rated_cost_usd"] > 0
        assert budget["settled_safety_usd"] > 0
    else:
        assert rows[0]["rated_cost_usd"] is None
        assert budget["active_reserved_usd"] > 0
    assert json.loads((run_dir / "state.json").read_text())["status"] == "halted"


def test_bad_auth_spends_nothing(tmp_path):
    campaign = prepare_campaign(EXPERIMENT)
    run_dir = tmp_path / "no-auth"
    with pytest.raises(ValueError, match="authentication"):
        execute_campaign(campaign, run_dir, token_provider=lambda: "",
                         transport=lambda *args: pytest.fail("must not dispatch"))
    assert json.loads((run_dir / "budget.json").read_text())["active_reserved_usd"] == 0


@pytest.mark.parametrize("channel", [
    "input_tokens", "output_tokens", "total_tokens", "input_tokens_details",
    "input_tokens_details.cached_tokens", "output_tokens_details",
    "output_tokens_details.reasoning_tokens",
])
def test_missing_raw_channels_retain_reservation_and_halt(tmp_path, channel):
    campaign = prepare_campaign(EXPERIMENT)
    run_dir = tmp_path / "missing-channel"
    transport = FakeTransport(campaign, run_dir, f"missing:{channel}")
    with pytest.raises(FoundryDispatchError, match="missing measured usage channel"):
        execute_campaign(campaign, run_dir, token_provider=lambda: "fake",
                         transport=transport, count_probe=lambda _: SimpleNamespace(input_tokens=1))
    assert transport.calls == 1
    row = json.loads((run_dir / "records.json").read_text())[0]
    assert row["status"] == "failed"
    assert row["usage"] is None
    assert row["rated_cost_usd"] is None
    budget = json.loads((run_dir / "budget.json").read_text())
    assert budget["active_reserved_usd"] == row["reservation_usd"]
    assert budget["settled_safety_usd"] == 0
    assert (run_dir / f"{row['call_id']}.response.json").exists()


def test_settlement_failure_persists_known_usage_and_charge(tmp_path):
    campaign = prepare_campaign(EXPERIMENT)
    run_dir = tmp_path / "over-budget"
    transport = FakeTransport(campaign, run_dir, "over-budget")
    with pytest.raises(RuntimeError, match="exceeded the hard cap"):
        execute_campaign(campaign, run_dir, token_provider=lambda: "fake",
                         transport=transport, count_probe=lambda _: SimpleNamespace(input_tokens=1))
    assert transport.calls == 1
    row = json.loads((run_dir / "records.json").read_text())[0]
    assert row["status"] == "settlement_failed"
    assert row["usage"]["input_tokens"] == 3_000_000
    assert row["rated_cost_usd"] > 0
    budget = json.loads((run_dir / "budget.json").read_text())
    assert budget["settled_safety_usd"] > budget["cap_usd"]
    assert budget["active_reserved_usd"] == 0
    assert json.loads((run_dir / "state.json").read_text())["status"] == "halted"


def test_counter_auth_failure_is_not_treated_as_unsupported(tmp_path):
    def failed_probe(_):
        raise FoundryCountError("authentication failed")
    run_dir = tmp_path / "count-auth"
    with pytest.raises(FoundryCountError):
        execute_campaign(prepare_campaign(EXPERIMENT), run_dir,
                         token_provider=lambda: "fake", count_probe=failed_probe,
                         transport=lambda *args: pytest.fail("must not dispatch"))


def test_budget_denial_happens_before_generation(tmp_path):
    campaign = prepare_campaign(EXPERIMENT)
    campaign["config"]["cap_usd"] = campaign["config"]["stop_usd"] = 0.01
    run_dir = tmp_path / "too-small"
    with pytest.raises(RuntimeError, match="budget blocks"):
        execute_campaign(campaign, run_dir, token_provider=lambda: "fake",
                         count_probe=lambda _: SimpleNamespace(input_tokens=1),
                         transport=lambda *args: pytest.fail("must not dispatch"))
    assert not (run_dir / "records.json").exists()


def test_prior_spend_cannot_be_ignored(tmp_path):
    config = json.loads((EXPERIMENT / "pilot.json").read_text())
    config["cap_usd"] = 50
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="cumulative"):
        load_config(path)


def test_deployment_version_mismatch_stops_before_generation(tmp_path, fake_deployment_probe):
    fake_deployment_probe["model"]["version"] = "different"
    run_dir = tmp_path / "wrong-version"
    with pytest.raises(RuntimeError, match="deployment differs"):
        execute_campaign(prepare_campaign(EXPERIMENT), run_dir, token_provider=lambda: "fake",
                         transport=lambda *args: pytest.fail("must not generate"))
    assert not (run_dir / "records.json").exists()
    assert (run_dir / "deployment-before.json").exists()


def test_deployment_change_stops_before_holdout(tmp_path, monkeypatch):
    observations = iter([
        {"model": {"name": "gpt-5.4", "version": "2026-03-05"},
         "sku": "GlobalStandard", "state": "Succeeded"},
        {"model": {"name": "gpt-5.4", "version": "different"},
         "sku": "GlobalStandard", "state": "Succeeded"},
    ])
    monkeypatch.setattr(customer_pilot, "_deployment_probe", lambda: next(observations))
    campaign = prepare_campaign(EXPERIMENT)
    run_dir = tmp_path / "drift"
    transport = FakeTransport(campaign, run_dir)
    with pytest.raises(RuntimeError, match="deployment differs"):
        execute_campaign(campaign, run_dir, token_provider=lambda: "fake", transport=transport,
                         count_probe=lambda _: SimpleNamespace(input_tokens=1))
    assert transport.calls == 40
    assert not (run_dir / "predictions.json").exists()


@pytest.mark.parametrize("bad_cap", [51, -1, True, float("nan")])
def test_config_rejects_unapproved_or_invalid_cap(tmp_path, bad_cap):
    config = json.loads((EXPERIMENT / "pilot.json").read_text())
    config["cap_usd"] = bad_cap
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_incomplete_measurements_cannot_win_batching():
    campaign = prepare_campaign(EXPERIMENT)
    result = summarize_batching(campaign, [])
    assert all(p["cheapest_contract_passing_arm"] is None for p in result["projects"])
    assert not result["unmeasured_layouts_eligible"]


def test_changed_plan_rejected_by_optimizer():
    campaign = prepare_campaign(EXPERIMENT)
    row = copy.deepcopy(campaign["calls"][0])
    row["plan_sha256"] = "different-scope"
    with pytest.raises(ValueError, match="frozen layout"):
        summarize_batching(campaign, [row])
