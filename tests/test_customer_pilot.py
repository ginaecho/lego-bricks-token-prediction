"""Campaign integration tests with synthetic transport, never paid inference."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from token_yield.customer_decomposition import content_hash, load_catalog, load_template
from token_yield.customer_pilot import (
    execute_campaign, load_config, prepare_campaign, summarize_batching,
)
from token_yield.foundry_count import FoundryCountError
from token_yield.foundry_dispatch import FoundryDispatchError
from token_yield import customer_pilot


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "customer_requests"
AUGMENTED = EXPERIMENT.parent / "customer_requests_augmented_20260916"
FRESH = EXPERIMENT.parent / "customer_requests_fresh_holdout_20260917"
DEPLOYMENT_PROBE = customer_pilot._deployment_probe


@pytest.fixture
def augmented_experiment(tmp_path):
    for name in ("catalog.json", "template.json", "pilot.json"):
        (tmp_path / name).write_bytes((AUGMENTED / name).read_bytes())
    return tmp_path


@pytest.fixture
def fresh_experiment(tmp_path):
    for name in ("catalog.json", "template.json", "pilot.json"):
        (tmp_path / name).write_bytes((FRESH / name).read_bytes())
    return tmp_path


@pytest.fixture(autouse=True)
def fake_deployment_probe(monkeypatch):
    metadata = {
        "model": {"name": "gpt-5.4", "version": "2026-03-05"},
        "sku": "GlobalStandard", "state": "Succeeded",
        "versionUpgradeOption": "OnceNewDefaultVersionAvailable",
    }
    monkeypatch.setattr(customer_pilot, "_deployment_probe", lambda config: copy.deepcopy(metadata))
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
            assert {row["call_id"] for row in frozen["predictions"]} == {
                c["call_id"] for c in self.campaign["calls"] if c["split"] == "holdout"
            }
            models = json.loads((self.run_dir / "models.json").read_text())
            assert set(models["training_project_ids"]) == {
                c["project_id"] for c in self.campaign["calls"] if c["split"] == "train"
            }
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


def test_augmented_prepare_preserves_sources_splits_and_protocol(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preparation must remain offline")
    monkeypatch.setattr(customer_pilot, "_deployment_probe", forbidden)
    monkeypatch.setattr(customer_pilot, "urlopen", forbidden)
    campaign = prepare_campaign(AUGMENTED)
    assert campaign == prepare_campaign(AUGMENTED)
    assert len(campaign["plans"]) == 27
    assert len(campaign["calls"]) == 270
    assert len({call["call_id"] for call in campaign["calls"]}) == 270
    assert [c["split"] for c in campaign["calls"]] == ["train"] * 190 + ["holdout"] * 80
    assert all(len(plan["candidates"]) == 8 for plan in campaign["plans"])
    assert campaign["catalog"]["projects"][:6] == load_catalog(EXPERIMENT / "catalog.json")["projects"]
    assert campaign["template"] == load_template(EXPERIMENT / "template.json")
    assert campaign["catalog_sha256"] == campaign["config"]["catalog_sha256"]
    assert campaign["template_sha256"] == campaign["config"]["template_sha256"]
    assert campaign["config"]["execution_approved"] is False
    assert "19 train and 8 outcome-holdout" in " ".join(campaign["limitations"])
    for project in campaign["catalog"]["projects"]:
        calls = [call for call in campaign["calls"] if call["project_id"] == project["id"]]
        assert len(calls) == 10
        assert {call["split"] for call in calls} == {project["split"]}
        assert all("usage" not in call for call in calls)


def test_augmented_budget_carries_both_prior_runs():
    config = load_config(AUGMENTED / "pilot.json")
    previous_run = EXPERIMENT.parents[1] / "runs" / "20260914_customer_scoping_v2"
    analysis = json.loads((previous_run / "analysis.json").read_text())
    budget = json.loads((previous_run / "budget.json").read_text())
    prior = config["prior_attempt"]
    assert prior["rated_cost_usd"] == analysis["cumulative_rated_cost_usd"]
    assert prior["settled_safety_usd"] == pytest.approx(
        budget["settled_safety_usd"] + analysis["prior_attempt"]["settled_safety_usd"])
    assert prior["active_reserved_usd"] == budget["active_reserved_usd"] == 0
    assert config["cap_usd"] + prior["settled_safety_usd"] == pytest.approx(50)
    assert config["stop_usd"] + prior["settled_safety_usd"] == pytest.approx(48)


@pytest.mark.parametrize("counts", [
    None, [], {}, {"train": 19}, {"train": 19, "holdout": 8, "other": 1},
    {"train": 2, "holdout": 8}, {"train": 19, "holdout": 0},
    {"train": 19.0, "holdout": 8}, {"train": True, "holdout": 8},
    {"train": 19, "holdout": True},
])
def test_v2_rejects_invalid_project_counts(augmented_experiment, counts):
    path = augmented_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["project_counts"] = counts
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="project_counts"):
        load_config(path)


@pytest.mark.parametrize("approval", [None, 0, 1, "false", "true"])
def test_v2_requires_boolean_execution_approval(augmented_experiment, approval):
    path = augmented_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["execution_approved"] = approval
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="execution_approved"):
        load_config(path)


@pytest.mark.parametrize("field", ["catalog_sha256", "template_sha256"])
@pytest.mark.parametrize("digest", [None, "", "0" * 63, "z" * 64])
def test_v2_requires_valid_source_pins(augmented_experiment, field, digest):
    path = augmented_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config[field] = digest
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        load_config(path)


@pytest.mark.parametrize("version", ["customer-pilot-v1", "customer-pilot-v2"])
def test_population_must_match_configured_version(augmented_experiment, version):
    path = augmented_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["schema_version"] = version
    config["project_counts"] = {"train": 18, "holdout": 9}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="project counts differ"):
        prepare_campaign(augmented_experiment)


def test_v2_population_is_configurable_not_hardcoded_to_27(augmented_experiment):
    catalog = load_catalog(augmented_experiment / "catalog.json")
    catalog["projects"] = (
        [p for p in catalog["projects"] if p["split"] == "train"][:3]
        + [p for p in catalog["projects"] if p["split"] == "holdout"][:1]
    )
    (augmented_experiment / "catalog.json").write_text(json.dumps(catalog))
    config = json.loads((augmented_experiment / "pilot.json").read_text())
    config["project_counts"] = {"train": 3, "holdout": 1}
    config["catalog_sha256"] = content_hash(catalog)
    (augmented_experiment / "pilot.json").write_text(json.dumps(config))
    campaign = prepare_campaign(augmented_experiment)
    assert len(campaign["plans"]) == 4
    assert [c["split"] for c in campaign["calls"]] == ["train"] * 30 + ["holdout"] * 10


@pytest.mark.parametrize("change", ["requirement", "split_swap", "output_limit", "instruction"])
def test_v2_rejects_unreviewed_input_changes(augmented_experiment, change):
    name = "catalog" if change in ("requirement", "split_swap") else "template"
    path = augmented_experiment / f"{name}.json"
    value = json.loads(path.read_text())
    if change == "requirement":
        value["projects"][0]["requirements"][0]["text"] += " Additional scope."
    elif change == "split_swap":
        value["projects"][0]["split"], value["projects"][4]["split"] = "holdout", "train"
    elif change == "output_limit":
        value["max_output_tokens_per_operation"] = 3200
    else:
        value["operations"][0]["instruction"] += " Additional instruction."
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=f"frozen {name} differs"):
        prepare_campaign(augmented_experiment)


@pytest.mark.parametrize("approval", [False, None, 1, "true"])
def test_unapproved_execution_stops_before_files_or_network(tmp_path, approval):
    campaign = prepare_campaign(AUGMENTED)
    campaign["config"]["execution_approved"] = approval
    run_dir = tmp_path / "unapproved"

    def forbidden(*args, **kwargs):
        pytest.fail("unapproved setup must not authenticate, count, or generate")
    with pytest.raises(ValueError, match="explicit execution approval"):
        execute_campaign(campaign, run_dir, token_provider=forbidden,
                         transport=forbidden, count_probe=forbidden)
    assert not run_dir.exists()


def test_fake_augmented_execution_after_explicit_approval(augmented_experiment):
    path = augmented_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["execution_approved"] = True
    path.write_text(json.dumps(config))
    campaign = prepare_campaign(augmented_experiment)
    run_dir = augmented_experiment / "fake-run"
    transport = FakeTransport(campaign, run_dir)
    result = execute_campaign(
        campaign, run_dir, token_provider=lambda: "FAKE-SECRET",
        transport=transport, count_probe=lambda _: SimpleNamespace(input_tokens=100),
    )
    assert transport.calls == result["calls"] == 270
    assert result["contract_passed_calls"] == 270
    assert result["budget"]["active_reserved_usd"] == 0
    assert result["holdout"]["n_calls"] == 80
    assert result["holdout"]["n_projects"] == 8
    assert result["cumulative_rated_cost_usd"] == pytest.approx(
        result["total_rated_cost_usd"] + config["prior_attempt"]["rated_cost_usd"])
    rows = json.loads((run_dir / "records.json").read_text())
    assert [row["split"] for row in rows] == ["train"] * 190 + ["holdout"] * 80


def test_fresh_holdout_uses_only_unmeasured_projects_and_preserves_source_content():
    campaign = prepare_campaign(FRESH)
    original_ids = {p["id"] for p in load_catalog(EXPERIMENT / "catalog.json")["projects"]}
    held = {p["id"] for p in campaign["catalog"]["projects"] if p["split"] == "holdout"}
    assert len(held) == 8 and not held & original_ids
    assert len(campaign["calls"]) == 270
    assert campaign["calls_sha256"] == content_hash(campaign["calls"])
    assert [c["split"] for c in campaign["calls"]] == ["train"] * 190 + ["holdout"] * 80
    previous = load_catalog(AUGMENTED / "catalog.json")
    for old, new in zip(previous["projects"], campaign["catalog"]["projects"]):
        assert {k: v for k, v in old.items() if k != "split"} == {
            k: v for k, v in new.items() if k != "split"}
    rates = customer_pilot.SafetyRateCard()
    total_reservations = sum(
        rates.price(campaign["config"]["max_input_tokens_per_call"],
                    2 * call["quote"]["planned_output_tokens"])
        for call in campaign["calls"])
    assert total_reservations == pytest.approx(453.4272)
    assert total_reservations <= campaign["config"]["stop_usd"]
    assert campaign["config"]["cap_usd"] == 453.43


@pytest.mark.parametrize("endpoint", [
    "https://msfoundry-hackathon.services.ai.azure.com/api/projects/foundry-hackathon",
    "https://other.openai.azure.com/openai/v1",
    "http://msfoundry-hackathon.openai.azure.com/openai/v1",
    "https://msfoundry-hackathon.openai.azure.com/openai/v1?extra=true",
    "https://user:password@msfoundry-hackathon.openai.azure.com/openai/v1",
])
def test_v3_rejects_project_or_unrelated_endpoints(fresh_experiment, endpoint):
    path = fresh_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["endpoint"] = endpoint
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="inference URL"):
        load_config(path)


@pytest.mark.parametrize("field,value", [
    ("tenant_id", "not-a-uuid"), ("subscription_id", ""),
    ("resource_name", "bad/name"), ("resource_group", None),
])
def test_v3_requires_valid_resource_identity(fresh_experiment, field, value):
    path = fresh_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["azure_resource"][field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="azure_resource"):
        load_config(path)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, -1])
def test_v3_budget_approval_must_be_finite(fresh_experiment, value):
    path = fresh_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["budget_approval"]["cap_usd"] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="finite positive cumulative"):
        load_config(path)


def test_v3_budget_keeps_prior_safety_accounting(fresh_experiment):
    path = fresh_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["budget_approval"]["cap_usd"] = config["cap_usd"]
    config["budget_approval"]["stop_usd"] = config["stop_usd"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="cumulative"):
        load_config(path)


def test_v3_cannot_expand_the_approved_call_count(fresh_experiment):
    path = fresh_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["budget_approval"]["max_calls"] = 269
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="max_calls"):
        prepare_campaign(fresh_experiment)


@pytest.mark.parametrize("change", ["config", "prompt", "quote", "extra_call"])
def test_v3_rejects_mutated_campaign_before_creating_run(tmp_path, change):
    campaign = prepare_campaign(FRESH)
    if change == "config":
        campaign["config"]["endpoint"] = "https://other.openai.azure.com/openai/v1"
    elif change == "prompt":
        campaign["calls"][0]["prompt"] += " Unreviewed instruction."
    elif change == "quote":
        campaign["calls"][0]["quote"]["planned_output_tokens"] += 1
    else:
        campaign["calls"].append(copy.deepcopy(campaign["calls"][0]))
    run_dir = tmp_path / "mutated"
    with pytest.raises(ValueError, match="changed after preparation"):
        execute_campaign(campaign, run_dir,
                         token_provider=lambda: pytest.fail("must not authenticate"))
    assert not run_dir.exists()


def test_v3_rejects_execution_without_approval(fresh_experiment):
    path = fresh_experiment / "pilot.json"
    config = json.loads(path.read_text())
    config["execution_approved"] = False
    path.write_text(json.dumps(config))
    run_dir = fresh_experiment / "unapproved"
    with pytest.raises(ValueError, match="explicit execution approval"):
        execute_campaign(prepare_campaign(fresh_experiment), run_dir,
                         token_provider=lambda: pytest.fail("must not authenticate"))
    assert not run_dir.exists()


def test_deployment_probe_uses_configured_resource_and_deployment(monkeypatch):
    invocations = []
    config = load_config(FRESH / "pilot.json")
    config["deployment"] = "scoping-deployment"
    expected = {"model": {"name": "gpt-5.4", "version": "2026-03-05"},
                "sku": "GlobalStandard", "state": "Succeeded"}
    monkeypatch.setattr(customer_pilot.shutil, "which", lambda name: "az")

    def runner(command, **kwargs):
        invocations.append(command)
        return SimpleNamespace(stdout=json.dumps(expected))
    monkeypatch.setattr(customer_pilot.subprocess, "run", runner)
    assert DEPLOYMENT_PROBE(config) == expected
    command = invocations[0]
    assert command[command.index("--name") + 1] == "msfoundry-hackathon"
    assert command[command.index("--resource-group") + 1] == "brb-dev-ne"
    assert command[command.index("--subscription") + 1] == config["azure_resource"]["subscription_id"]
    assert command[command.index("--deployment-name") + 1] == "scoping-deployment"


def test_v3_default_auth_is_scoped_and_refreshed_for_each_request(tmp_path, monkeypatch):
    campaign = prepare_campaign(FRESH)
    auth_calls = []

    def scoped_token(**kwargs):
        auth_calls.append(kwargs)
        return "FAKE-SCOPED-TOKEN"
    monkeypatch.setattr(customer_pilot, "acquire_entra_token", scoped_token)
    run_dir = tmp_path / "fake-fresh"
    transport = FakeTransport(campaign, run_dir)
    result = execute_campaign(
        campaign, run_dir, transport=transport,
        count_probe=lambda _: SimpleNamespace(input_tokens=100))
    assert transport.calls == 270
    assert result["contract_passed_calls"] == 270
    assert len(auth_calls) == 270
    assert all(call == {
        "subscription_id": campaign["config"]["azure_resource"]["subscription_id"],
        "tenant_id": campaign["config"]["azure_resource"]["tenant_id"],
    } for call in auth_calls)
    assert result["holdout"]["n_projects"] == 8
    rows = json.loads((run_dir / "records.json").read_text())
    original_ids = {p["id"] for p in load_catalog(EXPERIMENT / "catalog.json")["projects"]}
    assert not {r["project_id"] for r in rows if r["split"] == "holdout"} & original_ids
    assert "FAKE-SCOPED-TOKEN" not in (run_dir / "records.json").read_text()


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
    monkeypatch.setattr(customer_pilot, "_deployment_probe", lambda config: next(observations))
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
