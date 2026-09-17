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


@pytest.fixture
def prospective_inputs(tmp_path):
    campaign = prepare_campaign(EXPERIMENT)
    source, model_run = tmp_path / "measured", tmp_path / "model"
    execute_campaign(
        campaign, source, token_provider=lambda: "FAKE-SECRET",
        transport=FakeTransport(campaign, source),
        count_probe=lambda payload: SimpleNamespace(input_tokens=100),
    )
    customer_pilot.train_project_forecast(source, model_run)
    catalog = copy.deepcopy(campaign["catalog"])
    catalog["projects"] = catalog["projects"][:4]
    for index, project in enumerate(catalog["projects"]):
        project.update(id=f"new-request-{index}", split="holdout",
                       source_url=f"https://example.org/request-{index}")
        for requirement in project["requirements"]:
            requirement["brick"] = "plan"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return model_run, path, source


class ProspectiveTransport:
    def __init__(self, campaign, run_dir, fail_after=None):
        self.campaign, self.run_dir = campaign, run_dir
        self.calls, self.fail_after = 0, fail_after

    def __call__(self, url, headers, body, timeout):
        call = self.campaign["calls"][self.calls]
        frozen = json.loads((self.run_dir / "predictions.json").read_text())
        assert frozen["predictions"] == self.campaign["prospective"]["predictions"]
        assert len(frozen["predictions"]) == 16
        assert all(len(forms) == 5 for prediction in frozen["predictions"]
                   for forms in prediction["predictions_by_form"].values())
        assert call["call_id"] in json.loads(
            (self.run_dir / "budget.json").read_text())["active_reservations"]
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return 429, b'{"error":{"code":"RateLimitReached"}}'
        payload = json.loads(body)
        assert set(payload) == {"input", "model", "reasoning", "text", "max_output_tokens"}
        assert payload["max_output_tokens"] == 1600 * len(call["operations"])
        project = next(p for p in self.campaign["catalog"]["projects"]
                       if p["id"] == call["project_id"])
        text = fake_output(project, call["operations"])
        inp, out = len(payload["input"]) // 4, len(text) // 4
        return 200, json.dumps({
            "id": f"new-fake-{self.calls}", "status": "completed", "model": "gpt-5.4",
            "output_text": text, "output": [], "usage": {
                "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }).encode()


def test_offline_token_training_preserves_sources_and_reloads(prospective_inputs, tmp_path, monkeypatch):
    from token_yield.customer_models import forecast_tokens
    from token_yield.robust import ConstantModel, RidgeLinearModel

    _, _, source = prospective_inputs
    original = {path: path.read_bytes() for path in source.iterdir() if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("offline training must not contact providers; reload must not fit")

    monkeypatch.setattr(customer_pilot, "_default_transport", forbidden)
    monkeypatch.setattr(customer_pilot, "acquire_entra_token", forbidden)
    destination = tmp_path / "token-models"
    result = customer_pilot.train_token_forecast(source, destination)
    assert result["spent_usd"] == 0 and result["status"] == "trained_exploratory"
    assert result["training_calls"] == 40 and result["holdout_calls"] == 20
    assert result["training_projects"] == 4 and result["holdout_projects"] == 2
    assert all(path.read_bytes() == raw for path, raw in original.items())
    artifact = json.loads((destination / "models.json").read_text(encoding="utf-8"))
    predictions = json.loads((destination / "predictions.json").read_text(encoding="utf-8"))
    assert "pricing" not in artifact["runtime"]
    assert "records.json" in artifact["provenance"]["source_file_sha256"]
    assert len(predictions) == 60
    assert json.loads((destination / "analysis.json").read_text(encoding="utf-8")) == result
    measured = {row["call_id"]: row for row in json.loads(original[source / "records.json"])}
    monkeypatch.setattr(ConstantModel, "fit", forbidden)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden)
    for prediction in predictions:
        reloaded = forecast_tokens(artifact, measured[prediction["call_id"]]["quote"], artifact["runtime"])
        assert reloaded["point_estimate"] == prediction["point_estimate"]
    with pytest.raises(FileExistsError):
        customer_pilot.train_token_forecast(source, destination)
    with pytest.raises(ValueError, match="outside"):
        customer_pilot.train_token_forecast(source, source / "nested")


def test_offline_token_training_rejects_tampered_measurements(prospective_inputs, tmp_path):
    _, _, source = prospective_inputs
    path = next(source.glob("*.response.json"))
    response = json.loads(path.read_text(encoding="utf-8"))
    response["usage"]["input_tokens"] += 1
    path.write_text(json.dumps(response), encoding="utf-8")
    destination = tmp_path / "rejected-token-models"
    with pytest.raises(ValueError, match="raw response"):
        customer_pilot.train_token_forecast(source, destination)
    assert not destination.exists()


def test_prospective_freezes_all_channels_without_fit_and_preserves_sources(
        prospective_inputs, tmp_path, monkeypatch):
    from token_yield import customer_models
    from token_yield.robust import ConstantModel, RidgeLinearModel

    model_run, catalog_path, source = prospective_inputs
    original = {path: path.read_bytes() for root in (model_run, source)
                for path in root.iterdir() if path.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("prospective evaluation must not fit or select models")

    for name in ("fit_models", "fit_project_models"):
        monkeypatch.setattr(customer_models, name, forbidden)
    monkeypatch.setattr(ConstantModel, "fit", forbidden)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden)
    campaign = customer_pilot.prepare_prospective_campaign(model_run, catalog_path)
    assert len(campaign["calls"]) == 40
    assert campaign["config"]["cap_usd"] == 10 and campaign["config"]["stop_usd"] == 9.5
    assert all(call["split"] == "holdout" for call in campaign["calls"])
    assert any("not independent organizations" in item for item in campaign["limitations"])
    assert all(p["support"]["status"] == "unsupported"
               for p in campaign["prospective"]["predictions"])
    destination = tmp_path / "test"
    transport = ProspectiveTransport(campaign, destination)
    result = execute_campaign(
        campaign, destination, token_provider=lambda: "FAKE-SECRET", transport=transport,
        count_probe=lambda payload: SimpleNamespace(input_tokens=100),
    )
    assert result["status"] == "completed"
    assert result["n_projects"] == 4 and result["n_observations"] == 16
    assert result["source_model_run"] == model_run.name
    assert transport.calls == 40
    assert result["selected_forms"] == campaign["prospective"]["selected_forms"]
    assert all(path.read_bytes() == raw for path, raw in original.items())
    for target, forms in result["scores"].items():
        for form, score in forms.items():
            errors = [abs(p["predictions_by_form"][target][form] - p["actual"][target])
                      for p in result["observations"]]
            assert score["mae"] == pytest.approx(sum(errors) / len(errors))
    assert result["totals"]["input_tokens"] == sum(
        p["actual"]["input_tokens"] for p in result["observations"])
    assert json.loads((destination / "prospective_evaluation.json").read_text()) == result
    assert len(json.loads((destination / "project_records.json").read_text())) == 16


def test_prospective_halt_retains_spend_and_never_claims_complete(
        prospective_inputs, tmp_path):
    model_run, catalog_path, _ = prospective_inputs
    campaign = customer_pilot.prepare_prospective_campaign(model_run, catalog_path)
    destination = tmp_path / "halted"
    transport = ProspectiveTransport(campaign, destination, fail_after=1)
    with pytest.raises(FoundryDispatchError):
        execute_campaign(
            campaign, destination, token_provider=lambda: "FAKE-SECRET", transport=transport,
            count_probe=lambda payload: SimpleNamespace(input_tokens=100),
        )
    result = json.loads((destination / "prospective_evaluation.json").read_text())
    assert transport.calls == 2
    assert result["status"] == "halted" and result["scores"] == {}
    assert result["n_observations"] == 0
    assert result["totals"]["rated_cost_usd"] > 0
    assert json.loads((destination / "budget.json").read_text())["active_reservations"]
    with pytest.raises(FileExistsError):
        execute_campaign(campaign, destination)


@pytest.mark.parametrize("mutation", ["train", "small", "brick", "old-url", "old-id", "same-url"])
def test_prospective_rejects_unsupported_catalogs(prospective_inputs, mutation):
    model_run, catalog_path, source = prospective_inputs
    catalog = json.loads(catalog_path.read_text())
    old = json.loads((source / "protocol.json").read_text())["catalog"]["projects"][0]
    if mutation == "train":
        catalog["projects"][0]["split"] = "train"
    elif mutation == "small":
        catalog["projects"].pop()
    elif mutation == "brick":
        catalog["projects"][0]["requirements"][0]["brick"] = "unsupported-brick"
    elif mutation == "old-url":
        catalog["projects"][0]["source_url"] = old["source_url"] + "#duplicate"
    elif mutation == "old-id":
        catalog["projects"][0]["id"] = old["id"]
    else:
        catalog["projects"][0]["source_url"] = catalog["projects"][1]["source_url"]
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        customer_pilot.prepare_prospective_campaign(model_run, catalog_path)


@pytest.mark.parametrize("mutation", ["prediction", "catalog", "model", "source-output"])
def test_prospective_rejects_post_freeze_changes_before_auth(
        prospective_inputs, tmp_path, mutation):
    model_run, catalog_path, source = prospective_inputs
    campaign = customer_pilot.prepare_prospective_campaign(model_run, catalog_path)
    destination = tmp_path / "rejected"
    if mutation == "prediction":
        campaign["prospective"]["predictions"][0]["predicted"]["input_tokens"] += 1
    elif mutation in ("catalog", "model"):
        path = catalog_path if mutation == "catalog" else model_run / "models.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        destination = source / "new"
    with pytest.raises(ValueError):
        execute_campaign(campaign, destination, token_provider=lambda: pytest.fail("no auth"))
    assert not destination.exists()


def test_prospective_cli_preview_never_dispatches(prospective_inputs, tmp_path, monkeypatch):
    from examples import customer_request_pilot as cli

    model_run, catalog_path, _ = prospective_inputs
    catalog = json.loads(catalog_path.read_text())
    catalog["selection_notes"] = ["Only two buyer organizations, not four independent buyers."]
    catalog["scope"] = "Scoping only, not execution of deliverables."
    catalog["rejected_projects"] = []
    catalog["projects"][0]["compatibility"] = "scoping_only_full_delivery_unsupported"
    catalog["projects"][0]["source_sections"] = ["Research metadata must not enter the prompt."]
    catalog["projects"][0]["requirements"][0]["brick"] = "draft"
    catalog_path.write_text(json.dumps(catalog))
    destination = tmp_path / "preview"
    monkeypatch.setattr(cli, "execute_campaign", lambda *a: pytest.fail("preview must be offline"))
    monkeypatch.setattr(customer_pilot, "_deployment_probe",
                        lambda: pytest.fail("preview must not probe Azure"))
    monkeypatch.setattr("sys.argv", [
        "pilot", "--test-model", str(model_run), "--catalog", str(catalog_path),
        "--run-dir", str(destination),
    ])
    assert cli.main() == 0
    preview = json.loads((destination / "preview.json").read_text())
    assert len(preview["prospective"]["predictions"]) == 16
    assert preview["prospective"]["source_catalog_metadata"]["selection_notes"] == catalog["selection_notes"]
    assert all("Research metadata must not enter the prompt." not in call["prompt"]
               for call in preview["calls"])
    assert all(set(call["operations"]) <= {"extract", "classify", "plan", "report"}
               for call in preview["calls"])
    assert not (destination / "records.json").exists()
    catalog["unknown_setting"] = "must not be silently ignored"
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="unsupported prospective catalog"):
        customer_pilot.prepare_prospective_campaign(model_run, catalog_path)


def test_local_snapshot_lock_retries_are_bounded_without_reissuing_requests(tmp_path, monkeypatch):
    original = Path.replace
    attempts = []

    def briefly_locked(path, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("synthetic Windows scanner lock")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", briefly_locked)
    monkeypatch.setattr(customer_pilot.time, "sleep", lambda seconds: None)
    target = tmp_path / "snapshot.json"
    customer_pilot._write_json(target, {"status": "reserved"})
    assert len(attempts) == 3
    assert json.loads(target.read_text()) == {"status": "reserved"}
    attempts.clear()

    def always_locked(path, target):
        attempts.append(target)
        raise PermissionError("persistent lock")

    monkeypatch.setattr(Path, "replace", always_locked)
    with pytest.raises(PermissionError):
        customer_pilot._write_json(target, {"status": "completed"})
    assert len(attempts) == 5
    assert json.loads(target.read_text()) == {"status": "reserved"}


def test_prospective_rejects_unknown_runtime_and_changed_critical_code(prospective_inputs):
    model_run, catalog_path, _ = prospective_inputs
    model_path = model_run / "models.json"
    model = json.loads(model_path.read_text())
    model["runtime"]["temperature"] = 0.5
    model_path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="unknown frozen model runtime"):
        customer_pilot.prepare_prospective_campaign(model_run, catalog_path)
    model["runtime"].pop("temperature")
    model["runtime"]["execution_code_sha256"]["budget.py"] = "0" * 64
    with pytest.raises(ValueError, match="critical execution component"):
        customer_pilot._runtime_compatibility(model, {})


def test_offline_project_training_preserves_source_and_reloads(tmp_path, monkeypatch):
    from token_yield.customer_models import forecast_project
    from token_yield.customer_pilot import aggregate_project_rows, train_project_forecast
    from token_yield.robust import ConstantModel, RidgeLinearModel

    campaign = prepare_campaign(EXPERIMENT)
    source = tmp_path / "measured"
    transport = FakeTransport(campaign, source)
    execute_campaign(
        campaign, source, token_provider=lambda: "FAKE-SECRET", transport=transport,
        count_probe=lambda payload: SimpleNamespace(input_tokens=100),
    )
    original = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    frozen = json.loads(original["protocol.json"])
    measured = json.loads(original["records.json"])
    grouped = aggregate_project_rows(frozen, measured)
    assert len(grouped) == 24
    assert {row["project_id"] for row in grouped} == {
        project["id"] for project in campaign["catalog"]["projects"]
    }
    for name in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens",
                 "reasoning_tokens"):
        assert sum(row["usage"][name] for row in grouped) == sum(
            row["usage"][name] for row in measured)
    assert sum(row["rated_cost_usd"] for row in grouped) == pytest.approx(
        sum(row["rated_cost_usd"] for row in measured))
    for row in grouped:
        assert row["quote"]["planned_calls"] == (4 if row["arm"] == "split" else 1)
        assert row["quote"]["max_operations_per_call"] == (1 if row["arm"] == "split" else 4)
        assert row["quote"]["counts"]["report"] == 1
        assert row["quote"]["planned_output_tokens"] == 6400

    def forbidden(*args, **kwargs):
        raise AssertionError("offline training must not contact the model or cloud")

    monkeypatch.setattr(customer_pilot, "_default_transport", forbidden)
    monkeypatch.setattr(customer_pilot, "_deployment_probe", forbidden)
    monkeypatch.setattr(customer_pilot, "acquire_entra_token", forbidden)
    destination = tmp_path / "trained"
    summary = train_project_forecast(source, destination)
    assert (summary["training_projects"], summary["holdout_projects"]) == (4, 2)
    assert (summary["training_observations"], summary["holdout_observations"]) == (16, 8)
    assert summary["spent_usd"] == 0
    assert "retrospective" in summary["holdout"]["method"]
    assert original == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    saved = json.loads((destination / "models.json").read_text())
    forecasts = json.loads((destination / "predictions.json").read_text())
    assert saved["provenance"]["source_file_sha256"]["records.json"]
    assert len(saved["provenance"]["source_file_sha256"]) == 122
    monkeypatch.setattr(ConstantModel, "fit", forbidden)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden)
    for row, prediction in zip(grouped, forecasts):
        reloaded = forecast_project(saved, row["quote"], saved["runtime"])
        assert reloaded["point_estimate"] == prediction["point_estimate"]
        assert prediction["calibrated_interval"] is None
        assert prediction["upper_budget_bound"] is None
    with pytest.raises(FileExistsError):
        train_project_forecast(source, destination)
    with pytest.raises(ValueError, match="outside"):
        train_project_forecast(source, source / "new-model")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "split", "quote", "usage", "raw"])
def test_project_training_rejects_incomplete_or_inconsistent_measurements(tmp_path, mutation):
    from token_yield.customer_pilot import aggregate_project_rows, train_project_forecast

    campaign = prepare_campaign(EXPERIMENT)
    source = tmp_path / "measured"
    execute_campaign(
        campaign, source, token_provider=lambda: "FAKE-SECRET",
        transport=FakeTransport(campaign, source),
        count_probe=lambda payload: SimpleNamespace(input_tokens=100),
    )
    campaign = json.loads((source / "protocol.json").read_text())
    rows = json.loads((source / "records.json").read_text())
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "split":
        rows[0]["split"] = "holdout"
    elif mutation == "quote":
        rows[0]["quote"]["prompt_bytes"] += 1
    elif mutation == "usage":
        rows[0]["usage"]["output_tokens"] += 1
        rows[0]["usage"]["total_tokens"] += 1
    else:
        response = source / f"{rows[0]['call_id']}.response.json"
        response.write_bytes(response.read_bytes() + b" ")
        destination = tmp_path / "rejected"
        with pytest.raises(ValueError, match="hash mismatch"):
            train_project_forecast(source, destination)
        assert not destination.exists()
        return
    with pytest.raises(ValueError):
        aggregate_project_rows(campaign, rows)


def test_project_aggregation_keeps_quality_failures_and_uses_only_declared_features(tmp_path):
    from token_yield.customer_pilot import aggregate_project_rows

    campaign = prepare_campaign(EXPERIMENT)
    source = tmp_path / "measured"
    execute_campaign(
        campaign, source, token_provider=lambda: "FAKE-SECRET",
        transport=FakeTransport(campaign, source),
        count_probe=lambda payload: SimpleNamespace(input_tokens=100),
    )
    campaign = json.loads((source / "protocol.json").read_text())
    rows = json.loads((source / "records.json").read_text())
    original = aggregate_project_rows(campaign, rows)
    for row in rows:
        row["quality"]["contract_passed"] = False
        row["output"] = "Different post-execution text must not become a predictor."
    altered = aggregate_project_rows(campaign, rows)
    assert [row["quote"] for row in original] == [row["quote"] for row in altered]
    assert [row["rated_cost_usd"] for row in original] == [row["rated_cost_usd"] for row in altered]
    assert len(altered) == 24 and not any(row["contract_passed"] for row in altered)


def test_pilot_cli_training_is_exclusive_and_does_not_prepare_live_campaign(tmp_path, monkeypatch):
    from examples import customer_request_pilot as cli

    with pytest.raises(SystemExit):
        cli.create_parser().parse_args(
            ["--run-dir", str(tmp_path), "--execute", "--train-from-run", str(tmp_path)])
    calls = []
    monkeypatch.setattr(cli, "train_project_forecast",
                        lambda source, target: calls.append((source, target)) or {"spent_usd": 0})
    monkeypatch.setattr(cli, "prepare_campaign",
                        lambda *args: pytest.fail("offline training must use the frozen protocol"))
    monkeypatch.setattr("sys.argv", ["pilot", "--run-dir", str(tmp_path / "out"),
                                     "--train-from-run", str(tmp_path / "in")])
    assert cli.main() == 0
    assert calls == [(tmp_path / "in", tmp_path / "out")]


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
