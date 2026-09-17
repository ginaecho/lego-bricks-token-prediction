"""Synthetic, offline T011 fixtures. No active measurement directory is opened."""

import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from examples.marketplace_models import main
from token_yield import marketplace_measurements as measurements
from token_yield import marketplace_models as models
from token_yield.budget import HardBudget
from token_yield.customer_decomposition import check_output, content_hash
from token_yield.economics import Pricing


ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-09-16T10:00:00+00:00"


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def output(project, operations):
    artifacts = {}
    for operation in operations:
        if operation == "report":
            artifacts[operation] = {
                "source_url": project["source_url"],
                "requirement_ids": [r["id"] for r in project["requirements"]],
                "exclusions": project["exclusions"], "delivery_status": "not_executed",
                "summary": "Draft scoping only; private evidence and human approval remain missing.",
            }
            continue
        artifacts[operation] = []
        for requirement in project["requirements"]:
            item = {"requirement_id": requirement["id"]}
            if operation == "extract":
                item["evidence"] = requirement["text"]
            elif operation == "classify":
                item.update(brick=requirement["brick"], owner=requirement["owner"])
            else:
                item.update(owner=requirement["owner"], prerequisite_status="unverified",
                            review_required=True,
                            proposed_artifact="Proposed draft scoping checklist for this requirement.")
            artifacts[operation].append(item)
    return json.dumps(artifacts)


@pytest.fixture
def completed():
    root = ROOT / (".workflow-test-" + uuid.uuid4().hex)
    source = root / "experiments" / "customer_requests"
    source.mkdir(parents=True)
    (root / "token_yield").mkdir()
    (root / "experiments" / "marketplace").mkdir()
    directory = root / "runs" / "synthetic-completed"
    directory.mkdir(parents=True)
    try:
        for name in ("catalog.json", "template.json", "pilot.json"):
            shutil.copyfile(ROOT / "experiments" / "customer_requests" / name, source / name)
        for name in measurements.RUNTIME_FILES:
            shutil.copyfile(ROOT / "token_yield" / name, root / "token_yield" / name)
        campaign = measurements.build_campaign(root)
        write(root / "experiments" / "marketplace" / "manifest.json", measurements.manifest_record(campaign))
        write(directory / "protocol.json", campaign)
        config = campaign["manifest"]["config"]
        for phase in ("before", "after"):
            write(directory / f"deployment-{phase}.json", {
                "observed_at": NOW,
                "metadata": {"model": {"name": config["deployment"], "version": config["deployment_version"]},
                             "sku": config["deployment_sku"], "state": "Succeeded"},
            })
        budget = HardBudget(config["cap_usd"], config["stop_usd"])
        pricing, rows = Pricing(**config["pricing"]), []
        projects = {p["id"]: p for p in campaign["manifest"]["catalog"]["projects"]}
        for call in campaign["calls"]:
            project = projects[call["project_id"]]
            text = output(project, call["operations"])
            # Each layout has actual observed overhead; no assumed discount target.
            inp = 100 + call["quote"]["context_bytes"] // 10 + len(call["operations"]) * 20
            out = 150 + len(project["requirements"]) * len(call["operations"]) * 10
            usage = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                     "cached_tokens": 0, "reasoning_tokens": 0}
            response = {
                "id": "synthetic-" + call["call_id"], "status": "completed",
                "model": config["expected_response_model"], "output": [], "output_text": text,
                "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}},
            }
            request_raw, response_raw = json.dumps(call["payload"]).encode(), json.dumps(response).encode()
            (directory / (call["call_id"] + ".request.json")).write_bytes(request_raw)
            (directory / (call["call_id"] + ".response.json")).write_bytes(response_raw)
            request_hash = hashlib.sha256(request_raw).hexdigest()
            response_hash = hashlib.sha256(response_raw).hexdigest()
            rated = pricing.attempt_cost(input_tokens=inp, cached_input_tokens=0, output_tokens=out)
            rows.append({
                **{key: value for key, value in call.items() if key != "payload"},
                "started_at": NOW, "dispatched_at": NOW, "response_received_at": NOW, "finished_at": NOW,
                "status": "completed", "response_status": "completed", "dispatched": True,
                "request_sha256": request_hash, "response_sha256": response_hash,
                "stored_response_sha256": response_hash, "response_redacted": False,
                "response_model": config["expected_response_model"], "output": text,
                "quality": check_output(project, tuple(call["operations"]), text),
                "usage": usage, "rated_cost_usd": rated, "http_status": 200,
                "response_calls": [{
                    "target": call["call_id"], "response_id": response["id"], "status": "completed",
                    "model": response["model"], "elapsed_ms": 1, "usage": usage,
                    "request_sha256": request_hash, "response_sha256": response_hash,
                }],
            })
            budget.reserve(call["call_id"], call["max_input_tokens"], 2 * call["max_output_tokens"])
            budget.settle(call["call_id"], usage, rated)
        write(directory / "records.json", rows)
        write(directory / "budget.json", budget.snapshot())
        write(directory / "analysis.json", measurements._summary(campaign, rows, budget, "completed_research_only"))
        write(directory / "state.json", {"status": "completed_research_only", "at": NOW, "no_automatic_retry": True})
        yield root, directory, campaign
    finally:
        shutil.rmtree(root)


def fit(completed):
    root, directory, _ = completed
    audit = models.audit_run(root, directory)
    training = [r for r in audit["workflows"] if r["group"] == "train"]
    holdout = [r for r in audit["workflows"] if r["group"] == "historical_holdout"]
    model = models.fit_workflow_models(training, audit["runtime"], audit["provenance"])
    return audit, training, holdout, model


def test_complete_audit_quote_only_and_persistent_offline_pipeline(completed, monkeypatch):
    root, directory, campaign = completed
    monkeypatch.setattr(measurements, "acquire_entra_token", lambda: pytest.fail("network auth forbidden"))
    monkeypatch.setattr(measurements, "_default_transport", lambda *args: pytest.fail("network forbidden"))
    audit, training, _, model = fit(completed)
    assert len(audit["workflows"]) == 48 and len(training) == 32
    validation = model["selection"]["validation"]["input_tokens"]
    assert validation["composition"]["group_macro_mae"] < validation["constant"]["group_macro_mae"]
    assert len(audit["provenance"]["source_file_sha256"]) == 247
    row = training[0]
    project = next(p for p in campaign["manifest"]["catalog"]["projects"] if p["id"] == row["project_id"])
    quote = models.workflow_quote(project, audit["template"], row["variant"])
    assert quote == row["calls"]
    assert models.predict_workflow(model, quote, row["variant"], audit["runtime"])["status"] == "research_estimate"
    report = models.run_offline(root, directory, root / "runs" / "composition-research")
    assert report["saved_baseline_replaced"] is False
    assert report["production_recommendation"] is None
    assert report["historical_holdout"]["selection_influence"] is False
    saved = models.load_model(read(root / "runs" / "composition-research" / "model.json"))
    assert saved == model
    for name, digest in audit["provenance"]["source_file_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("target", [
    "protocol", "record_quote", "split", "missing_call", "duplicate_call", "request_hash",
    "request_payload", "response_hash", "usage", "output", "quality", "ledger", "model",
    "rated_cost", "budget", "analysis", "deployment_before", "deployment_after", "time", "runtime",
])
def test_tampering_rejected(completed, target):
    root, directory, campaign = completed
    rows = read(directory / "records.json")
    row = rows[0]
    if target == "protocol":
        value = read(directory / "protocol.json")
        value["calls"][0]["payload"]["input"] += "changed"
        write(directory / "protocol.json", value)
    elif target in ("request_hash", "request_payload"):
        path = directory / (row["call_id"] + ".request.json")
        value = read(path)
        value["input"] += "changed"
        write(path, value)
        if target == "request_payload":
            row["request_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif target == "response_hash":
        path = directory / (row["call_id"] + ".response.json")
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "runtime":
        path = root / "token_yield" / "budget.py"
        path.write_bytes(path.read_bytes() + b"\n")
    elif target.startswith("deployment_"):
        path = directory / (target.replace("_", "-") + ".json")
        value = read(path)
        value["metadata"]["model"]["version"] = "unapproved"
        write(path, value)
    elif target in ("budget", "analysis"):
        path = directory / (target + ".json")
        value = read(path)
        value["fabricated"] = True
        write(path, value)
    elif target == "record_quote":
        row["quote"]["prompt_bytes"] += 1
    elif target == "split":
        row["group"] = "historical_holdout"
    elif target == "missing_call":
        rows.pop()
    elif target == "duplicate_call":
        rows[-1] = rows[0]
    elif target == "usage":
        row["usage"]["input_tokens"] += 1
    elif target == "output":
        row["output"] = "{}"
    elif target == "quality":
        row["quality"]["human_review_required"] = False
    elif target == "ledger":
        row["response_calls"][0]["target"] = "another-call"
    elif target == "model":
        row["response_model"] = "another-model"
    elif target == "rated_cost":
        row["rated_cost_usd"] += 0.01
    elif target == "time":
        row["finished_at"] = "2026-09-17T10:00:00+00:00"
    write(directory / "records.json", rows)
    with pytest.raises((ValueError, RuntimeError)):
        models.audit_run(root, directory)


@pytest.mark.parametrize("target", ["missing_usage", "invalid_usage", "contract", "incomplete", "identity", "tools"])
def test_raw_evidence_not_just_hashes_is_required(completed, target):
    root, directory, _ = completed
    rows = read(directory / "records.json")
    row = rows[0]
    path = directory / (row["call_id"] + ".response.json")
    raw = read(path)
    if target == "missing_usage":
        del raw["usage"]["input_tokens_details"]
    elif target == "invalid_usage":
        raw["usage"]["input_tokens"] = True
    elif target == "contract":
        raw["output_text"] = "{}"
        row["output"] = "{}"
    elif target == "incomplete":
        raw["status"] = "incomplete"
    elif target == "identity":
        raw["model"] = "unapproved"
    else:
        raw["output"] = [{"type": "function_call", "name": "unapproved"}]
    write(path, raw)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row["response_sha256"] = row["stored_response_sha256"] = digest
    row["response_calls"][0]["response_sha256"] = digest
    write(directory / "records.json", rows)
    with pytest.raises((ValueError, RuntimeError)):
        models.audit_run(root, directory)


@pytest.mark.parametrize("status", ["running", "halted", "budget_stopped"])
def test_partial_live_labels_never_opened(completed, monkeypatch, status):
    root, directory, _ = completed
    write(directory / "state.json", {"status": status})
    original = Path.read_bytes

    def guarded(path):
        if path.name in ("records.json", "protocol.json"):
            pytest.fail("partial live labels were opened")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="labels not read"):
        models.audit_run(root, directory)
    destination = root / "runs" / "must-not-be-created"
    with pytest.raises(ValueError, match="source run is incomplete; labels not read"):
        models.run_offline(root, directory, destination)
    assert not destination.exists()


def test_leakage_exclusion_and_deterministic_grouped_selection(completed, monkeypatch):
    audit, training, holdout, model = fit(completed)
    fitted_groups = []
    original = models._fit_model

    def tracked(rows, target, form):
        groups = {r["project_id"] for r in rows}
        assert not groups & {r["project_id"] for r in holdout}
        fitted_groups.append(groups)
        return original(rows, target, form)

    monkeypatch.setattr(models, "_fit_model", tracked)
    reordered = models.fit_workflow_models(list(reversed(training)), audit["runtime"], audit["provenance"])
    assert reordered == model
    expected_groups = model["support"]["training_projects"]
    assert len(fitted_groups) == 3 * 3 * 5
    for offset in range(0, len(fitted_groups), 5):
        for fold, held in enumerate(expected_groups):
            assert fitted_groups[offset + fold] == set(expected_groups) - {held}
        assert fitted_groups[offset + 4] == set(expected_groups)
    for target in models.TARGETS:
        for candidate in models.FORMS:
            for fold in model["selection"]["validation"][target][candidate]["folds"]:
                assert fold["validation_project"] not in fold["training_projects"]
                assert len(fold["predictions"]) == 8
    with pytest.raises(ValueError, match="holdout"):
        models.fit_workflow_models(training + holdout, audit["runtime"], audit["provenance"])
    relabelled = copy.deepcopy(training)
    old_id = relabelled[0]["project_id"]
    for row in relabelled:
        if row["project_id"] == old_id:
            row["project_id"] = holdout[0]["project_id"]
    with pytest.raises(ValueError, match="fixed training"):
        models.fit_workflow_models(relabelled, audit["runtime"], audit["provenance"])
    altered = copy.deepcopy(holdout)
    for row in altered:
        row["targets"]["output_tokens"] *= 100
        for targets in row["call_targets"]:
            targets["output_tokens"] *= 100
    before = copy.deepcopy(model)
    models.evaluate_historical_holdout(model, altered)
    assert model == before


@pytest.mark.parametrize("target", ["runtime", "template", "variant", "order", "missing", "quantity", "range"])
def test_unsupported_quotes_abstain(completed, target):
    audit, training, _, model = fit(completed)
    row = copy.deepcopy(training[0])
    runtime = copy.deepcopy(audit["runtime"])
    if target == "runtime":
        runtime["deployment_version"] = "new-version"
    elif target == "template":
        row["calls"][0]["service_selections"][0]["version"] = "unknown"
    elif target == "variant":
        row["variant"] = "sequential-handoff-v2"
    elif target == "order":
        row["calls"][0]["operations"] = ["unknown"]
    elif target == "missing":
        row["calls"].pop()
    elif target == "quantity":
        row["calls"][0]["quote"]["counts"]["report"] += 1
    else:
        row["calls"][0]["quote"]["prompt_bytes"] += 100000
    result = models.predict_workflow(model, row["calls"], row["variant"], runtime)
    assert result["status"] == "abstained"
    assert result["predictions"] is None
    assert result["calibrated_intervals"] is None


def test_json_reload_without_refit_and_serialized_tampering(completed, monkeypatch):
    audit, training, _, model = fit(completed)
    expected = models.predict_workflow(model, training[0]["calls"], training[0]["variant"], audit["runtime"])
    monkeypatch.setattr(models, "_fit_model", lambda *args: pytest.fail("refitting on reload"))
    reloaded = models.load_model(json.loads(json.dumps(model)))
    assert models.predict_workflow(reloaded, training[0]["calls"], training[0]["variant"], audit["runtime"]) == expected
    tampered = copy.deepcopy(model)
    tampered["candidates"]["input_tokens"]["constant"]["value"] += 1
    with pytest.raises(ValueError, match="digest"):
        models.load_model(tampered)
    tampered = copy.deepcopy(model)
    tampered["candidates"]["input_tokens"]["composition"]["scales"] = []
    tampered["model_sha256"] = content_hash({k: v for k, v in tampered.items() if k != "model_sha256"})
    with pytest.raises(ValueError, match="dimensions"):
        models.load_model(tampered)


def test_holdout_out_of_range_is_reported_not_silently_scored(completed):
    _, _, holdout, model = fit(completed)
    rows = copy.deepcopy(holdout)
    for row in rows:
        row["calls"][0]["quote"]["prompt_bytes"] += 100000
        row["features"] = models._features(row["calls"], row["variant"], model["support"]["runtime"])
    report = models.evaluate_historical_holdout(model, rows)
    assert report["supported_workflows"] == 0
    assert report["total_workflows"] == 16
    assert all(row["candidate_comparison"] is None for row in report["rows"])
    assert report["supported_only_group_macro_mae"]["input_tokens"]["composition"] is None


def test_redacted_source_and_duplicate_json_fail_closed(completed):
    root, directory, _ = completed
    rows = read(directory / "records.json")
    rows[0]["response_redacted"] = True
    write(directory / "records.json", rows)
    with pytest.raises(ValueError, match="redacted"):
        models.audit_run(root, directory)
    with pytest.raises(ValueError, match="duplicate JSON"):
        models._json(b'{"status":"running","status":"completed_research_only"}')


def test_source_change_during_audit_rejected(completed, monkeypatch):
    root, directory, _ = completed
    original = models._summary

    def changed(*args):
        value = original(*args)
        (directory / "state.json").write_bytes((directory / "state.json").read_bytes() + b" ")
        return value

    monkeypatch.setattr(models, "_summary", changed)
    with pytest.raises(ValueError, match="source changed"):
        models.audit_run(root, directory)


def test_incomplete_training_matrix_rejected(completed):
    audit, training, _, _ = fit(completed)
    with pytest.raises(ValueError, match="incomplete workflow matrix"):
        models.fit_workflow_models(training[:-1], audit["runtime"], audit["provenance"])


def test_cli_destination_protection_before_source_read(completed, monkeypatch, capsys):
    root, source, _ = completed
    monkeypatch.setattr(models, "audit_run", lambda *args: pytest.fail("destination rejected before source audit"))
    for destination in (source, root / "outside", root / "runs" / "nested" / "child"):
        with pytest.raises(ValueError, match="destination"):
            models.run_offline(root, source, destination)
    existing = root / "runs" / "existing-baseline"
    existing.mkdir()
    write(existing / "baseline.json", {"unchanged": True})
    with pytest.raises(FileExistsError):
        models.run_offline(root, source, existing)
    assert read(existing / "baseline.json") == {"unchanged": True}
    with pytest.raises(SystemExit) as caught:
        main(["--source-run", str(source), "--run-dir", str(ROOT / "README.md")])
    assert caught.value.code == 2
    assert "refused" in capsys.readouterr().err


def test_cli_success_routes_only_offline(monkeypatch, capsys):
    from examples import marketplace_models as cli

    calls = []

    def offline(root, source, destination):
        calls.append((root, source, destination))
        return {"status": "completed_research_only", "selected": {"input_tokens": "constant"}}

    monkeypatch.setattr(cli, "run_offline", offline)
    assert cli.main(["--source-run", "runs\\completed", "--run-dir", "runs\\research-new"]) == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["production_recommendation"] is None


def test_cli_runtime_audit_failure_is_a_refusal(monkeypatch, capsys):
    from examples import marketplace_models as cli

    def failed(*args):
        raise RuntimeError("invalid measured response")

    monkeypatch.setattr(cli, "run_offline", failed)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--source-run", "runs\\completed", "--run-dir", "runs\\research-new"])
    assert caught.value.code == 2
    assert "refused" in capsys.readouterr().err


def test_cli_preview_is_audit_only(monkeypatch, capsys):
    from examples import marketplace_models as cli

    monkeypatch.setattr(cli, "run_offline", lambda *args: pytest.fail("preview must not fit or write"))
    monkeypatch.setattr(cli, "audit_run", lambda *args: {
        "workflows": [None] * 48, "provenance": {"campaign_sha256": "synthetic-digest"},
    })
    assert cli.main(["--source-run", "runs\\completed", "--audit-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["workflow_count"] == 48 and result["files_written"] == 0 and result["fitted"] is False


def test_cli_preview_and_fit_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--source-run", "runs\\completed", "--audit-only", "--run-dir", "runs\\new"])
    assert caught.value.code == 2


@pytest.mark.parametrize("error,expected", [(KeyboardInterrupt, 130), (BrokenPipeError, 1)])
def test_cli_handles_interrupt_and_broken_pipe(monkeypatch, error, expected):
    from examples import marketplace_models as cli

    def failed(*args):
        raise error()

    monkeypatch.setattr(cli, "audit_run", failed)
    assert cli.main(["--source-run", "runs\\completed", "--audit-only"]) == expected
