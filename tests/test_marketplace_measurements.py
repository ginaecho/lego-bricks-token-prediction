"""Offline-only measurement tests; scratch artifacts remain inside the repository."""

import copy
import json
import shutil
import uuid
from pathlib import Path

import pytest

from token_yield import marketplace_measurements as measurements
from token_yield.customer_decomposition import SCOPING_BRICKS, batch_candidates, content_hash
from token_yield.foundry_dispatch import FoundryDispatchError


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_manifest_matches_runtime():
    campaign = measurements.prepare_campaign(ROOT)
    assert campaign["manifest_sha256"] == content_hash(campaign["manifest"])
    assert campaign["manifest"]["preview"]["total_reservation_usd"] > 48


def test_measurement_service_versions_match_marketplace_catalog():
    from token_yield.customer_models import fit_token_models
    from token_yield.marketplace import service_catalog

    campaign = measurements.prepare_campaign(ROOT)
    template = campaign["manifest"]["template"]
    runtime = {"deployment": "synthetic-model", "deployment_version": "test-v1",
               "template_sha256": content_hash(template), "execution_policy": "no_tools_no_retries"}
    rows = [{
        "call_id": call["call_id"], "project_id": call["project_id"], "split": "train",
        "status": "completed", "quote": call["quote"], "rated_cost_usd": 0.01,
        "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300,
                  "cached_tokens": 0, "reasoning_tokens": 0},
    } for call in campaign["calls"] if call["group"] == "train"]
    catalog = service_catalog(template, fit_token_models(rows, runtime))
    versions = {service["id"]: service["version"] for service in catalog["services"]}
    assert versions == measurements.manifest_record(campaign)["service_versions"]
    assert all(selection["version"] == versions[selection["service_id"]]
               for call in campaign["calls"] for selection in call["service_selections"])


@pytest.fixture
def workspace(monkeypatch):
    root = ROOT / (".measurement-test-" + uuid.uuid4().hex)
    source = root / "experiments" / "customer_requests"
    source.mkdir(parents=True)
    (root / "token_yield").mkdir()
    (root / "experiments" / "marketplace").mkdir()
    for name in ("catalog.json", "template.json", "pilot.json"):
        shutil.copyfile(ROOT / "experiments" / "customer_requests" / name, source / name)
    for name in measurements.RUNTIME_FILES:
        shutil.copyfile(ROOT / "token_yield" / name, root / "token_yield" / name)
    campaign = measurements.build_campaign(root)
    (root / "experiments" / "marketplace" / "manifest.json").write_text(
        json.dumps(measurements.manifest_record(campaign)), encoding="utf-8")
    monkeypatch.setattr(measurements, "_audit_deployment", lambda *args: None)
    try:
        yield root, campaign
    finally:
        shutil.rmtree(root)


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
        else:
            artifacts[operation] = []
            for requirement in project["requirements"]:
                row = {"requirement_id": requirement["id"]}
                if operation == "extract":
                    row["evidence"] = requirement["text"]
                elif operation == "classify":
                    row.update(brick=requirement["brick"], owner=requirement["owner"])
                else:
                    row.update(owner=requirement["owner"], prerequisite_status="unverified",
                               review_required=True,
                               proposed_artifact="Proposed draft scoping checklist for this requirement.")
                artifacts[operation].append(row)
    return json.dumps(artifacts)


class Transport:
    def __init__(self, campaign, directory, mode="success"):
        self.campaign, self.directory, self.mode = campaign, directory, mode
        self.calls = 0

    def __call__(self, url, headers, body, timeout):
        call = self.campaign["calls"][self.calls]
        assert json.loads(body) == call["payload"]
        assert (self.directory / (call["call_id"] + ".request.json")).read_bytes() == body
        budget = json.loads((self.directory / "budget.json").read_text())
        assert call["call_id"] in budget["active_reservations"]
        assert budget["settled_safety_usd"] + budget["active_reserved_usd"] <= 48
        self.calls += 1
        if self.mode == "timeout":
            raise TimeoutError("Authorization: Bearer FAKE-SECRET")
        if self.mode == "http":
            return 429, b'{"error": "FAKE-SECRET rate limit"}'
        project = next(p for p in self.campaign["manifest"]["catalog"]["projects"]
                       if p["id"] == call["project_id"])
        response = {
            "id": f"fake-{self.calls}", "status": "completed", "model": "gpt-5.4",
            "output": [], "output_text": output(project, call["operations"]),
            "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        }
        if self.mode == "incomplete":
            response["status"] = "incomplete"
        if self.mode == "missing_usage":
            del response["usage"]["input_tokens_details"]
        if self.mode == "invalid_usage":
            response["usage"]["input_tokens"] = -1
        if self.mode == "identity":
            response["model"] = "other"
        if self.mode == "contract":
            response["output_text"] = "{}"
        if self.mode == "bounds":
            response["usage"]["input_tokens"] = call["max_input_tokens"] + 1
            response["usage"]["total_tokens"] = response["usage"]["input_tokens"] + 200
        if self.mode == "expensive":
            inp, out = call["max_input_tokens"], call["max_output_tokens"]
            response["usage"].update(input_tokens=inp, output_tokens=out, total_tokens=inp + out,
                                     output_tokens_details={"reasoning_tokens": out})
        return 200, json.dumps(response).encode()


def run(workspace, mode="success"):
    root, campaign = workspace
    directory = root / "runs" / "measurement"
    transport = Transport(campaign, directory, mode)
    result = measurements.execute_campaign(
        campaign, root, directory, execute=True,
        token_provider=lambda: "FAKE-SECRET", transport=transport)
    return directory, transport, result


def test_offline_deterministic_manifest_and_grouping(workspace, monkeypatch):
    root, campaign = workspace
    monkeypatch.setattr(measurements, "_default_transport", lambda *args: pytest.fail("network"))
    monkeypatch.setattr(measurements, "acquire_entra_token", lambda: pytest.fail("authentication"))
    assert measurements.prepare_campaign(root) == campaign == measurements.build_campaign(root)
    manifest = campaign["manifest"]
    assert manifest["preview"]["planned_calls"] == 120
    assert manifest["preview"]["planned_project_layouts"] == 48
    assert len({call["call_id"] for call in campaign["calls"]}) == 120
    assert set(measurements.LAYOUTS.values()) == set(batch_candidates())
    assert manifest["authorization"]["additional_cap_usd"] == 50
    assert manifest["authorization"]["operational_stop_usd"] == 48
    assert manifest["grouping"]["calibration_projects"] == 0
    assert manifest["grouping"]["fresh_test_projects"] == 0
    assert sum(p["group"] == "historical_holdout" for p in manifest["projects"]) == 2
    assert not (root / "runs").exists()
    for project in manifest["projects"]:
        for variant in measurements.LAYOUTS:
            calls = [call for call in campaign["calls"]
                     if call["project_id"] == project["project_id"] and call["variant"] == variant]
            assert tuple(op for call in calls for op in call["operations"]) == SCOPING_BRICKS
            assert {call["group"] for call in calls} == {project["group"]}
            assert {call["plan_sha256"] for call in calls} == {project["plan_sha256"]}
            for op in SCOPING_BRICKS:
                assert sum(call["quote"]["counts"][op] for call in calls) == project["counts"][op]
            for call in calls:
                assert call["payload_sha256"] == content_hash(call["payload"])
                assert "previous_response_id" not in call["payload"]
                assert "tools" not in call["payload"]
                for service in call["service_selections"]:
                    assert service["version"] == (
                        "marketplace-services-v1:" + manifest["provenance"]["template_sha256"])
                    assert service["quantity"] == project["counts"][service["service_id"]]


@pytest.mark.parametrize("target", ["catalog", "runtime", "manifest", "campaign"])
def test_changed_hashes_and_payloads_fail_closed(workspace, target):
    root, campaign = workspace
    if target == "campaign":
        campaign = copy.deepcopy(campaign)
        campaign["calls"][0]["payload"]["input"] += "tampered"
        with pytest.raises(ValueError, match="differs"):
            measurements.execute_campaign(campaign, root, root / "runs" / "bad", execute=True)
    else:
        path = {
            "catalog": root / "experiments" / "customer_requests" / "catalog.json",
            "runtime": root / "token_yield" / "budget.py",
            "manifest": root / "experiments" / "marketplace" / "manifest.json",
        }[target]
        if target == "manifest":
            data = json.loads(path.read_text())
            data["authorization"]["additional_cap_usd"] = 500
            path.write_text(json.dumps(data), encoding="utf-8")
        else:
            path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="hash changed|differs"):
            measurements.prepare_campaign(root)
    assert not (root / "runs").exists()


def test_explicit_gate_and_unsupported_variants(workspace):
    root, campaign = workspace
    with pytest.raises(ValueError, match="explicit"):
        measurements.execute_campaign(campaign, root, root / "runs" / "not-paid")
    with pytest.raises(ValueError, match="unsupported"):
        measurements.prepare_campaign(root, measurements.UNSUPPORTED_VARIANT)
    with pytest.raises(ValueError, match="complete frozen matrix"):
        measurements.prepare_campaign(root, "paired-v1")
    assert not (root / "runs").exists()


def test_success_preserves_sources_and_approval_cannot_reset(workspace):
    root, campaign = workspace
    originals = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    directory, transport, result = run(workspace)
    assert transport.calls == 120
    assert result["status"] == "completed_research_only"
    assert result["completed_calls"] == 120
    assert result["budget"]["active_reservations"] == {}
    assert all(w["complete"] and w["all_contract_checks_passed"] for w in result["workflows"])
    assert result["quality_noninferiority_established"] is False
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert len(list(directory.glob("*.request.json"))) == 120
    with pytest.raises(FileExistsError):
        measurements.execute_campaign(campaign, root, root / "runs" / "another", execute=True)
    with pytest.raises(FileExistsError):
        measurements.execute_campaign(campaign, root, directory, execute=True)
    with pytest.raises(ValueError, match="direct child"):
        measurements.execute_campaign(campaign, root, directory / "nested", execute=True)


@pytest.mark.parametrize("mode,known", [
    ("timeout", False), ("http", False), ("missing_usage", False), ("invalid_usage", False),
    ("incomplete", True), ("identity", True), ("contract", True), ("bounds", True),
])
def test_failure_evidence_reservations_and_no_retry(workspace, mode, known):
    root, campaign = workspace
    directory = root / "runs" / "failure"
    transport = Transport(campaign, directory, mode)
    with pytest.raises((FoundryDispatchError, ValueError, RuntimeError)):
        measurements.execute_campaign(
            campaign, root, directory, execute=True,
            token_provider=lambda: "FAKE-SECRET", transport=transport)
    assert transport.calls == 1
    summary = json.loads((directory / "analysis.json").read_text())
    row = json.loads((directory / "records.json").read_text())[0]
    assert summary["status"] == "halted"
    assert row["status"] != "completed"
    assert (row["usage"] is not None) == known
    assert bool(summary["budget"]["active_reservations"]) != known
    assert (row["rated_cost_usd"] is not None) == known
    assert row["dispatched"] and row["started_at"] and row["finished_at"]
    assert not any(w["complete"] for w in summary["workflows"])
    assert len(list(directory.glob("*.request.json"))) == 1
    assert len(list(directory.glob("*.response.json"))) == (mode != "timeout")
    for path in directory.iterdir():
        if path.is_file():
            assert b"FAKE-SECRET" not in path.read_bytes()


def test_budget_stop_persists_blocked_attempt_without_dispatch(workspace):
    directory, transport, result = run(workspace, "expensive")
    assert result["status"] == "budget_stopped"
    assert 0 < transport.calls < 120
    assert result["budget"]["settled_safety_usd"] <= 48
    rows = json.loads((directory / "records.json").read_text())
    assert rows[-1]["status"] == "budget_blocked"
    assert not rows[-1]["dispatched"]
    assert len(rows) == transport.calls + 1
    assert rows[-1]["usage"] is None
    assert result["budget"]["active_reservations"] == {}


def test_auth_failure_is_persisted_without_secret_or_spend(workspace):
    root, campaign = workspace
    directory = root / "runs" / "auth-failure"
    def no_token():
        raise RuntimeError("secret may appear in authentication errors")
    with pytest.raises(RuntimeError):
        measurements.execute_campaign(campaign, root, directory, execute=True, token_provider=no_token)
    summary = json.loads((directory / "analysis.json").read_text())
    assert summary["status"] == "halted"
    assert summary["attempt_records"] == 0
    assert summary["budget"]["active_reserved_usd"] == 0
    assert "secret" not in (directory / "failure.json").read_text()
