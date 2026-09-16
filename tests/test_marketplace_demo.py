"""Regression coverage for actual offline fitting and the loopback HTTP boundary."""

from __future__ import annotations

import http.client
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import pytest

from examples import marketplace_demo_server as server_module
from token_yield.marketplace_demo import (
    BASE_FEATURES,
    RATES,
    extract_requirements,
    run_pipeline,
    sample_documents,
    validate_request,
)
from token_yield.robust import RidgeLinearModel

ROOT = Path(__file__).resolve().parents[1]
REQUEST = {
    "description": "Review this fictional security documentation project and identify gaps.",
    "model_id": "gpt-mini",
    "runs_per_month": 20,
}


@pytest.fixture
def artifact_dir():
    """Keep all test artifacts within this isolated worktree, never OS temp."""
    parent = ROOT / ".demo-test-runs"
    folder = parent / uuid.uuid4().hex
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        shutil.rmtree(folder)
        try:
            parent.rmdir()
        except OSError:
            pass


def test_original_sources_link_and_deliberate_gaps():
    documents = sample_documents()
    ids = {doc["id"] for doc in documents}
    assert len(ids) == 4
    assert all(link in ids for doc in documents for link in doc["links"])
    assert next(doc for doc in documents if doc["id"] == "harbor-security")["links"] == [
        "harbor-operations"
    ]
    requirements = extract_requirements(documents)
    assert [req["status"] for req in requirements] == [
        "evidence_found", "partial", "partial", "missing", "missing", "evidence_found",
    ]
    project_text = {doc["id"]: doc["text"] for doc in documents if doc["role"] == "project"}
    for req in requirements:
        assert req["source_id"] == "demo-reference"
        for evidence in req["evidence"]:
            assert evidence["quote"] in project_text[evidence["document_id"]]
    assert sample_documents() == documents


def test_pipeline_fit_persistence_events_and_holdout(artifact_dir, monkeypatch):
    fit_calls = []
    original_fit = RidgeLinearModel.fit

    def recording_fit(records, alpha=1.0):
        fit_calls.append(records)
        return original_fit(records, alpha)

    monkeypatch.setattr(RidgeLinearModel, "fit", recording_fit)
    events = []
    result = run_pipeline(REQUEST, artifact_dir, on_event=events.append)
    assert len(fit_calls) == 4  # Baseline + expanded fit, input and output each.
    assert all(records for records in fit_calls)
    assert all("test" not in str(record.group) for records in fit_calls for record in records)
    assert [event["stage"] for event in events] == [
        "decompose", "wiki", "requirements", "predict_before", "simulate",
        "features", "train", "evaluate", "predict_after", "complete",
    ]
    assert [event["seq"] for event in events] == list(range(1, 11))
    assert all(event["time"] and isinstance(event["data"], dict) for event in events)
    training = result["training"]
    assert training["source"] == "synthetic"
    assert training["train_count"] == 200 and training["test_count"] == 50
    rows = training["rows"]
    train_groups = {row["group"] for row in rows if row["split"] == "train"}
    test_groups = {row["group"] for row in rows if row["split"] == "test"}
    assert train_groups.isdisjoint(test_groups)
    assert {record.group for record in fit_calls[-1]} == train_groups
    assert all(row["synthetic"] is True for row in rows)
    assert all(row["unit"] == "one entire project execution" for row in rows)
    folder = artifact_dir / result["id"]
    assert json.loads((folder / "request.json").read_text()) == validate_request(REQUEST)
    assert json.loads((folder / "result.json").read_text()) == result
    assert json.loads((folder / "training.json").read_text()) == training
    assert [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()] == events
    persisted = json.loads((folder / "model.json").read_text())
    for channel in ("input", "output"):
        fitted = RidgeLinearModel(**persisted["models"][channel])
        features = [result["features"][name] for name in result["feature_names"]]
        assert result["after"][f"{channel}_tokens"] == round(fitted.predict(features))
        test_rows = [row for row in rows if row["split"] == "test"]
        actual_mae = sum(abs(row[f"{channel}_tokens"] - fitted.predict([
            row["features"][name] for name in result["feature_names"]
        ])) for row in test_rows) / len(test_rows)
        assert training[f"{channel}_mae"] == pytest.approx(actual_mae)
    assert result["before"]["supported"] is True
    assert result["after"]["total_tokens"] != result["before"]["total_tokens"]
    assert any("not establish real-world" in limitation for limitation in result["limitations"])


@pytest.mark.parametrize("novel_request", [
    {"new_function": "Policy-as-code parsing"},
    {"description": REQUEST["description"] + " Include policy-as-code parsing."},
    {"new_function": "Custom evidence graph scoring"},
])
def test_novelty_abstains_expands_and_changes_fitted_prediction(artifact_dir, novel_request):
    baseline = run_pipeline(REQUEST, artifact_dir)
    result = run_pipeline({**REQUEST, **novel_request}, artifact_dir)
    before = result["before"]
    assert before["supported"] is False
    assert before["input_tokens"] is before["output_tokens"] is before["total_tokens"] is None
    assert "Missing catalog feature" in before["reason"]
    new_features = set(result["feature_names"]) - set(BASE_FEATURES)
    assert len(new_features) == 1
    novel_feature = new_features.pop()
    assert result["features"][novel_feature] == 1
    assert sum(brick["novel"] for brick in result["bricks"]) == 1
    training = result["training"]
    assert any(row["features"][novel_feature] > 0 for row in training["rows"])
    assert all(novel_feature in row["features"] for row in training["rows"])
    for channel in ("input", "output"):
        coeff = training["coefficients"][channel]
        assert coeff[novel_feature] > 0
        assert coeff["document_units"] != before["coefficients"][channel]["document_units"]
        raw_prediction = coeff["intercept"] + sum(
            result["features"][name] * coeff[name] for name in result["feature_names"]
        )
        assert result["after"][f"{channel}_tokens"] == round(raw_prediction)
        assert result["after"][f"{channel}_tokens"] > baseline["after"][f"{channel}_tokens"]


@pytest.mark.parametrize("model_id", list(RATES))
def test_illustrative_rates_and_zero_monthly_execution(artifact_dir, model_id):
    result = run_pipeline({**REQUEST, "model_id": model_id, "runs_per_month": 0}, artifact_dir)
    after = result["after"]
    input_rate, output_rate = RATES[model_id]
    assert after["usd_per_run"] == pytest.approx(
        (after["input_tokens"] * input_rate + after["output_tokens"] * output_rate) / 1_000_000
    )
    assert after["usd_per_run"] > 0 and after["usd_per_month"] == 0
    assert after["total_tokens"] == after["input_tokens"] + after["output_tokens"]
    assert all(type(after[key]) is int for key in ("input_tokens", "output_tokens", "total_tokens"))
    assert result["rates"]["illustrative"] is True


def test_deep_research_is_offline_only(artifact_dir):
    result = run_pipeline({**REQUEST, "description": REQUEST["description"] + " Deep research."}, artifact_dir)
    assert result["features"]["offline_source_reviews"] == 1
    assert any(brick["name"] == "Offline source review" for brick in result["bricks"])
    assert result["before"]["supported"] is True
    assert any("no internet or LLM calls" in value for value in result["limitations"])


@pytest.mark.parametrize("body", [
    None, [], {}, {**REQUEST, "description": "short"},
    {**REQUEST, "description": "a" * 6001},
    {**REQUEST, "model_id": "unknown"}, {**REQUEST, "model_id": []},
    {**REQUEST, "runs_per_month": -1}, {**REQUEST, "runs_per_month": True},
    {**REQUEST, "runs_per_month": 1.5}, {**REQUEST, "runs_per_month": 1_000_001},
    {**REQUEST, "new_function": None}, {**REQUEST, "new_function": "a" * 121},
    {**REQUEST, "run_dir": ".."}, {**REQUEST, "description": "a" * 20 + "\ud800"},
])
def test_invalid_request_rejected(body):
    with pytest.raises(ValueError):
        validate_request(body)


def _http(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        return response.status, response.getheaders(), data
    finally:
        connection.close()


def _wait_terminal(store, run_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run = store.get(run_id)
        if run["status"] in ("completed", "failed"):
            return run
        threading.Event().wait(0.005)
    pytest.fail("offline worker did not finish within 10 seconds")


@pytest.fixture
def http_server(artifact_dir):
    store = server_module.RunStore(artifact_dir)
    server = server_module.make_server(0, store=store)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server, store
    finally:
        for run in store.listing()["runs"]:
            _wait_terminal(store, run["id"])
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_contract_full_run_and_json_stdout(http_server, capsys):
    server, store = http_server
    status, _, body = _http(server, "GET", "/api/health")
    assert status == 200 and json.loads(body) == {"status": "ok"}
    status, _, body = _http(server, "POST", "/api/runs", json.dumps(REQUEST), {
        "Content-Type": "application/json", "Origin": f"http://127.0.0.1:{server.server_port}",
    })
    assert status == 202
    created = json.loads(body)
    run_id = created["id"]
    assert created["operations_url"] == f"/marketplace-operations-demo.html?run={run_id}"
    run = _wait_terminal(store, run_id)
    assert run["status"] == "completed" and run["error"] is None
    assert run["result"]["after"]["usd_per_month"] == run["result"]["after"]["usd_per_run"] * 20
    status, _, body = _http(server, "GET", f"/api/runs/{run_id}")
    assert status == 200 and json.loads(body) == run
    status, _, body = _http(server, "GET", "/api/runs")
    assert json.loads(body)["runs"] == [{
        "id": run_id, "status": "completed", "description": REQUEST["description"],
    }]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["seq"] for line in lines] == list(range(1, 11))
    assert all(line["run_id"] == run_id for line in lines)
    run["events"].clear()
    assert len(store.get(run_id)["events"]) == 10  # Returned snapshots cannot mutate storage.


@pytest.mark.parametrize("path", [
    "/README.md", "/.git/config", "/.demo-runs/result.json", "/token_yield/robust.py",
    "/../README.md", "/%2e%2e/README.md", "/api/runs/not-an-id",
])
def test_http_static_allowlist(http_server, path):
    server, _ = http_server
    assert _http(server, "GET", path)[0] == 404


def test_http_static_page_and_host_origin_security(http_server):
    server, _ = http_server
    status, headers, body = _http(server, "GET", "/marketplace-sales-demo.html")
    assert status == 200 and b"<html" in body.lower()
    assert dict(headers)["X-Content-Type-Options"] == "nosniff"
    assert server.server_address[0] == "127.0.0.1"
    for headers in [
        {"Host": f"attacker.example:{server.server_port}"},
        {"Host": "localhost:99999"},
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
        {"Origin": f"http://127.0.0.1:{server.server_port + 1}"},
        {"Sec-Fetch-Site": "cross-site"},
    ]:
        assert _http(server, "GET", "/api/health", headers=headers)[0] == 403


@pytest.mark.parametrize("body,headers,expected", [
    ("{", {"Content-Type": "application/json"}, 400),
    ("[]", {"Content-Type": "application/json"}, 400),
    (json.dumps({**REQUEST, "model_id": "oops"}), {"Content-Type": "application/json"}, 400),
    (json.dumps({**REQUEST, "runs_per_month": True}), {"Content-Type": "application/json"}, 400),
    ('{"description":NaN}', {"Content-Type": "application/json"}, 400),
    ('{"description":"a","description":"b"}', {"Content-Type": "application/json"}, 400),
    (b"\xff", {"Content-Type": "application/json"}, 400),
    (json.dumps(REQUEST), {"Content-Type": "text/plain"}, 415),
    (" " * (server_module.MAX_BODY + 1), {"Content-Type": "application/json"}, 413),
    ("{}", {"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
], ids=["malformed", "array", "model", "boolean", "nan", "duplicate", "encoding",
        "content-type", "oversized", "transfer-encoding"])
def test_http_invalid_bodies_do_not_launch_workers(http_server, body, headers, expected):
    server, store = http_server
    assert _http(server, "POST", "/api/runs", body, headers)[0] == expected
    assert store.listing() == {"runs": []}


def test_worker_failure_is_visible_and_persisted(artifact_dir, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(server_module, "run_pipeline", fail)
    store = server_module.RunStore(artifact_dir)
    run_id = store.submit(REQUEST)
    run = _wait_terminal(store, run_id)
    assert run["status"] == "failed" and run["result"] is None
    assert "deliberate test failure" in run["error"]
    assert run["events"][-1]["stage"] == "failed"
    assert json.loads((artifact_dir / run_id / "run.json").read_text())["status"] == "failed"


def test_run_listing_is_newest_first(http_server):
    server, store = http_server
    first = store.submit(REQUEST)
    _wait_terminal(store, first)
    second = store.submit({**REQUEST, "new_function": "Policy-as-code parsing"})
    _wait_terminal(store, second)
    status, _, body = _http(server, "GET", "/api/runs")
    assert status == 200
    assert [run["id"] for run in json.loads(body)["runs"]] == [second, first]


def test_running_and_total_run_limits(artifact_dir, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_pipeline = server_module.run_pipeline

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr(server_module, "run_pipeline", blocked)
    store = server_module.RunStore(artifact_dir, max_running=1, max_runs=1)
    run_id = store.submit(REQUEST)
    try:
        assert entered.wait(5)
        with pytest.raises(server_module.RunLimitError):
            store.submit(REQUEST)
    finally:
        release.set()
    assert _wait_terminal(store, run_id)["status"] == "completed"
    with pytest.raises(server_module.RunLimitError):
        store.submit(REQUEST)
    restarted = server_module.RunStore(artifact_dir, max_runs=1)
    with pytest.raises(server_module.RunLimitError):
        restarted.submit(REQUEST)
