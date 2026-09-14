"""Tests for the studio.

The studio is a demonstration, and a demonstration that quietly lies is worse than
no demonstration. So these tests are mostly about honesty properties rather than
pixels: that the evidence class travels with the payload, that the null arm cannot
be made to look like the brick arm, that unmeasured gates still cap the verdict,
and that a second click cannot interleave two runs over one state object.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from assay.studio import demo as demo_mod
from assay.studio.demo import ARMS, STAGES, DemoState, run_demo, start
from assay.studio.server import APP_HTML, serve


@pytest.fixture(scope="module")
def finished(tmp_path_factory) -> DemoState:
    """One completed run, shared. It takes a couple of seconds; do it once."""
    state = DemoState()
    run_demo(state, tmp_path_factory.mktemp("studio"))
    return state


# -- the run itself ------------------------------------------------------------------


def test_demo_completes_both_arms(finished: DemoState) -> None:
    snap = finished.snapshot()
    assert snap["status"] == "done", snap["error"]
    assert snap["error"] is None
    assert snap["overall_progress"] == 1.0
    for arm in ARMS:
        stages = snap["arms"][arm]["stages"]
        assert len(stages) == len(STAGES)
        assert all(s["status"] == "done" for s in stages), (arm, stages)


def test_every_payload_carries_its_evidence_class(finished: DemoState) -> None:
    snap = finished.snapshot()
    assert snap["evidence_class"] == "pipeline_only"
    for arm in ARMS:
        assert snap["arms"][arm]["result"]["verdict"]["evidence_class"] == "pipeline_only"
        assert snap["arms"][arm]["result"]["example"]["quote"]["evidence_class"] == "pipeline_only"


def test_the_two_arms_separate(finished: DemoState) -> None:
    """The whole point. Signal present -> found; signal absent -> refused."""
    snap = finished.snapshot()
    brick = snap["arms"]["brick"]["result"]["ladder"]
    null = snap["arms"]["null"]["result"]["ladder"]
    assert brick["chose_a_decoder"] is True, "failed to find a signal that is there"
    assert null["chose_a_decoder"] is False, "invented a signal that is not there"


def test_the_null_arm_withholds_prices_it_did_not_earn(finished: DemoState) -> None:
    fit = finished.snapshot()["arms"]["null"]["result"]["fit"]
    assert fit["show_per_brick"] is False
    assert fit["marginals"] == {}


def test_the_null_arm_quote_ignores_which_bricks_were_ordered(finished: DemoState) -> None:
    """A size-only model must be visibly indifferent to the order, not subtly so."""
    from assay.studio.server import _quote

    a = _quote(finished, "null", {"Extract": 1}, 6000)
    b = _quote(finished, "null", {"Retrieve": 9, "Reconcile": 4}, 6000)
    assert a["predicted_units"] == pytest.approx(b["predicted_units"])

    c = _quote(finished, "brick", {"Extract": 1}, 6000)
    d = _quote(finished, "brick", {"Retrieve": 9, "Reconcile": 4}, 6000)
    assert c["predicted_units"] != pytest.approx(d["predicted_units"])


def test_unmeasured_gates_cap_the_verdict(finished: DemoState) -> None:
    """A simulator cannot observe the encoder or the human raters, and must not pretend to."""
    snap = finished.snapshot()
    for arm in ARMS:
        obs = snap["arms"][arm]["result"]["observations"]
        assert obs["M7_encoder_exact_pct"] is None
        assert obs["M13_alpha_macro"] is None
        assert snap["arms"][arm]["result"]["verdict"]["outcome"] != "feasible"


def test_invoice_lines_reconcile_to_the_measured_total(finished: DemoState) -> None:
    for arm in ARMS:
        inv = finished.snapshot()["arms"][arm]["result"]["example"]["invoice"]
        assert inv["reconciles"] is True
        assert sum(line["units"] for line in inv["lines"]) == pytest.approx(inv["sum_check"])


def test_snapshot_is_json_serialisable(finished: DemoState) -> None:
    """The fitted models must stay out of the wire format."""
    payload = json.dumps(finished.snapshot())
    assert "predicted_units" in payload


def test_blind_split_is_sealed_before_measurement(finished: DemoState) -> None:
    corpus = finished.snapshot()["corpus"]
    assert corpus["seal_sha256"]
    assert set(corpus["blind"]).isdisjoint(corpus["fit"])
    assert corpus["span_x"] > 5, "sizes must span enough to be told apart from brick count"


# -- concurrency ---------------------------------------------------------------------


def test_a_second_start_cannot_race_a_running_one(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def slow(state: DemoState, work_dir=None) -> None:
        release.wait(timeout=5)
        state.set_top(status="done")

    monkeypatch.setattr(demo_mod, "run_demo", slow)
    state = DemoState()
    first = start(state)
    assert first is not None
    assert start(state) is None, "a second run would interleave two writers"
    release.set()
    first.join(timeout=5)


def test_claim_resets_the_previous_run(finished: DemoState) -> None:
    state = DemoState()
    state.say("brick", "stale")
    assert state.claim() is True
    assert state.snapshot()["log"] == []
    state.set_top(status="idle")


# -- the server ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(finished: DemoState):
    httpd = serve(host="127.0.0.1", port=0, open_browser=False, autorun=False, state=finished)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def test_index_is_served_and_self_contained(server: str) -> None:
    status, body = _get(server + "/")
    assert status == 200
    text = body.decode("utf-8")
    assert "<!DOCTYPE html>" in text
    # No external fetches: the project's claim is that it runs on a fresh clone.
    for marker in ("https://", "http://cdn", "<script src=", "<link rel=\"stylesheet\""):
        assert marker not in text, f"studio must not reach the network ({marker})"


def test_index_uses_the_required_theme(server: str) -> None:
    text = APP_HTML.read_text(encoding="utf-8")
    assert "scoutTheme" in text
    assert "--cp-accent: #b11f4b;" in text
    assert '"Segoe UI", Aptos, Calibri' in text


def test_state_endpoint_matches_the_snapshot(server: str) -> None:
    status, body = _get(server + "/api/state")
    assert status == 200
    payload = json.loads(body)
    assert payload["status"] == "done"
    assert payload["evidence_class"] == "pipeline_only"


def test_quote_endpoint_prices_a_typed_request(server: str) -> None:
    status, body = _get(server + "/api/quote?arm=brick&bytes=5000&u.Extract=2")
    assert status == 200
    q = json.loads(body)
    assert q["predicted_units"] > 0
    assert q["interval_lo"] < q["predicted_units"] < q["interval_hi"]
    assert q["evidence_class"] == "pipeline_only"


def test_quote_endpoint_rejects_nonsense(server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server + "/api/quote?arm=brick&bytes=lots")
    assert excinfo.value.code == 400


def test_unknown_paths_are_not_found(server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server + "/../etc/passwd")
    assert excinfo.value.code == 404


def test_quote_for_an_unfitted_arm_says_so() -> None:
    from assay.studio.server import _quote

    assert "error" in _quote(DemoState(), "brick", {"Extract": 1}, 1000)


def test_run_endpoint_answers_and_guards_a_double_click(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Build button posts here. It regressed once by returning a Thread."""
    release = threading.Event()
    monkeypatch.setattr(demo_mod, "run_demo",
                        lambda state, work_dir=None: release.wait(timeout=5))

    state = DemoState()
    httpd = serve(host="127.0.0.1", port=0, open_browser=False, autorun=False, state=state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/api/run", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=10) as r:
            first = json.loads(r.read())
        assert first == {"started": True, "status": "running"}

        req = urllib.request.Request(base + "/api/run", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=10) as r:
            second = json.loads(r.read())
        assert second["started"] is False
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()
