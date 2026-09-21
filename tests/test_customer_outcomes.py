"""Outcome arithmetic, evidence boundaries and actual reward-driven learning."""

import copy
import json
import random
import threading
import urllib.error
import urllib.request

import pytest

from examples.customer_outcomes_demo import seed_demo
from examples.customer_outcomes_server import make_server
from token_yield.customer_outcomes import (
    OutcomeStore, outcome_signals, probabilities, validate_feedback,
)


@pytest.fixture
def function():
    return {
        "id": "sow", "name": "Statement of work", "kind": "sow", "complexity": "low",
        "assigned_units": 100, "estimated_tokens": 10000, "customer_budget_usd": 3000,
        "provider_budget_usd": 2000, "reference_benefit_usd": 5000, "observation_days": 30,
    }


@pytest.fixture
def feedback():
    return {
        "decision_id": "pending", "respondent": "Client", "completed_units": 80,
        "kept_units": 60, "useful_units": 30, "actual_tokens": 10000,
        "satisfaction": 4, "sow_quality": 3, "staffing_fit": 5, "impact": 4,
        "customer_benefit_usd": 6000, "customer_cost_usd": 2500,
        "provider_revenue_usd": 2000, "provider_cost_usd": 1500,
        "financial_evidence": "verified", "window_closed": True,
        "safety_ok": True, "budget_ok": True, "evidence": "Customer review and cost ledger",
        "comments": "SOW scope was good but staffing needed rework.",
    }


@pytest.fixture
def store(tmp_path):
    return OutcomeStore(tmp_path / "outcomes.sqlite3")


def episode(store, function, feedback, *, source="real", split="train", approved=True):
    project = store.create_project({
        "title": "Client project", "description": "Prepare scope and team.",
        "source": source, "functions": [function]})
    decision = store.recommend(project["id"], function["id"], split, rng=random.Random(1))
    store.approve(decision["id"], "PM", True)
    result = store.feedback({**feedback, "decision_id": decision["id"]})
    if approved:
        result = store.review(decision["id"], "Reviewer", True, "Evidence checked")
    return project, result


def test_aes_exact_funnel_and_separate_profit(function, feedback):
    signals = outcome_signals(function, validate_feedback(feedback, function))
    assert signals["completion_rate"] == .8
    assert signals["retention_rate"] == .75
    assert signals["usefulness_rate"] == .5
    assert signals["aes"] == 30
    assert signals["token_yield_per_1000"] == 3
    assert signals["customer_roi"] == 1.4
    assert signals["provider_profit_usd"] == 500
    assert signals["net_benefit_usd"] == 3500
    assert signals["reward"] == pytest.approx(.12 + .16 + .06 + .1 + .14)
    assert signals["themes"] == ["SOW / scope", "Staffing / expertise", "Quality / rework"]


@pytest.mark.parametrize("field", ["completed_units", "useful_units", "satisfaction",
                                   "customer_benefit_usd", "actual_tokens"])
def test_missing_evidence_never_becomes_success(function, feedback, field):
    feedback[field] = None
    if field == "completed_units":
        feedback["kept_units"] = feedback["useful_units"] = None
    signal = outcome_signals(function, validate_feedback(feedback, function))
    assert signal["reward"] is None
    assert signal["reward_pending"]


@pytest.mark.parametrize("field,value", [
    ("completed_units", 101), ("kept_units", 81), ("useful_units", 61),
    ("actual_tokens", True), ("actual_tokens", -1), ("satisfaction", 6),
    ("customer_cost_usd", float("nan")), ("customer_cost_usd", float("inf")),
    ("window_closed", "true"), ("financial_evidence", "fantasy"),
])
def test_feedback_rejects_bad_values(function, feedback, field, value):
    feedback[field] = value
    with pytest.raises((ValueError, TypeError)):
        validate_feedback(feedback, function)


def test_known_zero_and_unknown_cost_do_not_fabricate_ratios(function, feedback):
    feedback.update(completed_units=0, kept_units=None, useful_units=None,
                    customer_cost_usd=0)
    signal = outcome_signals(function, validate_feedback(feedback, function))
    assert signal["aes"] == 0
    assert signal["retention_rate"] is None
    assert signal["usefulness_rate"] is None
    assert signal["customer_roi"] is None
    assert signal["token_yield_per_1000"] == 0


def test_negative_roi_profit_and_reward_preserved(function, feedback):
    feedback.update(completed_units=0, kept_units=0, useful_units=0,
                    satisfaction=1, sow_quality=1, staffing_fit=1,
                    customer_benefit_usd=0, provider_revenue_usd=0)
    signal = outcome_signals(function, validate_feedback(feedback, function))
    assert signal["customer_roi"] == -1
    assert signal["provider_profit_usd"] == -1500
    assert signal["reward"] < 0


@pytest.mark.parametrize("change", [
    {"window_closed": False}, {"financial_evidence": "reported"},
    {"financial_evidence": "unknown"}, {"budget_ok": False}, {"safety_ok": False},
    {"customer_cost_usd": 4000}, {"provider_cost_usd": 2500}, {"actual_tokens": 0},
])
def test_pending_and_failed_gates_block_reward(function, feedback, change):
    feedback.update(change)
    assert outcome_signals(function, validate_feedback(feedback, function))["reward"] is None


def test_feedback_review_approval_and_revisions(store, function, feedback):
    project = store.create_project({
        "title": "Example", "description": "Example scope",
        "source": "real", "functions": [function]})
    decision = store.recommend(project["id"], "sow")
    payload = {**feedback, "decision_id": decision["id"]}
    with pytest.raises(ValueError, match="approved"):
        store.feedback(payload)
    store.approve(decision["id"], "PM", True)
    store.feedback(payload)
    with pytest.raises(ValueError, match="reviewed"):
        store.train("real")
    store.review(decision["id"], "Reviewer", True, "Checked")
    revised = store.feedback({**payload, "satisfaction": 2})
    assert revised["feedback"]["review"] is None
    store.review(decision["id"], "Reviewer", True, "Rechecked")
    model = store.train("real")
    assert model["training_count"] == model["policy_updates"] == 1
    assert store.train("real")["version"] == model["version"]
    with pytest.raises(ValueError, match="immutable"):
        store.feedback(payload)
    with pytest.raises(ValueError, match="immutable"):
        store.review(decision["id"], "Reviewer", False, "Changed mind")
    with pytest.raises(ValueError, match="already has"):
        store.recommend(project["id"], "sow")


def test_real_synthetic_and_validation_never_mix(store, function, feedback):
    episode(store, function, feedback, source="synthetic")
    _, validation = episode(store, function, feedback, source="real", split="validation")
    with pytest.raises(ValueError, match="no human-reviewed"):
        store.train("real")
    _, training = episode(store, function, feedback, source="real")
    trained = store.train("real")
    assert trained["trained_ids"] == [training["id"]]
    assert validation["id"] not in trained["trained_ids"]
    synthetic = store.train("synthetic")
    assert synthetic["training_count"] == 1
    assert set(synthetic["trained_ids"]).isdisjoint(trained["trained_ids"])
    assert store.state("real")["active_policy_version"] == 0
    with pytest.raises(ValueError, match="blocked"):
        store.promote("real", "Owner")
    evaluated = store.evaluate("real")
    assert not evaluated["evaluation"]["eligible"]
    assert evaluated["evaluation"]["validation_count"] == 1


def test_freeze_project_split_and_reject_cohort_mutation(store, function):
    second = {**function, "id": "staff", "kind": "staffing"}
    project = store.create_project({
        "title": "Project", "description": "Scope", "source": "real",
        "functions": [function, second]})
    store.recommend(project["id"], "sow", "validation")
    with pytest.raises(ValueError, match="frozen split"):
        store.recommend(project["id"], "staff", "train")


def test_pending_outcomes_train_ratings_but_not_reward(store, function, feedback):
    feedback.update(financial_evidence="reported", customer_benefit_usd=None)
    episode(store, function, feedback)
    model = store.train("real")
    assert model["policy_updates"] == 0
    assert model["predictor_metrics"]["satisfaction"]["count"] == 1
    assert "customer_roi" not in model["predictors"]


def test_delayed_finances_refit_once_without_losing_snapshots(store, function, feedback):
    pending = {**feedback, "financial_evidence": "reported", "customer_benefit_usd": None}
    _, decision = episode(store, function, pending)
    first = store.train("real")
    store.feedback({**feedback, "decision_id": decision["id"]})
    store.review(decision["id"], "Reviewer", True, "Delayed financial evidence checked")
    second = store.train("real")
    assert second["version"] != first["version"]
    assert second["training_count"] == second["policy_updates"] == 1
    assert first["training_rows"][0]["targets"]["customer_roi"] is None
    assert second["training_rows"][0]["targets"]["customer_roi"] == 1.4
    assert store.train("real")["policy_updates"] == 1


def test_pending_validation_can_be_completed_later(store, function, feedback):
    episode(store, function, feedback)
    store.train("real")
    _, decision = episode(store, function, {**feedback, "window_closed": False}, split="validation")
    with pytest.raises(ValueError, match="reward-complete"):
        store.evaluate("real")
    store.feedback({**feedback, "decision_id": decision["id"]})
    store.review(decision["id"], "Reviewer", True, "Window closed")
    assert store.evaluate("real")["evaluation"]["validation_count"] == 1


def test_validation_projects_cannot_be_reused_for_next_candidate(store, function, feedback):
    episode(store, function, feedback)
    episode(store, function, feedback, split="validation")
    store.train("real")
    store.evaluate("real")
    episode(store, function, feedback)
    store.train("real")
    with pytest.raises(ValueError, match="fresh"):
        store.evaluate("real")


def test_no_feedback_and_sparse_context_abstain(store, function, feedback):
    project, _ = episode(store, function, feedback)
    assert store.predict(project["id"], "sow")["model_version"] is None
    store.train("real")
    result = store.predict(project["id"], "sow")
    assert all(all(value is None for value in option["estimates"].values())
               for option in result["options"])


def test_prediction_abstains_on_extrapolation_and_missing_target_support(store, function, feedback):
    for _ in range(3):
        episode(store, function, {**feedback, "customer_benefit_usd": None})
    model = store.train("real")
    oversized = {**function, "estimated_tokens": function["estimated_tokens"] + 1}
    project = store.create_project({
        "title": "Bigger request", "description": "Outside measured range",
        "source": "real", "functions": [oversized]})
    result = store.predict(project["id"], oversized["id"])
    option = next(o for o in result["options"] if o["support"]["training_examples"])
    assert option["support"]["outside_observed_ranges"] == ["estimated_tokens"]
    assert all(value is None for value in option["estimates"].values())
    assert "customer_roi" not in model["predictors"]


def test_policy_probabilities_are_valid_and_contextual(function):
    policy = {"sow:low": {"lean": {"n": 10, "q": .8},
                          "assured": {"n": 10, "q": .1}}}
    low = probabilities(policy, function)
    assert sum(low.values()) == pytest.approx(1)
    assert min(low.values()) > 0
    assert low["lean"] == pytest.approx(1 - .2 + .2 / 3)
    high = probabilities(policy, {**function, "complexity": "high"})
    assert len(set(high.values())) == 1


def test_actual_training_evaluation_promotion_and_restart(store):
    result = seed_demo(store)
    assert result["inserted"] == 540
    assert seed_demo(store)["inserted"] == 0
    state = store.state("synthetic")
    assert state["model"] is None
    before = copy.deepcopy(state["decisions"][0]["probabilities"])
    candidate = store.train("synthetic")
    assert candidate["training_count"] == candidate["policy_updates"] == 180
    assert store.state("synthetic")["active_policy_version"] == 0
    evaluated = store.evaluate("synthetic")
    assert evaluated["evaluation"]["validation_count"] == 360
    assert evaluated["evaluation"]["eligible"], evaluated["evaluation"]
    for target in ("satisfaction", "customer_roi", "provider_profit_usd", "aes"):
        metric = evaluated["predictor_metrics"][target]
        assert metric["mae"] < metric["baseline_mae"]
    promoted = store.promote("synthetic", "Release owner")
    assert promoted["status"] == "active"
    restarted = OutcomeStore(store.path)
    assert restarted.state("synthetic")["active_policy_version"] == candidate["version"]
    first_project = state["projects"][0]
    prediction = restarted.predict(first_project["id"], first_project["functions"][0]["id"])
    best = max(prediction["options"], key=lambda option: option["probability"])
    assert best["action"] == "lean"
    assert best["probability"] > before["lean"]
    assert best["estimates"]["satisfaction"] > 4
    assert restarted.state("real")["model"] is None
    new_project = restarted.create_project({
        "title": "Next synthetic assignment", "description": "New work after promotion",
        "source": "synthetic", "functions": first_project["functions"]})
    new_decision = restarted.recommend(new_project["id"], first_project["functions"][0]["id"],
                                      rng=random.Random(1))
    assert new_decision["action"] == "lean"
    assert new_decision["policy_version"] == promoted["version"]
    assert new_decision["status"] == "proposed"
    assert restarted.train("synthetic")["version"] == candidate["version"]
    assert restarted.evaluate("synthetic")["evaluation"] == evaluated["evaluation"]


def test_http_validation_and_origin_guards(store):
    server = make_server(store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/health") as response:
            assert json.load(response)["status"] == "ok"
        with urllib.request.urlopen(base + "/api/state?source=real") as response:
            assert json.load(response)["projects"] == []
        for content, headers, status in [
            (b"{}", {"Origin": "https://not-local.example"}, 403),
            (b"{}", {"Host": "not-local.example"}, 403),
            (b'{"source":"real","source":"synthetic"}', {}, 400),
            (b'{"source":NaN}', {}, 400),
            (b"[]", {}, 400),
            (b"{}", {"Content-Type": "text/plain"}, 415),
        ]:
            request = urllib.request.Request(
                base + "/api/train", content,
                headers={"Content-Type": "application/json", **headers})
            with pytest.raises(urllib.error.HTTPError) as failure:
                urllib.request.urlopen(request)
            assert failure.value.code == status
            assert json.load(failure.value)["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
