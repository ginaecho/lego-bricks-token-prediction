"""Delivered-scope feedback and prospective reward-loop regression tests."""

import copy
import hashlib
import json
import random
import threading
import urllib.error
import urllib.request

import pytest

from examples.customer_outcomes_server import make_server
from token_yield.customer_outcomes import OutcomeStore
from token_yield.delivery_feedback import DeliveryFeedback, OPTIONAL_NUMBERS, action_key


def receipt_id(index=0, split="train"):
    for offset in range(100):
        value = f"delivery-{index}-{offset}"
        validation = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 5 == 0
        if validation == (split == "validation"):
            return value
    raise AssertionError("could not find split fixture")


def payload(index=0, split="train", variant="basic", decision=None):
    return {
        "handoff": {
            "schema": "token-yield-delivery-handoff-v1", "receipt_id": receipt_id(index, split),
            "title": "Existing purchased project", "description": "Original approved description",
            "flow": "manual", "quote_basis": "illustrative forecast",
            "functions": [{"id": "research:basic", "feature_id": "research", "variant_id": variant,
                           "name": "Research", "model_id": "small", "model_name": "Small",
                           "estimated_tokens": 2500, "runs": 100, "decision_id": decision}],
            "staffing": [{"role": "Consultant", "hours": 24}], "estimated_project_tokens": 250000,
        },
        "source": "synthetic", "respondent": "Example client", "delivered": True, "consent": True,
        "sow_quality": 4, "staffing_fit": 5, "observation_days": 30,
        "feedback": [{"function_id": "research:basic", "satisfaction": 5, "kept": "all",
                      "usefulness": "yes", "comments": "Helpful, but slow.",
                      **dict.fromkeys(OPTIONAL_NUMBERS)}],
    }


@pytest.fixture
def service(tmp_path):
    return DeliveryFeedback(OutcomeStore(tmp_path / "PROTOTYPE-delivery.sqlite3"))


def test_submission_trains_once_preserves_scope_and_unknown_actuals(service):
    request = payload()
    result = service.submit(request)
    signal = result["signals"][0]
    assert result["saved"] and result["learning"]["reward_updates"] == 1
    assert signal["experience_reward"] == 1
    assert signal["reported_roi"] is None
    assert signal["reported_aes"] is None
    assert signal["actual_tokens"] is None
    assert signal["financial_status"] == "customer_reported_not_verified"
    assert service.submit(request) == result
    assert service.get(request["handoff"]["receipt_id"])["handoff"] == request["handoff"]
    assert service.learning("synthetic")["active"]["version"] == 0
    assert service.learning("real")["candidate"] is None
    restarted = DeliveryFeedback(OutcomeStore(service.store.path))
    assert restarted.listing()["deliveries"][0]["function_count"] == 1


def test_revision_replaces_reward_without_duplicate_credit(service):
    request = payload()
    first = service.submit(request)
    request["feedback"][0].update(satisfaction=1, kept="none", usefulness="no")
    second = service.submit(request)
    assert second["learning"]["candidate_version"] > first["learning"]["candidate_version"]
    model = service.learning("synthetic")["candidate"]
    assert model["reward_updates"] == 1
    assert model["policy"]["research"][action_key(request["handoff"]["functions"][0])] == {"n": 1, "q": -1}
    assert model["training_snapshots"][0]["feedback"][0]["satisfaction"] == 1


@pytest.mark.parametrize("field,value", [
    ("delivered", False), ("delivered", "true"), ("consent", False),
    ("source", "invented"), ("sow_quality", 0), ("staffing_fit", 6),
    ("observation_days", 0), ("respondent", ""),
])
def test_requires_delivery_consent_and_valid_metadata(service, field, value):
    request = payload()
    request[field] = value
    with pytest.raises((ValueError, TypeError)):
        service.submit(request)
    assert service.listing() == {"deliveries": []}


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "object", "id"])
def test_feedback_cannot_change_purchased_functions(service, case):
    request = payload()
    if case == "missing":
        request["feedback"] = []
    elif case == "extra":
        request["feedback"].append({**request["feedback"][0], "function_id": "not-purchased"})
    elif case == "duplicate":
        request["feedback"] *= 2
    elif case == "object":
        request["feedback"] = ["invalid"]
    else:
        request["feedback"][0]["function_id"] = []
    with pytest.raises((ValueError, TypeError)):
        service.submit(request)


def test_unknown_reward_stays_pending_but_ratings_train(service):
    request = payload()
    request["feedback"][0]["kept"] = "unknown"
    result = service.submit(request)
    assert result["signals"][0]["experience_reward"] is None
    assert result["learning"]["reward_updates"] == 0
    assert service.learning("synthetic")["candidate"]["policy"] == {}


def test_reported_finances_aes_and_zero_are_not_verified(service):
    request = payload()
    request["feedback"][0].update(benefit_usd=300, cost_usd=100, revenue_usd=80,
                                  provider_cost_usd=100, autonomy_pct=80, kept_pct=50,
                                  useful_pct=50, actual_tokens=4000)
    signal = service.submit(request)["signals"][0]
    assert signal["reported_roi"] == 2
    assert signal["reported_profit_usd"] == -20
    assert signal["reported_aes"] == 20
    request["feedback"][0].update(autonomy_pct=0, kept_pct=None, useful_pct=None, cost_usd=0)
    signal = service.submit(request)["signals"][0]
    assert signal["reported_aes"] == 0
    assert signal["reported_roi"] is None


@pytest.mark.parametrize("field,value", [
    ("satisfaction", 6), ("satisfaction", True), ("kept", "most"),
    ("usefulness", ""), ("benefit_usd", -1), ("autonomy_pct", 101),
    ("cost_usd", float("nan")), ("comments", {}),
])
def test_invalid_signal_rejected(service, field, value):
    request = payload()
    request["feedback"][0][field] = value
    with pytest.raises((ValueError, TypeError)):
        service.submit(request)


@pytest.mark.parametrize("change", ["scope", "source", "window"])
def test_saved_receipt_freezes_scope_source_and_window(service, change):
    request = payload()
    service.submit(request)
    if change == "scope":
        request["handoff"]["functions"][0]["model_id"] = "another"
    elif change == "source":
        request["source"] = "real"
    else:
        request["observation_days"] = 60
    with pytest.raises(ValueError, match="frozen"):
        service.submit(request)


def test_suggestion_cannot_be_reassigned_or_reused(service):
    choice = service.suggest("synthetic", "research", [{"variant_id": "basic", "model_id": "small"}])
    request = payload(decision=choice["id"])
    incorrect = copy.deepcopy(request)
    incorrect["handoff"]["functions"][0]["model_id"] = "other"
    with pytest.raises(ValueError, match="original"):
        service.submit(incorrect)
    duplicate = copy.deepcopy(request)
    duplicate["handoff"]["functions"].append({**duplicate["handoff"]["functions"][0], "id": "second"})
    duplicate["feedback"].append({**duplicate["feedback"][0], "function_id": "second"})
    with pytest.raises(ValueError, match="already credited"):
        service.submit(duplicate)
    service.submit(request)
    with pytest.raises(ValueError, match="already credited"):
        service.submit(payload(1, decision=choice["id"]))


def test_protected_receipts_never_train_and_historical_choices_cannot_validate(service):
    service.submit(payload())
    candidate = service.learning("synthetic")["candidate"]
    for index in range(22):
        service.submit(payload(index, "validation"))
    assert service.learning("synthetic")["candidate"] == candidate
    with pytest.raises(ValueError, match="prospectively logged"):
        service.promote("synthetic", "Reviewer")


def test_end_to_end_reward_training_human_promotion_and_next_decision(service):
    options = [{"variant_id": v, "model_id": "small"} for v in ("basic", "deep")]
    rng = random.Random(812)
    before = service.suggest("synthetic", "research", options, rng=rng)
    assert set(before["probabilities"].values()) == {.5}
    for index in range(32):
        variant = "basic" if index % 2 == 0 else "deep"
        request = payload(index, variant=variant)
        if variant == "deep":
            request["feedback"][0].update(satisfaction=1, kept="none", usefulness="no")
        service.submit(request)
    assert service.predict("synthetic", payload()["handoff"], 30)["policy_version"] == 0
    for index in range(80):
        decision = service.suggest("synthetic", "research", options, rng=rng)
        request = payload(index, "validation", variant=decision["option"]["variant_id"], decision=decision["id"])
        if decision["option"]["variant_id"] == "deep":
            request["feedback"][0].update(satisfaction=1, kept="none", usefulness="no")
        service.submit(request)
    model = service.promote("synthetic", "Named human reviewer")
    assert model["evaluation"]["eligible"]
    assert model["evaluation"]["projects"] == 80
    assert model["evaluation"]["lower_bound"] > 0
    assert len(model["training_ids"]) == 32
    assert not set(model["training_ids"]) & set(model["evaluation_ids"])
    next_choice = service.suggest("synthetic", "research", options, rng=rng)
    assert next_choice["policy_version"] == model["version"]
    assert next_choice["probabilities"][action_key(options[0])] == pytest.approx(.9)
    estimates = service.predict("synthetic", payload()["handoff"], 30)["functions"][0]["targets"]
    assert estimates["satisfaction"]["estimate"] == pytest.approx(5)
    assert estimates["reported_roi"]["estimate"] is None
    outside = payload()["handoff"]
    outside["functions"][0]["estimated_tokens"] = 5000
    assert service.predict("synthetic", outside, 30)["functions"][0]["targets"]["satisfaction"]["estimate"] is None
    request = payload()
    request["feedback"][0]["comments"] = "Changed after policy approval"
    with pytest.raises(ValueError, match="locked"):
        service.submit(request)
    assert service.learning("real")["active"]["version"] == 0
    assert service.promote("synthetic", "Reviewer")["version"] == model["version"]
    service.submit(payload(1000))
    with pytest.raises(ValueError, match="fresh"):
        service.promote("synthetic", "Reviewer")


def test_simple_page_and_feedback_api(service):
    server = make_server(service.store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base) as response:
            page = response.read().decode()
            assert 'id="feedback-form"' in page
            assert 'id="project-form"' not in page
        with urllib.request.urlopen(base + "/admin") as response:
            assert b"Customer Outcomes" in response.read()
        request = urllib.request.Request(base + "/api/deliveries/feedback", json.dumps(payload()).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        assert result["saved"]
        with urllib.request.urlopen(base + "/api/deliveries?receipt=" + result["receipt_id"]) as response:
            assert json.load(response)["handoff"] == payload()["handoff"]
        request.add_header("Origin", "https://untrusted.example")
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request)
        assert failure.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_failed_evaluation_cannot_promote_or_reuse_its_holdout(service):
    options = [{"variant_id": v, "model_id": "small"} for v in ("basic", "deep")]
    for index in range(20):
        service.submit(payload(index))
    validation = []
    for index in range(40):
        choice = service.suggest("synthetic", "research", options, rng=random.Random(index % 2))
        request = payload(index, "validation", choice["option"]["variant_id"], choice["id"])
        validation.append(request)
        service.submit(request)
    model = service.promote("synthetic", "Reviewer")
    assert model["evaluation"]["gain"] == pytest.approx(0)
    assert not model["evaluation"]["eligible"]
    assert "promoted_by" not in model
    assert service.learning("synthetic")["active"]["version"] == 0
    assert service.promote("synthetic", "Reviewer")["evaluation"] == model["evaluation"]
    validation[0]["feedback"][0]["satisfaction"] = 1
    with pytest.raises(ValueError, match="locked"):
        service.submit(validation[0])
    service.submit(payload(999))
    with pytest.raises(ValueError, match="fresh"):
        service.promote("synthetic", "Reviewer")
