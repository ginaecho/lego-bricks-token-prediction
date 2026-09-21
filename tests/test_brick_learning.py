"""Ranking evidence, constrained choices, and the visible learning walkthrough."""

import copy
import hashlib
import json

import pytest

from examples.customer_outcomes_server import BRICK_CATALOG
from token_yield.brick_learning import BrickLearning, validate_catalog
from token_yield.customer_outcomes import OutcomeStore
from token_yield.delivery_feedback import DeliveryFeedback, OPTIONAL_NUMBERS, action_key


@pytest.fixture
def learning(tmp_path):
    return BrickLearning(DeliveryFeedback(OutcomeStore(tmp_path / "PROTOTYPE-rankings.sqlite3")))


@pytest.fixture
def catalog():
    return json.loads(BRICK_CATALOG.read_text(encoding="utf-8"))["catalog"]


def submit(learning, index, *, split="train", satisfaction=5):
    receipt = f"ranking-{index}"
    while (int(hashlib.sha256(receipt.encode()).hexdigest()[:8], 16) % 5 == 0) != (split == "validation"):
        receipt += "x"
    learning.deliveries.submit({
        "source": "real", "respondent": "Customer", "delivered": True, "consent": True,
        "sow_quality": None, "staffing_fit": None, "observation_days": 30,
        "handoff": {"schema": "token-yield-delivery-handoff-v1", "receipt_id": receipt,
                    "title": "Delivered project", "description": "", "flow": "manual",
                    "quote_basis": "forecast", "staffing": [], "estimated_project_tokens": None,
                    "functions": [{"id": "research:web", "feature_id": "research", "name": "Web research",
                                   "variant_id": "web", "model_id": "gpt", "model_name": "GPT",
                                   "estimated_tokens": None, "runs": 10, "decision_id": None}]},
        "feedback": [{"function_id": "research:web", "satisfaction": satisfaction,
                      "kept": "all", "usefulness": "yes", "comments": "",
                      **dict.fromkeys(OPTIONAL_NUMBERS)}],
    })


def test_cold_start_and_minimum_independent_evidence(learning, catalog):
    assert all(r["rank"] is None and r["score"] is None for r in learning.rankings("real", "candidate", catalog)["rows"])
    for index in range(2):
        submit(learning, index)
    assert all(r["rank"] is None for r in learning.rankings("real", "candidate", catalog)["rows"])
    submit(learning, 2)
    report = learning.rankings("real", "candidate", catalog)
    best = report["rows"][0]
    assert best["feature_id"] == "research"
    assert best["score"] == 100
    assert best["rank"] == 1 and best["project_count"] == 3
    assert best["options"][0]["variant_id"] == "web"
    assert best["options"][0]["model_id"] == "gpt"
    assert all(r["rank"] is None for r in report["rows"][1:])
    assert all(r["score"] is None for r in learning.rankings("synthetic", "candidate", catalog)["rows"])
    assert all(r["score"] is None for r in learning.rankings("real", "active", catalog)["rows"])
    submit(learning, 30, split="validation", satisfaction=1)
    assert learning.rankings("real", "candidate", catalog)["rows"] == report["rows"]
    submit(learning, 2, satisfaction=1)
    revised = learning.rankings("real", "candidate", catalog)["rows"][0]
    assert revised["project_count"] == 3
    assert revised["score"] < best["score"]


@pytest.mark.parametrize("corruption", ["duplicate", "options", "option", "name"])
def test_catalog_validation(catalog, corruption):
    broken = copy.deepcopy(catalog)
    if corruption == "duplicate":
        broken.append(broken[0])
    elif corruption == "options":
        broken[0]["options"] = []
    elif corruption == "option":
        broken[0]["options"][0] = None
    else:
        broken[0]["name"] = 33
    with pytest.raises((ValueError, TypeError)):
        validate_catalog(broken)


def test_custom_needs_are_preserved_and_only_matching_actions_are_logged(learning, catalog):
    matches = [{"feature_id": "research", "variant_ids": ["web"], "reason": "Cited external sources"},
               {"feature_id": "documents", "variant_ids": ["extract"], "reason": "Extract structured fields"}]
    report = learning.recommend("real", catalog, matches)
    choices = report["recommendations"]
    assert report["policy_version"] == 0
    assert {(r["feature_id"], r["variant_id"]) for r in choices} == {("research", "web"), ("documents", "extract")}
    assert all(r["score"] is None and r["decision_id"] for r in choices)
    with learning.deliveries.store.connection() as db:
        for recommendation in choices:
            decision = json.loads(db.execute("SELECT body FROM delivery_choices WHERE id=?",
                                            (recommendation["decision_id"],)).fetchone()[0])
            assert decision["action"] == action_key(recommendation)
            assert len(decision["probabilities"]) == 4
            assert all(json.loads(key)[0] == recommendation["variant_id"] for key in decision["probabilities"])


@pytest.mark.parametrize("matches", [
    [],
    [{"feature_id": "unrelated", "variant_ids": ["web"], "reason": "No match"}],
    [{"feature_id": "research", "variant_ids": ["invented"], "reason": "No such option"}],
    [{"feature_id": "research", "variant_ids": ["web"], "reason": "A"},
     {"feature_id": "research", "variant_ids": ["web"], "reason": "Duplicate"}],
])
def test_invalid_custom_requests_do_not_log_decisions(learning, catalog, matches):
    with pytest.raises(ValueError):
        learning.recommend("real", catalog, matches)
    with learning.deliveries.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM delivery_choices").fetchone()[0] == 0


def test_demo_drives_actual_feedback_training_then_human_approved_rankings(learning, catalog):
    seeded = learning.seed_demo(catalog)
    assert seeded["source"] == "synthetic" and seeded["seeded"] >= 160
    assert learning.seed_demo(catalog)["seeded"] == 0
    assert learning.deliveries.listing()["deliveries"] == []
    assert len(learning.deliveries.listing(include_examples=True)["deliveries"]) == seeded["seeded"]
    candidate = learning.rankings("synthetic", "candidate", catalog)
    assert candidate["active_version"] == 0
    assert all(r["score"] is not None for r in candidate["rows"])
    assert candidate["validation_projects"] == 140
    assert all(r["score"] is None for r in learning.rankings("real", "candidate", catalog)["rows"])
    assert all(r["score"] is None for r in learning.rankings("synthetic", "active", catalog)["rows"])
    model = learning.deliveries.promote("synthetic", "Demo reviewer")
    assert model["evaluation"]["eligible"], model["evaluation"]
    active = learning.rankings("synthetic", "active", catalog)
    assert active["rows"] == candidate["rows"]
    assert active["active_version"] == candidate["model_version"]
    matches = [{"feature_id": "research", "variant_ids": ["web"], "reason": "Needs cited web sources"}]
    report = learning.recommend("synthetic", catalog, matches)
    assert report["policy_version"] == model["version"]
    with learning.deliveries.store.connection() as db:
        decision = json.loads(db.execute("SELECT body FROM delivery_choices WHERE id=?",
                                        (report["recommendations"][0]["decision_id"],)).fetchone()[0])
    assert decision["probabilities"][action_key({"variant_id": "web", "model_id": "gpt"})] == pytest.approx(.85)
