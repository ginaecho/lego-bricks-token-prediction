"""Deterministic synthetic bandit tests; no provider or real cost/quality claims."""

import random
import sqlite3
import json

import pytest

from token_yield.measurement_policy import MeasurementPolicy, reward_value
from token_yield.robust import Record, RidgeLinearModel


def policy(tmp_path, **kwargs):
    return MeasurementPolicy(tmp_path / "audit.sqlite", compatibility="synthetic-fixture-v1",
                             source="mocked-test-provider", actions=["useful", "redundant"], **kwargs)


def observation(key, before=10., after=9., cost=.001, source="mocked-test-provider"):
    return {"before_mae": before, "after_mae": after, "cost_usd": cost,
            "measurements": [{"id": key, "source": source, "status": "measured",
                              "rated_usd": cost, "usage": {"input_tokens": 20, "output_tokens": 10}}],
            "predictor_before": "synthetic-before", "predictor_after": "synthetic-after",
            "calibration_ids": ["calibration-only"]}


def test_reward_changes_selection_and_survives_restart(tmp_path):
    p = policy(tmp_path)
    first = p.choose("one", {"phase": "synthetic"}, rng=random.Random(1))
    assert first["action"] == "useful" and first["propensity"] == .5
    assert p.choose("one", {"phase": "synthetic"}) == first
    with pytest.raises(RuntimeError, match="unsettled"):
        p.choose("premature", {})
    good = observation("call-1", after=0.)
    p.feedback("one", **good)
    second = p.choose("two", {})
    assert second["action"] == "redundant" and second["propensity"] == 1
    p.feedback("two", **observation("call-2", after=20.))
    restarted = policy(tmp_path)
    assert restarted.snapshot() == p.snapshot()
    third = restarted.choose("three", {}, rng=random.Random(1))
    assert third["action"] == "useful"
    assert third["probabilities"] == {"useful": .9, "redundant": .1}
    assert third["baseline"]["probabilities"] == {"useful": .5, "redundant": .5}
    assert restarted.feedback("one", **good)["reward"] == 1.
    assert restarted.snapshot()["updates"] == 2
    with pytest.raises(ValueError, match="conflicting duplicate"):
        restarted.feedback("one", **observation("call-1", after=1.))
    with pytest.raises(ValueError, match="already credited"):
        restarted.feedback("three", **good)
    restarted.reject("three", "Cancelled before dispatch")
    assert restarted.snapshot()["updates"] == 2
    assert not restarted.snapshot()["pending"]
    with sqlite3.connect(p.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("DELETE FROM events")


@pytest.mark.parametrize("before,after,cost", [
    (float("nan"), 1, .1), (1, float("inf"), .1), (-1, 0, .1),
    (1, -1, .1), (True, 1, .1), (1, 0, 0), (1, 0, None), (1, 0, -.1),
])
def test_invalid_observation_never_creates_reward(tmp_path, before, after, cost):
    p = policy(tmp_path)
    p.choose("one", {})
    with pytest.raises(ValueError):
        p.feedback("one", **observation("call", before, after, cost))
    assert p.snapshot()["updates"] == 0
    assert p.snapshot()["pending"] == ["one"]


@pytest.mark.parametrize("change", ["provenance", "cost", "usage", "leak", "duplicate"])
def test_reject_incomplete_telemetry_and_leakage(tmp_path, change):
    p = policy(tmp_path)
    p.choose("one", {})
    data = observation("call")
    if change == "provenance":
        data["measurements"][0]["source"] = "measured-foundry"
    elif change == "cost":
        data["cost_usd"] *= 2
    elif change == "usage":
        del data["measurements"][0]["usage"]["output_tokens"]
    elif change == "leak":
        data["calibration_ids"] = ["call"]
    else:
        data["measurements"] *= 2
    with pytest.raises(ValueError):
        p.feedback("one", **data)
    assert p.snapshot()["updates"] == 0


def test_compatibility_and_real_provenance_partition(tmp_path):
    p = policy(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        MeasurementPolicy(p.path, compatibility="synthetic-fixture-v1",
                          source="measured-foundry", actions=["useful", "redundant"])
    with pytest.raises(ValueError, match="mismatch"):
        MeasurementPolicy(p.path, compatibility="another-source-version",
                          source="mocked-test-provider", actions=["useful", "redundant"])
    real = MeasurementPolicy(tmp_path / "real.sqlite", compatibility="separate-real-partition",
                             source="measured-foundry", actions=["one"])
    real.choose("real-protocol-test", {})
    data = observation("synthetic-protocol-fixture", source="measured-foundry")
    real.feedback("real-protocol-test", **data)
    assert real.snapshot()["cost_basis"] == "rated-provider-usage"
    assert p.snapshot()["updates"] == 0


def test_signed_and_zero_baseline_rewards():
    assert reward_value(10, 12, .001) == -.2
    assert reward_value(0, 0, .001) == 0
    assert reward_value(0, 1, .001) == -1
    assert reward_value(10, 0, .001) > reward_value(10, 0, .01)


def test_deterministic_synthetic_ridge_baseline_comparison(tmp_path):
    """Same seed set, calibration point, calls and cost; final test never rewards."""
    def experiment(adaptive):
        rng = random.Random(6)
        p = policy(tmp_path) if adaptive else None
        train = [Record((0.,), 0., "seed")]
        errors, selected = [], []
        for step in range(24):
            before = abs(RidgeLinearModel.fit(train, alpha=1.).predict((1.,)) - 10.)
            action = (p.choose(str(step), {"train_count": len(train)}, rng=rng)["action"]
                      if p else rng.choice(["useful", "redundant"]))
            x = 1. if action == "useful" else 0.
            train.append(Record((x,), 10. * x, f"train-{step}"))
            model = RidgeLinearModel.fit(train, alpha=1.)
            after = abs(model.predict((1.,)) - 10.)
            if p:
                p.feedback(str(step), **observation(f"call-{step}", before, after))
            errors.append(after)
            selected.append(action)
        final_test_error = abs(model.predict((2.,)) - 20.)
        return {"calibration_error_area": sum(errors), "final_test_error": final_test_error,
                "selections": selected, "virtual_usd": 24 * .001}
    bandit, uniform = experiment(True), experiment(False)
    assert bandit["virtual_usd"] == uniform["virtual_usd"]
    assert bandit["selections"].count("useful") > uniform["selections"].count("useful")
    # Cold-start exploration can lose on cumulative error despite learned choices.
    # Retain this counterexample instead of tuning the fixture to claim a win.
    assert bandit["calibration_error_area"] > uniform["calibration_error_area"]
    print(json.dumps({"synthetic_bandit": bandit, "synthetic_uniform": uniform}))
    assert policy(tmp_path).snapshot()["updates"] == 24
