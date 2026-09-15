"""Tests for the paid-experiment hard budget."""

import pytest

from token_yield.budget import HardBudget, SafetyRateCard


def test_conservative_rate_prices_reasoning_as_output():
    budget = HardBudget(
        cap_usd=10,
        stop_usd=9,
        rates=SafetyRateCard(1, 10, "test"),
    )
    budget.reserve("a", 1_000_000, 100_000)
    settled = budget.settle(
        "a",
        {"input_tokens": 500_000, "output_tokens": 20_000,
         "reasoning_tokens": 30_000},
    )
    assert settled == pytest.approx(1.0)


def test_reservation_blocks_before_dispatch():
    budget = HardBudget(
        cap_usd=2,
        stop_usd=1.9,
        rates=SafetyRateCard(1, 10, "test"),
    )
    with pytest.raises(RuntimeError, match="hard budget blocks"):
        budget.reserve("too-large", 1_000_000, 100_000)
    assert budget.snapshot()["settled_safety_usd"] == 0


def test_provider_charge_is_not_hidden_by_lower_usage_estimate():
    budget = HardBudget(cap_usd=200, stop_usd=190)
    budget.reserve("a", 1_000, 100)
    assert budget.settle(
        "a", {"input_tokens": 1, "output_tokens": 1}, provider_charge_usd=3
    ) == 3


def test_cancel_releases_failed_pre_dispatch_reservation():
    budget = HardBudget()
    budget.reserve("a", 10_000, 100)
    assert budget.reserved_usd > 0
    budget.cancel("a")
    assert budget.reserved_usd == 0
