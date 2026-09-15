"""Tests for reservations and preregistered pilot gates."""

from pathlib import Path

from token_yield.pilot_cases import load_pilot_cases
from token_yield.pilot_runner import _restore_budget, reservation_bounds
from token_yield.public_api import load_manifests

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "foundry_wave2"


def test_all_worst_case_reservations_fit_far_inside_hard_cap():
    cases = load_pilot_cases(EXPERIMENT)
    manifests = load_manifests(EXPERIMENT / "api_manifests.json")
    from token_yield.budget import SafetyRateCard
    rates = SafetyRateCard()
    total = 0
    for case in cases:
        manifest = manifests.get(case.manifest_id) if case.manifest_id else None
        total += rates.price(*reservation_bounds(case, manifest))
    assert total < 50


def test_live_fetch_reserves_two_outputs_and_api_response():
    case = next(
        item for item in load_pilot_cases(EXPERIMENT)
        if item.case_id == "fetch-2019-24499-live"
    )
    manifest = load_manifests(
        EXPERIMENT / "api_manifests.json"
    )[case.manifest_id]
    maximum_input, maximum_output = reservation_bounds(case, manifest)
    assert maximum_input > manifest.max_response_bytes
    assert maximum_output == case.max_output_tokens * 2


def test_restore_budget_retains_full_reservation_for_unknown_usage():
    budget = _restore_budget([{
        "case_id": "failed",
        "reserved_safety_usd": 7.5,
        "usage": {"input_tokens": None, "output_tokens": None},
    }])
    assert budget.settled_usd == 7.5
