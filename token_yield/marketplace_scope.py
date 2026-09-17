"""Prepare and validate an additive Archive v2 scope, never new funding.

The CLI writes a disabled request for independent review and later user approval.
It only reads the existing campaign/state; it cannot authenticate or dispatch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .marketplace_agent_contracts import canonical, fingerprint, strict_json
from .marketplace_scenarios import SCENARIOS
from .marketplace_source_fixtures import fixture_documents, fixture_scenario

SCOPE_ID = "archive-atelier-v2-existing50-v1"
FIXTURE_ID = "archive-exceptions-v2"
HISTORICAL_SAFETY_USD = 1.49220
HISTORICAL_CALLS = 22


def source_identity() -> str:
    return fingerprint({"scenario": fixture_scenario(FIXTURE_ID), "documents": [
        fixture_documents(FIXTURE_ID, f"fixture-{i}", i) for i in range(6)]})


def _funding(campaign: dict, state: dict) -> None:
    if (not isinstance(state, dict) or not isinstance(state.get("pin"), dict)
            or not isinstance(state.get("budget"), dict) or "calls" not in state
            or not {"settled_safety_usd", "settled_requests", "active_reservations",
                    "active_reserved_usd"} <= set(state["budget"])
            or not isinstance(state["budget"]["settled_requests"], dict)):
        raise ValueError("complete existing funding ledger required")
    if campaign != {
        "approval_id": "marketplace-feedback-three-demos-50usd", "cap_usd": 50,
        "stop_usd": 48, "execution_enabled": True,
        "scenario_ids": [s["id"] for s in SCENARIOS],
    } or campaign["execution_enabled"] is not True:
        raise ValueError("exact unchanged original USD50/USD48 approval required")
    pin, budget = state["pin"], state["budget"]
    if (pin.get("campaign") != campaign or pin.get("mocked") is not False
            or pin.get("scenario_fingerprint") != fingerprint(SCENARIOS)
            or pin.get("cap_usd") != 50 or pin.get("stop_usd") != 48
            or budget.get("cap_usd") != 50 or budget.get("operational_stop_usd") != 48):
        raise ValueError("scope must use the existing real campaign funding pin")
    spent = budget["settled_safety_usd"]
    amounts = list(budget["settled_requests"].values())
    if (type(spent) not in (int, float) or not math.isfinite(spent)
            or spent + 1e-9 < HISTORICAL_SAFETY_USD or spent >= 48
            or type(state["calls"]) is not int or state["calls"] < HISTORICAL_CALLS
            or any(type(a) not in (int, float) or not math.isfinite(a) or a < 0 for a in amounts)
            or not math.isclose(sum(amounts), spent, rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError("historical settled spending/calls must be preserved within USD48 stop")
    if (state.get("halted") is not None or budget["active_reservations"]
            or budget["active_reserved_usd"] != 0):
        raise RuntimeError("unknown or reserved telemetry blocks scope authorization; no recovery")


def prepare_scope(campaign: dict, state: dict) -> dict:
    """Return a disabled, source-bound approval request without changing state."""
    _funding(campaign, state)
    return {
        "scope_id": SCOPE_ID, "version": 1, "funding": "existing-campaign-only",
        "campaign_fingerprint": fingerprint(campaign), "funding_pin": fingerprint(state["pin"]),
        "cap_usd": 50, "stop_usd": 48, "source_fixture": FIXTURE_ID,
        "source_fingerprint": source_identity(), "measurement_policy_enabled": False,
        "baseline": {"calls": state["calls"],
                     "settled_requests": dict(state["budget"]["settled_requests"]),
                     "settled_safety_usd": state["budget"]["settled_safety_usd"]},
        "review": {"verdict": "PENDING", "reference": ""},
        "run_authorization": {"approved": False, "reference": ""},
    }


def validate_scope(scope: dict, campaign: dict, state: dict) -> None:
    """Require separate review/run consent and preserve every baseline settlement."""
    _funding(campaign, state)
    expected = prepare_scope(campaign, state)
    if not isinstance(scope, dict) or set(scope) != set(expected):
        raise ValueError("exact versioned scope authorization fields required")
    for key in set(expected) - {"baseline", "review", "run_authorization"}:
        if scope[key] != expected[key] or type(scope[key]) is not type(expected[key]):
            raise ValueError(f"scope authorization mismatch: {key}")
    baseline = scope["baseline"]
    if (not isinstance(baseline, dict) or set(baseline) != set(expected["baseline"])
            or type(baseline["calls"]) is not int
            or not HISTORICAL_CALLS <= baseline["calls"] <= state["calls"]):
        raise ValueError("scope cannot reset historical calls")
    settlements = baseline["settled_requests"]
    if (not isinstance(settlements, dict) or not settlements
            or any(type(a) not in (int, float) or not math.isfinite(a) or a < 0
                   for a in settlements.values())
            or type(baseline["settled_safety_usd"]) not in (int, float)
            or not math.isfinite(baseline["settled_safety_usd"])
            or baseline["settled_safety_usd"] + 1e-9 < HISTORICAL_SAFETY_USD
            or not math.isclose(sum(settlements.values()), baseline["settled_safety_usd"],
                                rel_tol=1e-12, abs_tol=1e-9)
            or any(state["budget"]["settled_requests"].get(k) != v for k, v in settlements.items())):
        raise ValueError("scope cannot remove or rewrite prior settled requests")
    review, approval = scope["review"], scope["run_authorization"]
    if (not isinstance(review, dict) or set(review) != {"verdict", "reference"}
            or review["verdict"] != "PASS" or not isinstance(review["reference"], str)
            or not review["reference"].strip()):
        raise ValueError("independent scope review PASS and evidence reference required")
    if (not isinstance(approval, dict) or set(approval) != {"approved", "reference"}
            or approval["approved"] is not True or not isinstance(approval["reference"], str)
            or not approval["reference"].strip()):
        raise ValueError("subsequent explicit user run authorization required")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a disabled Archive v2 scope under existing funds.")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        campaign = strict_json(args.campaign.read_text(encoding="utf-8"))
        state = strict_json((args.state / "budget.json").read_text(encoding="utf-8"))
        scope = prepare_scope(campaign, state)
        # Exclusive creation cannot overwrite an existing approval or budget.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(canonical(scope) + "\n")
        print(f"Disabled scope written to {args.output}; no execution authorized.")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Scope preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
