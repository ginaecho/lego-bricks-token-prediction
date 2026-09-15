"""Tests for wave-3 preregistration, budget bounds, and gate analysis."""

import json
from pathlib import Path

from token_yield.public_api import load_manifests
from token_yield.wave3_repair_cases import load_repair_cases
from token_yield.wave3_repair_runner import (
    PRIOR_CAMPAIGN_SAFETY_USD,
    analyze_repair,
    build_preregistration,
    reservation_bounds,
    run_repair,
    _unavailable_feature_plan,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "foundry_wave3"


def test_preregistration_freezes_36_ordered_cases_and_audit_hashes():
    value = build_preregistration(EXPERIMENT)
    assert value["session_count"] == 36
    assert [case["order"] for case in value["cases"]] == list(range(1, 37))
    assert len({case["case_id"] for case in value["cases"]}) == 36
    assert len(value["input_hashes"]["feature_registry"]) == 64
    assert value["repair_data_role"].startswith(
        "calibration-and-instrumentation-only"
    )


def test_every_reservation_includes_counting_and_generation():
    manifests = load_manifests(EXPERIMENT / "api_manifests.json")
    for case in load_repair_cases(EXPERIMENT):
        manifest = manifests.get(case.manifest_id) if case.manifest_id else None
        maximum_input, maximum_output = reservation_bounds(case, manifest)
        assert maximum_input > len(case.prompt.encode("utf-8")) * 2
        assert maximum_output >= case.output_bound_tokens


def test_dry_run_freezes_inputs_and_stays_under_cumulative_stop(tmp_path):
    result = run_repair(
        EXPERIMENT,
        tmp_path / "run",
        "https://example.openai.azure.com/openai/v1",
        dry_run=True,
    )
    assert result["status"] == "dry-run"
    assert result["cases_total"] == result["cases_remaining"] == 36
    assert result["projected_cumulative_safety_usd"] < 190
    assert result["budget"]["settled_safety_usd"] == PRIOR_CAMPAIGN_SAFETY_USD
    frozen = tmp_path / "run" / "frozen_inputs"
    assert (frozen / "preregistration.json").is_file()
    assert json.loads(
        (frozen / "preregistration.json").read_text(encoding="utf-8")
    )["pause_after_logical_sessions"] == 36


def test_analysis_is_explicitly_calibration_only():
    records = []
    for index in range(36):
        records.append({
            "case_id": f"case-{index}",
            "family": "fetch" if index < 6 else "control",
            "source_id": "source",
            "arm": "snapshot" if index < 3 else "live",
            "cache_warm_assignment": index % 2 == 0,
            "observed_cached": index % 3 == 0,
            "features": {
                "provider_input_tokens": 20,
                "task_payload_tokens": 10,
            },
            "usage": {"total_tokens": 30},
            "acceptance": {
                "provider_incomplete": False,
                "structural_acceptance": True,
                "semantic_acceptance": True,
                "overall_acceptance": True,
            },
        })
    result = analyze_repair(records, provider_count_available=False)
    assert result["status"] == "repair-complete-paused"
    assert "no model promotion" in result["claim_scope"]
    assert result["overall_accepted"] == 36
    assert result["provider_count_unavailable"] == 36
    assert result["known_measured_usage"]["total_tokens"] == 1080
    assert result["fetch_calibration"]["source"][
        "live_minus_snapshot_median"
    ] == 0


def test_unavailable_provider_count_blocks_model_row_but_keeps_plan():
    case = load_repair_cases(EXPERIMENT)[0]
    plan = _unavailable_feature_plan(case, "unsupported")
    assert plan["model_row_eligible"] is False
    assert plan["provider_input_tokens"] is None
    assert plan["fixed_overhead_tokens"] is None
    assert plan["task_payload_tokens"] > 0
