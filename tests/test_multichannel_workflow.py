"""End-to-end offline diagnostic tests for the multichannel quote model."""

from pathlib import Path

import pytest

from token_yield.economics import Pricing
from token_yield.multichannel_workflow import (
    HistoricalQuote,
    fit_wave2_historical_diagnostic,
    fit_wave3_repair_diagnostic,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260826_1627_wave2"
EXPERIMENT = ROOT / "experiments" / "foundry_wave2"
WAVE3_RUN = ROOT / "runs" / "20260827_1152_wave3"
WAVE3_EXPERIMENT = ROOT / "experiments" / "foundry_wave3"


def test_wave2_diagnostic_preserves_historical_only_status_and_groups():
    diagnostic = fit_wave2_historical_diagnostic(RUN, EXPERIMENT)

    assert diagnostic.status == "historical-exploratory-only"
    assert diagnostic.rows_used == 29
    assert diagnostic.rows_excluded == 1
    assert diagnostic.independent_groups == 10
    assert diagnostic.promotion_eligible is False
    assert diagnostic.model.channel_comparisons[
        "non_reasoning_output_tokens"
    ].status == "IDENTIFIED"
    assert diagnostic.model.channel_comparisons[
        "reasoning_tokens"
    ].status == "UNIDENTIFIED"
    assert diagnostic.model.channel_comparisons[
        "branch_or_retry_probability"
    ].status == "UNIDENTIFIED"


def test_historical_quote_reconstructs_bounded_channels_and_cost():
    diagnostic = fit_wave2_historical_diagnostic(RUN, EXPERIMENT)
    quote = HistoricalQuote(
        task_payload_tokens=800,
        declared_fetch_bytes=0,
        output_bound_tokens=512,
        output_spec_units=4,
        k_declared=1,
        family="summarise",
        arm=None,
        input_tokens=850,
    )
    forecast = diagnostic.forecast(
        quote,
        pricing=Pricing(
            input_per_million=10,
            cached_input_per_million=2,
            output_per_million=50,
        ),
    )

    assert forecast.p95.non_reasoning_output_tokens <= 512
    assert forecast.p95.reasoning_tokens is None
    assert forecast.p95.output_tokens is None
    assert forecast.p50.total_tokens is None
    assert forecast.p50.cost is None
    assert forecast.p95.calibration_status == "EXPLORATORY"
    assert forecast.p50.accepted_work_cost is None


def test_diagnostic_report_exposes_channel_baselines_without_promotion():
    report = fit_wave2_historical_diagnostic(RUN, EXPERIMENT).report()

    assert report["status"] == "historical-exploratory-only"
    assert report["promotion_eligible"] is False
    assert report["channels"]["non_reasoning_output_tokens"]["n_groups"] >= 3
    assert report["channels"]["non_reasoning_output_tokens"][
        "relative_mae_improvement"
    ] < 0.20
    assert report["channels"]["reasoning_tokens"]["model_mae"] is None
    assert report["channels"]["acceptance_probability"]["model_mae"] == pytest.approx(
        0.29629, rel=1e-4
    )


def test_wave3_retraining_is_calibration_only_and_uses_quote_time_features():
    diagnostic = fit_wave3_repair_diagnostic(WAVE3_RUN, WAVE3_EXPERIMENT)
    report = diagnostic.report()

    assert diagnostic.status == "wave3-repair-calibration-only"
    assert diagnostic.rows_used == 36
    assert diagnostic.rows_excluded == 0
    assert diagnostic.independent_groups == 16
    assert diagnostic.promotion_eligible is False
    assert report["promotion_eligible"] is False
    assert "cache_warm" in report["feature_names"]
    assert "effort_medium" in report["feature_names"]
    assert report["channels"]["accepted_work_cost"]["status"] == "UNAVAILABLE"
    assert any("repair gate" in item for item in report["limitations"])
