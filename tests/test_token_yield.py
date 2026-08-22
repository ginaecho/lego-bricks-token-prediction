"""Tests for the token yield prediction pipeline."""

from token_yield.models import (
    CalibrationRecord,
    ComplexityTier,
    ProjectSpec,
    TaskTypeStats,
    TaskUnit,
)
from token_yield.calibrate import CalibrationStore
from token_yield.predict import TokenPredictor
from token_yield.forecast import ProjectForecaster
from token_yield.report import text_report, markdown_report


# ── Calibration ──────────────────────────────────────────────────────────

def _make_store() -> CalibrationStore:
    store = CalibrationStore()
    for tokens in (1000, 1200, 1100, 900, 1300):
        store.add(CalibrationRecord("bug_fix", tokens, duration_seconds=60.0))
    for tokens in (3000, 3500, 2800, 3200):
        store.add(CalibrationRecord("feature", tokens, duration_seconds=180.0))
    for tokens in (500, 600, 550):
        store.add(CalibrationRecord("docs", tokens, duration_seconds=30.0))
    return store


def test_calibration_store_basics():
    store = _make_store()
    assert store.total_records() == 12
    assert sorted(store.task_types) == ["bug_fix", "docs", "feature"]


def test_calibration_stats():
    store = _make_store()
    stats = store.stats("bug_fix")
    assert stats is not None
    assert stats.task_type == "bug_fix"
    assert stats.sample_count == 5
    assert stats.mean_tokens == 1100.0
    assert stats.min_tokens == 900
    assert stats.max_tokens == 1300
    assert stats.success_rate == 1.0


def test_calibration_stats_missing():
    store = CalibrationStore()
    assert store.stats("nonexistent") is None


def test_all_stats():
    store = _make_store()
    all_s = store.all_stats()
    assert len(all_s) == 3
    assert "bug_fix" in all_s
    assert "feature" in all_s
    assert "docs" in all_s


# ── Prediction (single) ─────────────────────────────────────────────────

def test_predict_single_base():
    store = _make_store()
    pred = TokenPredictor(store)
    p = pred.predict_single("bug_fix")
    assert p is not None
    assert p.task_type == "bug_fix"
    assert p.complexity == ComplexityTier.BASE
    assert p.multiplier == 1.0
    assert p.predicted_tokens == 1100


def test_predict_single_plus():
    store = _make_store()
    pred = TokenPredictor(store)
    p = pred.predict_single("bug_fix", ComplexityTier.PLUS)
    assert p is not None
    assert p.multiplier == 2.0
    assert p.predicted_tokens == 2200


def test_predict_single_plus_plus():
    store = _make_store()
    pred = TokenPredictor(store)
    p = pred.predict_single("bug_fix", ComplexityTier.PLUS_PLUS)
    assert p is not None
    assert p.multiplier == 4.0
    assert p.predicted_tokens == 4400


def test_predict_single_custom_multiplier():
    store = _make_store()
    pred = TokenPredictor(store)
    p = pred.predict_single("bug_fix", custom_multiplier=3.5)
    assert p is not None
    assert p.multiplier == 3.5
    assert p.predicted_tokens == 3850


def test_predict_missing_type():
    store = _make_store()
    pred = TokenPredictor(store)
    assert pred.predict_single("nonexistent") is None


# ── Prediction (scaled) ─────────────────────────────────────────────────

def test_predict_scaled():
    store = _make_store()
    pred = TokenPredictor(store)
    scaled = pred.predict_scaled("bug_fix")
    assert ComplexityTier.BASE in scaled
    assert ComplexityTier.PLUS in scaled
    assert ComplexityTier.PLUS_PLUS in scaled
    assert scaled[ComplexityTier.PLUS].predicted_tokens == 2 * scaled[ComplexityTier.BASE].predicted_tokens


# ── Prediction (combined) ───────────────────────────────────────────────

def test_predict_combined_single_type():
    store = _make_store()
    pred = TokenPredictor(store)
    combined = pred.predict_combined([("bug_fix", ComplexityTier.BASE, 1)])
    assert len(combined) == 1
    assert combined[0].predicted_tokens == 1100


def test_predict_combined_multi_type_has_overhead():
    store = _make_store()
    pred = TokenPredictor(store)

    single_a = pred.predict_single("bug_fix")
    single_b = pred.predict_single("feature")

    combined = pred.predict_combined([
        ("bug_fix", ComplexityTier.BASE, 1),
        ("feature", ComplexityTier.BASE, 1),
    ], interaction_overhead=0.15)

    sum_without_overhead = single_a.predicted_tokens + single_b.predicted_tokens
    sum_with_overhead = combined[0].predicted_tokens + combined[1].predicted_tokens

    assert sum_with_overhead > sum_without_overhead


def test_predict_combined_count_scales():
    store = _make_store()
    pred = TokenPredictor(store)
    combined = pred.predict_combined([("bug_fix", ComplexityTier.BASE, 3)])
    assert combined[0].predicted_tokens == 1100 * 3


# ── Compare scenarios ───────────────────────────────────────────────────

def test_compare_scenarios():
    store = _make_store()
    pred = TokenPredictor(store)
    totals = pred.compare_scenarios({
        "minimal": [("bug_fix", ComplexityTier.BASE, 1)],
        "full": [
            ("bug_fix", ComplexityTier.PLUS, 3),
            ("feature", ComplexityTier.BASE, 2),
        ],
    })
    assert "minimal" in totals
    assert "full" in totals
    assert totals["full"] > totals["minimal"]


# ── Forecaster ──────────────────────────────────────────────────────────

def test_forecast_basic():
    store = _make_store()
    forecaster = ProjectForecaster(store)
    spec = ProjectSpec("Test Project")
    spec.add("bug_fix", ComplexityTier.BASE, count=2)
    spec.add("feature", ComplexityTier.PLUS, count=1)

    fc = forecaster.forecast(spec)
    assert fc.project_name == "Test Project"
    assert len(fc.task_predictions) == 2
    assert fc.total_with_overhead > 0
    assert fc.estimated_hours >= 0


def test_forecast_cost():
    store = _make_store()
    forecaster = ProjectForecaster(store)
    spec = ProjectSpec("Cost Test")
    spec.add("bug_fix", count=5)

    fc = forecaster.forecast(spec)
    cost = fc.cost_at_rate(3.0)
    assert cost >= 0
    low, high = fc.cost_range(3.0)
    assert low <= cost <= high


def test_forecast_with_cost_dict():
    store = _make_store()
    forecaster = ProjectForecaster(store)
    spec = ProjectSpec("Dict Test")
    spec.add("bug_fix", count=2)
    spec.add("docs", count=1)

    result = forecaster.forecast_with_cost(spec, dollars_per_million_tokens=3.0)
    assert result["project"] == "Dict Test"
    assert "total_tokens" in result
    assert "estimated_cost" in result
    assert "task_breakdown" in result
    assert len(result["task_breakdown"]) == 2


# ── Reports ─────────────────────────────────────────────────────────────

def test_text_report():
    store = _make_store()
    forecaster = ProjectForecaster(store)
    spec = ProjectSpec("Report Test")
    spec.add("bug_fix", ComplexityTier.PLUS, count=3)
    spec.add("feature", count=2)
    spec.add("docs", count=5)

    fc = forecaster.forecast(spec)
    report = text_report(fc)
    assert "Report Test" in report
    assert "bug_fix" in report
    assert "Budget Summary" in report


def test_markdown_report():
    store = _make_store()
    forecaster = ProjectForecaster(store)
    spec = ProjectSpec("MD Test")
    spec.add("bug_fix", count=1)

    fc = forecaster.forecast(spec)
    report = markdown_report(fc)
    assert "# Token Yield Forecast" in report
    assert "| Type |" in report


# ── ProjectSpec builder ─────────────────────────────────────────────────

def test_project_spec_builder():
    spec = (ProjectSpec("Builder Test")
            .add("bug_fix", count=3)
            .add("feature", ComplexityTier.PLUS, count=2)
            .add("docs", count=5))
    assert spec.total_task_count == 10
    assert len(spec.tasks) == 3


# ── Confidence interval ─────────────────────────────────────────────────

def test_confidence_interval_narrows_with_samples():
    store = CalibrationStore()
    for t in (1000, 1100, 1050, 950, 1000, 1050, 1100, 950, 1000, 1050):
        store.add(CalibrationRecord("stable_type", t))
    stats = store.stats("stable_type")
    assert stats is not None
    assert stats.confidence_width < 0.5


def test_confidence_interval_wide_with_few_samples():
    store = CalibrationStore()
    store.add(CalibrationRecord("rare_type", 1000))
    stats = store.stats("rare_type")
    assert stats is not None
    assert stats.confidence_width == 1.0


# ── TaskTypeStats properties ────────────────────────────────────────────

def test_cv_zero_mean():
    stats = TaskTypeStats(task_type="test", mean_tokens=0.0, stddev_tokens=0.0)
    assert stats.cv == 0.0


def test_cv_nonzero():
    stats = TaskTypeStats(task_type="test", mean_tokens=1000.0, stddev_tokens=100.0)
    assert abs(stats.cv - 0.1) < 1e-9


# ── ComplexityTier ──────────────────────────────────────────────────────

def test_complexity_multipliers():
    assert ComplexityTier.BASE.default_multiplier == 1.0
    assert ComplexityTier.PLUS.default_multiplier == 2.0
    assert ComplexityTier.PLUS_PLUS.default_multiplier == 4.0
    assert ComplexityTier.CUSTOM.default_multiplier == 1.0


# ── Custom multipliers per task type ────────────────────────────────────

def test_custom_multipliers_override():
    store = _make_store()
    pred = TokenPredictor(store, custom_multipliers={
        "bug_fix": {ComplexityTier.PLUS: 1.8},
    })
    p = pred.predict_single("bug_fix", ComplexityTier.PLUS)
    assert p is not None
    assert p.multiplier == 1.8
    assert p.predicted_tokens == int(1100 * 1.8)
