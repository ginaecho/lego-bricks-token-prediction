import math

import pytest

from assay.metrics import (
    ape,
    bias,
    bootstrap,
    coverage,
    cv_pct,
    drift_pct,
    krippendorff_alpha_ordinal,
    mape,
    noise_floor,
    quantile,
    relative_lift,
    score,
    vwape,
    wape,
    wilson_ci,
)


def test_ape_hand_computed():
    assert ape(100.0, 110.0) == pytest.approx(10.0)
    assert ape(100.0, 90.0) == pytest.approx(10.0)


def test_ape_refuses_a_zero_actual():
    with pytest.raises(ZeroDivisionError):
        ape(0.0, 10.0)


def test_mape_and_wape_hand_computed():
    actual = [100.0, 200.0]
    pred = [110.0, 180.0]
    assert mape(actual, pred) == pytest.approx((10.0 + 10.0) / 2)
    assert wape(actual, pred) == pytest.approx((10 + 20) / 300 * 100)


def test_vwape_removes_the_toll_from_both_sides():
    boot = 30_000.0
    actual = [30_100.0, 30_400.0]
    pred = [30_150.0, 30_300.0]
    # variable actuals are 100 and 400; variable predictions 150 and 300.
    assert vwape(actual, pred, boot) == pytest.approx((50 + 100) / 500 * 100)


def test_the_constant_predictor_looks_excellent_on_total_and_terrible_on_variable():
    """The whole plan exists because of this asymmetry. Pin it down."""
    boot = 30_000.0
    actual = [30_100.0, 30_400.0, 31_000.0, 35_000.0]
    constant = [boot] * 4

    card = score(actual, constant, boot)
    assert card.mape_total < 5.0, "a constant scores brilliantly on total error"
    assert card.vwape == pytest.approx(100.0), "and explains none of the variable portion"


def test_scorecard_reports_negative_variable_rows_rather_than_hiding_them():
    boot = 30_000.0
    actual = [29_500.0, 30_500.0]
    card = score(actual, [30_000.0, 30_400.0], boot)
    assert card.negative_variable_rows == 1


def test_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score([1.0, 2.0], [1.0], boot=0.0)


def test_bias_sign_means_over_quoting():
    assert bias([100.0], [110.0]) == pytest.approx(10.0)
    assert bias([100.0], [90.0]) == pytest.approx(-10.0)


def test_relative_lift_is_zero_when_the_model_matches_the_baseline():
    assert relative_lift(20.0, 20.0) == pytest.approx(0.0)
    assert relative_lift(20.0, 10.0) == pytest.approx(50.0)
    assert relative_lift(20.0, 30.0) == pytest.approx(-50.0)


def test_cv_and_noise_floor():
    assert cv_pct([100.0, 100.0, 100.0]) == pytest.approx(0.0)
    groups = {"a": [100.0, 101.0], "b": [200.0, 202.0]}
    assert noise_floor(groups) == pytest.approx(cv_pct([100.0, 101.0]), rel=1e-6)


def test_drift_pct():
    assert drift_pct([100.0, 100.0], [102.0, 102.0]) == pytest.approx(2.0)


def test_coverage_counts_inclusive_hits():
    hits, n = coverage([5.0, 15.0], [(0.0, 10.0), (0.0, 10.0)])
    assert (hits, n) == (1, 2)
    hits, _ = coverage([10.0], [(0.0, 10.0)])
    assert hits == 1


def test_wilson_ci_is_wide_at_n_24():
    lo, hi = wilson_ci(20, 24)
    assert lo < 0.83 - 0.05, "20/24 must not be reportable as proven 90% coverage"
    assert hi > 0.83


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi


def test_quantile_interpolates():
    assert quantile([0.0, 10.0], 0.5) == pytest.approx(5.0)
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0


def test_bootstrap_of_a_constant_has_zero_width():
    res = bootstrap([1.0] * 20, lambda s: sum(s) / len(s), iters=200)
    assert res.point == pytest.approx(1.0)
    assert res.lo == pytest.approx(1.0)
    assert res.hi == pytest.approx(1.0)


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    data = [float(i) for i in range(30)]
    stat = lambda s: sum(s) / len(s)  # noqa: E731
    a = bootstrap(data, stat, iters=300, seed=7)
    b = bootstrap(data, stat, iters=300, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_bootstrap_excludes_zero_only_when_the_effect_is_real():
    strong = bootstrap([5.0] * 30, lambda s: sum(s) / len(s), iters=500)
    assert strong.excludes_zero()
    noisy = bootstrap(
        [-1.0, 1.0] * 15, lambda s: sum(s) / len(s), iters=500
    )
    assert not noisy.excludes_zero()


def test_krippendorff_perfect_agreement():
    units = [[1, 1], [2, 2], [3, 3], [0, 0]]
    assert krippendorff_alpha_ordinal(units) == pytest.approx(1.0)


def test_krippendorff_ordinal_penalises_distant_confusions_more():
    near = [[1, 2], [2, 3], [1, 1], [3, 3], [2, 2], [0, 0]]
    far = [[1, 4], [2, 5], [1, 1], [3, 3], [2, 2], [0, 0]]
    assert krippendorff_alpha_ordinal(near) > krippendorff_alpha_ordinal(far)


def test_krippendorff_requires_a_pairable_unit():
    with pytest.raises(ValueError):
        krippendorff_alpha_ordinal([[1, None], [2, None]])


def test_krippendorff_single_valued_corpus_is_perfect_agreement():
    assert krippendorff_alpha_ordinal([[1, 1], [1, 1]]) == pytest.approx(1.0)


def test_wape_of_an_all_zero_target_is_infinite_not_zero():
    assert math.isinf(wape([0.0, 0.0], [1.0, 1.0]))
