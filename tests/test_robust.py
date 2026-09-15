import math

import pytest

from token_yield.robust import (
    ConstantModel,
    Record,
    RidgeLinearModel,
    empirical_interval_coverage,
    grouped_bootstrap_improvement_ci,
    grouped_bootstrap_coefficient_ci,
    grouped_kfold,
    quantile,
)


def _records():
    return [
        Record((float(i), 1.0), 3.0 + 2.0 * i, f"g{i // 2}")
        for i in range(12)
    ]


def test_grouped_folds_have_no_leakage():
    records = _records()
    for train, validation in grouped_kfold(records, 3, seed=7):
        assert {records[i].group for i in train}.isdisjoint(
            {records[i].group for i in validation}
        )


def test_grouped_folds_are_deterministic():
    assert grouped_kfold(_records(), 3, 19) == grouped_kfold(_records(), 3, 19)
    assert grouped_kfold(_records(), 3, 19) != grouped_kfold(_records(), 3, 20)


def test_constant_features_produce_intercept_only_model():
    records = [Record((4.0, 4.0), y, f"g{i}") for i, y in enumerate((2, 4, 9))]
    model = RidgeLinearModel.fit(records, alpha=1)
    assert not model.coefficients
    assert model.predict((4, 4)) == pytest.approx(5)
    assert ConstantModel.fit(records).value == pytest.approx(5)


def test_ridge_shrinks_coefficients():
    records = _records()
    unregularized = RidgeLinearModel.fit(records, alpha=0)
    regularized = RidgeLinearModel.fit(records, alpha=100)
    assert sum(abs(x) for x in regularized.coefficients) < sum(
        abs(x) for x in unregularized.coefficients
    )
    assert math.isfinite(regularized.intercept)


def test_grouped_bootstrap_is_deterministic():
    records = _records()
    predictions = [r.target for r in records]
    baselines = [10.0] * len(records)
    first = grouped_bootstrap_improvement_ci(
        records, predictions, baselines, n_bootstrap=100, seed=12
    )
    second = grouped_bootstrap_improvement_ci(
        records, predictions, baselines, n_bootstrap=100, seed=12
    )
    assert first == second
    assert first[0] == pytest.approx(1)


def test_grouped_bootstrap_coefficients_include_intercept_and_features():
    intervals = grouped_bootstrap_coefficient_ci(
        _records(), 1.0, n_bootstrap=20, seed=7
    )
    assert len(intervals) == 3
    assert all(low <= high for low, high in intervals)


def test_quantile_and_interval_coverage_including_tails():
    actual = [0, 1, 2, 3, 4]
    assert quantile(actual, 0.25) == 1
    lower = [-1, 0, 3, 2, 5]
    upper = [1, 2, 4, 4, 6]
    assert empirical_interval_coverage(actual, lower, upper) == pytest.approx(0.6)
    assert empirical_interval_coverage(
        actual, lower, upper, tail_probability=0.25
    ) == pytest.approx(0.75)
