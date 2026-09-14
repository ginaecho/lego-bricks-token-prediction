import math

import pytest

from assay.linalg import (
    condition_number,
    gram,
    matvec,
    nnls,
    r_squared,
    solve,
    solve_ols,
    spearman,
    vif,
)


def test_solve_recovers_known_solution():
    A = [[2.0, 1.0], [1.0, 3.0]]
    b = [5.0, 10.0]
    x = solve(A, b)
    assert x[0] == pytest.approx(1.0, abs=1e-12)
    assert x[1] == pytest.approx(3.0, abs=1e-12)


def test_solve_raises_on_singular():
    with pytest.raises(ZeroDivisionError):
        solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_ols_recovers_planted_coefficients_exactly():
    # y = 5 + 2*a - 3*b, no noise. The estimator must find it to machine precision.
    rows = [(a, b) for a in range(6) for b in range(6)]
    X = [[1.0, float(a), float(b)] for a, b in rows]
    y = [5.0 + 2.0 * a - 3.0 * b for a, b in rows]
    beta = solve_ols(X, y)
    assert beta[0] == pytest.approx(5.0, abs=1e-6)
    assert beta[1] == pytest.approx(2.0, abs=1e-6)
    assert beta[2] == pytest.approx(-3.0, abs=1e-6)


def test_nnls_matches_ols_when_truth_is_non_negative():
    rows = [(a, b) for a in range(6) for b in range(6)]
    X = [[1.0, float(a), float(b)] for a, b in rows]
    y = [5.0 + 2.0 * a + 3.0 * b for a, b in rows]
    beta = nnls(X, y)
    assert beta[0] == pytest.approx(5.0, abs=1e-4)
    assert beta[1] == pytest.approx(2.0, abs=1e-4)
    assert beta[2] == pytest.approx(3.0, abs=1e-4)


def test_nnls_never_returns_a_negative_coefficient():
    # Truth has a genuinely negative term; NNLS must floor it rather than report it.
    rows = [(a, b) for a in range(6) for b in range(6)]
    X = [[1.0, float(a), float(b)] for a, b in rows]
    y = [5.0 + 2.0 * a - 3.0 * b for a, b in rows]
    beta = nnls(X, y)
    assert all(v >= 0.0 for v in beta)
    ols = solve_ols(X, y)
    assert ols[2] < 0.0, "the OLS fit must still expose the negative sign"


def test_r_squared_perfect_and_null():
    y = [1.0, 2.0, 3.0, 4.0]
    assert r_squared(y, y) == pytest.approx(1.0)
    mean = [2.5] * 4
    assert r_squared(y, mean) == pytest.approx(0.0)


def test_condition_number_is_small_for_orthogonal_design():
    X = [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]
    assert condition_number(X) < 30.0


def test_condition_number_is_infinite_when_rank_deficient():
    X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    assert math.isinf(condition_number(X))


def test_vif_flags_a_collinear_column_and_ignores_the_intercept():
    X = [[1.0, float(i), float(2 * i) + 0.001 * (i % 2)] for i in range(20)]
    factors = vif(X)
    assert factors[0] == 1.0, "the intercept is the design, not a collinearity problem"
    assert factors[1] > 5.0 and factors[2] > 5.0


def test_vif_is_near_one_for_independent_columns():
    rows = [(a, b) for a in range(6) for b in range(6)]
    X = [[1.0, float(a), float(b)] for a, b in rows]
    assert all(f < 1.5 for f in vif(X)[1:])


def test_spearman_perfect_monotone_and_constant():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [7, 7, 7, 7]) == 0.0


def test_spearman_handles_ties():
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_gram_and_matvec_agree_with_naive_products():
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    g = gram(X)
    assert g == [[35.0, 44.0], [44.0, 56.0]]
    assert matvec(X, [1.0, 1.0]) == [3.0, 7.0, 11.0]
