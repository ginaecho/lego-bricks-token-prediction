"""The small amount of linear algebra the multi-head fit needs, dependency-free.

:mod:`token_yield.compose` already solves one ordinary least-squares problem and
carries its own private solver. This module exists because predicting *value*
and *impact* needs two things that one does not have:

* **weighted** least squares, the inner step of a logistic fit, and
* the ability to hand back ``(XᵀWX)⁻¹x`` so a probability can be quoted with a
  standard error rather than as a bare number.

Everything here is deliberately plain: Gaussian elimination, no dependencies, no
matrix class. The problems are tens of rows by tens of columns and will stay
that way, because the feature vector is a fixed vocabulary of task bricks.
"""

from __future__ import annotations

import math
from typing import List, Sequence

Matrix = Sequence[Sequence[float]]
Vector = Sequence[float]

#: Ridge for the logistic head, as a fraction of each column's own scale.
#: Chosen so that a perfectly separable column still yields probabilities in
#: roughly 0.01-0.99 rather than 0.0/1.0. It shrinks proportionally at any
#: sample size, because the penalty is scaled by the normal matrix itself.
LOGIT_RIDGE = 0.05


def solve(a: Matrix, b: Vector) -> List[float]:
    """Solve ``a x = b`` by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            m[col][col] += 1e-9          # singular column: nudge, don't crash
            piv = col
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / pv
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _normal_matrix(x: Matrix, w: Vector, ridge: float) -> List[List[float]]:
    """``XᵀWX`` with ridge applied relative to each column's own scale.

    The relative scaling matters: a single global ridge would be set by the
    largest feature (context bytes, whose squares run to 1e9) and would swamp
    the intercept column entirely — silently destroying the term it was meant
    to stabilise.
    """
    k = len(x[0])
    xtx = [[sum(w[r] * x[r][i] * x[r][j] for r in range(len(x)))
            for j in range(k)] for i in range(k)]
    for i in range(k):
        xtx[i][i] += ridge * (abs(xtx[i][i]) or 1.0)
    return xtx


def weighted_least_squares(x: Matrix, y: Vector, w: Vector = None,
                           ridge: float = 1e-6) -> List[float]:
    """Fit ``y ≈ x·beta``, optionally weighting each row."""
    if w is None:
        w = [1.0] * len(y)
    k = len(x[0])
    xty = [sum(w[r] * x[r][i] * y[r] for r in range(len(x))) for i in range(k)]
    return solve(_normal_matrix(x, w, ridge), xty)


def least_squares(x: Matrix, y: Vector, ridge: float = 1e-6) -> List[float]:
    return weighted_least_squares(x, y, None, ridge)


def inverse(a: Matrix) -> List[List[float]]:
    """``a⁻¹``, by solving against each basis vector.

    Only ever called on the normal matrix, whose side is the number of
    features — tens, not thousands — so the cubic cost is irrelevant and the
    clarity is worth more than a factorisation.
    """
    n = len(a)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve(a, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def leverage(x: Matrix, w: Vector = None, ridge: float = 1e-6) -> List[float]:
    """The diagonal of the hat matrix: ``hᵢ = wᵢ · xᵢᵀ (XᵀWX)⁻¹ xᵢ``.

    Leverage is how much a row pulls the fit toward itself. It is what makes
    exact leave-one-out affordable: for a least-squares fit the deletion
    residual is just ``eᵢ / (1 - hᵢ)``, so honest cross-validation costs one
    fit rather than one fit per row.
    """
    if w is None:
        w = [1.0] * len(x)
    inv = inverse(_normal_matrix(x, w, ridge))
    out = []
    for r, row in enumerate(x):
        q = sum(row[i] * sum(inv[i][j] * row[j] for j in range(len(row)))
                for i in range(len(row)))
        out.append(min(max(w[r] * q, 0.0), 1.0 - 1e-9))
    return out


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def irls_logistic(x: Matrix, y: Vector, ridge: float = LOGIT_RIDGE,
                  iterations: int = 30) -> List[float]:
    """Logistic regression by iteratively reweighted least squares.

    Ridge is deliberately larger than for the linear heads. Historical
    engagement outcomes are close to separable — big compliance programmes in
    regulated industries almost always land — and an unpenalised fit answers
    separability with infinite coefficients and a confidence of exactly 1.0.
    A win probability of 1.0 is never true and is the most expensive kind of
    wrong, so the fit is kept finite by construction.
    """
    n, k = len(y), len(x[0])
    beta = [0.0] * k
    for _ in range(iterations):
        eta = [sum(b * f for b, f in zip(beta, x[r])) for r in range(n)]
        p = [sigmoid(e) for e in eta]
        w = [max(pi * (1.0 - pi), 1e-6) for pi in p]
        z = [eta[r] + (y[r] - p[r]) / w[r] for r in range(n)]
        nxt = weighted_least_squares(x, z, w, ridge)
        if max(abs(a - b) for a, b in zip(nxt, beta)) < 1e-8:
            return nxt
        beta = nxt
    return beta


def standard_error(x: Matrix, w: Vector, point: Vector,
                   ridge: float = LOGIT_RIDGE) -> float:
    """Standard error of the linear predictor at ``point``.

    ``se² = xᵀ (XᵀWX)⁻¹ x``, computed by solving rather than inverting.
    """
    z = solve(_normal_matrix(x, w, ridge), list(point))
    var = sum(a * b for a, b in zip(point, z))
    return math.sqrt(max(var, 0.0))


def quantile(values: Sequence[float], q: float) -> float:
    """Empirical quantile by linear interpolation; ``[]`` gives 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
