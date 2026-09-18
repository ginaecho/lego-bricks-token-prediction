"""Small dense linear algebra, pure standard library.

Everything here operates on lists of lists. The matrices in this project are tiny -- at
most a few hundred rows by eleven columns -- so clarity beats speed, and a dependency-free
core is worth more than a fast one.
"""

from __future__ import annotations

import math

Matrix = list[list[float]]
Vector = list[float]


def transpose(X: Matrix) -> Matrix:
    return [list(col) for col in zip(*X)]


def gram(X: Matrix) -> Matrix:
    """X^T X."""
    n = len(X[0])
    out = [[0.0] * n for _ in range(n)]
    for row in X:
        for i in range(n):
            ri = row[i]
            if ri == 0.0:
                continue
            for j in range(n):
                out[i][j] += ri * row[j]
    return out


def matvec_t(X: Matrix, y: Vector) -> Vector:
    """X^T y."""
    n = len(X[0])
    out = [0.0] * n
    for row, yi in zip(X, y):
        if yi == 0.0:
            continue
        for j in range(n):
            out[j] += row[j] * yi
    return out


def matvec(X: Matrix, b: Vector) -> Vector:
    return [sum(xij * bj for xij, bj in zip(row, b)) for row in X]


def solve(A: Matrix, b: Vector) -> Vector:
    """Gaussian elimination with partial pivoting. Raises on a singular system."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-14:
            raise ZeroDivisionError(f"singular system at column {col}")
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] / pv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = acc / M[r][r]
    return x


def solve_ols(X: Matrix, y: Vector, ridge: float = 1e-9) -> Vector:
    """Least squares via normal equations with a stabilising ridge.

    The ridge is 1e-9 by default: enough to survive a collinear column, small enough that
    it never shows up in a reported coefficient. If you need real regularisation, pass a
    real value -- do not rely on this one.
    """
    A = gram(X)
    for i in range(len(A)):
        A[i][i] += ridge
    return solve(A, matvec_t(X, y))


def nnls(X: Matrix, y: Vector, tol: float = 1e-10, max_iter: int | None = None) -> Vector:
    """Lawson-Hanson non-negative least squares.

    This is the principled way to impose "a brick cannot cost negative tokens". The
    alternative -- fitting OLS and clamping negatives to zero afterwards -- silently
    biases every *other* coefficient, because the clamped variance has to go somewhere.
    Both are reported; only this one is used as a candidate model.
    """
    m, n = len(X), len(X[0])
    if max_iter is None:
        max_iter = 3 * n
    passive = [False] * n
    x = [0.0] * n

    for _ in range(max_iter):
        r = [y[i] - sum(X[i][j] * x[j] for j in range(n)) for i in range(m)]
        w = matvec_t(X, r)
        candidates = [j for j in range(n) if not passive[j]]
        if not candidates:
            break
        j = max(candidates, key=lambda k: w[k])
        if w[j] <= tol:
            break
        passive[j] = True

        for _inner in range(max_iter):
            idx = [k for k in range(n) if passive[k]]
            if not idx:
                break
            Xp = [[row[k] for k in idx] for row in X]
            try:
                sp = solve_ols(Xp, y, ridge=1e-12)
            except ZeroDivisionError:
                passive[j] = False
                break
            s = [0.0] * n
            for pos, k in enumerate(idx):
                s[k] = sp[pos]
            if min(s[k] for k in idx) > tol:
                x = s
                break
            ratios = [
                x[k] / (x[k] - s[k])
                for k in idx
                if s[k] <= tol and abs(x[k] - s[k]) > 1e-15
            ]
            if not ratios:
                x = [max(0.0, s[k]) for k in range(n)]
                break
            alpha = min(ratios)
            x = [x[k] + alpha * (s[k] - x[k]) for k in range(n)]
            for k in idx:
                if abs(x[k]) < tol:
                    passive[k] = False
                    x[k] = 0.0
        else:
            break

    return [max(0.0, v) for v in x]


def jacobi_eigenvalues(A: Matrix, sweeps: int = 100, tol: float = 1e-12) -> list[float]:
    """Eigenvalues of a symmetric matrix by cyclic Jacobi rotation. Descending."""
    n = len(A)
    M = [list(row) for row in A]
    for _ in range(sweeps):
        off = math.sqrt(sum(M[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(M[p][q]) < tol:
                    continue
                theta = (M[q][q] - M[p][p]) / (2.0 * M[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = M[k][p], M[k][q]
                    M[k][p] = c * mkp - s * mkq
                    M[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = M[p][k], M[q][k]
                    M[p][k] = c * mpk - s * mqk
                    M[q][k] = s * mpk + c * mqk
    return sorted((M[i][i] for i in range(n)), reverse=True)


def _column_normalise(X: Matrix) -> Matrix:
    n = len(X[0])
    norms = [math.sqrt(sum(row[j] ** 2 for row in X)) or 1.0 for j in range(n)]
    return [[row[j] / norms[j] for j in range(n)] for row in X]


def condition_number(X: Matrix) -> float:
    """Belsley's scaled condition number: ``sqrt(lambda_max / lambda_min)`` of ``X^T X``
    after normalising each column to unit length.

    The normalisation is not optional. An intercept column of 1.0 sitting beside a byte
    count in the tens of thousands produces a condition number in the tens of thousands
    purely from the difference in units, which says nothing about whether the columns are
    collinear. Scaling first is what makes the conventional threshold of 30 mean anything.

    Returns ``inf`` when rank-deficient.
    """
    eig = jacobi_eigenvalues(gram(_column_normalise(X)))
    lo, hi = eig[-1], eig[0]
    if lo <= 1e-12 or hi <= 0:
        return math.inf
    return math.sqrt(hi / lo)


def r_squared(y: Vector, yhat: Vector) -> float:
    mean = sum(y) / len(y)
    ss_tot = sum((yi - mean) ** 2 for yi in y)
    ss_res = sum((yi - hi) ** 2 for yi, hi in zip(y, yhat))
    if ss_tot <= 1e-15:
        return 1.0 if ss_res <= 1e-15 else 0.0
    return 1.0 - ss_res / ss_tot


def vif(X: Matrix) -> list[float]:
    """Variance inflation factor per column, regressing each on the others plus intercept.

    A constant column has no variance to inflate and reports 1.0 rather than infinity --
    the intercept is not a collinearity problem, it is the design.
    """
    n = len(X[0])
    out: list[float] = []
    for j in range(n):
        target = [row[j] for row in X]
        if max(target) - min(target) < 1e-12:
            out.append(1.0)
            continue
        others = [[1.0] + [row[k] for k in range(n) if k != j] for row in X]
        try:
            beta = solve_ols(others, target)
        except ZeroDivisionError:
            out.append(math.inf)
            continue
        pred = matvec(others, beta)
        r2 = r_squared(target, pred)
        out.append(math.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
    return out


def _ranks(values: Vector) -> Vector:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Vector, b: Vector) -> float:
    """Rank correlation with tie averaging. 0.0 when either side is constant."""
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return num / (da * db)
