"""Accuracy, uncertainty and agreement statistics.

The organising idea: a start-up toll dominates small agent tasks, so total error is easy
and nearly uninformative. ``return 30969`` scores an excellent MAPE on the reference data.
Every accuracy number therefore comes as a :class:`Scorecard` carrying *both* the total
error and the variable-portion error, so it is structurally impossible to report one
without the other.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

Vector = Sequence[float]


# -- point accuracy ---------------------------------------------------------------


def ape(actual: float, predicted: float) -> float:
    """Absolute percentage error. Undefined at zero actual, which is a measurement bug."""
    if actual == 0:
        raise ZeroDivisionError("actual is zero; that is a failed measurement, not a datum")
    return abs(predicted - actual) / abs(actual) * 100.0


def mape(actual: Vector, predicted: Vector) -> float:
    return sum(ape(a, p) for a, p in zip(actual, predicted)) / len(list(actual))


def wape(actual: Vector, predicted: Vector) -> float:
    """Weighted absolute percentage error: sum |error| / sum |actual|.

    Preferred over MAPE on the variable portion, where individual actuals get small and
    per-row percentages explode.
    """
    denom = sum(abs(a) for a in actual)
    if denom <= 1e-12:
        return math.inf
    return sum(abs(p - a) for a, p in zip(actual, predicted)) / denom * 100.0


def vwape(actual: Vector, predicted: Vector, boot: float) -> float:
    """WAPE on the variable portion, with the start-up toll removed from both sides.

    Rows where ``actual - boot <= 0`` are kept. They are measurement failures -- a task
    that cost less than the empty task -- and hiding them would flatter the model.
    """
    a_var = [a - boot for a in actual]
    p_var = [p - boot for p in predicted]
    return wape(a_var, p_var)


def bias(actual: Vector, predicted: Vector) -> float:
    """Mean signed relative error. Positive means the model over-quotes."""
    return sum((p - a) / a for a, p in zip(actual, predicted)) / len(list(actual)) * 100.0


def p90_ape(actual: Vector, predicted: Vector) -> float:
    errs = sorted(ape(a, p) for a, p in zip(actual, predicted))
    return _quantile(errs, 0.90)


def relative_lift(baseline_error: float, model_error: float) -> float:
    """How much of the baseline's error the model removed, as a percentage.

    Negative means the model is worse than the baseline. This is the number the plan gates
    on, not the raw error, because a raw error can look excellent while removing nothing.
    """
    if baseline_error <= 1e-12:
        return 0.0
    return (baseline_error - model_error) / baseline_error * 100.0


@dataclass(frozen=True)
class Scorecard:
    """Total and variable-portion error, reported together or not at all."""

    n: int
    boot: float
    mape_total: float
    wape_total: float
    vwape: float
    bias_pct: float
    p90_ape: float
    negative_variable_rows: int
    """Rows whose actual fell below the empty-task cost. Never silently dropped."""

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "boot": self.boot,
            "mape_total_pct": round(self.mape_total, 4),
            "wape_total_pct": round(self.wape_total, 4),
            "vwape_pct": round(self.vwape, 4),
            "bias_pct": round(self.bias_pct, 4),
            "p90_ape_pct": round(self.p90_ape, 4),
            "negative_variable_rows": self.negative_variable_rows,
        }

    def summary(self) -> str:
        return (
            f"n={self.n}  total MAPE {self.mape_total:.2f}%  "
            f"vWAPE {self.vwape:.2f}%  bias {self.bias_pct:+.2f}%"
        )


def score(actual: Vector, predicted: Vector, boot: float) -> Scorecard:
    actual, predicted = list(actual), list(predicted)
    if not actual:
        raise ValueError("nothing to score")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted differ in length")
    return Scorecard(
        n=len(actual),
        boot=boot,
        mape_total=mape(actual, predicted),
        wape_total=wape(actual, predicted),
        vwape=vwape(actual, predicted, boot),
        bias_pct=bias(actual, predicted),
        p90_ape=p90_ape(actual, predicted),
        negative_variable_rows=sum(1 for a in actual if a - boot <= 0),
    )


# -- dispersion and drift ---------------------------------------------------------


def cv_pct(values: Vector) -> float:
    """Coefficient of variation. The noise floor is this, over replicate groups."""
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if abs(mean) < 1e-12:
        return math.inf
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var) / abs(mean) * 100.0


def noise_floor(groups: dict[str, list[float]]) -> float:
    """Pooled CV over replicate groups keyed by instruction hash."""
    cvs = [cv_pct(v) for v in groups.values() if len(v) >= 2]
    if not cvs:
        return 0.0
    return sum(cvs) / len(cvs)


def drift_pct(start: Vector, end: Vector) -> float:
    """Relative movement of the null probe between the opening and closing brackets."""
    s = sum(start) / len(list(start))
    e = sum(end) / len(list(end))
    if abs(s) < 1e-12:
        return math.inf
    return abs(e - s) / abs(s) * 100.0


# -- intervals --------------------------------------------------------------------


def coverage(actual: Vector, intervals: Sequence[tuple[float, float]]) -> tuple[int, int]:
    hits = sum(1 for a, (lo, hi) in zip(actual, intervals) if lo <= a <= hi)
    return hits, len(list(actual))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Coverage on 24 sealed cases is a *screen*, not a proof. Reporting "20/24 hits" as
    "90% coverage proven" is the mistake this function exists to make awkward: it always
    hands back an interval, and at n=24 that interval is wide.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def quantile(values: Vector, q: float) -> float:
    return _quantile(sorted(values), q)


# -- resampling -------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lo: float
    hi: float
    level: float
    iters: int

    def excludes_zero(self) -> bool:
        return self.lo > 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": round(self.point, 6),
            "lo": round(self.lo, 6),
            "hi": round(self.hi, 6),
            "level": self.level,
            "iters": self.iters,
        }


def bootstrap(
    items: Sequence[object],
    statistic: Callable[[Sequence[object]], float],
    *,
    iters: int = 10_000,
    level: float = 0.95,
    seed: int = 1337,
) -> BootstrapResult:
    """Percentile bootstrap over whole items.

    The *item* is the unit of analysis and it matters: for batching the item is a bundle,
    not a task, because tasks inside a bundle are not independent.
    """
    rng = random.Random(seed)
    n = len(items)
    if n == 0:
        raise ValueError("nothing to resample")
    point = statistic(items)
    draws = []
    for _ in range(iters):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        draws.append(statistic(sample))
    draws.sort()
    tail = (1.0 - level) / 2.0
    return BootstrapResult(
        point=point,
        lo=_quantile(draws, tail),
        hi=_quantile(draws, 1.0 - tail),
        level=level,
        iters=iters,
    )


def cluster_bootstrap(
    clusters: dict[str, list[object]],
    statistic: Callable[[Sequence[object]], float],
    *,
    iters: int = 10_000,
    level: float = 0.95,
    seed: int = 1337,
) -> BootstrapResult:
    """Resample whole clusters (e.g. source documents), not rows within them.

    With five clusters this is descriptive only. It is reported to show the spread, never
    to claim significance.
    """
    keys = list(clusters)
    rng = random.Random(seed)
    flat = [row for k in keys for row in clusters[k]]
    point = statistic(flat)
    draws = []
    for _ in range(iters):
        sample: list[object] = []
        for _ in range(len(keys)):
            sample.extend(clusters[keys[rng.randrange(len(keys))]])
        draws.append(statistic(sample))
    draws.sort()
    tail = (1.0 - level) / 2.0
    return BootstrapResult(point, _quantile(draws, tail), _quantile(draws, 1.0 - tail), level, iters)


# -- inter-rater agreement --------------------------------------------------------


def krippendorff_alpha_ordinal(units: Sequence[Sequence[int | None]]) -> float:
    """Krippendorff's alpha with an ordinal difference function.

    ``units`` is one row per item, one column per coder, ``None`` for missing. Ordinal is
    the right metric here because brick counts are ordered: confusing 1 with 2 is a
    smaller error than confusing 1 with 4, and nominal agreement would treat them alike.

    Returns 1.0 for perfect agreement, 0.0 for chance, negative for systematic disagreement.
    """
    pairable = [[v for v in row if v is not None] for row in units]
    pairable = [row for row in pairable if len(row) >= 2]
    if not pairable:
        raise ValueError("no unit has two or more codings")

    values = sorted({v for row in pairable for v in row})
    if len(values) == 1:
        return 1.0
    index = {v: i for i, v in enumerate(values)}
    k = len(values)

    coincidence = [[0.0] * k for _ in range(k)]
    for row in pairable:
        m = len(row)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coincidence[index[row[i]]][index[row[j]]] += 1.0 / (m - 1)

    marginals = [sum(coincidence[c]) for c in range(k)]
    total = sum(marginals)

    def delta_sq(c: int, g: int) -> float:
        lo, hi = (c, g) if c <= g else (g, c)
        inner = sum(marginals[t] for t in range(lo, hi + 1))
        return (inner - (marginals[lo] + marginals[hi]) / 2.0) ** 2

    d_obs = sum(
        coincidence[c][g] * delta_sq(c, g) for c in range(k) for g in range(k) if c != g
    )
    d_exp = sum(
        marginals[c] * marginals[g] * delta_sq(c, g)
        for c in range(k)
        for g in range(k)
        if c != g
    ) / (total - 1)

    if d_exp <= 1e-12:
        return 1.0
    return 1.0 - d_obs / d_exp


@dataclass
class AgreementReport:
    exact_vector_pct: float
    mae: float
    alpha_macro: float
    alpha_per_brick: dict[str, float] = field(default_factory=dict)
    n_units: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_vector_pct": round(self.exact_vector_pct, 2),
            "mae": round(self.mae, 4),
            "alpha_macro": round(self.alpha_macro, 4),
            "alpha_per_brick": {k: round(v, 4) for k, v in self.alpha_per_brick.items()},
            "n_units": self.n_units,
        }
