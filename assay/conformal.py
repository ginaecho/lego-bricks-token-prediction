"""Calibrated prediction intervals by split conformal.

A point estimate is not a budget. The reference reported single numbers and derived a
"band" by rounding up its own held-out error, which is a description of past performance
rather than a statement about the next task.

Split conformal gives a real coverage guarantee under exchangeability, from a calibration
fold the model never saw. Two choices are made *before* any held-out data is opened, and
recorded so they cannot be revisited once the intervals look inconvenient:

* absolute versus relative nonconformity, decided by a heteroscedasticity check on
  training residuals;
* the level, frozen at 90% for the demo.

Coverage is then *screened* on the blind set and reported with a Wilson interval. With 24
sealed tasks, "20 out of 24 hits" is not "90% coverage proven", and nothing in this module
will let it be written that way.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from assay.linalg import spearman
from assay.metrics import quantile


class Nonconformity(str, Enum):
    ABSOLUTE = "absolute"
    """Errors are roughly the same size regardless of the quote. Band is +/- a constant."""

    RELATIVE = "relative"
    """Errors grow with the quote. Band is a percentage."""


def choose_nonconformity(
    actual: Sequence[float], predicted: Sequence[float], *, threshold: float = 0.30
) -> tuple[Nonconformity, float]:
    """Pick the score function from how the residuals behave, not from how they score.

    Returns the choice and the rank correlation that drove it, so the decision is auditable.
    """
    residuals = [abs(a - p) for a, p in zip(actual, predicted)]
    rho = spearman(list(predicted), residuals)
    return (Nonconformity.RELATIVE if abs(rho) > threshold else Nonconformity.ABSOLUTE), rho


@dataclass(frozen=True)
class Conformal:
    level: float
    method: Nonconformity
    q: float
    n_calibration: int
    floor: float = 0.0
    """A quote can never fall below the empty-task cost; that is a measurement, not a bound."""
    heteroscedasticity_rho: float = 0.0

    def interval(self, point: float) -> tuple[float, float]:
        if self.method is Nonconformity.ABSOLUTE:
            lo, hi = point - self.q, point + self.q
        else:
            lo, hi = point * (1 - self.q), point * (1 + self.q)
        return (max(self.floor, lo), hi)

    def half_width(self, point: float) -> float:
        lo, hi = self.interval(point)
        return (hi - lo) / 2.0

    def relative_half_width(self, point: float) -> float:
        return self.half_width(point) / point * 100.0 if point else math.inf

    def as_dict(self) -> dict[str, object]:
        # q is stored at full precision deliberately: it is the half-width of every
        # interval this model will ever quote, and rounding it would make a reloaded
        # model disagree with the one that was calibrated.
        return {
            "level": self.level,
            "method": self.method.value,
            "q": self.q,
            "n_calibration": self.n_calibration,
            "floor": self.floor,
            "heteroscedasticity_rho": self.heteroscedasticity_rho,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "Conformal":
        return cls(
            level=float(data["level"]),
            method=Nonconformity(data["method"]),
            q=float(data["q"]),
            n_calibration=int(data["n_calibration"]),
            floor=float(data.get("floor", 0.0)),
            heteroscedasticity_rho=float(data.get("heteroscedasticity_rho", 0.0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Conformal":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    level: float = 0.90,
    method: Nonconformity | None = None,
    floor: float = 0.0,
) -> Conformal:
    """Split conformal on a held-out calibration fold.

    The quantile index uses the finite-sample correction ``ceil((n+1)(1-alpha))/n``. With
    small calibration sets this matters: the naive empirical quantile under-covers, and
    under-covering is exactly the failure mode that turns a budget range into a surprise.
    """
    actual, predicted = list(actual), list(predicted)
    n = len(actual)
    if n < 5:
        raise ValueError(f"conformal calibration needs at least 5 points, got {n}")

    rho = 0.0
    if method is None:
        method, rho = choose_nonconformity(actual, predicted)

    if method is Nonconformity.ABSOLUTE:
        scores = [abs(a - p) for a, p in zip(actual, predicted)]
    else:
        scores = [abs(a - p) / abs(p) if p else math.inf for a, p in zip(actual, predicted)]

    k = math.ceil((n + 1) * level)
    if k > n:
        q = max(scores)
    else:
        q = quantile(scores, (k - 1) / (n - 1)) if n > 1 else scores[0]

    return Conformal(
        level=level,
        method=method,
        q=q,
        n_calibration=n,
        floor=floor,
        heteroscedasticity_rho=rho,
    )
