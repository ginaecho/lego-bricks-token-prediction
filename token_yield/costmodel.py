"""Cost models that are **fitted**, and a rule for choosing between them.

The first version of this package asserted its model: `A+` cost twice `A` and
`A++` four times, because the enum said so. Measurement says otherwise — over
the shipped probe suite, an 8× increase in scope moved tokens by 1.39×, because
a large fixed cost dominates. A model that cannot represent that is not
mis-tuned, it is the wrong shape.

So the shape is not chosen here either. Four forms are offered:

* ``ConstantModel``      — ``tokens = c``               (scope-blind)
* ``ProportionalModel``  — ``tokens = b·scope``         (the old multiplicative assumption)
* ``AffineModel``        — ``tokens = a + b·scope``     (fixed overhead + marginal)
* ``PowerModel``         — ``tokens = a·scope^b``       (sub- or super-linear)

:func:`select_model` fits all of them, scores each by leave-one-out
cross-validation, and returns the winner. The data picks the shape; when the
data is too thin to tell, the most parsimonious form wins by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .taxonomy import ScopedRecord


@dataclass(frozen=True)
class CostModel:
    """A fitted token-cost curve for one task kind.

    ``scope_min``/``scope_max`` record the range the fit was made over, so a
    caller can tell an interpolation from an extrapolation — the very mistake
    the 2×/4× multipliers made silently.
    """

    kind: str
    form: str
    params: tuple[float, ...]
    n: int
    scope_min: float
    scope_max: float
    residual_sigma: float

    # -- prediction ------------------------------------------------------
    def predict(self, scope: float) -> float:
        raise NotImplementedError

    def interval(self, scope: float, z: float = 1.96) -> tuple[float, float]:
        """A prediction interval from the fit's own residual spread.

        This is the honest width: it reflects how far real runs sat from the
        fitted curve, not how tightly the mean of a sample is known.
        """
        mid = self.predict(scope)
        margin = z * self.residual_sigma
        return max(0.0, mid - margin), mid + margin

    # -- regime ----------------------------------------------------------
    def in_regime(self, scope: float) -> bool:
        return self.scope_min <= scope <= self.scope_max

    def extrapolation_factor(self, scope: float) -> float:
        """How far outside the fitted range this scope sits (1.0 = inside)."""
        if self.in_regime(scope):
            return 1.0
        if scope > self.scope_max:
            return scope / self.scope_max if self.scope_max else float("inf")
        return self.scope_min / scope if scope else float("inf")

    def describe(self) -> str:
        return (f"{self.kind}: {self.equation()}  "
                f"[n={self.n}, scope {self.scope_min:g}–{self.scope_max:g}, "
                f"σ={self.residual_sigma:,.0f}]")

    def equation(self) -> str:
        raise NotImplementedError

    def decompose(self, scope: float) -> tuple[float, float]:
        """Split the prediction into (per-invocation fixed, scope-driven marginal).

        The split is what makes composition predictable: run two tasks in one
        agent and the fixed part is paid once, not twice. Forms that cannot
        separate the two report everything as marginal, which is the
        conservative answer.
        """
        return 0.0, self.predict(scope)


@dataclass(frozen=True)
class ConstantModel(CostModel):
    def predict(self, scope: float) -> float:
        return self.params[0]

    def decompose(self, scope: float) -> tuple[float, float]:
        # scope carried no signal, so all of it behaves as per-invocation cost
        return self.params[0], 0.0

    def equation(self) -> str:
        return f"tokens = {self.params[0]:,.0f}"


@dataclass(frozen=True)
class ProportionalModel(CostModel):
    def predict(self, scope: float) -> float:
        return self.params[0] * scope

    def equation(self) -> str:
        return f"tokens = {self.params[0]:,.0f} × scope"


@dataclass(frozen=True)
class AffineModel(CostModel):
    def predict(self, scope: float) -> float:
        a, b = self.params
        return a + b * scope

    @property
    def fixed(self) -> float:
        return self.params[0]

    @property
    def marginal(self) -> float:
        return self.params[1]

    def decompose(self, scope: float) -> tuple[float, float]:
        a, b = self.params
        return a, b * scope

    def equation(self) -> str:
        a, b = self.params
        return f"tokens = {a:,.0f} + {b:,.0f} × scope"


@dataclass(frozen=True)
class PowerModel(CostModel):
    def predict(self, scope: float) -> float:
        a, b = self.params
        return a * (scope ** b)

    def equation(self) -> str:
        a, b = self.params
        return f"tokens = {a:,.0f} × scope^{b:.2f}"


# ── fitters ──────────────────────────────────────────────────────────────
# Each returns None when the data cannot support that form.

def _xy(records: Sequence[ScopedRecord]) -> tuple[list[float], list[float]]:
    return [r.scope for r in records], [float(r.tokens) for r in records]


def _sigma(ys: Sequence[float], preds: Sequence[float], n_params: int) -> float:
    dof = max(len(ys) - n_params, 1)
    return math.sqrt(sum((y - p) ** 2 for y, p in zip(ys, preds)) / dof)


def _bounds(xs: Sequence[float]) -> tuple[float, float]:
    return min(xs), max(xs)


def fit_constant(kind: str, records: Sequence[ScopedRecord]) -> Optional[ConstantModel]:
    if not records:
        return None
    xs, ys = _xy(records)
    c = sum(ys) / len(ys)
    lo, hi = _bounds(xs)
    return ConstantModel(kind, "constant", (c,), len(ys), lo, hi,
                         _sigma(ys, [c] * len(ys), 1))


def fit_proportional(kind: str, records: Sequence[ScopedRecord]) -> Optional[ProportionalModel]:
    if not records:
        return None
    xs, ys = _xy(records)
    denom = sum(x * x for x in xs)
    if denom == 0:
        return None
    b = sum(x * y for x, y in zip(xs, ys)) / denom
    lo, hi = _bounds(xs)
    return ProportionalModel(kind, "proportional", (b,), len(ys), lo, hi,
                             _sigma(ys, [b * x for x in xs], 1))


def fit_affine(kind: str, records: Sequence[ScopedRecord]) -> Optional[AffineModel]:
    if len(records) < 2:
        return None
    xs, ys = _xy(records)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:                      # no variation in scope: slope unidentifiable
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    lo, hi = _bounds(xs)
    return AffineModel(kind, "affine", (a, b), n, lo, hi,
                       _sigma(ys, [a + b * x for x in xs], 2))


def fit_power(kind: str, records: Sequence[ScopedRecord]) -> Optional[PowerModel]:
    usable = [r for r in records if r.scope > 0 and r.tokens > 0]
    if len(usable) < 2:
        return None
    xs, ys = _xy(usable)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    a = math.exp(my - b * mx)
    lo, hi = _bounds(xs)
    return PowerModel(kind, "power", (a, b), n, lo, hi,
                      _sigma(ys, [a * (x ** b) for x in xs], 2))


Fitter = Callable[[str, Sequence[ScopedRecord]], Optional[CostModel]]

#: form name -> (fitter, free parameters, minimum records to fit)
FORMS: dict[str, tuple[Fitter, int, int]] = {
    "constant": (fit_constant, 1, 1),
    "proportional": (fit_proportional, 1, 1),
    "affine": (fit_affine, 2, 2),
    "power": (fit_power, 2, 2),
}


# ── scoring and selection ────────────────────────────────────────────────

def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error, as a fraction (0.05 = 5%)."""
    pairs = [(a, p) for a, p in zip(actual, predicted) if a]
    if not pairs:
        return float("inf")
    return sum(abs(a - p) / a for a, p in pairs) / len(pairs)


def loo_mape(kind: str, records: Sequence[ScopedRecord], form: str) -> Optional[float]:
    """Leave-one-out cross-validated MAPE for one model form.

    Cross-validation rather than in-sample error, so a flexible form cannot win
    merely by having more parameters to bend.
    """
    fitter, _, min_n = FORMS[form]
    n = len(records)
    if n < min_n + 1:
        return None
    errs = []
    for i in range(n):
        rest = list(records[:i]) + list(records[i + 1:])
        model = fitter(kind, rest)
        if model is None:
            return None
        actual = float(records[i].tokens)
        if not actual:
            continue
        errs.append(abs(actual - model.predict(records[i].scope)) / actual)
    return sum(errs) / len(errs) if errs else None


@dataclass(frozen=True)
class Selection:
    """The outcome of choosing a model form for one kind."""

    kind: str
    model: CostModel
    scores: dict[str, float]          # form -> LOO MAPE (only scorable forms)
    reason: str

    @property
    def form(self) -> str:
        return self.model.form


def select_model(kind: str, records: Sequence[ScopedRecord]) -> Optional[Selection]:
    """Fit every viable form, score by LOO CV, return the best.

    Ties and unscorable cases fall back to parsimony: fewer parameters wins.
    This is what makes the layer *tunable* — feed it different data and it can
    return a different shape, not merely different coefficients.
    """
    records = list(records)
    if not records:
        return None

    fitted: dict[str, CostModel] = {}
    for form, (fitter, _, min_n) in FORMS.items():
        if len(records) >= min_n:
            m = fitter(kind, records)
            if m is not None:
                fitted[form] = m
    if not fitted:
        return None

    scores = {}
    for form in fitted:
        s = loo_mape(kind, records, form)
        if s is not None and math.isfinite(s):
            scores[form] = s

    if scores:
        best = min(scores, key=lambda f: (round(scores[f], 6), FORMS[f][1]))
        reason = f"lowest leave-one-out MAPE ({scores[best]:.1%}) of {len(scores)} forms"
    else:
        best = min(fitted, key=lambda f: FORMS[f][1])
        reason = (f"too few records to cross-validate (n={len(records)}); "
                  f"fell back to the most parsimonious form")
    return Selection(kind, fitted[best], scores, reason)


def group_by_kind(records: Iterable[ScopedRecord]) -> dict[str, list[ScopedRecord]]:
    out: dict[str, list[ScopedRecord]] = {}
    for r in records:
        out.setdefault(r.kind, []).append(r)
    return out
