"""Cost models that are **fitted**, and a rule for choosing between them.

The first version of this package asserted its model: `A+` cost twice `A` and
`A++` four times, because the enum said so. Measurement says otherwise — an 8×
increase in scope moved tokens by 1.39×, because a large fixed cost dominates.
A model that cannot represent that is not mis-tuned, it is the wrong shape.

So two things are left to the data rather than decided here.

**The shape.** Four forms are offered:

* ``ConstantModel``      — ``tokens = c``            (signal-blind)
* ``ProportionalModel``  — ``tokens = b·x``          (the old multiplicative assumption)
* ``AffineModel``        — ``tokens = a + b·x``      (fixed overhead + marginal)
* ``PowerModel``         — ``tokens = a·x^b``        (sub- or super-linear)

**The signal.** A record can carry several competing measures of "how much
work" — files read, bytes read, functions written. Which one predicts cost is
not obvious and must not be assumed. Measuring scope in *files* produced slopes
that disagreed 7× across three repositories; measuring the same runs in *bytes*
produced one model that fitted all of them to 2.8%. Had the signal been
hardcoded, that would never have surfaced.

:func:`select_model` fits every (signal, form) pair, scores each by
leave-one-out cross-validation, and returns the winner — ties broken toward
fewer parameters, and toward the plain ``scope`` signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .taxonomy import ScopedRecord, common_signals

#: When a fit is saturated (as many parameters as points) the residual spread is
#: not merely small, it is unmeasurable. Rather than report a zero-width
#: interval implying certainty, fall back to this relative half-width.
SATURATED_HALF_WIDTH = 0.5


@dataclass(frozen=True)
class CostModel:
    """A fitted token-cost curve for one task kind.

    ``scope_min``/``scope_max`` record the range of the *signal* the fit was
    made over, so a caller can tell an interpolation from an extrapolation —
    the mistake the 2×/4× multipliers made silently.
    """

    kind: str
    form: str
    signal: str
    params: tuple[float, ...]
    n: int
    scope_min: float
    scope_max: float
    residual_sigma: float
    saturated: bool = False

    # -- prediction ------------------------------------------------------
    def predict(self, x: float) -> float:
        raise NotImplementedError

    def predict_for(self, record: ScopedRecord) -> float:
        """Predict this record's cost, reading whichever signal was selected."""
        return self.predict(record.value(self.signal))

    def interval(self, x: float, z: float = 1.96) -> tuple[float, float]:
        """A prediction interval from the fit's own residual spread.

        This is the honest width: it reflects how far real runs sat from the
        fitted curve. A saturated fit cannot measure that at all, so it widens
        to a stated fraction rather than claiming precision it does not have.
        """
        mid = self.predict(x)
        margin = z * self.residual_sigma
        if self.saturated:
            margin = max(margin, SATURATED_HALF_WIDTH * mid)
        return max(0.0, mid - margin), mid + margin

    # -- regime ----------------------------------------------------------
    def in_regime(self, x: float) -> bool:
        return self.scope_min <= x <= self.scope_max

    def extrapolation_factor(self, x: float) -> float:
        """How far outside the fitted range this value sits (1.0 = inside)."""
        if self.in_regime(x):
            return 1.0
        if x > self.scope_max:
            return x / self.scope_max if self.scope_max else float("inf")
        return self.scope_min / x if x else float("inf")

    def describe(self) -> str:
        flag = " SATURATED" if self.saturated else ""
        return (f"{self.kind}: {self.equation()}  "
                f"[n={self.n}, {self.signal} {self.scope_min:g}–{self.scope_max:g}, "
                f"σ={self.residual_sigma:,.0f}{flag}]")

    def equation(self) -> str:
        raise NotImplementedError

    def decompose(self, x: float) -> tuple[float, float]:
        """Split the prediction into (per-invocation fixed, work-driven marginal).

        The split is what makes composition predictable: run two tasks in one
        agent and the fixed part is paid once, not twice. Forms that cannot
        separate the two report everything as marginal, the conservative answer.
        """
        return 0.0, self.predict(x)


@dataclass(frozen=True)
class ConstantModel(CostModel):
    def predict(self, x: float) -> float:
        return self.params[0]

    def decompose(self, x: float) -> tuple[float, float]:
        # the signal carried no information, so all of it behaves as fixed cost
        return self.params[0], 0.0

    def equation(self) -> str:
        return f"tokens = {self.params[0]:,.0f}"


@dataclass(frozen=True)
class ProportionalModel(CostModel):
    def predict(self, x: float) -> float:
        return self.params[0] * x

    def equation(self) -> str:
        return f"tokens = {self.params[0]:,.4g} × {self.signal}"


@dataclass(frozen=True)
class AffineModel(CostModel):
    def predict(self, x: float) -> float:
        a, b = self.params
        return a + b * x

    @property
    def fixed(self) -> float:
        return self.params[0]

    @property
    def marginal(self) -> float:
        return self.params[1]

    def decompose(self, x: float) -> tuple[float, float]:
        a, b = self.params
        return a, b * x

    def equation(self) -> str:
        a, b = self.params
        return f"tokens = {a:,.0f} + {b:,.4g} × {self.signal}"


@dataclass(frozen=True)
class PowerModel(CostModel):
    def predict(self, x: float) -> float:
        a, b = self.params
        if x <= 0:                     # x**negative would divide by zero
            return a if b >= 0 else float("inf")
        return a * (x ** b)

    def equation(self) -> str:
        a, b = self.params
        return f"tokens = {a:,.4g} × {self.signal}^{b:.2f}"


# ── fitters ──────────────────────────────────────────────────────────────
# Each returns None when the data cannot support that form on that signal.

def _xy(records: Sequence[ScopedRecord], signal: str) -> tuple[list, list]:
    return ([r.value(signal) for r in records],
            [float(r.tokens) for r in records])


def _sigma(ys: Sequence[float], preds: Sequence[float], n_params: int) -> tuple[float, bool]:
    """Residual spread, and whether the fit was saturated (spread unmeasurable)."""
    n = len(ys)
    if n <= n_params:
        mean = sum(ys) / n if n else 0.0
        return SATURATED_HALF_WIDTH * mean / 1.96, True
    resid = math.sqrt(sum((y - p) ** 2 for y, p in zip(ys, preds)) / (n - n_params))
    return resid, False


def fit_constant(kind: str, records: Sequence[ScopedRecord],
                 signal: str = "scope") -> Optional[ConstantModel]:
    if not records:
        return None
    xs, ys = _xy(records, signal)
    c = sum(ys) / len(ys)
    sig, sat = _sigma(ys, [c] * len(ys), 1)
    return ConstantModel(kind, "constant", signal, (c,), len(ys),
                         min(xs), max(xs), sig, sat)


def fit_proportional(kind: str, records: Sequence[ScopedRecord],
                     signal: str = "scope") -> Optional[ProportionalModel]:
    if not records:
        return None
    xs, ys = _xy(records, signal)
    denom = sum(x * x for x in xs)
    if denom == 0:
        return None
    b = sum(x * y for x, y in zip(xs, ys)) / denom
    sig, sat = _sigma(ys, [b * x for x in xs], 1)
    return ProportionalModel(kind, "proportional", signal, (b,), len(ys),
                             min(xs), max(xs), sig, sat)


def fit_affine(kind: str, records: Sequence[ScopedRecord],
               signal: str = "scope") -> Optional[AffineModel]:
    if len(records) < 2:
        return None
    xs, ys = _xy(records, signal)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:                      # no variation in the signal: slope unidentifiable
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sig, sat = _sigma(ys, [a + b * x for x in xs], 2)
    return AffineModel(kind, "affine", signal, (a, b), n,
                       min(xs), max(xs), sig, sat)


def fit_power(kind: str, records: Sequence[ScopedRecord],
              signal: str = "scope") -> Optional[PowerModel]:
    usable = [r for r in records if r.value(signal) > 0 and r.tokens > 0]
    if len(usable) < 2:
        return None
    xs, ys = _xy(usable, signal)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    a = math.exp(my - b * mx)
    sig, sat = _sigma(ys, [a * (x ** b) for x in xs], 2)
    return PowerModel(kind, "power", signal, (a, b), n,
                      min(xs), max(xs), sig, sat)


Fitter = Callable[..., Optional[CostModel]]

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
    return sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs)


#: A form must survive at least this many leave-one-out folds to be scored.
MIN_FOLDS = 2


def loo_mape(kind: str, records: Sequence[ScopedRecord], form: str,
             signal: str = "scope") -> Optional[float]:
    """Leave-one-out cross-validated MAPE for one (form, signal) pair.

    Cross-validation rather than in-sample error, so a flexible form cannot win
    merely by having more parameters to bend.

    A fold that fails to fit is *skipped*, not fatal. Returning ``None`` for the
    whole form on one degenerate split would drop it from the comparison
    entirely, handing selection to a worse-fitting but always-fittable rival.
    """
    fitter, _, min_n = FORMS[form]
    n = len(records)
    if n < min_n + 1:
        return None
    errs = []
    for i in range(n):
        rest = list(records[:i]) + list(records[i + 1:])
        model = fitter(kind, rest, signal)
        if model is None:
            continue
        actual = float(records[i].tokens)
        if not actual:
            continue
        errs.append(abs(actual - model.predict_for(records[i])) / actual)
    if len(errs) < MIN_FOLDS:
        return None
    return sum(errs) / len(errs)


@dataclass(frozen=True)
class Selection:
    """The outcome of choosing a signal and a form for one kind."""

    kind: str
    model: CostModel
    scores: dict[str, float]          # "form@signal" -> LOO MAPE
    reason: str

    @property
    def form(self) -> str:
        return self.model.form

    @property
    def signal(self) -> str:
        return self.model.signal

    def scores_for_signal(self, signal: str) -> dict[str, float]:
        return {k.split("@")[0]: v for k, v in self.scores.items()
                if k.endswith("@" + signal)}


def select_model(kind: str, records: Sequence[ScopedRecord]) -> Optional[Selection]:
    """Fit every viable (signal, form) pair, score by LOO CV, return the best.

    Ties and unscorable cases fall back to parsimony — fewer parameters, and the
    plain ``scope`` signal. This is what makes the layer *tunable*: feed it
    different data and it can return a different shape **and a different
    explanatory variable**, not merely different coefficients.
    """
    records = list(records)
    if not records:
        return None

    signals = common_signals(records) or ["scope"]
    fitted: dict[str, CostModel] = {}
    for signal in signals:
        for form, (fitter, _, min_n) in FORMS.items():
            if len(records) >= min_n:
                m = fitter(kind, records, signal)
                if m is not None:
                    fitted[f"{form}@{signal}"] = m
    if not fitted:
        return None

    scores = {}
    for key, model in fitted.items():
        s = loo_mape(kind, records, model.form, model.signal)
        if s is not None and math.isfinite(s):
            scores[key] = s

    def parsimony(key: str) -> tuple:
        form, signal = key.split("@")
        return (FORMS[form][1], 0 if signal == "scope" else 1)

    if scores:
        best = min(scores, key=lambda k: (round(scores[k], 6),) + parsimony(k))
        model = fitted[best]
        reason = (f"lowest leave-one-out MAPE ({scores[best]:.1%}) of "
                  f"{len(scores)} fitted (form, signal) pairs")
    else:
        best = min(fitted, key=parsimony)
        model = fitted[best]
        reason = (f"too few records to cross-validate (n={len(records)}); "
                  f"fell back to the most parsimonious fit")
    return Selection(kind, model, scores, reason)


def group_by_kind(records: Iterable[ScopedRecord]) -> dict[str, list[ScopedRecord]]:
    out: dict[str, list[ScopedRecord]] = {}
    for r in records:
        out.setdefault(r.kind, []).append(r)
    return out
