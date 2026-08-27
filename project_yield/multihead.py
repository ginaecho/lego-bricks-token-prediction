"""Fitting one head per outcome, each choosing its own shape from the data.

:mod:`token_yield.compose` fits a single model: brick counts in, tokens out,
functional form selected by leave-one-out cross-validation. This module does the
same thing six times over — for contract value, success rate, three staffing
lines and elapsed time — and keeps the same rule that makes the token model
trustworthy: **a richer form is only selected if it predicts engagements it was
not fitted on better than a simpler one does.**

Two departures from the token model, both forced by what is being predicted:

*Every head is scored, and every head reports the score.* A staffing estimate at
30% error and a win probability at 30% error are different products. Continuous
heads are scored by leave-one-out MAPE; the binary head by leave-one-out Brier
score against a base-rate baseline, because MAPE on a probability is meaningless
and a probability model that cannot beat "everything succeeds at the historical
average" is worse than the average.

*Leave-one-out is computed in closed form.* For a least-squares fit the deletion
residual is exactly ``eᵢ / (1 - hᵢ)``, so honest cross-validation over seven
forms and six heads costs seven fits, not seven hundred. The logistic head uses
the standard one-step deletion approximation of the same identity — it is an
approximation, it is labelled as one, and it errs toward *pessimism* about the
model rather than flattery.

Intervals
---------
Every estimate is returned as a band, not a point. The band is the empirical
10th-to-90th percentile of the head's own leave-one-out residuals, applied in
the fitting space and transformed back. No normality is assumed: with a hundred
or so engagements the empirical quantiles are the more honest instrument, and
they carry the corpus's real skew — which for contract value is considerable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import features as F
from .features import FeatureRow
from .linalg import (LOGIT_RIDGE, irls_logistic, least_squares, leverage,
                     quantile, sigmoid, standard_error)
from .outcomes import ORDER, OUTCOMES, Outcome

#: Central interval reported for every estimate. 80% is the band an estimator
#: can actually stand behind at this sample size; 95% would be mostly extrapolation.
INTERVAL = 0.80
_Z80 = 1.2815515655446004


@dataclass(frozen=True)
class Estimate:
    """One head's answer for one use case."""

    outcome: str
    value: float
    low: float
    high: float
    unit: str

    def format(self) -> str:
        o = OUTCOMES[self.outcome]
        return f"{o.format(self.value)}  ({o.format(self.low)} – {o.format(self.high)})"


@dataclass
class Head:
    """A fitted model for one outcome."""

    outcome: Outcome
    form: str
    coef: List[float]
    n: int
    #: LOO MAPE for continuous heads; LOO Brier score for the binary head.
    loo_score: float
    in_sample_score: float
    #: The same metric for the do-nothing model — the corpus mean, or base rate.
    baseline_score: float
    scores: Dict[str, float] = field(default_factory=dict)
    resid_low: float = 0.0
    resid_high: float = 0.0
    _design: List[List[float]] = field(default_factory=list, repr=False)
    _weights: List[float] = field(default_factory=list, repr=False)

    # -- prediction ------------------------------------------------------

    def _eta(self, row: FeatureRow) -> float:
        x = F.build(self.form, row)
        return sum(c * f for c, f in zip(self.coef, x))

    def predict(self, row: FeatureRow) -> float:
        return self.outcome.link.inverse(self._eta(row))

    def interval(self, row: FeatureRow) -> Tuple[float, float]:
        """The 80% band, in the observed units."""
        eta = self._eta(row)
        if self.outcome.binary:
            # For a probability the uncertainty that matters is in the fit, not
            # in a residual: a 0/1 outcome has no residual spread to quantile.
            se = standard_error(self._design, self._weights,
                                F.build(self.form, row), LOGIT_RIDGE)
            lo, hi = eta - _Z80 * se, eta + _Z80 * se
        else:
            lo, hi = eta + self.resid_low, eta + self.resid_high
        inv = self.outcome.link.inverse
        return inv(lo), inv(hi)

    def estimate(self, row: FeatureRow) -> Estimate:
        lo, hi = self.interval(row)
        return Estimate(self.outcome.slug, self.predict(row), lo, hi,
                        self.outcome.unit)

    # -- interpretation --------------------------------------------------

    def contributions(self, row: FeatureRow) -> List[Tuple[str, float, float]]:
        """``(feature name, value, contribution)`` in the fitting space.

        A forecast a buyer cannot interrogate is a guess with a decimal point,
        so every head can show its terms — the same obligation
        :func:`token_yield.decompose.explain` meets for the token model.
        """
        x = F.build(self.form, row)
        return [(name, xi, c * xi)
                for name, xi, c in zip(F.names(self.form), x, self.coef)]

    @property
    def beats_baseline(self) -> bool:
        return self.loo_score < self.baseline_score

    @property
    def skill(self) -> float:
        """Fraction of the baseline's error removed; negative means harmful."""
        if self.baseline_score <= 0:
            return 0.0
        return (self.baseline_score - self.loo_score) / self.baseline_score

    def describe(self) -> str:
        metric = "Brier" if self.outcome.binary else "MAPE"
        verdict = "beats" if self.beats_baseline else "LOSES TO"
        return (f"{self.outcome.name:<18} form={self.form:<32} "
                f"LOO {metric} {self.loo_score:.3f} ({verdict} baseline "
                f"{self.baseline_score:.3f}, skill {self.skill:+.0%}), n={self.n}")


# ── scoring ──────────────────────────────────────────────────────────────

def _loo_continuous(form: str, rows: Sequence[FeatureRow],
                    y_obs: Sequence[float], outcome: Outcome
                    ) -> Tuple[float, List[float]]:
    """Exact leave-one-out for a least-squares head, via the PRESS identity."""
    x = F.design(form, rows)
    z = [outcome.link.forward(v) for v in y_obs]
    coef = least_squares(x, z)
    h = leverage(x)
    inv = outcome.link.inverse
    errs, loo_resid = [], []
    for i, row_x in enumerate(x):
        fitted = sum(c * f for c, f in zip(coef, row_x))
        deletion = (z[i] - fitted) / (1.0 - h[i])
        loo_resid.append(deletion)
        pred = inv(z[i] - deletion)
        actual = float(y_obs[i])
        errs.append(abs(pred - actual) / actual if actual else 0.0)
    return sum(errs) / len(errs), loo_resid


def _loo_binary(form: str, rows: Sequence[FeatureRow],
                y_obs: Sequence[float]) -> Tuple[float, List[float]]:
    """One-step leave-one-out for the logistic head, scored by Brier."""
    x = F.design(form, rows)
    y = [float(v) for v in y_obs]
    coef = irls_logistic(x, y)
    eta = [sum(c * f for c, f in zip(coef, xi)) for xi in x]
    p = [sigmoid(e) for e in eta]
    w = [max(pi * (1.0 - pi), 1e-6) for pi in p]
    h = leverage(x, w, LOGIT_RIDGE)
    loo_p = []
    for i in range(len(y)):
        adj = h[i] * (y[i] - p[i]) / (w[i] * (1.0 - h[i]))
        loo_p.append(sigmoid(eta[i] - adj))
    brier = sum((pi - yi) ** 2 for pi, yi in zip(loo_p, y)) / len(y)
    return brier, loo_p


def _baseline_score(y_obs: Sequence[float], outcome: Outcome) -> float:
    """What you get for free: the corpus mean, or the base rate."""
    if outcome.binary:
        rate = sum(y_obs) / len(y_obs)
        return sum((rate - y) ** 2 for y in y_obs) / len(y_obs)
    # Geometric mean, because the head works in log space and the arithmetic
    # mean of a skewed corpus is a straw man that any model would beat.
    gm = math.exp(sum(math.log(max(y, 1e-9)) for y in y_obs) / len(y_obs))
    return sum(abs(gm - y) / y for y in y_obs if y) / len(y_obs)


# ── fitting one head ─────────────────────────────────────────────────────

def fit_head(outcome: Outcome, rows: Sequence[FeatureRow],
             y_obs: Sequence[float]) -> Head:
    """Fit every candidate form and keep the one that generalises.

    Forms wider than the data are skipped rather than fitted: with fewer rows
    than features the fit is exact, the leave-one-out score is meaningless, and
    the richest form would win every time on a corpus too small to have earned
    it.
    """
    usable = [f for f in F.FORM_ORDER if F.width(f) <= max(2, len(rows) // 4)]
    if not usable:
        usable = ["constant"]

    scores: Dict[str, float] = {}
    residuals: Dict[str, List[float]] = {}
    for form in usable:
        if outcome.binary:
            score, aux = _loo_binary(form, rows, y_obs)
        else:
            score, aux = _loo_continuous(form, rows, y_obs, outcome)
        scores[form] = score
        residuals[form] = aux

    best = min(usable, key=lambda f: (round(scores[f], 4), F.width(f),
                                      F.FORM_ORDER.index(f)))

    x = F.design(best, rows)
    if outcome.binary:
        coef = irls_logistic(x, [float(v) for v in y_obs])
        eta = [sum(c * f for c, f in zip(coef, xi)) for xi in x]
        p = [sigmoid(e) for e in eta]
        weights = [max(pi * (1.0 - pi), 1e-6) for pi in p]
        in_sample = sum((pi - float(yi)) ** 2
                        for pi, yi in zip(p, y_obs)) / len(y_obs)
        lo = hi = 0.0
    else:
        z = [outcome.link.forward(v) for v in y_obs]
        coef = least_squares(x, z)
        inv = outcome.link.inverse
        preds = [inv(sum(c * f for c, f in zip(coef, xi))) for xi in x]
        in_sample = sum(abs(pr - float(a)) / float(a)
                        for pr, a in zip(preds, y_obs) if a) / len(y_obs)
        weights = [1.0] * len(rows)
        lo = quantile(residuals[best], (1.0 - INTERVAL) / 2.0)
        hi = quantile(residuals[best], 1.0 - (1.0 - INTERVAL) / 2.0)

    return Head(
        outcome=outcome, form=best, coef=coef, n=len(rows),
        loo_score=scores[best], in_sample_score=in_sample,
        baseline_score=_baseline_score([float(v) for v in y_obs], outcome),
        scores=scores, resid_low=lo, resid_high=hi,
        _design=x, _weights=weights,
    )


# ── the whole model ──────────────────────────────────────────────────────

@dataclass
class MultiHeadModel:
    """Every outcome head, fitted on the same encoded feature vector."""

    heads: Dict[str, Head]
    n: int

    def __getitem__(self, slug: str) -> Head:
        return self.heads[slug]

    def estimate_all(self, row: FeatureRow) -> Dict[str, Estimate]:
        return {slug: self.heads[slug].estimate(row) for slug in ORDER
                if slug in self.heads}

    def report(self) -> str:
        lines = ["Fitted outcome heads",
                 "=" * 74,
                 f"  corpus: {self.n} engagements, "
                 f"{len(F.FORM_ORDER)} candidate forms per head", ""]
        for slug in ORDER:
            head = self.heads.get(slug)
            if head is None:
                lines.append(f"  {slug}: not fitted")
                continue
            lines.append("  " + head.describe())
        weak = [h.outcome.name for h in self.heads.values()
                if not h.beats_baseline]
        if weak:
            lines += ["", "  Heads that do NOT beat their baseline: "
                          + ", ".join(weak),
                      "  Report these as the base rate, not as a prediction."]
        return "\n".join(lines)


def fit(rows: Sequence[FeatureRow],
        observations: Dict[str, Sequence[float]]) -> MultiHeadModel:
    """Fit one head per outcome from aligned rows and observed values."""
    heads: Dict[str, Head] = {}
    for slug in ORDER:
        y_obs = observations.get(slug)
        if not y_obs:
            continue
        if len(y_obs) != len(rows):
            raise ValueError(f"{slug}: {len(y_obs)} observations for "
                             f"{len(rows)} rows")
        heads[slug] = fit_head(OUTCOMES[slug], rows, y_obs)
    return MultiHeadModel(heads=heads, n=len(rows))
