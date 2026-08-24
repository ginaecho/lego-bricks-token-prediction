"""Fitting what a *combination* of base tasks costs.

:mod:`token_yield.tasks` says what the primitives are. This module answers the
question that makes them useful: if a task is ``2xReview + Reconcile``, what
does it cost — without running it?

The naive answer is to add up the parts. That answer is wrong, and the
measurements say by how much. Every agent invocation pays a large fixed cost
before any work begins; running two primitives in one invocation pays it once,
running them separately pays it twice. So the model is affine in the work, not
proportional to it:

    tokens  =  boot  +  Σ marginal[p] x units[p]  +  slope x context_bytes

``boot`` is measured directly by the null probe — a task that asks for nothing.
The rest is fitted by least squares over the measured campaign.

Which terms actually earn their place is decided by leave-one-out
cross-validation over nested forms, not by assumption. A richer form is only
selected if it predicts *held-out* points better, so the model cannot win by
having more parameters to bend.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .tasks import ORDER, PRIMITIVES


# ── records ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Run:
    """One measured dispatch: what was asked for, and what it cost."""

    label: str
    notation: str
    counts: Dict[str, int]
    context_bytes: int
    arity: int
    tokens: int
    tool_uses: int
    held_out: bool

    @property
    def total_units(self) -> int:
        return sum(self.counts.values())


def load_runs(path: str) -> List[Run]:
    """Read the measured campaign from its JSONL."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Run(d["label"], d["notation"], d["counts"],
                           d["context_bytes"], d["arity"], d["tokens"],
                           d["tool_uses"], d["held_out"]))
    return out


# ── linear algebra, kept small and dependency-free ───────────────────────

def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve ``a x = b`` by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
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


def _least_squares(x: Sequence[Sequence[float]], y: Sequence[float],
                   ridge: float = 1e-6) -> List[float]:
    """Ordinary least squares with a whisper of ridge for conditioning."""
    k = len(x[0])
    xtx = [[sum(r[i] * r[j] for r in x) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * v for r, v in zip(x, y)) for i in range(k)]
    # Ridge must be applied *relative to each column's own scale*. A single
    # global scale would be set by the largest feature (context_bytes, whose
    # squares run to 1e9) and would swamp the intercept column entirely,
    # silently destroying the very term it was meant to stabilise.
    for i in range(k):
        xtx[i][i] += ridge * (abs(xtx[i][i]) or 1.0)
    return _solve(xtx, xty)


# ── candidate model forms ────────────────────────────────────────────────
#
# Each form is a feature builder. They are strictly nested in expressiveness,
# so cross-validation is deciding whether each added term pays for itself.

def _f_constant(r: Run) -> List[float]:
    return [1.0]


def _f_bytes(r: Run) -> List[float]:
    return [1.0, float(r.context_bytes)]


def _f_units(r: Run) -> List[float]:
    return [1.0, float(r.total_units)]


def _f_bytes_units(r: Run) -> List[float]:
    return [1.0, float(r.context_bytes), float(r.total_units)]


def _f_bytes_perprim(r: Run) -> List[float]:
    return ([1.0, float(r.context_bytes)]
            + [float(r.counts.get(p, 0)) for p in ORDER])


def _f_perprim(r: Run) -> List[float]:
    return [1.0] + [float(r.counts.get(p, 0)) for p in ORDER]


FORMS = {
    "constant": _f_constant,
    "bytes": _f_bytes,
    "units": _f_units,
    "bytes+units": _f_bytes_units,
    "per-primitive": _f_perprim,
    "bytes+per-primitive": _f_bytes_perprim,
}


# ── the fitted model ─────────────────────────────────────────────────────

@dataclass
class CompositionModel:
    """A fitted cost model over compositions of base tasks."""

    form: str
    coef: List[float]
    boot: float
    loo_mape: float
    in_sample_mape: float
    n: int
    scores: Dict[str, float] = field(default_factory=dict)

    def predict_run(self, r: Run) -> float:
        feats = FORMS[self.form](r)
        return sum(c * f for c, f in zip(self.coef, feats))

    def predict(self, counts: Dict[str, int], context_bytes: int = 0) -> float:
        """Price a composition that may never have been run."""
        r = Run("?", "?", dict(counts), int(context_bytes), 0, 0, 0, False)
        return self.predict_run(r)

    # -- readable parameters ---------------------------------------------

    def marginals(self) -> Dict[str, float]:
        """Tokens added per extra unit of each primitive, where fitted."""
        if self.form == "bytes+per-primitive":
            return {p: self.coef[2 + i] for i, p in enumerate(ORDER)}
        if self.form == "per-primitive":
            return {p: self.coef[1 + i] for i, p in enumerate(ORDER)}
        return {}

    def byte_slope(self) -> float:
        return self.coef[1] if self.form.startswith("bytes") else 0.0

    def equation(self) -> str:
        bits = [f"{self.coef[0]:,.0f}"]
        if self.form.startswith("bytes"):
            bits.append(f"{self.byte_slope():.4f} x context_bytes")
        for p, m in self.marginals().items():
            if abs(m) >= 1.0:
                bits.append(f"{m:,.0f} x {PRIMITIVES[p].name}")
        if self.form in ("units", "bytes+units"):
            bits.append(f"{self.coef[-1]:,.0f} x units")
        return "tokens = " + " + ".join(bits)


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return sum(abs(a - p) / a for a, p in zip(actual, predicted)) / len(actual)


def _fit_one(form: str, runs: Sequence[Run]) -> List[float]:
    x = [FORMS[form](r) for r in runs]
    y = [float(r.tokens) for r in runs]
    return _least_squares(x, y)


def _loo_mape(form: str, runs: Sequence[Run]) -> float:
    """Leave-one-out error: the only honest way to compare forms."""
    errs = []
    for i in range(len(runs)):
        rest = list(runs[:i]) + list(runs[i + 1:])
        coef = _fit_one(form, rest)
        feats = FORMS[form](runs[i])
        pred = sum(c * f for c, f in zip(coef, feats))
        errs.append(abs(runs[i].tokens - pred) / runs[i].tokens)
    return sum(errs) / len(errs)


def select_model(runs: Sequence[Run]) -> CompositionModel:
    """Fit every candidate form and keep the one that predicts unseen points.

    Ties break toward the simpler form: a model that needs fewer numbers to say
    the same thing is the one that will still be right next quarter.
    """
    fitted = [r for r in runs if not r.held_out]
    scores = {name: _loo_mape(name, fitted) for name in FORMS}
    best = min(FORMS, key=lambda n: (round(scores[n], 4), len(FORMS[n](fitted[0]))))
    coef = _fit_one(best, fitted)
    preds = [sum(c * f for c, f in zip(coef, FORMS[best](r))) for r in fitted]
    boot = next((float(r.tokens) for r in fitted if r.total_units == 0), coef[0])
    return CompositionModel(
        form=best, coef=coef, boot=boot, loo_mape=scores[best],
        in_sample_mape=mape([r.tokens for r in fitted], preds),
        n=len(fitted), scores=scores,
    )


# ── what composition actually buys you ───────────────────────────────────

def batching_saving(model: CompositionModel, counts: Dict[str, int],
                    context_bytes: int = 0) -> Tuple[float, float, float]:
    """One agent doing everything vs one agent per primitive.

    Returns ``(batched, separate, saving_fraction)``. The gap is the boot cost
    paid once instead of once per part — the single biggest lever a buyer has,
    and the one an additive model cannot see.
    """
    batched = model.predict(counts, context_bytes)
    parts = [p for p, n in counts.items() if n]
    separate = sum(model.predict({p: counts[p]}, context_bytes) for p in parts)
    saving = (separate - batched) / separate if separate else 0.0
    return batched, separate, saving


def noise_floor(runs: Sequence[Run]) -> float:
    """Spread between replicate pairs — the error no model can beat."""
    by_shape: Dict[Tuple, List[int]] = {}
    for r in runs:
        key = (tuple(sorted(r.counts.items())), r.context_bytes)
        by_shape.setdefault(key, []).append(r.tokens)
    spreads = []
    for vals in by_shape.values():
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            spreads.append(max(abs(v - m) / m for v in vals))
    return sum(spreads) / len(spreads) if spreads else 0.0


def default_runs_path() -> str:
    """The committed campaign that ships with the package."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "experiments", "train_runs.jsonl")
