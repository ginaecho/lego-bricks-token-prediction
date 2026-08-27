"""The feature vector every head is fitted on, and the candidate forms.

The input is the same encoded object the token model uses — a multiset of task
bricks plus the size of the material in scope — extended with the two things a
delivery outcome depends on that a token count does not: **who the work is for**
(industry, business goal) and **what it inherits** (lineage).

Which of those actually earn their place is not decided here. Seven candidate
forms are offered, from a bare constant up to bricks-plus-lineage-plus-context,
and :mod:`project_yield.multihead` picks per head by cross-validation. A head is
allowed to conclude that industry is noise, or that lineage is worth more than
the brick mix. That is the point: the same discipline
:mod:`token_yield.compose` applies to the token model, applied to money, risk,
staffing and time.

Why counts are compressed
-------------------------
Every head here has a log or logit link, so a feature entering linearly enters
the *observed* quantity exponentially — and an estimate that doubles the price
for ten extra documents is not wrong at the margin, it is wrong in a way that
makes the tool unusable at scale. Count-like features are therefore carried as
``log1p``, which makes the fitted relationship a power law: cost grows with
scope, sublinearly, with the exponent fitted rather than assumed. That is the
shape effort estimation has used since COCOMO, and it is the shape the batching
result in the token model already implies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from .lineage import GREENFIELD, LineageFeatures
from .usecase import BRICKS, GOALS, INDUSTRIES


@dataclass(frozen=True)
class FeatureRow:
    """Everything a head is allowed to look at, for one use case."""

    counts: Dict[str, int]
    context_bytes: int
    industry: str
    goal: str
    lineage: LineageFeatures = GREENFIELD

    @property
    def total_units(self) -> int:
        return sum(self.counts.values())

    def share(self, slug: str) -> float:
        total = self.total_units
        return (self.counts.get(slug, 0) / total) if total else 0.0


def row_of(subject, lineage: LineageFeatures = GREENFIELD) -> FeatureRow:
    """Build a feature row from a :class:`UseCase` or an ``Engagement``."""
    return FeatureRow(
        counts=dict(subject.counts), context_bytes=int(subject.context_bytes),
        industry=subject.industry, goal=subject.goal, lineage=lineage,
    )


# ── feature blocks ───────────────────────────────────────────────────────

def _size(r: FeatureRow) -> List[float]:
    return [math.log1p(r.total_units)]


def _bytes(r: FeatureRow) -> List[float]:
    return [math.log1p(r.context_bytes)]


def _brick_counts(r: FeatureRow) -> List[float]:
    return [math.log1p(r.counts.get(s, 0)) for s in BRICKS]


def _brick_mix(r: FeatureRow) -> List[float]:
    # The last share is dropped: shares sum to one, so keeping all nine would
    # make the block collinear with the intercept and the coefficients
    # uninterpretable even where the prediction is fine.
    return [r.share(s) for s in BRICKS[:-1]]


def _lineage(r: FeatureRow) -> List[float]:
    lin = r.lineage
    return [float(lin.reuse_depth), math.log1p(lin.sibling_count),
            float(lin.inherited_fraction)]


def _context(r: FeatureRow) -> List[float]:
    # First level of each is the reference category, absorbed by the intercept.
    return ([1.0 if r.industry == i else 0.0 for i in INDUSTRIES[1:]]
            + [1.0 if r.goal == g else 0.0 for g in GOALS[1:]])


_BLOCK_NAMES: Dict[str, List[str]] = {
    "size": ["log units"],
    "bytes": ["log context bytes"],
    "bricks": [f"log {s}" for s in BRICKS],
    "mix": [f"{s} share" for s in BRICKS[:-1]],
    "lineage": ["reuse depth", "log siblings", "inherited fraction"],
    "context": [f"industry={i}" for i in INDUSTRIES[1:]]
               + [f"goal={g}" for g in GOALS[1:]],
}

_BLOCKS: Dict[str, Callable[[FeatureRow], List[float]]] = {
    "size": _size, "bytes": _bytes, "bricks": _brick_counts,
    "mix": _brick_mix, "lineage": _lineage, "context": _context,
}


# ── the candidate forms ──────────────────────────────────────────────────
#
# Ordered from fewest features to most. Ties in cross-validated score break
# toward the earlier entry, so a richer form is only ever selected when it
# genuinely predicts held-out engagements better.

FORMS: Dict[str, Tuple[str, ...]] = {
    "constant": (),
    "size": ("size",),
    "size+bytes": ("size", "bytes"),
    "size+bytes+lineage": ("size", "bytes", "lineage"),
    "bricks+bytes": ("bricks", "bytes"),
    "size+mix+bytes+lineage": ("size", "mix", "bytes", "lineage"),
    "size+mix+bytes+lineage+context": ("size", "mix", "bytes", "lineage",
                                       "context"),
}

FORM_ORDER: Tuple[str, ...] = tuple(FORMS)


def build(form: str, r: FeatureRow) -> List[float]:
    """The feature vector for one row under one form; always intercept-first."""
    out = [1.0]
    for block in FORMS[form]:
        out.extend(_BLOCKS[block](r))
    return out


def names(form: str) -> List[str]:
    """Human-readable names, aligned with :func:`build`."""
    out = ["intercept"]
    for block in FORMS[form]:
        out.extend(_BLOCK_NAMES[block])
    return out


def width(form: str) -> int:
    return len(names(form))


def design(form: str, rows: Sequence[FeatureRow]) -> List[List[float]]:
    return [build(form, r) for r in rows]
