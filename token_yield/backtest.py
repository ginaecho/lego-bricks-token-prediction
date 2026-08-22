"""Validation: is the fitted model any good, and how would we know?

A cost model can always be fitted. The question is whether it *predicts*, and
the honest way to ask is to score it on records it was not fitted on and then
compare that error to something meaningful.

The reference point used here is the **noise floor**: run the identical task
twice and the token counts still differ. That spread is irreducible — no model
can predict a kind more precisely than the same task repeated predicts itself.
So the interesting quantity is not raw error but the *skill ratio*:

    skill = cross-validated MAPE / noise floor

At ``skill ≈ 1`` the model is as good as the process allows and more data will
not help; the way forward is reducing variance, not fitting harder. Well above
1 and there is real signal still on the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .costmodel import FORMS, loo_mape, select_model
from .taxonomy import ScopedRecord


def noise_floor(records: Sequence[ScopedRecord]) -> Optional[float]:
    """Pooled within-(kind, scope) relative spread — the irreducible error.

    Pools across every replicate group so a handful of small groups still give
    a usable estimate. Returns ``None`` when nothing was ever repeated, which
    is itself the finding: without replicates there is no way to tell model
    error from run-to-run noise.
    """
    groups: dict[tuple[str, float], list[float]] = {}
    for r in records:
        groups.setdefault((r.kind, r.scope), []).append(float(r.tokens))

    num = den = 0.0
    grand: list[float] = []
    for vals in groups.values():
        n = len(vals)
        if n < 2:
            continue
        m = sum(vals) / n
        var = sum((v - m) ** 2 for v in vals) / (n - 1)   # sample variance
        num += (n - 1) * var
        den += n - 1
        grand.extend(vals)
    if den == 0 or not grand:
        return None
    pooled_sd = math.sqrt(num / den)
    mean = sum(grand) / len(grand)
    return pooled_sd / mean if mean else None


@dataclass(frozen=True)
class KindReport:
    """How well each candidate form predicts one kind, against the noise floor."""

    kind: str
    n: int
    chosen_form: str
    scores: dict[str, float]
    floor: Optional[float]

    @property
    def best_mape(self) -> Optional[float]:
        return self.scores.get(self.chosen_form)

    @property
    def skill_ratio(self) -> Optional[float]:
        """Cross-validated error as a multiple of the irreducible noise."""
        if self.floor is None or not self.floor or self.best_mape is None:
            return None
        return self.best_mape / self.floor

    @property
    def verdict(self) -> str:
        s = self.skill_ratio
        if s is None:
            return "no replicates — cannot separate model error from noise"
        if s <= 1.25:
            return "at the noise floor — more data will not help; reduce variance instead"
        if s <= 2.0:
            return "close to the noise floor — modest headroom"
        return "well above the noise floor — real signal is still unmodelled"

    def summary(self) -> str:
        mape = f"{self.best_mape:.1%}" if self.best_mape is not None else "n/a"
        floor = f"{self.floor:.1%}" if self.floor is not None else "n/a"
        skill = f"{self.skill_ratio:.2f}×" if self.skill_ratio is not None else "n/a"
        return (f"{self.kind}: form={self.chosen_form}, n={self.n}, "
                f"LOO MAPE {mape} vs floor {floor} (skill {skill}) — {self.verdict}")


def backtest_kind(kind: str, records: Sequence[ScopedRecord]) -> Optional[KindReport]:
    rs = [r for r in records if r.kind == kind]
    if not rs:
        return None
    sel = select_model(kind, rs)
    if sel is None:
        return None
    scores = {}
    for form in FORMS:
        s = loo_mape(kind, rs, form)
        if s is not None and math.isfinite(s):
            scores[form] = s
    return KindReport(kind, len(rs), sel.form, scores, noise_floor(rs))


def backtest(records: Iterable[ScopedRecord]) -> dict[str, KindReport]:
    rs = list(records)
    out = {}
    for kind in sorted({r.kind for r in rs}):
        rep = backtest_kind(kind, rs)
        if rep is not None:
            out[kind] = rep
    return out


def learning_curve(kind: str, records: Sequence[ScopedRecord]) -> list[tuple[int, float]]:
    """Cross-validated error as records accumulate: does more data help?

    Answers the question the loop exists to answer — *is it still worth
    measuring more of this kind?* A curve that has gone flat says stop.
    """
    rs = [r for r in records if r.kind == kind]
    out = []
    for n in range(3, len(rs) + 1):
        prefix = rs[:n]
        sel = select_model(kind, prefix)
        if sel is None:
            continue
        score = loo_mape(kind, prefix, sel.form)
        if score is not None and math.isfinite(score):
            out.append((n, score))
    return out


def report(records: Iterable[ScopedRecord]) -> str:
    rs = list(records)
    lines = ["Backtest — cross-validated error vs the noise floor", "=" * 72]
    floor = noise_floor(rs)
    lines.append(f"Overall noise floor (pooled replicate spread): "
                 + (f"{floor:.1%}" if floor is not None else "n/a — no replicates"))
    lines.append("")
    for kind, rep in backtest(rs).items():
        lines.append("  " + rep.summary())
        ranked = sorted(rep.scores.items(), key=lambda kv: kv[1])
        lines.append("      by form: " + ", ".join(f"{f} {s:.1%}" for f, s in ranked))
    return "\n".join(lines)
