"""Forecasting a project from **fitted** cost models.

The tier-based path in :mod:`token_yield.forecast` prices work by asserting a
multiplier. This one prices it by asking the model that was fitted to real
runs, and it carries the two caveats that assertion could not:

* **Unmodelled kinds** — a kind with no fitted model is named, never dropped.
* **Extrapolation** — a scope outside the range the model was fitted over is
  flagged with how far outside it sits. Predicting far past your evidence is
  the original sin this package was built to stop repeating.

Intervals combine the right way. Repeating an item ``n`` times adds *variance*,
not standard deviation, so the band grows with ``√n`` rather than ``n`` — the
tier path's linear scaling overstated uncertainty for repeated work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .costmodel import CostModel
from .learn import LearningStore


@dataclass
class WorkItem:
    """``count`` tasks of ``kind``, each at this ``scope``.

    ``signals`` supplies the other measures a fitted model might have chosen —
    most importantly ``bytes``, which turned out to predict comprehension cost
    far better than a file count.
    """

    kind: str
    scope: float
    count: int = 1
    signals: dict = field(default_factory=dict)

    def value(self, signal: str) -> float:
        if signal == "scope":
            return self.signals.get("scope", self.scope)
        return self.signals.get(signal, 0.0)


@dataclass
class WorkPlan:
    """A project as measurable work: kinds, scopes, and how many of each."""

    name: str
    items: list[WorkItem] = field(default_factory=list)

    def add(self, kind: str, scope: float, count: int = 1,
            **signals: float) -> "WorkPlan":
        self.items.append(WorkItem(kind, scope, count, dict(signals)))
        return self

    @property
    def total_tasks(self) -> int:
        return sum(i.count for i in self.items)


@dataclass(frozen=True)
class LineItem:
    """One priced row of the plan."""

    kind: str
    scope: float
    count: int
    per_task_tokens: float
    tokens: float
    sigma: float
    form: str
    basis_n: int
    extrapolation: float      # 1.0 = inside the fitted range
    signal: str = "scope"

    @property
    def is_extrapolated(self) -> bool:
        return self.extrapolation > 1.0


@dataclass(frozen=True)
class PlanForecast:
    """A budget built from fitted models, with its own caveats attached."""

    plan_name: str
    line_items: tuple[LineItem, ...]
    unmodelled: tuple[str, ...]
    total_tokens: float
    total_sigma: float
    missing_signal: tuple[tuple[str, str], ...] = ()
    """(kind, signal) pairs the plan could not supply, so were left unpriced."""

    @property
    def is_complete(self) -> bool:
        return not self.unmodelled and not self.missing_signal

    @property
    def extrapolated(self) -> tuple[LineItem, ...]:
        return tuple(li for li in self.line_items if li.is_extrapolated)

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        margin = z * self.total_sigma
        return max(0.0, self.total_tokens - margin), self.total_tokens + margin

    def cost_at_rate(self, dollars_per_million_tokens: float) -> float:
        return self.total_tokens * dollars_per_million_tokens / 1_000_000

    def summary(self) -> str:
        lo, hi = self.interval()
        lines = [f"Plan forecast: {self.plan_name}", "=" * 68]
        lines.append(f"{'kind':<18}{'scope':>7}{'n':>5}{'per task':>12}"
                     f"{'tokens':>12}  form")
        for li in self.line_items:
            flag = "  ⚠ extrapolated" if li.is_extrapolated else ""
            lines.append(f"{li.kind:<18}{li.scope:>7g}{li.count:>5}"
                         f"{li.per_task_tokens:>12,.0f}{li.tokens:>12,.0f}"
                         f"  {li.form}{flag}")
        lines.append("-" * 68)
        lines.append(f"{'total':<18}{'':>7}{'':>5}{'':>12}{self.total_tokens:>12,.0f}")
        lines.append(f"95% interval: {lo:,.0f} – {hi:,.0f}")
        if self.unmodelled:
            lines.append("")
            lines.append("!! NOT IN THE TOTAL — no fitted model for: "
                         + ", ".join(self.unmodelled))
        for kind, sig in self.missing_signal:
            lines.append(f"!! NOT IN THE TOTAL — {kind} is priced by '{sig}', "
                         f"which this plan did not supply")
        for li in self.extrapolated:
            lines.append(f"!! {li.kind} @ scope {li.scope:g} is "
                         f"{li.extrapolation:.1f}× outside the fitted range")
        return "\n".join(lines)


class PlanForecaster:
    """Prices a :class:`WorkPlan` using whatever models the store has learned."""

    def __init__(self, store: LearningStore) -> None:
        self._store = store

    def _price(self, item: WorkItem, model: CostModel) -> LineItem:
        x = item.value(model.signal)
        per_task = model.predict(x)
        # A few near-collinear points can fit a line almost exactly and imply an
        # interval of a handful of tokens. Repeating the identical task varies
        # by several percent, so an interval tighter than that claims more
        # precision than the process can deliver. Floor it at the noise floor.
        sigma = model.residual_sigma
        floor = self._store.noise_floor()
        if floor:
            sigma = max(sigma, floor * per_task)
        return LineItem(
            kind=item.kind,
            scope=item.scope,
            count=item.count,
            per_task_tokens=per_task,
            tokens=per_task * item.count,
            sigma=sigma,
            form=model.form,
            basis_n=model.n,
            extrapolation=model.extrapolation_factor(x),
            signal=model.signal,
        )

    def forecast(self, plan: WorkPlan) -> PlanForecast:
        priced: list[LineItem] = []
        unmodelled: list[str] = []

        missing: list = []
        for item in plan.items:
            model = self._store.model_for(item.kind)
            if model is None:
                unmodelled.append(item.kind)
                continue
            # The fitted model reads a particular signal. If the plan does not
            # carry it, predicting anyway would quietly substitute zero and
            # under-price the item — the same silent failure as a missing kind.
            if model.signal != "scope" and item.value(model.signal) <= 0:
                missing.append((item.kind, model.signal))
                continue
            priced.append(self._price(item, model))

        total = sum(li.tokens for li in priced)
        # independent runs: variances add, so the band grows with sqrt(count)
        var = sum(li.count * (li.sigma ** 2) for li in priced)

        return PlanForecast(
            plan_name=plan.name,
            line_items=tuple(priced),
            unmodelled=tuple(dict.fromkeys(unmodelled)),
            total_tokens=total,
            total_sigma=math.sqrt(var),
            missing_signal=tuple(dict.fromkeys(missing)),
        )

    def boot_cost(self) -> Optional[float]:
        """The per-invocation fixed cost shared by every kind.

        Each kind's fitted intercept is ``boot + that kind's own fixed work``:
        a kind that always reads an 11 KB file before starting carries that
        read inside its intercept. So the intercepts are *upper bounds* on the
        shared floor, not independent estimates of it, and averaging them
        overstates it. The smallest observed intercept is the tightest bound
        the evidence supports.
        """
        fixed = []
        for kind in self._store.kinds():
            model = self._store.model_for(kind)
            if model is None:
                continue
            f, _ = model.decompose(model.scope_min)
            if f > 0:
                fixed.append(f)
        return min(fixed) if fixed else None

    def compare_batching(self, plan: WorkPlan) -> Optional[dict]:
        """Cost of running the plan as separate agents vs. batched into one.

        The fixed component is paid **per agent invocation**, so ``n`` separate
        tasks pay it ``n`` times while one batched agent pays it once. That is
        not a refinement — on the measured probe suite it dominates: two tasks
        batched cost 53% of the same two run separately.

        This is why the original ``interaction_overhead = +15%`` was not merely
        mis-tuned. It had the wrong sign.
        """
        boot = self.boot_cost()
        if boot is None:
            return None

        separate = 0.0
        marginal_total = 0.0
        n_tasks = 0
        for item in plan.items:
            model = self._store.model_for(item.kind)
            if model is None:
                continue
            fixed, marginal = model.decompose(item.value(model.signal))
            separate += (fixed + marginal) * item.count
            marginal_total += marginal * item.count
            n_tasks += item.count

        if n_tasks == 0:
            return None
        batched = boot + marginal_total
        return {
            "separate_agents": separate,
            "batched_single_agent": batched,
            "boot_cost": boot,
            "boot_paid_times_if_separate": n_tasks,
            "saving_if_batched": 1 - (batched / separate) if separate else None,
            "caveat": ("validated only for two kinds at scope 3 on the shipped "
                       "probe suite; a long batched context has its own costs "
                       "that this linear picture does not model."),
        }
