"""The feedback loop: observe real runs, refit, and notice when the model is stale.

This is the part that makes the layer a *learning* one rather than a lookup
table. Three obligations:

* **Absorb.** Every finished task is a new record, whether it came from a probe
  or from production traffic.
* **Refit.** Models are rebuilt from the accumulated record set — including the
  choice of *form*, not just the coefficients.
* **Notice.** New records are scored against the model that was current *before*
  they arrived. Systematic one-sided error is the signal that the old fit no
  longer describes reality, and it is reported rather than quietly averaged
  away.

That last point is the whole discipline. A model that silently absorbs
contradicting data looks healthy forever; one that reports its drift tells you
when to look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .costmodel import CostModel, Selection, group_by_kind, select_model
from .taxonomy import KINDS, Provenance, ScopedRecord


@dataclass(frozen=True)
class DriftReport:
    """How the standing model fared against records it was not fitted on."""

    kind: str
    n_new: int
    mape: float
    bias: float               # mean signed relative error; + means under-predicting
    within_interval: float    # fraction of new records inside the 95% interval
    verdict: str

    @property
    def should_refit(self) -> bool:
        return self.verdict != "stable"

    def summary(self) -> str:
        return (f"{self.kind}: {self.n_new} new records, MAPE {self.mape:.1%}, "
                f"bias {self.bias:+.1%}, {self.within_interval:.0%} inside the "
                f"interval → {self.verdict}")


#: A model under-predicting by more than this, consistently, is stale.
BIAS_THRESHOLD = 0.10
#: Error above this means the model is not describing the data at all.
MAPE_THRESHOLD = 0.25


def score_against(model: CostModel, records: Iterable[ScopedRecord]) -> Optional[DriftReport]:
    """Score a standing model against records it has not seen."""
    rs = [r for r in records if r.kind == model.kind and r.tokens]
    if not rs:
        return None

    rel = [(float(r.tokens) - model.predict(r.scope)) / float(r.tokens) for r in rs]
    mape = sum(abs(e) for e in rel) / len(rel)
    bias = sum(rel) / len(rel)
    inside = sum(1 for r in rs
                 if model.interval(r.scope)[0] <= r.tokens <= model.interval(r.scope)[1])
    coverage = inside / len(rs)

    direction = "under" if bias > 0 else "over"
    if mape > MAPE_THRESHOLD:
        # say which way it is wrong — a budget that is 50% low and one that is
        # 50% high call for opposite actions
        if abs(bias) > BIAS_THRESHOLD:
            verdict = (f"refit: far from the new runs, consistently "
                       f"{direction}-predicting by {abs(bias):.0%}")
        else:
            verdict = "refit: far from the new runs, with no consistent direction"
    elif abs(bias) > BIAS_THRESHOLD:
        verdict = f"refit: consistently {direction}-predicting"
    elif any(not model.in_regime(r.scope) for r in rs):
        verdict = "refit: new runs fall outside the fitted scope range"
    else:
        verdict = "stable"

    return DriftReport(model.kind, len(rs), mape, bias, coverage, verdict)


@dataclass
class LearningStore:
    """Accumulates measured runs and keeps a fitted model per kind current.

    Deliberately not a database. It is the smallest thing that can hold the
    loop's invariant: *the model in force is always the one implied by every
    record seen so far, and any record that surprised it has been reported.*
    """

    records: list[ScopedRecord] = field(default_factory=list)
    _models: dict[str, Selection] = field(default_factory=dict)
    _dirty: set = field(default_factory=set)

    # -- absorb ----------------------------------------------------------
    def observe(self, record: ScopedRecord) -> Optional[DriftReport]:
        """Add one record, scoring it against the standing model first.

        The scoring happens *before* the refit on purpose: once a record is
        folded into the fit, it can no longer surprise it.
        """
        prior = self.model_for(record.kind)
        report = score_against(prior, [record]) if prior else None
        KINDS.ensure(record.kind)
        self.records.append(record)
        self._dirty.add(record.kind)
        return report

    def observe_many(self, records: Iterable[ScopedRecord]) -> dict[str, DriftReport]:
        """Add a batch, scoring the whole batch against the standing models."""
        batch = list(records)
        reports: dict[str, DriftReport] = {}
        for kind, group in group_by_kind(batch).items():
            prior = self.model_for(kind)
            if prior is not None:
                r = score_against(prior, group)
                if r is not None:
                    reports[kind] = r
        for rec in batch:
            KINDS.ensure(rec.kind)
            self.records.append(rec)
            self._dirty.add(rec.kind)
        return reports

    # -- refit -----------------------------------------------------------
    def selection_for(self, kind: str) -> Optional[Selection]:
        if kind in self._dirty or kind not in self._models:
            group = [r for r in self.records if r.kind == kind]
            sel = select_model(kind, group)
            if sel is None:
                return None
            self._models[kind] = sel
            self._dirty.discard(kind)
        return self._models.get(kind)

    def model_for(self, kind: str) -> Optional[CostModel]:
        sel = self.selection_for(kind)
        return sel.model if sel else None

    def refit_all(self) -> dict[str, Selection]:
        return {k: s for k in self.kinds() if (s := self.selection_for(k))}

    # -- inspect ---------------------------------------------------------
    def kinds(self) -> list[str]:
        return sorted({r.kind for r in self.records})

    def records_for(self, kind: str) -> list[ScopedRecord]:
        return [r for r in self.records if r.kind == kind]

    def evidence(self, kind: str) -> dict[str, int]:
        """How many records back this kind, split by where they came from."""
        out = {p.value: 0 for p in Provenance}
        for r in self.records_for(kind):
            out[r.provenance.value] += 1
        return out

    def predict(self, kind: str, scope: float) -> Optional[float]:
        m = self.model_for(kind)
        return m.predict(scope) if m else None

    def report(self) -> str:
        lines = ["Fitted cost models", "=" * 68]
        if not self.records:
            lines.append("  (no records yet)")
            return "\n".join(lines)
        for kind in self.kinds():
            sel = self.selection_for(kind)
            if sel is None:
                lines.append(f"  {kind}: not enough data to fit")
                continue
            lines.append(f"  {sel.model.describe()}")
            lines.append(f"      form chosen: {sel.form} — {sel.reason}")
            if sel.scores:
                ranked = sorted(sel.scores.items(), key=lambda kv: kv[1])
                lines.append("      LOO MAPE by form: "
                             + ", ".join(f"{f} {s:.1%}" for f, s in ranked))
            ev = self.evidence(kind)
            lines.append(f"      evidence: " + ", ".join(
                f"{v} {k}" for k, v in ev.items() if v))
        return "\n".join(lines)


def seeded_store() -> LearningStore:
    """A store preloaded with the shipped probe measurements."""
    from .probes import MEASURED

    store = LearningStore()
    store.observe_many(MEASURED)
    return store
