"""Auditing the reference campaign.

This module reproduces the published leave-one-out table from the reference repository's
committed data -- and then, in the column beside it, scores the same six forms on the
*variable portion*, with the reference's own null probe as the start-up toll.

That second column is the entire reason this runs. A leave-one-out MAPE of 2.55% on a
target where the empty task costs 29,821 tokens and a median task costs about 33,000 is
mostly a measurement of how well a constant predicts a constant. Reproducing the first
column proves we transcribed the arithmetic correctly. It proves nothing about prediction,
and it authorises nothing -- which is why this is a check, not a gate.

Everything here is stamped ``replay`` and can never move a gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Sequence

from assay.evidence import EvidenceClass
from assay.linalg import solve_ols
from assay.metrics import mape, wape

REFERENCE_BRICKS = (
    "review", "extract", "classify", "retrieve",
    "reconcile", "draft", "remediate", "validate", "report",
)

# The six nested forms the reference scored, in its own order.
REFERENCE_FORMS = ("constant", "units", "bytes", "bytes_units", "per_brick", "bytes_per_brick")


@dataclass
class ReferenceRow:
    label: str
    counts: dict[str, int]
    context_bytes: int
    tokens: float
    held_out: bool
    provenance: str
    model: str

    @property
    def total_units(self) -> int:
        return sum(self.counts.values())


def load_reference(path: str | Path) -> list[ReferenceRow]:
    rows: list[ReferenceRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(
            ReferenceRow(
                label=raw["label"],
                counts={k: int(v) for k, v in raw["counts"].items()},
                context_bytes=int(raw["context_bytes"]),
                tokens=float(raw["tokens"]),
                held_out=bool(raw.get("held_out", False)),
                provenance=str(raw.get("provenance", "unknown")),
                model=str(raw.get("model", "unknown")),
            )
        )
    return rows


def _row_features(row: ReferenceRow, form: str) -> list[float]:
    x: list[float] = [1.0]
    if form in ("bytes", "bytes_units", "bytes_per_brick"):
        x.append(float(row.context_bytes))
    if form in ("units", "bytes_units"):
        x.append(float(row.total_units))
    if form in ("per_brick", "bytes_per_brick"):
        x.extend(float(row.counts.get(b, 0)) for b in REFERENCE_BRICKS)
    return x


def loo_predictions(rows: Sequence[ReferenceRow], form: str) -> list[float]:
    """Leave-one-out, exactly as the reference described it."""
    preds: list[float] = []
    for i in range(len(rows)):
        train = [r for j, r in enumerate(rows) if j != i]
        X = [_row_features(r, form) for r in train]
        y = [r.tokens for r in train]
        try:
            beta = solve_ols(X, y)
        except ZeroDivisionError:
            preds.append(sum(y) / len(y))
            continue
        preds.append(sum(b * v for b, v in zip(beta, _row_features(rows[i], form))))
    return preds


@dataclass
class AuditRow:
    form: str
    loo_mape_total: float
    loo_vwape: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "form": self.form,
            "loo_mape_total_pct": round(self.loo_mape_total, 3),
            "loo_vwape_pct": round(self.loo_vwape, 3),
        }


@dataclass
class AuditReport:
    boot: float
    n_fitted: int
    n_held_out: int
    rows: list[AuditRow]
    evidence_class: EvidenceClass = EvidenceClass.REPLAY
    notes: list[str] = field(default_factory=list)

    def best_total(self) -> AuditRow:
        return min(self.rows, key=lambda r: r.loo_mape_total)

    def best_variable(self) -> AuditRow:
        return min(self.rows, key=lambda r: r.loo_vwape)

    def constant(self) -> AuditRow:
        return next(r for r in self.rows if r.form == "constant")

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_class": self.evidence_class.value,
            "boot_from_null_probes": round(self.boot, 1),
            "n_fitted": self.n_fitted,
            "n_held_out": self.n_held_out,
            "forms": [r.as_dict() for r in self.rows],
            "best_on_total_error": self.best_total().form,
            "best_on_variable_portion": self.best_variable().form,
            "notes": self.notes,
        }

    def render(self) -> str:
        head = [
            "REFERENCE AUDIT  (evidence class: replay -- reference only, decides nothing)",
            "",
            f"  null probe / start-up toll : {self.boot:,.0f} tokens",
            f"  rows fitted                : {self.n_fitted}   held out: {self.n_held_out}",
            "",
            f"  {'form':<22}{'LOO MAPE (total)':>20}{'LOO WAPE (variable)':>22}",
            f"  {'-' * 62}",
        ]
        body = [
            f"  {r.form:<22}{r.loo_mape_total:>19.2f}%{r.loo_vwape:>21.2f}%"
            for r in self.rows
        ]
        best_t, best_v = self.best_total(), self.best_variable()
        tail = [
            "",
            f"  Best on total error     : {best_t.form} ({best_t.loo_mape_total:.2f}%)",
            f"  Best on variable portion: {best_v.form} ({best_v.loo_vwape:.2f}%)",
            "",
            "  The left column is the published result and is a transcription check.",
            "  The right column is what those same models know once the start-up toll --"
            f" {self.boot:,.0f} tokens, paid whatever you ask -- is taken off both sides.",
            f"  A constant scores {self.constant().loo_mape_total:.2f}% on the left.",
            "",
            *(f"  note: {n}" for n in self.notes),
            "",
            "  Neither column authorises anything. This is a different runtime.",
        ]
        return "\n".join(head + body + tail)


def audit(rows: Sequence[ReferenceRow]) -> AuditReport:
    fitted = [r for r in rows if not r.held_out]
    nulls = [r.tokens for r in fitted if r.total_units == 0 and r.context_bytes == 0]
    boot = median(nulls) if nulls else min(r.tokens for r in fitted)

    audit_rows: list[AuditRow] = []
    actual = [r.tokens for r in fitted]
    for form in REFERENCE_FORMS:
        preds = loo_predictions(fitted, form)
        audit_rows.append(
            AuditRow(
                form=form,
                loo_mape_total=mape(actual, preds),
                loo_vwape=wape([a - boot for a in actual], [p - boot for p in preds]),
            )
        )

    notes = []
    if not nulls:
        notes.append("no null probe found; the toll was approximated by the cheapest run")
    elif len(nulls) > 1:
        # The reference publishes 29,821 -- the first null probe. There are two, and the
        # median of the replicates is the honest estimate, so the toll printed here is
        # slightly below the published headline. Saying so stops the two numbers reading
        # as a transcription error.
        notes.append(
            f"{len(nulls)} null probes ({', '.join(f'{n:,.0f}' for n in sorted(nulls))}); "
            f"the toll is their median, {boot:,.0f}, not the single published figure "
            f"{max(nulls):,.0f}"
        )
    notes.append(
        "the reference's rows record a single scalar token count with no input/output "
        "split, so no dollar figure can be recovered from them at all"
    )
    return AuditReport(
        boot=boot,
        n_fitted=len(fitted),
        n_held_out=len(rows) - len(fitted),
        rows=audit_rows,
        notes=notes,
    )
