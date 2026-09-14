"""Invoices: where the model thought the money went, and where it actually went.

The final ``reconciliation`` line is the point of the whole artefact. It is the residual --
actual cost minus everything the model could account for -- and it always makes the invoice
sum exactly to the measured total. A large reconciliation line is not an embarrassment to
be smoothed away; it is the most informative number on the page, because it says the model
is pricing something other than what is really driving cost.

The invoice also carries ``variable_error_pct`` beside ``error_pct``. A 3% total error
concealing a 42% error on the portion bricks are supposed to explain is the single failure
mode this project exists to avoid, so both numbers travel together or neither does.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from assay.costmodel import CostModel
from assay.evidence import EvidenceClass
from assay.metrics import ape
from assay.quote import Quote
from assay.schemas import Run

TOLERANCE = 1e-6


@dataclass(frozen=True)
class InvoiceLine:
    item: str
    units: float
    detail: str = ""

    def render(self, usd_per_unit: float) -> str:
        cost = self.units * usd_per_unit
        detail = f"  ({self.detail})" if self.detail else ""
        return f"  {self.item:<24} {self.units:>12,.0f} units  ${cost:>9.4f}{detail}"


@dataclass
class Invoice:
    run_id: str
    actual_units: float
    predicted_units: float
    error_pct: float
    variable_error_pct: float
    interval_hit: bool | None
    lines: list[InvoiceLine]
    boot: float
    evidence_class: EvidenceClass
    notes: list[str] = field(default_factory=list)

    @property
    def sum_check(self) -> float:
        return sum(line.units for line in self.lines)

    @property
    def reconciles(self) -> bool:
        return abs(self.sum_check - self.actual_units) < TOLERANCE

    @property
    def residual(self) -> float:
        for line in self.lines:
            if line.item == "reconciliation":
                return line.units
        return 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "actual_units": round(self.actual_units, 4),
            "predicted_units": round(self.predicted_units, 4),
            "error_pct": round(self.error_pct, 4),
            "variable_error_pct": round(self.variable_error_pct, 4),
            "interval_hit": self.interval_hit,
            "boot": round(self.boot, 4),
            "lines": [asdict(line) for line in self.lines],
            "sum_check": round(self.sum_check, 4),
            "reconciles": self.reconciles,
            "evidence_class": self.evidence_class.value,
            "notes": self.notes,
        }

    def render(self, usd_per_unit: float) -> str:
        head = [
            f"INVOICE {self.run_id}",
            f"  quoted {self.predicted_units:,.0f}   actual {self.actual_units:,.0f}   "
            f"error {self.error_pct:.1f}% total / {self.variable_error_pct:.1f}% variable",
            "",
        ]
        body = [line.render(usd_per_unit) for line in self.lines]
        tail = [
            "",
            f"  {'TOTAL':<24} {self.sum_check:>12,.0f} units  "
            f"${self.sum_check * usd_per_unit:>9.4f}",
        ]
        if abs(self.residual) > 0.10 * self.actual_units:
            tail.append(
                "\n  The reconciliation line is more than 10% of the bill: the model is "
                "pricing something other than what drove this run."
            )
        return "\n".join(head + body + tail)


def make_invoice(
    run: Run,
    quote: Quote,
    model: CostModel,
    *,
    boot: float,
    show_per_brick: bool = True,
) -> Invoice:
    """Break the measured cost down the way the model believes it was incurred."""
    lines: list[InvoiceLine] = [
        InvoiceLine("start-up", model.intercept, "fixed toll, paid once per run")
    ]

    if model.bytes_coef:
        lines.append(
            InvoiceLine(
                "context",
                model.bytes_coef * run.context_bytes,
                f"{run.context_bytes:,} bytes x {model.bytes_coef:.4f}",
            )
        )

    notes: list[str] = []
    if show_per_brick:
        for brick, count in run.units.items():
            if not count:
                continue
            marginal = model.marginals.get(brick)
            if marginal is None:
                continue
            lines.append(
                InvoiceLine(f"{brick} x {count}", marginal * count, f"{marginal:,.0f}/unit")
            )
        negative = sorted(
            b for b, c in run.units.items()
            if c and (model.marginals.get(b) or 0.0) < 0.0
        )
        if negative:
            # Deliberately reported rather than clamped. A brick that appears to make a
            # run cheaper is not a discount; it is the fit telling you that coefficient
            # is not identified -- usually because the brick moves with something else in
            # the design. Clamping it to zero would hide exactly that.
            notes.append(
                "negative price on " + ", ".join(negative) + ": a brick cannot reduce "
                "cost, so this coefficient is not identified -- read it as a design "
                "problem, not a discount (the nnls form exists for shipping, not for "
                "explaining)"
            )
    else:
        notes.append(
            "per-brick lines withheld: the verdict did not support per-brick pricing"
        )

    modelled = sum(line.units for line in lines)
    lines.append(
        InvoiceLine(
            "reconciliation",
            run.billable_units - modelled,
            "actual minus everything the model could account for",
        )
    )

    variable_actual = run.billable_units - boot
    variable_predicted = quote.predicted_units - boot
    if abs(variable_actual) < TOLERANCE:
        variable_error = 0.0
        notes.append("this run cost no more than the empty task; variable error is undefined")
    else:
        variable_error = abs(variable_predicted - variable_actual) / abs(variable_actual) * 100.0
    if variable_actual <= 0:
        notes.append(
            "actual fell at or below the measured empty-task cost -- treat as a "
            "measurement failure, not a cheap run"
        )

    return Invoice(
        run_id=run.run_id,
        actual_units=run.billable_units,
        predicted_units=quote.predicted_units,
        error_pct=ape(run.billable_units, quote.predicted_units),
        variable_error_pct=variable_error,
        interval_hit=quote.contains(run.billable_units),
        lines=lines,
        boot=boot,
        evidence_class=run.evidence_class,
        notes=notes,
    )


def append_invoice(path: str | Path, invoice: Invoice) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(invoice.as_dict(), sort_keys=True) + "\n")
