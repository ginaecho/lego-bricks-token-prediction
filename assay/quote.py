"""Quotes: a price, a range, and the baselines it has to justify itself against.

A quote is never a bare number. It carries the interval, the baselines it beat (or did
not), the identity of the coefficients that produced it, and the moment it was committed.
That last field is what makes the blind test possible: a quote written after the run is
not a prediction, and the ordering is checked rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from assay.conformal import Conformal
from assay.costmodel import CostModel, Predictor
from assay.evidence import EvidenceClass
from assay.pricing import Pricing

OUTLIER_SHARE = 0.25


@dataclass(frozen=True)
class Quote:
    request_id: str
    units: dict[str, int]
    context_bytes: int
    predicted_units: float
    predicted_usd: float
    interval_lo: float
    interval_hi: float
    interval_level: float
    baselines: dict[str, float]
    outlier_flags: tuple[str, ...]
    coefficients_ref: str
    quoted_at: str
    evidence_class: EvidenceClass
    show_per_brick: bool = True
    """False under a Narrow verdict: the estimator ships, the per-brick prices do not."""
    notes: tuple[str, ...] = field(default=())

    @property
    def relative_half_width_pct(self) -> float:
        return (self.interval_hi - self.interval_lo) / 2.0 / self.predicted_units * 100.0

    def contains(self, actual: float) -> bool:
        return self.interval_lo <= actual <= self.interval_hi

    def hash(self) -> str:
        payload = json.dumps(
            {
                "request_id": self.request_id,
                "units": self.units,
                "context_bytes": self.context_bytes,
                "predicted_units": round(self.predicted_units, 6),
                "coefficients_ref": self.coefficients_ref,
                "quoted_at": self.quoted_at,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["evidence_class"] = self.evidence_class.value
        data["outlier_flags"] = list(self.outlier_flags)
        data["notes"] = list(self.notes)
        data["quote_sha256"] = self.hash()
        data["relative_half_width_pct"] = round(self.relative_half_width_pct, 3)
        if not self.show_per_brick:
            data["units"] = {"_hidden": "narrow verdict: per-brick detail withheld"}
        return data

    def summary(self) -> str:
        asked = ", ".join(f"{k} x{v}" for k, v in self.units.items() if v) or "no bricks"
        return (
            f"{self.request_id}: {asked} over {self.context_bytes:,} bytes\n"
            f"  quote  {self.predicted_units:,.0f} units  (${self.predicted_usd:.4f})\n"
            f"  {self.interval_level:.0%} band  "
            f"{self.interval_lo:,.0f} - {self.interval_hi:,.0f}  "
            f"(+/-{self.relative_half_width_pct:.1f}%)"
        )


def outlier_flags(model: CostModel, units: Mapping[str, int], predicted: float) -> tuple[str, ...]:
    """Operational warnings, not statistical ones.

    A brick that dominates the quote is a scoping decision waiting to be made: the useful
    output is not "this will cost a lot" but "this one clause is why, and here is the
    lever".
    """
    flags: list[str] = []
    for brick, count in units.items():
        if not count:
            continue
        share = model.marginals.get(brick, 0.0) * count
        if predicted > 0 and share / predicted >= OUTLIER_SHARE:
            flags.append(
                f"{brick} x{count} is {share / predicted:.0%} of this quote "
                f"({model.marginals.get(brick, 0.0):,.0f}/unit) -- narrow the scope of that clause"
            )
    if units.get("Retrieve"):
        flags.append(
            "Retrieve searches an unspecified source; naming the document turns it into "
            "Extract and is usually an order of magnitude cheaper"
        )
    return tuple(flags)


def make_quote(
    request_id: str,
    units: Mapping[str, int],
    context_bytes: int,
    *,
    model: CostModel,
    conformal: Conformal,
    pricing: Pricing,
    baselines: Mapping[str, Predictor] | None = None,
    show_per_brick: bool = True,
    evidence_class: EvidenceClass = EvidenceClass.PIPELINE_ONLY,
    quoted_at: str | None = None,
) -> Quote:
    units = dict(units)
    point = model.predict(units, context_bytes)
    lo, hi = conformal.interval(point)

    base = {
        key: predictor.predict(units, context_bytes)
        for key, predictor in (baselines or {}).items()
    }

    return Quote(
        request_id=request_id,
        units=units,
        context_bytes=context_bytes,
        predicted_units=point,
        predicted_usd=pricing.units_to_usd(point),
        interval_lo=lo,
        interval_hi=hi,
        interval_level=conformal.level,
        baselines=base,
        outlier_flags=outlier_flags(model, units, point) if show_per_brick else (),
        coefficients_ref=f"{model.form}@{model.data_sha256[:8]}",
        quoted_at=quoted_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        evidence_class=evidence_class,
        show_per_brick=show_per_brick,
    )


def append_quote(path: str | Path, quote: Quote) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(quote.as_dict(), sort_keys=True) + "\n")


def read_quotes(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["request_id"]] = row
    return out
