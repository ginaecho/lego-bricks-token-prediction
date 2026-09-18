"""Turning provider token counts into the thing we actually predict: billable cost.

Output tokens cost several times what input tokens cost, so ``prompt + completion`` is not
a cost -- it is two different currencies added together. The reference campaign recorded a
single scalar ``tokens`` field with no split, which is why none of its numbers can support
a dollar claim no matter how good they look.

The target is *billable units*::

    y = prompt_tokens + (price_out / price_in) * completion_tokens

which is denominated in input-token-equivalents, so dollars are one multiplication away
and the ratio is never hardcoded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pricing:
    """Freeze-day rates. ``provisional`` is sticky and taints every dollar claim."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    currency: str = "USD"
    provisional: bool = True
    source: str = "TODO(measure): set from the published rate on freeze day"

    def __post_init__(self) -> None:
        if self.input_per_mtok <= 0:
            raise ValueError("input_per_mtok must be positive")
        if self.output_per_mtok <= 0:
            raise ValueError("output_per_mtok must be positive")

    @property
    def ratio(self) -> float:
        """How many input tokens one output token costs."""
        return self.output_per_mtok / self.input_per_mtok

    def billable_units(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens + self.ratio * completion_tokens

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self.billable_units(prompt_tokens, completion_tokens) * self.input_per_mtok / 1e6

    def units_to_usd(self, units: float) -> float:
        return units * self.input_per_mtok / 1e6

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Pricing":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> "Pricing":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")


PROVISIONAL = Pricing(
    input_per_mtok=1.0,
    output_per_mtok=5.0,
    provisional=True,
    source="provisional placeholder; replace on freeze day before any dollar claim",
)


def reconciles_with_billing(campaign_usd: float, billing_export_usd: float) -> bool:
    """Campaign dollars must land within 1% of the provider's own export.

    If they do not, the arithmetic is wrong somewhere and every dollar figure in the
    report is void -- not approximately right.
    """
    if billing_export_usd <= 0:
        return False
    return abs(campaign_usd - billing_export_usd) / billing_export_usd <= 0.01
