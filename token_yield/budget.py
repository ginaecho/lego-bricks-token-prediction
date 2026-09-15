"""Hard pre-dispatch spending guard for paid model experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class SafetyRateCard:
    """Conservative rates used for authorization, not claimed provider prices."""

    input_per_million_usd: float = 20.0
    output_per_million_usd: float = 200.0
    source: str = (
        "campaign safety ceiling; not a provider billing quote; reasoning "
        "detail is conservatively charged in addition to output"
    )

    def price(self, input_tokens: int, output_tokens: int) -> float:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return (
            input_tokens * self.input_per_million_usd
            + output_tokens * self.output_per_million_usd
        ) / 1_000_000


class HardBudget:
    """Reserve worst-case spend before dispatch and settle measured usage."""

    def __init__(
        self,
        cap_usd: float = 200.0,
        stop_usd: float = 190.0,
        rates: SafetyRateCard | None = None,
    ):
        if cap_usd <= 0 or stop_usd <= 0 or stop_usd > cap_usd:
            raise ValueError("budget requires 0 < stop_usd <= cap_usd")
        self.cap_usd = float(cap_usd)
        self.stop_usd = float(stop_usd)
        self.rates = rates or SafetyRateCard()
        self._reserved: Dict[str, float] = {}
        self._settled: Dict[str, float] = {}

    @property
    def settled_usd(self) -> float:
        return sum(self._settled.values())

    @property
    def reserved_usd(self) -> float:
        return sum(self._reserved.values())

    def reserve(
        self, request_id: str, max_input_tokens: int, max_output_tokens: int
    ) -> float:
        if not request_id or request_id in self._reserved or request_id in self._settled:
            raise ValueError("request_id must be new and non-empty")
        amount = self.rates.price(max_input_tokens, max_output_tokens)
        projected = self.settled_usd + self.reserved_usd + amount
        if projected > self.stop_usd or projected > self.cap_usd:
            raise RuntimeError(
                f"hard budget blocks {request_id}: ${projected:.6f} projected "
                f"exceeds ${min(self.stop_usd, self.cap_usd):.2f} stop"
            )
        self._reserved[request_id] = amount
        return amount

    def settle(
        self,
        request_id: str,
        usage: Mapping[str, int],
        provider_charge_usd: float | None = None,
    ) -> float:
        if request_id not in self._reserved:
            raise ValueError(f"{request_id} has no active reservation")
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        reasoning_tokens = int(usage.get("reasoning_tokens", 0))
        safety_cost = self.rates.price(
            input_tokens, output_tokens + reasoning_tokens
        )
        if provider_charge_usd is not None:
            if provider_charge_usd < 0:
                raise ValueError("provider charge must be non-negative")
            safety_cost = max(safety_cost, float(provider_charge_usd))
        self._reserved.pop(request_id)
        self._settled[request_id] = safety_cost
        if self.settled_usd > self.cap_usd:
            raise RuntimeError("settled provider charge exceeded the hard cap")
        return safety_cost

    def cancel(self, request_id: str) -> None:
        if request_id not in self._reserved:
            raise ValueError(f"{request_id} has no active reservation")
        self._reserved.pop(request_id)

    def snapshot(self) -> Dict[str, object]:
        return {
            "cap_usd": self.cap_usd,
            "operational_stop_usd": self.stop_usd,
            "settled_safety_usd": self.settled_usd,
            "active_reserved_usd": self.reserved_usd,
            "remaining_to_stop_usd": self.stop_usd
            - self.settled_usd - self.reserved_usd,
            "rates": asdict(self.rates),
            "settled_requests": dict(self._settled),
            "active_reservations": dict(self._reserved),
        }
