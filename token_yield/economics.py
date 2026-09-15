"""Unit economics over measured runs whose work was independently accepted."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Pricing:
    """Provider rates in dollars per million tokens."""

    input_per_million: float = 0.0
    cached_input_per_million: Optional[float] = None
    output_per_million: float = 0.0
    blended_per_million: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "input_per_million", "output_per_million", "blended_per_million"
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("pricing rates must be finite and non-negative")
        cached = self.cached_input_per_million
        if cached is not None and (not math.isfinite(cached) or cached < 0):
            raise ValueError("pricing rates must be finite and non-negative")

    @property
    def effective_cached_input_per_million(self) -> float:
        return (
            self.input_per_million
            if self.cached_input_per_million is None
            else self.cached_input_per_million
        )

    def attempt_cost(
        self,
        *,
        input_tokens: float,
        cached_input_tokens: float,
        output_tokens: float,
    ) -> float:
        values = (input_tokens, cached_input_tokens, output_tokens)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("token counts must be finite and non-negative")
        if cached_input_tokens > input_tokens:
            raise ValueError(
                "cached_input_tokens must not exceed input_tokens"
            )
        return (
            (input_tokens - cached_input_tokens) * self.input_per_million
            + cached_input_tokens * self.effective_cached_input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class WorkOutcome:
    """One attempted case, including failed work because failures still cost."""

    case_id: str
    accepted: bool
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_center: str = "unassigned"

    def __post_init__(self) -> None:
        values = (
            self.input_tokens, self.cached_tokens, self.output_tokens,
            self.reasoning_tokens, self.total_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("token counts must be non-negative integers")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached_tokens must not exceed input_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens must not exceed output_tokens")
        if (
            self.total_tokens
            and (self.input_tokens or self.output_tokens)
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens must equal input_tokens + output_tokens")

    def tokens(self) -> int:
        measured = self.input_tokens + self.output_tokens
        return self.total_tokens or measured

    def cost(self, pricing: Pricing) -> float:
        if self.input_tokens or self.output_tokens:
            return pricing.attempt_cost(
                input_tokens=self.input_tokens,
                cached_input_tokens=self.cached_tokens,
                output_tokens=self.output_tokens,
            )
        return self.total_tokens * pricing.blended_per_million / 1_000_000


@dataclass(frozen=True)
class UnitEconomics:
    """Scoping range and accepted-work economics for a cohort of attempts."""

    attempts: int
    accepted: int
    acceptance_rate: float
    attempts_per_accepted_outcome: float
    tokens_per_accepted_outcome: float
    p50_tokens: float
    p90_tokens: float
    p95_tokens: float
    p99_tokens: float
    tail_reserve_tokens: float
    total_cost: float
    cost_per_accepted_outcome: float

    def quote(self, risk_percentile: int = 95) -> Dict[str, object]:
        """Quote tail budgets using an explicit IID geometric retry model."""

        percentiles = {
            50: self.p50_tokens,
            90: self.p90_tokens,
            95: self.p95_tokens,
            99: self.p99_tokens,
        }
        if risk_percentile not in percentiles:
            raise ValueError("risk_percentile must be one of 50, 90, 95, 99")
        attempt_budget = percentiles[risk_percentile]
        confidence = risk_percentile / 100.0
        if self.acceptance_rate >= 1.0:
            retry_attempts = 1
        else:
            retry_attempts = math.ceil(
                math.log(1.0 - confidence) / math.log(1.0 - self.acceptance_rate)
            )
        outcome_budget = attempt_budget * retry_attempts
        return {
            "expected_tokens_per_attempt": self.p50_tokens,
            "expected_tokens_per_accepted_outcome": self.tokens_per_accepted_outcome,
            "budget_tokens_per_attempt": attempt_budget,
            "budget_tokens_per_accepted_outcome": outcome_budget,
            "risk_reserve_tokens_per_attempt": max(
                0.0, attempt_budget - self.p50_tokens
            ),
            "risk_reserve_tokens_per_accepted_outcome": max(
                0.0, outcome_budget - self.tokens_per_accepted_outcome
            ),
            "risk_percentile": float(risk_percentile),
            "retry_attempts_at_percentile": retry_attempts,
            "retry_model": "iid_geometric",
        }


def percentile(values: Iterable[int], probability: float) -> float:
    """Linearly interpolated empirical percentile."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one value is required")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(outcomes: Iterable[WorkOutcome], pricing: Pricing) -> UnitEconomics:
    """Connect all spend to the accepted outputs produced by that spend."""

    outcomes = list(outcomes)
    if not outcomes:
        raise ValueError("at least one outcome is required")
    if any(outcome.tokens() <= 0 for outcome in outcomes):
        raise ValueError("every outcome must have a positive measured token count")
    accepted = [outcome for outcome in outcomes if outcome.accepted]
    if not accepted:
        raise ValueError("at least one accepted outcome is required")
    attempt_tokens = [outcome.tokens() for outcome in outcomes]
    total_cost = sum(outcome.cost(pricing) for outcome in outcomes)
    p50 = percentile(attempt_tokens, 0.50)
    p95 = percentile(attempt_tokens, 0.95)
    total_tokens = sum(attempt_tokens)
    return UnitEconomics(
        attempts=len(outcomes),
        accepted=len(accepted),
        acceptance_rate=len(accepted) / len(outcomes),
        attempts_per_accepted_outcome=len(outcomes) / len(accepted),
        tokens_per_accepted_outcome=total_tokens / len(accepted),
        p50_tokens=p50,
        p90_tokens=percentile(attempt_tokens, 0.90),
        p95_tokens=p95,
        p99_tokens=percentile(attempt_tokens, 0.99),
        tail_reserve_tokens=max(0.0, p95 - p50),
        total_cost=total_cost,
        cost_per_accepted_outcome=total_cost / len(accepted),
    )


def chargeback(outcomes: Iterable[WorkOutcome], pricing: Pricing) -> List[Dict[str, float]]:
    """Allocate actual spend and accepted outcomes to cost centers."""

    grouped: Dict[str, List[WorkOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.cost_center, []).append(outcome)
    rows = []
    for cost_center in sorted(grouped):
        cohort = grouped[cost_center]
        cost = sum(outcome.cost(pricing) for outcome in cohort)
        accepted = sum(outcome.accepted for outcome in cohort)
        rows.append({
            "cost_center": cost_center,
            "attempts": float(len(cohort)),
            "accepted_outcomes": float(accepted),
            "actual_cost": cost,
            "cost_per_accepted_outcome": cost / accepted if accepted else math.inf,
        })
    return rows
