"""Turning six predictions into the two numbers a decision actually needs.

A PM holding a price, a win probability, three staffing lines and a duration
still has to do arithmetic before they can say yes or no. This module does it,
and does it in the direction that survives contact with a finance review:

* **Cost is incurred, revenue is contingent.** Staff are assigned and tokens are
  burned whether or not the engagement is ultimately accepted, so the delivery
  cost is *not* discounted by the win probability and the contract value is.
  The alternative — multiplying both — flatters every marginal deal and is the
  arithmetic behind most overcommitted pipelines.
* **No recovery assumptions.** A failed engagement is worth zero here. Partially
  billed failures exist, but the fraction is a negotiation, not a statistic, and
  inventing one would put a made-up number inside the decision variable.

Rates are placeholders and are marked as such. They are the one input that is
genuinely internal to whoever runs this, and every function takes them as an
argument so nothing is hardcoded into a result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

from .outcomes import STAFF_OUTCOMES


@dataclass(frozen=True)
class Rates:
    """Blended internal day rates and the token price. **Placeholders.**

    Replace with your own before quoting anything. They are deliberately round
    numbers so that nobody mistakes them for a finance-supplied figure.
    """

    architect_day: float = 1850.0
    engineer_day: float = 1250.0
    pm_day: float = 1100.0
    #: Blended dollars per million tokens, input and output together.
    dollars_per_million_tokens: float = 5.00
    currency: str = "USD"
    source: str = "PLACEHOLDER — replace with internal rate card"

    def day_rate(self, outcome_slug: str) -> float:
        return {"architect_days": self.architect_day,
                "engineer_days": self.engineer_day,
                "pm_days": self.pm_day}[outcome_slug]

    def with_rates(self, **kwargs) -> "Rates":
        return replace(self, **kwargs)


DEFAULT_RATES = Rates()

#: How many times a pipeline is run during development and testing before it is
#: accepted. A working assumption, not a measurement — but assuming *one* run
#: would understate build-side inference by two orders of magnitude, and the
#: constant is exposed so it can be replaced by a measured figure from telemetry.
BUILD_RUNS = 250


@dataclass(frozen=True)
class Economics:
    """The decision arithmetic for one use case."""

    contract_value: float
    win_probability: float
    staff_days: Dict[str, float]
    #: Inference cost of building and testing the pipeline.
    token_cost: float
    labour_cost: float
    rates: Rates
    #: Tokens one production run of the pipeline consumes.
    tokens_per_run: float = 0.0
    #: Production runs per month, as given by the PM. Zero means not quoted.
    monthly_runs: int = 0

    @property
    def total_staff_days(self) -> float:
        return sum(self.staff_days.values())

    @property
    def delivery_cost(self) -> float:
        """What it costs to do, whether or not it is ultimately accepted."""
        return self.labour_cost + self.token_cost

    @property
    def gross_margin(self) -> float:
        """Margin if it lands. The number a delivery lead is measured on."""
        return self.contract_value - self.delivery_cost

    @property
    def gross_margin_pct(self) -> float:
        if not self.contract_value:
            return 0.0
        return self.gross_margin / self.contract_value

    @property
    def risk_adjusted_value(self) -> float:
        return self.win_probability * self.contract_value

    @property
    def expected_margin(self) -> float:
        """Contingent revenue less committed cost. The number to rank on."""
        return self.risk_adjusted_value - self.delivery_cost

    @property
    def is_worth_doing(self) -> bool:
        return self.expected_margin > 0

    @property
    def breakeven_win_rate(self) -> float:
        """The win probability at which this stops being worth staffing."""
        if not self.contract_value:
            return 1.0
        return min(self.delivery_cost / self.contract_value, 1.0)

    @property
    def margin_per_staff_day(self) -> float:
        """Ranking metric when people, not money, are the constraint.

        In practice they usually are: two engagements with the same expected
        margin are not equally attractive if one occupies an architect for a
        quarter.
        """
        days = self.total_staff_days
        return (self.expected_margin / days) if days else 0.0

    # -- run rate, which is a different question from build cost ---------

    @property
    def annual_token_cost(self) -> float:
        """What the finished pipeline costs to *operate* for a year.

        Build effort follows the scope; the inference bill follows the volume.
        At a hundred runs a month inference is a rounding error against people,
        and at fifty thousand it is the whole business case — which is the
        single most common thing a scoping conversation gets wrong.
        """
        return (self.tokens_per_run * self.monthly_runs * 12.0
                * self.rates.dollars_per_million_tokens / 1_000_000.0)

    @property
    def has_run_rate(self) -> bool:
        return self.monthly_runs > 0

    @property
    def first_year_total_cost(self) -> float:
        return self.delivery_cost + self.annual_token_cost

    @property
    def run_rate_dominates(self) -> bool:
        """True when a year of running it costs more than building it."""
        return self.annual_token_cost > self.delivery_cost

    @property
    def token_cost_share(self) -> float:
        """How much of delivery cost is inference. Usually startlingly little."""
        total = self.delivery_cost
        return (self.token_cost / total) if total else 0.0


def compute(contract_value: float, win_probability: float,
            staff_days: Dict[str, float], tokens: float,
            rates: Rates = DEFAULT_RATES, monthly_runs: int = 0,
            build_multiplier: float = BUILD_RUNS) -> Economics:
    """Assemble the economics from the head predictions and a token budget.

    ``tokens`` is the cost of *one* run of the pipeline, as priced by the token
    model. Building it costs more than one run — the thing is exercised
    repeatedly while it is developed and tested — so the build-side inference
    cost is that multiplied by :data:`BUILD_RUNS`.
    """
    labour = sum(rates.day_rate(slug) * staff_days.get(slug, 0.0)
                 for slug in STAFF_OUTCOMES)
    token_cost = (tokens * build_multiplier
                  * rates.dollars_per_million_tokens / 1_000_000.0)
    return Economics(
        contract_value=float(contract_value),
        win_probability=float(win_probability),
        staff_days={s: float(staff_days.get(s, 0.0)) for s in STAFF_OUTCOMES},
        token_cost=token_cost, labour_cost=labour, rates=rates,
        tokens_per_run=float(tokens), monthly_runs=int(monthly_runs),
    )
