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
from typing import Dict, Optional

from .roles import DEFAULT_ROSTER, Roster


@dataclass(frozen=True)
class Rates:
    """Day rates by role, and the token price. **Placeholders.**

    Day rates live on the roster, because the roster is the thing an
    organisation owns and edits — adding a role and forgetting to price it is
    not a state this should be able to reach. Replace both before quoting
    anything; the defaults are round numbers so nobody mistakes them for a
    finance-supplied figure.
    """

    roster: Roster = DEFAULT_ROSTER
    #: Blended dollars per million tokens, input and output together.
    dollars_per_million_tokens: float = 5.00
    currency: str = "USD"
    source: str = "PLACEHOLDER — replace with internal rate card"

    def day_rate(self, role_slug: str) -> float:
        role = self.roster.get(role_slug)
        if role is None:
            raise KeyError(
                f"no rate for {role_slug!r}: it is not on the roster "
                f"({self.roster.source}). Add it there rather than here, so "
                f"it is priced and predicted together.")
        return role.day_rate

    def with_day_rates(self, **rates: float) -> "Rates":
        """A copy with some role day rates replaced, by slug."""
        return replace(self, roster=self.roster.with_rates(rates),
                       source=self.source + " + overridden day rates")

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
    #: Expected days by role slug — presence probability times days-when-needed.
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
    def cost_by_role(self) -> Dict[str, float]:
        return {slug: self.rates.day_rate(slug) * days
                for slug, days in self.staff_days.items() if days}

    @property
    def biggest_role(self) -> Optional[str]:
        costs = self.cost_by_role
        return max(costs, key=lambda k: costs[k]) if costs else None

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
            #: Expected days by role slug — presence probability times days-when-needed.
    staff_days: Dict[str, float], tokens: float,
            rates: Rates = DEFAULT_RATES, monthly_runs: int = 0,
            build_multiplier: float = BUILD_RUNS) -> Economics:
    """Assemble the economics from the head predictions and a token budget.

    ``tokens`` is the cost of *one* run of the pipeline, as priced by the token
    model. Building it costs more than one run — the thing is exercised
    repeatedly while it is developed and tested — so the build-side inference
    cost is that multiplied by :data:`BUILD_RUNS`.
    """
    labour = sum(rates.day_rate(slug) * days
                 for slug, days in staff_days.items() if days)
    token_cost = (tokens * build_multiplier
                  * rates.dollars_per_million_tokens / 1_000_000.0)
    return Economics(
        contract_value=float(contract_value),
        win_probability=float(win_probability),
        staff_days={s: float(d) for s, d in staff_days.items()},
        token_cost=token_cost, labour_cost=labour, rates=rates,
        tokens_per_run=float(tokens), monthly_runs=int(monthly_runs),
    )
