"""What the client gets — the other half of the business case.

Everything else in this package prices the *engagement*: what it costs to
build, what the client pays, who is needed. None of that answers the question a
sponsor actually asks, which is what the thing is worth once it is running.

For the work this vocabulary describes — reading documents, pulling fields out
of them, sorting them, checking them — the benefit has one dominant term:
people currently do it by hand. So the impact is the handling time the pipeline
displaces, at the client's own loaded cost, less what the inference costs to
run. Every use case already carries what that calculation needs: the task
bricks say how much handling there is per run, and the production run rate says
how often.

This is the number that is usually missing at scoping time, and its absence is
why AI business cases get argued on cost. A build that costs $60,000 is
expensive or cheap depending entirely on whether it displaces $40,000 a year or
$4 million, and nothing in a delivery estimate can tell you which.

Assumptions, all of them replaceable
------------------------------------
The per-brick handling times below are **placeholders** — considered estimates
of how long a competent person takes to do one unit of each task by hand, not
measurements. So is the loaded hourly cost, and so is the deflection rate: no
automation removes all the work, and assuming it does is the single most common
way an AI business case is overstated. Every one of them is an argument of
:class:`ImpactAssumptions`, printed on every card, and meant to be replaced
with the client's own time-and-motion figures before anybody quotes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from token_yield.tasks import ORDER as BRICKS

#: Minutes of human handling one unit of each task takes today. **Placeholders.**
MANUAL_MINUTES: Dict[str, float] = {
    "review": 2.0,       # read one document and form a view
    "extract": 0.4,      # key one field
    "classify": 0.2,     # sort one item
    "retrieve": 3.0,     # find where something is stated
    "reconcile": 2.0,    # compare one pair and note the difference
    "draft": 5.0,        # write one piece
    "remediate": 4.0,    # diagnose and correct one error
    "validate": 0.8,     # write and run one check
    "report": 4.0,       # write up one aspect
}


@dataclass(frozen=True)
class ImpactAssumptions:
    """The three numbers the benefit case turns on. **All placeholders.**"""

    #: Fully loaded cost of an hour of the client's own staff time.
    client_hourly_cost: float = 55.0
    #: Share of the handling the pipeline actually removes. The rest still
    #: needs a person — exceptions, escalations, the cases the model declines.
    #: Assuming 100% is the commonest way a benefit case is overstated.
    deflection: float = 0.70
    minutes: Dict[str, float] = field(
        default_factory=lambda: dict(MANUAL_MINUTES))
    source: str = "PLACEHOLDER — replace with the client's own handling times"

    def with_values(self, **kwargs) -> "ImpactAssumptions":
        return replace(self, **kwargs)


DEFAULT_ASSUMPTIONS = ImpactAssumptions()


@dataclass(frozen=True)
class Impact:
    """The client-side value of one use case, once it is running."""

    minutes_per_run: float
    monthly_runs: int
    annual_token_cost: float
    delivery_cost: float
    assumptions: ImpactAssumptions

    @property
    def quoted(self) -> bool:
        """False when no production run rate was given — then there is no
        annual anything, and saying so beats printing a zero."""
        return self.monthly_runs > 0

    @property
    def annual_runs(self) -> int:
        return self.monthly_runs * 12

    @property
    def hours_saved(self) -> float:
        return (self.minutes_per_run * self.annual_runs
                * self.assumptions.deflection / 60.0)

    @property
    def annual_benefit(self) -> float:
        """Handling cost displaced in a year, before running costs."""
        return self.hours_saved * self.assumptions.client_hourly_cost

    @property
    def annual_net_benefit(self) -> float:
        """After what the pipeline costs to run. This is the impact figure."""
        return self.annual_benefit - self.annual_token_cost

    @property
    def fte_equivalent(self) -> float:
        """Hours saved as full-time people, at 1,700 productive hours a year.

        Sponsors think in people, not hours, and the translation is the
        sentence that gets a business case read.
        """
        return self.hours_saved / 1700.0

    @property
    def first_year_return(self) -> float:
        """Net benefit less what it cost to build. Year one, not steady state."""
        return self.annual_net_benefit - self.delivery_cost

    @property
    def payback_months(self) -> Optional[float]:
        """Months for the benefit to repay the build. ``None`` if it never does."""
        if not self.quoted or self.annual_net_benefit <= 0:
            return None
        return 12.0 * self.delivery_cost / self.annual_net_benefit

    @property
    def is_positive(self) -> bool:
        return self.quoted and self.first_year_return > 0

    @property
    def verdict(self) -> str:
        if not self.quoted:
            return ("no production run rate given, so the annual benefit "
                    "cannot be estimated")
        months = self.payback_months
        if months is None:
            return "the running cost exceeds the handling it displaces"
        if months <= 12:
            return f"pays back the build in {months:.1f} months"
        return f"pays back the build in {months / 12:.1f} years"


def manual_minutes(counts: Dict[str, int],
                   assumptions: ImpactAssumptions = DEFAULT_ASSUMPTIONS
                   ) -> float:
    """Human handling time for one end-to-end run, in minutes."""
    return sum(assumptions.minutes.get(slug, 0.0) * counts.get(slug, 0)
               for slug in BRICKS)


def compute(counts: Dict[str, int], monthly_runs: int,
            annual_token_cost: float, delivery_cost: float,
            assumptions: ImpactAssumptions = DEFAULT_ASSUMPTIONS) -> Impact:
    return Impact(
        minutes_per_run=manual_minutes(counts, assumptions),
        monthly_runs=int(monthly_runs),
        annual_token_cost=float(annual_token_cost),
        delivery_cost=float(delivery_cost),
        assumptions=assumptions,
    )
