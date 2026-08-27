"""What the model predicts, beyond tokens.

Token Yield answers one question — *what will this cost to run?* A Microsoft
project manager scoping a use case has to answer four more before they can
decide anything:

* **Will the client pay for it, and how much?**  (``contract_value``)
* **Will it actually land?**                     (``win_probability``)
* **Who do I need, and for how long?**           (one pair of heads per role
                                                  on the roster)
* **When is it done?**                           (``calendar_days``)

Each of these is a *head*: an independent model over the same encoded feature
vector — the same lego bricks the token model is built from. They are separate
heads rather than one multi-output model on purpose. Money is multiplicative and
long-tailed, a win is a coin flip with structure, and elapsed time is bounded
below by the critical path however many people you add. Forcing them through one
link function would make at least two of the four wrong in a predictable
direction, and a forecast that is wrong in a predictable direction is worse than
no forecast: people learn to apply a mental correction and then stop reading it.

So each head declares its own **link** and its own **scoring metric**, and each
independently selects its functional form by cross-validation.

The staffing heads are not written down here
--------------------------------------------
The three fixed outcomes below are properties of the *engagement*. Staffing is a
property of the *organisation*, and it is read off
:mod:`project_yield.roles` — a roster the user edits — rather than hardcoded.
:func:`build_outcomes` turns that roster into two heads per role:

``<role>_used``
    Does work like this need this role at all? A logit head over every
    engagement. Averaging a data scientist's days across jobs that never used
    one produces "3.1 days", which is not a thing anybody can book.

``<role>_days``
    How many days when it does? A log head, fitted only on the engagements that
    actually used the role, so the number means what it says.

Cost uses the product of the two; the staffing plan shows both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

from .linalg import logit, sigmoid

if TYPE_CHECKING:                                        # pragma: no cover
    from .roles import Role, Roster


@dataclass(frozen=True)
class Link:
    """How a head's target is transformed before a linear model is fitted."""

    name: str
    forward: Callable[[float], float]     # observed value -> fitting space
    inverse: Callable[[float], float]     # fitting space -> observed value
    rationale: str


IDENTITY = Link(
    "identity", lambda y: float(y), lambda z: float(z),
    "the quantity adds up: two of the thing costs twice as much",
)

LOG = Link(
    "log", lambda y: math.log(max(float(y), 1e-9)),
    lambda z: math.exp(min(float(z), 700.0)),
    "the quantity multiplies: effects are percentages, and it cannot go negative",
)

LOGIT = Link(
    "logit", logit, sigmoid,
    "the quantity is a probability, bounded to (0, 1) at both ends",
)


@dataclass(frozen=True)
class Outcome:
    """One thing the prototype predicts for a use case."""

    slug: str
    name: str
    unit: str
    link: Link
    question: str
    blurb: str
    #: True when the corpus records this as a 0/1 event rather than a magnitude.
    binary: bool = False

    def format(self, value: float) -> str:
        if self.binary:
            return f"{value:.0%}"
        if self.unit == "USD":
            return f"${value:,.0f}"
        if self.unit == "days":
            return f"{value:,.1f} days"
        return f"{value:,.0f} {self.unit}"


OUTCOMES: Dict[str, Outcome] = {
    o.slug: o for o in (
        Outcome(
            "contract_value", "Contract value", "USD", LOG,
            "How much has a client like this paid for work like this?",
            "The price the engagement was actually signed at — willingness to "
            "pay, read off comparable delivered work rather than off a rate "
            "card. Log link: deal sizes span two orders of magnitude and a "
            "10% error on a $2m deal is not the same mistake as $10k on $100k.",
        ),
        Outcome(
            "win_probability", "Success rate", "probability", LOGIT,
            "How often does a use case shaped like this actually succeed?",
            "Delivered to acceptance and renewed, as opposed to stalled, "
            "descoped or abandoned. Fitted on the realised 0/1 outcome of past "
            "engagements, so it is a base rate with structure, not an opinion.",
            binary=True,
        ),
        Outcome(
            "calendar_days", "Time to finish", "days", LOG,
            "How long from kickoff to acceptance?",
            "Elapsed time, not effort. Bounded below by the critical path, so "
            "it does not fall in proportion when staff are added — which is "
            "exactly why it is a separate head and not staff-days divided by "
            "a team size.",
        ),
    )
}

#: The outcomes that are properties of the engagement rather than of the
#: organisation delivering it. Stable ordering for tables and reports.
ORDER: Tuple[str, ...] = ("contract_value", "win_probability", "calendar_days")


def days_outcome(role: "Role") -> Outcome:
    """The head that answers *how many days of this role, when it is needed*."""
    return Outcome(
        slug=role.days_outcome, name=role.name, unit="days", link=LOG,
        question=f"How many {role.name.lower()} days does it take?",
        blurb=(role.blurb + " ") if role.blurb else ""
              + "Fitted only on engagements that actually used the role, so "
                "the figure is days-when-needed rather than an average "
                "flattened by the jobs that needed none.",
    )


def used_outcome(role: "Role") -> Outcome:
    """The head that answers *does work like this need this role at all*."""
    return Outcome(
        slug=role.used_outcome, name=f"{role.name} needed", unit="probability",
        link=LOGIT, binary=True,
        question=f"How often does work like this need a {role.name.lower()}?",
        blurb=(f"The base rate at which comparable engagements staffed a "
               f"{role.name.lower()} at all. A project manager who knows the "
               f"answer overrides it; one who does not gets the history."),
    )


def build_outcomes(roster: "Roster") -> Dict[str, Outcome]:
    """Every head for a given roster: the three fixed ones, then two per role."""
    out: Dict[str, Outcome] = dict(OUTCOMES)
    for role in roster:
        out[role.used_outcome] = used_outcome(role)
        out[role.days_outcome] = days_outcome(role)
    return out


def build_order(roster: "Roster") -> Tuple[str, ...]:
    """Report order: engagement outcomes first, then the staffing plan."""
    staffing: List[str] = []
    for role in roster:
        staffing += [role.used_outcome, role.days_outcome]
    return ORDER + tuple(staffing)
