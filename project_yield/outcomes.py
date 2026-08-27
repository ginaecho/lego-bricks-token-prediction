"""What the model predicts, beyond tokens.

Token Yield answers one question — *what will this cost to run?* A Microsoft
project manager scoping a use case has to answer four more before they can
decide anything:

* **Will the client pay for it, and how much?**  (``contract_value``)
* **Will it actually land?**                     (``win_probability``)
* **Who do I need, and for how long?**           (``architect_days``,
                                                  ``engineer_days``, ``pm_days``)
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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from .linalg import logit, sigmoid


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
            "architect_days", "Architect", "days", LOG,
            "How many solution-architect days does it take?",
            "Scales with novelty and integration surface, not with volume: "
            "the tenth extraction is free to design, the first retrieval "
            "across an unmapped corpus is not.",
        ),
        Outcome(
            "engineer_days", "Engineer", "days", LOG,
            "How many engineering days does it take?",
            "The bulk of delivery effort, and the line item that moves most "
            "with the brick composition.",
        ),
        Outcome(
            "pm_days", "Program manager", "days", LOG,
            "How many program-management days does it take?",
            "Scales with governance rather than build: regulated industries "
            "and multi-party scopes pay for coordination whatever gets built.",
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

#: Stable ordering for tables, feature vectors and reports.
ORDER: Tuple[str, ...] = ("contract_value", "win_probability", "architect_days",
                          "engineer_days", "pm_days", "calendar_days")

#: The heads that together make up the staffing plan.
STAFF_OUTCOMES: Tuple[str, ...] = ("architect_days", "engineer_days", "pm_days")
