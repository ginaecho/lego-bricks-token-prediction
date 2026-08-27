"""Who does the work — a roster the user owns, not a list baked into the model.

The first version of this predicted three staffing lines: architect, engineer,
program manager. That is not a delivery team, it is the three roles that
happened to be easy to name. Real engagements are staffed with data scientists,
data engineers, industry consultants and change managers, in proportions that
are the *whole* difference between a document-extraction pipeline and a
forecasting model, and a tool that cannot express that difference cannot price
either one.

So the roster is data. It is a JSON file the user edits, and everything
downstream is built from it: one pair of heads per role, one day rate per role,
one line per role on every card. Add "MLOps engineer" to the file and — provided
the corpus has a column for it — it is fitted, priced and reported with no code
change at all.

Two rates of change, kept apart
-------------------------------
*The roster* changes when an organisation changes how it staffs work. It is
configuration.

*The roles on one use case* change with every use case, and are a judgement the
PM makes at scoping time: "this one needs a data scientist and no change
manager". That is :attr:`~project_yield.usecase.UseCase.roles`, not a roster
edit, and it overrides what the history would otherwise have inferred — see
:mod:`project_yield.predict`.

Presence and days are separate questions
----------------------------------------
A data scientist is not needed on every engagement. Averaging their days over
jobs that never used one gives "3.1 data scientist days", which is not a thing
anybody can book. Each role therefore carries two predictions — how often work
like this needs the role at all, and how many days when it does — and the cost
uses their product. See :mod:`project_yield.multihead`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Role:
    """One staffing line: what they are called, and what a day of them costs."""

    slug: str
    name: str
    day_rate: float
    blurb: str = ""
    #: Set on roles that appear on nearly every engagement. Purely descriptive —
    #: presence is fitted from the corpus like everything else — but it lets a
    #: report say "core team" without hardcoding which roles those are.
    core: bool = False

    @property
    def days_outcome(self) -> str:
        return f"{self.slug}_days"

    @property
    def used_outcome(self) -> str:
        return f"{self.slug}_used"


#: The roster the prototype ships with. Rates are **placeholders** — round
#: numbers chosen so nobody mistakes them for a finance-supplied figure — and
#: the whole file is meant to be replaced. See ``roles.json``.
DEFAULT_ROLES: Tuple[Role, ...] = (
    Role("solution_architect", "Solution architect", 1850.0,
         "Shapes the solution and owns the technical decisions. Scales with "
         "novelty and integration surface, not with volume.", core=True),
    Role("data_scientist", "Data scientist", 1650.0,
         "Model selection, evaluation design and error analysis. Needed where "
         "the answer has to be measured rather than asserted."),
    Role("data_engineer", "Data engineer", 1400.0,
         "Pipelines, indexing and access to the source material. The role most "
         "often missed at scoping and most often the critical path."),
    Role("software_engineer", "Software engineer", 1250.0,
         "Builds and integrates the thing. Usually the largest line.",
         core=True),
    Role("security_expert", "Security expert", 1700.0,
         "Threat modelling, data handling and the security review the client's "
         "own team will run anyway. Needed wherever regulated or personal data "
         "moves, which is most places."),
    Role("consultant", "Industry consultant", 1500.0,
         "Domain and process work: what the current process is, what it should "
         "become, and what the regulator will accept."),
    Role("project_manager", "Project manager", 1100.0,
         "Plan, governance and the client relationship. Scales with "
         "coordination load rather than with build.", core=True),
    Role("change_manager", "Change manager", 1200.0,
         "Adoption, training and the operating-model change. Needed wherever "
         "the work displaces what people currently do."),
)


class Roster:
    """An ordered set of roles, with lookup by slug."""

    def __init__(self, roles: Sequence[Role], source: str = "built-in") -> None:
        if not roles:
            raise ValueError("a roster needs at least one role")
        seen = set()
        for role in roles:
            if role.slug in seen:
                raise ValueError(f"duplicate role slug {role.slug!r}")
            seen.add(role.slug)
        self._roles: Tuple[Role, ...] = tuple(roles)
        self.source = source

    # -- access ----------------------------------------------------------

    def __iter__(self) -> Iterator[Role]:
        return iter(self._roles)

    def __len__(self) -> int:
        return len(self._roles)

    def __contains__(self, slug: str) -> bool:
        return any(r.slug == slug for r in self._roles)

    def __getitem__(self, slug: str) -> Role:
        for role in self._roles:
            if role.slug == slug:
                return role
        raise KeyError(f"no role {slug!r} in the roster; it has: "
                       + ", ".join(self.slugs))

    def get(self, slug: str) -> Optional[Role]:
        return next((r for r in self._roles if r.slug == slug), None)

    @property
    def slugs(self) -> Tuple[str, ...]:
        return tuple(r.slug for r in self._roles)

    @property
    def days_outcomes(self) -> Tuple[str, ...]:
        return tuple(r.days_outcome for r in self._roles)

    @property
    def used_outcomes(self) -> Tuple[str, ...]:
        return tuple(r.used_outcome for r in self._roles)

    def core(self) -> Tuple[Role, ...]:
        return tuple(r for r in self._roles if r.core)

    # -- validation ------------------------------------------------------

    def validate(self, slugs: Sequence[str], what: str = "role") -> List[str]:
        """Check names against the roster, naming what is wrong and offering
        what is right. A silently dropped role is a silently missing cost."""
        unknown = [s for s in slugs if s not in self]
        if unknown:
            raise ValueError(
                f"unknown {what}(s): {', '.join(sorted(unknown))}. The roster "
                f"({self.source}) has: {', '.join(self.slugs)}")
        return list(slugs)

    # -- serialisation ---------------------------------------------------

    def to_list(self) -> List[Dict[str, object]]:
        return [{"slug": r.slug, "name": r.name, "day_rate": r.day_rate,
                 "blurb": r.blurb, "core": r.core} for r in self._roles]

    def rates(self) -> Dict[str, float]:
        return {r.slug: r.day_rate for r in self._roles}

    def with_rates(self, rates: Dict[str, float]) -> "Roster":
        """A copy with some day rates replaced — the usual first customisation."""
        self.validate(list(rates), "role rate")
        return Roster([Role(r.slug, r.name, float(rates.get(r.slug, r.day_rate)),
                            r.blurb, r.core) for r in self._roles],
                      source=self.source + " + overridden rates")


DEFAULT_ROSTER = Roster(DEFAULT_ROLES)


def default_roster_path() -> str:
    """``roles.json`` beside the package, if the user has written one."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "roles.json")


def load_roster(path: Optional[str] = None) -> Roster:
    """Read a roster from JSON, falling back to the built-in one.

    The file is a list of ``{slug, name, day_rate, blurb?, core?}``. Absent is
    not an error — the built-in roster is the default and says so in its
    ``source``, which every report prints alongside the rates.
    """
    path = path or default_roster_path()
    if not os.path.exists(path):
        return DEFAULT_ROSTER
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("roles", [])
    roles = []
    for i, entry in enumerate(data, 1):
        try:
            roles.append(Role(
                slug=str(entry["slug"]).strip().lower().replace(" ", "_"),
                name=str(entry.get("name") or entry["slug"]),
                day_rate=float(entry["day_rate"]),
                blurb=str(entry.get("blurb", "")),
                core=bool(entry.get("core", False)),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{os.path.basename(path)}: role {i} is not "
                             f"usable ({exc})") from None
    return Roster(roles, source=os.path.basename(path))
