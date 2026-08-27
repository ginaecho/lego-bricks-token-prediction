"""What a project manager types in, and what it turns into.

A :class:`UseCase` is the unit of input for the prototype. It carries what a
Microsoft PM or architect actually knows at scoping time — a description, the
industry, the business goal, roughly how much material is in scope — plus the
one thing that is usually lost and matters most: **where this use case came
from**.

Lineage
-------
Use cases are rarely greenfield. A customer who bought invoice extraction buys
claims triage next; a proven compliance pipeline is re-pointed at a second
regulator. Those are not new projects and pricing them as new projects is how
both the estimate and the margin go wrong:

* a **continuation** (``parent_id``) inherits architecture, connectors and a
  working relationship — cheaper to build, faster to land, more likely to
  succeed, and typically signed at a *lower* price because the client knows the
  hard part is done;
* a **sibling** (``sibling_ids``) is a use case delivered alongside this one for
  the same client, which shares fixed setup but competes for the same people.

The prototype does not assume what those relationships are worth. They become
features — :attr:`UseCase.reuse_depth`, sibling count, and the measured overlap
in task bricks with everything upstream — and every head is free to find them
worthless. :mod:`project_yield.lineage` computes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from token_yield.tasks import ORDER as BRICKS


#: Industries the corpus is stratified over. Each carries a different
#: governance load, a different willingness to pay, and a different base rate
#: of delivery success — which is why it is a feature and not a label.
INDUSTRIES: Tuple[str, ...] = (
    "financial_services", "healthcare", "manufacturing",
    "retail", "public_sector", "energy",
)

#: The business goal the use case is bought against. A PM knows this on day one
#: and it moves price harder than anything technical does.
GOALS: Tuple[str, ...] = (
    "cost_reduction", "revenue_growth", "compliance_risk", "customer_experience",
)


def normalise_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """Coerce a partial brick dict into the full, ordered vocabulary."""
    out = {slug: 0 for slug in BRICKS}
    for key, value in (counts or {}).items():
        slug = str(key).strip().lower()
        if slug in out:
            try:
                out[slug] += max(0, int(value))
            except (TypeError, ValueError):
                continue
    return out


@dataclass
class UseCase:
    """One scoped piece of work, as a PM describes it.

    ``counts`` is the encoded form — the lego bricks. It is produced from
    ``description`` by :mod:`project_yield.encode`, either by an agent or by the
    keyword fallback, and ``encoder`` records which, because a number derived
    from a keyword match should never be presented like one derived from
    judgement.
    """

    id: str
    title: str
    description: str = ""
    industry: str = "financial_services"
    goal: str = "cost_reduction"
    counts: Dict[str, int] = field(default_factory=dict)
    context_bytes: int = 0
    #: How many times the finished pipeline runs per month in production.
    #: Zero means "build only, no run-rate quoted". This is the input that
    #: decides whether inference is a rounding error or the main cost: build
    #: effort follows the scope, but the token bill follows the volume.
    monthly_runs: int = 0
    #: The use case this one continues, if any.
    parent_id: Optional[str] = None
    #: Use cases delivered alongside this one for the same client.
    sibling_ids: List[str] = field(default_factory=list)
    #: The roles the PM says are in play. ``None`` means "let the history
    #: decide" — each role is included at the rate comparable work used it. A
    #: list is a statement of knowledge and overrides that: the named roles are
    #: staffed, and every other role on the roster is not. Naming roles is how
    #: a PM says "this one needs a data scientist and no change manager"
    #: without editing the roster, which is organisation-wide configuration.
    roles: Optional[List[str]] = None
    #: Days per role entered by hand, which win outright over any prediction.
    #: For the role the corpus has never seen, and for the one the PM has
    #: already negotiated.
    role_days: Dict[str, float] = field(default_factory=dict)
    client: str = ""
    encoder: str = "manual"
    rationale: str = ""
    #: What the encoder had to guess rather than read. A defaulted industry is
    #: not a small thing — it is a feature in every head — so it travels with
    #: the use case and every surface that shows a number shows these too.
    assumptions: List[str] = field(default_factory=list)

    @property
    def staffing_is_specified(self) -> bool:
        """True when the PM has said who is on this, rather than leaving it
        to the base rates from comparable work."""
        return self.roles is not None or bool(self.role_days)

    def __post_init__(self) -> None:
        self.counts = normalise_counts(self.counts)
        if self.industry not in INDUSTRIES:
            raise ValueError(
                f"unknown industry {self.industry!r}; expected one of "
                + ", ".join(INDUSTRIES))
        if self.goal not in GOALS:
            raise ValueError(
                f"unknown goal {self.goal!r}; expected one of " + ", ".join(GOALS))

    @property
    def total_units(self) -> int:
        return sum(self.counts.values())

    @property
    def is_continuation(self) -> bool:
        return self.parent_id is not None

    def notation(self) -> str:
        """The use case written in bricks, e.g. ``3xExtract + Reconcile``."""
        from token_yield.tasks import PRIMITIVES
        parts = [(s, self.counts[s]) for s in BRICKS if self.counts.get(s)]
        if not parts:
            return "(nothing)"
        return " + ".join(f"{n}x{PRIMITIVES[s].name}" if n > 1
                          else PRIMITIVES[s].name for s, n in parts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id, "title": self.title, "description": self.description,
            "industry": self.industry, "goal": self.goal, "counts": self.counts,
            "context_bytes": self.context_bytes,
            "monthly_runs": self.monthly_runs, "parent_id": self.parent_id,
            "sibling_ids": list(self.sibling_ids), "client": self.client,
            "encoder": self.encoder, "rationale": self.rationale,
            "assumptions": list(self.assumptions),
            "roles": list(self.roles) if self.roles is not None else None,
            "role_days": dict(self.role_days),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "UseCase":
        return cls(
            id=str(d.get("id", "")), title=str(d.get("title", "")),
            description=str(d.get("description", "")),
            industry=str(d.get("industry", "financial_services")),
            goal=str(d.get("goal", "cost_reduction")),
            counts=dict(d.get("counts") or {}),
            context_bytes=int(d.get("context_bytes", 0) or 0),
            monthly_runs=int(d.get("monthly_runs", 0) or 0),
            parent_id=(str(d["parent_id"]) if d.get("parent_id") else None),
            sibling_ids=[str(s) for s in (d.get("sibling_ids") or [])],
            client=str(d.get("client", "")),
            encoder=str(d.get("encoder", "manual")),
            rationale=str(d.get("rationale", "")),
            assumptions=[str(a) for a in (d.get("assumptions") or [])],
            roles=([str(r) for r in d["roles"]]
                   if d.get("roles") is not None else None),
            role_days={str(k): float(v)
                       for k, v in (d.get("role_days") or {}).items()},
        )
