"""Turning "this continues that" into numbers a model can be fitted on.

A PM linking a new use case to an earlier one is stating something with real
economic content — the connectors exist, the data access is signed, the client
has seen it work — and the prototype's job is to price that rather than to
nod at it. This module turns a lineage graph into three features:

``reuse_depth``
    How many generations upstream the use case sits. 0 is greenfield, 1 is a
    continuation, 2 is a continuation of a continuation. Capped, because the
    difference between the fifth and sixth generation is not real.

``sibling_count``
    How many use cases run alongside this one for the same client. Siblings cut
    both ways — shared setup, contended people — so the sign is left to the fit.

``inherited_fraction``
    The share of this use case's task bricks that already appear somewhere
    upstream, in 0..1. This is the feature that actually carries the reuse
    signal: a "continuation" that is 90% new bricks is a new project wearing a
    continuation's badge, and this number is what says so.

None of the three is given a coefficient here. They go into the feature vector
and every head is free to price them at zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .usecase import BRICKS, UseCase

#: Beyond this many generations, further ancestry stops carrying information.
MAX_DEPTH = 4


@dataclass(frozen=True)
class LineageFeatures:
    """The lineage of one use case, reduced to model inputs."""

    reuse_depth: int
    sibling_count: int
    inherited_fraction: float
    ancestor_ids: Tuple[str, ...] = ()

    def explain(self) -> str:
        if not self.ancestor_ids and not self.sibling_count:
            return "greenfield: no upstream use case, no siblings"
        bits = []
        if self.ancestor_ids:
            bits.append(f"generation {self.reuse_depth} of "
                        + " <- ".join(self.ancestor_ids))
            bits.append(f"{self.inherited_fraction:.0%} of its bricks already "
                        "appear upstream")
        if self.sibling_count:
            bits.append(f"{self.sibling_count} sibling use case(s) alongside")
        return "; ".join(bits)


GREENFIELD = LineageFeatures(0, 0, 0.0)


class LineageIndex:
    """A lookup over known use cases, resolving parents, siblings and neighbours.

    Built from whatever is known — the historical corpus, plus any use cases
    already scoped in this session — because a PM linking today's use case to
    one they scoped an hour ago should get the same treatment as one that
    closed last year.
    """

    def __init__(self, known: Iterable[UseCase] = ()) -> None:
        self._by_id: Dict[str, UseCase] = {}
        for uc in known:
            self.add(uc)

    def add(self, usecase: UseCase) -> None:
        self._by_id[usecase.id] = usecase

    def __contains__(self, uid: str) -> bool:
        return uid in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, uid: str) -> Optional[UseCase]:
        return self._by_id.get(uid)

    def all(self) -> List[UseCase]:
        return list(self._by_id.values())

    # -- the graph -------------------------------------------------------

    def ancestors(self, usecase: UseCase) -> List[UseCase]:
        """The parent chain, nearest first, stopping at a cycle or the cap."""
        out: List[UseCase] = []
        seen = {usecase.id}
        cur = usecase.parent_id
        while cur and cur not in seen and len(out) < MAX_DEPTH:
            parent = self._by_id.get(cur)
            if parent is None:
                break
            out.append(parent)
            seen.add(cur)
            cur = parent.parent_id
        return out

    def siblings(self, usecase: UseCase) -> List[UseCase]:
        """Declared siblings, plus other children of the same parent.

        A PM naming one sibling should not be penalised relative to a PM who
        names all of them, so the graph fills in what it can infer.
        """
        found: Dict[str, UseCase] = {}
        for sid in usecase.sibling_ids:
            sib = self._by_id.get(sid)
            if sib is not None and sib.id != usecase.id:
                found[sib.id] = sib
        if usecase.parent_id:
            for other in self._by_id.values():
                if other.id != usecase.id and other.parent_id == usecase.parent_id:
                    found[other.id] = other
        return list(found.values())

    # -- the features ----------------------------------------------------

    def features_for(self, usecase: UseCase) -> LineageFeatures:
        ancestors = self.ancestors(usecase)
        siblings = self.siblings(usecase)

        upstream: Dict[str, int] = {slug: 0 for slug in BRICKS}
        for anc in ancestors:
            for slug in BRICKS:
                upstream[slug] = max(upstream[slug], anc.counts.get(slug, 0))

        total = usecase.total_units
        if total:
            covered = sum(min(usecase.counts.get(s, 0), upstream[s])
                          for s in BRICKS)
            inherited = covered / total
        else:
            inherited = 0.0

        return LineageFeatures(
            reuse_depth=min(len(ancestors), MAX_DEPTH),
            sibling_count=len(siblings),
            inherited_fraction=inherited,
            ancestor_ids=tuple(a.id for a in ancestors),
        )

    # -- similarity ------------------------------------------------------

    def nearest(self, usecase: UseCase, k: int = 5,
                exclude_self: bool = True
                ) -> List[Tuple[UseCase, float, Tuple[str, ...]]]:
        """The most similar known use cases, most similar first.

        Ranked by cosine similarity over the brick vector, nudged by industry,
        goal and client agreement. Returns ``(use case, brick cosine, what else
        matched)`` — the bare cosine rather than the boosted score, and the
        reasons alongside it, so a row that outranks a geometrically closer one
        can be seen to have earned it rather than looking like a sort bug.

        This is the "show me the comparable deals" panel. In a deployed system
        it is the query a vector index (Azure AI Search) would serve rather
        than a scan; the scoring is kept here so the prototype needs no service
        to run.
        """
        scored = []
        for other in self._by_id.values():
            if exclude_self and other.id == usecase.id:
                continue
            cos = _cosine(usecase.counts, other.counts)
            rank, why = cos, []
            for bonus, label, same in (
                    (0.10, "industry", other.industry == usecase.industry),
                    (0.05, "goal", other.goal == usecase.goal),
                    (0.05, "client",
                     bool(other.client) and other.client == usecase.client)):
                if same:
                    rank += bonus
                    why.append(label)
            scored.append((other, cos, tuple(why), rank))
        scored.sort(key=lambda t: (-t[3], t[0].id))
        return [(uc, cos, why) for uc, cos, why, _ in scored[:k]]


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    num = sum(a.get(s, 0) * b.get(s, 0) for s in BRICKS)
    na = math.sqrt(sum(a.get(s, 0) ** 2 for s in BRICKS))
    nb = math.sqrt(sum(b.get(s, 0) ** 2 for s in BRICKS))
    return num / (na * nb) if na and nb else 0.0


def index_from_engagements(engagements: Sequence["object"]) -> LineageIndex:
    """Build the index from a corpus of :class:`~project_yield.corpus.Engagement`."""
    return LineageIndex(e.as_usecase() for e in engagements)
