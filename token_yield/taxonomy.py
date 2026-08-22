"""What *kind* of task is this, and how much of it is there?

The original calibration layer keyed everything on a bare task-type string and
had no notion of *how much work* a task contained. That made "A+" and "A++"
unanswerable except by decree — hence the hardcoded 2× and 4×.

This module supplies the two things a fitted cost model actually needs:

* a **kind** — an open registry, because kinds are discovered from real work,
  not fixed in advance by whoever wrote the enum; and
* a **scope** — a number saying how many units of that kind's work the task
  contains, where each kind declares what one unit means.

With those, "A+" stops being a decree and becomes a point on a curve you fit.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable, Optional


class Provenance(enum.Enum):
    """Where a measurement came from. Kept on every record, and never inferred.

    The repository's standing rule is that synthetic data is labelled as such.
    A cost model fitted on invented numbers must not be mistakable for one
    fitted on measured runs.
    """

    PROBE = "probe"            # a real agent run, dispatched to measure it
    PRODUCTION = "production"  # a real agent run doing real work
    SYNTHETIC = "synthetic"    # invented; illustrative only

    @property
    def is_measured(self) -> bool:
        return self is not Provenance.SYNTHETIC


@dataclass(frozen=True)
class TaskKind:
    """A category of work, plus the unit its scope is counted in."""

    id: str
    scope_unit: str
    summary: str = ""

    def describe(self, scope: float) -> str:
        n = int(scope) if float(scope).is_integer() else scope
        unit = self.scope_unit if scope == 1 else self.scope_unit + "s"
        return f"{self.id} ({n} {unit})"


class KindRegistry:
    """An open registry of task kinds.

    Open on purpose: the point of the learning loop is that new kinds show up
    in real traffic. A closed enum would force every new kind to be a code
    change before it could be measured.
    """

    def __init__(self) -> None:
        self._kinds: dict[str, TaskKind] = {}

    def register(self, id: str, scope_unit: str, summary: str = "") -> TaskKind:
        kind = TaskKind(id=id, scope_unit=scope_unit, summary=summary)
        self._kinds[id] = kind
        return kind

    def get(self, id: str) -> Optional[TaskKind]:
        return self._kinds.get(id)

    def ensure(self, id: str, scope_unit: str = "unit") -> TaskKind:
        """Return the kind, registering an unknown one rather than rejecting it."""
        return self._kinds.get(id) or self.register(id, scope_unit)

    def __contains__(self, id: object) -> bool:
        return id in self._kinds

    def __iter__(self) -> Iterable[TaskKind]:
        return iter(sorted(self._kinds.values(), key=lambda k: k.id))

    def __len__(self) -> int:
        return len(self._kinds)


#: The kinds the shipped probe suite measures. Not a closed list — a project
#: registers its own, and unknown kinds seen in traffic register themselves.
KINDS = KindRegistry()
COMPREHENSION = KINDS.register(
    "comprehension", "file",
    "read existing code and answer a question about it")
CODE_WRITE = KINDS.register(
    "code_write", "function",
    "write new code to a specification, no repo reading")


@dataclass(frozen=True)
class ScopedRecord:
    """One measured run: a kind, how much of it, and what it actually cost.

    This is the atom the cost models are fitted on. Unlike the original
    ``CalibrationRecord`` it carries ``scope``, which is what makes a *slope*
    estimable at all, and ``provenance``, which is what keeps a synthetic point
    from silently becoming evidence.
    """

    kind: str
    scope: float
    tokens: int
    duration_seconds: float = 0.0
    tool_uses: int = 0
    provenance: Provenance = Provenance.SYNTHETIC
    label: str = ""

    def __post_init__(self) -> None:
        if self.scope <= 0:
            raise ValueError(f"scope must be positive, got {self.scope}")
        if self.tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {self.tokens}")


def measured_only(records: Iterable[ScopedRecord]) -> list[ScopedRecord]:
    """Drop synthetic records — for when a claim has to rest on real runs."""
    return [r for r in records if r.provenance.is_measured]
