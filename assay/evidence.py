"""Evidence classes.

Every row and every report carries exactly one. Only ``feasibility`` may move a gate --
this is enforced here rather than by convention, because the whole plan depends on it.
"""

from __future__ import annotations

from enum import Enum


class EvidenceClass(str, Enum):
    REPLAY = "replay"
    """Recomputed from the reference repository's committed data. Reference only."""

    PIPELINE_ONLY = "pipeline_only"
    """Fixture or MockAdapter. Proves the code path; proves nothing about the model."""

    FEASIBILITY = "feasibility"
    """Real dispatch under the locked contract."""

    @property
    def can_move_a_gate(self) -> bool:
        return self is EvidenceClass.FEASIBILITY


def assert_gateable(classes: set[EvidenceClass] | list[EvidenceClass]) -> None:
    """Raise unless every class present is allowed to decide something."""
    bad = sorted({c.value for c in classes if not EvidenceClass(c).can_move_a_gate})
    if bad:
        raise ValueError(
            f"refusing to evaluate a gate on non-feasibility evidence: {', '.join(bad)}"
        )


def mixed(classes: set[EvidenceClass] | list[EvidenceClass]) -> bool:
    return len({EvidenceClass(c) for c in classes}) > 1
