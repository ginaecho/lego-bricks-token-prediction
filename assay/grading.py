"""Blinded quality grading for the batching arm.

Batching is only a saving if the answers survive it. Cheaper output that quietly drops the
last task in the bundle, or fabricates a figure because the context got crowded, is not a
discount -- it is a defect with a discount attached.

Two design rules, both borrowed from clinical practice for the same reason:

* Graders never see which arm produced an answer until after the scores are joined. The
  arm label lives in a separate manifest.
* Agreement between graders is checked *before* unblinding. If they do not agree with each
  other, their verdict on the arms means nothing, and the honest report is "inconclusive"
  rather than a number.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from assay.metrics import bootstrap, krippendorff_alpha_ordinal

RUBRIC: tuple[tuple[str, int], ...] = (
    ("factual accuracy", 45),
    ("completeness", 25),
    ("citations", 15),
    ("no fabrication", 10),
    ("structure", 5),
)
RUBRIC_TOTAL = sum(points for _, points in RUBRIC)

CRITICAL_FAILURES = (
    "wrong numeric value",
    "fabricated fact",
    "missed the central reconciliation",
    "unusable result",
)

ALPHA_FLOOR = 0.67


@dataclass(frozen=True)
class Grade:
    answer_id: str
    grader: str
    score: float
    critical_failure: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.score <= RUBRIC_TOTAL:
            raise ValueError(f"score must be within 0..{RUBRIC_TOTAL}, got {self.score}")
        if self.critical_failure and self.critical_failure not in CRITICAL_FAILURES:
            raise ValueError(f"unknown critical failure: {self.critical_failure!r}")


@dataclass
class BlindingManifest:
    """The only place the arm label lives until scores are in."""

    assignment: dict[str, str]  # answer_id -> "separate" | "batched"

    def hash(self) -> str:
        payload = "|".join(f"{k}:{v}" for k, v in sorted(self.assignment.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def arm(self, answer_id: str) -> str:
        return self.assignment[answer_id]

    @classmethod
    def shuffled(cls, arms: Mapping[str, str], seed: int = 1337) -> "BlindingManifest":
        """Randomise presentation order so a grader cannot infer the arm from position."""
        items = list(arms.items())
        random.Random(seed).shuffle(items)
        return cls(assignment=dict(items))


@dataclass
class QualityReport:
    alpha: float
    conclusive: bool
    mean_separate: float
    mean_batched: float
    delta_q: float
    delta_lower_90: float
    critical_separate: int
    critical_batched: int
    n_bundles: int
    notes: list[str] = field(default_factory=list)

    @property
    def non_inferior(self) -> bool:
        """A cheaper worse answer is not a saving."""
        return (
            self.conclusive
            and self.delta_lower_90 >= -5.0
            and self.critical_batched <= self.critical_separate + 1
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": round(self.alpha, 4),
            "conclusive": self.conclusive,
            "mean_separate": round(self.mean_separate, 3),
            "mean_batched": round(self.mean_batched, 3),
            "delta_q": round(self.delta_q, 3),
            "delta_lower_90": round(self.delta_lower_90, 3),
            "critical_separate": self.critical_separate,
            "critical_batched": self.critical_batched,
            "non_inferior": self.non_inferior,
            "n_bundles": self.n_bundles,
            "notes": self.notes,
        }


def grader_agreement(grades: Sequence[Grade]) -> float:
    """Krippendorff's alpha over graders, computed before any unblinding."""
    by_answer: dict[str, dict[str, float]] = {}
    graders: list[str] = []
    for g in grades:
        by_answer.setdefault(g.answer_id, {})[g.grader] = g.score
        if g.grader not in graders:
            graders.append(g.grader)
    units = [
        [
            int(round(by_answer[a][gr])) if gr in by_answer[a] else None
            for gr in graders
        ]
        for a in sorted(by_answer)
    ]
    return krippendorff_alpha_ordinal(units)


def score_quality(
    grades: Sequence[Grade],
    manifest: BlindingManifest,
    bundle_of: Mapping[str, str],
    *,
    iters: int = 2000,
    seed: int = 1337,
) -> QualityReport:
    """Join grades to arms, but only after agreement has been checked."""
    notes: list[str] = []
    alpha = grader_agreement(grades)
    conclusive = alpha >= ALPHA_FLOOR
    if not conclusive:
        notes.append(
            f"grader agreement {alpha:.2f} is below {ALPHA_FLOOR}; the quality arm is "
            "inconclusive and batching cannot be recommended on this evidence"
        )

    mean_by_answer: dict[str, float] = {}
    critical_by_answer: dict[str, bool] = {}
    for g in grades:
        mean_by_answer.setdefault(g.answer_id, 0.0)
        critical_by_answer.setdefault(g.answer_id, False)
    for answer in mean_by_answer:
        scores = [g.score for g in grades if g.answer_id == answer]
        mean_by_answer[answer] = sum(scores) / len(scores)
        critical_by_answer[answer] = any(
            g.critical_failure for g in grades if g.answer_id == answer
        )

    paired: list[tuple[float, float]] = []
    bundles: dict[str, dict[str, float]] = {}
    for answer, mean in mean_by_answer.items():
        bundles.setdefault(bundle_of[answer], {})[manifest.arm(answer)] = mean
    for arms in bundles.values():
        if "separate" in arms and "batched" in arms:
            paired.append((arms["separate"], arms["batched"]))

    if not paired:
        return QualityReport(
            alpha=alpha, conclusive=False, mean_separate=0.0, mean_batched=0.0,
            delta_q=0.0, delta_lower_90=-100.0, critical_separate=0, critical_batched=0,
            n_bundles=0, notes=notes + ["no bundle had both arms graded"],
        )

    res = bootstrap(
        paired,
        lambda s: sum(b - a for a, b in s) / len(s),  # type: ignore[misc]
        iters=iters,
        level=0.90,
        seed=seed,
    )

    crit_sep = sum(
        1 for a, bad in critical_by_answer.items() if bad and manifest.arm(a) == "separate"
    )
    crit_bat = sum(
        1 for a, bad in critical_by_answer.items() if bad and manifest.arm(a) == "batched"
    )

    return QualityReport(
        alpha=alpha,
        conclusive=conclusive,
        mean_separate=sum(a for a, _ in paired) / len(paired),
        mean_batched=sum(b for _, b in paired) / len(paired),
        delta_q=res.point,
        delta_lower_90=res.lo,
        critical_separate=crit_sep,
        critical_batched=crit_bat,
        n_bundles=len(paired),
        notes=notes,
    )
