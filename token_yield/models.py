"""Data models for the token yield prediction pipeline.

Vocabulary:
- A *calibration record* is one measured run of a task type.
- *Task type stats* are the aggregated statistics for a task type across
  all calibration records — mean, stddev, min, max.
- A *complexity tier* scales a baseline measurement to predict harder
  variants (A -> A+ -> A++).
- A *task prediction* is the predicted token usage for one task unit.
- A *project spec* describes a business project as a list of task units.
- A *project forecast* is the full budget prediction for a project.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional


class ComplexityTier(enum.Enum):
    """Scaling tiers for task complexity relative to the measured baseline."""

    BASE = "base"
    PLUS = "plus"
    PLUS_PLUS = "plus_plus"
    CUSTOM = "custom"

    @property
    def default_multiplier(self) -> float:
        return {
            ComplexityTier.BASE: 1.0,
            ComplexityTier.PLUS: 2.0,
            ComplexityTier.PLUS_PLUS: 4.0,
            ComplexityTier.CUSTOM: 1.0,
        }[self]


@dataclass(frozen=True)
class CalibrationRecord:
    """One measured run of a task type."""

    task_type: str
    tokens_used: int
    duration_seconds: float = 0.0
    events_count: int = 0
    harness_tokens: int = 0
    success: bool = True
    label: str = ""

    @property
    def total_tokens(self) -> int:
        return self.tokens_used + self.harness_tokens


@dataclass
class TaskTypeStats:
    """Aggregated statistics for a task type from calibration records."""

    task_type: str
    sample_count: int = 0
    mean_tokens: float = 0.0
    stddev_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    mean_duration: float = 0.0
    stddev_duration: float = 0.0
    mean_harness_tokens: float = 0.0
    success_rate: float = 1.0

    @property
    def cv(self) -> float:
        """Coefficient of variation — how spread out the measurements are."""
        if self.mean_tokens == 0:
            return 0.0
        return self.stddev_tokens / self.mean_tokens

    @property
    def confidence_width(self) -> float:
        """95% confidence interval half-width as a fraction of the mean."""
        if self.sample_count < 2 or self.mean_tokens == 0:
            return 1.0
        se = self.stddev_tokens / math.sqrt(self.sample_count)
        return (1.96 * se) / self.mean_tokens


@dataclass(frozen=True)
class TaskPrediction:
    """Predicted token usage for one task unit."""

    task_type: str
    complexity: ComplexityTier
    multiplier: float
    predicted_tokens: int
    predicted_duration: float
    confidence_low: int
    confidence_high: int
    basis_samples: int
    harness_overhead: int = 0

    @property
    def total_predicted(self) -> int:
        return self.predicted_tokens + self.harness_overhead


@dataclass
class TaskUnit:
    """One line item in a project spec: a task type at a complexity, repeated N times."""

    task_type: str
    complexity: ComplexityTier = ComplexityTier.BASE
    count: int = 1
    custom_multiplier: Optional[float] = None

    @property
    def multiplier(self) -> float:
        if self.custom_multiplier is not None:
            return self.custom_multiplier
        return self.complexity.default_multiplier


@dataclass
class ProjectSpec:
    """A business project described as a list of task units."""

    name: str
    tasks: list[TaskUnit] = field(default_factory=list)
    interaction_overhead: float = 0.15

    def add(self, task_type: str, complexity: ComplexityTier = ComplexityTier.BASE,
            count: int = 1, custom_multiplier: Optional[float] = None) -> "ProjectSpec":
        self.tasks.append(TaskUnit(task_type, complexity, count, custom_multiplier))
        return self

    @property
    def total_task_count(self) -> int:
        return sum(t.count for t in self.tasks)


@dataclass(frozen=True)
class ProjectForecast:
    """The full budget prediction for a business project."""

    project_name: str
    task_predictions: tuple[TaskPrediction, ...]
    task_counts: tuple[int, ...]
    interaction_overhead_rate: float
    total_tokens: int
    total_tokens_low: int
    total_tokens_high: int
    estimated_duration_seconds: float
    interaction_overhead_tokens: int
    uncalibrated: tuple[str, ...] = ()
    """Task types in the spec with no calibration data, so absent from the totals.

    A budget that quietly omits part of the project is worse than no budget, so
    the omission travels with the forecast and every report prints it.
    """

    @property
    def is_complete(self) -> bool:
        """True when every task type in the spec could actually be priced."""
        return not self.uncalibrated

    @property
    def total_with_overhead(self) -> int:
        return self.total_tokens + self.interaction_overhead_tokens

    def cost_at_rate(self, dollars_per_million_tokens: float) -> float:
        return self.total_with_overhead * dollars_per_million_tokens / 1_000_000

    def cost_range(self, dollars_per_million_tokens: float) -> tuple[float, float]:
        low = self.total_tokens_low * dollars_per_million_tokens / 1_000_000
        high_with_overhead = int(self.total_tokens_high * (1 + self.interaction_overhead_rate))
        high = high_with_overhead * dollars_per_million_tokens / 1_000_000
        return low, high

    @property
    def estimated_hours(self) -> float:
        return self.estimated_duration_seconds / 3600
