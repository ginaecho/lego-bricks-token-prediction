"""Prediction engine — from calibrated baselines to scaled/combined estimates.

Three prediction modes:
1. **Single** — predict tokens for one task at a given complexity tier.
2. **Scaled** — predict A+, A++ by applying complexity multipliers to
   measured A.
3. **Combined** — predict A + B + C with an interaction overhead factor
   that accounts for context switching, shared setup, and cross-task
   dependencies.
"""

from __future__ import annotations

import math
from typing import Optional

from .calibrate import CalibrationStore
from .models import (
    CalibrationRecord,
    ComplexityTier,
    TaskPrediction,
    TaskTypeStats,
)


class TokenPredictor:
    """Predicts token usage from calibrated task-type statistics."""

    def __init__(self, store: CalibrationStore,
                 custom_multipliers: Optional[dict[str, dict[ComplexityTier, float]]] = None) -> None:
        self._store = store
        self._stats_cache: dict[str, TaskTypeStats] = {}
        self._multipliers: dict[str, dict[ComplexityTier, float]] = custom_multipliers or {}

    def _get_stats(self, task_type: str) -> Optional[TaskTypeStats]:
        if task_type not in self._stats_cache:
            s = self._store.stats(task_type)
            if s is not None:
                self._stats_cache[task_type] = s
        return self._stats_cache.get(task_type)

    def multiplier_for(self, task_type: str, complexity: ComplexityTier) -> float:
        if task_type in self._multipliers and complexity in self._multipliers[task_type]:
            return self._multipliers[task_type][complexity]
        return complexity.default_multiplier

    def predict_single(self, task_type: str,
                       complexity: ComplexityTier = ComplexityTier.BASE,
                       custom_multiplier: Optional[float] = None) -> Optional[TaskPrediction]:
        """Predict token usage for a single task at a given complexity."""
        stats = self._get_stats(task_type)
        if stats is None:
            return None

        mult = custom_multiplier if custom_multiplier is not None else self.multiplier_for(task_type, complexity)

        predicted = int(stats.mean_tokens * mult)
        predicted_dur = stats.mean_duration * mult

        # confidence interval: scale the stddev by multiplier too
        scaled_std = stats.stddev_tokens * mult
        if stats.sample_count >= 2:
            se = scaled_std / math.sqrt(stats.sample_count)
            margin = 1.96 * se
        else:
            margin = predicted * 0.5

        low = max(0, int(predicted - margin))
        high = int(predicted + margin)
        harness = int(stats.mean_harness_tokens * mult)

        return TaskPrediction(
            task_type=task_type,
            complexity=complexity,
            multiplier=mult,
            predicted_tokens=predicted,
            predicted_duration=predicted_dur,
            confidence_low=low,
            confidence_high=high,
            basis_samples=stats.sample_count,
            harness_overhead=harness,
        )

    def predict_scaled(self, task_type: str) -> dict[ComplexityTier, TaskPrediction]:
        """Predict all standard complexity tiers for a task type."""
        results = {}
        for tier in (ComplexityTier.BASE, ComplexityTier.PLUS, ComplexityTier.PLUS_PLUS):
            pred = self.predict_single(task_type, tier)
            if pred is not None:
                results[tier] = pred
        return results

    def predict_combined(self, task_specs: list[tuple[str, ComplexityTier, int]],
                         interaction_overhead: float = 0.15) -> list[TaskPrediction]:
        """Predict token usage for a combined workload.

        Each spec is (task_type, complexity, count).  The interaction overhead
        accounts for cross-task context: switching, shared setup, dependency
        chains.  It is applied proportionally to distinct-type count.
        """
        predictions = []
        distinct_types = len(set(tt for tt, _, _ in task_specs))

        for task_type, complexity, count in task_specs:
            base_pred = self.predict_single(task_type, complexity)
            if base_pred is None:
                continue

            # interaction overhead scales with the number of distinct task types
            overhead_factor = 1.0 + interaction_overhead * max(0, distinct_types - 1)

            scaled_tokens = int(base_pred.predicted_tokens * count * overhead_factor)
            scaled_dur = base_pred.predicted_duration * count * overhead_factor
            scaled_low = int(base_pred.confidence_low * count * overhead_factor)
            scaled_high = int(base_pred.confidence_high * count * overhead_factor)
            scaled_harness = int(base_pred.harness_overhead * count * overhead_factor)

            predictions.append(TaskPrediction(
                task_type=task_type,
                complexity=complexity,
                multiplier=base_pred.multiplier,
                predicted_tokens=scaled_tokens,
                predicted_duration=scaled_dur,
                confidence_low=scaled_low,
                confidence_high=scaled_high,
                basis_samples=base_pred.basis_samples,
                harness_overhead=scaled_harness,
            ))

        return predictions

    def compare_scenarios(self, scenarios: dict[str, list[tuple[str, ComplexityTier, int]]],
                          interaction_overhead: float = 0.15) -> dict[str, int]:
        """Compare total predicted tokens across named scenarios."""
        totals = {}
        for name, specs in scenarios.items():
            preds = self.predict_combined(specs, interaction_overhead)
            totals[name] = sum(p.total_predicted for p in preds)
        return totals
