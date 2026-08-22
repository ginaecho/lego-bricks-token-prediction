"""Project-level forecasting — from a project spec to a budget.

Takes a ProjectSpec (a list of task units with types, complexities, and
counts) and produces a ProjectForecast with total tokens, cost estimates
at configurable rates, time estimates, and confidence intervals.
"""

from __future__ import annotations

from typing import Optional

from .calibrate import CalibrationStore
from .models import (
    ProjectForecast,
    ProjectSpec,
    TaskPrediction,
)
from .predict import TokenPredictor


class ProjectForecaster:
    """End-to-end forecaster: project spec -> budget."""

    def __init__(self, store: CalibrationStore,
                 predictor: Optional[TokenPredictor] = None) -> None:
        self._store = store
        self._predictor = predictor or TokenPredictor(store)

    def forecast(self, spec: ProjectSpec) -> ProjectForecast:
        """Produce a full forecast from a project specification."""
        task_specs = [
            (unit.task_type, unit.complexity, unit.count)
            for unit in spec.tasks
        ]

        predictions = self._predictor.predict_combined(
            task_specs,
            interaction_overhead=spec.interaction_overhead,
        )

        counts = tuple(unit.count for unit in spec.tasks)

        total = sum(p.total_predicted for p in predictions)
        total_low = sum(p.confidence_low for p in predictions)
        total_high = sum(p.confidence_high + p.harness_overhead for p in predictions)
        total_dur = sum(p.predicted_duration for p in predictions)

        base_tokens = sum(p.predicted_tokens for p in predictions)
        overhead_tokens = sum(p.harness_overhead for p in predictions)

        return ProjectForecast(
            project_name=spec.name,
            task_predictions=tuple(predictions),
            task_counts=counts,
            interaction_overhead_rate=spec.interaction_overhead,
            total_tokens=base_tokens,
            total_tokens_low=total_low,
            total_tokens_high=total_high,
            estimated_duration_seconds=total_dur,
            interaction_overhead_tokens=overhead_tokens,
        )

    def forecast_with_cost(self, spec: ProjectSpec,
                           dollars_per_million_tokens: float = 3.0) -> dict:
        """Produce a forecast plus cost breakdown as a plain dict."""
        fc = self.forecast(spec)
        cost = fc.cost_at_rate(dollars_per_million_tokens)
        cost_low, cost_high = fc.cost_range(dollars_per_million_tokens)

        return {
            "project": fc.project_name,
            "total_tokens": fc.total_with_overhead,
            "total_tokens_base": fc.total_tokens,
            "harness_overhead_tokens": fc.interaction_overhead_tokens,
            "confidence_interval": {
                "low": fc.total_tokens_low,
                "high": fc.total_tokens_high,
            },
            "estimated_cost": {
                "rate_per_million": dollars_per_million_tokens,
                "estimated": round(cost, 4),
                "range_low": round(cost_low, 4),
                "range_high": round(cost_high, 4),
            },
            "estimated_hours": round(fc.estimated_hours, 2),
            "task_breakdown": [
                {
                    "task_type": p.task_type,
                    "complexity": p.complexity.value,
                    "multiplier": p.multiplier,
                    "tokens": p.total_predicted,
                    "duration_s": round(p.predicted_duration, 1),
                    "samples": p.basis_samples,
                }
                for p in fc.task_predictions
            ],
        }
