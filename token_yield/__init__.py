"""Token Yield — predict token usage and budget for business projects.

The prediction pipeline:
1. **Calibrate** — run baseline task types (A, B, C) and record actual token
   usage, duration, and harness overhead.
2. **Model** — fit per-type statistics and complexity scaling factors so that
   A+ and A++ can be inferred from measured A.
3. **Compose** — predict combined workloads (A + B + C) with interaction
   overhead, not just a naive sum.
4. **Forecast** — given a project specification, output a budget: predicted
   tokens, estimated cost, time, and confidence interval.
"""

from .models import (
    ComplexityTier,
    TaskTypeStats,
    CalibrationRecord,
    TaskPrediction,
    ProjectSpec,
    ProjectForecast,
)
from .calibrate import CalibrationStore
from .predict import TokenPredictor
from .forecast import ProjectForecaster

# The fitted layer: models learned from measured runs rather than asserted.
from .taxonomy import KINDS, Provenance, ScopedRecord, TaskKind
from .costmodel import (
    AffineModel, ConstantModel, CostModel, PowerModel, ProportionalModel,
    Selection, select_model,
)
from .learn import DriftReport, LearningStore, seeded_store
from .plan import PlanForecast, PlanForecaster, WorkItem, WorkPlan
from .backtest import backtest, noise_floor

__all__ = [
    # tier-based path (assumed multipliers)
    "ComplexityTier",
    "TaskTypeStats",
    "CalibrationRecord",
    "TaskPrediction",
    "ProjectSpec",
    "ProjectForecast",
    "CalibrationStore",
    "TokenPredictor",
    "ProjectForecaster",
    # fitted path (models learned from measurement)
    "KINDS",
    "Provenance",
    "ScopedRecord",
    "TaskKind",
    "CostModel",
    "ConstantModel",
    "ProportionalModel",
    "AffineModel",
    "PowerModel",
    "Selection",
    "select_model",
    "LearningStore",
    "DriftReport",
    "seeded_store",
    "WorkItem",
    "WorkPlan",
    "PlanForecast",
    "PlanForecaster",
    "backtest",
    "noise_floor",
]
