"""Token Yield — predict token usage and budget for business projects.

The prediction pipeline:

1. **Name the work.** A fixed vocabulary of base business tasks — Review,
   Extract, Classify, Retrieve, Reconcile, Draft, Remediate, Validate, Report —
   each one a task type enterprises already buy agents to do (:mod:`tasks`).
2. **Measure them.** Run each base task at more than one size against real
   documents, and record what it actually cost (:mod:`trainsuite`).
3. **Compose.** Fit what *combinations* cost. Not a naive sum: an agent pays a
   large start-up cost once per invocation, so batching several base tasks into
   one agent is far cheaper than running them apart (:mod:`compose`).
4. **Decompose.** Express a request nobody has run in that vocabulary, then
   recompose it through the fitted model to price it in advance
   (:mod:`decompose`).

Everything the model rests on is measured. Nothing is asserted.
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
from .backtest import noise_floor  # NB: `backtest` stays a module, unshadowed
from .duration import duration_selection, predict_hours, seconds_for

# The compositional layer: named base tasks, and pricing their combinations.
from .tasks import ORDER, PRIMITIVES, Primitive, TaskSpec
from .compose import (
    CompositionModel, Run, batching_saving, default_runs_path, load_runs,
    select_model as select_composition_model,
)
from .decompose import (
    Decomposition, decompose_prompt, explain, heuristic_decompose,
    parse_decomposition, price, reconstruction_error,
)

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
    "noise_floor",
    "duration_selection",
    "predict_hours",
    "seconds_for",
    # compositional path (named base tasks -> combinations -> unseen requests)
    "Primitive",
    "PRIMITIVES",
    "ORDER",
    "TaskSpec",
    "Run",
    "CompositionModel",
    "load_runs",
    "default_runs_path",
    "select_composition_model",
    "batching_saving",
    "Decomposition",
    "decompose_prompt",
    "parse_decomposition",
    "heuristic_decompose",
    "price",
    "explain",
    "reconstruction_error",
]
