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

__all__ = [
    "ComplexityTier",
    "TaskTypeStats",
    "CalibrationRecord",
    "TaskPrediction",
    "ProjectSpec",
    "ProjectForecast",
    "CalibrationStore",
    "TokenPredictor",
    "ProjectForecaster",
]
