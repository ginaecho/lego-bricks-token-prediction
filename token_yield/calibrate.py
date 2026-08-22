"""Calibration — collect actual measurements and compute per-type statistics.

Feed it records from real task runs (or from the benchmark harness) and it
produces the TaskTypeStats that the predictor needs.  The store can also
ingest OpenHarness traces directly: each observation carries tokens and
task_type, so the bridge is straightforward.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

from .models import CalibrationRecord, TaskTypeStats


class CalibrationStore:
    """Accumulates calibration records and computes statistics per task type."""

    def __init__(self) -> None:
        self._records: dict[str, list[CalibrationRecord]] = defaultdict(list)

    def add(self, record: CalibrationRecord) -> None:
        self._records[record.task_type].append(record)

    def add_many(self, records: Iterable[CalibrationRecord]) -> None:
        for r in records:
            self.add(r)

    def from_observations(self, observations: Iterable) -> None:
        """Ingest OpenHarness Observation objects as calibration data.

        Each observation is one harness check.  We group by (task_type, task_id)
        and sum the tokens per task to get one CalibrationRecord per task.
        """
        task_tokens: dict[tuple[str, str], int] = defaultdict(int)
        task_types: dict[tuple[str, str], str] = {}
        for obs in observations:
            key = (obs.task_type, obs.task_id)
            task_tokens[key] += obs.tokens
            task_types[key] = obs.task_type
        for key, tokens in task_tokens.items():
            self.add(CalibrationRecord(
                task_type=task_types[key],
                tokens_used=tokens,
                label=f"from observation {key[1]}",
            ))

    @property
    def task_types(self) -> list[str]:
        return sorted(self._records.keys())

    def records_for(self, task_type: str) -> list[CalibrationRecord]:
        return list(self._records.get(task_type, []))

    def stats(self, task_type: str) -> Optional[TaskTypeStats]:
        """Compute statistics for a single task type."""
        records = self._records.get(task_type)
        if not records:
            return None

        tokens = [r.total_tokens for r in records]
        durations = [r.duration_seconds for r in records]
        harness = [r.harness_tokens for r in records]
        successes = sum(1 for r in records if r.success)
        n = len(records)

        mean_t = sum(tokens) / n
        mean_d = sum(durations) / n
        mean_h = sum(harness) / n

        var_t = sum((t - mean_t) ** 2 for t in tokens) / n if n > 1 else 0.0
        var_d = sum((d - mean_d) ** 2 for d in durations) / n if n > 1 else 0.0

        return TaskTypeStats(
            task_type=task_type,
            sample_count=n,
            mean_tokens=mean_t,
            stddev_tokens=math.sqrt(var_t),
            min_tokens=min(tokens),
            max_tokens=max(tokens),
            mean_duration=mean_d,
            stddev_duration=math.sqrt(var_d),
            mean_harness_tokens=mean_h,
            success_rate=successes / n,
        )

    def all_stats(self) -> dict[str, TaskTypeStats]:
        result = {}
        for tt in self.task_types:
            s = self.stats(tt)
            if s is not None:
                result[tt] = s
        return result

    def total_records(self) -> int:
        return sum(len(recs) for recs in self._records.values())
