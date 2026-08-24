"""Predicting **hours**, with the honesty that wall-clock deserves.

Budgets are asked for in two currencies: tokens and time. Tokens are the
well-behaved one. Identical tasks, repeated, differ by about 5% in tokens and
about **23% in wall-clock** — nearly five times noisier. Duration absorbs
queueing, tool latency, retries, and how chatty a particular run happened to
be, none of which the work itself determines.

So hours are predicted here with exactly the same machinery as tokens — the
same four forms, the same signal search, the same cross-validation — and
reported with the same skill ratio, which is what makes the gap visible instead
of hidden. A predicted hour figure that does not carry its own error bar is a
worse answer than no figure at all.

Implementation note: rather than duplicating every fitter, a record's duration
is moved into the field the fitters already read, in milliseconds. The models
that come back therefore predict **milliseconds**; :func:`predict_seconds` and
:func:`predict_hours` convert.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional, Sequence

from .costmodel import CostModel, Selection, select_model
from .taxonomy import ScopedRecord

MS_PER_HOUR = 3_600_000.0


def as_duration_records(records: Iterable[ScopedRecord]) -> list:
    """Re-point records at duration, so the token fitters can be reused verbatim.

    Milliseconds keep the values integral and comfortably above zero, which the
    power fit's log transform requires.
    """
    out = []
    for r in records:
        ms = max(int(round(r.duration_seconds * 1000)), 1)
        out.append(replace(r, tokens=ms))
    return out


def duration_selection(kind: str, records: Sequence[ScopedRecord]) -> Optional[Selection]:
    """Fit a duration model for one kind. Predictions are in milliseconds."""
    rs = [r for r in records if r.kind == kind and r.duration_seconds > 0]
    if not rs:
        return None
    return select_model(kind, as_duration_records(rs))


def predict_seconds(model: CostModel, x: float) -> float:
    return model.predict(x) / 1000.0


def predict_hours(model: CostModel, x: float) -> float:
    return model.predict(x) / MS_PER_HOUR


def seconds_for(selection: Optional[Selection], signals: dict) -> Optional[float]:
    """Seconds for a task described by ``signals``, reading the model's own signal.

    A duration model routinely selects a *different* explanatory signal than the
    token model for the same kind — comprehension tokens track bytes read while
    its wall-clock tracks file count, because each file is a separate tool call.
    Passing one model's input to the other silently produces nonsense, so the
    signal lookup happens here rather than at every call site.
    """
    if selection is None:
        return None
    m = selection.model
    x = signals.get(m.signal)
    if x is None or x <= 0:
        return None
    return predict_seconds(m, x)


def duration_models(records: Sequence[ScopedRecord]) -> dict:
    """A duration model per kind, for every kind with timed runs."""
    out = {}
    for kind in sorted({r.kind for r in records}):
        sel = duration_selection(kind, records)
        if sel is not None:
            out[kind] = sel
    return out
