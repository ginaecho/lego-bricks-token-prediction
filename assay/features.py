"""The model forms -- which are also the baseline ladder.

This is the load-bearing idea of the whole plan. The six nested forms the reference used to
*select* a model are exactly the baselines a decoder has to *beat*, so there is one ladder,
not two, and "did bricks help?" is answered by climbing it:

    boot -> constant -> units -> bytes -> bytes_units -> log_bytes_units │ per_brick -> bytes_per_brick

Everything left of the bar is a baseline: it knows nothing about which bricks were asked
for, only how much material there was and how many things were wanted. Everything right of
it is a decoder candidate. A decoder that cannot beat ``bytes_units`` has not shown that
brick decomposition carries information -- it has shown that size does.

All forms are fitted against ``y`` (billable units) with an intercept, so they stay nested
and directly comparable. The variable-portion score subtracts the *measured* boot from both
sides afterwards; it is not a different fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from assay.schemas import Run
from assay.vocabulary import Vocabulary


@dataclass(frozen=True)
class Form:
    key: str
    role: str  # "baseline" | "decoder" | "diagnostic"
    solver: str  # "ols" | "nnls" | "fixed" | "cell"
    description: str
    uses_bytes: bool = False
    uses_log_bytes: bool = False
    uses_total_units: bool = False
    uses_per_brick: bool = False
    has_intercept: bool = True

    def columns(self, vocab: Vocabulary) -> tuple[str, ...]:
        cols: list[str] = []
        if self.solver in {"fixed", "cell"}:
            return ()
        if self.has_intercept:
            cols.append("intercept")
        if self.uses_bytes:
            cols.append("context_bytes")
        if self.uses_log_bytes:
            cols.append("log_context_bytes")
        if self.uses_total_units:
            cols.append("total_units")
        if self.uses_per_brick:
            cols.extend(vocab.names)
        return tuple(cols)

    def n_params(self, vocab: Vocabulary) -> int:
        return len(self.columns(vocab))


FORMS: dict[str, Form] = {
    f.key: f
    for f in (
        Form(
            "boot",
            "baseline",
            "fixed",
            "B0 -- always predict the measured empty-task cost. The fixed-cost illusion, "
            "named. If this scores well on total error, that is the illusion, not a result.",
        ),
        Form("constant", "baseline", "ols", "B1 -- fitted mean of the training runs."),
        Form(
            "units",
            "baseline",
            "ols",
            "B2 -- how many things were asked for, ignoring which.",
            uses_total_units=True,
        ),
        Form(
            "bytes",
            "baseline",
            "ols",
            "B3 -- how much material there was, ignoring the request.",
            uses_bytes=True,
        ),
        Form(
            "bytes_units",
            "baseline",
            "ols",
            "B4 -- size plus count. The real competitor: it needs no vocabulary, no "
            "encoder and no labelling, so a decoder must beat it to be worth anything.",
            uses_bytes=True,
            uses_total_units=True,
        ),
        Form(
            "log_bytes_units",
            "baseline",
            "ols",
            "B4b -- the same competitor with a diminishing-returns size term. Context "
            "cost need not be linear in bytes: a summarising harness, truncation, or "
            "chunked retrieval all bend the curve. Without this rung a decoder can be "
            "credited with lift it only earned by being *curved*, which brick counts are "
            "incidentally able to imitate. It is a baseline, so it can only ever raise "
            "the bar a decoder has to clear.",
            uses_bytes=True,
            uses_log_bytes=True,
            uses_total_units=True,
        ),
        Form(
            "per_brick",
            "decoder",
            "ols",
            "One coefficient per brick, no size term.",
            uses_per_brick=True,
        ),
        Form(
            "bytes_per_brick",
            "decoder",
            "ols",
            "The reference's published form: size plus a coefficient per brick.",
            uses_bytes=True,
            uses_per_brick=True,
        ),
        Form(
            "bytes_per_brick_nnls",
            "decoder",
            "nnls",
            "As above under a non-negativity constraint -- the principled alternative to "
            "fitting OLS and clamping negative marginals to zero afterwards.",
            uses_bytes=True,
            uses_per_brick=True,
        ),
        Form(
            "cell_mean",
            "diagnostic",
            "cell",
            "B5 -- the mean of each (brick, size) cell. A ceiling on what any model of "
            "this design could achieve. Excluded from the beat-the-baseline gate.",
        ),
    )
}

BASELINE_KEYS = tuple(k for k, f in FORMS.items() if f.role == "baseline")
DECODER_KEYS = tuple(k for k, f in FORMS.items() if f.role == "decoder")
DIAGNOSTIC_KEYS = tuple(k for k, f in FORMS.items() if f.role == "diagnostic")

LADDER = BASELINE_KEYS + DECODER_KEYS
"""Simple to rich. Selection walks this order and only climbs on real evidence."""


def feature_row(
    form_key: str, units: dict[str, int], context_bytes: int, vocab: Vocabulary
) -> list[float]:
    form = FORMS[form_key]
    row: list[float] = []
    if form.has_intercept:
        row.append(1.0)
    if form.uses_bytes:
        row.append(float(context_bytes))
    if form.uses_log_bytes:
        # log1p, not log: a null probe mounts zero bytes and must stay in the design,
        # where it pins the intercept. log(0) would drop exactly the rows that measure
        # the start-up toll.
        row.append(math.log1p(float(context_bytes)))
    if form.uses_total_units:
        row.append(float(sum(units.values())))
    if form.uses_per_brick:
        row.extend(float(units.get(name, 0)) for name in vocab.names)
    return row


def design_matrix(
    runs: Sequence[Run], form_key: str, vocab: Vocabulary
) -> tuple[list[list[float]], list[float]]:
    X = [feature_row(form_key, r.units, r.context_bytes, vocab) for r in runs]
    y = [r.billable_units for r in runs]
    return X, y


def run_group(run: Run) -> str:
    """The CV grouping key: the source material a run drew on.

    Splitting by group rather than by row is what stops the model being scored on a
    document it was fitted on. Runs with no context (null probes) belong to no group and
    are held in training for every fold -- they pin the intercept and cannot leak.
    """
    if not run.context_files:
        return "_nocontext"
    return sorted(run.context_files)[0]


def cell_key(run: Run, vocab: Vocabulary) -> tuple[str, ...]:
    """Identity of a design cell: which bricks, at what counts, in which size band."""
    present = tuple(f"{n}x{run.units[n]}" for n in vocab.names if run.units.get(n))
    band = "0" if run.context_bytes == 0 else str(len(str(run.context_bytes)))
    return present + (f"band{band}",)
