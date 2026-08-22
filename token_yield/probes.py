"""The probe suite: real agent runs, dispatched purely to measure what they cost.

This is the answer to "where does the calibration data come from?". Each
:class:`ProbeSpec` is a self-contained task at a stated kind and scope. Running
one dispatches a fresh, memoryless subagent and records what it actually spent.

The measurements in :data:`MEASURED` were produced by running this suite. They
are **real** (``Provenance.PROBE``) — not invented — and they are committed so
that the fitted models are reproducible without re-spending the tokens.

What the suite is for
---------------------
Graded scope within a kind is what makes a *slope* estimable. A single
measurement per kind can only ever support a scope-blind constant model, which
is exactly the dead end the first version of this package reached.

What the suite is not
---------------------
It measures small tasks (scope 1–8) on one harness, one model, one repository.
Section "regime" in every fitted model records that. Predicting a 500-file
refactor from these points is the same extrapolation error the 2×/4×
multipliers made, just with better arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import Provenance, ScopedRecord

SCRATCH = ("/tmp/claude-0/-home-user-harness-dose/"
           "d6e84d11-a4c1-5e60-8722-4897492f2f5d/scratchpad/probes")


@dataclass(frozen=True)
class ProbeSpec:
    """A reproducible unit of measurement: dispatch this, record what it cost."""

    label: str
    kind: str
    scope: float
    prompt: str

    def describe(self) -> str:
        return f"{self.label}: {self.kind} @ scope {self.scope:g}"


_COMPREHENSION_FILES = [
    "/home/user/harness-dose/token_yield/models.py",
    "/home/user/harness-dose/token_yield/calibrate.py",
    "/home/user/harness-dose/token_yield/predict.py",
    "/home/user/harness-dose/token_yield/forecast.py",
    "/home/user/harness-dose/token_yield/report.py",
    "/home/user/harness-dose/token_yield/__init__.py",
    "/home/user/harness-dose/openharness/trace.py",
    "/home/user/harness-dose/openharness/card.py",
]

_WRITE_FUNCS = [
    "`clamp(value, lo, hi)` — returns value bounded to [lo, hi].",
    "`mean(xs)` — arithmetic mean, returns 0.0 for an empty sequence.",
    "`pct_change(a, b)` — percent change from a to b, returns None if a == 0.",
    "`median(xs)` — median, returns None for an empty sequence.",
    "`stddev(xs)` — population standard deviation, 0.0 for fewer than 2 items.",
    "`normalize(xs)` — scale a sequence to sum to 1.0, returns [] if the sum is 0.",
    "`chunk(xs, n)` — split a sequence into lists of at most n items.",
    "`fmt_bytes(n)` — format a byte count as B / KB / MB / GB with one decimal.",
]


def comprehension_probe(scope: int, question: str) -> str:
    files = "\n   ".join(_COMPREHENSION_FILES[:scope])
    return (f"MEASUREMENT PROBE — comprehension, scope {scope}.\n\n"
            f"Do exactly this, nothing more. Do not edit any file. "
            f"Do not run tests.\n\n"
            f"1. Read these {scope} files:\n   {files}\n"
            f"2. {question}\n\nStop after answering.")


def code_write_probe(scope: int, out_file: str) -> str:
    funcs = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(_WRITE_FUNCS[:scope]))
    return (f"MEASUREMENT PROBE — code_write, scope {scope}.\n\n"
            f"Do exactly this, nothing more. Do not read any repo files. "
            f"Do not run tests.\n\n"
            f"Write {scope} Python function(s) to the new file\n"
            f"{SCRATCH}/{out_file}\n\nThe functions:\n{funcs}\n\n"
            f"Each with type hints and a one-line docstring.\n\n"
            f"Stop after writing the file. Reply with just the word DONE.")


_Q5 = ("In exactly 5 sentences, explain how a CalibrationRecord becomes a "
       "TaskPrediction.")

#: The suite as dispatched. Replicates repeat a spec verbatim — identical
#: prompts are what make run-to-run variance measurable rather than confounded.
PROBE_SUITE: tuple[ProbeSpec, ...] = (
    ProbeSpec("comprehension_s1", "comprehension", 1,
              comprehension_probe(1, "In exactly 2 sentences, state what "
                                     "ComplexityTier is for.")),
    ProbeSpec("comprehension_s3", "comprehension", 3, comprehension_probe(3, _Q5)),
    ProbeSpec("comprehension_s5", "comprehension", 5,
              comprehension_probe(5, "In exactly 5 sentences, explain how a "
                                     "CalibrationRecord becomes a rendered "
                                     "budget report.")),
    ProbeSpec("comprehension_s8", "comprehension", 8,
              comprehension_probe(8, "Produce a dependency map: which module "
                                     "imports which, and which dataclass flows "
                                     "between which stage.")),
    ProbeSpec("code_write_s1", "code_write", 1, code_write_probe(1, "w1.py")),
    ProbeSpec("code_write_s3", "code_write", 3, code_write_probe(3, "w3.py")),
    ProbeSpec("code_write_s8", "code_write", 8, code_write_probe(8, "w8.py")),
)


# ── the measurements ─────────────────────────────────────────────────────
# Collected by dispatching the suite above to fresh general-purpose subagents.
# `tokens` is the harness-reported `subagent_tokens` for that run.
#
# Harness: Claude Code remote session, general-purpose subagent, 2026-08-22.
# Every row is a real run. Nothing here is estimated or interpolated.

MEASURED: tuple[ScopedRecord, ...] = (
    # -- comprehension: read N files, answer a question about them ---------
    ScopedRecord("comprehension", 1, 40_540, 4.804, 1, Provenance.PROBE,
                 "pilot; same kind and scope, slightly shorter question"),
    ScopedRecord("comprehension", 1, 41_651, 5.389, 1, Provenance.PROBE,
                 "comprehension_s1"),
    ScopedRecord("comprehension", 3, 46_359, 12.216, 3, Provenance.PROBE,
                 "comprehension_s3 r1"),
    ScopedRecord("comprehension", 3, 42_402, 11.210, 3, Provenance.PROBE,
                 "comprehension_s3 r2"),
    ScopedRecord("comprehension", 3, 42_402, 12.618, 3, Provenance.PROBE,
                 "comprehension_s3 r3"),
    ScopedRecord("comprehension", 5, 47_236, 30.904, 5, Provenance.PROBE,
                 "comprehension_s5"),
    ScopedRecord("comprehension", 8, 57_732, 38.461, 4, Provenance.PROBE,
                 "comprehension_s8"),

    # -- code_write: write N functions to spec, no repo reading ------------
    ScopedRecord("code_write", 1, 39_242, 4.558, 1, Provenance.PROBE,
                 "code_write_s1"),
    ScopedRecord("code_write", 3, 39_749, 12.159, 2, Provenance.PROBE,
                 "code_write_s3 r1"),
    ScopedRecord("code_write", 3, 36_203, 6.325, 1, Provenance.PROBE,
                 "code_write_s3 r2"),
    ScopedRecord("code_write", 8, 40_337, 11.680, 1, Provenance.PROBE,
                 "code_write_s8"),
)


@dataclass(frozen=True)
class CompositionMeasurement:
    """One agent doing several kinds of work in a single invocation.

    Deliberately kept out of :data:`MEASURED`: these are not single-kind
    records and must not be fitted as if they were. They exist to test how cost
    *composes*, which is a separate question from how it scales.
    """

    parts: tuple[tuple[str, float], ...]      # (kind, scope) done in ONE agent
    tokens: int
    duration_seconds: float
    label: str = ""


#: The composition experiment. Each row is one agent asked to do a 3-file
#: comprehension task AND write 3 functions, so it can be compared against the
#: two kinds measured separately.
#:
#: Result: the batched runs average ~43.5k against ~81.7k for the same work in
#: two agents — 53% of the separate cost. Combining kinds *saves*; it does not
#: surcharge. The original ``interaction_overhead = +15%`` had the wrong sign.
COMPOSITION_MEASURED: tuple[CompositionMeasurement, ...] = (
    CompositionMeasurement((("comprehension", 3), ("code_write", 3)),
                           43_727, 25.207, "combined_s3 r1"),
    CompositionMeasurement((("comprehension", 3), ("code_write", 3)),
                           43_255, 22.809, "combined_s3 r2"),
)


def composition_evidence() -> dict:
    """Batched vs separate cost for the measured composition experiment."""
    batched = [float(c.tokens) for c in COMPOSITION_MEASURED]
    if not batched:
        return {}
    batched_mean = sum(batched) / len(batched)

    separate = 0.0
    for kind, scope in COMPOSITION_MEASURED[0].parts:
        n, mean, _ = replicate_spread(kind, scope)
        separate += mean
    return {
        "batched_mean": batched_mean,
        "separate_sum": separate,
        "ratio": batched_mean / separate if separate else None,
        "saving": 1 - (batched_mean / separate) if separate else None,
    }


def replicate_spread(kind: str, scope: float) -> tuple[int, float, float]:
    """(n, mean, stddev) over replicates at one (kind, scope).

    Run-to-run spread is the noise floor: no model can predict this kind more
    precisely than the same task repeated predicts itself.
    """
    vals = [float(r.tokens) for r in MEASURED
            if r.kind == kind and r.scope == scope]
    if not vals:
        return 0, 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return len(vals), m, var ** 0.5
