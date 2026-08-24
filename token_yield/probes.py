"""The probe suite: real agent runs, dispatched purely to measure what they cost.

This is the answer to "where does the calibration data come from?". Each probe
is a self-contained task at a stated kind and scope. Running one dispatches a
fresh, memoryless subagent and records what it actually spent.

Everything in :data:`MEASURED` is a **real run** (``Provenance.PROBE``), across
**three unrelated repositories** — this one, ``psf/requests`` and
``pallets/click`` — and is committed so the fitted models are reproducible
without re-spending the tokens.

Why three repositories
----------------------
Probing one repository can only ever tell you about that repository. Running
the identical graded task against three revealed that the obvious scope unit
was wrong: measured in **files**, the fitted slope disagreed 7× between repos
(2,305 vs 3,265 vs 15,794 tokens per file). Measured in **bytes**, the same runs
gave one model that fitted all three to 2.8%. Every record therefore carries
both candidates and the selector decides.
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
    repo: str = "harness-dose"
    input_bytes: int = 0

    def describe(self) -> str:
        return f"{self.label}: {self.kind} @ scope {self.scope:g} ({self.repo})"


# ── probe templates ──────────────────────────────────────────────────────

def comprehension_probe(files: list, question: str) -> str:
    listed = "\n   ".join(files)
    return (f"MEASUREMENT PROBE — comprehension, scope {len(files)}.\n\n"
            f"Do exactly this, nothing more. Do not edit any file. "
            f"Do not run tests.\n\n"
            f"1. Read these {len(files)} files:\n   {listed}\n"
            f"2. {question}\n\nStop after answering.")


def code_write_probe(scope: int, out_file: str, funcs: list) -> str:
    listed = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(funcs[:scope]))
    return (f"MEASUREMENT PROBE — code_write, scope {scope}.\n\n"
            f"Do exactly this, nothing more. Do not read any repo files.\n\n"
            f"Write {scope} Python function(s) to {SCRATCH}/{out_file}\n\n"
            f"{listed}\n\nEach with type hints and a one-line docstring.\n\n"
            f"Reply with just the word DONE.")


def test_write_probe(scope: int, targets: list, out_file: str) -> str:
    return (f"MEASUREMENT PROBE — test_write, scope {scope}.\n\n"
            f"Do exactly this, nothing more. Do not run tests.\n\n"
            f"1. Read /home/user/harness-dose/token_yield/costmodel.py\n"
            f"2. Write pytest tests for these {scope} function(s): "
            f"{', '.join(targets[:scope])}\n"
            f"3. Save to {SCRATCH}/{out_file}\n\nReply with just the word DONE.")


def code_review_probe(scope: int, targets: list) -> str:
    return (f"MEASUREMENT PROBE — code_review, scope {scope}.\n\n"
            f"Do exactly this, nothing more. Do not edit files. "
            f"Do not run tests.\n\n"
            f"1. Read /home/user/harness-dose/token_yield/costmodel.py\n"
            f"2. Review these {scope} function(s) for correctness bugs: "
            f"{', '.join(targets[:scope])}\n"
            f"3. Report findings in at most 3 sentences.\n\nStop after answering.")


def docs_probe(scope: int, targets: list, out_file: str) -> str:
    return (f"MEASUREMENT PROBE — docs, scope {scope}.\n\n"
            f"Do exactly this, nothing more. Do not run tests.\n\n"
            f"1. Read /home/user/harness-dose/token_yield/costmodel.py\n"
            f"2. Write reference documentation (Markdown) for these {scope} "
            f"function(s): {', '.join(targets[:scope])}\n"
            f"3. Save to {SCRATCH}/{out_file}\n\nReply with just the word DONE.")


FIT_TARGETS = ["fit_constant", "fit_proportional", "fit_affine",
               "fit_power", "mape", "loo_mape"]

WRITE_FUNCS = [
    "`clamp(value, lo, hi)` — returns value bounded to [lo, hi].",
    "`mean(xs)` — arithmetic mean, returns 0.0 for an empty sequence.",
    "`pct_change(a, b)` — percent change from a to b, returns None if a == 0.",
    "`median(xs)` — median, returns None for an empty sequence.",
    "`stddev(xs)` — population standard deviation, 0.0 for fewer than 2 items.",
    "`normalize(xs)` — scale a sequence to sum to 1.0, [] if the sum is 0.",
    "`chunk(xs, n)` — split a sequence into lists of at most n items.",
    "`fmt_bytes(n)` — format a byte count as B / KB / MB / GB, one decimal.",
]


def _rec(kind, scope, tokens, secs, tools, label, repo, in_bytes=0, out_units=0):
    signals = {}
    if in_bytes:
        signals["bytes"] = in_bytes
    if out_units:
        signals["output_units"] = out_units
    return ScopedRecord(kind, scope, tokens, secs, tools, Provenance.PROBE,
                        label, signals, repo)


# ── the measurements ─────────────────────────────────────────────────────
# Harness: Claude Code remote session, general-purpose subagent, 2026-08-22.
# `tokens` is the harness-reported `subagent_tokens`. Every row is a real run.

MEASURED: tuple[ScopedRecord, ...] = (
    # -- comprehension: read N files, answer a question -------------------
    # harness-dose
    _rec("comprehension", 1, 40_540, 4.804, 1, "hd s1 pilot", "harness-dose", 5_544),
    _rec("comprehension", 1, 41_651, 5.389, 1, "hd s1", "harness-dose", 5_544),
    _rec("comprehension", 3, 46_359, 12.216, 3, "hd s3 r1", "harness-dose", 15_216),
    _rec("comprehension", 3, 42_402, 11.210, 3, "hd s3 r2", "harness-dose", 15_216),
    _rec("comprehension", 3, 42_402, 12.618, 3, "hd s3 r3", "harness-dose", 15_216),
    _rec("comprehension", 5, 47_236, 30.904, 5, "hd s5", "harness-dose", 24_644),
    _rec("comprehension", 8, 57_732, 38.461, 4, "hd s8", "harness-dose", 35_518),
    # psf/requests
    _rec("comprehension", 1, 36_901, 9.707, 1, "req s1", "requests", 2_494),
    _rec("comprehension", 3, 40_839, 13.820, 3, "req s3", "requests", 10_696),
    _rec("comprehension", 5, 45_171, 14.625, 5, "req s5", "requests", 19_611),
    _rec("comprehension", 8, 59_851, 30.013, 6, "req s8", "requests", 60_374),
    # pallets/click — the decisive case: 8 files, 272 KB
    _rec("comprehension", 1, 37_909, 17.462, 1, "click s1", "click", 5_145),
    _rec("comprehension", 3, 44_766, 11.396, 3, "click s3", "click", 21_859),
    _rec("comprehension", 5, 53_925, 12.344, 5, "click s5", "click", 41_721),
    _rec("comprehension", 8, 149_655, 35.402, 11, "click s8", "click", 271_778),

    # -- code_write: write N functions to spec, no repo reading -----------
    _rec("code_write", 1, 39_242, 4.558, 1, "w1", "harness-dose", 0, 1),
    _rec("code_write", 3, 39_749, 12.159, 2, "w3 r1", "harness-dose", 0, 3),
    _rec("code_write", 3, 36_203, 6.325, 1, "w3 r2", "harness-dose", 0, 3),
    _rec("code_write", 8, 40_337, 11.680, 1, "w8", "harness-dose", 0, 8),

    # -- test_write: fixed input, graded OUTPUT ---------------------------
    # bytes read is identical in all three, so only the output count can
    # explain the variation — a clean test that the selector reads signals
    # rather than defaulting to the first one it finds.
    _rec("test_write", 1, 44_434, 28.301, 3, "t1", "harness-dose", 11_165, 1),
    _rec("test_write", 3, 47_158, 53.128, 3, "t3", "harness-dose", 11_165, 3),
    _rec("test_write", 6, 50_289, 85.489, 3, "t6", "harness-dose", 11_165, 6),

    # -- docs: acted on the coverage backlog (24% of mined real work) ------
    _rec("docs", 1, 43_906, 31.101, 2, "d1", "harness-dose", 11_165, 1),
    _rec("docs", 3, 44_657, 37.062, 2, "d3", "harness-dose", 11_165, 3),
    _rec("docs", 6, 46_358, 54.246, 2, "d6", "harness-dose", 11_165, 6),

    # -- code_review: fixed input, tiny fixed output ----------------------
    _rec("code_review", 1, 40_773, 12.650, 1, "cr1", "harness-dose", 11_165, 1),
    _rec("code_review", 3, 40_782, 20.680, 1, "cr3", "harness-dose", 11_165, 3),
    _rec("code_review", 6, 40_800, 33.942, 1, "cr6", "harness-dose", 11_165, 6),
)


PROBE_SUITE: tuple[ProbeSpec, ...] = tuple(
    [ProbeSpec(f"comprehension_{r.repo}_s{int(r.scope)}", "comprehension",
               r.scope, comprehension_probe(["<n files>"] * int(r.scope),
                                            "Summarise what these files do."),
               r.repo, int(r.signals.get("bytes", 0)))
     for r in MEASURED if r.kind == "comprehension" and "r2" not in r.label
     and "r3" not in r.label and "pilot" not in r.label]
    + [ProbeSpec(f"code_write_s{s}", "code_write", s,
                 code_write_probe(s, f"w{s}.py", WRITE_FUNCS)) for s in (1, 3, 8)]
    + [ProbeSpec(f"test_write_s{s}", "test_write", s,
                 test_write_probe(s, FIT_TARGETS, f"t{s}.py"), "harness-dose", 11_165)
       for s in (1, 3, 6)]
    + [ProbeSpec(f"code_review_s{s}", "code_review", s,
                 code_review_probe(s, FIT_TARGETS), "harness-dose", 11_165)
       for s in (1, 3, 6)]
    + [ProbeSpec(f"docs_s{s}", "docs", s, docs_probe(s, FIT_TARGETS, f"d{s}.md"),
                 "harness-dose", 11_165) for s in (1, 3, 6)]
)


# ── composition experiment (held out of the per-kind fits) ───────────────

@dataclass(frozen=True)
class CompositionMeasurement:
    """One agent doing several kinds of work in a single invocation.

    Deliberately kept out of :data:`MEASURED`: these are not single-kind
    records and must not be fitted as if they were. They exist to test how cost
    *composes*, which is a separate question from how it scales.
    """

    parts: tuple
    tokens: int
    duration_seconds: float
    label: str = ""


COMPOSITION_MEASURED: tuple[CompositionMeasurement, ...] = (
    CompositionMeasurement((("comprehension", 3), ("code_write", 3)),
                           43_727, 25.207, "combined_s3 r1"),
    CompositionMeasurement((("comprehension", 3), ("code_write", 3)),
                           43_255, 22.809, "combined_s3 r2"),
)


def replicate_spread(kind: str, scope: float, repo: str = "") -> tuple:
    """(n, mean, stddev) over replicates at one (kind, scope[, repo])."""
    vals = [float(r.tokens) for r in MEASURED
            if r.kind == kind and r.scope == scope and (not repo or r.repo == repo)]
    if not vals:
        return 0, 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return len(vals), m, var ** 0.5


def composition_evidence() -> dict:
    """Batched vs separate cost for the measured composition experiment."""
    batched = [float(c.tokens) for c in COMPOSITION_MEASURED]
    if not batched:
        return {}
    batched_mean = sum(batched) / len(batched)
    separate = 0.0
    for kind, scope in COMPOSITION_MEASURED[0].parts:
        _, mean, _ = replicate_spread(kind, scope, "harness-dose")
        separate += mean
    return {
        "batched_mean": batched_mean,
        "separate_sum": separate,
        "ratio": batched_mean / separate if separate else None,
        "saving": 1 - (batched_mean / separate) if separate else None,
    }


def repos() -> list:
    return sorted({r.repo for r in MEASURED if r.repo})
