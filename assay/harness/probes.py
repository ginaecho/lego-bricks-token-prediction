"""Probe instructions -- the things we actually dispatch.

A probe is one instruction plus the documents it may see plus the brick vector it was
*designed* to ask for. That designed vector is the generator's intent, not gold: Phase 1
labels the same text independently, and the two disagreeing on more than 10% of the fit
split is a signal that the wording is ambiguous, not that the labellers are wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from assay.schemas import Split, Tier

NULL_INSTRUCTION = "Reply with the single word DONE."

TEMPLATES: dict[str, str] = {
    "Review": "Read the attached material and answer {n} judgement question(s) about what it covers.",
    "Extract": "From the attached document, pull out {n} named field(s) and report each value.",
    "Classify": "Assign each of {n} supplied item(s) to exactly one of the given categories.",
    "Retrieve": "Search the mounted directory and locate {n} fact(s); say which document holds each.",
    "Reconcile": "Compare {n} named item(s) across the two supplied sources and report any differences.",
    "Draft": "Write {n} short piece(s) of new prose to the brief in the attached material.",
    "Remediate": "Correct {n} known defect(s) in the attached artefact.",
    "Validate": "Run {n} independently scorable check(s) against the attached source.",
    "Report": "Present the findings already made as {n} report(s) for the stated audience.",
}

LADDER_INSTRUCTION = "Read the attached material and state in one sentence what it covers."


@dataclass(frozen=True)
class Probe:
    probe_id: str
    tier: Tier
    split: Split
    instruction: str
    units: dict[str, int]
    context_files: tuple[str, ...] = ()
    replicate: int = 1
    bundle_id: str | None = None
    arm: str | None = None
    notes: str = ""
    tasks: tuple[str, ...] = field(default=(), repr=False)

    @property
    def instruction_sha256(self) -> str:
        return hashlib.sha256(self.instruction.encode("utf-8")).hexdigest()

    @property
    def total_units(self) -> int:
        return sum(self.units.values())


def brick_instruction(brick: str, n: int) -> str:
    return TEMPLATES[brick].format(n=n)


def compose_instruction(parts: list[tuple[str, int]]) -> str:
    """Join several brick clauses into one request.

    The batched arm of the batching experiment adds only the neutral joining language --
    every per-task clause is byte-identical to its separate-arm counterpart, so a cost
    difference cannot be a difference in what was asked.
    """
    clauses = [brick_instruction(b, n) for b, n in parts]
    if len(clauses) == 1:
        return clauses[0]
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(clauses, 1))
    return f"Answer all sections below.\n\n{numbered}"


def null_probe(index: int, position: str) -> Probe:
    """The empty task. Its median across replicates is the boot estimate, and the drift
    between the opening and closing brackets is how we notice the runtime moving."""
    return Probe(
        probe_id=f"null-{position}-{index}",
        tier=Tier.NULL,
        split=Split.FIT,
        instruction=NULL_INSTRUCTION,
        units={},
        context_files=(),
        replicate=index,
        notes=f"drift bracket ({position})",
    )


def ladder_probe(doc: str, index: int) -> Probe:
    return Probe(
        probe_id=f"ladder-bytes-{index}",
        tier=Tier.LADDER,
        split=Split.PROBE,
        instruction=LADDER_INSTRUCTION,
        units={},
        context_files=(doc,),
        notes="context slope",
    )


def orthogonality_probe(docs: tuple[str, ...], index: int) -> Probe:
    """Same total bytes delivered as a different number of documents.

    This is what turns "units and bytes are orthogonal by construction" from an assumption
    into a measurement: if document *count* carries cost independently of size, the
    composite grid has to be rebalanced before any per-brick coefficient is believed.
    """
    return Probe(
        probe_id=f"ladder-orth-{len(docs)}doc-{index}",
        tier=Tier.LADDER,
        split=Split.PROBE,
        instruction=LADDER_INSTRUCTION,
        units={},
        context_files=docs,
        notes=f"units-vs-bytes: {len(docs)} documents",
    )
