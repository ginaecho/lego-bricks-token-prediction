"""Strict parsing of an encoder's output.

Fail-closed, deliberately. A decomposition that cannot be parsed is a *failure*, recorded
as one -- never quietly repaired into a plausible-looking vector, and never silently
replaced by the keyword encoder. A confidently wrong brick vector produces a
confidently wrong price, and the reference project's encoder was never scored at all, so
this is the part of the system with the least prior evidence behind it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Mapping

from assay.errors import DecompositionError
from assay.vocabulary import Vocabulary

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(frozen=True)
class Decomposition:
    units: dict[str, int]
    context_files: tuple[str, ...] = ()
    rationale: str = ""
    confidence: float = 0.0
    source: str = "llm"
    needs_clarification: bool = False
    out_of_scope: bool = False
    parse_retries: int = 0
    notes: tuple[str, ...] = field(default=())

    @property
    def total_units(self) -> int:
        return sum(self.units.values())

    @property
    def is_empty(self) -> bool:
        return self.total_units == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "units": self.units,
            "context_files": list(self.context_files),
            "rationale": self.rationale,
            "confidence": self.confidence,
            "source": self.source,
            "needs_clarification": self.needs_clarification,
            "out_of_scope": self.out_of_scope,
            "parse_retries": self.parse_retries,
            "notes": list(self.notes),
        }


def parse(
    text: str,
    vocab: Vocabulary,
    *,
    available_files: Mapping[str, object] | None = None,
    source: str = "llm",
) -> Decomposition:
    """Turn raw encoder output into a validated vector, or raise.

    Accepts a fenced code block because models emit them, and that is a formatting habit
    rather than a wrong answer. Everything else is strict.
    """
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DecompositionError(f"output is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise DecompositionError(f"expected a JSON object, got {type(data).__name__}")

    raw_units = data.get("units")
    if not isinstance(raw_units, dict):
        raise DecompositionError("`units` must be an object with one key per brick")

    expected = set(vocab.names)
    got = set(raw_units)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        problem = []
        if missing:
            problem.append(f"missing {missing}")
        if extra:
            problem.append(f"unknown {extra}")
        raise DecompositionError("`units` keys are wrong: " + "; ".join(problem))

    units: dict[str, int] = {}
    for name in vocab.names:
        value = raw_units[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise DecompositionError(f"{name}: count must be an integer, got {value!r}")
        if value < 0:
            raise DecompositionError(f"{name}: count must be >= 0, got {value}")
        units[name] = value

    files = data.get("context_files", [])
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise DecompositionError("`context_files` must be a list of strings")
    if available_files is not None:
        unknown = sorted(f for f in files if f not in available_files)
        if unknown:
            raise DecompositionError(f"context_files names documents that do not exist: {unknown}")

    confidence = data.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise DecompositionError(f"`confidence` must be between 0 and 1, got {confidence!r}")

    needs_clarification = bool(data.get("needs_clarification", False))
    out_of_scope = bool(data.get("out_of_scope", False))
    if (needs_clarification or out_of_scope) and sum(units.values()):
        raise DecompositionError(
            "a request flagged needs_clarification or out_of_scope must have an all-zero vector"
        )

    return Decomposition(
        units=units,
        context_files=tuple(files),
        rationale=str(data.get("rationale", "")),
        confidence=float(confidence),
        source=source,
        needs_clarification=needs_clarification,
        out_of_scope=out_of_scope,
    )
