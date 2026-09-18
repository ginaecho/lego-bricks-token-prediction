"""What a dispatcher must provide, and nothing more.

The interface is deliberately narrow: an instruction, some documents, a temperature, and
a usage record back. Everything the plan cares about -- fresh session per call, one frozen
system prompt, one frozen tool set -- is a property of the *implementation*, and each
adapter states in its docstring how it satisfies them, because that statement is part of
what the campaign measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from assay.evidence import EvidenceClass


@dataclass(frozen=True)
class RunResult:
    prompt_tokens: int
    completion_tokens: int
    model_echoed: str
    elapsed_s: float
    output_text: str
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    tool_calls: int = 0
    tool_result_bytes: int = 0
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts cannot be negative")


@runtime_checkable
class AgentAdapter(Protocol):
    name: str
    model: str
    evidence_class: EvidenceClass

    caching_disabled: bool
    """The adapter asserts that prompt caching is off for every call it makes.

    This is a declaration, not a hope, and the runner refuses to dispatch without it.
    Caching is on by default on several providers, and a cached run is not a measurement
    of what the task costs -- it is a measurement of what the task costs *given* what was
    asked five minutes ago. ``Usage.contract_violation`` still catches a cache hit that
    happens anyway, but by then the money is spent and, on a 24-task blind set, four such
    runs void the whole set. The cheap check belongs before the call.
    """

    def run(
        self,
        instruction: str,
        context_files: list[Path],
        *,
        temperature: float = 0.0,
    ) -> RunResult: ...
