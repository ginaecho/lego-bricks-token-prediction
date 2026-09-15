"""Auditable execution records for business cases.

The runner is provider-neutral: callers supply a dispatcher for their agent
runtime. CI can use :func:`dry_run` without spending tokens; live adapters must
return the provider's measured usage rather than estimate it locally.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from .business_cases import BusinessCase, evaluate_output


@dataclass(frozen=True)
class AgentResult:
    output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_uses: Optional[int]
    duration_ms: int
    model: str
    model_version: str


@dataclass(frozen=True)
class RunRecord:
    case_id: str
    accepted: bool
    output: str
    checks: tuple
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_uses: Optional[int]
    duration_ms: int
    model: str
    model_version: str
    timestamp: str
    prompt_sha256: str
    source_manifest_sha256: str
    context_bytes: int
    counts: Dict[str, int]
    held_out: bool
    provenance: str = "measured"


Dispatcher = Callable[[str], AgentResult]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dry_run(case: BusinessCase) -> Dict[str, object]:
    """Return immutable dispatch material without invoking an agent."""

    return {
        "case_id": case.case_id,
        "prompt": case.prompt,
        "prompt_sha256": _sha256(case.prompt),
        "source_manifest_sha256": _sha256("\n".join(case.source_urls)),
        "counts": dict(case.counts),
        "context_bytes": case.context_bytes,
        "held_out": case.held_out,
    }


def run_case(case: BusinessCase, dispatch: Dispatcher) -> RunRecord:
    """Dispatch one fresh case and bind its measured spend to acceptance."""

    started = time.perf_counter()
    result = dispatch(case.prompt)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if result.total_tokens <= 0:
        raise ValueError("live dispatch must report positive measured total_tokens")
    if (
        result.input_tokens < 0
        or result.output_tokens < 0
        or (result.tool_uses is not None and result.tool_uses < 0)
    ):
        raise ValueError("token and tool measurements must be non-negative")
    verdict = evaluate_output(case, result.output)
    return RunRecord(
        case_id=case.case_id,
        accepted=verdict.accepted,
        output=result.output,
        checks=verdict.checks,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        tool_uses=result.tool_uses,
        duration_ms=result.duration_ms or elapsed_ms,
        model=result.model,
        model_version=result.model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_sha256=_sha256(case.prompt),
        source_manifest_sha256=_sha256("\n".join(case.source_urls)),
        context_bytes=case.context_bytes,
        counts=dict(case.counts),
        held_out=case.held_out,
    )


def append_record(path: str, record: RunRecord) -> None:
    """Append one durable JSONL result after a successful dispatch."""

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
