"""Auditable, quote-time-only feature definitions for wave 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .foundry_count import plain_text_token_lower_bound


DESIGN_CITATION = "docs/wave3-taxonomy-and-industry-design.md#62-revised-critic-feature-grammar"
TOKEN_CITATION = "https://developers.openai.com/api/docs/guides/token-counting"


@dataclass(frozen=True)
class FeatureDefinition:
    definition: str
    unit: str
    measurement: str
    valid_range: str
    missing_value_rule: str
    leakage_classification: str
    citations: Tuple[str, ...]
    compatible_bricks: Tuple[str, ...]
    role: str = "predictor"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["citations"] = list(self.citations)
        value["compatible_bricks"] = list(self.compatible_bricks)
        return value


ALL_BRICKS = ("*",)


def _feature(
    definition: str,
    unit: str,
    measurement: str,
    valid_range: str,
    missing: str,
    *,
    compatible: Tuple[str, ...] = ALL_BRICKS,
    role: str = "predictor",
    citations: Tuple[str, ...] = (DESIGN_CITATION,),
) -> FeatureDefinition:
    return FeatureDefinition(
        definition, unit, measurement, valid_range, missing,
        "quote_time: available before the first response byte; no observed-run data",
        citations, compatible, role,
    )


FEATURE_REGISTRY: Dict[str, FeatureDefinition] = {
    "task_payload_tokens": _feature(
        "Plain-text task payload token count, excluding the fixed harness.",
        "tokens",
        "At request construction, encode payload with tiktoken o200k_base. "
        "This is a lower-bound diagnostic relative to the complete provider request.",
        "integer >= 0",
        "Required; a missing tokenizer blocks the row rather than being imputed.",
        citations=(DESIGN_CITATION, TOKEN_CITATION),
    ),
    "provider_input_tokens": _feature(
        "Exact provider count for the complete input request, including request formatting and tool schemas.",
        "tokens",
        "Before generation, call the provider input-token counting endpoint with the exact request object.",
        "integer >= 0",
        "Nullable when the provider count endpoint is unavailable; do not substitute observed billed input.",
        role="audit",
        citations=(DESIGN_CITATION, TOKEN_CITATION),
    ),
    "fixed_overhead_tokens": _feature(
        "Exact provider input count for system prompt, tool schemas, and instruction prefix with an empty task payload.",
        "tokens",
        "Count the frozen zero-content harness with the provider input-token endpoint and bind it to the harness SHA-256.",
        "integer >= 0",
        "Required before fitting; absence blocks the row rather than being fitted or imputed.",
        role="fixed_constant",
        citations=(DESIGN_CITATION, TOKEN_CITATION),
    ),
    "output_bound_tokens": _feature(
        "Maximum permitted generated output for the request.",
        "tokens",
        "At request construction use min(API max-output setting, frozen schema/item token cap) when both exist.",
        "integer >= 0",
        "Required; use the API cap when no tighter frozen schema cap exists.",
    ),
    "output_spec_units": _feature(
        "Number of discrete fields, records, or prose facts required by the frozen output specification.",
        "declared items",
        "Count items in the prompt template or output schema before dispatch.",
        "integer >= 0",
        "Required; zero only when the task explicitly requests no output items.",
    ),
    "declared_fetch_bytes_kib": _feature(
        "Frozen expected bytes returned by declared fetches.",
        "KiB",
        "At preregistration compute round(sum(len(snapshot response UTF-8 bytes))/1024, 3).",
        "number >= 0",
        "Use 0 for a no-fetch task; a live fetch without a frozen snapshot is ineligible.",
        compatible=("fetch", "correlate", "retrieve", "provision"),
    ),
    "k_declared": _feature(
        "Number of model calls architecturally required by the task specification.",
        "calls",
        "Count fixed call stages in the frozen execution plan before dispatch.",
        "integer >= 1",
        "Required; never backfill from observed calls.",
    ),
    "k_free": _feature(
        "Whether additional model-call count is agent-chosen rather than fixed by the plan.",
        "binary flag",
        "Set from the frozen retry/branching policy: 1 when the agent may choose extra calls, otherwise 0.",
        "integer in {0, 1}",
        "Required; this is a quote-time uncertainty flag, not observed_calls - k_declared.",
    ),
    "effort_level": _feature(
        "Configured reasoning effort level.",
        "category",
        "Copy reasoning.effort from the request configuration before dispatch.",
        "one of minimal, low, medium, high",
        "Required; do not infer it from reasoning tokens.",
    ),
    "verbosity_level": _feature(
        "Configured response verbosity level.",
        "category",
        "Copy text.verbosity from the request configuration before dispatch.",
        "one of low, medium, high",
        "Required; do not infer it from response length.",
    ),
    "cache_warm": _feature(
        "Experimental assignment to repeated (warm) rather than unique (cold) harness prefix.",
        "binary flag",
        "Copy the preregistered randomized assignment before request construction.",
        "integer in {0, 1}",
        "Required; unknown cache hit telemetry is not a substitute.",
    ),
    "brick_counts": _feature(
        "Declared multiset count for each compatible brick in the composition.",
        "mapping of brick id to count",
        "Count occurrences in the frozen task decomposition before dispatch.",
        "mapping values are integers >= 0 with at least one positive count",
        "Required; undeclared bricks have count 0.",
    ),
    "brick_effect_coding": _feature(
        "Sum-to-zero effect-coding metadata and columns derived from declared brick counts.",
        "mapping",
        "Sort the frozen brick vocabulary; use the final brick as reference and emit J-1 columns count[j]-count[reference].",
        "J-1 numeric columns plus frozen vocabulary/reference metadata",
        "Required when J > 1; empty columns are valid for a one-brick vocabulary.",
    ),
    "composition_arity": _feature(
        "Number of distinct brick types with positive declared counts.",
        "brick types",
        "Count positive entries in brick_counts before dispatch.",
        "integer >= 1",
        "Required and derived from brick_counts.",
    ),
    "composition_mode": _feature(
        "Whether declared brick stages execute sequentially or in parallel.",
        "category",
        "Copy the frozen orchestration mode from the task plan.",
        "one of sequential, parallel",
        "Required for compositions; use sequential for a single-stage task.",
    ),
    "attempt_index": _feature(
        "One-based index reserved for this attempt before it starts.",
        "attempt",
        "Assign from the pre-dispatch retry ledger.",
        "integer >= 1",
        "Required; never derive it by counting successful responses.",
        role="audit",
    ),
    "retry_policy": _feature(
        "Frozen policy controlling whether and how another attempt may be dispatched.",
        "category",
        "Copy the preregistered policy identifier from the request plan.",
        "one of none, fixed, bounded",
        "Required; use none when retries are prohibited.",
        role="audit",
    ),
}


REJECTED_PREDICTORS: Dict[str, str] = {
    "observed_model_calls": "post-response execution outcome",
    "observed_tool_calls": "post-response execution outcome",
    "model_calls": "ambiguous/post-response call count; use k_declared",
    "tool_calls": "ambiguous/post-response call count; use the frozen plan",
    "k_observed": "post-response execution outcome",
    "reasoning_tokens": "post-response usage and direct leakage",
    "observed_response_bytes": "post-response output size",
    "response_size_observed": "post-response output size",
    "acceptance": "post-response oracle outcome",
    "latency": "post-response timing outcome",
    "latency_ms": "post-response timing outcome",
}


def registry_as_dict() -> Dict[str, Dict[str, Any]]:
    """Return a JSON-serializable copy of the frozen registry."""

    return {name: item.to_dict() for name, item in FEATURE_REGISTRY.items()}


def local_payload_token_count(payload: str) -> Tuple[int, str]:
    """Count plain text locally with the preregistered o200k_base tokenizer."""

    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    return plain_text_token_lower_bound(payload), "tiktoken:o200k_base"


def effect_code_bricks(
    counts: Mapping[str, int], vocabulary: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Create deterministic sum-to-zero count contrasts and their audit metadata."""

    if not counts or any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                         for v in counts.values()) or not any(counts.values()):
        raise ValueError("brick counts require at least one positive non-negative integer")
    names = tuple(sorted(vocabulary or counts))
    if not names or not set(counts).issubset(names):
        raise ValueError("brick vocabulary must contain every counted brick")
    reference = names[-1]
    reference_count = counts.get(reference, 0)
    columns = {
        f"brick_effect__{name}": counts.get(name, 0) - reference_count
        for name in names[:-1]
    }
    return {
        "coding": "sum_to_zero",
        "vocabulary": list(names),
        "reference": reference,
        "columns": columns,
    }


def extract_quote_features(
    *,
    task_payload: str,
    fixed_overhead_tokens: int,
    output_bound_tokens: int,
    output_spec_units: int,
    k_declared: int,
    k_free: bool,
    effort_level: str,
    verbosity_level: str,
    cache_warm: bool,
    brick_counts: Mapping[str, int],
    composition_mode: str,
    attempt_index: int = 1,
    retry_policy: str = "none",
    declared_fetch_bytes: int = 0,
    provider_input_tokens: Optional[int] = None,
    brick_vocabulary: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Extract a complete row exclusively from the pre-dispatch request plan."""

    if not isinstance(declared_fetch_bytes, int) or isinstance(declared_fetch_bytes, bool) \
            or declared_fetch_bytes < 0:
        raise ValueError("declared_fetch_bytes must be a non-negative integer")
    payload_tokens, payload_measurement = local_payload_token_count(task_payload)
    counts = dict(brick_counts)
    row: Dict[str, Any] = {
        "task_payload_tokens": payload_tokens,
        "task_payload_tokens_measurement": payload_measurement,
        "provider_input_tokens": provider_input_tokens,
        "fixed_overhead_tokens": fixed_overhead_tokens,
        "output_bound_tokens": output_bound_tokens,
        "output_spec_units": output_spec_units,
        "declared_fetch_bytes_kib": round(declared_fetch_bytes / 1024, 3),
        "k_declared": k_declared,
        "k_free": int(k_free),
        "effort_level": effort_level,
        "verbosity_level": verbosity_level,
        "cache_warm": int(cache_warm),
        "brick_counts": counts,
        "brick_effect_coding": effect_code_bricks(counts, brick_vocabulary),
        "composition_arity": sum(value > 0 for value in counts.values()),
        "composition_mode": composition_mode,
        "attempt_index": attempt_index,
        "retry_policy": retry_policy,
    }
    if provider_input_tokens is not None and (
        not isinstance(provider_input_tokens, int)
        or isinstance(provider_input_tokens, bool)
        or provider_input_tokens < 0
    ):
        raise ValueError("provider_input_tokens must be a non-negative integer or None")
    validate_quote_features(row)
    return row


# Explicit long name for callers that want the phase constraint visible.
extract_quote_time_features = extract_quote_features


def validate_quote_features(features: Mapping[str, Any]) -> None:
    """Reject leakage and enforce the core quote-time row invariants."""

    leaked = sorted(set(features) & set(REJECTED_PREDICTORS))
    if leaked:
        raise ValueError("post-run predictors are prohibited: " + ", ".join(leaked))
    required = {
        "task_payload_tokens", "fixed_overhead_tokens", "output_bound_tokens",
        "output_spec_units", "declared_fetch_bytes_kib", "k_declared", "k_free",
        "effort_level", "verbosity_level", "cache_warm", "brick_counts",
        "composition_arity", "composition_mode", "attempt_index", "retry_policy",
    }
    missing = sorted(required - set(features))
    if missing:
        raise ValueError("missing quote-time features: " + ", ".join(missing))
    nonnegative = ("task_payload_tokens", "fixed_overhead_tokens",
                   "output_bound_tokens", "output_spec_units",
                   "declared_fetch_bytes_kib")
    if any(not isinstance(features[n], (int, float)) or isinstance(features[n], bool)
           or features[n] < 0 for n in nonnegative):
        raise ValueError("token, unit, and byte features must be non-negative numbers")
    for name in ("k_declared", "composition_arity", "attempt_index"):
        if not isinstance(features[name], int) or isinstance(features[name], bool) or features[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("k_free", "cache_warm"):
        if features[name] not in (0, 1, False, True):
            raise ValueError(f"{name} must be binary")
    if features["effort_level"] not in {"minimal", "low", "medium", "high"}:
        raise ValueError("invalid effort_level")
    if features["verbosity_level"] not in {"low", "medium", "high"}:
        raise ValueError("invalid verbosity_level")
    if features["composition_mode"] not in {"sequential", "parallel"}:
        raise ValueError("invalid composition_mode")
    if features["retry_policy"] not in {"none", "fixed", "bounded"}:
        raise ValueError("invalid retry_policy")
    effect_code_bricks(features["brick_counts"])
    arity = sum(1 for value in features["brick_counts"].values() if value > 0)
    if features["composition_arity"] != arity:
        raise ValueError("composition_arity must equal positive brick count")
