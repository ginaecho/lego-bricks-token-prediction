"""Offline historical wiring for the leak-free multichannel quote model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .economics import Pricing
from .foundry_count import plain_text_token_lower_bound
from .multichannel_data import (
    normalize_record,
    read_jsonl_records,
    read_run_metadata,
)
from .multichannel_model import (
    ChannelObservation,
    MultiChannelForecast,
    MultiChannelModel,
)
from .pilot_cases import PilotCase, load_pilot_cases


FEATURE_NAMES = (
    "task_payload_tokens",
    "declared_fetch_bytes_kib",
    "output_bound_tokens",
    "output_spec_units",
    "k_declared",
    "brick_summarise",
    "brick_transform",
    "brick_fetch",
    "live_fetch",
)

WAVE3_FEATURE_NAMES = (
    "task_payload_tokens",
    "declared_fetch_bytes_kib",
    "output_bound_tokens",
    "output_spec_units",
    "k_declared",
    "k_free",
    "cache_warm",
    "effort_low",
    "effort_medium",
    "effort_high",
    "brick_effect_control",
    "brick_effect_summarise",
    "brick_effect_transform",
    "live_fetch",
)

_FAMILIES = {"control", "summarise", "transform", "fetch"}


@dataclass(frozen=True)
class HistoricalQuote:
    """Quote-time values accepted by the historical diagnostic seam."""

    task_payload_tokens: int
    declared_fetch_bytes: int
    output_bound_tokens: int
    output_spec_units: int
    k_declared: int
    family: str
    arm: Optional[str]
    input_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "task_payload_tokens", "declared_fetch_bytes",
            "output_bound_tokens", "output_spec_units", "k_declared",
            "input_tokens",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.output_bound_tokens < 1 or self.k_declared < 1:
            raise ValueError(
                "output_bound_tokens and k_declared must be positive"
            )
        if self.family not in _FAMILIES:
            raise ValueError(f"unknown historical family {self.family!r}")
        if self.arm not in (None, "snapshot", "live"):
            raise ValueError("arm must be None, snapshot, or live")
        if self.arm == "live" and self.family != "fetch":
            raise ValueError("only Fetch quotes may use the live arm")

    def features(self) -> Tuple[float, ...]:
        return (
            float(self.task_payload_tokens),
            self.declared_fetch_bytes / 1024.0,
            float(self.output_bound_tokens),
            float(self.output_spec_units),
            float(self.k_declared),
            float(self.family == "summarise"),
            float(self.family == "transform"),
            float(self.family == "fetch"),
            float(self.arm == "live"),
        )


@dataclass(frozen=True)
class HistoricalDiagnostic:
    """A fitted plumbing diagnostic that can never be promoted."""

    model: MultiChannelModel
    rows_used: int
    rows_excluded: int
    independent_groups: int
    status: str = "historical-exploratory-only"
    promotion_eligible: bool = False
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    limitations: Tuple[str, ...] = (
        "Wave 2 is historical exploratory evidence only.",
        "Reasoning, cache, and unforced branching are underidentified.",
        "p90/p95 are not publishable calibrated tail claims.",
        "No Wave 3 repair or blind row entered model fitting.",
    )

    def forecast(
        self,
        quote: HistoricalQuote,
        *,
        pricing: Optional[Pricing] = None,
    ) -> MultiChannelForecast:
        return self.model.forecast(
            quote.features(),
            input_tokens=quote.input_tokens,
            output_bound_tokens=quote.output_bound_tokens,
            pricing=pricing,
        )

    def report(self) -> Dict[str, Any]:
        channels = {}
        for name, comparison in self.model.channel_comparisons.items():
            value = asdict(comparison)
            baseline = comparison.baseline_mae
            model = comparison.model_mae
            value["relative_mae_improvement"] = (
                (baseline - model) / baseline
                if baseline not in (None, 0.0) and model is not None
                else None
            )
            channels[name] = value
        return {
            "status": self.status,
            "promotion_eligible": False,
            "rows_used": self.rows_used,
            "rows_excluded": self.rows_excluded,
            "independent_groups": self.independent_groups,
            "feature_names": list(self.feature_names),
            "channels": channels,
            "limitations": list(self.limitations),
        }


def _declared_calls(case: PilotCase) -> int:
    return 2 if case.arm == "live" else 1


def _quote_for_case(
    case: PilotCase,
    declared_fetch_bytes_by_group: Mapping[str, int],
    input_tokens: int,
) -> HistoricalQuote:
    calls = _declared_calls(case)
    return HistoricalQuote(
        task_payload_tokens=plain_text_token_lower_bound(case.prompt),
        declared_fetch_bytes=declared_fetch_bytes_by_group.get(
            case.group_id, 0
        ),
        output_bound_tokens=case.max_output_tokens * calls,
        output_spec_units=case.units,
        k_declared=calls,
        family=case.family,
        arm=case.arm,
        input_tokens=input_tokens,
    )


def fit_wave2_historical_diagnostic(
    run_dir: Path,
    experiment_dir: Path,
    *,
    pricing: Optional[Pricing] = None,
    seed: int = 260826,
) -> HistoricalDiagnostic:
    """Fit an offline plumbing diagnostic from frozen Wave 2 artifacts."""

    pricing = pricing or Pricing(
        input_per_million=20.0,
        cached_input_per_million=20.0,
        output_per_million=200.0,
    )
    rows = read_jsonl_records(run_dir / "records.jsonl")
    metadata = read_run_metadata(run_dir / "run_metadata.json")
    cases = load_pilot_cases(experiment_dir)
    cases_by_id = {case.case_id: case for case in cases}
    if len(cases_by_id) != 30:
        raise ValueError("Wave 2 case catalog must contain exactly 30 cases")
    declared_fetch_bytes_by_group: Dict[str, int] = {}
    for case in cases:
        if case.family == "fetch":
            declared_fetch_bytes_by_group[case.group_id] = max(
                declared_fetch_bytes_by_group.get(case.group_id, 0),
                case.context_bytes,
            )

    feature_rows = []
    observations = []
    excluded = 0
    for row in rows:
        case_id = row.get("case_id")
        case = cases_by_id.get(case_id)
        if case is None:
            raise ValueError(f"run contains unknown Wave 2 case {case_id!r}")
        normalized = normalize_record(
            row, run_metadata=metadata, pricing=pricing
        )
        usage = normalized.targets.usage
        if usage is None:
            excluded += 1
            continue
        quote = _quote_for_case(
            case, declared_fetch_bytes_by_group, usage.input_tokens
        )
        model_calls = row.get("model_calls")
        if (
            isinstance(model_calls, bool)
            or not isinstance(model_calls, int)
            or model_calls < 1
        ):
            raise ValueError(f"{case_id}: measured model_calls is unavailable")
        output_censored = normalized.targets.provider_incomplete
        observed_output_bound = quote.output_bound_tokens
        if usage.output_tokens > observed_output_bound:
            raise ValueError(
                f"{case_id}: output exceeds the frozen aggregate call cap"
            )
        feature_rows.append(quote.features())
        observations.append(ChannelObservation(
            group=case.group_id,
            cached_tokens=usage.cached_input_tokens,
            non_reasoning_output_tokens=usage.non_reasoning_output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            branch_or_retry=model_calls > quote.k_declared,
            output_bound_tokens=observed_output_bound,
            output_censored=output_censored,
            attempt_cost=normalized.targets.attempt_cost,
            accepted=normalized.targets.overall_accepted,
        ))
    if not feature_rows:
        raise ValueError("Wave 2 has no measured rows for the diagnostic")
    if any(
        not all(math.isfinite(value) for value in feature_row)
        for feature_row in feature_rows
    ):
        raise ValueError("Wave 2 quote-time feature matrix is not finite")
    model = MultiChannelModel.fit(
        feature_rows,
        observations,
        feature_names=FEATURE_NAMES,
        seed=seed,
    )
    return HistoricalDiagnostic(
        model=model,
        rows_used=len(observations),
        rows_excluded=excluded,
        independent_groups=len({item.group for item in observations}),
    )


def _wave3_feature_row(
    case: Mapping[str, Any],
    *,
    task_payload_tokens: int,
) -> Tuple[float, ...]:
    family = str(case["family"])
    reference_fetch = float(family == "fetch")
    effort = str(case["effort_level"])
    return (
        float(task_payload_tokens),
        float(case["declared_fetch_bytes"]) / 1024.0,
        float(case["output_bound_tokens_per_call"]) * int(case["k_declared"]),
        float(case["output_spec_units"]),
        float(case["k_declared"]),
        float(bool(case["k_free_allowed"])),
        float(bool(case["cache_warm_assignment"])),
        float(effort == "low"),
        float(effort == "medium"),
        float(effort == "high"),
        float(family == "control") - reference_fetch,
        float(family == "summarise") - reference_fetch,
        float(family == "tokenizer_confirmation") - reference_fetch,
        float(case.get("arm") == "live"),
    )


def fit_wave3_repair_diagnostic(
    run_dir: Path,
    experiment_dir: Path,
    *,
    pricing: Optional[Pricing] = None,
    seed: int = 260827,
) -> HistoricalDiagnostic:
    """Retrain Wave 3 offline while preserving its calibration-only role."""

    pricing = pricing or Pricing(
        input_per_million=20.0,
        cached_input_per_million=20.0,
        output_per_million=200.0,
    )
    rows = read_jsonl_records(run_dir / "records.jsonl")
    metadata = read_run_metadata(run_dir / "run_metadata.json")
    preregistration = json.loads(
        (experiment_dir / "preregistration.json").read_text(encoding="utf-8")
    )
    role = str(preregistration.get("repair_data_role") or "")
    if "prohibited from final model selection" not in role:
        raise ValueError("Wave 3 repair calibration-only role is not frozen")
    cases = preregistration.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Wave 3 preregistration cases are unavailable")
    cases_by_id = {case["case_id"]: case for case in cases}
    rows_by_id = {row.get("case_id"): row for row in rows}
    if len(cases_by_id) != 36 or set(cases_by_id) != set(rows_by_id):
        raise ValueError("Wave 3 repair run must match all 36 frozen cases")

    task_tokens_by_prompt = {
        row["prompt_sha256"]: int(row["features"]["task_payload_tokens"])
        for row in rows
        if isinstance(row.get("features"), Mapping)
        and row["features"].get("task_payload_tokens") is not None
    }
    feature_rows = []
    observations = []
    for case_id, case in cases_by_id.items():
        row = rows_by_id[case_id]
        prompt_sha256 = row.get("prompt_sha256")
        if prompt_sha256 not in task_tokens_by_prompt:
            raise ValueError(f"{case_id}: quote-time task token count is unavailable")
        normalized = normalize_record(row, run_metadata=metadata, pricing=pricing)
        usage = normalized.targets.usage
        if usage is None:
            raise ValueError(f"{case_id}: Wave 3 repair usage is unavailable")
        feature_row = _wave3_feature_row(
            case,
            task_payload_tokens=task_tokens_by_prompt[prompt_sha256],
        )
        output_bound = (
            int(case["output_bound_tokens_per_call"]) * int(case["k_declared"])
        )
        if usage.output_tokens > output_bound:
            raise ValueError(f"{case_id}: output exceeds frozen aggregate call cap")
        feature_rows.append(feature_row)
        observations.append(ChannelObservation(
            group=str(case["group_id"]),
            cached_tokens=usage.cached_input_tokens,
            non_reasoning_output_tokens=usage.non_reasoning_output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            branch_or_retry=int(row.get("model_calls") or 0) > int(case["k_declared"]),
            output_bound_tokens=float(output_bound),
            output_censored=normalized.targets.provider_incomplete,
            attempt_cost=normalized.targets.attempt_cost,
            accepted=normalized.targets.semantic_accepted,
        ))

    model = MultiChannelModel.fit(
        feature_rows,
        observations,
        feature_names=WAVE3_FEATURE_NAMES,
        seed=seed,
    )
    return HistoricalDiagnostic(
        model=model,
        rows_used=len(observations),
        rows_excluded=0,
        independent_groups=len({item.group for item in observations}),
        status="wave3-repair-calibration-only",
        feature_names=WAVE3_FEATURE_NAMES,
        limitations=(
            "Wave 3 repair rows failed the repair gate and are calibration-only.",
            "Provider preflight input-token counts were unavailable for this deployment.",
            "Provider-incomplete rows are censored for output and reasoning magnitudes.",
            "No blind row entered fitting and no model is eligible for promotion.",
        ),
    )
