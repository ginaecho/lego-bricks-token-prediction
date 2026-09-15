"""Leak-free loading and normalization of measured run records.

This is the *data* seam described in
``docs/multichannel-model-experiment.md`` (Sections 1-3): it turns raw
``records.jsonl`` rows produced by ``wave2``/``wave3``/``wave3-repair`` runners
into two kinds of typed, auditable rows -- a quote-time predictor row that can
never contain post-run information, and a post-run target row with the
measured token/acceptance identities enforced -- plus an explicit,
target-specific eligibility verdict for every row. It never fits, scores, or
selects a model; that is the job of ``multichannel_model.py``.

Two record shapes are read directly from the repository's run history:

* wave2 (``runs/20260826_1627_wave2/records.jsonl``): a boolean ``accepted``
  field, no ``features`` block, and a flat ``usage`` mapping.
* wave3 / wave3-repair (``runs/20260827_11??_wave3/records.jsonl``): a richer
  ``acceptance`` mapping (``overall_acceptance``, ``semantic_acceptance``,
  ``structural_acceptance``, ``provider_incomplete``, ``telemetry_complete``),
  a ``features`` block produced by ``wave3_features.extract_quote_features``,
  and the same flat ``usage`` mapping.

Row eligibility is *target-specific*, per
``docs/multichannel-model-experiment.md#3-row-eligibility-and-data-firewall``:
a row can be useless for one channel and perfectly usable for another. This
module therefore never drops a row outright; it annotates each row with the
reasons (``ExclusionReason``) a consumer would need to exclude it from a
particular channel, and lets the caller decide.

``Targets.attempt_cost`` prices every row with known usage, whether or not
that attempt was accepted. Pricing only accepted rows is survivorship bias:
it silently erases the cost of every rejected or incomplete attempt. Real
accepted-work cost is a property of a *logical-work ledger* -- every attempt
behind one quoted job, priced and then reconciled against that job's
terminal acceptance state (see ``docs/multichannel-model-experiment.md``
Section 2, ``LogicalWorkLedger``) -- and must be aggregated downstream from
many rows' ``attempt_cost`` plus their acceptance labels, never read off of
a single accepted row here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .economics import Pricing
from .wave3_features import REJECTED_PREDICTORS, validate_quote_features

DESIGN_CITATION = "docs/multichannel-model-experiment.md"
WAVE3_CITATION = "token_yield/wave3_features.py"

# Fields wave3's runner attaches to ``features`` purely for audit purposes.
# They describe the row (e.g. why the exact provider count was unavailable);
# they are never predictor values and must not enter a feature matrix.
_FEATURE_DIAGNOSTIC_KEYS = frozenset({
    "model_row_eligible", "ineligibility_reason", "provider_count_error",
    "task_payload_tokens_measurement",
})

# provenance values that this codebase's runners use for a genuine, dispatched
# metering attempt (see budget/business_cases/pilot_runner/wave3_repair_runner).
# Anything else was never going to carry usage in the first place.
_METERED_PROVENANCE = frozenset({"measured", "measured-incomplete"})

_USAGE_KEYS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


class ExclusionReason:
    """Canonical, machine-readable reasons a row may be excluded from a target.

    ``MISSING_QUOTE_TIME_FEATURES`` is not one of the four reasons named in
    the task description but is required to make ``model_row_eligible``
    auditable (a wave2 row has no ``features`` block at all, and a wave3 row
    can fail its own quote-time schema); it is kept separate from the
    predictor-leakage hard failure below, which is not recoverable.
    """

    ACCEPTANCE_ONLY_UNMETERED = "acceptance_only_unmetered"
    UNKNOWN_USAGE = "unknown_usage"
    WAVE3_REPAIR_CALIBRATION_ONLY = "wave3_repair_calibration_only"
    PROVIDER_INCOMPLETE_TARGET_UNAVAILABLE = "provider_incomplete_target_unavailable"
    MISSING_QUOTE_TIME_FEATURES = "missing_quote_time_features"


ALL_EXCLUSION_REASONS = frozenset(
    value for key, value in vars(ExclusionReason).items() if key.isupper()
)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RuntimeStratum:
    """The runtime context a row was measured under.

    Per ``docs/multichannel-model-experiment.md`` Section 3.2, the runtime
    stratum (endpoint, deployment/model, and protocol/run identity) "is
    recorded and is not pooled with another stratum" -- two runs with the
    same endpoint but a different ``protocol_version`` (e.g. a repair-block
    rerun) are distinct strata.
    """

    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    runtime_stratum: Optional[str] = None
    protocol_version: Optional[str] = None

    @property
    def is_wave3_repair(self) -> bool:
        return bool(self.protocol_version) and self.protocol_version.startswith("wave3-repair")

    @property
    def key(self) -> Tuple[Optional[str], ...]:
        """A hashable identity that must never be pooled with another stratum's."""
        return (self.endpoint, self.deployment, self.runtime_stratum, self.protocol_version)


@dataclass(frozen=True)
class UsageBundle:
    """Actual, post-run token usage with every identity enforced at construction.

    ``non_reasoning_output_tokens`` is derived, never stored independently, so
    it can never drift from ``output_tokens - reasoning_tokens``.
    """

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    non_reasoning_output_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens", "cached_input_tokens", "output_tokens",
            "reasoning_tokens", "total_tokens",
        ):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens must not exceed input_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens must not exceed output_tokens")
        object.__setattr__(
            self, "non_reasoning_output_tokens",
            self.output_tokens - self.reasoning_tokens,
        )
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")


@dataclass(frozen=True)
class QuoteTimeObservation:
    """A validated, leak-free predictor row available before the first response byte.

    ``features`` contains only quote-time fields that passed
    ``wave3_features.validate_quote_features``; it is ``None`` when the row
    carries no ``features`` block at all (every wave2 record, and any wave3
    record whose feature extraction failed upstream).
    """

    record_id: str
    group_id: str
    runtime_stratum: RuntimeStratum
    features: Optional[Mapping[str, Any]]
    model_row_eligible: bool
    ineligibility_reason: Optional[str]
    upstream_model_row_eligible: Optional[bool] = None
    upstream_ineligibility_reason: Optional[str] = None


@dataclass(frozen=True)
class Targets:
    """Post-run measurements. Never constructed from quote-time inputs.

    ``attempt_cost`` is always computed by this module from usage and an
    explicitly supplied ``pricing`` (an ``economics.Pricing``) whenever usage
    is known -- regardless of ``overall_accepted``/``semantic_accepted`` --
    and it is never read from a raw record field. It prices *this one
    attempt*; it is deliberately not called "accepted work cost", since that
    concept can only be computed correctly by aggregating every attempt's
    ``attempt_cost`` in a job's logical-work ledger against that ledger's
    terminal acceptance state, not by looking at one row in isolation. It
    remains a field separate from every token count and from the acceptance
    labels.
    """

    record_id: str
    group_id: str
    runtime_stratum: RuntimeStratum
    usage: Optional[UsageBundle]
    overall_accepted: Optional[bool]
    semantic_accepted: Optional[bool]
    provider_incomplete: bool
    telemetry_complete: bool
    attempt_cost: Optional[float] = None


@dataclass(frozen=True)
class RowEligibility:
    """Target-specific eligibility verdicts and their machine-readable reasons.

    ``accepted_work_eligible`` is about whether this row's semantic
    acceptance *label* is usable (known, and not provider-incomplete) -- it
    says nothing about cost. ``Targets.attempt_cost`` is priced independently
    of this flag; see ``Targets`` for why accepted-work *cost* cannot be a
    per-row concept at all.
    """

    token_targets_eligible: bool
    accepted_work_eligible: bool
    final_model_selection_eligible: bool
    wave3_repair_calibration_only: bool
    exclusion_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class NormalizedObservation:
    """One measured run record, split into leak-free predictors and targets."""

    record_id: str
    group_id: str
    runtime_stratum: RuntimeStratum
    provenance: str
    quote: QuoteTimeObservation
    targets: Targets
    eligibility: RowEligibility


def _runtime_stratum(row: Mapping[str, Any], run_metadata: Mapping[str, Any]) -> RuntimeStratum:
    return RuntimeStratum(
        endpoint=row.get("endpoint"),
        deployment=row.get("deployment"),
        runtime_stratum=run_metadata.get("runtime_stratum"),
        protocol_version=run_metadata.get("protocol_version"),
    )


def _is_wave3_repair_calibration_only(
    stratum: RuntimeStratum, run_metadata: Mapping[str, Any]
) -> bool:
    if stratum.is_wave3_repair:
        return True
    role = str(run_metadata.get("repair_data_role") or "")
    return "final model selection" in role.lower()


def _parse_usage(raw_usage: Any) -> Tuple[Optional[UsageBundle], str, Optional[str]]:
    """Return ``(bundle_or_None, status, detail)``.

    ``status`` is one of ``"ok"``, ``"absent"`` (no usage was ever recorded),
    or ``"invalid"`` (usage exists but is partial or violates an identity).
    """

    if not isinstance(raw_usage, Mapping):
        return None, "absent", None
    values = {key: raw_usage.get(key) for key in _USAGE_KEYS}
    if all(value is None for value in values.values()):
        return None, "absent", None
    if any(value is None for value in values.values()):
        return None, "invalid", "usage is partially populated; some channels are missing"
    try:
        bundle = UsageBundle(
            input_tokens=values["input_tokens"],
            cached_input_tokens=values["cached_tokens"],
            output_tokens=values["output_tokens"],
            reasoning_tokens=values["reasoning_tokens"],
            total_tokens=values["total_tokens"],
        )
    except ValueError as exc:
        return None, "invalid", str(exc)
    return bundle, "ok", None


def _parse_acceptance(
    row: Mapping[str, Any], usage_status: str
) -> Tuple[Optional[bool], Optional[bool], bool, bool]:
    """Return ``(overall_accepted, semantic_accepted, provider_incomplete, telemetry_complete)``."""

    raw_acceptance = row.get("acceptance")
    if isinstance(raw_acceptance, Mapping):
        overall_accepted = raw_acceptance.get("overall_acceptance")
        semantic_accepted = raw_acceptance.get("semantic_acceptance")
        provider_incomplete = bool(raw_acceptance.get("provider_incomplete", False))
        telemetry_complete = bool(
            raw_acceptance.get("telemetry_complete", usage_status == "ok")
        )
        return overall_accepted, semantic_accepted, provider_incomplete, telemetry_complete

    # wave2 shape: a single boolean ``accepted`` verdict, no separate
    # structural/semantic split, and completion state only inferable from the
    # error text.
    overall_accepted = row.get("accepted") if "accepted" in row else None
    semantic_accepted = overall_accepted
    error_text = str(row.get("error") or "")
    provider_incomplete = "incomplete" in error_text.lower() and "status" in error_text.lower()
    telemetry_complete = usage_status == "ok"
    return overall_accepted, semantic_accepted, provider_incomplete, telemetry_complete


def _quote_time_observation(
    row: Mapping[str, Any],
    record_id: str,
    group_id: str,
    stratum: RuntimeStratum,
) -> QuoteTimeObservation:
    raw_features = row.get("features")
    if not isinstance(raw_features, Mapping):
        return QuoteTimeObservation(
            record_id=record_id,
            group_id=group_id,
            runtime_stratum=stratum,
            features=None,
            model_row_eligible=False,
            ineligibility_reason="no features were recorded for this row",
        )

    upstream_model_row_eligible = raw_features.get("model_row_eligible")
    upstream_ineligibility_reason = raw_features.get("ineligibility_reason")
    predictor_features = {
        key: value for key, value in raw_features.items()
        if key not in _FEATURE_DIAGNOSTIC_KEYS
    }

    leaked = sorted(set(predictor_features) & set(REJECTED_PREDICTORS))
    if leaked:
        # Predictor-leakage contamination is never recoverable for this row;
        # it must never be returned to a caller, so this raises rather than
        # being folded into the eligibility/exclusion-reason machinery below.
        raise ValueError(
            f"record {record_id!r}: post-run predictors are prohibited in quote-time "
            f"features: {', '.join(leaked)}"
        )

    own_valid = True
    own_reason: Optional[str] = None
    try:
        validate_quote_features(predictor_features)
    except ValueError as exc:
        own_valid = False
        own_reason = str(exc)

    # An explicitly frozen upstream verdict of False is binding: independent
    # schema validity here must never silently overrule a runner's own
    # ineligibility determination (e.g. "provider input-token endpoint
    # unavailable"). Combine both signals with AND semantics -- upstream
    # True cannot rescue a row that fails our own validation either.
    if upstream_model_row_eligible is False:
        model_row_eligible = False
        reasons = [
            reason for reason in (upstream_ineligibility_reason, own_reason)
            if reason
        ]
        ineligibility_reason = (
            "; ".join(reasons) if reasons
            else "upstream marked this row ineligible for model use"
        )
    elif not own_valid:
        model_row_eligible = False
        ineligibility_reason = own_reason
    else:
        model_row_eligible = True
        ineligibility_reason = None

    return QuoteTimeObservation(
        record_id=record_id,
        group_id=group_id,
        runtime_stratum=stratum,
        features=predictor_features,
        model_row_eligible=model_row_eligible,
        ineligibility_reason=ineligibility_reason,
        upstream_model_row_eligible=upstream_model_row_eligible,
        upstream_ineligibility_reason=upstream_ineligibility_reason,
    )


def _attempt_cost(usage: UsageBundle, pricing: Pricing) -> float:
    """Price one attempt's known usage, independent of its acceptance label."""

    return pricing.attempt_cost(
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
    )


def normalize_record(
    row: Mapping[str, Any],
    *,
    run_metadata: Optional[Mapping[str, Any]] = None,
    pricing: Optional[Pricing] = None,
) -> NormalizedObservation:
    """Normalize one raw measured-run record into leak-free typed rows.

    ``run_metadata`` is the parsed ``run_metadata.json`` sibling of a
    ``records.jsonl`` file (or ``{}``); it supplies the run-level
    ``runtime_stratum`` label and ``protocol_version`` used to detect a
    wave-3 repair-block run. ``pricing`` is an optional ``economics.Pricing``
    used only to derive ``targets.attempt_cost`` for every row with known
    usage, regardless of its acceptance label -- a raw ``attempt_cost`` (or
    legacy ``accepted_work_cost``) field on ``row``, if present, is always
    ignored.

    Raises ``ValueError`` immediately if the row's quote-time features
    contain a post-run predictor (predictor-leakage contamination), or if
    the row lacks the ``case_id``/``group_id`` identity this module promises
    to preserve.
    """

    run_metadata = run_metadata or {}
    record_id = row.get("case_id")
    group_id = row.get("group_id")
    if not record_id or not isinstance(record_id, str):
        raise ValueError("row is missing a string case_id")
    if not group_id or not isinstance(group_id, str):
        raise ValueError("row is missing a string group_id")

    provenance = row.get("provenance") or "unknown"
    stratum = _runtime_stratum(row, run_metadata)
    wave3_repair_calibration_only = _is_wave3_repair_calibration_only(stratum, run_metadata)

    quote = _quote_time_observation(row, record_id, group_id, stratum)

    usage, usage_status, usage_detail = _parse_usage(row.get("usage"))
    overall_accepted, semantic_accepted, provider_incomplete, telemetry_complete = (
        _parse_acceptance(row, usage_status)
    )

    reasons = set()
    if wave3_repair_calibration_only:
        reasons.add(ExclusionReason.WAVE3_REPAIR_CALIBRATION_ONLY)
    if usage_status == "absent":
        if provenance not in _METERED_PROVENANCE:
            reasons.add(ExclusionReason.ACCEPTANCE_ONLY_UNMETERED)
        else:
            reasons.add(ExclusionReason.UNKNOWN_USAGE)
    elif usage_status == "invalid":
        reasons.add(ExclusionReason.UNKNOWN_USAGE)
    if provider_incomplete:
        reasons.add(ExclusionReason.PROVIDER_INCOMPLETE_TARGET_UNAVAILABLE)
    if not quote.model_row_eligible:
        reasons.add(ExclusionReason.MISSING_QUOTE_TIME_FEATURES)

    token_targets_eligible = usage_status == "ok"
    accepted_work_eligible = not provider_incomplete and semantic_accepted is not None
    final_model_selection_eligible = (
        quote.model_row_eligible
        and token_targets_eligible
        and not wave3_repair_calibration_only
        and not provider_incomplete
        and provenance == "measured"
    )

    attempt_cost = (
        _attempt_cost(usage, pricing) if usage is not None and pricing is not None else None
    )

    targets = Targets(
        record_id=record_id,
        group_id=group_id,
        runtime_stratum=stratum,
        usage=usage,
        overall_accepted=overall_accepted,
        semantic_accepted=semantic_accepted,
        provider_incomplete=provider_incomplete,
        telemetry_complete=telemetry_complete,
        attempt_cost=attempt_cost,
    )
    eligibility = RowEligibility(
        token_targets_eligible=token_targets_eligible,
        accepted_work_eligible=accepted_work_eligible,
        final_model_selection_eligible=final_model_selection_eligible,
        wave3_repair_calibration_only=wave3_repair_calibration_only,
        exclusion_reasons=tuple(sorted(reasons)),
    )
    return NormalizedObservation(
        record_id=record_id,
        group_id=group_id,
        runtime_stratum=stratum,
        provenance=provenance,
        quote=quote,
        targets=targets,
        eligibility=eligibility,
    )


def normalize_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_metadata: Optional[Mapping[str, Any]] = None,
    pricing: Optional[Pricing] = None,
) -> List[NormalizedObservation]:
    """Normalize a batch of raw rows, preserving their order."""

    return [
        normalize_record(row, run_metadata=run_metadata, pricing=pricing)
        for row in rows
    ]


_ELIGIBILITY_PREDICATES = {
    "model_row": lambda observation: observation.quote.model_row_eligible,
    "token_targets": lambda observation: observation.eligibility.token_targets_eligible,
    "accepted_work": lambda observation: observation.eligibility.accepted_work_eligible,
    "final_model_selection": lambda observation: observation.eligibility.final_model_selection_eligible,
}


def filter_eligible(
    observations: Iterable[NormalizedObservation], target: str
) -> List[NormalizedObservation]:
    """Return only the rows eligible for ``target``.

    ``target`` is one of ``"model_row"``, ``"token_targets"``,
    ``"accepted_work"``, or ``"final_model_selection"``.
    """

    try:
        predicate = _ELIGIBILITY_PREDICATES[target]
    except KeyError as exc:
        raise ValueError(
            f"unknown eligibility target {target!r}; expected one of "
            f"{sorted(_ELIGIBILITY_PREDICATES)}"
        ) from exc
    return [observation for observation in observations if predicate(observation)]


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Read a ``records.jsonl`` file into a list of raw row dictionaries."""

    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_run_metadata(path: Path) -> Dict[str, Any]:
    """Read a ``run_metadata.json`` file, or ``{}`` when it does not exist."""

    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(
    run_dir: Path, *, pricing: Optional[Pricing] = None
) -> List[NormalizedObservation]:
    """Load and normalize every row in one run directory's ``records.jsonl``."""

    run_dir = Path(run_dir)
    rows = read_jsonl_records(run_dir / "records.jsonl")
    run_metadata = read_run_metadata(run_dir / "run_metadata.json")
    return normalize_records(rows, run_metadata=run_metadata, pricing=pricing)
