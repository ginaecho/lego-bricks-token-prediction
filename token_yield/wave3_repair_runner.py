"""Resumable execution and calibration-only analysis for wave-3 repair."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .budget import HardBudget
from .foundry_count import FoundryCountError, FoundryInputTokenCounter
from .foundry_dispatch import (
    FoundryDispatchError,
    FoundryDispatcher,
    HttpRequestSpec,
    ResponseProtocolError,
    UsageLedger,
    acquire_entra_token,
)
from .public_api import ApiManifest, execute_manifest, load_manifests
from .wave3_features import (
    extract_quote_features,
    local_payload_token_count,
    registry_as_dict,
)
from .wave3_oracles import evaluate_oracle
from .wave3_repair_cases import RepairCase, load_repair_cases


PRIOR_WAVE2_SAFETY_USD = 5.05074
PRIOR_ABORTED_WAVE3_CANARY_SAFETY_USD = 5.67438
PRIOR_CAMPAIGN_SAFETY_USD = (
    PRIOR_WAVE2_SAFETY_USD + PRIOR_ABORTED_WAVE3_CANARY_SAFETY_USD
)
MODEL = "gpt-5-mini"
PROTOCOL_VERSION = "wave3-repair-v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(body)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, Mapping):
            return {str(key): safe(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(inner) for inner in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_preregistration(experiment_dir: Path) -> Dict[str, Any]:
    cases = load_repair_cases(experiment_dir)
    source_registry = experiment_dir / "source_registry.json"
    api_manifests = experiment_dir / "api_manifests.json"
    oracle_module = Path(__file__).with_name("wave3_oracles.py")
    feature_registry = registry_as_dict()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": MODEL,
        "seed": 260827,
        "session_count": len(cases),
        "allocation": {
            "tokenizer_confirmation": 2,
            "summarise": 12,
            "fetch": 18,
            "control": 4,
        },
        "repair_data_role": (
            "calibration-and-instrumentation-only; prohibited from final "
            "model selection"
        ),
        "retry_policy": "none",
        "pause_after_logical_sessions": 36,
        "prior_safety_ledger": {
            "wave2_usd": PRIOR_WAVE2_SAFETY_USD,
            "aborted_wave3_canary_usd":
                PRIOR_ABORTED_WAVE3_CANARY_SAFETY_USD,
            "cumulative_usd": PRIOR_CAMPAIGN_SAFETY_USD,
        },
        "cumulative_hard_cap_usd": 200.0,
        "operational_stop_usd": 190.0,
        "safety_rates_are_provider_prices": False,
        "input_hashes": {
            "source_registry.json": _sha256(source_registry),
            "api_manifests.json": _sha256(api_manifests),
            "wave3_oracles.py": _sha256(oracle_module),
            "feature_registry": _canonical_hash(feature_registry),
        },
        "feature_registry": feature_registry,
        "cases": [
            {
                "order": order,
                "case_id": case.case_id,
                "family": case.family,
                "group_id": case.group_id,
                "source_id": case.source_id,
                "arm": case.arm,
                "manifest_id": case.manifest_id,
                "prompt_sha256": case.prompt_sha256,
                "oracle_sha256": _canonical_hash(case.oracle),
                "effort_level": case.effort_level,
                "verbosity_level": case.verbosity_level,
                "cache_warm_assignment": case.cache_warm,
                "output_bound_tokens_per_call": case.output_bound_tokens,
                "output_spec_units": case.output_spec_units,
                "k_declared": case.k_declared,
                "k_free_allowed": case.k_free_allowed,
                "attempt_index": case.attempt_index,
                "declared_fetch_bytes": case.declared_fetch_bytes,
            }
            for order, case in enumerate(cases, start=1)
        ],
    }


def ensure_preregistration(experiment_dir: Path) -> Path:
    path = experiment_dir / "preregistration.json"
    expected = build_preregistration(experiment_dir)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError("wave-3 preregistration differs from frozen design")
    else:
        _write_json_atomic(path, expected)
    return path


def freeze_inputs(experiment_dir: Path, run_dir: Path, endpoint: str) -> None:
    """Freeze all protocol inputs before any provider request."""

    preregistration = ensure_preregistration(experiment_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "frozen_inputs"
    frozen.mkdir(exist_ok=True)
    module_dir = Path(__file__).parent
    sources = [
        preregistration,
        experiment_dir / "source_registry.json",
        experiment_dir / "api_manifests.json",
        module_dir / "wave3_features.py",
        module_dir / "wave3_oracles.py",
        *sorted((experiment_dir / "snapshots").glob("*")),
    ]
    hashes: Dict[str, str] = {}
    for source in sources:
        target = frozen / source.name
        if not target.exists():
            shutil.copy2(source, target)
        if _sha256(target) != _sha256(source):
            raise RuntimeError(f"frozen input changed: {source.name}")
        hashes[source.name] = _sha256(target)
    metadata = {
        "created_at": _now(),
        "runtime_stratum": "microsoft-foundry-direct-responses",
        "endpoint": endpoint.rstrip("/"),
        "deployment": MODEL,
        "protocol_version": PROTOCOL_VERSION,
        "input_hashes": hashes,
    }
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("endpoint", "deployment", "protocol_version", "input_hashes"):
            if existing.get(key) != metadata[key]:
                raise RuntimeError("run metadata differs from frozen checkpoint")
    else:
        _write_json_atomic(metadata_path, metadata)


def _manifest_bridge(
    manifests: Mapping[str, ApiManifest],
) -> tuple[Dict[str, HttpRequestSpec], Any]:
    specs = {
        key: HttpRequestSpec(value.method, value.url, value.headers)
        for key, value in manifests.items()
    }
    by_url = {value.url: value for value in manifests.values()}

    def fetch(spec: HttpRequestSpec):
        manifest = by_url.get(spec.url)
        if manifest is None:
            raise ValueError("request URL is outside the loaded allowlist")
        evidence = execute_manifest(manifest)
        return evidence.status, evidence.body.encode("utf-8")

    return specs, fetch


def reservation_bounds(
    case: RepairCase, manifest: Optional[ApiManifest] = None
) -> tuple[int, int]:
    """Bound generation plus exact-count and empty-harness count requests."""

    prompt_bytes = len(case.prompt.encode("utf-8"))
    if case.arm == "live":
        if manifest is None:
            raise ValueError("live Fetch case requires a manifest")
        generation_input = (
            prompt_bytes * 2 + manifest.max_response_bytes + 20_000
        )
        generation_output = case.output_bound_tokens * 2
    else:
        generation_input = prompt_bytes + 10_000
        generation_output = case.output_bound_tokens
    # Treat input-count requests as if their counted input were billable and
    # include a separate empty-harness count. This is deliberately conservative.
    reserved_input = generation_input * 2 + 10_000
    return reserved_input, generation_output


def _seed_budget() -> HardBudget:
    budget = HardBudget(cap_usd=200.0, stop_usd=190.0)
    budget.reserve("prior-campaign-safety-ledger", 0, 0)
    budget.settle(
        "prior-campaign-safety-ledger",
        {"input_tokens": 0, "output_tokens": 0},
        provider_charge_usd=PRIOR_CAMPAIGN_SAFETY_USD,
    )
    return budget


def _restore_budget(records: Iterable[Mapping[str, Any]]) -> HardBudget:
    budget = _seed_budget()
    for row in records:
        request_id = str(row["case_id"])
        budget.reserve(request_id, 0, 0)
        usage = row.get("safety_usage")
        if (
            isinstance(usage, Mapping)
            and isinstance(usage.get("input_tokens"), int)
            and isinstance(usage.get("output_tokens"), int)
        ):
            budget.settle(request_id, usage)
        else:
            budget.settle(
                request_id,
                {"input_tokens": 0, "output_tokens": 0},
                provider_charge_usd=float(row["reserved_safety_usd"]),
            )
    return budget


def _count_record(kind: str, index: int, result: Any) -> Dict[str, Any]:
    return {
        "kind": kind,
        "dispatch_index": index,
        "input_tokens": result.input_tokens,
        "endpoint": result.endpoint,
        "model": result.model,
        "request_sha256": result.request_sha256,
        "response_sha256": result.response_sha256,
        "response_id": result.response_id,
        "request_id": result.request_id,
        "latency_ms": result.latency_ms,
        "raw_fields": dict(result.raw_fields),
    }


def _brick_counts(case: RepairCase) -> Dict[str, int]:
    if case.family == "tokenizer_confirmation":
        return {"transform": 1}
    return {case.family: 1}


def _unavailable_feature_plan(
    case: RepairCase, error: str
) -> Dict[str, Any]:
    local_tokens, measurement = local_payload_token_count(case.prompt)
    return {
        "model_row_eligible": False,
        "ineligibility_reason": "provider input-token endpoint unavailable",
        "provider_count_error": error,
        "task_payload_tokens": local_tokens,
        "task_payload_tokens_measurement": measurement,
        "provider_input_tokens": None,
        "fixed_overhead_tokens": None,
        "output_bound_tokens": case.output_bound_tokens,
        "output_spec_units": case.output_spec_units,
        "declared_fetch_bytes_kib": round(
            case.declared_fetch_bytes / 1024, 3
        ),
        "k_declared": case.k_declared,
        "k_free": int(case.k_free_allowed),
        "effort_level": case.effort_level,
        "verbosity_level": case.verbosity_level,
        "cache_warm": int(case.cache_warm),
        "brick_counts": _brick_counts(case),
        "composition_arity": 1,
        "composition_mode": "sequential",
        "attempt_index": case.attempt_index,
        "retry_policy": "none",
    }


def analyze_repair(
    records: List[Mapping[str, Any]],
    *,
    provider_count_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Produce calibration diagnostics without model-promotion or tail claims."""

    complete = [row for row in records if row.get("usage")]
    discrepancies = [
        row["features"]["provider_input_tokens"]
        - row["features"]["task_payload_tokens"]
        for row in records
        if isinstance(row.get("features"), Mapping)
        and isinstance(row["features"].get("provider_input_tokens"), int)
    ]
    fetch_groups: Dict[str, Dict[str, List[float]]] = {}
    for row in records:
        if row.get("family") != "fetch" or not row.get("usage"):
            continue
        source = str(row["source_id"])
        arm = str(row["arm"])
        fetch_groups.setdefault(source, {}).setdefault(arm, []).append(
            float(row["usage"]["total_tokens"])
        )
    fetch_diagnostics = {}
    for source, arms in fetch_groups.items():
        fetch_diagnostics[source] = {}
        for arm, values in arms.items():
            median = statistics.median(values)
            fetch_diagnostics[source][arm] = {
                "n": len(values),
                "median_total_tokens": median,
                "mad_total_tokens": statistics.median(
                    abs(value - median) for value in values
                ),
            }
        if {"snapshot", "live"} <= set(arms):
            fetch_diagnostics[source]["live_minus_snapshot_median"] = (
                statistics.median(arms["live"])
                - statistics.median(arms["snapshot"])
            )
    regime_calibration: Dict[str, Dict[str, Any]] = {}
    for row in records:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            continue
        features = row.get("features")
        effort = (
            features.get("effort_level")
            if isinstance(features, Mapping)
            else None
        )
        verbosity = (
            features.get("verbosity_level")
            if isinstance(features, Mapping)
            else None
        )
        if not isinstance(effort, str) or not isinstance(verbosity, str):
            case_id = str(row.get("case_id", ""))
            effort = "medium" if "-medium-" in case_id else "minimal"
            verbosity = "medium" if effort == "medium" else "low"
        key = f"{effort}/{verbosity}"
        bucket = regime_calibration.setdefault(key, {
            "input_tokens": [],
            "output_tokens": [],
            "reasoning_tokens": [],
            "total_tokens": [],
            "accepted": 0,
            "provider_incomplete": 0,
            "n": 0,
        })
        bucket["n"] += 1
        for channel in (
            "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"
        ):
            value = usage.get(channel)
            if isinstance(value, int):
                bucket[channel].append(value)
        bucket["accepted"] += int(
            row.get("acceptance", {}).get("overall_acceptance") is True
        )
        bucket["provider_incomplete"] += int(
            row.get("acceptance", {}).get("provider_incomplete") is True
        )
    for bucket in regime_calibration.values():
        for channel in (
            "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"
        ):
            values = bucket[channel]
            bucket[f"median_{channel}"] = (
                statistics.median(values) if values else None
            )
            del bucket[channel]
    usage_totals = {
        channel: sum(
            int(row["usage"][channel])
            for row in records
            if isinstance(row.get("usage"), Mapping)
            and isinstance(row["usage"].get(channel), int)
        )
        for channel in (
            "input_tokens", "cached_tokens", "output_tokens",
            "reasoning_tokens", "total_tokens"
        )
    }
    count_unavailable = (
        len(records)
        if provider_count_available is False
        else sum(
            isinstance(row.get("features"), Mapping)
            and row["features"].get("model_row_eligible") is False
            for row in records
        )
    )
    return {
        "status": (
            "repair-complete-paused"
            if len(records) == 36 else "repair-checkpoint"
        ),
        "claim_scope": (
            "calibration-only; no model promotion, tail calibration, or "
            "post-repair campaign authorization"
        ),
        "logical_sessions": len(records),
        "telemetry_complete": len(complete),
        "provider_incomplete": sum(
            bool(row.get("acceptance", {}).get("provider_incomplete"))
            for row in records
        ),
        "provider_count_available": provider_count_available,
        "provider_count_unavailable": count_unavailable,
        "known_measured_usage": usage_totals,
        "structural_accepted": sum(
            row.get("acceptance", {}).get("structural_acceptance") is True
            for row in records
        ),
        "semantic_accepted": sum(
            row.get("acceptance", {}).get("semantic_acceptance") is True
            for row in records
        ),
        "overall_accepted": sum(
            row.get("acceptance", {}).get("overall_acceptance") is True
            for row in records
        ),
        "provider_minus_local_input_tokens": {
            "n": len(discrepancies),
            "median": statistics.median(discrepancies)
            if discrepancies else None,
            "minimum": min(discrepancies) if discrepancies else None,
            "maximum": max(discrepancies) if discrepancies else None,
        },
        "cache_assignment_observation": {
            "assigned_warm_observed_cached": sum(
                bool(row.get("cache_warm_assignment"))
                and bool(row.get("observed_cached"))
                for row in records
            ),
            "assigned_warm_observed_uncached": sum(
                bool(row.get("cache_warm_assignment"))
                and row.get("observed_cached") is False
                for row in records
            ),
            "assigned_cold_observed_cached": sum(
                not bool(row.get("cache_warm_assignment"))
                and bool(row.get("observed_cached"))
                for row in records
            ),
            "assigned_cold_observed_uncached": sum(
                not bool(row.get("cache_warm_assignment"))
                and row.get("observed_cached") is False
                for row in records
            ),
        },
        "fetch_calibration": fetch_diagnostics,
        "effort_verbosity_calibration": regime_calibration,
        "failure_modes": {
            "provider_output_cap_incomplete": sum(
                "incomplete status" in str(row.get("error", ""))
                for row in records
            ),
            "frozen_semantic_mismatch": sum(
                row.get("acceptance", {}).get("structural_acceptance") is True
                and row.get("acceptance", {}).get("semantic_acceptance") is False
                for row in records
            ),
        },
    }


def run_repair(
    experiment_dir: Path,
    run_dir: Path,
    endpoint: str,
    *,
    dry_run: bool = False,
    max_new_cases: Optional[int] = None,
    continue_after_errors: bool = False,
) -> Dict[str, Any]:
    """Run only the frozen 36-session repair block, then pause."""

    freeze_inputs(experiment_dir, run_dir, endpoint)
    cases = load_repair_cases(experiment_dir)
    manifests = load_manifests(experiment_dir / "api_manifests.json")
    specs, fetch_executor = _manifest_bridge(manifests)
    records_path = run_dir / "records.jsonl"
    pending_path = run_dir / "pending_request.json"
    count_capability_path = run_dir / "count_capability.json"
    existing_rows = _read_jsonl(records_path)
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        if pending["case_id"] not in {
            row["case_id"] for row in existing_rows
        }:
            _append_jsonl(records_path, {
                **pending,
                "completed_at": _now(),
                "error": (
                    "indeterminate provider attempt recovered after process "
                    "interruption; full reservation retained"
                ),
                "usage": None,
                "safety_usage": None,
                "acceptance": evaluate_oracle(
                    "", pending["oracle"], telemetry_complete=False
                ),
                "provenance": "measured-incomplete",
            })
            existing_rows = _read_jsonl(records_path)
        pending_path.unlink()
    existing = {row["case_id"] for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise RuntimeError("duplicate case IDs in run checkpoint")
    cases_by_id = {case.case_id: case for case in cases}
    for row in existing_rows:
        case = cases_by_id.get(row["case_id"])
        if case is None or row.get("prompt_sha256") != case.prompt_sha256:
            raise RuntimeError(
                f"checkpoint prompt identity changed: {row['case_id']}"
            )
    budget = _restore_budget(existing_rows)
    count_unavailable_reason: Optional[str] = None
    if count_capability_path.exists():
        capability = json.loads(
            count_capability_path.read_text(encoding="utf-8")
        )
        if capability.get("available") is False:
            count_unavailable_reason = str(capability["error"])
    if count_unavailable_reason is None:
        for row in existing_rows:
            error = str(row.get("error", ""))
            if "model is not supported by Responses API" in error:
                count_unavailable_reason = error
                _write_json_atomic(count_capability_path, {
                    "available": False,
                    "deployment": MODEL,
                    "endpoint": endpoint.rstrip("/")
                    + "/responses/input_tokens",
                    "error": error,
                    "observed_at": row.get("completed_at"),
                })
                break
    remaining = [case for case in cases if case.case_id not in existing]
    worst = sum(
        budget.rates.price(*reservation_bounds(
            case,
            manifests.get(case.manifest_id) if case.manifest_id else None,
        ))
        for case in remaining
    )
    plan = {
        "status": "dry-run" if dry_run else "ready",
        "cases_total": len(cases),
        "cases_complete": len(existing),
        "cases_remaining": len(remaining),
        "remaining_worst_case_reservation_usd": worst,
        "projected_cumulative_safety_usd": budget.settled_usd + worst,
        "budget": budget.snapshot(),
    }
    if budget.settled_usd + worst > budget.stop_usd:
        raise RuntimeError("frozen repair matrix exceeds the operational stop")
    if dry_run:
        return plan
    if max_new_cases is not None and max_new_cases < 1:
        raise ValueError("max_new_cases must be positive")

    token = acquire_entra_token()
    counter = FoundryInputTokenCounter(
        endpoint, model=MODEL, token_provider=lambda: token
    )
    dispatched = 0
    for case in remaining:
        if max_new_cases is not None and dispatched >= max_new_cases:
            break
        manifest = manifests.get(case.manifest_id) if case.manifest_id else None
        max_input, max_output = reservation_bounds(case, manifest)
        reserved = budget.reserve(case.case_id, max_input, max_output)
        pending = {
            "case_id": case.case_id,
            "family": case.family,
            "group_id": case.group_id,
            "source_id": case.source_id,
            "arm": case.arm,
            "prompt_sha256": case.prompt_sha256,
            "oracle": case.oracle,
            "reserved_safety_usd": reserved,
            "started_at": _now(),
        }
        _write_json_atomic(pending_path, pending)
        selected_specs = (
            {case.manifest_id: specs[case.manifest_id]}
            if case.manifest_id else {}
        )
        count_audits: List[Dict[str, Any]] = []
        ledger = UsageLedger()

        def count_before_dispatch(
            payload: Mapping[str, Any], index: int
        ) -> None:
            result = counter.count_dispatch_payload(payload)
            count_audits.append(_count_record("generation_input", index, result))

        dispatcher = FoundryDispatcher(
            endpoint,
            selected_specs,
            fetch_executor if selected_specs else None,
            model=MODEL,
            token_provider=lambda: token,
            max_response_calls=2 if case.arm == "live" else 1,
            max_tool_calls=1 if case.arm == "live" else 0,
            max_output_tokens=case.output_bound_tokens,
            reasoning_effort=case.effort_level,
            text_verbosity=case.verbosity_level,
            require_tool=case.arm == "live",
            ledger=ledger,
            pre_dispatch_hook=(
                None if count_unavailable_reason else count_before_dispatch
            ),
        )
        try:
            overhead = None
            if count_unavailable_reason is None:
                try:
                    # Foundry rejects an empty string. One ASCII space remains
                    # content-free while satisfying the request contract.
                    overhead = counter.count_dispatch_payload(
                        dispatcher.initial_payload(" ")
                    )
                    count_audits.append(
                        _count_record("empty_harness", -1, overhead)
                    )
                except FoundryCountError as exc:
                    if "model is not supported by Responses API" not in str(exc):
                        raise
                    count_unavailable_reason = f"{type(exc).__name__}: {exc}"
                    _write_json_atomic(count_capability_path, {
                        "available": False,
                        "deployment": MODEL,
                        "endpoint": counter.endpoint,
                        "error": count_unavailable_reason,
                        "observed_at": _now(),
                    })
                    dispatcher.pre_dispatch_hook = None
            result = dispatcher.dispatch(case.prompt, target=case.case_id)
            usage = asdict(result.usage)
            if overhead is not None:
                initial_count = next(
                    item["input_tokens"] for item in count_audits
                    if item["kind"] == "generation_input"
                    and item["dispatch_index"] == 0
                )
                features = extract_quote_features(
                    task_payload=case.prompt,
                    fixed_overhead_tokens=overhead.input_tokens,
                    provider_input_tokens=initial_count,
                    output_bound_tokens=case.output_bound_tokens,
                    output_spec_units=case.output_spec_units,
                    declared_fetch_bytes=case.declared_fetch_bytes,
                    k_declared=case.k_declared,
                    k_free=case.k_free_allowed,
                    effort_level=case.effort_level,
                    verbosity_level=case.verbosity_level,
                    cache_warm=case.cache_warm,
                    brick_counts=_brick_counts(case),
                    composition_mode="sequential",
                    attempt_index=case.attempt_index,
                    retry_policy="none",
                )
                features["model_row_eligible"] = True
            else:
                features = _unavailable_feature_plan(
                    case, count_unavailable_reason or "unavailable"
                )
            acceptance = evaluate_oracle(
                result.output, case.oracle, telemetry_complete=True
            )
            safety_usage = {
                **usage,
                "input_tokens": usage["input_tokens"] + sum(
                    item["input_tokens"] for item in count_audits
                ),
            }
            row = {
                **{key: value for key, value in pending.items()
                   if key != "oracle"},
                "cache_warm_assignment": case.cache_warm,
                "observed_cached": usage["cached_tokens"] > 0,
                "features": features,
                "count_audits": count_audits,
                "acceptance": acceptance,
                "output": result.output,
                "usage": usage,
                "safety_usage": safety_usage,
                "model_calls": len(result.response_calls),
                "tool_calls": len(result.fetches),
                "response_calls": [
                    asdict(item) for item in result.response_calls
                ],
                "fetches": [asdict(item) for item in result.fetches],
                "endpoint": endpoint.rstrip("/"),
                "deployment": MODEL,
                "completed_at": _now(),
                "error": None,
                "provenance": "measured",
            }
            row["settled_safety_usd"] = budget.settle(
                case.case_id, safety_usage
            )
            _append_jsonl(records_path, row)
            _write_json_atomic(run_dir / "budget.json", budget.snapshot())
            pending_path.unlink(missing_ok=True)
            dispatched += 1
        except (FoundryDispatchError, FoundryCountError) as exc:
            calls = ledger.calls(case.case_id)
            usage = asdict(ledger.totals()["__all__"]) if calls else None
            provider_incomplete = (
                isinstance(exc, ResponseProtocolError)
                and "incomplete status" in str(exc)
            )
            safety_usage = None
            if usage is not None:
                safety_usage = {
                    **usage,
                    "input_tokens": usage["input_tokens"] + sum(
                        item["input_tokens"] for item in count_audits
                    ),
                }
                budget.settle(case.case_id, safety_usage)
            else:
                budget.settle(
                    case.case_id,
                    {"input_tokens": 0, "output_tokens": 0},
                    provider_charge_usd=reserved,
                )
            _append_jsonl(records_path, {
                **{key: value for key, value in pending.items()
                   if key != "oracle"},
                "count_audits": count_audits,
                "usage": usage,
                "safety_usage": safety_usage,
                "model_calls": len(calls) if calls else None,
                "tool_calls": None,
                "acceptance": evaluate_oracle(
                    "",
                    case.oracle,
                    telemetry_complete=usage is not None,
                    provider_incomplete=provider_incomplete,
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at": _now(),
                "provenance": "measured-incomplete",
            })
            _write_json_atomic(run_dir / "budget.json", budget.snapshot())
            pending_path.unlink(missing_ok=True)
            dispatched += 1
            if not continue_after_errors:
                raise
    records = _read_jsonl(records_path)
    analysis = analyze_repair(
        records,
        provider_count_available=(
            False if count_unavailable_reason is not None else True
        ),
    )
    analysis["budget"] = budget.snapshot()
    _write_json_atomic(run_dir / "analysis.json", analysis)
    return analysis
