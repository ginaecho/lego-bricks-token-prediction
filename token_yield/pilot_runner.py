"""Resumable execution and preregistered analysis for Foundry wave 2."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .budget import HardBudget
from .foundry_dispatch import (
    FoundryDispatchError,
    FoundryDispatcher,
    HttpRequestSpec,
    UsageLedger,
    acquire_entra_token,
)
from .pilot_cases import PilotCase, evaluate_pilot_output, load_pilot_cases
from .public_api import ApiManifest, execute_manifest, load_manifests
from .robust import (
    Record,
    RidgeLinearModel,
    diagnostics,
    empirical_interval_coverage,
    grouped_bootstrap_improvement_ci,
    grouped_bootstrap_coefficient_ci,
    nested_grouped_cv,
    quantile,
    select_ridge_alpha,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): safe(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(inner) for inner in item]
        return item

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze_inputs(experiment_dir: Path, run_dir: Path, endpoint: str) -> None:
    """Copy protocol inputs and hashes before the first paid request."""

    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "frozen_inputs"
    frozen.mkdir(exist_ok=True)
    sources = [
        experiment_dir / "preregistration.json",
        experiment_dir / "api_manifests.json",
        *sorted((experiment_dir / "snapshots").glob("*")),
    ]
    hashes = {}
    for source in sources:
        target = frozen / source.name
        if not target.exists():
            shutil.copy2(source, target)
        if _sha256(target) != _sha256(source):
            raise RuntimeError(f"frozen input changed: {source.name}")
        hashes[source.name] = _sha256(target)
    metadata_path = run_dir / "run_metadata.json"
    metadata = {
        "created_at": _now(),
        "runtime_stratum": "microsoft-foundry-direct-responses",
        "endpoint": endpoint.rstrip("/"),
        "deployment": "gpt-5-mini",
        "input_hashes": hashes,
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing["endpoint"] != metadata["endpoint"]
            or existing["input_hashes"] != hashes
        ):
            raise RuntimeError("run metadata differs from the frozen checkpoint")
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
        if spec.url not in by_url:
            raise ValueError("request URL is outside the loaded allowlist")
        evidence = execute_manifest(by_url[spec.url])
        return evidence.status, evidence.body.encode("utf-8")

    return specs, fetch


def reservation_bounds(
    case: PilotCase,
    manifest: ApiManifest | None = None,
) -> tuple[int, int]:
    """Conservative one-byte-per-token bounds including tool continuation."""

    prompt_bytes = len(case.prompt.encode("utf-8"))
    if case.arm == "live":
        if manifest is None:
            raise ValueError("live Fetch case requires a manifest")
        max_input = prompt_bytes * 2 + manifest.max_response_bytes + 20_000
        max_output = case.max_output_tokens * 2
    else:
        max_input = prompt_bytes + 10_000
        max_output = case.max_output_tokens
    return max_input, max_output


def _restore_budget(records: Iterable[Mapping[str, Any]]) -> HardBudget:
    budget = HardBudget()
    for row in records:
        request_id = str(row["case_id"])
        budget.reserve(request_id, 0, 0)
        usage = row.get("usage")
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


def run_pilot(
    experiment_dir: Path,
    run_dir: Path,
    endpoint: str,
    *,
    dry_run: bool = False,
    max_new_cases: int | None = None,
    continue_after_errors: bool = False,
) -> Dict[str, Any]:
    """Run at most the frozen 30 sessions, then stop for gate analysis."""

    freeze_inputs(experiment_dir, run_dir, endpoint)
    cases = load_pilot_cases(experiment_dir)
    manifests = load_manifests(experiment_dir / "api_manifests.json")
    specs, fetch_executor = _manifest_bridge(manifests)
    records_path = run_dir / "records.jsonl"
    pending_path = run_dir / "pending_request.json"
    existing_rows = _read_jsonl(records_path)
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        if pending["case_id"] not in {row["case_id"] for row in existing_rows}:
            _append(records_path, {
                **pending,
                "accepted": False,
                "completed_at": _now(),
                "error": (
                    "indeterminate paid attempt recovered after process "
                    "interruption; full reservation retained"
                ),
                "usage": {
                    "input_tokens": None, "cached_tokens": None,
                    "output_tokens": None, "reasoning_tokens": None,
                    "total_tokens": None,
                },
                "response_calls": [],
                "model_calls": None,
                "tool_calls": None,
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
            raise RuntimeError(f"checkpoint prompt identity changed: {row['case_id']}")
    budget = _restore_budget(existing_rows)
    plan = {
        "cases_total": len(cases),
        "cases_complete": len(existing),
        "cases_remaining": len(cases) - len(existing),
        "budget": budget.snapshot(),
    }
    if dry_run:
        worst = 0.0
        for case in cases:
            manifest = manifests.get(case.manifest_id) if case.manifest_id else None
            bounds = reservation_bounds(case, manifest)
            worst += budget.rates.price(*bounds)
        plan["all_cases_worst_case_reservation_usd"] = worst
        return plan
    if max_new_cases is not None and max_new_cases < 1:
        raise ValueError("max_new_cases must be positive")

    token = acquire_entra_token()
    dispatched = 0
    for case in cases:
        if case.case_id in existing:
            continue
        if max_new_cases is not None and dispatched >= max_new_cases:
            break
        manifest = manifests.get(case.manifest_id) if case.manifest_id else None
        max_input, max_output = reservation_bounds(case, manifest)
        reserved = budget.reserve(case.case_id, max_input, max_output)
        _write_json_atomic(pending_path, {
            "case_id": case.case_id,
            "family": case.family,
            "group_id": case.group_id,
            "arm": case.arm,
            "context_bytes": case.context_bytes,
            "units": case.units,
            "prompt_sha256": case.prompt_sha256,
            "reserved_safety_usd": reserved,
            "started_at": _now(),
        })
        ledger = UsageLedger()
        selected_specs = (
            {case.manifest_id: specs[case.manifest_id]}
            if case.manifest_id else {}
        )
        dispatcher = FoundryDispatcher(
            endpoint,
            selected_specs,
            fetch_executor if selected_specs else None,
            model="gpt-5-mini",
            token_provider=lambda: token,
            max_response_calls=2 if case.arm == "live" else 1,
            max_tool_calls=1 if case.arm == "live" else 0,
            max_output_tokens=case.max_output_tokens,
            reasoning_effort="minimal",
            text_verbosity="low",
            require_tool=case.arm == "live",
            ledger=ledger,
        )
        started = _now()
        try:
            result = dispatcher.dispatch(case.prompt, target=case.case_id)
            usage = asdict(result.usage)
            verdict = evaluate_pilot_output(case, result.output)
            row = {
                "case_id": case.case_id,
                "family": case.family,
                "group_id": case.group_id,
                "arm": case.arm,
                "context_bytes": case.context_bytes,
                "units": case.units,
                "prompt_sha256": case.prompt_sha256,
                "accepted": verdict["accepted"],
                "acceptance_detail": verdict["detail"],
                "output": result.output,
                "usage": usage,
                "model_calls": len(result.response_calls),
                "tool_calls": len(result.fetches),
                "response_calls": [asdict(item) for item in result.response_calls],
                "fetches": [asdict(item) for item in result.fetches],
                "endpoint": endpoint.rstrip("/"),
                "deployment": "gpt-5-mini",
                "started_at": started,
                "completed_at": _now(),
                "reserved_safety_usd": reserved,
                "error": None,
                "provenance": "measured",
            }
            budget.settle(case.case_id, usage)
            row["settled_safety_usd"] = budget.snapshot()[
                "settled_requests"
            ][case.case_id]
            _append(records_path, row)
            _write_json_atomic(run_dir / "budget.json", budget.snapshot())
            pending_path.unlink(missing_ok=True)
            dispatched += 1
        except FoundryDispatchError as exc:
            calls = ledger.calls(case.case_id)
            usage = asdict(ledger.totals().get(case.case_id, ledger.totals()["__all__"]))
            if calls:
                budget.settle(case.case_id, usage)
            else:
                budget.cancel(case.case_id)
            _append(records_path, {
                "case_id": case.case_id,
                "family": case.family,
                "group_id": case.group_id,
                "arm": case.arm,
                "prompt_sha256": case.prompt_sha256,
                "usage": usage,
                "model_calls": len(calls),
                "tool_calls": None,
                "response_calls": [asdict(item) for item in calls],
                "accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
                "started_at": started,
                "completed_at": _now(),
                "reserved_safety_usd": reserved,
                "provenance": "measured",
            })
            _write_json_atomic(run_dir / "budget.json", budget.snapshot())
            pending_path.unlink(missing_ok=True)
            dispatched += 1
            if not continue_after_errors:
                raise
    records = _read_jsonl(records_path)
    if len(records) != 30:
        return {
            "status": "checkpoint",
            "sessions_complete": len(records),
            "sessions_remaining": 30 - len(records),
            "budget": budget.snapshot(),
        }
    analysis = analyze_pilot(records)
    _write_json_atomic(run_dir / "analysis.json", analysis)
    return analysis


def _median_mad(values: List[float]) -> tuple[float, float]:
    median = statistics.median(values)
    return median, statistics.median(abs(value - median) for value in values)


def _ladder_gate(
    records: List[Mapping[str, Any]], family: str, level_key: str
) -> Dict[str, Any]:
    rows = [
        row for row in records
        if row["family"] == family
        and isinstance(row.get("usage", {}).get("total_tokens"), int)
    ]
    levels = sorted({int(row[level_key]) for row in rows})
    medians = {}
    deviations = []
    for level in levels:
        values = [
            float(row["usage"]["total_tokens"])
            for row in rows if int(row[level_key]) == level
        ]
        median, _ = _median_mad(values)
        medians[str(level)] = median
        deviations.extend(abs(value - median) for value in values)
    pooled_mad = statistics.median(deviations)
    effect = medians[str(levels[-1])] - medians[str(levels[0])]
    monotonic = all(
        medians[str(left)] <= medians[str(right)]
        for left, right in zip(levels, levels[1:])
    )
    return {
        "levels": levels,
        "median_total_tokens": medians,
        "pooled_within_shape_mad": pooled_mad,
        "high_minus_low": effect,
        "effect_over_mad": (
            None if pooled_mad == 0
            else effect / pooled_mad if pooled_mad else 0
        ),
        "zero_observed_within_shape_mad": pooled_mad == 0,
        "monotonic": monotonic,
        "passed": effect > 0 and monotonic and effect >= 3 * pooled_mad,
    }


def _fetch_gate(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [
        row for row in records
        if row["family"] == "fetch"
        and isinstance(row.get("usage", {}).get("total_tokens"), int)
    ]
    by_group: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], {})[row["arm"]] = row
    deltas = []
    arm_values = {"snapshot": [], "live": []}
    for pair in by_group.values():
        if set(pair) != {"snapshot", "live"}:
            continue
        for arm in arm_values:
            arm_values[arm].append(float(pair[arm]["usage"]["total_tokens"]))
        deltas.append(
            float(pair["live"]["usage"]["total_tokens"])
            - float(pair["snapshot"]["usage"]["total_tokens"])
        )
    deviations = []
    for values in arm_values.values():
        median = statistics.median(values)
        deviations.extend(abs(value - median) for value in values)
    mad = statistics.median(deviations)
    incremental = statistics.median(deltas) if deltas else 0
    delta_mad = (
        statistics.median(abs(value - incremental) for value in deltas)
        if deltas else 0
    )
    branching = all(
        row["model_calls"] == (2 if row["arm"] == "live" else 1)
        and row["tool_calls"] == (1 if row["arm"] == "live" else 0)
        for row in rows
    )
    return {
        "paired_deltas": deltas,
        "median_incremental_total_tokens": incremental,
        "pooled_within_arm_mad": mad,
        "effect_over_mad": (
            float("inf") if mad == 0 and incremental > 0
            else incremental / mad if mad else 0
        ),
        "preregistered_within_shape_mad_identifiable": False,
        "paired_delta_mad_post_hoc": delta_mad,
        "effect_over_paired_delta_mad_post_hoc": (
            incremental / delta_mad if delta_mad else None
        ),
        "gate_interpretation": (
            "conservative pre-run implementation failed; preregistered "
            "within-shape MAD is unidentifiable with one row per arm/shape"
        ),
        "planned_branching": branching,
        "passed": branching and incremental > 0 and incremental >= 3 * mad,
    }


def _feature_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    family = row["family"]
    return (
        float(row.get("context_bytes", 0)) / 1024,
        float(row.get("units", 0)),
        float(family == "transform"),
        float(family == "fetch"),
        float(row.get("arm") == "live"),
    )


def _target_value(row: Mapping[str, Any], target: str) -> float | None:
    if target in {"model_calls", "tool_calls"}:
        value = row.get(target)
    else:
        value = row.get("usage", {}).get(target)
    return float(value) if isinstance(value, int) else None


def _model_gate(
    records: List[Mapping[str, Any]], target: str
) -> Dict[str, Any]:
    rows = [
        row for row in records
        if row["family"] != "control"
        and _target_value(row, target) is not None
    ]
    model_rows = [
        Record(
            _feature_vector(row),
            _target_value(row, target),
            row["group_id"],
            {"family": row["family"], "arm": row.get("arm") or "none"},
        )
        for row in rows
    ]
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv = nested_grouped_cv(model_rows, alphas, 5, 3, 260826)
    ci = grouped_bootstrap_improvement_ci(
        model_rows,
        cv.predictions,
        cv.baseline_predictions,
        confidence=0.95,
        n_bootstrap=2000,
        seed=260826,
    )
    residuals = [
        abs(row.target - predicted)
        for row, predicted in zip(model_rows, cv.predictions)
    ]
    coverage = {}
    for level in (0.9, 0.95):
        radii = []
        for row in model_rows:
            calibration = [
                residual for other, residual in zip(model_rows, residuals)
                if other.group != row.group
            ]
            radii.append(quantile(calibration, level))
        lower = [
            value - radius for value, radius in zip(cv.predictions, radii)
        ]
        upper = [
            value + radius for value, radius in zip(cv.predictions, radii)
        ]
        coverage[str(level)] = empirical_interval_coverage(
            [row.target for row in model_rows], lower, upper
        )
    tail_pass = all(
        abs(coverage[str(level)] - level) <= 0.1
        for level in (0.9, 0.95)
    )
    alpha = select_ridge_alpha(model_rows, alphas, 5, 260826)
    model = RidgeLinearModel.fit(model_rows, alpha)
    checks = diagnostics(model_rows, model, alpha=alpha)
    coefficient_intervals = grouped_bootstrap_coefficient_ci(
        model_rows, alpha, confidence=0.95, n_bootstrap=500, seed=260826
    )
    return {
        "target": target,
        "feature_order": [
            "context_kib", "units", "transform", "fetch", "live_fetch",
        ],
        "nested_grouped_cv": asdict(cv),
        "relative_improvement_ci95": ci,
        "interval_coverage": coverage,
        "interval_method": (
            "cross-group calibration from other groups' out-of-fold residuals"
        ),
        "tail_coverage_passed": tail_pass,
        "selected_alpha": alpha,
        "raw_intercept": model.raw_intercept,
        "raw_coefficients": model.raw_coefficients,
        "coefficient_ci95": {
            name: interval for name, interval in zip(
                ["intercept", "context_kib", "units", "transform", "fetch",
                 "live_fetch"],
                coefficient_intervals,
            )
        },
        "diagnostics": asdict(checks),
        "promoted": cv.relative_improvement >= 0.2 and ci[0] > 0 and tail_pass,
    }


def _semantic_regrade(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Post-hoc normalized view; never replaces the frozen acceptance gate."""

    rates = {}
    verdicts = {}
    for row in records:
        passed = bool(row.get("accepted"))
        if row["family"] == "summarise" and row.get("output"):
            try:
                value = json.loads(row["output"])
            except json.JSONDecodeError:
                value = {}
            title = str(value.get("document_title", "")).casefold()
            agency = str(value.get("agency", "")).casefold()
            date = str(value.get("publication_date", "")).casefold()
            passed = (
                title.startswith(
                    "medicare program; inpatient rehabilitation facility "
                    "prospective payment system for federal fiscal year 2025"
                )
                and "centers for medicare & medicaid services" in agency
                and ("march 29, 2024" in date or date == "2024-03-29")
                and bool(str(value.get("summary", "")).strip())
            )
        verdicts[row["case_id"]] = passed
    for group in sorted({row["group_id"] for row in records}):
        ids = [row["case_id"] for row in records if row["group_id"] == group]
        rates[group] = sum(verdicts[item] for item in ids) / len(ids)
    return {
        "status": "post-hoc normalized semantic diagnostic; frozen gate unchanged",
        "acceptance_by_shape": rates,
        "quality_gate_passed": all(rate >= 0.8 for rate in rates.values()),
        "case_verdicts": verdicts,
    }


def analyze_pilot(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(records) != 30 or len({row["case_id"] for row in records}) != 30:
        raise ValueError("analysis requires exactly 30 unique pilot records")
    telemetry = all(
        row.get("error") is None
        and row.get("response_calls")
        and all(
            call.get("request_sha256") and call.get("response_sha256")
            for call in row["response_calls"]
        )
        for row in records
    )
    shape_quality = {}
    for group in sorted({row["group_id"] for row in records}):
        rows = [row for row in records if row["group_id"] == group]
        shape_quality[group] = sum(bool(row["accepted"]) for row in rows) / len(rows)
    quality_pass = all(rate >= 0.8 for rate in shape_quality.values())
    summarise = _ladder_gate(records, "summarise", "context_bytes")
    transform = _ladder_gate(records, "transform", "units")
    fetch = _fetch_gate(records)
    targets = {
        target: _model_gate(records, target)
        for target in (
            "input_tokens", "cached_tokens", "output_tokens",
            "reasoning_tokens", "total_tokens", "model_calls", "tool_calls",
        )
    }
    model = targets["total_tokens"]
    return {
        "created_at": _now(),
        "sessions": 30,
        "telemetry_complete": telemetry,
        "acceptance_by_shape": shape_quality,
        "quality_gate_passed": quality_pass,
        "semantic_regrade": _semantic_regrade(records),
        "signal_gates": {
            "summarise": summarise,
            "transform": transform,
            "fetch": fetch,
        },
        "mechanism_expansion_passed": (
            telemetry and quality_pass and summarise["passed"]
            and transform["passed"] and fetch["passed"]
        ),
        "model_gate": model,
        "model_targets": targets,
        "decision": "pause-for-approval",
    }
