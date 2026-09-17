"""Audited, offline T011 composition research; never a deployment recommendation.

The comparison is direct workflow regression versus a sum of call predictions,
not a purported batching discount. Every call baseline is fitted again without
the validation project in each fold. Historical holdout is descriptive.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .budget import HardBudget
from .customer_decomposition import SCOPING_BRICKS, check_output, content_hash, quote_features
from .customer_models import _model_from_params, _model_params
from .customer_pilot import _require_measured_channels
from .economics import Pricing
from .foundry_dispatch import _response_text, parse_usage
from .marketplace_measurements import LAYOUTS, _summary, prepare_campaign
from .robust import ConstantModel, Record, RidgeLinearModel


SCHEMA = "marketplace-workflow-research-v1"
TARGETS = ("input_tokens", "output_tokens", "rated_cost_usd")
BASE_FEATURES = ("context_bytes", *SCOPING_BRICKS)
CALL_FEATURES = ("context_bytes", "prompt_bytes", "planned_output_tokens", *SCOPING_BRICKS)
COMPOSITION_FEATURES = (
    *BASE_FEATURES, "prompt_bytes", "planned_output_tokens", "planned_calls",
    *(f"layout:{name}" for name in LAYOUTS),
)
FORMS = {"constant": (), "baseline": CALL_FEATURES, "composition": COMPOSITION_FEATURES}
LIMITATIONS = [
    "Research only: no production recommendation or automatic baseline replacement.",
    "Four training projects, one repeat per layout, one shared template; high selection uncertainty.",
    "Historical holdout comprises two previously known briefs, not fresh independent testing.",
    "Selection CV is exploratory and reused for selection, not an unbiased post-selection estimate.",
    "CV errors include held-project extrapolation diagnostics; public quotes still abstain outside training ranges.",
    "No calibration set, calibrated intervals, coverage, specialist, or unseen-runtime claims.",
    "Independent original-brief services only; serial scheduling is not sequential handoff semantics.",
    "Measured composition association is not a universal batching discount or causal benefit.",
    "Structural contract acceptance does not establish human quality or noninferiority.",
    "Rated cost uses frozen retail prices and observed cache usage, not an invoice or future cache guarantee.",
    "Control-plane observations bracket calls but do not attest each response's backend version.",
    "Local hashes detect changes relative to the audited snapshot; they are not signed provider attestations.",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite numeric value required")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("numeric value exceeds supported range") from exc
    _require(math.isfinite(number), "finite numeric value required")
    return number


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("nonnegative integer required")
    return value


def _same(actual: object, expected: object, label: str) -> None:
    _require(content_hash(actual) == content_hash(expected), f"{label} differs from frozen evidence")


def _json(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "audit timestamp must include timezone")
    return parsed


def implementation_hashes() -> dict:
    """Hash the actual local regression, feature, parser, and CLI implementations."""
    package = Path(__file__).resolve().parent
    paths = [package / name for name in (
        "marketplace_models.py", "robust.py", "customer_models.py",
        "customer_decomposition.py", "foundry_dispatch.py", "customer_pilot.py",
        "marketplace_measurements.py", "budget.py", "economics.py",
    )]
    paths.append(package.parent / "examples" / "marketplace_models.py")
    return {str(path.relative_to(package.parent)): _sha(path.read_bytes()) for path in paths}


def runtime_signature(manifest: dict) -> dict:
    config = manifest["config"]
    return {
        **{key: config[key] for key in (
            "endpoint", "deployment", "expected_response_model", "deployment_version",
            "deployment_sku", "reasoning_effort", "text_verbosity", "pricing",
        )},
        "template_sha256": manifest["provenance"]["template_sha256"],
        "runtime_file_sha256": manifest["provenance"]["runtime_file_sha256"],
        "semantics": "independent-original-brief",
        "tools": 0, "retries": 0,
    }


def workflow_quote(project: dict, template: dict, variant: str) -> list[dict]:
    """Build quote-time features without reading outputs or making any calls."""
    _require(variant in LAYOUTS, "unsupported layout; sequential handoffs are blocked")
    version = "marketplace-services-v1:" + content_hash(template)
    result = []
    for operations in LAYOUTS[variant]:
        quote = quote_features(project, template, operations)
        result.append({
            "operations": list(operations), "quote": quote,
            "service_selections": [
                {"service_id": op, "version": version, "quantity": quote["counts"][op]}
                for op in operations
            ],
        })
    return result


def _features(calls: list[dict], variant: str, runtime: dict) -> dict:
    _require(variant in LAYOUTS, "unsupported layout")
    _require(isinstance(calls, list) and len(calls) == len(LAYOUTS[variant]),
             "incomplete workflow")
    totals = dict.fromkeys(SCOPING_BRICKS, 0)
    contexts, prompt_bytes, output_cap = [], 0, 0
    for call, operations in zip(calls, LAYOUTS[variant]):
        _same(call["operations"], list(operations), "operation combination/order")
        quote = call["quote"]
        _require(set(quote) == {"counts", "context_bytes", "prompt_bytes", "planned_output_tokens"},
                 "unknown quote fields")
        _require(set(quote["counts"]) == set(SCOPING_BRICKS), "unknown service counts")
        contexts.append(_integer(quote["context_bytes"]))
        prompt_bytes += _integer(quote["prompt_bytes"])
        output_cap += _integer(quote["planned_output_tokens"])
        for op in SCOPING_BRICKS:
            count = _integer(quote["counts"][op])
            _require((count > 0) == (op in operations), "service quantity/combination mismatch")
            totals[op] += count
        _same(call["service_selections"], [
            {"service_id": op, "version": "marketplace-services-v1:" + runtime["template_sha256"],
             "quantity": quote["counts"][op]} for op in operations
        ], "service version or quantity")
    _require(len(set(contexts)) == 1 and contexts[0] > 0, "inconsistent original brief")
    _require(totals["extract"] == totals["classify"] == totals["plan"] and totals["report"] == 1,
             "unsupported service quantity combination")
    return {
        **totals, "context_bytes": contexts[0], "prompt_bytes": prompt_bytes,
        "planned_output_tokens": output_cap, "planned_calls": len(calls),
        **{f"layout:{name}": int(name == variant) for name in LAYOUTS},
    }


def audit_run(root: Path, source_run: Path) -> dict:
    """Reject partial evidence before opening labels; replay every completed call offline."""
    _require(not Path(source_run).is_symlink(), "linked source directories are unsupported")
    root, source_run = Path(root).resolve(), Path(source_run).resolve()
    _require(source_run.parent == root / "runs", "source must be a direct child of repository runs")
    hashes = {}

    def read(name):
        path = source_run / name
        _require(not path.is_symlink() and path.is_file(), f"missing or linked evidence: {name}")
        raw = path.read_bytes()
        hashes[name] = _sha(raw)
        return raw

    state = _json(read("state.json"))
    _require(state.get("status") == "completed_research_only", "source run is incomplete; labels not read")
    campaign = prepare_campaign(root)
    _same(_json(read("protocol.json")), campaign, "protocol")
    manifest, config = campaign["manifest"], campaign["manifest"]["config"]
    runtime = runtime_signature(manifest)
    bracket = []
    for phase in ("before", "after"):
        audit = _json(read(f"deployment-{phase}.json"))
        metadata = audit["metadata"]
        _require(metadata.get("model", {}).get("name") == config["deployment"]
                 and metadata["model"].get("version") == config["deployment_version"]
                 and metadata.get("sku") == config["deployment_sku"]
                 and metadata.get("state") == "Succeeded", "deployment identity mismatch")
        bracket.append(_time(audit["observed_at"]))
    _require(bracket[0] <= bracket[1] <= _time(state["at"]), "invalid deployment time bracket")
    rows = _json(read("records.json"))
    _require(isinstance(rows, list) and len(rows) == len(campaign["calls"]),
             "incomplete run/workflows")
    _require(len({row["call_id"] for row in rows}) == len(rows), "duplicate call record")
    projects = {p["id"]: p for p in manifest["catalog"]["projects"]}
    budget, pricing = HardBudget(config["cap_usd"], config["stop_usd"]), Pricing(**config["pricing"])
    response_ids = set()
    for row, call in zip(rows, campaign["calls"]):
        frozen = {key: value for key, value in call.items() if key != "payload"}
        _same({key: row[key] for key in frozen}, frozen, "call record")
        _require(row["status"] == row["response_status"] == "completed"
                 and row["dispatched"] is True, "incomplete or undispatched call")
        _require(bracket[0] <= _time(row["started_at"]) <= _time(row["dispatched_at"])
                 <= _time(row["response_received_at"]) <= _time(row["finished_at"]) <= bracket[1],
                 "call lies outside deployment time bracket")
        request = read(call["call_id"] + ".request.json")
        response = read(call["call_id"] + ".response.json")
        _require(_sha(request) == row["request_sha256"], "request hash mismatch")
        _same(_json(request), call["payload"], "request payload")
        _require(row["response_redacted"] is False
                 and _sha(response) == row["response_sha256"] == row["stored_response_sha256"],
                 "response hash mismatch or redacted evidence")
        _require(type(row["http_status"]) is int and 200 <= row["http_status"] < 300,
                 "unsuccessful HTTP response")
        raw = _json(response)
        _require_measured_channels(response)
        usage = asdict(parse_usage(raw))
        _same(row["usage"], usage, "measured usage")
        _require(raw.get("status") == "completed"
                 and raw.get("model") == row["response_model"] == config["expected_response_model"],
                 "raw response identity or completion mismatch")
        response_id = raw.get("id")
        _require(isinstance(response_id, str) and bool(response_id)
                 and response_id not in response_ids, "missing or duplicate raw response id")
        response_ids.add(response_id)
        raw_output = raw.get("output", [])
        _require(isinstance(raw_output, list) and not raw.get("tools") and all(
            isinstance(item, dict) and item.get("type") in ("message", "reasoning") for item in raw_output
        ), "unexpected tool use or raw output type")
        output = _response_text(raw)
        _same(row["output"], output, "raw output")
        quality = check_output(projects[call["project_id"]], tuple(call["operations"]), output)
        _require(quality["contract_passed"] is True, "raw output failed structural contract")
        _same(row["quality"], quality, "structural acceptance")
        _require(isinstance(row["response_calls"], list) and len(row["response_calls"]) == 1,
                 "expected exactly one measured response call")
        response_call = row["response_calls"][0]
        _integer(response_call["elapsed_ms"])
        _same({k: v for k, v in response_call.items() if k != "elapsed_ms"}, {
            "target": call["call_id"], "response_id": response_id, "status": "completed",
            "model": config["expected_response_model"], "usage": usage,
            "request_sha256": _sha(request), "response_sha256": _sha(response),
        }, "response ledger")
        _require(usage["input_tokens"] <= call["max_input_tokens"]
                 and usage["output_tokens"] <= call["max_output_tokens"]
                 and usage["reasoning_tokens"] <= call["max_output_tokens"],
                 "usage exceeds reservation")
        rated = pricing.attempt_cost(input_tokens=usage["input_tokens"],
                                    cached_input_tokens=usage["cached_tokens"],
                                    output_tokens=usage["output_tokens"])
        _same(row["rated_cost_usd"], rated, "rated cost")
        budget.reserve(call["call_id"], call["max_input_tokens"], 2 * call["max_output_tokens"])
        budget.settle(call["call_id"], usage, rated)
    _same(_json(read("budget.json")), budget.snapshot(), "replayed budget")
    _same(_json(read("analysis.json")),
          _summary(campaign, rows, budget, "completed_research_only"), "run/workflow completeness")
    workflows = []
    for project in manifest["projects"]:
        for variant in LAYOUTS:
            observed = [r for r in rows if r["project_id"] == project["project_id"]
                        and r["variant"] == variant]
            calls = [{key: r[key] for key in ("operations", "quote", "service_selections")}
                     for r in observed]
            workflows.append({
                "project_id": project["project_id"], "group": project["group"],
                "variant": variant, "calls": calls,
                "features": _features(calls, variant, runtime),
                "targets": {
                    "input_tokens": sum(r["usage"]["input_tokens"] for r in observed),
                    "output_tokens": sum(r["usage"]["output_tokens"] for r in observed),
                    "rated_cost_usd": sum(r["rated_cost_usd"] for r in observed),
                },
                "call_targets": [{
                    "input_tokens": r["usage"]["input_tokens"],
                    "output_tokens": r["usage"]["output_tokens"],
                    "rated_cost_usd": r["rated_cost_usd"],
                } for r in observed],
                "measured_usage": {key: sum(r["usage"][key] for r in observed)
                                   for key in rows[0]["usage"]},
                "contract_passed": True, "human_quality_established": False,
            })
    for name, digest in hashes.items():
        _require(_sha((source_run / name).read_bytes()) == digest, "source changed during audit")
    return {
        "schema_version": SCHEMA, "runtime": runtime, "workflows": workflows,
        "provenance": {
            "source_run": source_run.name, "source_file_sha256": hashes,
            "campaign_sha256": campaign["manifest_sha256"],
            "frozen_inputs": manifest["provenance"],
            "project_groups": {p["project_id"]: p["group"] for p in manifest["projects"]},
            "original_splits": {p["project_id"]: p["original_split"] for p in manifest["projects"]},
            "implementation_sha256": implementation_hashes(),
        },
        "projects": manifest["projects"], "template": manifest["template"],
        "limitations": LIMITATIONS,
    }


def _vector(row: dict, form: str) -> tuple:
    return tuple(row["features"][key] for key in FORMS[form]) or (0.0,)


def _call_vector(call: dict) -> tuple:
    features = {**call["quote"], **call["quote"]["counts"]}
    return tuple(features[key] for key in CALL_FEATURES)


def _fit_model(rows: list[dict], target: str, form: str):
    if form == "baseline":
        records = [
            Record(_call_vector(call), targets[target], row["project_id"])
            for row in rows for call, targets in zip(row["calls"], row["call_targets"])
        ]
        return RidgeLinearModel.fit(records, alpha=10.0)
    records = [Record(_vector(row, form), row["targets"][target], row["project_id"]) for row in rows]
    return ConstantModel.fit(records) if form == "constant" else RidgeLinearModel.fit(records, alpha=10.0)


def _predict(model, row: dict, form: str) -> float:
    if form == "baseline":
        return sum(max(0.0, _number(model.predict(_call_vector(call)))) for call in row["calls"])
    return max(0.0, _number(model.predict(_vector(row, form))))


def _validated_rows(rows: list[dict], runtime: dict, group: str, expected_groups: int) -> list[dict]:
    _require(isinstance(rows, list) and bool(rows), "workflows required")
    _require(all(row["group"] == group for row in rows),
             "holdout must never enter training or selection; fixed split required")
    seen, projects = set(), set()
    for row in rows:
        project = row["project_id"]
        _require(isinstance(project, str) and bool(project), "project identity required")
        pair = (project, row["variant"])
        _require(pair not in seen, "duplicate project-layout target")
        seen.add(pair)
        projects.add(project)
        _same(row["features"], _features(row["calls"], row["variant"], runtime), "workflow features")
        _require(set(row["targets"]) == set(TARGETS), "target channels differ")
        for value in row["targets"].values():
            _require(_number(value) >= 0, "targets must be nonnegative")
        _require(len(row["call_targets"]) == len(row["calls"]), "incomplete call targets")
        for targets in row["call_targets"]:
            _require(set(targets) == set(TARGETS), "call target channels differ")
            for value in targets.values():
                _require(_number(value) >= 0, "call targets must be nonnegative")
        _same(row["targets"], {key: sum(t[key] for t in row["call_targets"]) for key in TARGETS},
              "workflow target sum")
        _require(row["contract_passed"] is True and row["human_quality_established"] is False,
                 "structural contract is required but not human quality evidence")
    _require(len(projects) == expected_groups, f"expected {expected_groups} fixed project groups")
    _require(seen == {(project, variant) for project in projects for variant in LAYOUTS},
             "incomplete workflow matrix")
    return sorted(rows, key=lambda row: (row["project_id"], row["variant"]))


def _support(rows: list[dict], runtime: dict) -> dict:
    layouts = {}
    for variant in LAYOUTS:
        subset = [r for r in rows if r["variant"] == variant]
        layouts[variant] = {
            "batches": [list(batch) for batch in LAYOUTS[variant]],
            "feature_ranges": {
                key: [min(r["features"][key] for r in subset), max(r["features"][key] for r in subset)]
                for key in COMPOSITION_FEATURES
            },
            "call_ranges": [{
                key: [min(r["calls"][index]["quote"][key] for r in subset),
                      max(r["calls"][index]["quote"][key] for r in subset)]
                for key in ("context_bytes", "prompt_bytes", "planned_output_tokens")
            } for index in range(len(LAYOUTS[variant]))],
        }
    return {
        "runtime": runtime, "training_projects": sorted({r["project_id"] for r in rows}),
        "layouts": layouts, "policy": "training-only per-layout marginal ranges; not joint-distribution proof",
        "calibrated_intervals": None, "production_recommendation": None,
    }


def fit_workflow_models(training_rows: list[dict], runtime: dict, provenance: dict) -> dict:
    """Fit only four complete training groups; candidate order is the deterministic tie-break."""
    rows = _validated_rows(training_rows, runtime, "train", 4)
    groups = sorted({row["project_id"] for row in rows})
    _same(groups, sorted(key for key, role in provenance["project_groups"].items() if role == "train"),
          "fixed training project roles")
    validation, candidates, selected = {}, {}, {}
    for target in TARGETS:
        validation[target], candidates[target] = {}, {}
        for form in FORMS:
            folds = []
            for held in groups:
                training = [row for row in rows if row["project_id"] != held]
                held_rows = [row for row in rows if row["project_id"] == held]
                fitted = _fit_model(training, target, form)
                predictions = [{
                    "project_id": held, "variant": row["variant"], "actual": row["targets"][target],
                    "predicted": _predict(fitted, row, form),
                } for row in held_rows]
                folds.append({
                    "training_projects": [g for g in groups if g != held], "validation_project": held,
                    "predictions": predictions,
                    "predictions_are_diagnostic_not_supported_quotes": True,
                    "mae": sum(abs(p["actual"] - p["predicted"]) for p in predictions) / len(predictions),
                })
            validation[target][form] = {
                "group_macro_mae": sum(fold["mae"] for fold in folds) / len(folds), "folds": folds,
            }
            candidates[target][form] = _model_params(_fit_model(rows, target, form))
        selected[target] = min(FORMS, key=lambda f: validation[target][f]["group_macro_mae"])
    model = {
        "schema_version": SCHEMA, "selected": selected, "candidates": candidates,
        "feature_names": {key: list(value) for key, value in FORMS.items()},
        "alpha": 10.0, "clamp_predictions_at_zero": True,
        "selection": {
            "method": "leave-one-training-project-out; fixed candidates; macro project MAE",
            "candidate_order": list(FORMS), "validation": validation,
            "baseline_definition": "sum of call-level size/count ridge predictions, freshly fitted within each fold",
            "composition_definition": "direct workflow ridge with layout/overhead features, plus constant comparator",
            "holdout_used": False,
        },
        "support": _support(rows, runtime), "provenance": provenance,
        "training_rows_sha256": content_hash(rows), "limitations": LIMITATIONS,
    }
    model["model_sha256"] = content_hash(model)
    return model


def load_model(value: dict) -> dict:
    """Validate and reconstruct serialized coefficients; never refit on reload."""
    model = _json(json.dumps(value, allow_nan=False).encode("utf-8"))
    _require(model["schema_version"] == SCHEMA, "unsupported model schema")
    _same(model["model_sha256"], content_hash({k: v for k, v in model.items() if k != "model_sha256"}),
          "model digest")
    _same(model["feature_names"], {key: list(value) for key, value in FORMS.items()}, "feature specification")
    _same(model["provenance"]["implementation_sha256"], implementation_hashes(), "model implementation")
    _require(model["alpha"] == 10.0 and model["clamp_predictions_at_zero"] is True
             and model["selection"]["holdout_used"] is False, "unsupported fitting policy")
    _require(set(model["selected"]) == set(model["candidates"]) == set(TARGETS), "target models differ")
    _require(set(model["support"]["layouts"]) == set(LAYOUTS), "unsupported support layouts")
    _require(len(set(model["support"]["training_projects"])) == 4, "four training projects required")
    _same(model["support"]["training_projects"], sorted(
        key for key, role in model["provenance"]["project_groups"].items() if role == "train"
    ), "fixed training project roles")
    for target in TARGETS:
        _require(model["selected"][target] in FORMS and set(model["candidates"][target]) == set(FORMS),
                 "unknown candidate")
        for form in FORMS:
            _model_from_params(model["candidates"][target][form],
                               "constant" if form == "constant" else "lego", FORMS[form])
    for variant, support in model["support"]["layouts"].items():
        _same(support["batches"], [list(batch) for batch in LAYOUTS[variant]], "support layout")
        _require(set(support["feature_ranges"]) == set(COMPOSITION_FEATURES)
                 and len(support["call_ranges"]) == len(LAYOUTS[variant]), "invalid support dimensions")
        for ranges in support["call_ranges"]:
            _require(set(ranges) == {"context_bytes", "prompt_bytes", "planned_output_tokens"},
                     "invalid call support dimensions")
        for ranges in [support["feature_ranges"], *support["call_ranges"]]:
            for low, high in ranges.values():
                _require(0 <= _number(low) <= _number(high), "invalid support range")
    _same(model["limitations"], LIMITATIONS, "research limitations")
    return model


def predict_workflow(model: dict, calls: list[dict], variant: str, runtime: dict) -> dict:
    """Abstain on unknown runtime, contract, composition or training-range extrapolation."""
    model = load_model(model)
    abstention = {
        "status": "abstained", "predictions": None, "calibrated_intervals": None,
        "production_recommendation": None,
    }
    try:
        _same(runtime, model["support"]["runtime"], "runtime")
        features = _features(calls, variant, runtime)
        support = model["support"]["layouts"][variant]
        for key, (low, high) in support["feature_ranges"].items():
            _require(low <= features[key] <= high, f"out-of-range quote: {key}")
        for call, ranges in zip(calls, support["call_ranges"]):
            for key, (low, high) in ranges.items():
                _require(low <= call["quote"][key] <= high, f"out-of-range call quote: {key}")
    except (KeyError, TypeError, ValueError) as exc:
        return {**abstention, "reason": str(exc)}
    predictions = {}
    for target in TARGETS:
        form = model["selected"][target]
        fitted = _model_from_params(model["candidates"][target][form],
                                    "constant" if form == "constant" else "lego", FORMS[form])
        predictions[target] = _predict(fitted, {"features": features, "calls": calls}, form)
    return {
        "status": "research_estimate", "predictions": predictions, "calibrated_intervals": None,
        "production_recommendation": None, "selected": model["selected"],
        "model_sha256": model["model_sha256"], "limits": LIMITATIONS,
    }


def evaluate_historical_holdout(model: dict, holdout_rows: list[dict]) -> dict:
    """Report all candidates descriptively, without changing selection or refitting."""
    model = load_model(model)
    rows = _validated_rows(holdout_rows, model["support"]["runtime"], "historical_holdout", 2)
    _same(sorted({row["project_id"] for row in rows}), sorted(
        key for key, role in model["provenance"]["project_groups"].items() if role == "historical_holdout"
    ), "fixed historical holdout project roles")
    _require(not ({row["project_id"] for row in rows} & set(model["support"]["training_projects"])),
             "historical holdout overlaps training projects")
    results = []
    for row in rows:
        quote = predict_workflow(model, row["calls"], row["variant"], model["support"]["runtime"])
        comparison = None
        if quote["status"] == "research_estimate":
            comparison = {}
            for target in TARGETS:
                comparison[target] = {}
                for form in FORMS:
                    fitted = _model_from_params(model["candidates"][target][form],
                                                "constant" if form == "constant" else "lego", FORMS[form])
                    estimate = _predict(fitted, row, form)
                    comparison[target][form] = {
                        "predicted": estimate, "absolute_error": abs(estimate - row["targets"][target]),
                    }
        results.append({
            "project_id": row["project_id"], "variant": row["variant"], "actual": row["targets"],
            "quote": quote, "candidate_comparison": comparison,
        })
    metrics = {}
    for target in TARGETS:
        metrics[target] = {}
        for form in FORMS:
            group_errors = []
            for project in sorted({row["project_id"] for row in rows}):
                supported = [r for r in results if r["project_id"] == project and r["candidate_comparison"]]
                if supported:
                    group_errors.append(sum(r["candidate_comparison"][target][form]["absolute_error"]
                                            for r in supported) / len(supported))
            metrics[target][form] = sum(group_errors) / len(group_errors) if group_errors else None
    return {
        "role": "historical_holdout_descriptive_only", "selection_influence": False,
        "rows": results, "supported_workflows": sum(r["candidate_comparison"] is not None for r in results),
        "total_workflows": len(rows), "supported_only_group_macro_mae": metrics,
        "limitations": LIMITATIONS,
    }


def run_offline(root: Path, source_run: Path, destination: Path) -> dict:
    """Write new research artifacts only, after complete audit and a no-refit round trip."""
    _require(not Path(destination).is_symlink() and not Path(source_run).is_symlink(),
             "linked source or destination directories are unsupported")
    root, source_run, destination = Path(root).resolve(), Path(source_run).resolve(), Path(destination).resolve()
    _require(destination.parent == root / "runs" and destination != source_run,
             "destination must be a new, separate direct child of runs")
    if destination.exists():
        raise FileExistsError(destination)
    audited = audit_run(root, source_run)
    train = [row for row in audited["workflows"] if row["group"] == "train"]
    holdout = [row for row in audited["workflows"] if row["group"] == "historical_holdout"]
    model = fit_workflow_models(train, audited["runtime"], audited["provenance"])
    reloaded = load_model(_json(json.dumps(model, allow_nan=False).encode("utf-8")))
    _same(model, reloaded, "JSON model round trip")
    report = {
        "schema_version": SCHEMA, "status": "completed_research_only",
        "source_audit": audited["provenance"], "selected": model["selected"],
        "training_validation": model["selection"],
        "historical_holdout": evaluate_historical_holdout(reloaded, holdout),
        "measured_workflows": audited["workflows"],
        "production_recommendation": None, "saved_baseline_replaced": False,
        "quality_noninferiority_established": False, "calibrated_intervals": None,
        "limitations": LIMITATIONS,
    }
    for name, digest in audited["provenance"]["source_file_sha256"].items():
        _require(_sha((source_run / name).read_bytes()) == digest, "source changed before publication")
    destination.mkdir(exist_ok=False)
    for name, value in (("model.json", reloaded), ("evaluation.json", report)):
        with (destination / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    _same(load_model(_json((destination / "model.json").read_bytes())), model, "persisted model")
    return report
