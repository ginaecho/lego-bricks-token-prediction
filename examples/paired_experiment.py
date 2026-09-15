"""Run the paired train-predict-execute token experiment.

The training phase meters atomic bricks and fitted compositions, writes the
feature matrix, fits the model, and freezes held-out predictions. The evaluate
phase then dispatches those exact held-out cases and compares actual usage with
the already-written predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from token_yield.business_cases import BusinessCase, load_cases, to_run
from token_yield.compose import Run, fit_form, mape, select_model
from token_yield.copilot_dispatch import CopilotDispatcher
from token_yield.economics import percentile
from token_yield.runner import run_case
from token_yield.tasks import ORDER


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "20260826_1416_wave1"
CATALOG = ROOT / "experiments" / "business_cases" / "cases.jsonl"
SCALED = ROOT / "experiments" / "paired_cases" / "scaled_cases.jsonl"
REAL = ROOT / "experiments" / "paired_cases" / "real_cases.jsonl"
MODEL = "claude-haiku-4.5"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def training_cases() -> List[BusinessCase]:
    """Atomic size ladders plus compositions, with no held-out cases."""

    original = [case for case in load_cases(str(CATALOG)) if not case.held_out]
    return original + load_cases(str(SCALED))


def held_out_cases() -> List[BusinessCase]:
    original = [case for case in load_cases(str(CATALOG)) if case.held_out]
    return original + load_cases(str(REAL))


def _append(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_to_run(record: Dict) -> Run:
    return Run(
        label=record["case_id"],
        notation=record["notation"],
        counts=record["counts"],
        context_bytes=record["context_bytes"],
        arity=record["arity"],
        tokens=record["total_tokens"],
        tool_uses=record.get("tool_uses"),
        held_out=False,
    )


def _write_features(path: Path, runs: Iterable[Run]) -> None:
    fields = ["label", "context_bytes", *ORDER, "total_units", "arity", "tokens"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "label": run.label,
                "context_bytes": run.context_bytes,
                "total_units": run.total_units,
                "arity": run.arity,
                "tokens": run.tokens,
            }
            row.update({slug: run.counts.get(slug, 0) for slug in ORDER})
            writer.writerow(row)


def _training_sha256(records: Sequence[Dict]) -> str:
    canonical = sorted(records, key=lambda row: row["case_id"])
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _model_payload(
    model,
    training_ids: Sequence[str],
    training_sha256: str,
) -> Dict:
    predictive_core = {
        "model_runtime": MODEL,
        "form": model.form,
        "coef": model.coef,
        "boot": model.boot,
        "feature_order": list(model.feature_names),
        "training_sha256": training_sha256,
    }
    payload = {
        **predictive_core,
        "loo_mape": model.loo_mape,
        "excess_loo_error": model.excess_loo_error,
        "excess_skill_vs_constant": model.excess_skill_vs_constant,
        "scores": model.scores,
        "training_ids": list(training_ids),
        "equation": model.equation(),
    }
    encoded = json.dumps(
        predictive_core, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = {"created_at": _now(), **payload}
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _expected_training_ids() -> set:
    return {"null_r1", "null_r2", *(case.case_id for case in training_cases())}


def _expected_held_out_ids() -> set:
    return {case.case_id for case in held_out_cases()}


def _validate_checkpoint(run_dir: Path) -> Dict[str, Dict]:
    model_path = run_dir / "model.json"
    predictions_path = run_dir / "predictions.jsonl"
    records_path = run_dir / "train_records.jsonl"
    if not (model_path.is_file() and predictions_path.is_file()):
        raise RuntimeError("model and predictions must exist as one checkpoint")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    records = _read_jsonl(records_path)
    record_ids = {row["case_id"] for row in records}
    if record_ids != _expected_training_ids():
        raise RuntimeError("frozen training IDs differ from the current partition")
    if record_ids != set(model["training_ids"]):
        raise RuntimeError("training records differ from the persisted model")
    if _training_sha256(records) != model.get("training_sha256"):
        raise RuntimeError("training records changed after model fitting")
    rows = _read_jsonl(predictions_path)
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != _expected_held_out_ids():
        raise RuntimeError("frozen predictions are incomplete or duplicated")
    if any(row.get("model_sha256") != model["sha256"] for row in rows):
        raise RuntimeError("prediction model hash does not match model artifact")
    return {row["case_id"]: row for row in rows}


def run_training(run_dir: Path) -> None:
    """Meter training cases, fit, and freeze held-out predictions."""

    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "train_records.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    model_path = run_dir / "model.json"
    if model_path.exists():
        _validate_checkpoint(run_dir)
        return
    if predictions_path.exists():
        # Predictions publish before the model commit marker. Without the
        # marker they are an interrupted checkpoint and safe to rebuild.
        predictions_path.unlink()
    (run_dir / "model.json.tmp").unlink(missing_ok=True)
    (run_dir / "predictions.jsonl.tmp").unlink(missing_ok=True)
    existing = {record["case_id"] for record in _read_jsonl(records_path)}

    for replicate in (1, 2):
        label = f"null_r{replicate}"
        if label in existing:
            continue
        dispatcher = CopilotDispatcher(run_dir / "raw", label, MODEL)
        result = dispatcher(
            "MEASUREMENT PROBE - null. Do not use tools. Reply with exactly: DONE"
        )
        _append(records_path, {
            "case_id": label,
            "notation": "",
            "counts": {slug: 0 for slug in ORDER},
            "context_bytes": 0,
            "arity": 0,
            "accepted": result.output.strip() == "DONE",
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "tool_uses": result.tool_uses,
            "duration_ms": result.duration_ms,
            "model": result.model,
            "model_version": result.model_version,
            "timestamp": _now(),
            "provenance": "measured",
        })

    for case in training_cases():
        if case.case_id in existing:
            continue
        dispatcher = CopilotDispatcher(run_dir / "raw", case.case_id, MODEL)
        record = run_case(case, dispatcher)
        row = asdict(record)
        row["notation"] = case.notation()
        row["arity"] = case.arity
        _append(records_path, row)

    records = _read_jsonl(records_path)
    if {record["case_id"] for record in records} != _expected_training_ids():
        raise RuntimeError("training partition is incomplete")
    runs = [_record_to_run(record) for record in records]
    _write_features(run_dir / "feature_matrix.csv", runs)
    model = select_model(runs)
    payload = _model_payload(
        model,
        [run.label for run in runs],
        _training_sha256(records),
    )
    model_tmp = run_dir / "model.json.tmp"
    model_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    prediction_rows = []
    for case in held_out_cases():
        predicted = model.predict(case.counts, case.context_bytes)
        prediction_rows.append({
            "case_id": case.case_id,
            "predicted_at": _now(),
            "predicted_tokens": round(predicted),
            "counts": case.counts,
            "context_bytes": case.context_bytes,
            "model_sha256": payload["sha256"],
            "model_form": model.form,
        })
    predictions_tmp = run_dir / "predictions.jsonl.tmp"
    predictions_tmp.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows
        ),
        encoding="utf-8",
    )
    predictions_tmp.replace(predictions_path)
    model_tmp.replace(model_path)


def run_evaluation(run_dir: Path) -> None:
    """Execute held-out tasks only after predictions have been frozen."""

    predictions = _validate_checkpoint(run_dir)
    records_path = run_dir / "held_out_records.jsonl"
    existing = {record["case_id"] for record in _read_jsonl(records_path)}
    cases = held_out_cases()
    for case in cases:
        if case.case_id in existing:
            continue
        dispatcher = CopilotDispatcher(
            run_dir / "raw", f"heldout_{case.case_id}", MODEL
        )
        record = run_case(case, dispatcher)
        row = asdict(record)
        row["notation"] = case.notation()
        row["arity"] = case.arity
        row["predicted_tokens"] = predictions[case.case_id]["predicted_tokens"]
        row["absolute_percentage_error"] = abs(
            row["total_tokens"] - row["predicted_tokens"]
        ) / row["total_tokens"]
        _append(records_path, row)

    evaluated = _read_jsonl(records_path)
    actual = [row["total_tokens"] for row in evaluated]
    predicted = [row["predicted_tokens"] for row in evaluated]
    summary = {
        "created_at": _now(),
        "model_runtime": MODEL,
        "cases": len(evaluated),
        "accepted": sum(row["accepted"] for row in evaluated),
        "mape": mape(actual, predicted),
        "mean_actual_tokens": sum(actual) / len(actual),
        "mean_predicted_tokens": sum(predicted) / len(predicted),
        "rows": [{
            "case_id": row["case_id"],
            "accepted": row["accepted"],
            "predicted_tokens": row["predicted_tokens"],
            "actual_tokens": row["total_tokens"],
            "absolute_percentage_error": row["absolute_percentage_error"],
        } for row in evaluated],
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_diagnostics(run_dir, evaluated, cases)


def _request_count(run_dir: Path, case_id: str, held_out: bool) -> int:
    prefix = "heldout_" if held_out else ""
    path = run_dir / "raw" / f"{prefix}{case_id}_usage.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["modelMetrics"][MODEL]["requests"]["count"])


def _write_diagnostics(
    run_dir: Path,
    evaluated: Sequence[Dict],
    cases: Sequence[BusinessCase],
) -> None:
    """Compare the forced LEGO form and empirical tail without refitting truth."""

    training_rows = _read_jsonl(run_dir / "train_records.jsonl")
    training_runs = [_record_to_run(record) for record in training_rows]
    brick_model = fit_form(training_runs, "bytes+per-primitive")
    cases_by_id = {case.case_id: case for case in cases}
    forced_predictions = []
    for row in evaluated:
        case = cases_by_id[row["case_id"]]
        predicted = round(brick_model.predict(case.counts, case.context_bytes))
        forced_predictions.append({
            "case_id": case.case_id,
            "predicted_tokens": predicted,
            "actual_tokens": row["total_tokens"],
            "absolute_percentage_error": abs(
                row["total_tokens"] - predicted
            ) / row["total_tokens"],
        })
    tokens = [row["total_tokens"] for row in training_rows]
    training_requests = {
        row["case_id"]: _request_count(run_dir, row["case_id"], False)
        for row in training_rows
    }
    one_call_tokens = [
        row["total_tokens"] for row in training_rows
        if training_requests[row["case_id"]] == 1
    ]
    multi_call_tokens = [
        row["total_tokens"] for row in training_rows
        if training_requests[row["case_id"]] > 1
    ]
    diagnostics = {
        "created_at": _now(),
        "forced_brick_model": {
            "status": "post-hoc hypothesis diagnostic; not the frozen selected model",
            "form": brick_model.form,
            "equation": brick_model.equation(),
            "loo_mape": brick_model.loo_mape,
            "held_out_mape": mape(
                [row["actual_tokens"] for row in forced_predictions],
                [row["predicted_tokens"] for row in forced_predictions],
            ),
            "rows": forced_predictions,
        },
        "training_token_distribution": {
            "p50": percentile(tokens, 0.50),
            "p90": percentile(tokens, 0.90),
            "p95": percentile(tokens, 0.95),
            "p99": percentile(tokens, 0.99),
            "max": max(tokens),
        },
        "held_out_request_counts": {
            row["case_id"]: _request_count(run_dir, row["case_id"], True)
            for row in evaluated
        },
        "training_request_counts": training_requests,
        "request_count_effect": {
            "one_call_runs": len(one_call_tokens),
            "one_call_mean_tokens": sum(one_call_tokens) / len(one_call_tokens),
            "multi_call_runs": len(multi_call_tokens),
            "multi_call_mean_tokens": (
                sum(multi_call_tokens) / len(multi_call_tokens)
                if multi_call_tokens else None
            ),
        },
    }
    (run_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--phase", choices=("train", "evaluate", "all"), default="all"
    )
    args = parser.parse_args()
    if args.phase in ("train", "all"):
        run_training(args.run_dir)
    if args.phase in ("evaluate", "all"):
        run_evaluation(args.run_dir)


if __name__ == "__main__":
    main()
