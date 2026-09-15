"""Offline, project-grouped comparisons of quote-time customer cost models.

Only completed, fully metered training calls are targets. Quality acceptance is
deliberately irrelevant: an unsuccessful deliverable still incurred its cost.
Cross-validation residuals are exploratory, not calibrated coverage estimates.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict

from token_yield.robust import ConstantModel, Record, RidgeLinearModel, grouped_kfold


_SCHEMA_VERSION = 1
_ALPHA = 10.0
_UNITS = ("extract", "classify", "plan", "report")
_FEATURES = {
    "constant": (),
    "size": ("prompt_bytes",),
    "size+units": ("prompt_bytes", "total_scoping_units", "planned_output_tokens"),
    "lego": ("prompt_bytes", "planned_output_tokens", *_UNITS),
}
_CALIBRATION_STATUS = "insufficient_independent_groups_shared_template"


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dict")
    return value


def _required(value: dict, key: str) -> object:
    if key not in value:
        raise ValueError(f"missing required field: {key}")
    return value[key]


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must be nonempty")
    return value


def _number(value: object, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    if not -sys.float_info.max <= value <= sys.float_info.max or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    _number(value, label)
    return value


def _quote_features(quote: object) -> dict:
    quote = _mapping(quote, "quote")
    values = {
        name: _count(_required(quote, name), name)
        for name in ("context_bytes", "prompt_bytes", "planned_output_tokens")
    }
    counts = _mapping(_required(quote, "counts"), "counts")
    if set(counts) != set(_UNITS):
        raise ValueError("counts must contain exactly extract, classify, plan, report")
    values.update({name: _count(counts[name], name) for name in _UNITS})
    values["total_scoping_units"] = sum(counts.values())
    _number(values["total_scoping_units"], "total_scoping_units")
    return values


def _validate_usage(value: object) -> None:
    usage = _mapping(value, "usage")
    for name in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
        _count(_required(usage, name), name)
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise ValueError("usage total_tokens must equal input_tokens + output_tokens")
    for name, parent in (("cached_tokens", "input_tokens"), ("reasoning_tokens", "output_tokens")):
        if usage[name] > usage[parent]:
            raise ValueError(f"usage {name} must not exceed {parent}")


def _completed_rows(rows: list[dict], *, training: bool) -> tuple[list[dict], list[str]]:
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    if not rows:
        raise ValueError("rows must not be empty")
    seen = set()
    completed = []
    excluded = []
    for value in rows:
        row = _mapping(value, "row")
        call_id = _identifier(_required(row, "call_id"), "call_id")
        _identifier(_required(row, "project_id"), "project_id")
        if call_id in seen:
            raise ValueError(f"duplicate call_id: {call_id}")
        seen.add(call_id)
        split = _identifier(_required(row, "split"), "split")
        if split not in ("train", "holdout"):
            raise ValueError("split must be train or holdout")
        if training and split == "holdout":
            raise ValueError("holdout rows must never be passed to fit_models")
        status = _identifier(_required(row, "status"), "status")
        if status != "completed":
            excluded.append(call_id)
            continue
        _number(_required(row, "rated_cost_usd"), "rated_cost_usd")
        _validate_usage(_required(row, "usage"))
        _quote_features(_required(row, "quote"))
        completed.append(row)
    if not completed:
        raise ValueError("at least one completed, metered target row is required")
    return sorted(completed, key=lambda row: row["call_id"]), sorted(excluded)


def _features(values: dict, form: str) -> tuple:
    # Record requires a nonempty vector even for an intercept-only model.
    return tuple(values[name] for name in _FEATURES[form]) or (0.0,)


def _fit(records: list[Record], form: str) -> ConstantModel | RidgeLinearModel:
    try:
        model = (
            ConstantModel.fit(records)
            if form == "constant"
            else RidgeLinearModel.fit(records, alpha=_ALPHA)
        )
    except OverflowError as exc:
        raise ValueError("model inputs exceed supported numeric range") from exc
    _model_from_params(_model_params(model), form)
    return model


def _model_params(model: ConstantModel | RidgeLinearModel) -> dict:
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in asdict(model).items()
    }


def _model_from_params(params: object, form: str) -> ConstantModel | RidgeLinearModel:
    params = _mapping(params, "model")
    if form == "constant":
        return ConstantModel(_number(_required(params, "value"), "value", nonnegative=False))
    width = len(_FEATURES[form])
    vectors = {}
    for name in ("coefficients", "means", "scales", "active"):
        value = _required(params, name)
        if not isinstance(value, list):
            raise TypeError(f"model {name} must be a list")
        vectors[name] = value
    if len(vectors["means"]) != width or len(vectors["scales"]) != width:
        raise ValueError("model feature dimensions do not match form")
    active = vectors["active"]
    for index in active:
        _count(index, "active index")
        if index >= width:
            raise ValueError("active index exceeds model feature dimensions")
    if active != sorted(set(active)) or len(vectors["coefficients"]) != len(active):
        raise ValueError("model active indices and coefficients are inconsistent")
    for name in ("coefficients", "means", "scales"):
        for value in vectors[name]:
            _number(value, f"model {name}", nonnegative=(name == "scales"))
    if active != [j for j, scale in enumerate(vectors["scales"]) if scale > 1e-12]:
        raise ValueError("model active indices do not match nonconstant scales")
    alpha = _number(_required(params, "alpha"), "alpha")
    if alpha != _ALPHA:
        raise ValueError("model alpha must equal preregistered 10.0")
    return RidgeLinearModel(
        intercept=_number(_required(params, "intercept"), "intercept", nonnegative=False),
        coefficients=tuple(vectors["coefficients"]),
        means=tuple(vectors["means"]),
        scales=tuple(vectors["scales"]),
        active=tuple(active),
        alpha=alpha,
    )


def _predict(model: ConstantModel | RidgeLinearModel, features: tuple) -> float:
    value = _number(model.predict(features), "model prediction", nonnegative=False)
    return max(0.0, value)


def _errors(rows: list[dict], predictions: dict[str, float]) -> dict:
    projects = {}
    residuals = []
    for row in rows:
        residual = row["rated_cost_usd"] - predictions[row["call_id"]]
        _number(residual, "residual", nonnegative=False)
        residuals.append(residual)
        projects.setdefault(row["project_id"], []).append(abs(residual))
    project_mae = {
        project: sum(error / len(errors) for error in errors)
        for project, errors in sorted(projects.items())
    }
    mae = sum(error / len(project_mae) for error in project_mae.values())
    _number(mae, "MAE")
    return {
        "mae": mae,
        "per_project_mae": project_mae,
        "call_weighted_mae": sum(abs(error) / len(residuals) for error in residuals),
        "residual_summary": {
            "residual_mae": mae,
            "mean_signed_residual": sum(error / len(residuals) for error in residuals),
            "min_signed_residual": min(residuals),
            "max_signed_residual": max(residuals),
            "interpretation": "exploratory_leave_one_project_out_residuals_not_coverage",
        },
    }


def fit_models(rows: list[dict]) -> dict:
    """Fit four frozen models using equal-project-weight leave-one-project-out MAE.

    Raises TypeError/ValueError for malformed completed measurements, holdout
    input, duplicate IDs, or fewer than three measured training projects.
    Noncompleted calls are excluded explicitly; quality-rejected completions
    remain. Ridge alpha is fixed at 10.0 and never selected using targets.
    Complete usage requires input_tokens, output_tokens, total_tokens,
    cached_tokens, and reasoning_tokens, including explicit measured zeros.
    """
    rows, excluded = _completed_rows(rows, training=True)
    projects = sorted({row["project_id"] for row in rows})
    if len(projects) < 3:
        raise ValueError("at least 3 completed training projects are required")
    values = [_quote_features(row["quote"]) for row in rows]
    forms = {}
    for form in _FEATURES:
        records = [
            Record(_features(features, form), row["rated_cost_usd"], row["project_id"])
            for row, features in zip(rows, values)
        ]
        predictions = {}
        folds = []
        for train, validation in grouped_kfold(records, n_splits=len(projects), seed=0):
            model = _fit([records[i] for i in train], form)
            for i in validation:
                predictions[rows[i]["call_id"]] = _predict(model, records[i].features)
            folds.append({
                "validation_project_id": rows[validation[0]]["project_id"],
                "training_project_ids": sorted({rows[i]["project_id"] for i in train}),
                "validation_call_ids": [rows[i]["call_id"] for i in validation],
            })
        metrics = _errors(rows, predictions)
        forms[form] = {
            "features": list(_FEATURES[form]),
            "model": _model_params(_fit(records, form)),
            "cv_mae": metrics["mae"],
            "per_project_mae": metrics["per_project_mae"],
            "cv_predictions": dict(sorted(predictions.items())),
            "cv_folds": sorted(folds, key=lambda fold: fold["validation_project_id"]),
            "residual_summary": metrics["residual_summary"],
        }
    selected = min(forms, key=lambda form: forms[form]["cv_mae"])
    return {
        "schema_version": _SCHEMA_VERSION,
        "selected_form": selected,
        "training_project_ids": projects,
        "training_call_ids": [row["call_id"] for row in rows],
        "excluded_call_ids": excluded,
        "feature_definitions": {
            "prompt_bytes": "Declared quote-time prompt byte count; no measured usage.",
            "context_bytes": "Validated quote metadata; not an additional model feature.",
            "planned_output_tokens": "Declared quote-time output allowance, not actual output.",
            "total_scoping_units": "Sum of extract, classify, plan, and report counts.",
            **{name: f"Declared quote-time {name} scoping-unit count." for name in _UNITS},
        },
        "forms": forms,
        "cv_mae": {form: result["cv_mae"] for form, result in forms.items()},
        "cv_method": "leave_one_project_out",
        "selection_metric": "equal_project_weight_mae_usd",
        "tie_break_order": list(_FEATURES),
        "ridge_alpha": _ALPHA,
        "prediction_floor_usd": 0.0,
        "residual_summary": forms[selected]["residual_summary"],
        "calibrated_interval": None,
        "calibration_status": _CALIBRATION_STATUS,
    }


def predict_cost(artifact: dict, quote: dict, form: str | None = None) -> float:
    """Predict a nonnegative USD cost using saved parameters, without fitting."""
    artifact = _mapping(artifact, "artifact")
    version = _required(artifact, "schema_version")
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema_version")
    selected = _identifier(_required(artifact, "selected_form"), "selected_form")
    if selected not in _FEATURES:
        raise ValueError("unknown selected_form")
    form = selected if form is None else _identifier(form, "form")
    if form not in _FEATURES:
        raise ValueError(f"unknown form: {form}")
    forms = _mapping(_required(artifact, "forms"), "forms")
    if set(forms) != set(_FEATURES):
        raise ValueError("artifact must contain all four model forms")
    result = _mapping(_required(forms, form), "form result")
    if _required(result, "features") != list(_FEATURES[form]):
        raise ValueError("artifact feature definitions do not match form")
    model = _model_from_params(_required(result, "model"), form)
    return _predict(model, _features(_quote_features(quote), form))


def evaluate_predictions(rows: list[dict], predictions: list[dict]) -> dict:
    """Score frozen predictions against completed targets with exact call IDs.

    Each prediction must provide the same nonempty subset of the four forms.
    MAE weights projects equally; call-weighted MAE is separately identified.
    This function neither fits models nor chooses a form using evaluation data.
    """
    rows, excluded = _completed_rows(rows, training=False)
    if not isinstance(predictions, list):
        raise TypeError("predictions must be a list")
    indexed = {}
    expected_forms = set()
    for value in predictions:
        prediction = _mapping(value, "prediction")
        call_id = _identifier(_required(prediction, "call_id"), "call_id")
        if call_id in indexed:
            raise ValueError(f"duplicate prediction call_id: {call_id}")
        forms = _mapping(_required(prediction, "predictions"), "predictions by form")
        if not forms or not set(forms).issubset(_FEATURES):
            raise ValueError("predictions must contain known model forms")
        if expected_forms and set(forms) != expected_forms:
            raise ValueError("all predictions must contain identical forms")
        expected_forms = set(forms)
        indexed[call_id] = {
            form: _number(value, f"prediction {form}") for form, value in forms.items()
        }
    target_ids = {row["call_id"] for row in rows}
    if set(indexed) != target_ids:
        raise ValueError("prediction call_ids must exactly match completed target call_ids")
    scores = {}
    for form in _FEATURES:
        if form in expected_forms:
            metrics = _errors(rows, {key: value[form] for key, value in indexed.items()})
            scores[form] = {
                name: metrics[name] for name in ("mae", "per_project_mae", "call_weighted_mae")
            }
    return {
        "n_calls": len(rows),
        "n_projects": len({row["project_id"] for row in rows}),
        "call_ids": sorted(target_ids),
        "excluded_call_ids": excluded,
        "metric": "equal_project_weight_mae_usd",
        "forms": scores,
        "mae": {form: metrics["mae"] for form, metrics in scores.items()},
    }
