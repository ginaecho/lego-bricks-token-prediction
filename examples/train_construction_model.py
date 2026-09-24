"""Fit and evaluate the construction-token predictor on measured build points.

Targets are input and output construction tokens, fitted separately in log space.
Only pre-build input features are used. Test groups are evaluated once per frozen
candidate fingerprint; re-running cannot silently redefine the test result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from token_yield.marketplace_agent_contracts import fingerprint

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "build_simulations"
EXCLUDED = ("staff_", "estimated_months_to_finish")
TARGETS = ("input_tokens", "output_tokens")
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def load(root: Path) -> list[dict]:
    points = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "data_points").glob("*.json"))]
    return [point for point in points if point["status"] == "measured_build_passed"
            and point["origin"] == "catalog_designed_simulation"]


def feature_names(points: list[dict]) -> list[str]:
    """Union over all points; wave-3 features (generic parts, industry) are 0 for earlier records."""
    names = {name for point in points for name, value in point["input_features"].items()
             if isinstance(value, (int, float)) and not isinstance(value, bool)
             and not name.startswith(EXCLUDED[0]) and name != EXCLUDED[1]}
    return sorted(names)


def matrix(points: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[float(point["input_features"].get(name, 0)) for name in names] for point in points])


def level(point: dict) -> str:
    return point["input_features"].get("composition_level", "BT")


def breakdown(points: list[dict], actual, predicted, key) -> dict:
    labels = [key(point) for point in points]
    return {str(label): metrics(actual[mask], predicted[mask])
            for label in sorted(set(labels), key=str)
            for mask in [np.array([item == label for item in labels])]}


def targets(points: list[dict]) -> dict[str, np.ndarray]:
    return {name: np.array([point["build_token_usage"][name] for point in points], dtype=float) for name in TARGETS}


def model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {"n": int(len(actual)), "mae": float(np.mean(np.abs(error))),
            "mape": float(np.mean(np.abs(error) / actual)), "bias": float(np.mean(error))}


def grouped_cv(x, ys, groups, alpha, folds):
    """Returns out-of-fold log residuals and predictions for the total-token target."""
    splitter = GroupKFold(n_splits=folds)
    totals = np.zeros(len(groups))
    residuals = {name: np.zeros(len(groups)) for name in TARGETS}
    for train, held in splitter.split(x, groups=groups):
        total = np.zeros(len(held))
        for name in TARGETS:
            fitted = model(alpha).fit(x[train], np.log(ys[name][train]))
            predicted = fitted.predict(x[held])
            residuals[name][held] = np.log(ys[name][held]) - predicted
            total += np.exp(predicted)
        totals[held] = total
    return totals, residuals


def fit_predict(x_train, ys_train, x_eval, alpha):
    fitted = {name: model(alpha).fit(x_train, np.log(ys_train[name])) for name in TARGETS}
    predictions = {name: np.exp(fitted[name].predict(x_eval)) for name in TARGETS}
    return fitted, predictions


def interval_bounds(residuals: dict, level: float) -> dict:
    tail = (1 - level) / 2
    return {name: [float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))]
            for name, values in residuals.items()}


def coverage(prediction_total, actual_total, predictions, bounds):
    low = sum(predictions[name] * math.e ** bounds[name][0] for name in TARGETS)
    high = sum(predictions[name] * math.e ** bounds[name][1] for name in TARGETS)
    return float(np.mean((actual_total >= low) & (actual_total <= high)))


def run(root: Path, wave: str, evaluate_test: bool) -> dict:
    gates = json.loads((root / "waves" / f"{wave}.json").read_text(encoding="utf-8"))["evaluation_rules"]
    level = gates["acceptance_gates"]["interval_coverage_target"]
    points = load(root)
    names = feature_names(points)
    split = {key: [point for point in points if point["split"] == key] for key in ("train", "validation", "test")}
    development = split["train"] + split["validation"]
    x_dev, y_dev = matrix(development, names), targets(development)
    groups = np.array([point["split_group"] for point in development])
    folds = min(5, len(set(groups)))
    total_dev = y_dev["input_tokens"] + y_dev["output_tokens"]
    search = []
    for alpha in ALPHAS:
        totals, _ = grouped_cv(x_dev, y_dev, groups, alpha, folds)
        search.append({"alpha": alpha, **metrics(total_dev, totals)})
    best = min(search, key=lambda row: row["mae"])["alpha"]
    _, cv_residuals = grouped_cv(x_dev, y_dev, groups, best, folds)
    bounds = interval_bounds(cv_residuals, level)

    x_train, y_train = matrix(split["train"], names), targets(split["train"])
    x_val, y_val = matrix(split["validation"], names), targets(split["validation"])
    _, val_pred = fit_predict(x_train, y_train, x_val, best)
    val_total = y_val["input_tokens"] + y_val["output_tokens"]
    val_pred_total = val_pred["input_tokens"] + val_pred["output_tokens"]
    baseline_total = np.full(len(val_total), float(np.mean(y_train["input_tokens"] + y_train["output_tokens"])))
    _, train_fit = fit_predict(x_train, y_train, x_train, best)
    train_total = y_train["input_tokens"] + y_train["output_tokens"]
    validation = {
        "candidate": metrics(val_total, val_pred_total),
        "training_mean_baseline": metrics(val_total, baseline_total),
        "per_target": {name: metrics(y_val[name], val_pred[name]) for name in TARGETS},
        "by_level": breakdown(split["validation"], val_total, val_pred_total, level),
        "train_fit": metrics(train_total, train_fit["input_tokens"] + train_fit["output_tokens"]),
        "interval_coverage": coverage(val_pred_total, val_total, val_pred, bounds),
    }
    rules = gates["acceptance_gates"]
    passed = (validation["candidate"]["mape"] <= rules["validation_total_mape_max"]
              and validation["candidate"]["mae"] < validation["training_mean_baseline"]["mae"])

    final, _ = fit_predict(x_dev, y_dev, x_dev[:1], best)
    artifact = {
        "model_family": "ridge_log_tokens_v1", "alpha": best, "feature_names": names,
        "interval_level": level, "log_residual_bounds": bounds,
        "trained_on": sorted(point["id"] for point in development),
        "targets": {name: {"scaler_mean": final[name][0].mean_.tolist(), "scaler_scale": final[name][0].scale_.tolist(),
                           "coef": final[name][1].coef_.tolist(), "intercept": float(final[name][1].intercept_)}
                    for name in TARGETS},
    }
    artifact["fingerprint"] = fingerprint(artifact)
    out = root / "models"
    out.mkdir(exist_ok=True)
    report_path = out / f"construction_model_{wave}_report.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    test = previous.get("test") if previous.get("model_fingerprint") == artifact["fingerprint"] else None
    if evaluate_test and test is None and split["test"]:
        if previous.get("test"):
            raise ValueError("The test set was already evaluated for another candidate; do not re-select on test")
        x_test, y_test = matrix(split["test"], names), targets(split["test"])
        predictions = {name: np.exp(final[name].predict(x_test)) for name in TARGETS}
        actual = y_test["input_tokens"] + y_test["output_tokens"]
        predicted = predictions["input_tokens"] + predictions["output_tokens"]
        test = {"evaluated_once": True, **metrics(actual, predicted),
                "interval_coverage": coverage(predicted, actual, predictions, bounds),
                "by_size": breakdown(split["test"], actual, predicted, lambda p: p["input_features"]["functionality_count"]),
                "by_level": breakdown(split["test"], actual, predicted, level),
                "by_industry": breakdown(split["test"], actual, predicted, lambda p: p["input_features"].get("industry") or "none"),
                "by_wave": breakdown(split["test"], actual, predicted, lambda p: p.get("wave") or "wave1")}
    report = {
        "wave": wave, "model_fingerprint": artifact["fingerprint"], "promoted": passed,
        "counts": {key: len(value) for key, value in split.items()},
        "counts_by_level": {key: dict(Counter(level(p) for p in value)) for key, value in split.items()},
        "groups": {key: len({p["split_group"] for p in value}) for key, value in split.items()},
        "alpha_search_grouped_cv_total_tokens": search, "selected_alpha": best,
        "validation": validation, "gates": rules, "gates_passed_on_validation": passed,
        "test": test,
        "limitations": [
            "Bounded Python CLI builds by one builder model; not enterprise delivery.",
            "Staffing and months are excluded: they are deterministic declared assumptions, not observed drivers.",
            "Unseen type/order interactions beyond the measured memberships are extrapolations.",
        ],
    }
    (out / f"construction_model_{wave}.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CORPUS)
    parser.add_argument("--wave", default="wave2")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    report = run(args.root, args.wave, args.evaluate_test)
    print(json.dumps({key: report[key] for key in ("counts", "selected_alpha", "gates_passed_on_validation", "test")}
                     | {"validation": report["validation"]["candidate"],
                        "baseline": report["validation"]["training_mean_baseline"]}, indent=1))


if __name__ == "__main__":
    main()
