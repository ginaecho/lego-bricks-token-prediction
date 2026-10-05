"""Feature engineering and training through main's real training functions.

Everything here calls ``examples.train_construction_model`` (``load``,
``feature_names``, ``matrix``, ``grouped_cv``, ``fit_predict``,
``interval_bounds``). Nothing writes model files; the stored wave-3 report
supplies the once-evaluated test result, which is never re-run here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from examples import train_construction_model as tcm

WAVE = "wave3"


class Trainer:
    def __init__(self, root: Path):
        corpus = root / "data" / "build_simulations"
        self.points = tcm.load(corpus)
        self.names = tcm.feature_names(self.points)
        rules = json.loads((corpus / "waves" / f"{WAVE}.json").read_text(encoding="utf-8"))["evaluation_rules"]
        self.gates = rules["acceptance_gates"]
        self.split = {key: [p for p in self.points if p["split"] == key] for key in ("train", "validation", "test")}
        self.development = self.split["train"] + self.split["validation"]
        self.stored_report = json.loads((corpus / "models" / f"construction_model_{WAVE}_report.json")
                                        .read_text(encoding="utf-8"))
        self._fit()

    def _fit(self) -> None:
        x_dev, y_dev = tcm.matrix(self.development, self.names), tcm.targets(self.development)
        groups = np.array([p["split_group"] for p in self.development])
        folds = min(5, len(set(groups)))
        total = y_dev["input_tokens"] + y_dev["output_tokens"]
        self.search = []
        for alpha in tcm.ALPHAS:
            totals, _ = tcm.grouped_cv(x_dev, y_dev, groups, alpha, folds)
            self.search.append({"alpha": alpha, **tcm.metrics(total, totals)})
        self.alpha = min(self.search, key=lambda row: row["mae"])["alpha"]
        _, residuals = tcm.grouped_cv(x_dev, y_dev, groups, self.alpha, folds)
        self.bounds = tcm.interval_bounds(residuals, self.gates["interval_coverage_target"])
        self.validation = self._validate()
        self.final, _ = tcm.fit_predict(x_dev, y_dev, x_dev[:1], self.alpha)

    def _validate(self) -> dict:
        train, held = self.split["train"], self.split["validation"]
        x_train, y_train = tcm.matrix(train, self.names), tcm.targets(train)
        x_val, y_val = tcm.matrix(held, self.names), tcm.targets(held)
        _, predicted = tcm.fit_predict(x_train, y_train, x_val, self.alpha)
        actual = y_val["input_tokens"] + y_val["output_tokens"]
        estimate = predicted["input_tokens"] + predicted["output_tokens"]
        baseline = np.full(len(actual), float(np.mean(y_train["input_tokens"] + y_train["output_tokens"])))
        candidate, mean = tcm.metrics(actual, estimate), tcm.metrics(actual, baseline)
        return {
            "candidate": candidate, "training_mean_baseline": mean,
            "passed": candidate["mape"] <= self.gates["validation_total_mape_max"] and candidate["mae"] < mean["mae"],
            "points": [{"id": p["id"], "actual": float(a), "predicted": float(e)}
                       for p, a, e in zip(held, actual, estimate)],
        }

    def vector(self, features: dict) -> list[dict]:
        """The engineered numeric input row exactly as the trainer sees it."""
        row = tcm.matrix([{"input_features": features}], self.names)[0]
        return [{"name": name, "value": float(value)} for name, value in zip(self.names, row)]

    def predict(self, features: dict, models: dict | None = None) -> dict:
        models = models or self.final
        x = tcm.matrix([{"input_features": features}], self.names)
        per_target = {name: float(np.exp(models[name].predict(x))[0]) for name in tcm.TARGETS}
        total = sum(per_target.values())
        low = sum(per_target[n] * math.e ** self.bounds[n][0] for n in tcm.TARGETS)
        high = sum(per_target[n] * math.e ** self.bounds[n][1] for n in tcm.TARGETS)
        return {**per_target, "total_tokens": total, "interval": [low, high],
                "interval_level": self.gates["interval_coverage_target"]}

    def contributions(self, features: dict, target: str = "input_tokens", top: int = 10) -> list[dict]:
        """Log-space contribution of each standardized feature to one target."""
        scaler, ridge = self.final[target][0], self.final[target][1]
        x = tcm.matrix([{"input_features": features}], self.names)[0]
        terms = ridge.coef_ * (x - scaler.mean_) / scaler.scale_
        ranked = sorted(zip(self.names, terms, x), key=lambda item: -abs(item[1]))[:top]
        return [{"name": n, "log_effect": float(t), "multiplier": float(math.exp(t)), "value": float(v)}
                for n, t, v in ranked]

    def without_group(self, split_group: str, features: dict) -> dict | None:
        """Prediction from a model refitted without this membership: the forecast before the data point existed."""
        kept = [p for p in self.development if p["split_group"] != split_group]
        if len(kept) == len(self.development):
            return None
        fitted, _ = tcm.fit_predict(tcm.matrix(kept, self.names), tcm.targets(kept),
                                    tcm.matrix(kept[:1], self.names), self.alpha)
        return {**self.predict(features, fitted), "trained_on": len(kept)}

    def summary(self) -> dict:
        stored = self.stored_report
        return {
            "feature_names": self.names, "excluded": ["staff_* (declared FTE)", "estimated_months_to_finish"],
            "counts": {key: len(value) for key, value in self.split.items()},
            "alpha_search": self.search, "selected_alpha": self.alpha, "gates": self.gates,
            "validation": self.validation,
            "stored_test": stored.get("test"), "stored_fingerprint": stored.get("model_fingerprint"),
            "stored_alpha": stored.get("selected_alpha"),
        }
