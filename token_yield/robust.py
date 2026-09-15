"""Small, dependency-free utilities for robust grouped model evaluation.

The group is the unit of independence throughout this module.  In particular,
model selection happens again inside every outer cross-validation fold.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Record:
    """One observation with numeric features, a target, and an independent group."""

    features: tuple[float, ...]
    target: float
    group: Hashable
    factors: Mapping[str, Hashable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(float(x) for x in self.features))
        object.__setattr__(self, "target", float(self.target))
        if not self.features:
            raise ValueError("records must have at least one feature")
        if not all(math.isfinite(x) for x in self.features + (self.target,)):
            raise ValueError("features and target must be finite")
        try:
            hash(self.group)
        except TypeError as exc:
            raise ValueError("group IDs must be hashable") from exc


def _validate(records: Sequence[Record], *, min_groups: int = 1) -> int:
    if not records:
        raise ValueError("at least one record is required")
    width = len(records[0].features)
    if width < 1:
        raise ValueError("at least one feature is required")
    if any(len(r.features) != width for r in records):
        raise ValueError("all feature vectors must have the same length")
    groups = {r.group for r in records}
    if len(groups) < min_groups:
        raise ValueError(f"at least {min_groups} distinct groups are required")
    return width


@dataclass(frozen=True)
class ConstantModel:
    value: float

    @classmethod
    def fit(cls, records: Sequence[Record]) -> "ConstantModel":
        _validate(records)
        return cls(sum(r.target for r in records) / len(records))

    def predict(self, features: Sequence[float]) -> float:
        return self.value


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a nonsingular system using partial-pivoted Gauss-Jordan elimination."""
    n = len(rhs)
    augmented = [matrix[i][:] + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) <= 1e-12:
            raise ValueError("linear system is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [x / scale for x in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            scale = augmented[row][col]
            if scale:
                augmented[row] = [
                    a - scale * b
                    for a, b in zip(augmented[row], augmented[col])
                ]
    return [augmented[i][-1] for i in range(n)]


@dataclass(frozen=True)
class RidgeLinearModel:
    """Ridge regression on standardized, nonconstant features.

    ``coefficients`` are in standardized-feature units.  The intercept is never
    penalized.  Constant columns are retained in metadata but not fitted.
    """

    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    active: tuple[int, ...]
    alpha: float

    @classmethod
    def fit(cls, records: Sequence[Record], alpha: float = 1.0) -> "RidgeLinearModel":
        width = _validate(records)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("ridge alpha must be a finite nonnegative number")
        n = len(records)
        means = tuple(sum(r.features[j] for r in records) / n for j in range(width))
        variances = tuple(
            sum((r.features[j] - means[j]) ** 2 for r in records) / n
            for j in range(width)
        )
        scales = tuple(math.sqrt(v) for v in variances)
        active = tuple(j for j, scale in enumerate(scales) if scale > 1e-12)
        target_mean = sum(r.target for r in records) / n
        if not active:
            return cls(target_mean, (), means, scales, (), float(alpha))

        z = [[(r.features[j] - means[j]) / scales[j] for j in active] for r in records]
        centered_y = [r.target - target_mean for r in records]
        p = len(active)
        gram = [[sum(row[a] * row[b] for row in z) for b in range(p)] for a in range(p)]
        for j in range(p):
            gram[j][j] += alpha
        rhs = [sum(row[j] * y for row, y in zip(z, centered_y)) for j in range(p)]
        # alpha=0 and collinear columns need a deterministic least-squares fallback.
        try:
            coefficients = _solve(gram, rhs)
        except ValueError:
            if alpha == 0:
                for j in range(p):
                    gram[j][j] += 1e-12
                coefficients = _solve(gram, rhs)
            else:
                raise
        return cls(
            target_mean, tuple(coefficients), means, scales, active, float(alpha)
        )

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.means):
            raise ValueError(f"expected {len(self.means)} features, got {len(features)}")
        return self.intercept + sum(
            coefficient * (float(features[j]) - self.means[j]) / self.scales[j]
            for coefficient, j in zip(self.coefficients, self.active)
        )

    @property
    def raw_coefficients(self) -> tuple[float, ...]:
        out = [0.0] * len(self.means)
        for coefficient, j in zip(self.coefficients, self.active):
            out[j] = coefficient / self.scales[j]
        return tuple(out)

    @property
    def raw_intercept(self) -> float:
        return self.intercept - sum(
            coefficient * self.means[j] / self.scales[j]
            for coefficient, j in zip(self.coefficients, self.active)
        )


def grouped_kfold(
    records: Sequence[Record], n_splits: int = 5, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Return deterministic ``(train_indices, validation_indices)`` folds."""
    _validate(records, min_groups=2)
    groups = list(dict.fromkeys(r.group for r in records))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(groups):
        raise ValueError("n_splits cannot exceed the number of distinct groups")
    random.Random(seed).shuffle(groups)
    fold_groups = [set(groups[i::n_splits]) for i in range(n_splits)]
    result = []
    for held_out in fold_groups:
        validation = [i for i, r in enumerate(records) if r.group in held_out]
        train = [i for i, r in enumerate(records) if r.group not in held_out]
        result.append((train, validation))
    return result


grouped_kfold_splits = grouped_kfold


def _mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def _mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    errors = [abs(a - p) / abs(a) for a, p in zip(actual, predicted) if a != 0]
    return sum(errors) / len(errors) if errors else math.nan


@dataclass(frozen=True)
class CVMetrics:
    mae: float
    mape: float
    baseline_mae: float
    baseline_mape: float
    excess_over_baseline: float
    relative_improvement: float
    predictions: tuple[float, ...]
    baseline_predictions: tuple[float, ...]
    selected_alphas: tuple[float, ...] = ()


def _metrics(
    records: Sequence[Record],
    predictions: Sequence[float],
    baseline_predictions: Sequence[float],
    selected_alphas: Sequence[float] = (),
) -> CVMetrics:
    actual = [r.target for r in records]
    mae = _mae(actual, predictions)
    baseline_mae = _mae(actual, baseline_predictions)
    improvement = (
        (baseline_mae - mae) / baseline_mae if baseline_mae else 0.0
    )
    return CVMetrics(
        mae, _mape(actual, predictions), baseline_mae,
        _mape(actual, baseline_predictions), mae - baseline_mae, improvement,
        tuple(predictions), tuple(baseline_predictions), tuple(selected_alphas),
    )


def grouped_cv_metrics(
    records: Sequence[Record],
    alpha: float = 1.0,
    n_splits: int = 5,
    seed: int = 0,
) -> CVMetrics:
    predictions = [math.nan] * len(records)
    baselines = [math.nan] * len(records)
    for train_indices, validation_indices in grouped_kfold(records, n_splits, seed):
        train = [records[i] for i in train_indices]
        model = RidgeLinearModel.fit(train, alpha)
        baseline = ConstantModel.fit(train)
        for i in validation_indices:
            predictions[i] = model.predict(records[i].features)
            baselines[i] = baseline.predict(records[i].features)
    return _metrics(records, predictions, baselines)


cross_validate = grouped_cv_metrics


def select_ridge_alpha(
    records: Sequence[Record],
    alphas: Sequence[float],
    n_splits: int = 3,
    seed: int = 0,
) -> float:
    """Select alpha by grouped CV, resolving ties toward stronger regularization."""
    _validate(records, min_groups=2)
    candidates = sorted({float(a) for a in alphas})
    if not candidates:
        raise ValueError("at least one ridge alpha is required")
    if any(not math.isfinite(a) or a < 0 for a in candidates):
        raise ValueError("ridge alphas must be finite and nonnegative")
    scores = [(grouped_cv_metrics(records, a, n_splits, seed).mae, -a, a)
              for a in candidates]
    return min(scores)[2]


def nested_grouped_cv(
    records: Sequence[Record],
    alphas: Sequence[float],
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 0,
) -> CVMetrics:
    """Grouped outer CV with ridge selection confined to each training fold."""
    predictions = [math.nan] * len(records)
    baselines = [math.nan] * len(records)
    chosen: list[float] = []
    for fold, (train_indices, validation_indices) in enumerate(
        grouped_kfold(records, outer_splits, seed)
    ):
        train = [records[i] for i in train_indices]
        train_groups = len({r.group for r in train})
        if train_groups < 2:
            raise ValueError("outer training folds need at least two groups")
        effective_inner = min(inner_splits, train_groups)
        if effective_inner < 2:
            raise ValueError("inner_splits must be at least 2")
        alpha = select_ridge_alpha(train, alphas, effective_inner, seed + fold + 1)
        chosen.append(alpha)
        model, baseline = RidgeLinearModel.fit(train, alpha), ConstantModel.fit(train)
        for i in validation_indices:
            predictions[i] = model.predict(records[i].features)
            baselines[i] = baseline.predict(records[i].features)
    return _metrics(records, predictions, baselines, chosen)


def quantile(values: Iterable[float], probability: float) -> float:
    """Linearly interpolated empirical quantile (the common type-7 definition)."""
    data = sorted(float(value) for value in values)
    if not data:
        raise ValueError("quantile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if any(not math.isfinite(value) for value in data):
        raise ValueError("quantile values must be finite")
    position = (len(data) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return data[lower]
    return data[lower] + (position - lower) * (data[upper] - data[lower])


def empirical_interval_coverage(
    actual: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    tail_probability: Optional[float] = None,
) -> float:
    """Fraction covered, optionally restricted to both empirical target tails."""
    if not (len(actual) == len(lower) == len(upper)) or not actual:
        raise ValueError("actual, lower, and upper must be nonempty and equal length")
    indices = list(range(len(actual)))
    if tail_probability is not None:
        if not 0 < tail_probability <= 0.5:
            raise ValueError("tail_probability must be in (0, 0.5]")
        low_cut = quantile(actual, tail_probability)
        high_cut = quantile(actual, 1 - tail_probability)
        indices = [i for i, value in enumerate(actual)
                   if value <= low_cut or value >= high_cut]
    if any(lower[i] > upper[i] for i in indices):
        raise ValueError("interval lower bounds cannot exceed upper bounds")
    return sum(lower[i] <= actual[i] <= upper[i] for i in indices) / len(indices)


def grouped_bootstrap_improvement_ci(
    records: Sequence[Record],
    predictions: Optional[Sequence[float]] = None,
    baseline_predictions: Optional[Sequence[float]] = None,
    *,
    alphas: Sequence[float] = (0.0, 0.1, 1.0, 10.0),
    n_splits: int = 5,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile grouped-bootstrap CI for MAE improvement over a constant.

    If predictions are omitted, leakage-free nested grouped-CV predictions are
    generated first.  Bootstrap resampling then treats whole groups as units.
    """
    _validate(records, min_groups=2)
    if predictions is None or baseline_predictions is None:
        if predictions is not None or baseline_predictions is not None:
            raise ValueError("provide both prediction sequences or neither")
        result = nested_grouped_cv(records, alphas, n_splits, min(3, n_splits), seed)
        predictions, baseline_predictions = result.predictions, result.baseline_predictions
    if len(predictions) != len(records) or len(baseline_predictions) != len(records):
        raise ValueError("prediction sequences must match records")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    by_group: dict[Hashable, list[int]] = {}
    for i, record in enumerate(records):
        by_group.setdefault(record.group, []).append(i)
    groups = list(by_group)
    rng = random.Random(seed)
    improvements = []
    for _ in range(n_bootstrap):
        sampled = [rng.choice(groups) for _ in groups]
        indices = [i for group in sampled for i in by_group[group]]
        model_error = sum(abs(records[i].target - predictions[i]) for i in indices)
        base_error = sum(abs(records[i].target - baseline_predictions[i]) for i in indices)
        improvements.append((base_error - model_error) / base_error if base_error else 0.0)
    tail = (1 - confidence) / 2
    return quantile(improvements, tail), quantile(improvements, 1 - tail)


grouped_bootstrap_ci = grouped_bootstrap_improvement_ci


def _matrix_rank(matrix: Sequence[Sequence[float]], tolerance: float = 1e-10) -> int:
    work = [list(map(float, row)) for row in matrix]
    if not work:
        return 0
    rows, columns, rank = len(work), len(work[0]), 0
    for column in range(columns):
        pivot = max(range(rank, rows), key=lambda row: abs(work[row][column]),
                    default=rank)
        if rank >= rows or abs(work[pivot][column]) <= tolerance:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for row in range(rank + 1, rows):
            ratio = work[row][column] / work[rank][column]
            for j in range(column, columns):
                work[row][j] -= ratio * work[rank][j]
        rank += 1
        if rank == rows:
            break
    return rank


@dataclass(frozen=True)
class Diagnostics:
    feature_variance: tuple[float, ...]
    matrix_rank: int
    design_columns: int
    collinear: bool
    condition_number: float
    residual_by_factor: dict[str, dict[Hashable, dict[str, float]]]
    influential_group_deltas: dict[Hashable, float]


def diagnostics(
    records: Sequence[Record],
    model: Optional[RidgeLinearModel] = None,
    *,
    alpha: float = 1.0,
) -> Diagnostics:
    """Return compact variance, collinearity, underfit, and influence checks."""
    width = _validate(records, min_groups=2)
    model = model or RidgeLinearModel.fit(records, alpha)
    n = len(records)
    means = [sum(r.features[j] for r in records) / n for j in range(width)]
    variances = tuple(
        sum((r.features[j] - means[j]) ** 2 for r in records) / n
        for j in range(width)
    )
    active = [j for j, variance in enumerate(variances) if variance > 1e-24]
    design = [[1.0] + [r.features[j] for j in active] for r in records]
    rank = _matrix_rank(design)
    standardized = [
        [1.0] + [
            (r.features[j] - means[j]) / math.sqrt(variances[j])
            for j in active
        ]
        for r in records
    ]
    gram = [
        [sum(row[i] * row[j] for row in standardized)
         for j in range(len(standardized[0]))]
        for i in range(len(standardized[0]))
    ]
    eigenvalues = _symmetric_eigenvalues(gram)
    positive = [value for value in eigenvalues if value > 1e-10]
    condition = (
        math.sqrt(max(positive) / min(positive))
        if len(positive) == len(gram) else math.inf
    )
    residuals = [r.target - model.predict(r.features) for r in records]

    buckets: dict[str, dict[Hashable, list[float]]] = {}
    for record, residual in zip(records, residuals):
        for factor, level in record.factors.items():
            buckets.setdefault(factor, {}).setdefault(level, []).append(residual)
    factor_summary = {
        factor: {
            level: {
                "count": float(len(values)),
                "mean_residual": sum(values) / len(values),
                "mae": sum(abs(value) for value in values) / len(values),
            }
            for level, values in levels.items()
        }
        for factor, levels in buckets.items()
    }

    full_mae = _mae([r.target for r in records],
                     [model.predict(r.features) for r in records])
    influence = {}
    for group in dict.fromkeys(r.group for r in records):
        remaining = [r for r in records if r.group != group]
        if not remaining:
            continue
        leave_one_out = RidgeLinearModel.fit(remaining, alpha)
        loo_mae = _mae(
            [r.target for r in records],
            [leave_one_out.predict(r.features) for r in records],
        )
        influence[group] = loo_mae - full_mae
    return Diagnostics(
        variances, rank, len(active) + 1, rank < len(active) + 1, condition,
        factor_summary, influence,
    )


def _symmetric_eigenvalues(
    matrix: Sequence[Sequence[float]], tolerance: float = 1e-12
) -> list[float]:
    """Jacobi eigenvalues for the small symmetric Gram matrices used here."""

    values = [list(map(float, row)) for row in matrix]
    n = len(values)
    for _ in range(max(1, 50 * n * n)):
        p, q, maximum = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(values[i][j]) > maximum:
                    p, q, maximum = i, j, abs(values[i][j])
        if maximum <= tolerance:
            break
        angle = 0.5 * math.atan2(
            2 * values[p][q], values[q][q] - values[p][p]
        )
        cosine, sine = math.cos(angle), math.sin(angle)
        for i in range(n):
            if i in (p, q):
                continue
            left, right = values[i][p], values[i][q]
            values[i][p] = values[p][i] = cosine * left - sine * right
            values[i][q] = values[q][i] = sine * left + cosine * right
        app, aqq, apq = values[p][p], values[q][q], values[p][q]
        values[p][p] = (
            cosine * cosine * app - 2 * sine * cosine * apq
            + sine * sine * aqq
        )
        values[q][q] = (
            sine * sine * app + 2 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        values[p][q] = values[q][p] = 0.0
    return [values[i][i] for i in range(n)]


def grouped_bootstrap_coefficient_ci(
    records: Sequence[Record],
    alpha: float,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Grouped-bootstrap intervals for raw intercept and coefficients."""

    width = _validate(records, min_groups=2)
    if not 0 < confidence < 1 or n_bootstrap < 1:
        raise ValueError("invalid bootstrap confidence or iteration count")
    by_group: dict[Hashable, list[Record]] = {}
    for record in records:
        by_group.setdefault(record.group, []).append(record)
    groups = list(by_group)
    rng = random.Random(seed)
    draws = [[] for _ in range(width + 1)]
    for _ in range(n_bootstrap):
        sampled = [rng.choice(groups) for _ in groups]
        rows = [record for group in sampled for record in by_group[group]]
        model = RidgeLinearModel.fit(rows, alpha)
        values = (model.raw_intercept, *model.raw_coefficients)
        for bucket, value in zip(draws, values):
            bucket.append(value)
    tail = (1 - confidence) / 2
    return tuple(
        (quantile(values, tail), quantile(values, 1 - tail))
        for values in draws
    )


RobustRecord = Record
RidgeModel = RidgeLinearModel
group_kfold = grouped_kfold
nested_cv = nested_grouped_cv
bootstrap_relative_improvement = grouped_bootstrap_improvement_ci
model_diagnostics = diagnostics
interval_coverage = empirical_interval_coverage


def fit_constant(records: Sequence[Record]) -> ConstantModel:
    return ConstantModel.fit(records)


def fit_ridge(records: Sequence[Record], alpha: float = 1.0) -> RidgeLinearModel:
    return RidgeLinearModel.fit(records, alpha)
