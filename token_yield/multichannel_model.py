"""Small-sample, grouped forecasts for separately measured usage channels."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Hashable, Mapping, Optional, Sequence, Tuple

from .robust import (
    ConstantModel,
    Record,
    RidgeLinearModel,
    grouped_kfold,
    quantile,
    select_ridge_alpha,
)
from .economics import Pricing
from .wave3_features import REJECTED_PREDICTORS


IDENTIFIED = "IDENTIFIED"
UNIDENTIFIED = "UNIDENTIFIED"
CENSORED = "CENSORED"
UNAVAILABLE = "UNAVAILABLE"
CALIBRATED = "CALIBRATED"
EXPLORATORY = "EXPLORATORY"

_POST_RUN_FEATURES = frozenset({
    "cached_tokens", "cache_hit", "observed_cache_hit",
    "output_tokens", "non_reasoning_output_tokens", "reasoning_tokens",
    "reasoning_occurred", "branch_or_retry", "accepted_work_cost",
    "input_tokens", "total_tokens",
}) | frozenset(REJECTED_PREDICTORS)


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


@dataclass(frozen=True)
class ChannelObservation:
    """One group-labelled observation from an attempt ledger.

    Censored output rows are retained for non-output channels but never used
    as exact output or reasoning magnitude observations.  Accepted-work
    economics requires both ``attempt_cost`` and ``accepted`` for every row.
    """

    group: Hashable
    cached_tokens: float
    non_reasoning_output_tokens: float
    reasoning_tokens: float
    branch_or_retry: bool
    output_bound_tokens: float
    output_censored: bool = False
    attempt_cost: Optional[float] = None
    accepted: Optional[bool] = None

    def __post_init__(self) -> None:
        try:
            hash(self.group)
        except TypeError as exc:
            raise ValueError("group IDs must be hashable") from exc
        for name in (
            "cached_tokens", "non_reasoning_output_tokens", "reasoning_tokens",
            "output_bound_tokens",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.non_reasoning_output_tokens + self.reasoning_tokens > self.output_bound_tokens:
            raise ValueError("observed output tokens exceed output_bound_tokens")
        if not isinstance(self.branch_or_retry, bool):
            raise ValueError("branch_or_retry must be boolean")
        if not isinstance(self.output_censored, bool):
            raise ValueError("output_censored must be boolean")
        if self.attempt_cost is not None:
            object.__setattr__(self, "attempt_cost", _number(self.attempt_cost, "attempt_cost"))
        if self.accepted is not None and not isinstance(self.accepted, bool):
            raise ValueError("accepted must be boolean or None")


@dataclass(frozen=True)
class ChannelComparison:
    """One channel's grouped out-of-fold comparison and identification state."""

    status: str
    model_mae: Optional[float]
    baseline_mae: Optional[float]
    residual_count: int
    n_groups: int
    attainable_conformal_level: Optional[float]
    censored_rows: int
    censored_fraction: float


@dataclass(frozen=True)
class ForecastPoint:
    """One bounded token and cost reconstruction at a requested percentile."""

    input_tokens: float
    cached_tokens: Optional[float]
    non_reasoning_output_tokens: Optional[float]
    reasoning_tokens: Optional[float]
    output_tokens: Optional[float]
    total_tokens: Optional[float]
    cost: Optional[float]
    accepted_work_cost: Optional[float]
    cap_binding: Optional[bool]
    calibration_status: str


@dataclass(frozen=True)
class MultiChannelForecast:
    """p50/p90/p95 forecasts with identification and hurdle-tail context."""

    cache_hit_probability: Optional[float]
    reasoning_probability: Optional[float]
    branch_or_retry_probability: Optional[float]
    cache_conditional_positive_p95: Optional[float]
    reasoning_conditional_positive_p95: Optional[float]
    channel_status: Mapping[str, str]
    promotion_eligible: bool
    p50: ForecastPoint
    p90: ForecastPoint
    p95: ForecastPoint


@dataclass(frozen=True)
class _CalibratedChannel:
    model: Optional[object]
    residuals: Tuple[float, ...]
    comparison: ChannelComparison

    def center(self, features: Sequence[float]) -> Optional[float]:
        if self.model is None:
            return None
        return max(0.0, float(self.model.predict(features)))

    def at(self, features: Sequence[float], probability: float) -> Optional[float]:
        center = self.center(features)
        if center is None:
            return None
        return max(0.0, center + quantile(self.residuals, probability))


@dataclass(frozen=True)
class _RidgeLogisticModel:
    intercept: float
    coefficients: Tuple[float, ...]
    means: Tuple[float, ...]
    scales: Tuple[float, ...]
    active: Tuple[int, ...]

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            probability = 1.0 / (1.0 + math.exp(-min(value, 700.0)))
        else:
            exp_value = math.exp(max(value, -700.0))
            probability = exp_value / (1.0 + exp_value)
        return min(1.0 - 1e-9, max(1e-9, probability))

    @classmethod
    def fit(
        cls, records: Sequence[Record], alpha: float,
    ) -> "_RidgeLogisticModel":
        width = len(records[0].features)
        means = tuple(
            sum(record.features[index] for record in records) / len(records)
            for index in range(width)
        )
        scales = tuple(
            math.sqrt(
                sum(
                    (record.features[index] - means[index]) ** 2
                    for record in records
                ) / len(records)
            )
            for index in range(width)
        )
        active = tuple(index for index, scale in enumerate(scales) if scale > 1e-12)
        design = [
            tuple(
                (record.features[index] - means[index]) / scales[index]
                for index in active
            )
            for record in records
        ]
        prevalence = (sum(record.target for record in records) + 0.5) / (
            len(records) + 1.0
        )
        intercept = math.log(prevalence / (1.0 - prevalence))
        coefficients = [0.0] * len(active)
        max_norm = max(1.0 + sum(value * value for value in row) for row in design)
        step = 1.0 / (0.25 * max_norm + alpha / len(records) + 1e-12)
        for _ in range(10_000):
            probabilities = [
                cls._sigmoid(
                    intercept + sum(c * value for c, value in zip(coefficients, row))
                )
                for row in design
            ]
            intercept_gradient = sum(
                probability - record.target
                for probability, record in zip(probabilities, records)
            ) / len(records)
            coefficient_gradients = [
                (
                    sum(
                        (probability - record.target) * row[index]
                        for probability, record, row
                        in zip(probabilities, records, design)
                    )
                    + alpha * coefficients[index]
                ) / len(records)
                for index in range(len(active))
            ]
            deltas = [step * intercept_gradient] + [
                step * gradient for gradient in coefficient_gradients
            ]
            intercept -= deltas[0]
            coefficients = [
                coefficient - delta
                for coefficient, delta in zip(coefficients, deltas[1:])
            ]
            if max(abs(delta) for delta in deltas) < 1e-10:
                break
        return cls(intercept, tuple(coefficients), means, scales, active)

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.means):
            raise ValueError(f"expected {len(self.means)} features, got {len(features)}")
        score = self.intercept + sum(
            coefficient * (float(features[index]) - self.means[index])
            / self.scales[index]
            for coefficient, index in zip(self.coefficients, self.active)
        )
        return self._sigmoid(score)


def _unidentified(
    status: str,
    n_groups: int = 0,
    censored_rows: int = 0,
    total_rows: int = 0,
) -> _CalibratedChannel:
    return _CalibratedChannel(
        None,
        (),
        ChannelComparison(
            status, None, None, 0, n_groups, None, censored_rows,
            censored_rows / total_rows if total_rows else 0.0,
        ),
    )


def _validate_rows(
    feature_rows: Sequence[Sequence[float]],
    observations: Sequence[ChannelObservation],
) -> Tuple[Tuple[float, ...], ...]:
    if len(feature_rows) != len(observations) or not feature_rows:
        raise ValueError("feature rows and observations must be nonempty and aligned")
    try:
        rows = tuple(tuple(float(value) for value in row) for row in feature_rows)
    except (TypeError, ValueError) as exc:
        raise ValueError("quote-time features must be finite numbers") from exc
    if not rows[0]:
        raise ValueError("feature rows must contain at least one value")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("all feature rows must have the same width")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("quote-time features must be finite numbers")
    if len({observation.group for observation in observations}) < 3:
        raise ValueError("at least three distinct groups are required for grouped calibration")
    return rows


def _group_means(values: Sequence[float], groups: Sequence[Hashable]) -> Tuple[float, ...]:
    buckets = {}
    for value, group in zip(values, groups):
        buckets.setdefault(group, []).append(value)
    return tuple(sum(bucket) / len(bucket) for bucket in buckets.values())


def _group_mae(
    actual: Sequence[float],
    predicted: Sequence[float],
    groups: Sequence[Hashable],
) -> float:
    return sum(_group_means(
        [abs(left - right) for left, right in zip(actual, predicted)], groups
    )) / len(set(groups))


def _fit_identified(
    rows: Sequence[Tuple[float, ...]],
    groups: Sequence[Hashable],
    targets: Sequence[float],
    *,
    alphas: Sequence[float],
    seed: int,
) -> _CalibratedChannel:
    records = [
        Record(row, target, group)
        for row, target, group in zip(rows, targets, groups)
    ]
    n_groups = len(set(groups))
    folds = min(5, n_groups)
    alpha = select_ridge_alpha(records, alphas, n_splits=folds, seed=seed)
    default_alpha = sorted(float(candidate) for candidate in alphas)[len(alphas) // 2]
    predictions = [0.0] * len(records)
    baseline_predictions = [0.0] * len(records)
    for fold, (train_indices, validation_indices) in enumerate(
        grouped_kfold(records, folds, seed)
    ):
        train = [records[index] for index in train_indices]
        train_groups = len({record.group for record in train})
        fold_alpha = (
            select_ridge_alpha(
                train, alphas, n_splits=min(5, train_groups), seed=seed + fold + 1
            )
            if train_groups >= 2 else default_alpha
        )
        model = RidgeLinearModel.fit(train, fold_alpha)
        baseline = ConstantModel.fit(train)
        for index in validation_indices:
            predictions[index] = model.predict(records[index].features)
            baseline_predictions[index] = baseline.predict(records[index].features)
    residuals = _group_means(
        [target - prediction for target, prediction in zip(targets, predictions)],
        groups,
    )
    return _CalibratedChannel(
        RidgeLinearModel.fit(records, alpha),
        residuals,
        ChannelComparison(
            IDENTIFIED,
            _group_mae(targets, predictions, groups),
            _group_mae(targets, baseline_predictions, groups),
            len(residuals),
            n_groups,
            n_groups / (n_groups + 1),
            0,
            0.0,
        ),
    )


def _fit_amount(
    rows: Sequence[Tuple[float, ...]],
    groups: Sequence[Hashable],
    values: Sequence[float],
    *,
    alphas: Sequence[float],
    seed: int,
    censored_rows: int = 0,
) -> _CalibratedChannel:
    positive = [
        (row, group, value)
        for row, group, value in zip(rows, groups, values)
        if value > 0
    ]
    positive_groups = {group for _, group, _ in positive}
    positive_values = {value for _, _, value in positive}
    usable_groups = len(set(groups))
    total_rows = len(values) + censored_rows
    if len(positive_groups) < 3 or len(positive_values) < 2:
        return _unidentified(
            CENSORED if censored_rows else UNIDENTIFIED,
            usable_groups,
            censored_rows,
            total_rows,
        )
    positive_rows, positive_group_ids, positive_values = zip(*positive)
    fitted = _fit_identified(
        positive_rows, positive_group_ids, positive_values,
        alphas=alphas, seed=seed,
    )
    censor_fraction = censored_rows / total_rows if total_rows else 0.0
    status = CENSORED if censor_fraction > 0.20 else IDENTIFIED
    return _CalibratedChannel(
        fitted.model,
        fitted.residuals,
        replace(
            fitted.comparison,
            status=status,
            censored_rows=censored_rows,
            censored_fraction=censor_fraction,
        ),
    )


def _fit_hurdle(
    rows: Sequence[Tuple[float, ...]],
    groups: Sequence[Hashable],
    occurred: Sequence[bool],
    *,
    alphas: Sequence[float],
    seed: int,
    censored_rows: int = 0,
) -> _CalibratedChannel:
    targets = tuple(float(value) for value in occurred)
    positive_groups = {
        group for group, target in zip(groups, targets) if target > 0
    }
    if len(positive_groups) < 3 or len(set(targets)) < 2:
        return _unidentified(
            CENSORED if censored_rows else UNIDENTIFIED,
            len(set(groups)),
            censored_rows,
            len(targets),
        )
    records = [
        Record(row, target, group)
        for row, target, group in zip(rows, targets, groups)
    ]
    n_groups = len(set(groups))
    folds = min(5, n_groups)

    def select_alpha(
        candidates_records: Sequence[Record],
        *,
        n_splits: int,
        selection_seed: int,
    ) -> float:
        candidate_losses = []
        for candidate in sorted(float(value) for value in alphas):
            losses = []
            for train_indices, validation_indices in grouped_kfold(
                candidates_records, n_splits, selection_seed
            ):
                train = [candidates_records[index] for index in train_indices]
                model = _RidgeLogisticModel.fit(train, candidate)
                for index in validation_indices:
                    probability = model.predict(
                        candidates_records[index].features
                    )
                    target = candidates_records[index].target
                    losses.append(
                        -target * math.log(probability)
                        - (1.0 - target) * math.log(1.0 - probability)
                    )
            candidate_losses.append(
                (sum(losses) / len(losses), -candidate, candidate)
            )
        return min(candidate_losses)[2]

    alpha = select_alpha(records, n_splits=folds, selection_seed=seed)
    default_alpha = sorted(float(value) for value in alphas)[len(alphas) // 2]
    predictions = [0.0] * len(records)
    baseline_predictions = [0.0] * len(records)
    for fold, (train_indices, validation_indices) in enumerate(
        grouped_kfold(records, folds, seed)
    ):
        train = [records[index] for index in train_indices]
        train_groups = len({record.group for record in train})
        fold_alpha = (
            select_alpha(
                train,
                n_splits=min(5, train_groups),
                selection_seed=seed + fold + 1,
            )
            if train_groups >= 2 else default_alpha
        )
        model = _RidgeLogisticModel.fit(train, fold_alpha)
        prevalence = sum(record.target for record in train) / len(train)
        for index in validation_indices:
            predictions[index] = model.predict(records[index].features)
            baseline_predictions[index] = prevalence
    residuals = _group_means(
        [target - prediction for target, prediction in zip(targets, predictions)],
        groups,
    )
    censor_fraction = censored_rows / len(targets) if targets else 0.0
    return _CalibratedChannel(
        _RidgeLogisticModel.fit(records, alpha),
        residuals,
        ChannelComparison(
            CENSORED if censor_fraction > 0.20 else IDENTIFIED,
            _group_mae(targets, predictions, groups),
            _group_mae(targets, baseline_predictions, groups),
            len(residuals),
            n_groups,
            n_groups / (n_groups + 1),
            censored_rows=censored_rows,
            censored_fraction=censor_fraction,
        ),
    )


class MultiChannelModel:
    """Fit quote-time features and grouped target channels, then forecast one row."""

    def __init__(
        self,
        *,
        width: int,
        channels: Mapping[str, _CalibratedChannel],
    ) -> None:
        self._width = width
        self._channels = dict(channels)
        self.channel_comparisons = {
            name: channel.comparison for name, channel in self._channels.items()
        }

    @classmethod
    def fit(
        cls,
        feature_rows: Sequence[Sequence[float]],
        observations: Sequence[ChannelObservation],
        *,
        feature_names: Sequence[str],
        alphas: Sequence[float] = (0.1, 1.0, 10.0),
        seed: int = 0,
    ) -> "MultiChannelModel":
        """Fit only named quote-time features; post-run feature names are rejected."""

        observations = tuple(observations)
        rows = _validate_rows(feature_rows, observations)
        if feature_names is None:
            raise ValueError("feature_names are required for quote-time leakage checks")
        if len(feature_names) != len(rows[0]):
            raise ValueError("feature_names must align with feature rows")
        leaked = sorted(
            str(name) for name in feature_names
            if str(name).strip().lower() in _POST_RUN_FEATURES
        )
        if leaked:
            raise ValueError("post-run feature names are prohibited: " + ", ".join(leaked))
        if not alphas or any(
            not math.isfinite(float(alpha)) or float(alpha) < 0
            for alpha in alphas
        ):
            raise ValueError("alphas must be finite non-negative values")

        groups = tuple(item.group for item in observations)
        uncensored = [
            (row, item) for row, item in zip(rows, observations)
            if not item.output_censored
        ]
        output_rows = tuple(row for row, _ in uncensored)
        output_observations = tuple(item for _, item in uncensored)
        output_groups = tuple(item.group for item in output_observations)
        censored_rows = len(observations) - len(output_observations)

        channels = {
            "cache_hit_probability": _fit_hurdle(
                rows, groups, [item.cached_tokens > 0 for item in observations],
                alphas=alphas, seed=seed,
            ),
            "cached_tokens": _fit_amount(
                rows, groups, [item.cached_tokens for item in observations],
                alphas=alphas, seed=seed + 1,
            ),
            "non_reasoning_output_tokens": _fit_amount(
                output_rows, output_groups,
                [item.non_reasoning_output_tokens for item in output_observations],
                alphas=alphas, seed=seed + 2, censored_rows=censored_rows,
            ) if output_rows else _unidentified(
                CENSORED, 0, censored_rows, len(observations)
            ),
            "reasoning_occurrence": _fit_hurdle(
                rows, groups,
                [item.reasoning_tokens > 0 for item in observations],
                alphas=alphas, seed=seed + 3, censored_rows=censored_rows,
            ),
            "reasoning_tokens": _fit_amount(
                output_rows, output_groups,
                [item.reasoning_tokens for item in output_observations],
                alphas=alphas, seed=seed + 4, censored_rows=censored_rows,
            ) if output_rows else _unidentified(
                CENSORED, 0, censored_rows, len(observations)
            ),
            "branch_or_retry_probability": _fit_hurdle(
                rows, groups, [item.branch_or_retry for item in observations],
                alphas=alphas, seed=seed + 5,
            ),
        }
        if any(item.non_reasoning_output_tokens == 0 for item in observations):
            channels["non_reasoning_occurrence"] = _fit_hurdle(
                rows, groups,
                [item.non_reasoning_output_tokens > 0 for item in observations],
                alphas=alphas, seed=seed + 8, censored_rows=censored_rows,
            )
        cost_rows = [
            (row, item.group, float(item.attempt_cost))
            for row, item in zip(rows, observations)
            if item.attempt_cost is not None
        ]
        if cost_rows:
            cost_features, cost_groups, cost_values = zip(*cost_rows)
            channels["attempt_cost"] = _fit_amount(
                cost_features, cost_groups, cost_values,
                alphas=alphas, seed=seed + 6,
            )
        else:
            channels["attempt_cost"] = _unidentified(UNAVAILABLE)

        acceptance_rows = [
            (row, item.group, bool(item.accepted))
            for row, item in zip(rows, observations)
            if item.accepted is not None
        ]
        if acceptance_rows:
            acceptance_features, acceptance_groups, acceptance_values = zip(
                *acceptance_rows
            )
            channels["acceptance_probability"] = _fit_hurdle(
                acceptance_features, acceptance_groups, acceptance_values,
                alphas=alphas, seed=seed + 7,
            )
        else:
            channels["acceptance_probability"] = _unidentified(UNAVAILABLE)

        if cost_rows and acceptance_rows:
            n_groups = min(
                channels["attempt_cost"].comparison.n_groups,
                channels["acceptance_probability"].comparison.n_groups,
            )
            channels["accepted_work_cost"] = _unidentified(UNAVAILABLE, n_groups)
        else:
            channels["accepted_work_cost"] = _unidentified(UNAVAILABLE)
        return cls(width=len(rows[0]), channels=channels)

    def forecast(
        self,
        features: Sequence[float],
        *,
        input_tokens: float = 0.0,
        output_bound_tokens: float,
        pricing: Optional[Pricing] = None,
    ) -> MultiChannelForecast:
        """Return bounded p50/p90/p95 forecasts for one quote-time feature row."""

        try:
            features = tuple(float(value) for value in features)
        except (TypeError, ValueError) as exc:
            raise ValueError("quote-time features must be finite numbers") from exc
        if len(features) != self._width or any(
            not math.isfinite(value) for value in features
        ):
            raise ValueError(f"expected {self._width} finite quote-time features")
        input_tokens = _number(input_tokens, "input_tokens")
        output_bound_tokens = _number(output_bound_tokens, "output_bound_tokens")
        pricing = pricing or Pricing()

        def probability(name: str) -> Optional[float]:
            value = self._channels[name].center(features)
            return None if value is None else min(1.0, value)

        cache_probability = probability("cache_hit_probability")
        reasoning_probability = probability("reasoning_occurrence")
        branch_probability = probability("branch_or_retry_probability")

        def hurdle(
            amount_name: str, probability_value: Optional[float], percentile: float,
        ) -> Optional[float]:
            if probability_value is None or self._channels[amount_name].model is None:
                return None
            if percentile <= 1.0 - probability_value:
                return 0.0
            conditional = (percentile - (1.0 - probability_value)) / probability_value
            return self._channels[amount_name].at(
                features, min(1.0, max(0.0, conditional))
            )

        core_channels = (
            "cache_hit_probability",
            "cached_tokens",
            "non_reasoning_output_tokens",
            "reasoning_occurrence",
            "reasoning_tokens",
            "branch_or_retry_probability",
        ) + (
            ("non_reasoning_occurrence",)
            if "non_reasoning_occurrence" in self._channels else ()
        )
        core_comparisons = [
            self._channels[name].comparison for name in core_channels
        ]
        minimum_groups = (
            min(item.n_groups for item in core_comparisons)
            if all(item.status == IDENTIFIED for item in core_comparisons)
            else 0
        )

        def quantile_status(percentile: float) -> str:
            required = 19 if percentile == 0.95 else 9 if percentile == 0.90 else 3
            return CALIBRATED if minimum_groups >= required else EXPLORATORY

        def point(percentile: float) -> ForecastPoint:
            cached = hurdle("cached_tokens", cache_probability, percentile)
            if cached is not None:
                cached = min(input_tokens, cached)
            if "non_reasoning_occurrence" in self._channels:
                non_reasoning = hurdle(
                    "non_reasoning_output_tokens",
                    probability("non_reasoning_occurrence"),
                    percentile,
                )
            else:
                non_reasoning = self._channels["non_reasoning_output_tokens"].at(
                    features, percentile
                )
            reasoning = hurdle("reasoning_tokens", reasoning_probability, percentile)
            raw_output = (
                non_reasoning + reasoning
                if non_reasoning is not None and reasoning is not None
                else None
            )
            if non_reasoning is not None:
                non_reasoning = min(non_reasoning, output_bound_tokens)
            if reasoning is not None:
                remaining_cap = (
                    output_bound_tokens - non_reasoning
                    if non_reasoning is not None else output_bound_tokens
                )
                reasoning = min(reasoning, remaining_cap)
            output = (
                non_reasoning + reasoning
                if non_reasoning is not None and reasoning is not None
                else None
            )
            total = input_tokens + output if output is not None else None
            cost = (
                pricing.attempt_cost(
                    input_tokens=input_tokens,
                    cached_input_tokens=cached,
                    output_tokens=output,
                )
                if cached is not None and output is not None else None
            )
            return ForecastPoint(
                input_tokens, cached, non_reasoning, reasoning, output,
                total, cost, None,
                raw_output >= output_bound_tokens if raw_output is not None else None,
                quantile_status(percentile),
            )

        cache_tail = self._channels["cached_tokens"].at(features, 0.95)
        reasoning_tail = self._channels["reasoning_tokens"].at(features, 0.95)
        p50, p90, p95 = point(0.50), point(0.90), point(0.95)
        # Component-wise tails do not determine their joint cost tail.  Keep
        # a conservative nondecreasing envelope rather than lowering a quote.
        p90 = replace(
            p90,
            cost=max(p50.cost, p90.cost)
            if p50.cost is not None and p90.cost is not None else None,
        )
        p95 = replace(
            p95,
            cost=max(p90.cost, p95.cost)
            if p90.cost is not None and p95.cost is not None else None,
        )
        return MultiChannelForecast(
            cache_probability,
            reasoning_probability,
            branch_probability,
            cache_tail,
            reasoning_tail,
            {
                name: channel.comparison.status
                for name, channel in self._channels.items()
            },
            minimum_groups >= 30,
            p50,
            p90,
            p95,
        )
