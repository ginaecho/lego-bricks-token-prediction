import pytest

from token_yield.economics import Pricing
from token_yield.multichannel_model import (
    ChannelObservation,
    MultiChannelModel,
)


FEATURE_NAMES = ("declared_size", "cache_warm")


def _training_rows(*, censored=False, ledger=True):
    features = []
    observations = []
    for group, offset in enumerate((0, 1, 2, 3, 4, 5), start=1):
        for size in (1.0, 2.0):
            features.append((size, float(offset % 2)))
            reasoning = 20.0 + 5.0 * size if offset % 2 else 0.0
            cached = 50.0 + 5.0 * size if offset % 3 else 0.0
            observations.append(ChannelObservation(
                group=f"case-{group}",
                cached_tokens=cached,
                non_reasoning_output_tokens=100.0 + 10.0 * size,
                reasoning_tokens=reasoning,
                branch_or_retry=bool(offset % 2),
                output_bound_tokens=200.0,
                output_censored=censored,
                attempt_cost=0.02 + 0.001 * size if ledger else None,
                accepted=bool(offset % 2) if ledger else None,
            ))
    return features, observations


def test_fit_requires_named_quote_time_features_and_rejects_all_known_aliases():
    features, observations = _training_rows()

    with pytest.raises(TypeError, match="feature_names"):
        MultiChannelModel.fit(features, observations)
    with pytest.raises(ValueError, match="post-run"):
        MultiChannelModel.fit(
            features, observations, feature_names=("declared_size", "model_calls")
        )


def test_forecast_reconstructs_channels_without_double_counting():
    features, observations = _training_rows()
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=17
    )

    forecast = model.forecast(
        (1.5, 1.0),
        input_tokens=400.0,
        output_bound_tokens=140.0,
        pricing=Pricing(
            input_per_million=1.0,
            cached_input_per_million=0.1,
            output_per_million=2.0,
        ),
    )

    assert 0.0 <= forecast.cache_hit_probability <= 1.0
    assert 0.0 <= forecast.reasoning_probability <= 1.0
    assert 0.0 <= forecast.branch_or_retry_probability <= 1.0
    assert forecast.p50.output_tokens == pytest.approx(
        forecast.p50.non_reasoning_output_tokens + forecast.p50.reasoning_tokens
    )
    assert forecast.p50.total_tokens == pytest.approx(
        forecast.p50.input_tokens + forecast.p50.output_tokens
    )
    assert forecast.p50.output_tokens <= 140.0
    assert forecast.p95.output_tokens <= 140.0
    assert forecast.p95.cap_binding
    assert forecast.p50.cost == pytest.approx(
        (forecast.p50.input_tokens - forecast.p50.cached_tokens) * 0.000001
        + forecast.p50.cached_tokens * 0.0000001
        + forecast.p50.output_tokens * 0.000002
    )
    assert forecast.p50.accepted_work_cost is None
    assert forecast.channel_status["accepted_work_cost"] == "UNAVAILABLE"


def test_group_level_calibration_and_small_sample_tails_are_explicit():
    features, observations = _training_rows()
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=23
    )
    forecast = model.forecast(
        (2.0, 0.0), input_tokens=300.0, output_bound_tokens=300.0
    )

    comparison = model.channel_comparisons["non_reasoning_output_tokens"]
    assert comparison.status == "IDENTIFIED"
    assert comparison.n_groups == 6
    assert comparison.residual_count == 6
    assert comparison.attainable_conformal_level == pytest.approx(6 / 7)
    assert forecast.p90.calibration_status == "EXPLORATORY"
    assert forecast.p95.calibration_status == "EXPLORATORY"
    assert forecast.cache_conditional_positive_p95 is not None
    assert forecast.reasoning_conditional_positive_p95 is not None
    assert forecast.p50.cost <= forecast.p90.cost <= forecast.p95.cost
    assert forecast == MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=23
    ).forecast((2.0, 0.0), input_tokens=300.0, output_bound_tokens=300.0)


def test_unidentified_hurdles_are_not_presented_as_zero_risk():
    features, observations = _training_rows()
    observations = [
        ChannelObservation(
            group=item.group,
            cached_tokens=item.cached_tokens,
            non_reasoning_output_tokens=item.non_reasoning_output_tokens,
            reasoning_tokens=0.0,
            branch_or_retry=False,
            output_bound_tokens=item.output_bound_tokens,
            attempt_cost=item.attempt_cost,
            accepted=item.accepted,
        )
        for item in observations
    ]
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=4
    )
    forecast = model.forecast(
        (1.0, 0.0), input_tokens=200.0, output_bound_tokens=200.0
    )

    comparison = model.channel_comparisons["reasoning_tokens"]
    assert comparison.status == "UNIDENTIFIED"
    assert comparison.model_mae is None
    assert forecast.reasoning_probability is None
    assert forecast.reasoning_conditional_positive_p95 is None
    assert forecast.channel_status["reasoning_tokens"] == "UNIDENTIFIED"
    assert forecast.p95.reasoning_tokens is None
    assert forecast.p95.output_tokens is None
    assert forecast.p95.total_tokens is None
    assert forecast.p95.cost is None
    assert forecast.p95.cap_binding is None


def test_censored_outputs_are_excluded_and_reported_unfit_when_needed():
    features, observations = _training_rows(censored=True)
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=9
    )
    forecast = model.forecast(
        (1.0, 0.0), input_tokens=100.0, output_bound_tokens=90.0
    )

    assert model.channel_comparisons["non_reasoning_output_tokens"].status == "CENSORED"
    assert model.channel_comparisons["reasoning_tokens"].status == "CENSORED"
    assert forecast.channel_status["non_reasoning_output_tokens"] == "CENSORED"
    assert forecast.p50.output_tokens is None
    assert forecast.p50.cap_binding is None


def test_positive_amount_channel_does_not_treat_zero_hurdle_rows_as_amounts():
    features = [(4.0,), (4.0,), (4.0,), (1.0,), (2.0,), (4.0,)]
    observations = [
        ChannelObservation(
            group=f"case-{index}",
            cached_tokens=cached,
            non_reasoning_output_tokens=100.0,
            reasoning_tokens=0.0,
            branch_or_retry=bool(index % 2),
            output_bound_tokens=200.0,
        )
        for index, cached in enumerate((0.0, 0.0, 0.0, 100.0, 200.0, 400.0))
    ]
    model = MultiChannelModel.fit(
        features, observations, feature_names=("declared_size",), seed=5
    )
    forecast = model.forecast(
        (4.0,), input_tokens=500.0, output_bound_tokens=200.0
    )

    assert forecast.cache_conditional_positive_p95 > 350.0


def test_heavily_censored_amount_channel_reports_censoring_metadata():
    features, observations = _training_rows()
    observations = [
        ChannelObservation(
            group=item.group,
            cached_tokens=item.cached_tokens,
            non_reasoning_output_tokens=item.non_reasoning_output_tokens,
            reasoning_tokens=item.reasoning_tokens,
            branch_or_retry=item.branch_or_retry,
            output_bound_tokens=item.output_bound_tokens,
            output_censored=index < 3,
            attempt_cost=item.attempt_cost,
            accepted=item.accepted,
        )
        for index, item in enumerate(observations)
    ]
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=8
    )

    comparison = model.channel_comparisons["non_reasoning_output_tokens"]
    assert comparison.status == "CENSORED"
    assert comparison.censored_rows == 3
    assert comparison.censored_fraction == pytest.approx(0.25)


def test_reasoning_occurrence_uses_censored_rows_but_reports_censoring():
    features, observations = _training_rows()
    observations = [
        ChannelObservation(
            group=item.group,
            cached_tokens=item.cached_tokens,
            non_reasoning_output_tokens=item.non_reasoning_output_tokens,
            reasoning_tokens=item.reasoning_tokens,
            branch_or_retry=item.branch_or_retry,
            output_bound_tokens=item.output_bound_tokens,
            output_censored=index < 3,
            attempt_cost=item.attempt_cost,
            accepted=item.accepted,
        )
        for index, item in enumerate(observations)
    ]
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=18
    )

    comparison = model.channel_comparisons["reasoning_occurrence"]
    assert comparison.status == "CENSORED"
    assert comparison.n_groups == 6
    assert comparison.censored_rows == 3
    assert comparison.censored_fraction == pytest.approx(0.25)


def test_zero_non_reasoning_output_is_modeled_with_a_hurdle():
    features, observations = _training_rows()
    observations = [
        ChannelObservation(
            group=item.group,
            cached_tokens=item.cached_tokens,
            non_reasoning_output_tokens=0.0 if index % 2 == 0 else 120.0 + index,
            reasoning_tokens=item.reasoning_tokens,
            branch_or_retry=item.branch_or_retry,
            output_bound_tokens=200.0,
            attempt_cost=item.attempt_cost,
            accepted=item.accepted,
        )
        for index, item in enumerate(observations)
    ]
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=19
    )
    forecast = model.forecast(
        (1.0, 1.0), input_tokens=200.0, output_bound_tokens=200.0
    )

    assert "non_reasoning_occurrence" in model.channel_comparisons
    assert model.channel_comparisons["non_reasoning_occurrence"].status == "IDENTIFIED"
    assert forecast.p50.non_reasoning_output_tokens == 0.0


def test_fitted_hurdle_does_not_publish_hard_zero_probability():
    features, observations = _training_rows()
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=20
    )
    forecast = model.forecast(
        (1.0, 100.0), input_tokens=200.0, output_bound_tokens=200.0
    )

    assert 0.0 < forecast.reasoning_probability < 1.0
    assert 0.0 < forecast.branch_or_retry_probability < 1.0


def test_unidentified_cache_amount_keeps_rate_sensitive_cost_unknown():
    features, observations = _training_rows()
    observations = [
        ChannelObservation(
            group=item.group,
            cached_tokens=0.0,
            non_reasoning_output_tokens=item.non_reasoning_output_tokens,
            reasoning_tokens=item.reasoning_tokens,
            branch_or_retry=item.branch_or_retry,
            output_bound_tokens=item.output_bound_tokens,
            attempt_cost=item.attempt_cost,
            accepted=item.accepted,
        )
        for item in observations
    ]
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=12
    )
    forecast = model.forecast(
        (1.0, 1.0), input_tokens=100.0, output_bound_tokens=200.0
    )

    assert forecast.p95.cached_tokens is None
    assert forecast.p95.output_tokens is not None
    assert forecast.p95.total_tokens is not None
    assert forecast.p95.cost is None


def test_accepted_work_economics_requires_complete_attempt_and_acceptance_ledger():
    features, observations = _training_rows(ledger=False)
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=3
    )
    forecast = model.forecast(
        (1.0, 0.0), input_tokens=100.0, output_bound_tokens=100.0
    )

    assert model.channel_comparisons["accepted_work_cost"].status == "UNAVAILABLE"
    assert forecast.p50.accepted_work_cost is None


def test_attempt_cost_and_acceptance_use_their_own_eligible_rows():
    features, observations = _training_rows()
    observations = [
        ChannelObservation(
            group=item.group,
            cached_tokens=item.cached_tokens,
            non_reasoning_output_tokens=item.non_reasoning_output_tokens,
            reasoning_tokens=item.reasoning_tokens,
            branch_or_retry=item.branch_or_retry,
            output_bound_tokens=item.output_bound_tokens,
            attempt_cost=item.attempt_cost,
            accepted=None if index < 2 else item.accepted,
        )
        for index, item in enumerate(observations)
    ]
    model = MultiChannelModel.fit(
        features, observations, feature_names=FEATURE_NAMES, seed=24
    )

    assert model.channel_comparisons["attempt_cost"].status == "IDENTIFIED"
    assert model.channel_comparisons["attempt_cost"].n_groups == 6
    assert model.channel_comparisons["acceptance_probability"].status == "IDENTIFIED"
    assert model.channel_comparisons["acceptance_probability"].n_groups == 5
    assert model.channel_comparisons["accepted_work_cost"].status == "UNAVAILABLE"
