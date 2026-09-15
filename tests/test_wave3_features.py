import json

import pytest

from token_yield.wave3_features import (
    FEATURE_REGISTRY,
    REJECTED_PREDICTORS,
    effect_code_bricks,
    extract_quote_features,
    local_payload_token_count,
    registry_as_dict,
    validate_quote_features,
)


def _row():
    return {
        "task_payload_tokens": 10, "provider_input_tokens": 110,
        "fixed_overhead_tokens": 100, "output_bound_tokens": 50,
        "output_spec_units": 2, "declared_fetch_bytes_kib": 0,
        "k_declared": 1, "k_free": 0, "effort_level": "minimal",
        "verbosity_level": "low", "cache_warm": 1,
        "brick_counts": {"summarise": 1}, "composition_arity": 1,
        "composition_mode": "sequential", "attempt_index": 1,
        "retry_policy": "none",
    }


def test_registry_is_complete_machine_readable_and_auditable():
    expected = {
        "task_payload_tokens", "provider_input_tokens", "fixed_overhead_tokens",
        "output_bound_tokens", "output_spec_units", "declared_fetch_bytes_kib",
        "k_declared", "k_free", "effort_level", "verbosity_level",
        "cache_warm", "brick_counts", "brick_effect_coding",
        "composition_arity", "composition_mode", "attempt_index", "retry_policy",
    }
    assert expected <= set(FEATURE_REGISTRY)
    serialized = registry_as_dict()
    json.dumps(serialized)
    required_metadata = {
        "definition", "unit", "measurement", "valid_range",
        "missing_value_rule", "leakage_classification", "citations",
        "compatible_bricks", "role",
    }
    assert all(set(item) == required_metadata for item in serialized.values())
    assert all(item["citations"] and item["compatible_bricks"]
               for item in serialized.values())
    assert all("quote_time" in item["leakage_classification"]
               for item in serialized.values())


def test_rejected_predictors_cover_all_post_run_leakage():
    assert {
        "observed_model_calls", "observed_tool_calls", "reasoning_tokens",
        "observed_response_bytes", "acceptance", "latency",
    } <= set(REJECTED_PREDICTORS)
    row = _row()
    row["latency"] = 2.5
    with pytest.raises(ValueError, match="post-run predictors"):
        validate_quote_features(row)


def test_local_token_count_is_deterministic_and_unicode_safe():
    count, method = local_payload_token_count("Café 日本")
    assert isinstance(count, int) and count > 0
    assert method == "tiktoken:o200k_base"
    assert local_payload_token_count("Café 日本") == (count, method)


def test_effect_coding_is_frozen_sorted_and_sum_to_zero_contrast():
    encoded = effect_code_bricks({"zeta": 2, "alpha": 1, "beta": 0})
    assert encoded == {
        "coding": "sum_to_zero",
        "vocabulary": ["alpha", "beta", "zeta"],
        "reference": "zeta",
        "columns": {"brick_effect__alpha": -1, "brick_effect__beta": -2},
    }


def test_quote_row_invariants_and_missing_rules():
    validate_quote_features(_row())
    bad = _row()
    bad["composition_arity"] = 2
    with pytest.raises(ValueError, match="composition_arity"):
        validate_quote_features(bad)
    bad = _row()
    del bad["attempt_index"]
    with pytest.raises(ValueError, match="missing quote-time"):
        validate_quote_features(bad)


def test_extraction_derives_kib_arity_effects_and_preserves_count_provenance():
    row = extract_quote_features(
        task_payload="Summarise this.", fixed_overhead_tokens=100,
        provider_input_tokens=104, output_bound_tokens=50,
        output_spec_units=2, declared_fetch_bytes=1536, k_declared=1,
        k_free=True, effort_level="medium", verbosity_level="medium",
        cache_warm=False, brick_counts={"summarise": 1, "fetch": 1},
        brick_vocabulary=["summarise", "fetch"], composition_mode="parallel",
        attempt_index=2, retry_policy="bounded",
    )
    assert row["declared_fetch_bytes_kib"] == 1.5
    assert row["composition_arity"] == 2
    assert row["k_free"] == 1
    assert row["task_payload_tokens_measurement"] == "tiktoken:o200k_base"
    assert row["brick_effect_coding"]["coding"] == "sum_to_zero"
