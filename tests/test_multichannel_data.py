import json

import pytest

from token_yield.economics import Pricing
from token_yield.multichannel_data import (
    ALL_EXCLUSION_REASONS,
    ExclusionReason,
    NormalizedObservation,
    RuntimeStratum,
    UsageBundle,
    filter_eligible,
    load_run,
    normalize_record,
    normalize_records,
    read_jsonl_records,
    read_run_metadata,
)


def _quote_features(**overrides):
    features = {
        "task_payload_tokens": 10, "provider_input_tokens": 110,
        "fixed_overhead_tokens": 100, "output_bound_tokens": 50,
        "output_spec_units": 2, "declared_fetch_bytes_kib": 0,
        "k_declared": 1, "k_free": 0, "effort_level": "minimal",
        "verbosity_level": "low", "cache_warm": 1,
        "brick_counts": {"summarise": 1}, "composition_arity": 1,
        "composition_mode": "sequential", "attempt_index": 1,
        "retry_policy": "none",
    }
    features.update(overrides)
    return features


def _usage(**overrides):
    usage = {
        "input_tokens": 100, "cached_tokens": 20, "output_tokens": 50,
        "reasoning_tokens": 10, "total_tokens": 150,
    }
    usage.update(overrides)
    return usage


def _wave3_row(**overrides):
    row = {
        "case_id": "case-1",
        "group_id": "group-a",
        "provenance": "measured",
        "endpoint": "https://example/openai/v1",
        "deployment": "gpt-5-mini",
        "usage": _usage(),
        "features": _quote_features(),
        "acceptance": {
            "overall_acceptance": True,
            "semantic_acceptance": True,
            "structural_acceptance": True,
            "provider_incomplete": False,
            "telemetry_complete": True,
            "errors": [],
        },
    }
    row.update(overrides)
    return row


def _wave2_row(**overrides):
    row = {
        "case_id": "case-2",
        "group_id": "group-b",
        "provenance": "measured",
        "endpoint": "https://example/openai/v1",
        "deployment": "gpt-5-mini",
        "usage": _usage(),
        "accepted": True,
        "error": None,
    }
    row.update(overrides)
    return row


# --- Usage identities -------------------------------------------------


def test_usage_bundle_derives_non_reasoning_output_tokens():
    usage = UsageBundle(
        input_tokens=100, cached_input_tokens=20, output_tokens=50,
        reasoning_tokens=10, total_tokens=150,
    )
    assert usage.non_reasoning_output_tokens == 40
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


def test_usage_bundle_rejects_reasoning_exceeding_output():
    with pytest.raises(ValueError, match="reasoning_tokens"):
        UsageBundle(
            input_tokens=100, cached_input_tokens=0, output_tokens=10,
            reasoning_tokens=20, total_tokens=110,
        )


def test_usage_bundle_rejects_cached_exceeding_input():
    with pytest.raises(ValueError, match="cached_input_tokens"):
        UsageBundle(
            input_tokens=10, cached_input_tokens=20, output_tokens=5,
            reasoning_tokens=0, total_tokens=15,
        )


def test_usage_bundle_rejects_inconsistent_total():
    with pytest.raises(ValueError, match="total_tokens"):
        UsageBundle(
            input_tokens=10, cached_input_tokens=0, output_tokens=5,
            reasoning_tokens=0, total_tokens=999,
        )


# --- Predictor leakage --------------------------------------------------


def test_normalize_record_rejects_leaked_predictor_in_features():
    row = _wave3_row()
    row["features"] = _quote_features()
    row["features"]["reasoning_tokens"] = 10
    with pytest.raises(ValueError, match="post-run predictors"):
        normalize_record(row)


def test_normalize_record_rejects_leaked_acceptance_in_features():
    row = _wave3_row()
    row["features"]["acceptance"] = True
    with pytest.raises(ValueError, match="post-run predictors"):
        normalize_record(row)


# --- Quote-time observation ----------------------------------------------


def test_wave3_row_with_valid_features_is_model_row_eligible():
    observation = normalize_record(_wave3_row())
    assert observation.quote.model_row_eligible is True
    assert observation.quote.ineligibility_reason is None
    assert observation.quote.features["task_payload_tokens"] == 10
    # Diagnostic-only upstream keys must not leak into the predictor dict.
    assert "model_row_eligible" not in observation.quote.features
    assert "ineligibility_reason" not in observation.quote.features


def test_upstream_ineligibility_is_binding_even_when_schema_is_valid():
    # An explicitly frozen upstream "ineligible" verdict must never be
    # silently overridden just because our independent schema validation
    # happens to pass.
    row = _wave3_row()
    row["features"]["model_row_eligible"] = False
    row["features"]["ineligibility_reason"] = "provider input-token endpoint unavailable"
    observation = normalize_record(row)
    assert observation.quote.upstream_model_row_eligible is False
    assert observation.quote.upstream_ineligibility_reason == (
        "provider input-token endpoint unavailable"
    )
    assert observation.quote.model_row_eligible is False
    assert "provider input-token endpoint unavailable" in observation.quote.ineligibility_reason
    assert (
        ExclusionReason.MISSING_QUOTE_TIME_FEATURES
        in observation.eligibility.exclusion_reasons
    )
    assert observation.eligibility.final_model_selection_eligible is False


def test_own_schema_failure_is_binding_even_when_upstream_claims_eligible():
    # The converse: upstream saying "eligible" cannot rescue a row whose
    # quote-time features fail our own independent validation.
    row = _wave3_row()
    row["features"]["model_row_eligible"] = True
    del row["features"]["k_declared"]
    observation = normalize_record(row)
    assert observation.quote.model_row_eligible is False
    assert "k_declared" in observation.quote.ineligibility_reason


def test_upstream_eligible_and_own_validation_pass_yields_model_row_eligible():
    row = _wave3_row()
    row["features"]["model_row_eligible"] = True
    observation = normalize_record(row)
    assert observation.quote.upstream_model_row_eligible is True
    assert observation.quote.model_row_eligible is True
    assert observation.quote.ineligibility_reason is None


def test_wave2_row_without_features_is_not_model_row_eligible():
    observation = normalize_record(_wave2_row())
    assert observation.quote.features is None
    assert observation.quote.model_row_eligible is False
    assert (
        ExclusionReason.MISSING_QUOTE_TIME_FEATURES
        in observation.eligibility.exclusion_reasons
    )


def test_wave3_row_missing_required_feature_is_not_model_row_eligible():
    row = _wave3_row()
    del row["features"]["k_declared"]
    observation = normalize_record(row)
    assert observation.quote.model_row_eligible is False
    assert "k_declared" in observation.quote.ineligibility_reason
    assert (
        ExclusionReason.MISSING_QUOTE_TIME_FEATURES
        in observation.eligibility.exclusion_reasons
    )


# --- Row identity requirements --------------------------------------------


def test_normalize_record_requires_case_id():
    row = _wave3_row()
    del row["case_id"]
    with pytest.raises(ValueError, match="case_id"):
        normalize_record(row)


def test_normalize_record_requires_group_id():
    row = _wave3_row()
    del row["group_id"]
    with pytest.raises(ValueError, match="group_id"):
        normalize_record(row)


def test_group_id_is_preserved_verbatim():
    row = _wave3_row(group_id="fetch:sec-apple-revenue:live")
    observation = normalize_record(row)
    assert observation.group_id == "fetch:sec-apple-revenue:live"
    assert observation.quote.group_id == "fetch:sec-apple-revenue:live"
    assert observation.targets.group_id == "fetch:sec-apple-revenue:live"


# --- Runtime strata --------------------------------------------------------


def test_runtime_stratum_combines_row_and_run_metadata():
    row = _wave3_row(endpoint="https://a", deployment="gpt-5-mini")
    metadata = {"runtime_stratum": "microsoft-foundry-direct-responses"}
    observation = normalize_record(row, run_metadata=metadata)
    stratum = observation.runtime_stratum
    assert stratum.endpoint == "https://a"
    assert stratum.deployment == "gpt-5-mini"
    assert stratum.runtime_stratum == "microsoft-foundry-direct-responses"
    assert stratum.protocol_version is None


def test_runtime_strata_are_not_pooled_across_different_protocol_versions():
    row = _wave3_row()
    stratum_v1 = normalize_record(
        row, run_metadata={"protocol_version": "wave3-repair-v1"}
    ).runtime_stratum
    stratum_v2 = normalize_record(
        row, run_metadata={"protocol_version": "wave3-repair-v2"}
    ).runtime_stratum
    assert stratum_v1.key != stratum_v2.key


# --- Wave 3 repair calibration-only ----------------------------------------


def test_wave3_repair_protocol_version_marks_calibration_only():
    row = _wave3_row()
    observation = normalize_record(
        row, run_metadata={"protocol_version": "wave3-repair-v2"}
    )
    assert observation.eligibility.wave3_repair_calibration_only is True
    assert observation.eligibility.final_model_selection_eligible is False
    assert (
        ExclusionReason.WAVE3_REPAIR_CALIBRATION_ONLY
        in observation.eligibility.exclusion_reasons
    )
    # A repair row otherwise usable for cache/input diagnostics stays usable.
    assert observation.eligibility.token_targets_eligible is True
    assert observation.eligibility.accepted_work_eligible is True


def test_non_repair_row_is_eligible_for_final_model_selection():
    row = _wave3_row()
    observation = normalize_record(
        row, run_metadata={"protocol_version": "some-future-protocol"}
    )
    assert observation.eligibility.wave3_repair_calibration_only is False
    assert observation.eligibility.final_model_selection_eligible is True


def test_repair_data_role_text_also_marks_calibration_only():
    row = _wave3_row()
    metadata = {
        "repair_data_role": (
            "calibration-and-instrumentation-only; prohibited from final "
            "model selection"
        ),
    }
    observation = normalize_record(row, run_metadata=metadata)
    assert observation.eligibility.wave3_repair_calibration_only is True


# --- Provider incomplete / target-specific eligibility ---------------------


def test_provider_incomplete_blocks_accepted_work_but_not_token_targets():
    row = _wave3_row(usage=_usage(output_tokens=0, reasoning_tokens=0, total_tokens=100))
    row["acceptance"] = {
        "overall_acceptance": False,
        "semantic_acceptance": None,
        "structural_acceptance": None,
        "provider_incomplete": True,
        "telemetry_complete": True,
        "errors": ["provider response incomplete"],
    }
    observation = normalize_record(row)
    assert observation.targets.provider_incomplete is True
    assert observation.eligibility.token_targets_eligible is True
    assert observation.eligibility.accepted_work_eligible is False
    # A provider-incomplete row can never be a generic final-model-selection
    # row, even though its features and usage are otherwise clean.
    assert observation.eligibility.final_model_selection_eligible is False
    assert (
        ExclusionReason.PROVIDER_INCOMPLETE_TARGET_UNAVAILABLE
        in observation.eligibility.exclusion_reasons
    )
    # Usage is still there for cache/input diagnostics.
    assert observation.targets.usage.input_tokens == 100


def test_wave2_incomplete_error_text_is_detected_as_provider_incomplete():
    row = _wave2_row(
        accepted=False,
        error="ResponseProtocolError: response has incomplete status 'incomplete'",
    )
    observation = normalize_record(row)
    assert observation.targets.provider_incomplete is True
    assert observation.eligibility.accepted_work_eligible is False


# --- Acceptance-only / unmetered vs unknown usage --------------------------


def test_non_metered_provenance_without_usage_is_acceptance_only_unmetered():
    row = _wave3_row(provenance="source-backed", usage=None)
    observation = normalize_record(row)
    assert observation.targets.usage is None
    assert observation.eligibility.token_targets_eligible is False
    assert (
        ExclusionReason.ACCEPTANCE_ONLY_UNMETERED
        in observation.eligibility.exclusion_reasons
    )
    assert ExclusionReason.UNKNOWN_USAGE not in observation.eligibility.exclusion_reasons


def test_measured_incomplete_without_usage_is_unknown_usage():
    row = _wave2_row(
        provenance="measured-incomplete",
        accepted=False,
        usage={
            "input_tokens": None, "cached_tokens": None, "output_tokens": None,
            "reasoning_tokens": None, "total_tokens": None,
        },
        error=(
            "ResponseProtocolError: provider usage was not captured because "
            "status validation preceded metering"
        ),
    )
    observation = normalize_record(row)
    assert observation.targets.usage is None
    assert observation.eligibility.token_targets_eligible is False
    assert (
        ExclusionReason.UNKNOWN_USAGE in observation.eligibility.exclusion_reasons
    )
    assert (
        ExclusionReason.ACCEPTANCE_ONLY_UNMETERED
        not in observation.eligibility.exclusion_reasons
    )
    assert observation.targets.semantic_accepted is False
    assert observation.eligibility.accepted_work_eligible is True


def test_partially_populated_usage_is_unknown_usage():
    row = _wave3_row(usage={
        "input_tokens": 100, "cached_tokens": None, "output_tokens": 50,
        "reasoning_tokens": 10, "total_tokens": 150,
    })
    observation = normalize_record(row)
    assert observation.targets.usage is None
    assert (
        ExclusionReason.UNKNOWN_USAGE in observation.eligibility.exclusion_reasons
    )


def test_usage_failing_an_identity_is_unknown_usage_not_a_hard_failure():
    row = _wave3_row(usage=_usage(cached_tokens=999))
    observation = normalize_record(row)
    assert observation.targets.usage is None
    assert (
        ExclusionReason.UNKNOWN_USAGE in observation.eligibility.exclusion_reasons
    )


# --- attempt_cost: every metered attempt, never gated on acceptance --------
#
# accepted_work_cost was survivorship-biased: it only priced rows that were
# already known to be accepted, which silently drops the cost of every
# rejected/incomplete attempt from the picture. attempt_cost instead prices
# every row with known usage, and any later "accepted-work cost" must be
# aggregated across a whole logical-work ledger (all attempts, including
# failed ones) rather than being read off of a single accepted row here.


def test_attempt_cost_is_none_without_pricing():
    row = _wave3_row()
    observation = normalize_record(row)
    assert observation.targets.attempt_cost is None


def test_attempt_cost_uses_economics_pricing_and_ignores_hand_entered_raw_field():
    row = _wave3_row()
    row["attempt_cost"] = 999.0  # a hand-entered value that must be ignored
    row["accepted_work_cost"] = 999.0  # likewise, for the retired field name
    pricing = Pricing(
        input_per_million=1.0, cached_input_per_million=0.1, output_per_million=2.0,
    )
    observation = normalize_record(row, pricing=pricing)
    usage = observation.targets.usage
    expected = (
        (usage.input_tokens - usage.cached_input_tokens) * pricing.input_per_million
        + usage.cached_input_tokens * pricing.cached_input_per_million
        + usage.output_tokens * pricing.output_per_million
    ) / 1_000_000.0
    assert observation.targets.attempt_cost == pytest.approx(expected)
    assert observation.targets.attempt_cost != 999.0


def test_attempt_cost_defaults_cached_rate_to_input_rate_when_unset():
    row = _wave3_row()
    pricing = Pricing(input_per_million=3.0, output_per_million=6.0)
    observation = normalize_record(row, pricing=pricing)
    usage = observation.targets.usage
    expected = (
        (usage.input_tokens - usage.cached_input_tokens) * pricing.input_per_million
        + usage.cached_input_tokens * pricing.input_per_million
        + usage.output_tokens * pricing.output_per_million
    ) / 1_000_000.0
    assert observation.targets.attempt_cost == pytest.approx(expected)


def test_attempt_cost_is_populated_even_when_not_accepted():
    row = _wave3_row()
    row["acceptance"]["overall_acceptance"] = False
    row["acceptance"]["semantic_acceptance"] = False
    pricing = Pricing(input_per_million=1.0, output_per_million=2.0)
    observation = normalize_record(row, pricing=pricing)
    assert observation.targets.overall_accepted is False
    assert observation.targets.attempt_cost is not None


def test_attempt_cost_is_populated_even_when_provider_incomplete():
    row = _wave3_row()
    row["acceptance"]["provider_incomplete"] = True
    row["acceptance"]["semantic_acceptance"] = None
    pricing = Pricing(input_per_million=1.0, output_per_million=2.0)
    observation = normalize_record(row, pricing=pricing)
    # The semantic/accepted-work label is unavailable, but the attempt still
    # incurred a real, knowable cost that must not be dropped from the ledger.
    assert observation.eligibility.accepted_work_eligible is False
    assert observation.targets.attempt_cost is not None


def test_attempt_cost_is_none_when_usage_is_unknown():
    row = _wave3_row(usage=None)
    pricing = Pricing(input_per_million=1.0, output_per_million=2.0)
    observation = normalize_record(row, pricing=pricing)
    assert observation.targets.usage is None
    assert observation.targets.attempt_cost is None


# --- Batch normalization and filtering -------------------------------------


def test_normalize_records_preserves_order():
    rows = [_wave3_row(case_id=f"case-{i}", group_id=f"group-{i}") for i in range(3)]
    observations = normalize_records(rows)
    assert [o.record_id for o in observations] == ["case-0", "case-1", "case-2"]


def test_filter_eligible_selects_only_matching_rows():
    eligible_row = _wave3_row(case_id="eligible", group_id="g1")
    ineligible_row = _wave2_row(case_id="ineligible", group_id="g2")
    observations = normalize_records([eligible_row, ineligible_row])
    model_rows = filter_eligible(observations, "model_row")
    assert [o.record_id for o in model_rows] == ["eligible"]
    final_rows = filter_eligible(observations, "final_model_selection")
    assert [o.record_id for o in final_rows] == ["eligible"]


def test_filter_eligible_rejects_unknown_target():
    observations = normalize_records([_wave3_row()])
    with pytest.raises(ValueError, match="unknown eligibility target"):
        filter_eligible(observations, "not-a-real-target")


def test_all_exclusion_reasons_matches_class_constants():
    assert ALL_EXCLUSION_REASONS == {
        ExclusionReason.ACCEPTANCE_ONLY_UNMETERED,
        ExclusionReason.UNKNOWN_USAGE,
        ExclusionReason.WAVE3_REPAIR_CALIBRATION_ONLY,
        ExclusionReason.PROVIDER_INCOMPLETE_TARGET_UNAVAILABLE,
        ExclusionReason.MISSING_QUOTE_TIME_FEATURES,
    }


# --- IO helpers --------------------------------------------------------


def test_read_jsonl_records_and_run_metadata_round_trip(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_wave3_row(case_id="r1", group_id="g1"), _wave2_row(case_id="r2", group_id="g2")]
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {"runtime_stratum": "microsoft-foundry-direct-responses"}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    loaded_rows = read_jsonl_records(run_dir / "records.jsonl")
    assert [row["case_id"] for row in loaded_rows] == ["r1", "r2"]
    loaded_metadata = read_run_metadata(run_dir / "run_metadata.json")
    assert loaded_metadata == metadata


def test_read_jsonl_records_returns_empty_list_for_missing_file(tmp_path):
    assert read_jsonl_records(tmp_path / "missing.jsonl") == []


def test_read_run_metadata_returns_empty_dict_for_missing_file(tmp_path):
    assert read_run_metadata(tmp_path / "missing.json") == {}


def test_load_run_normalizes_records_with_metadata(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = _wave3_row(case_id="r1", group_id="g1")
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"protocol_version": "wave3-repair-v2"}), encoding="utf-8"
    )
    observations = load_run(run_dir)
    assert len(observations) == 1
    assert observations[0].eligibility.wave3_repair_calibration_only is True


# --- Integration against real repository run history -----------------------


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_real_wave3_repair_run_is_excluded_from_final_selection():
    run_dir = REPO_ROOT / "runs" / "20260827_1152_wave3"
    if not run_dir.is_dir():
        pytest.skip("wave3 repair fixture run is not present")
    observations = load_run(run_dir)
    assert len(observations) == 36
    assert all(o.eligibility.wave3_repair_calibration_only for o in observations)
    assert all(not o.eligibility.final_model_selection_eligible for o in observations)
    assert {o.group_id for o in observations}
    assert len({o.record_id for o in observations}) == 36
    # Every row in this run is model-row ineligible: most recorded their own
    # upstream features.model_row_eligible == False (the provider
    # input-token count endpoint was unavailable), which must bind rather
    # than being overridden by our independent schema validation (which
    # would otherwise pass); the remainder simply recorded no features block
    # at all.
    with_features = [o for o in observations if o.quote.upstream_model_row_eligible is not None]
    assert with_features
    assert all(o.quote.upstream_model_row_eligible is False for o in with_features)
    assert all(not o.quote.model_row_eligible for o in observations)


def test_real_wave2_run_has_a_mix_of_eligibility_outcomes():
    run_dir = REPO_ROOT / "runs" / "20260826_1627_wave2"
    if not run_dir.is_dir():
        pytest.skip("wave2 fixture run is not present")
    observations = load_run(run_dir)
    assert observations
    assert all(not o.eligibility.wave3_repair_calibration_only for o in observations)
    # Every wave2 row lacks a features block, so none are model-row eligible.
    assert all(not o.quote.model_row_eligible for o in observations)
    reasons = {reason for o in observations for reason in o.eligibility.exclusion_reasons}
    assert ExclusionReason.MISSING_QUOTE_TIME_FEATURES in reasons
