import json

import pytest

from assay.errors import SchemaError
from assay.evidence import EvidenceClass, assert_gateable, mixed
from assay.pricing import PROVISIONAL, Pricing, reconciles_with_billing
from assay.schemas import (
    ExclusionCode,
    Run,
    Split,
    Status,
    Tier,
    Usage,
    append_run,
    exclusion_rate,
    existing_run_ids,
    make_run_id,
    read_runs,
)


def make_run(**over) -> Run:
    base = dict(
        run_id="r1",
        campaign_id="c1",
        evidence_class=EvidenceClass.PIPELINE_ONLY,
        split=Split.FIT,
        tier=Tier.BASE,
        probe_id="p1",
        replicate=1,
        instruction_sha256="deadbeef",
        units={"Extract": 2},
        context_files=["doc-01.txt"],
        context_bytes=1000,
        rendered_prompt_bytes=1100,
        model_sent="m",
        model_echoed="m",
        prompt_tokens=1000,
        completion_tokens=100,
    )
    base.update(over)
    return Run(**base)  # type: ignore[arg-type]


# -- pricing ----------------------------------------------------------------------


def test_billable_units_weights_output_by_the_price_ratio():
    p = Pricing(input_per_mtok=1.0, output_per_mtok=5.0)
    assert p.ratio == 5.0
    assert p.billable_units(1000, 100) == 1500.0


def test_dollars_follow_from_units_without_a_hardcoded_ratio():
    p = Pricing(input_per_mtok=3.0, output_per_mtok=15.0)
    units = p.billable_units(1_000_000, 0)
    assert p.units_to_usd(units) == pytest.approx(3.0)


def test_changing_the_price_ratio_changes_the_target():
    cheap = Pricing(input_per_mtok=1.0, output_per_mtok=2.0)
    dear = Pricing(input_per_mtok=1.0, output_per_mtok=10.0)
    assert dear.billable_units(100, 100) > cheap.billable_units(100, 100)


def test_pricing_rejects_nonsense_rates():
    with pytest.raises(ValueError):
        Pricing(input_per_mtok=0.0, output_per_mtok=5.0)


def test_provisional_flag_is_carried_so_dollar_claims_can_be_blocked():
    assert PROVISIONAL.provisional is True


def test_pricing_round_trips_and_hashes_stably(tmp_path):
    path = tmp_path / "pricing.json"
    PROVISIONAL.save(path)
    again = Pricing.load(path)
    assert again == PROVISIONAL
    assert again.hash() == PROVISIONAL.hash()


def test_billing_reconciliation_is_a_one_percent_rule():
    assert reconciles_with_billing(100.0, 100.5)
    assert not reconciles_with_billing(100.0, 90.0)


# -- evidence ---------------------------------------------------------------------


def test_only_feasibility_can_move_a_gate():
    assert EvidenceClass.FEASIBILITY.can_move_a_gate
    assert not EvidenceClass.PIPELINE_ONLY.can_move_a_gate
    assert not EvidenceClass.REPLAY.can_move_a_gate


def test_assert_gateable_refuses_mock_evidence():
    with pytest.raises(ValueError, match="non-feasibility"):
        assert_gateable([EvidenceClass.PIPELINE_ONLY])
    assert_gateable([EvidenceClass.FEASIBILITY])


def test_mixed_detects_a_blended_report():
    assert mixed([EvidenceClass.REPLAY, EvidenceClass.FEASIBILITY])
    assert not mixed([EvidenceClass.FEASIBILITY])


# -- run records ------------------------------------------------------------------


def test_derived_cost_is_recomputed_not_trusted(tmp_path):
    run = make_run().attach_pricing(PROVISIONAL)
    path = tmp_path / "runs.jsonl"
    append_run(path, run)

    # Tamper with the stored cost the way a spreadsheet edit would.
    line = json.loads(path.read_text(encoding="utf-8"))
    line["billable_units"] = 999_999.0
    line["cost_usd"] = 999.0
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    loaded = read_runs(path, PROVISIONAL)[0]
    assert loaded.billable_units == PROVISIONAL.billable_units(1000, 100)
    assert loaded.cost_usd != 999.0


def test_cache_tokens_exclude_the_row():
    run = make_run(cache_read_tokens=5).enforce_contract()
    assert run.exclusion_code is ExclusionCode.CACHE_NONZERO
    assert not run.accepted


def test_hidden_reasoning_tokens_exclude_the_row():
    run = make_run(reasoning_tokens=1).enforce_contract()
    assert run.exclusion_code is ExclusionCode.REASONING_NONZERO


def test_a_different_model_excludes_the_row():
    run = make_run(model_echoed="something-else").enforce_contract(expected_model="m")
    assert run.exclusion_code is ExclusionCode.MODEL_MISMATCH


def test_a_clean_run_survives_the_contract():
    assert make_run().enforce_contract(expected_model="m").accepted


def test_usage_is_never_silently_zero():
    """A failed dispatch is an excluded measurement, not a free run."""
    run = make_run(
        prompt_tokens=0, completion_tokens=0,
        status=Status.FAILED, exclusion_code=ExclusionCode.USAGE_MISSING,
    ).enforce_contract()
    assert not run.accepted
    assert run.status is Status.EXCLUDED


def test_only_fit_split_fittable_tiers_enter_the_model():
    assert make_run(tier=Tier.BASE, split=Split.FIT).fittable
    assert not make_run(tier=Tier.LADDER, split=Split.PROBE).fittable
    assert not make_run(tier=Tier.BLIND, split=Split.BLIND).fittable
    assert not make_run(tier=Tier.BATCHING, split=Split.BATCHING).fittable


def test_run_id_is_stable_and_input_sensitive():
    a = make_run_id("p1", ["h1"], "rt", "pr", 1)
    assert a == make_run_id("p1", ["h1"], "rt", "pr", 1)
    assert a != make_run_id("p1", ["h1"], "rt", "pr", 2)
    assert a != make_run_id("p1", ["h2"], "rt", "pr", 1)
    assert a != make_run_id("p1", ["h1"], "rt2", "pr", 1)


def test_storage_is_append_only(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_run(path, make_run(run_id="a"))
    append_run(path, make_run(run_id="b"))
    assert existing_run_ids(path) == {"a", "b"}
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_unknown_fields_are_rejected_rather_than_ignored(tmp_path):
    path = tmp_path / "runs.jsonl"
    payload = make_run().to_dict()
    payload["surprise"] = 1
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="unknown fields"):
        read_runs(path, PROVISIONAL)


def test_a_bad_line_names_its_line_number(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=":1:"):
        read_runs(path, PROVISIONAL)


def test_exclusion_rate_ignores_transport_failures():
    runs = [
        make_run(run_id="a"),
        make_run(run_id="b", exclusion_code=ExclusionCode.TRANSPORT_ERROR),
        make_run(run_id="c", exclusion_code=ExclusionCode.TRUNCATED),
    ]
    for r in runs:
        r.enforce_contract()
    rate, excluded, considered = exclusion_rate(runs)
    assert (excluded, considered) == (1, 2)
    assert rate == pytest.approx(50.0)


def test_usage_contract_violation_returns_none_when_clean():
    assert Usage(10, 1).contract_violation() is None
