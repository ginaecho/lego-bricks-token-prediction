import pytest

from assay.evidence import EvidenceClass
from assay.verdict import (
    DEFAULT_POLICY,
    Direction,
    GateSpec,
    Outcome,
    evaluate,
    policy_hash,
    save_policy,
)

ALL_PASS = {
    "M1_noise_floor_pct": 0.5,
    "M2_drift_pct": 0.4,
    "M5b_cv_lift_pct": 45.0,
    "M6_blind_mape_pct": 9.0,
    "M6b_blind_lift_pct": 31.0,
    "M7_encoder_exact_pct": 90.0,
    "M9_e2e_mape_pct": 12.0,
    "M9b_coverage_hits": 22.0,
    "M11_batching_saving_pct": 52.0,
    "M12_batching_quality_lb": -1.2,
    "M13_alpha_macro": 0.87,
    "M14_exclusion_pct": 3.0,
    "M15_identifiable": 1.0,
}


def verdict(**over):
    obs = dict(ALL_PASS)
    obs.update(over)
    return evaluate(obs, evidence_class=EvidenceClass.FEASIBILITY)


def test_everything_passing_is_feasible():
    assert verdict().outcome is Outcome.FEASIBLE


def test_a_missing_gate_is_narrow_not_a_pass():
    """Untested is not passed. Batching that was never run cannot make a project Feasible."""
    v = verdict(M11_batching_saving_pct=None)
    assert v.outcome is Outcome.NARROW
    assert any(g.reason == "not measured" for g in v.narrowing_gates())


def test_blind_error_above_the_stop_bar_stops():
    assert verdict(M6_blind_mape_pct=40.0).outcome is Outcome.STOP


def test_blind_error_inside_the_narrow_band_narrows():
    assert verdict(M6_blind_mape_pct=22.0).outcome is Outcome.NARROW


@pytest.mark.parametrize("value,expected", [(15.0, Outcome.FEASIBLE), (22.0, Outcome.NARROW), (31.0, Outcome.STOP)])
def test_blind_error_boundaries_are_exact(value, expected):
    assert verdict(M6_blind_mape_pct=value).outcome is expected


def test_a_lift_failure_narrows_but_never_stops():
    """The 'ship a runtime estimator' outcome has to be reachable.

    If bricks add nothing but a size-and-count baseline still predicts cost accurately,
    that is a smaller true result, not a dead project.
    """
    v = verdict(M5b_cv_lift_pct=-5.0, M6b_blind_lift_pct=-3.0)
    assert v.outcome is Outcome.NARROW
    assert not v.stopping_gates()


def test_cheap_but_worse_batching_narrows_and_does_not_stop():
    v = verdict(M11_batching_saving_pct=61.0, M12_batching_quality_lb=-14.0)
    assert v.outcome is Outcome.NARROW


def test_unusable_intervals_narrow_to_point_quotes():
    assert verdict(M9b_coverage_hits=12.0).outcome is Outcome.NARROW


def test_integrity_failures_stop():
    assert verdict(M14_exclusion_pct=18.0).outcome is Outcome.STOP
    assert verdict(M2_drift_pct=9.0).outcome is Outcome.STOP
    assert verdict(M13_alpha_macro=0.55).outcome is Outcome.STOP
    assert verdict(M1_noise_floor_pct=6.0).outcome is Outcome.STOP


def test_an_unusable_encoder_narrows_and_does_not_stop():
    """A bad encoder is not a dead project; the decoder underneath still quotes.

    The gate that stops on disagreement is M13: if trained humans cannot agree what a
    brick is, the vocabulary is not a measurable construct and every unit count in the
    study is uninterpretable. An *automatic* encoder failing is a different thing --
    quoting stays manual, and a manual quote is still a quote.
    """
    v = verdict(M7_encoder_exact_pct=41.0, M9_e2e_mape_pct=64.0)
    assert v.outcome is Outcome.NARROW
    assert not v.stopping_gates()
    assert any("capped at narrow" in n for n in v.notes)


def test_not_measuring_the_encoder_is_never_worse_than_measuring_it_badly():
    """Measuring must not be punished harder than declining to measure.

    Before M7/M9 were caps, a 41% encoder Stopped the project while an unmeasured one
    merely Narrowed it -- an incentive to leave the gate untested.
    """
    unmeasured = verdict(M7_encoder_exact_pct=None, M9_e2e_mape_pct=None)
    measured_badly = verdict(M7_encoder_exact_pct=41.0, M9_e2e_mape_pct=64.0)
    assert measured_badly.outcome.rank <= unmeasured.outcome.rank


def test_an_unidentifiable_design_caps_at_narrow():
    v = verdict(M15_identifiable=0.0)
    assert v.outcome is Outcome.NARROW
    assert any("capped at narrow" in n for n in v.notes)


def test_stop_exits_non_zero_and_pass_exits_zero():
    assert verdict().exit_code == 0
    assert verdict(M6_blind_mape_pct=99.0).exit_code == 1


def test_pipeline_only_evidence_cannot_quietly_read_as_a_finding():
    v = evaluate(ALL_PASS, evidence_class=EvidenceClass.PIPELINE_ONLY)
    assert any("not a finding" in n for n in v.notes)
    assert "THIS IS NOT A RESULT" in v.render()


def test_render_lists_every_gate_with_its_bar():
    text = verdict().render()
    for spec in DEFAULT_POLICY:
        assert spec.id in text
    assert "VERDICT: FEASIBLE" in text


def test_observations_naming_an_unknown_gate_are_refused():
    with pytest.raises(ValueError, match="not in the policy"):
        evaluate({"M99_invented": 1.0})


def test_policy_hash_is_stable_and_changes_with_a_moved_bar():
    before = policy_hash()
    moved = tuple(
        GateSpec("M6_blind_mape_pct", "x", Direction.LOWER_IS_BETTER, 99.0)
        if g.id == "M6_blind_mape_pct" else g
        for g in DEFAULT_POLICY
    )
    assert policy_hash() == before
    assert policy_hash(moved) != before, "moving a bar must change the policy hash"


def test_saved_policy_carries_its_hash(tmp_path):
    import json

    path = tmp_path / "verdict-policy.json"
    save_policy(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["policy_sha256"] == policy_hash()
    assert len(data["gates"]) == len(DEFAULT_POLICY)
