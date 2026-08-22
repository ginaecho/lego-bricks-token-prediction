"""Tests for the fitted calibration layer: taxonomy, cost models, learning, plans."""

import math

import pytest

from token_yield.taxonomy import (
    KindRegistry, Provenance, ScopedRecord, TaskKind, measured_only,
)
from token_yield.costmodel import (
    AffineModel, ConstantModel, PowerModel, ProportionalModel,
    fit_affine, fit_constant, fit_power, fit_proportional,
    group_by_kind, loo_mape, mape, select_model,
)
from token_yield.backtest import backtest, learning_curve, noise_floor
from token_yield.learn import LearningStore, score_against, seeded_store
from token_yield.plan import PlanForecaster, WorkPlan
from token_yield.probes import (
    COMPOSITION_MEASURED, MEASURED, PROBE_SUITE, composition_evidence,
    replicate_spread,
)


def rec(kind, scope, tokens, prov=Provenance.PROBE):
    return ScopedRecord(kind, scope, tokens, provenance=prov)


# ── taxonomy ────────────────────────────────────────────────────────────

def test_scoped_record_rejects_nonpositive_scope():
    with pytest.raises(ValueError):
        ScopedRecord("k", 0, 100)
    with pytest.raises(ValueError):
        ScopedRecord("k", -1, 100)


def test_scoped_record_rejects_negative_tokens():
    with pytest.raises(ValueError):
        ScopedRecord("k", 1, -5)


def test_provenance_marks_synthetic_as_unmeasured():
    assert Provenance.PROBE.is_measured
    assert Provenance.PRODUCTION.is_measured
    assert not Provenance.SYNTHETIC.is_measured


def test_measured_only_drops_synthetic():
    rs = [rec("k", 1, 10), rec("k", 1, 20, Provenance.SYNTHETIC)]
    assert len(measured_only(rs)) == 1


def test_registry_is_open():
    reg = KindRegistry()
    assert "novel" not in reg
    kind = reg.ensure("novel")
    assert isinstance(kind, TaskKind)
    assert "novel" in reg
    assert len(reg) == 1


def test_kind_describe_pluralises():
    k = TaskKind("comprehension", "file")
    assert k.describe(1) == "comprehension (1 file)"
    assert k.describe(3) == "comprehension (3 files)"


# ── fitters ─────────────────────────────────────────────────────────────

def test_fit_affine_recovers_known_line():
    rs = [rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)]
    m = fit_affine("k", rs)
    assert isinstance(m, AffineModel)
    assert m.fixed == pytest.approx(5000, abs=1e-6)
    assert m.marginal == pytest.approx(100, abs=1e-6)
    assert m.residual_sigma == pytest.approx(0.0, abs=1e-6)


def test_fit_proportional_recovers_known_slope():
    rs = [rec("k", x, 300 * x) for x in (1, 2, 5)]
    m = fit_proportional("k", rs)
    assert m.params[0] == pytest.approx(300, abs=1e-6)


def test_fit_power_recovers_known_exponent():
    rs = [rec("k", x, int(round(100 * x ** 0.5))) for x in (1, 4, 9, 16)]
    m = fit_power("k", rs)
    assert isinstance(m, PowerModel)
    assert m.params[1] == pytest.approx(0.5, abs=0.05)


def test_fit_constant_is_the_mean():
    rs = [rec("k", x, t) for x, t in [(1, 100), (2, 200), (3, 300)]]
    assert fit_constant("k", rs).params[0] == pytest.approx(200)


def test_affine_needs_scope_variation():
    rs = [rec("k", 3, 100), rec("k", 3, 120)]
    assert fit_affine("k", rs) is None       # slope unidentifiable


def test_fitters_return_none_on_empty():
    assert fit_affine("k", []) is None
    assert fit_constant("k", []) is None
    assert fit_proportional("k", []) is None
    assert fit_power("k", []) is None


# ── the model FORM is chosen by data, not asserted ──────────────────────

def test_selection_picks_affine_when_there_is_a_fixed_cost():
    rs = [rec("k", x, 5000 + 100 * x) for x in (1, 2, 3, 5, 8)]
    sel = select_model("k", rs)
    assert sel.form == "affine"


def test_selection_picks_proportional_for_through_origin_data():
    """Parsimony tie-break: affine fits this perfectly too, but costs a parameter."""
    rs = [rec("k", x, 1000 * x) for x in (1, 2, 3, 4, 6)]
    sel = select_model("k", rs)
    assert sel.form == "proportional"


def test_selection_picks_constant_when_scope_carries_no_signal():
    rs = [rec("k", x, 5000) for x in (1, 2, 3, 5, 8)]
    sel = select_model("k", rs)
    assert sel.form == "constant"


def test_selection_reports_its_scores_and_reason():
    rs = [rec("k", x, 5000 + 100 * x) for x in (1, 2, 3, 5, 8)]
    sel = select_model("k", rs)
    assert "proportional" in sel.scores
    assert sel.scores["affine"] < sel.scores["proportional"]
    assert "MAPE" in sel.reason


def test_selection_falls_back_to_parsimony_when_too_thin_to_cv():
    sel = select_model("k", [rec("k", 1, 100)])
    assert sel.form == "constant"
    assert "parsimon" in sel.reason


def test_select_model_none_on_empty():
    assert select_model("k", []) is None


# ── metrics ─────────────────────────────────────────────────────────────

def test_mape_is_a_fraction():
    assert mape([100, 200], [110, 180]) == pytest.approx(0.1)


def test_mape_ignores_zero_actuals():
    assert mape([0, 100], [50, 110]) == pytest.approx(0.1)


def test_loo_needs_enough_points():
    assert loo_mape("k", [rec("k", 1, 10), rec("k", 2, 20)], "affine") is None


def test_group_by_kind():
    rs = [rec("a", 1, 10), rec("b", 1, 20), rec("a", 2, 30)]
    g = group_by_kind(rs)
    assert set(g) == {"a", "b"}
    assert len(g["a"]) == 2


# ── regime / extrapolation ──────────────────────────────────────────────

def test_model_knows_its_fitted_range():
    rs = [rec("k", x, 5000 + 100 * x) for x in (1, 2, 4)]
    m = fit_affine("k", rs)
    assert m.in_regime(2)
    assert not m.in_regime(50)
    assert m.extrapolation_factor(2) == 1.0
    assert m.extrapolation_factor(8) == pytest.approx(2.0)


def test_interval_widens_with_residual_spread():
    tight = [rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)]
    noisy = [rec("k", 1, 5100), rec("k", 2, 5000), rec("k", 4, 6500),
             rec("k", 8, 5400)]
    lo1, hi1 = fit_affine("k", tight).interval(3)
    lo2, hi2 = fit_affine("k", noisy).interval(3)
    assert (hi2 - lo2) > (hi1 - lo1)


# ── decomposition ───────────────────────────────────────────────────────

def test_affine_decomposes_into_fixed_and_marginal():
    m = fit_affine("k", [rec("k", x, 5000 + 100 * x) for x in (1, 2, 4)])
    fixed, marginal = m.decompose(3)
    assert fixed == pytest.approx(5000)
    assert marginal == pytest.approx(300)


def test_constant_is_all_fixed():
    m = fit_constant("k", [rec("k", x, 5000) for x in (1, 2)])
    fixed, marginal = m.decompose(3)
    assert fixed == pytest.approx(5000)
    assert marginal == 0.0


def test_proportional_is_all_marginal():
    m = fit_proportional("k", [rec("k", x, 300 * x) for x in (1, 2)])
    fixed, marginal = m.decompose(2)
    assert fixed == 0.0
    assert marginal == pytest.approx(600)


# ── noise floor / backtest ──────────────────────────────────────────────

def test_noise_floor_none_without_replicates():
    assert noise_floor([rec("k", 1, 10), rec("k", 2, 20)]) is None


def test_noise_floor_measures_replicate_spread():
    rs = [rec("k", 1, 1000), rec("k", 1, 1100), rec("k", 2, 2000), rec("k", 2, 2100)]
    floor = noise_floor(rs)
    assert floor is not None and 0 < floor < 0.2


def test_backtest_reports_skill_against_the_floor():
    rs = ([rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)]
          + [rec("k", 2, 5210), rec("k", 2, 5190)])
    rep = backtest(rs)["k"]
    assert rep.skill_ratio is not None
    assert rep.verdict


def test_backtest_verdict_when_no_replicates():
    rs = [rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)]
    assert "cannot separate" in backtest(rs)["k"].verdict


def test_learning_curve_grows_with_n():
    rs = [rec("k", x, 5000 + 100 * x) for x in (1, 2, 3, 4, 5, 6)]
    curve = learning_curve("k", rs)
    assert [n for n, _ in curve] == [3, 4, 5, 6]


# ── the learning loop ───────────────────────────────────────────────────

def test_store_fits_lazily_and_refits_after_new_data():
    s = LearningStore()
    s.observe_many([rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)])
    first = s.model_for("k")
    assert first.form == "affine"
    s.observe(rec("k", 16, 5000 + 100 * 16))
    assert s.model_for("k").scope_max == 16       # refitted, wider regime


def test_first_observation_has_no_prior_to_score_against():
    s = LearningStore()
    assert s.observe(rec("k", 1, 100)) is None


def test_drift_fires_when_reality_shifts():
    s = LearningStore()
    s.observe_many([rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)])
    # the world got twice as expensive
    reports = s.observe_many([rec("k", x, 2 * (5000 + 100 * x)) for x in (2, 4)])
    d = reports["k"]
    assert d.should_refit
    assert d.bias > 0                 # under-predicting
    assert "under" in d.verdict


def test_drift_stays_quiet_when_the_model_still_holds():
    s = LearningStore()
    s.observe_many([rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)])
    reports = s.observe_many([rec("k", 3, 5300), rec("k", 5, 5500)])
    assert reports["k"].verdict == "stable"
    assert not reports["k"].should_refit


def test_drift_flags_scope_outside_the_fitted_range():
    s = LearningStore()
    s.observe_many([rec("k", x, 5000 + 100 * x) for x in (1, 2, 4)])
    reports = s.observe_many([rec("k", 40, 9000)])
    assert reports["k"].should_refit


def test_score_against_returns_none_for_other_kinds():
    m = fit_affine("k", [rec("k", x, 5000 + 100 * x) for x in (1, 2, 4)])
    assert score_against(m, [rec("other", 1, 100)]) is None


def test_evidence_splits_by_provenance():
    s = LearningStore()
    s.observe(rec("k", 1, 100, Provenance.PROBE))
    s.observe(rec("k", 2, 200, Provenance.SYNTHETIC))
    ev = s.evidence("k")
    assert ev["probe"] == 1 and ev["synthetic"] == 1


def test_unknown_kind_registers_itself():
    s = LearningStore()
    s.observe(rec("brand_new_kind", 1, 100))
    assert "brand_new_kind" in s.kinds()


def test_store_report_mentions_the_chosen_form():
    s = seeded_store()
    text = s.report()
    assert "comprehension" in text and "affine" in text


# ── plans ───────────────────────────────────────────────────────────────

def test_plan_surfaces_unmodelled_kinds():
    s = seeded_store()
    plan = WorkPlan("p").add("comprehension", 3).add("never_measured", 2)
    fc = PlanForecaster(s).forecast(plan)
    assert fc.unmodelled == ("never_measured",)
    assert not fc.is_complete
    assert len(fc.line_items) == 1


def test_plan_flags_extrapolation():
    s = seeded_store()
    fc = PlanForecaster(s).forecast(WorkPlan("p").add("comprehension", 400))
    assert fc.extrapolated
    assert fc.extrapolated[0].extrapolation == pytest.approx(50.0)
    assert "outside the fitted range" in fc.summary()


def test_plan_interval_grows_with_sqrt_of_count():
    s = seeded_store()
    one = PlanForecaster(s).forecast(WorkPlan("a").add("comprehension", 3, count=1))
    four = PlanForecaster(s).forecast(WorkPlan("b").add("comprehension", 3, count=4))
    assert four.total_sigma == pytest.approx(2 * one.total_sigma)


def test_plan_totals_scale_linearly_with_count():
    s = seeded_store()
    one = PlanForecaster(s).forecast(WorkPlan("a").add("comprehension", 3, count=1))
    four = PlanForecaster(s).forecast(WorkPlan("b").add("comprehension", 3, count=4))
    assert four.total_tokens == pytest.approx(4 * one.total_tokens)


def test_plan_cost_at_rate():
    s = seeded_store()
    fc = PlanForecaster(s).forecast(WorkPlan("p").add("comprehension", 3))
    assert fc.cost_at_rate(3.0) == pytest.approx(fc.total_tokens * 3.0 / 1e6)


# ── the measured probe suite ────────────────────────────────────────────

def test_probe_suite_is_reproducible_and_graded():
    kinds = {p.kind for p in PROBE_SUITE}
    assert kinds == {"comprehension", "code_write"}
    scopes = sorted({p.scope for p in PROBE_SUITE if p.kind == "comprehension"})
    assert len(scopes) >= 3           # graded scope is what makes a slope fittable
    assert all(p.prompt for p in PROBE_SUITE)


def test_all_shipped_measurements_are_real_runs():
    assert MEASURED
    assert all(r.provenance is Provenance.PROBE for r in MEASURED)


def test_replicates_exist_so_noise_is_estimable():
    n, mean, sd = replicate_spread("comprehension", 3)
    assert n >= 2 and mean > 0 and sd > 0


def test_measured_data_selects_affine_for_comprehension():
    s = seeded_store()
    assert s.model_for("comprehension").form == "affine"


def test_measured_data_rejects_the_old_multiplicative_assumption():
    """The 1x/2x/4x model is `proportional`. It must lose, badly, on real data."""
    s = seeded_store()
    for kind in ("comprehension", "code_write"):
        scores = s.selection_for(kind).scores
        assert scores["proportional"] > 0.4          # >40% error
        assert scores[s.selection_for(kind).form] < 0.10


def test_fitted_models_sit_near_the_noise_floor():
    reps = backtest(MEASURED)
    for kind, rep in reps.items():
        assert rep.skill_ratio is not None
        assert rep.skill_ratio < 2.0, f"{kind} leaves real signal unmodelled"


# ── out-of-sample validation: composition ───────────────────────────────

def test_fitted_models_predict_the_unseen_composition_experiment():
    """The strongest claim in the package, pinned.

    The per-kind models are fitted only on single-kind runs. The batched runs
    are held out entirely. If the fixed/marginal split is real, it should
    predict what batching costs — and it does, to inside the noise floor.
    """
    store = seeded_store()
    plan = WorkPlan("replica").add("comprehension", 3).add("code_write", 3)
    predicted = PlanForecaster(store).compare_batching(plan)
    measured = composition_evidence()

    err = (abs(predicted["batched_single_agent"] - measured["batched_mean"])
           / measured["batched_mean"])
    assert err < 0.06, f"batched prediction off by {err:.1%}"

    err_sep = (abs(predicted["separate_agents"] - measured["separate_sum"])
               / measured["separate_sum"])
    assert err_sep < 0.06, f"separate prediction off by {err_sep:.1%}"


def test_batching_saves_rather_than_surcharges():
    """The measured sign of the composition effect, pinned against regression."""
    ev = composition_evidence()
    assert ev["saving"] > 0.3, "batching measured as a large saving, not a surcharge"
    assert ev["ratio"] < 0.7


def test_composition_measurements_are_held_out_of_the_per_kind_fits():
    """A batched run must never be fitted as if it were a single-kind record."""
    batched_tokens = {c.tokens for c in COMPOSITION_MEASURED}
    assert not any(r.tokens in batched_tokens for r in MEASURED)


def test_boot_cost_is_pooled_across_kinds():
    store = seeded_store()
    boot = PlanForecaster(store).boot_cost()
    assert boot is not None
    assert 30_000 < boot < 45_000       # the measured per-invocation floor
