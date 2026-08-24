"""Tests for the fitted calibration layer: taxonomy, cost models, learning, plans."""

import math

from pathlib import Path

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

# Resolved from this file's own location so the suite runs wherever the
# checkout lives, rather than a path baked in from where it was written.
REPO_ROOT = str(Path(__file__).resolve().parents[1])


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
    by_form = sel.scores_for_signal("scope")
    assert by_form["affine"] < by_form["proportional"]
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
    plan = (WorkPlan("p").add("comprehension", 3, bytes=15_216)
            .add("never_measured", 2))
    fc = PlanForecaster(s).forecast(plan)
    assert fc.unmodelled == ("never_measured",)
    assert not fc.is_complete
    assert len(fc.line_items) == 1


def test_plan_flags_extrapolation():
    s = seeded_store()
    model = s.model_for("comprehension")
    far = model.scope_max * 10
    fc = PlanForecaster(s).forecast(
        WorkPlan("p").add("comprehension", 400, bytes=far))
    assert fc.extrapolated
    assert fc.extrapolated[0].extrapolation == pytest.approx(10.0)
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
    assert kinds == {"comprehension", "code_write", "test_write",
                     "code_review", "docs"}
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
    for kind in s.kinds():
        sel = s.selection_for(kind)
        by_form = sel.scores_for_signal(sel.signal)
        assert by_form["proportional"] > 0.4         # >40% error
        assert by_form[sel.form] < 0.10


def test_fitted_models_sit_near_the_noise_floor():
    reps = backtest(MEASURED)
    scored = {k: r for k, r in reps.items() if r.skill_ratio is not None}
    assert scored, "no kind had replicates to score against"
    for kind, rep in scored.items():
        assert rep.skill_ratio < 2.0, f"{kind} leaves real signal unmodelled"


def test_kinds_without_replicates_say_so_rather_than_guessing():
    reps = backtest(MEASURED)
    for rep in reps.values():
        if rep.floor is None:
            assert "cannot separate" in rep.verdict


# ── out-of-sample validation: composition ───────────────────────────────

def test_fitted_models_predict_the_unseen_composition_experiment():
    """The strongest claim in the package, pinned.

    The per-kind models are fitted only on single-kind runs. The batched runs
    are held out entirely. If the fixed/marginal split is real, it should
    predict what batching costs — and it does, to inside the noise floor.
    """
    store = seeded_store()
    plan = (WorkPlan("replica")
            .add("comprehension", 3, bytes=15_216)   # the files that probe read
            .add("code_write", 3))
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


def test_boot_cost_is_the_tightest_bound_not_the_average():
    """Intercepts bound the shared floor from above; the minimum is the bound."""
    store = seeded_store()
    f = PlanForecaster(store)
    boot = f.boot_cost()
    intercepts = [f._store.model_for(k).decompose(
        f._store.model_for(k).scope_min)[0] for k in store.kinds()]
    intercepts = [i for i in intercepts if i > 0]
    assert boot == pytest.approx(min(intercepts))
    assert boot < sum(intercepts) / len(intercepts)
    assert 30_000 < boot < 45_000       # the measured per-invocation floor


# ── multi-signal selection: the data picks the explanatory variable ─────

def test_selection_picks_the_signal_that_actually_predicts():
    """Two candidate signals; only one carries the relationship."""
    rs = [ScopedRecord("k", i, 1000 + 10 * b, provenance=Provenance.PROBE,
                       signals={"bytes": b})
          for i, b in enumerate([100, 400, 900, 1600, 2500], start=1)]
    sel = select_model("k", rs)
    assert sel.signal == "bytes"
    assert sel.form == "affine"


def test_selection_prefers_plain_scope_when_signals_tie():
    rs = [ScopedRecord("k", x, 1000 + 100 * x, provenance=Provenance.PROBE,
                       signals={"mirror": x})
          for x in (1, 2, 4, 8)]
    assert select_model("k", rs).signal == "scope"


def test_measured_comprehension_chooses_bytes_over_file_count():
    """The cross-repo finding, discovered by the selector rather than asserted."""
    s = seeded_store()
    sel = s.selection_for("comprehension")
    assert sel.signal == "bytes"
    by_signal = {k.split("@")[1]: v for k, v in sel.scores.items()
                 if k.startswith("affine@")}
    assert by_signal["bytes"] < by_signal["scope"] / 3


def test_probe_suite_spans_three_repositories():
    from token_yield.probes import repos
    assert set(repos()) == {"harness-dose", "requests", "click"}
    assert len({r.repo for r in MEASURED if r.kind == "comprehension"}) == 3


# ── the guards that keep a budget honest ───────────────────────────────

def test_plan_names_a_signal_it_cannot_supply():
    """A model priced by bytes must not silently read zero bytes."""
    s = seeded_store()
    fc = PlanForecaster(s).forecast(WorkPlan("p").add("comprehension", 3))
    assert fc.missing_signal == (("comprehension", "bytes"),)
    assert not fc.is_complete
    assert fc.total_tokens == 0
    assert "did not supply" in fc.summary()


def test_supplying_the_signal_prices_the_item():
    s = seeded_store()
    fc = PlanForecaster(s).forecast(
        WorkPlan("p").add("comprehension", 3, bytes=15_216))
    assert fc.is_complete
    assert fc.total_tokens > 0


def test_noise_floor_does_not_pool_different_repos():
    """3 files of one repo is not a replicate of 3 files of another."""
    rs = [ScopedRecord("comprehension", 3, 42_000, provenance=Provenance.PROBE,
                       signals={"bytes": 15_000}, repo="a"),
          ScopedRecord("comprehension", 3, 43_000, provenance=Provenance.PROBE,
                       signals={"bytes": 15_000}, repo="a"),
          ScopedRecord("comprehension", 3, 140_000, provenance=Provenance.PROBE,
                       signals={"bytes": 270_000}, repo="b")]
    floor = noise_floor(rs)
    assert floor is not None and floor < 0.10      # not inflated by repo b


def test_measured_noise_floor_is_a_few_percent():
    assert 0.01 < noise_floor(MEASURED) < 0.10


def test_saturated_fit_does_not_claim_certainty():
    """Two points fit a line exactly; the interval must not be zero-width."""
    m = fit_affine("k", [rec("k", 1, 1000), rec("k", 2, 2000)])
    assert m.saturated
    lo, hi = m.interval(3)
    assert hi - lo > 0.5 * m.predict(3)


def test_unsaturated_fit_reports_real_spread():
    m = fit_affine("k", [rec("k", x, 5000 + 100 * x) for x in (1, 2, 4, 8)])
    assert not m.saturated


def test_loo_skips_a_failed_fold_rather_than_dropping_the_form():
    """One degenerate split must not remove a form from the comparison."""
    rs = [rec("k", 1, 1000), rec("k", 1, 1010), rec("k", 4, 4000), rec("k", 8, 8000)]
    assert loo_mape("k", rs, "affine") is not None


def test_power_model_survives_a_zero_input():
    m = fit_power("k", [rec("k", x, int(100 * x ** -0.5)) for x in (1, 4, 9)])
    assert m.predict(0.0) == float("inf")           # flagged, not a crash


def test_plan_interval_is_never_tighter_than_the_noise_floor():
    s = seeded_store()
    fc = PlanForecaster(s).forecast(WorkPlan("p").add("code_review", 3))
    lo, hi = fc.interval()
    assert (hi - lo) / fc.total_tokens > 0.05       # not the raw sigma=1 fit


# ── mining feeds the loop back to measurement ──────────────────────────

def test_coverage_backlog_names_kinds_we_have_not_measured():
    """The framework must say what it cannot price, ranked by how much it matters."""
    from token_yield.mine import coverage, mine_repo
    mined = mine_repo(REPO_ROOT, limit=100)
    if not mined:
        pytest.skip("no history to mine")
    store = seeded_store()
    rep = coverage(mined, store.kinds())
    assert 0.0 <= rep.covered_share <= 1.0
    for kind, share in rep.backlog:
        assert kind not in store.kinds()
        assert share > 0


def test_acting_on_the_backlog_raises_coverage():
    """Measuring a kind on the backlog must move it out of the backlog."""
    from token_yield.mine import coverage, mine_repo
    mined = mine_repo(REPO_ROOT, limit=100)
    if not mined:
        pytest.skip("no history to mine")
    before = coverage(mined, ["comprehension"])
    after = coverage(mined, ["comprehension", "docs"])
    assert after.covered_share >= before.covered_share
    assert "docs" not in [k for k, _ in after.backlog]
