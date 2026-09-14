"""The regression backbone.

One campaign is executed twice against the same design, changing only whether the
simulated runtime *has* a per-brick cost structure. The pipeline must reach opposite
conclusions -- and the second arm is the one that matters, because a system that can only
say "yes" is not measuring anything.
"""

import pytest

from assay.costmodel import CostModel, assess_identifiability
from assay.evidence import EvidenceClass
from assay.features import FORMS
from assay.harness.mock_adapter import MockSpec
from assay.metrics import score
from assay.select import cross_validate, select_form, shuffle_target
from assay.verdict import Outcome, evaluate

from tests.assay.conftest import boot_estimate

BAR = 15.0


def fittable(runs):
    return [r for r in runs if r.fittable]


def billable_scale(pricing, spec: MockSpec) -> float:
    """The simulator emits total tokens; we predict billable units. This is the ratio."""
    s = spec.completion_share
    return (1.0 - s) + pricing.ratio * s


# -- shared invariants ------------------------------------------------------------


@pytest.mark.parametrize("arm", ["brick", "null"])
def test_boot_comes_from_the_null_probes_and_matches_the_simulator(
    arm, brick_runs, null_runs, pricing
):
    runs = brick_runs if arm == "brick" else null_runs
    boot = boot_estimate(runs)
    expected = MockSpec().boot * billable_scale(pricing, MockSpec())
    assert boot == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("arm", ["brick", "null"])
def test_the_design_is_identifiable(arm, brick_runs, null_runs, vocab):
    runs = fittable(brick_runs if arm == "brick" else null_runs)
    ident = assess_identifiability(runs, "bytes_per_brick", vocab)
    assert ident.ok, f"design cannot support per-brick coefficients: {ident.problems()}"
    assert abs(ident.spearman_units_bytes) <= 0.20


@pytest.mark.parametrize("arm", ["brick", "null"])
def test_the_constant_predictor_flatters_itself_on_total_error(arm, brick_runs, null_runs):
    runs = fittable(brick_runs if arm == "brick" else null_runs)
    boot = boot_estimate(brick_runs if arm == "brick" else null_runs)
    actual = [r.billable_units for r in runs]
    card = score(actual, [boot] * len(actual), boot)
    assert card.vwape == pytest.approx(100.0), "boot explains none of the variable portion"
    assert card.mape_total < card.vwape, "yet total error looks far kinder"


# -- arm 1: the runtime really does price bricks ----------------------------------


def test_brick_arm_recovers_the_planted_coefficients(brick_runs, vocab, pricing):
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot)
    scale = billable_scale(pricing, MockSpec())

    assert model.intercept == pytest.approx(MockSpec().boot * scale, rel=0.10)
    assert model.bytes_coef == pytest.approx(MockSpec().bytes_slope * scale, rel=0.10)
    for brick, planted in MockSpec().marginals.items():
        if planted > 500:  # the ones the design has power to see
            assert model.marginals[brick] == pytest.approx(planted * scale, rel=0.15), brick


def test_brick_arm_chooses_a_decoder_and_clears_the_lift_bar(brick_runs, vocab):
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    sel = select_form(runs, vocab, boot, bar_pct=BAR, iters=400)

    assert sel.bricks_helped, f"expected a decoder, selection chose {sel.chosen}"
    lift = sel.decoder_lift()
    assert lift is not None and lift.passed
    assert lift.lift_pct >= BAR
    assert lift.bootstrap_lo > 0.0


def test_brick_arm_beats_the_size_and_count_baseline_specifically(brick_runs, vocab):
    """Beating `boot` is trivial. `bytes_units` is the baseline that matters -- it needs
    no vocabulary, no encoder and no labelling."""
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    baseline = cross_validate(runs, "bytes_units", vocab, boot)
    decoder = cross_validate(runs, "bytes_per_brick", vocab, boot)
    assert decoder.pooled.vwape < baseline.pooled.vwape


# -- arm 2: the negative control ---------------------------------------------------


def test_null_arm_refuses_to_find_bricks_that_are_not_there(null_runs, vocab):
    """The most important test in the suite.

    The simulated runtime has no per-brick cost term at all. If selection still crowns a
    decoder here, then every positive result this pipeline can produce is an artefact of
    its own wiring.
    """
    runs = fittable(null_runs)
    boot = boot_estimate(null_runs)
    sel = select_form(runs, vocab, boot, bar_pct=BAR, iters=400)

    assert not sel.bricks_helped, f"invented a brick effect: chose {sel.chosen}"
    assert FORMS[sel.chosen].role == "baseline"
    assert FORMS[sel.chosen].uses_bytes, "the truth here is bytes; selection should say so"


def test_null_arm_yields_narrow_never_feasible(null_runs, vocab):
    runs = fittable(null_runs)
    boot = boot_estimate(null_runs)
    sel = select_form(runs, vocab, boot, bar_pct=BAR, iters=400)
    lift = sel.decoder_lift()

    v = evaluate(
        {
            "M5b_cv_lift_pct": lift.lift_pct if lift else 0.0,
            "M15_identifiable": 1.0,
        },
        evidence_class=EvidenceClass.PIPELINE_ONLY,
    )
    assert v.outcome is not Outcome.FEASIBLE
    assert v.outcome is Outcome.NARROW, "no brick signal is a Narrow result, not a dead project"


def test_null_arm_still_predicts_cost_well_from_size_alone(null_runs, vocab):
    """Narrow has to mean something. Here the estimator is genuinely useful."""
    runs = fittable(null_runs)
    boot = boot_estimate(null_runs)
    cv = cross_validate(runs, "bytes", vocab, boot)
    assert cv.pooled.mape_total < 5.0
    assert cv.pooled.vwape < 20.0


# -- controls ---------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["brick", "null"])
def test_shuffling_the_target_destroys_every_form(arm, brick_runs, null_runs, vocab):
    runs = fittable(brick_runs if arm == "brick" else null_runs)
    boot = boot_estimate(brick_runs if arm == "brick" else null_runs)
    shuffled = shuffle_target(runs, seed=7)
    sel = select_form(shuffled, vocab, boot, bar_pct=BAR, iters=200)
    assert not sel.bricks_helped, "a decoder won on noise; the evaluation is leaking"


def test_two_fits_of_the_same_data_are_identical(brick_runs, vocab):
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    a = CostModel.fit(runs, "bytes_per_brick", vocab, boot)
    b = CostModel.fit(runs, "bytes_per_brick", vocab, boot)
    assert a.coefficients == b.coefficients
    assert a.data_sha256 == b.data_sha256


def test_nnls_hides_no_negative_marginal_from_the_report(brick_runs, vocab):
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    model = CostModel.fit(runs, "bytes_per_brick_nnls", vocab, boot)
    assert all(v >= 0.0 for v in model.marginals.values())
    assert set(model.marginals_ols_unclamped) == set(vocab.names), (
        "the unconstrained sign pattern must stay visible even when it is not used"
    )


def test_a_model_fitted_on_mock_runs_is_stamped_pipeline_only(brick_runs, vocab):
    model = CostModel.fit(fittable(brick_runs), "bytes", vocab, boot_estimate(brick_runs))
    assert model.evidence_class is EvidenceClass.PIPELINE_ONLY


def test_model_round_trips_and_reproduces_its_predictions(brick_runs, vocab, tmp_path):
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot)
    path = tmp_path / "coefficients.json"
    model.save(path)
    again = CostModel.load(path)
    for run in runs[:20]:
        assert again.predict_run(run) == pytest.approx(model.predict_run(run), rel=1e-6)


# -- comparability of the ladder ---------------------------------------------------


def test_zero_units_and_zero_bytes_returns_the_intercept(brick_runs, vocab):
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot)
    assert model.predict(vocab.zero_units(), 0) == pytest.approx(model.intercept)


def test_lift_test_refuses_forms_that_were_not_scored_on_the_same_rows():
    """A form that failed to fit on one fold is not row-comparable with one that did.

    Zipping them pairs a prediction about one document against an actual from another, and
    the resulting 'lift' is an artefact of the misalignment rather than a measurement.
    """
    from assay.errors import MisalignedFoldsError
    from assay.select import CVResult, lift_test

    complete = CVResult(
        form="bytes_units", role="baseline", n_params=3, n_scored=3, pooled=None,
        mean_fold_vwape=0.0, actual=[10.0, 20.0, 30.0], predicted=[11.0, 21.0, 31.0],
        groups=["doc-a", "doc-b", "doc-c"], failed_folds=0,
    )
    partial = CVResult(
        form="bytes_per_brick", role="decoder", n_params=11, n_scored=2, pooled=None,
        mean_fold_vwape=0.0, actual=[20.0, 30.0], predicted=[20.0, 30.0],
        groups=["doc-b", "doc-c"], failed_folds=1,
    )

    with pytest.raises(MisalignedFoldsError, match="not row-comparable"):
        lift_test(complete, partial, boot=0.0, iters=10)


def test_an_incomparable_challenger_is_never_promoted(brick_runs, vocab, monkeypatch):
    """Selection must not climb the ladder on a comparison it could not make."""
    import assay.select as select_mod
    from assay.errors import MisalignedFoldsError

    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    real = select_mod.lift_test

    def refuse_decoders(incumbent, challenger, *args, **kwargs):
        if FORMS[challenger.form].role == "decoder":
            raise MisalignedFoldsError("not row-comparable (simulated fold failure)")
        return real(incumbent, challenger, *args, **kwargs)

    monkeypatch.setattr(select_mod, "lift_test", refuse_decoders)
    sel = select_mod.select_form(runs, vocab, boot, bar_pct=BAR, iters=200)

    assert not sel.bricks_helped, "promoted a decoder without a valid comparison"
    assert any("not row-comparable" in n for n in sel.notes)


def test_every_cross_validated_row_records_the_group_it_was_held_out_from(brick_runs, vocab):
    runs = fittable(brick_runs)
    cv = cross_validate(runs, "bytes_units", vocab, boot_estimate(brick_runs))
    assert len(cv.groups) == len(cv.actual) == cv.n_scored
    assert set(cv.groups) <= {r.context_files[0] for r in runs if r.context_files} | {"_nocontext"}


# -- the identifiability guard has to fire, not merely compile ---------------------


def _confounded_runs(brick_runs, vocab):
    """The same runs, rewritten so unit count is a deterministic function of size.

    This is the design failure the real campaign is built to avoid: `design_campaign`
    rotates the unit level against the brick index *and* the band precisely so that "more
    bricks" and "more bytes" are not the same column. Here they are, and the guard must
    say so -- a per-brick coefficient fitted on this data is an arbitrary split of one
    effect no matter how well it fits.
    """
    import copy

    out = []
    for r in brick_runs:
        if not r.fittable or not r.context_files:
            continue
        clone = copy.deepcopy(r)
        n = max(1, clone.context_bytes // 4_000)
        clone.units = {name: (n if name == vocab.names[0] else 0) for name in vocab.names}
        out.append(clone)
    return out


def test_the_guard_refuses_a_design_where_units_and_bytes_are_the_same_column(
    brick_runs, vocab
):
    confounded = _confounded_runs(brick_runs, vocab)
    ident = assess_identifiability(confounded, "bytes_per_brick", vocab)

    assert not ident.ok, "a perfectly confounded design was reported as identifiable"
    assert abs(ident.spearman_units_bytes) > 0.20
    assert ident.problems(), "an unidentifiable design must say what is wrong with it"


def test_a_confounded_design_caps_the_verdict_at_narrow(brick_runs, vocab):
    """M15 carries this. It caps rather than stops: the size baseline is still a perfectly
    good estimator, it just cannot honestly be broken down per brick."""
    confounded = _confounded_runs(brick_runs, vocab)
    ident = assess_identifiability(confounded, "bytes_per_brick", vocab)

    v = evaluate(
        {"M15_identifiable": 1.0 if ident.ok else 0.0},
        evidence_class=EvidenceClass.PIPELINE_ONLY,
    )
    assert v.outcome is Outcome.NARROW
    assert not v.stopping_gates()


# -- the size baseline is allowed to be curved -------------------------------------


def test_the_curved_size_baseline_is_a_baseline_and_is_always_scored(brick_runs, vocab):
    """A decoder must not be credited with lift it earned only by being non-linear.

    Brick counts correlate with size, so a per-brick form can imitate a diminishing-returns
    curve. `log_bytes_units` is the rung that takes that explanation away, and being a
    baseline it can only ever raise the bar.
    """
    from assay.features import LADDER

    assert FORMS["log_bytes_units"].role == "baseline"
    assert "log_bytes_units" in LADDER

    runs = fittable(brick_runs)
    sel = select_form(runs, vocab, boot_estimate(brick_runs), bar_pct=BAR, iters=200)
    assert "log_bytes_units" in sel.results, "the rung must be scored, not merely declared"


def test_the_curved_baseline_can_only_make_the_decoder_bar_harder(brick_runs, vocab):
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    linear = cross_validate(runs, "bytes_units", vocab, boot)
    curved = cross_validate(runs, "log_bytes_units", vocab, boot)

    sel = select_form(runs, vocab, boot, bar_pct=BAR, iters=200)
    best = min(linear.pooled.vwape, curved.pooled.vwape)
    assert sel.results[sel.best_baseline].pooled.vwape <= best + 1e-9, (
        "the best baseline must be at least as good as the curved rung"
    )


def test_the_curved_rung_keeps_the_null_probes_in_the_design(vocab):
    """log1p, not log. A null probe mounts zero bytes and pins the intercept; log(0) would
    drop exactly the rows that measure the start-up toll."""
    from assay.features import feature_row

    row = feature_row("log_bytes_units", vocab.zero_units(), 0, vocab)
    assert all(v == v and abs(v) != float("inf") for v in row), row
    assert row == [1.0, 0.0, 0.0, 0.0]


def test_the_brick_arm_still_beats_the_curved_baseline(brick_runs, vocab):
    """The positive control has to survive the harder bar, or the bar is simply wrong."""
    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    curved = cross_validate(runs, "log_bytes_units", vocab, boot)
    decoder = cross_validate(runs, "bytes_per_brick", vocab, boot)
    assert decoder.pooled.vwape < curved.pooled.vwape
