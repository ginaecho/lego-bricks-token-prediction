"""The end-to-end loop, and the guarantees that only hold when it is run as a whole.

These tests exist because the loop used to live only in `conftest.py`. A reader who
cannot reproduce quote-to-invoice without reading the fixtures cannot check the claim,
so the ordering is now library code and is pinned here.
"""

from __future__ import annotations

import json

import pytest

from assay.costmodel import CostModel
from assay.evidence import EvidenceClass
from assay.invoice import make_invoice
from assay.pipeline import Fitted, Priced, boot_estimate, fit_pipeline
from assay.quote import make_quote
from assay.schemas import Tier
from assay.select import FORMS

BAR = 15.0


# -- the start-up toll has exactly one definition ---------------------------------


def test_boot_is_the_median_of_the_null_probes_and_nothing_else(brick_runs):
    from statistics import median

    nulls = [r.billable_units for r in brick_runs if r.tier is Tier.NULL and r.accepted]
    assert boot_estimate(brick_runs) == median(nulls)


def test_boot_refuses_to_guess_when_no_null_probe_was_run(brick_runs):
    """The cheapest run is not the empty task; it is a cheap task that still did work."""
    without_nulls = [r for r in brick_runs if r.tier is not Tier.NULL]
    with pytest.raises(ValueError, match="never measured"):
        boot_estimate(without_nulls)


def test_boot_ignores_excluded_null_probes(brick_runs):
    import copy

    runs = copy.deepcopy(brick_runs)
    nulls = [r for r in runs if r.tier is Tier.NULL]
    assert len(nulls) >= 2
    nulls[0].billable_units = 10_000_000.0
    nulls[0].status = type(nulls[0].status).FAILED
    assert boot_estimate(runs) < 1_000_000.0


# -- the interval is calibrated on data the model did not fit ---------------------


def test_the_band_is_calibrated_out_of_fold_not_in_sample(brick_runs, vocab):
    """An in-sample band is narrower than the truth, and wrong in the flattering direction."""
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    held = fit.selection.results[fit.selection.chosen]
    assert fit.conformal.n_calibration == len(held.actual)
    assert fit.conformal.n_calibration > 0
    # Out-of-fold rows are the ones that survived cross-validation, which is not the same
    # set as the rows the final model was fitted on.
    assert fit.conformal.n_calibration <= fit.model.n_fitted


def test_the_band_never_quotes_below_the_measured_empty_task(brick_runs, vocab):
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    lo, _ = fit.conformal.interval(fit.boot * 0.5)
    assert lo >= fit.boot


# -- per-brick detail follows the verdict, not the fit ----------------------------


def test_the_brick_arm_earns_per_brick_pricing(brick_runs, vocab):
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    assert fit.selection.bricks_helped
    assert fit.show_per_brick


def test_the_null_arm_fits_a_model_but_withholds_per_brick_prices(null_runs, vocab):
    """The estimator may still ship. The per-brick prices may not: nothing established them."""
    fit = fit_pipeline(null_runs, vocab, bar_pct=BAR, iters=200)
    assert not fit.selection.bricks_helped
    assert not fit.show_per_brick
    assert FORMS[fit.selection.chosen].role != "decoder"


def test_a_quote_from_the_null_arm_carries_no_brick_detail(null_runs, vocab, pricing):
    fit = fit_pipeline(null_runs, vocab, bar_pct=BAR, iters=200)
    q = make_quote(
        "REQ", {"Extract": 3}, 9_000,
        model=fit.model, conformal=fit.conformal, pricing=pricing,
        show_per_brick=fit.show_per_brick,
    )
    assert q.outlier_flags == ()
    assert "_hidden" in q.as_dict()["units"]


# -- a reloaded fit must not forget the verdict -----------------------------------


def test_a_reloaded_fit_remembers_that_per_brick_prices_were_not_earned(
    null_runs, vocab, tmp_path
):
    """The coefficients survive a round trip either way; the entitlement to quote them
    is a conclusion of the ladder, and it has to survive too."""
    fit = fit_pipeline(null_runs, vocab, bar_pct=BAR, iters=200)
    fit.save(tmp_path)
    priced = Priced.load(tmp_path)
    assert priced.show_per_brick is False
    assert priced.chosen_form == fit.selection.chosen
    assert priced.boot == pytest.approx(fit.boot, abs=0.05)


def test_a_reloaded_fit_round_trips_the_brick_arm(brick_runs, vocab, tmp_path):
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    fit.save(tmp_path)
    priced = Priced.load(tmp_path)
    assert priced.show_per_brick is True
    assert priced.model.predict({"Extract": 2}, 9_000) == pytest.approx(
        fit.model.predict({"Extract": 2}, 9_000)
    )


def test_a_model_file_without_its_verdict_is_refused(brick_runs, vocab, tmp_path):
    """A bare model.json would happily publish per-brick prices nobody established."""
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    fit.save(tmp_path)
    (tmp_path / "fit.json").unlink()
    with pytest.raises(FileNotFoundError, match="does not say whether"):
        Priced.load(tmp_path)


# -- simulated evidence can never be relabelled as feasibility --------------------


def test_a_simulated_fit_reports_itself_as_unable_to_move_a_gate(brick_runs, vocab):
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    assert fit.evidence_class is EvidenceClass.PIPELINE_ONLY
    assert not fit.evidence_class.can_move_a_gate
    assert "cannot move a gate" in fit.render()


def test_a_mixed_run_set_is_downgraded_not_averaged(brick_runs, vocab):
    """One simulated row among real ones makes the whole fit simulated."""
    import copy

    runs = copy.deepcopy(brick_runs)
    for r in runs:
        r.evidence_class = EvidenceClass.FEASIBILITY
    runs[0].evidence_class = EvidenceClass.PIPELINE_ONLY
    fit = fit_pipeline(runs, vocab, bar_pct=BAR, iters=200)
    assert fit.evidence_class is EvidenceClass.PIPELINE_ONLY


# -- the invoice still reconciles when it comes out of the loop -------------------


def test_the_loop_produces_an_invoice_that_sums_to_the_measured_total(
    brick_runs, vocab, pricing
):
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    run = next(r for r in brick_runs if r.accepted and any(r.units.values()))
    q = make_quote(
        run.run_id, run.units, run.context_bytes,
        model=fit.model, conformal=fit.conformal, pricing=pricing,
        show_per_brick=fit.show_per_brick,
    )
    inv = make_invoice(run, q, fit.model, boot=fit.boot, show_per_brick=fit.show_per_brick)
    assert inv.reconciles
    assert inv.sum_check == pytest.approx(run.billable_units)


def test_a_negative_brick_price_is_reported_not_clamped(brick_runs, vocab, pricing):
    """A brick cannot make a run cheaper. If the fit says it does, say so on the bill."""
    fit = fit_pipeline(brick_runs, vocab, bar_pct=BAR, iters=200)
    run = next(r for r in brick_runs if r.accepted and any(r.units.values()))
    brick = next(b for b, c in run.units.items() if c)

    model = CostModel(
        form=fit.model.form,
        columns=fit.model.columns,
        coefficients=tuple(
            -abs(c) if fit.model.columns[i] == brick else c
            for i, c in enumerate(fit.model.coefficients)
        ),
        boot=fit.model.boot,
        brick_names=fit.model.brick_names,
        n_fitted=fit.model.n_fitted,
        data_sha256=fit.model.data_sha256,
        evidence_class=fit.model.evidence_class,
    )
    q = make_quote(
        run.run_id, run.units, run.context_bytes,
        model=model, conformal=fit.conformal, pricing=pricing,
    )
    inv = make_invoice(run, q, model, boot=fit.boot)
    assert any("negative price" in n for n in inv.notes)
    assert any(line.units < 0 for line in inv.lines if line.item.startswith(brick))
    # Reported, but not smoothed away: the bill still sums to what was measured.
    assert inv.reconciles
