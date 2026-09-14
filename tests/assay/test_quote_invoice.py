import math

import pytest

from assay.conformal import Conformal, Nonconformity, calibrate, choose_nonconformity
from assay.costmodel import BootModel, CostModel
from assay.evidence import EvidenceClass
from assay.features import run_group
from assay.invoice import make_invoice
from assay.metrics import coverage
from assay.quote import make_quote, read_quotes, append_quote
from assay.select import NO_GROUP

from tests.assay.conftest import boot_estimate


@pytest.fixture(scope="module")
def fitted(brick_runs, vocab):
    """Fit on most document groups, calibrate the interval on a disjoint set of them.

    Conformal calibration needs its own fold -- residuals from the training data are
    optimistic and would produce bands that are too narrow, which is the failure mode that
    turns a budget range into a surprise.
    """
    runs = [r for r in brick_runs if r.fittable]
    boot = boot_estimate(brick_runs)

    groups: dict[str, list] = {}
    for r in runs:
        groups.setdefault(run_group(r), []).append(r)
    doc_keys = sorted(k for k in groups if k != NO_GROUP)
    calib_keys = set(doc_keys[: max(3, len(doc_keys) // 3)])

    calib = [r for k in sorted(calib_keys) for r in groups[k]]
    train = [r for k in groups if k not in calib_keys for r in groups[k]]

    model = CostModel.fit(train, "bytes_per_brick", vocab, boot)
    preds = [model.predict_run(r) for r in calib]
    acts = [r.billable_units for r in calib]
    conf = calibrate(acts, preds, level=0.90, floor=boot)
    return model, conf, boot, calib


# -- conformal --------------------------------------------------------------------


def test_absolute_nonconformity_is_chosen_for_homoscedastic_errors():
    predicted = [float(i) for i in range(1, 60)]
    actual = [p + (1.0 if i % 2 else -1.0) for i, p in enumerate(predicted)]
    method, rho = choose_nonconformity(actual, predicted)
    assert method is Nonconformity.ABSOLUTE
    assert abs(rho) <= 0.30


def test_relative_nonconformity_is_chosen_when_error_grows_with_the_quote():
    predicted = [float(i) for i in range(1, 60)]
    actual = [p * (1.10 if i % 2 else 0.90) for i, p in enumerate(predicted)]
    method, _ = choose_nonconformity(actual, predicted)
    assert method is Nonconformity.RELATIVE


def test_the_score_function_is_chosen_before_the_level_is_applied(fitted):
    _, conf, _, _ = fitted
    assert conf.method in (Nonconformity.ABSOLUTE, Nonconformity.RELATIVE)
    assert conf.heteroscedasticity_rho == conf.heteroscedasticity_rho  # recorded, auditable


def test_wider_levels_give_wider_intervals():
    predicted = [100.0] * 40
    actual = [100.0 + (i % 11) - 5 for i in range(40)]
    widths = []
    for level in (0.50, 0.80, 0.90):
        c = calibrate(actual, predicted, level=level, method=Nonconformity.ABSOLUTE)
        widths.append(c.half_width(100.0))
    assert widths[0] <= widths[1] <= widths[2]


def test_calibration_refuses_a_tiny_fold():
    with pytest.raises(ValueError, match="at least 5"):
        calibrate([1.0, 2.0], [1.0, 2.0])


def test_the_lower_bound_never_falls_below_the_empty_task():
    c = Conformal(0.9, Nonconformity.ABSOLUTE, q=10_000.0, n_calibration=20, floor=5_000.0)
    lo, hi = c.interval(6_000.0)
    assert lo == 5_000.0
    assert hi == 16_000.0


def test_conformal_round_trips(tmp_path, fitted):
    _, conf, _, _ = fitted
    path = tmp_path / "conformal.json"
    conf.save(path)
    assert Conformal.load(path) == conf


def test_calibrated_interval_covers_about_the_stated_share(fitted):
    model, conf, _, calib = fitted
    intervals = [conf.interval(model.predict_run(r)) for r in calib]
    hits, n = coverage([r.billable_units for r in calib], intervals)
    assert hits / n >= 0.80, f"only {hits}/{n} covered"


def test_relative_half_width_is_finite_and_positive(fitted):
    _, conf, _, _ = fitted
    assert 0.0 < conf.relative_half_width(50_000.0) < math.inf


# -- quotes -----------------------------------------------------------------------


def test_a_quote_is_never_a_bare_number(fitted, pricing, vocab):
    model, conf, boot, _ = fitted
    q = make_quote(
        "demo-1", {"Extract": 3, "Validate": 2}, 20_000,
        model=model, conformal=conf, pricing=pricing,
        baselines={"boot": BootModel(boot)},
    )
    assert q.predicted_units > 0
    assert q.interval_lo < q.predicted_units < q.interval_hi
    assert q.baselines["boot"] == pytest.approx(boot)
    assert q.coefficients_ref.startswith("bytes_per_brick@")
    assert q.quoted_at


def test_quote_dollars_follow_the_pricing_record(fitted, pricing):
    model, conf, _, _ = fitted
    q = make_quote("d", {"Extract": 1}, 1000, model=model, conformal=conf, pricing=pricing)
    assert q.predicted_usd == pytest.approx(pricing.units_to_usd(q.predicted_units))


def test_retrieve_always_earns_an_operational_flag(fitted, pricing):
    model, conf, _, _ = fitted
    q = make_quote("d", {"Retrieve": 1}, 1000, model=model, conformal=conf, pricing=pricing)
    assert any("Retrieve" in f for f in q.outlier_flags)
    assert any("Extract" in f for f in q.outlier_flags), "the flag must name the cheaper lever"


def test_a_dominant_brick_is_flagged_with_its_share(fitted, pricing):
    model, conf, _, _ = fitted
    q = make_quote("d", {"Retrieve": 4}, 900, model=model, conformal=conf, pricing=pricing)
    assert any("%" in f for f in q.outlier_flags)


def test_a_narrow_verdict_hides_per_brick_detail(fitted, pricing):
    model, conf, _, _ = fitted
    q = make_quote(
        "d", {"Retrieve": 2}, 1000,
        model=model, conformal=conf, pricing=pricing, show_per_brick=False,
    )
    assert q.outlier_flags == ()
    assert "_hidden" in q.as_dict()["units"]


def test_quote_hash_changes_with_the_request_and_not_with_the_run(fitted, pricing):
    model, conf, _, _ = fitted
    kwargs = dict(model=model, conformal=conf, pricing=pricing, quoted_at="2026-09-04T00:00:00+00:00")
    a = make_quote("d", {"Extract": 1}, 1000, **kwargs)
    b = make_quote("d", {"Extract": 1}, 1000, **kwargs)
    c = make_quote("d", {"Extract": 2}, 1000, **kwargs)
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()


def test_quotes_append_and_reload(tmp_path, fitted, pricing):
    model, conf, _, _ = fitted
    path = tmp_path / "quotes.jsonl"
    for i in range(3):
        append_quote(path, make_quote(f"q{i}", {"Extract": i + 1}, 1000,
                                      model=model, conformal=conf, pricing=pricing))
    assert set(read_quotes(path)) == {"q0", "q1", "q2"}


# -- invoices ---------------------------------------------------------------------


def test_every_invoice_reconciles_exactly(fitted, pricing):
    model, conf, boot, calib = fitted
    for run in calib:
        q = make_quote(
            run.run_id, run.units, run.context_bytes,
            model=model, conformal=conf, pricing=pricing,
        )
        inv = make_invoice(run, q, model, boot=boot)
        assert inv.reconciles, f"{run.run_id}: {inv.sum_check} != {inv.actual_units}"
        assert inv.sum_check == pytest.approx(run.billable_units, abs=1e-6)


def test_the_invoice_carries_the_variable_error_beside_the_total(fitted, pricing):
    model, conf, boot, calib = fitted
    run = calib[0]
    q = make_quote(run.run_id, run.units, run.context_bytes,
                   model=model, conformal=conf, pricing=pricing)
    inv = make_invoice(run, q, model, boot=boot)
    d = inv.as_dict()
    assert "error_pct" in d and "variable_error_pct" in d


def test_the_reconciliation_line_absorbs_the_residual(fitted, pricing):
    model, conf, boot, calib = fitted
    run = calib[0]
    q = make_quote(run.run_id, run.units, run.context_bytes,
                   model=model, conformal=conf, pricing=pricing)
    inv = make_invoice(run, q, model, boot=boot)
    modelled = sum(line.units for line in inv.lines if line.item != "reconciliation")
    assert inv.residual == pytest.approx(run.billable_units - modelled, abs=1e-6)


def test_hiding_per_brick_lines_still_reconciles(fitted, pricing):
    model, conf, boot, calib = fitted
    run = calib[0]
    q = make_quote(run.run_id, run.units, run.context_bytes,
                   model=model, conformal=conf, pricing=pricing, show_per_brick=False)
    inv = make_invoice(run, q, model, boot=boot, show_per_brick=False)
    assert inv.reconciles
    assert not any(" x " in line.item for line in inv.lines), "no per-brick line survived"
    assert any("withheld" in n for n in inv.notes)


def test_a_run_below_the_empty_task_is_flagged_not_hidden(fitted, pricing):
    model, conf, boot, calib = fitted
    run = calib[0]
    q = make_quote(run.run_id, run.units, run.context_bytes,
                   model=model, conformal=conf, pricing=pricing)
    inv = make_invoice(run, q, model, boot=run.billable_units + 1_000.0)
    assert any("measurement failure" in n for n in inv.notes)


def test_rendered_invoice_warns_when_the_model_explains_little(fitted, pricing):
    model, conf, boot, calib = fitted
    run = calib[0]
    q = make_quote(run.run_id, run.units, run.context_bytes,
                   model=model, conformal=conf, pricing=pricing)
    inv = make_invoice(run, q, model, boot=boot)
    inv.lines[0] = type(inv.lines[0])("start-up", 0.0, "forced")
    inv.lines[-1] = type(inv.lines[-1])(
        "reconciliation", run.billable_units, "forced"
    )
    text = inv.render(pricing.input_per_mtok / 1e6)
    assert "pricing something other than" in text


def test_invoice_evidence_class_follows_the_run(fitted, pricing):
    model, conf, boot, calib = fitted
    run = calib[0]
    q = make_quote(run.run_id, run.units, run.context_bytes,
                   model=model, conformal=conf, pricing=pricing)
    inv = make_invoice(run, q, model, boot=boot)
    assert inv.evidence_class is EvidenceClass.PIPELINE_ONLY
