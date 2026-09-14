import pytest

from assay.batching import Bundle, analyse, collect_bundles
from assay.blind import commit_quotes, guard_dispatch, score_blind
from assay.conformal import calibrate
from assay.costmodel import BootModel, CostModel
from assay.errors import SealError
from assay.evidence import EvidenceClass
from assay.features import run_group
from assay.grading import (
    RUBRIC_TOTAL,
    BlindingManifest,
    Grade,
    grader_agreement,
    score_quality,
)
from assay.harness.campaign import design_batching, design_blind
from assay.harness.mock_adapter import MockAdapter, MockMode
from assay.harness.runner import run_campaign
from assay.schemas import ExclusionCode, Status, read_runs
from assay.select import NO_GROUP

from tests.assay.conftest import RUNTIME_HASH, boot_estimate


@pytest.fixture(scope="module")
def frozen(brick_runs, vocab):
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
    conf = calibrate(
        [r.billable_units for r in calib],
        [model.predict_run(r) for r in calib],
        level=0.90,
        floor=boot,
    )
    return model, conf, boot


@pytest.fixture(scope="module")
def blind_execution(frozen, vocab, sealed, pricing, tmp_path_factory):
    """Quote all sealed tasks, then dispatch them. In that order, provably."""
    model, conf, boot = frozen
    probes = design_blind(vocab, sealed, n_tasks=24)
    sizes = {p.probe_id: sum(sealed.size(d) for d in p.context_files) for p in probes}

    out = tmp_path_factory.mktemp("blind")
    quotes = commit_quotes(
        probes, model=model, conformal=conf, pricing=pricing,
        context_bytes=sizes, baselines={"boot": BootModel(boot)},
        path=out / "quotes.jsonl",
    )
    run_campaign(
        probes, MockAdapter(MockMode.BRICK, seed=77), out / "runs.jsonl",
        campaign_id="blind", pricing=pricing, corpus=sealed,
        runtime_hash=RUNTIME_HASH, allow_blind=True,
    )
    return quotes, read_runs(out / "runs.jsonl", pricing), boot


# -- blind: the ordering is the claim ---------------------------------------------


def test_a_task_without_a_committed_quote_will_not_dispatch(vocab, sealed):
    probe = design_blind(vocab, sealed, n_tasks=1)[0]
    with pytest.raises(SealError, match="no committed quote"):
        guard_dispatch(probe, {})


def test_every_quote_precedes_its_run(blind_execution):
    quotes, runs, boot = blind_execution
    report = score_blind(quotes, runs, boot=boot, n_planned=24)
    assert report.n_scored == 24


def test_a_quote_written_after_the_run_is_rejected(blind_execution):
    quotes, runs, boot = blind_execution
    from dataclasses import replace

    late = dict(quotes)
    victim = runs[0].probe_id
    late[victim] = replace(quotes[victim], quoted_at="2099-01-01T00:00:00+00:00")
    with pytest.raises(SealError, match="not a prediction"):
        score_blind(late, runs, boot=boot)


def test_an_offsetless_run_timestamp_is_read_as_utc_not_a_crash(blind_execution):
    """A replayed run file can lose its offset; the ordering check must still return a
    verdict rather than raise TypeError from comparing naive against aware datetimes."""
    import copy

    quotes, runs, boot = blind_execution
    naive = copy.deepcopy(runs)
    for r in naive:
        r.timestamp = r.timestamp.replace("+00:00", "")

    report = score_blind(quotes, naive, boot=boot)
    assert report.n_scored == len([r for r in naive if r.accepted])

    # And the invariant still bites through the same path.
    from dataclasses import replace

    late = dict(quotes)
    victim = naive[0].probe_id
    late[victim] = replace(quotes[victim], quoted_at="2099-01-01T00:00:00+00:00")
    with pytest.raises(SealError, match="not a prediction"):
        score_blind(late, naive, boot=boot)


def test_a_run_with_no_quote_at_all_is_rejected(blind_execution):
    quotes, runs, boot = blind_execution
    partial = {k: v for k, v in quotes.items() if k != runs[0].probe_id}
    with pytest.raises(SealError, match="without a committed quote"):
        score_blind(partial, runs, boot=boot)


def test_too_many_exclusions_void_the_set_rather_than_shrink_it(blind_execution):
    quotes, runs, boot = blind_execution
    import copy

    damaged = copy.deepcopy(runs)
    for r in damaged[:4]:
        r.exclusion_code = ExclusionCode.TRUNCATED
        r.status = Status.EXCLUDED

    report = score_blind(quotes, damaged, boot=boot, n_planned=24)
    assert report.voided
    assert report.scorecard is None
    assert report.observations()["M6_blind_mape_pct"] is None
    assert any("void" in n for n in report.notes)


def test_three_exclusions_are_survivable(blind_execution):
    quotes, runs, boot = blind_execution
    import copy

    damaged = copy.deepcopy(runs)
    for r in damaged[:3]:
        r.exclusion_code = ExclusionCode.TRANSPORT_ERROR
        r.status = Status.EXCLUDED
    report = score_blind(quotes, damaged, boot=boot, n_planned=24)
    assert not report.voided
    assert report.n_scored == 21


# -- blind: what it reports --------------------------------------------------------


def test_coverage_is_reported_as_a_screen_with_an_interval(blind_execution):
    quotes, runs, boot = blind_execution
    report = score_blind(quotes, runs, boot=boot, n_planned=24)
    assert "screen, not proof" in report.coverage_claim
    lo, hi = report.coverage_ci
    assert lo < hi
    assert report.coverage_n == 24


def test_the_blind_report_carries_both_error_measures(blind_execution):
    quotes, runs, boot = blind_execution
    report = score_blind(quotes, runs, boot=boot, n_planned=24)
    card = report.scorecard.as_dict()
    assert "mape_total_pct" in card and "vwape_pct" in card


def test_lift_is_measured_against_a_baseline_not_against_nothing(blind_execution):
    quotes, runs, boot = blind_execution
    report = score_blind(
        quotes, runs, boot=boot, baselines={"boot": BootModel(boot)}, n_planned=24
    )
    assert report.best_baseline == "boot"
    assert report.lift_pct is not None
    assert report.observations()["M6b_blind_lift_pct"] == report.lift_pct


def test_mock_blind_evidence_never_claims_feasibility(blind_execution):
    quotes, runs, boot = blind_execution
    report = score_blind(quotes, runs, boot=boot, n_planned=24)
    assert report.evidence_class is EvidenceClass.PIPELINE_ONLY


# -- batching: cost ---------------------------------------------------------------


def test_bundles_pair_both_arms_from_real_runs(vocab, sealed, pricing, tmp_path):
    probes = design_batching(vocab, sealed, n_two_task=8, n_four_task=3)
    out = tmp_path / "batch.jsonl"
    run_campaign(
        probes, MockAdapter(MockMode.BRICK, seed=11), out,
        campaign_id="batch", pricing=pricing, corpus=sealed, runtime_hash=RUNTIME_HASH,
    )
    bundles = collect_bundles(read_runs(out, pricing))
    assert len(bundles) == 11
    assert {b.n_tasks for b in bundles} == {2, 4}


def test_a_bundle_missing_an_arm_is_dropped_not_patched(vocab, sealed, pricing, tmp_path):
    probes = design_batching(vocab, sealed, n_two_task=2, n_four_task=0)
    out = tmp_path / "batch.jsonl"
    run_campaign(
        probes, MockAdapter(seed=11), out,
        campaign_id="b", pricing=pricing, corpus=sealed, runtime_hash=RUNTIME_HASH,
    )
    runs = [r for r in read_runs(out, pricing) if not r.probe_id.endswith("-batch")
            or r.bundle_id != "BATCH2-001"]
    assert len(collect_bundles(runs)) == 1


def test_saving_splits_into_the_toll_and_the_remainder():
    boot = 5_000.0
    # Two tasks, each 6,000 units apart; batched costs 7,000. Toll accounts for 5,000
    # of the 5,000 saved, so the residual is zero -- the mechanism is entirely amortisation.
    b = Bundle("B1", separate_units=12_000.0, batched_units=7_000.0, n_tasks=2)
    assert b.saving_pct == pytest.approx(41.667, abs=0.01)
    assert b.toll_component(boot) == pytest.approx(41.667, abs=0.01)
    assert b.residual_component(boot) == pytest.approx(0.0, abs=0.01)


def test_genuine_subadditivity_shows_up_in_the_residual():
    boot = 5_000.0
    b = Bundle("B1", separate_units=12_000.0, batched_units=5_500.0, n_tasks=2)
    assert b.residual_component(boot) > 10.0


def test_a_small_toll_can_miss_the_bar_with_the_mechanism_intact():
    """The distinction that separates a Narrow verdict from a Stop."""
    bundles = [Bundle(f"B{i}", 4_000.0, 3_100.0, 2) for i in range(8)]
    report = analyse(bundles, boot=800.0, iters=300)
    assert report.aggregate_saving_pct < 40.0
    assert report.n_positive == 8
    assert "Narrow result, not an absent effect" in report.interpretation()


def test_fewer_than_eight_bundles_is_untested_not_unsaving():
    report = analyse([Bundle(f"B{i}", 4_000.0, 2_000.0, 2) for i in range(5)], boot=800.0, iters=200)
    assert not report.tested
    assert "untested" in report.interpretation()
    assert report.observations()["M11_batching_saving_pct"] is None


def test_a_strong_saving_clears_the_cost_gate():
    bundles = [Bundle(f"B{i}", 10_000.0, 4_000.0, 2) for i in range(10)]
    report = analyse(bundles, boot=3_000.0, iters=500)
    assert report.cost_gate_passes
    assert report.bootstrap.lo >= 20.0


def test_bundles_are_the_unit_of_resampling():
    bundles = [Bundle(f"B{i}", 10_000.0, 4_000.0, 2) for i in range(10)]
    report = analyse(bundles, boot=3_000.0, iters=500)
    assert report.bootstrap.iters == 500


def test_no_bundles_reports_nothing_rather_than_zero():
    report = analyse([], boot=1000.0)
    assert report.bootstrap is None
    assert not report.tested


# -- batching: quality -------------------------------------------------------------


def manifest_and_grades(delta: float, *, agreement: bool = True, critical_batched: int = 0):
    arms, grades, bundle_of = {}, [], {}
    for i in range(8):
        sep, bat = f"a{i}-sep", f"a{i}-bat"
        arms[sep], arms[bat] = "separate", "batched"
        bundle_of[sep] = bundle_of[bat] = f"B{i}"
        base = 80.0 + i
        wobble = 0.0 if agreement else (25.0 if i % 2 else -25.0)
        for grader, offset in (("g1", 0.0), ("g2", wobble)):
            grades.append(Grade(sep, grader, min(RUBRIC_TOTAL, max(0.0, base + offset))))
            grades.append(
                Grade(
                    bat, grader,
                    min(RUBRIC_TOTAL, max(0.0, base + delta - offset)),
                    critical_failure="fabricated fact" if i < critical_batched else None,
                )
            )
    return BlindingManifest.shuffled(arms, seed=5), grades, bundle_of


def test_equal_quality_is_non_inferior():
    manifest, grades, bundle_of = manifest_and_grades(0.0)
    report = score_quality(grades, manifest, bundle_of, iters=500)
    assert report.conclusive
    assert report.non_inferior
    assert report.delta_q == pytest.approx(0.0, abs=1e-6)


def test_a_large_quality_drop_fails_non_inferiority():
    manifest, grades, bundle_of = manifest_and_grades(-15.0)
    report = score_quality(grades, manifest, bundle_of, iters=500)
    assert report.delta_lower_90 < -5.0
    assert not report.non_inferior


def test_extra_critical_failures_fail_non_inferiority():
    manifest, grades, bundle_of = manifest_and_grades(0.0, critical_batched=3)
    report = score_quality(grades, manifest, bundle_of, iters=500)
    assert report.critical_batched > report.critical_separate + 1
    assert not report.non_inferior


def test_graders_who_disagree_make_the_arm_inconclusive():
    manifest, grades, bundle_of = manifest_and_grades(0.0, agreement=False)
    report = score_quality(grades, manifest, bundle_of, iters=500)
    assert not report.conclusive
    assert not report.non_inferior
    assert any("inconclusive" in n for n in report.notes)


def test_agreement_is_computable_before_any_unblinding():
    _, grades, _ = manifest_and_grades(0.0)
    assert grader_agreement(grades) >= 0.67


def test_the_arm_label_lives_only_in_the_manifest():
    manifest, grades, _ = manifest_and_grades(0.0)
    assert all(not hasattr(g, "arm") for g in grades)
    assert len(manifest.hash()) == 64


def test_a_grade_outside_the_rubric_is_refused():
    with pytest.raises(ValueError):
        Grade("a", "g", RUBRIC_TOTAL + 1)


def test_an_unknown_critical_failure_is_refused():
    with pytest.raises(ValueError, match="unknown critical failure"):
        Grade("a", "g", 50.0, critical_failure="vibes")


def test_cheap_but_worse_batching_cannot_pass_on_price_alone():
    """The gate the reference never had."""
    bundles = [Bundle(f"B{i}", 10_000.0, 3_000.0, 2) for i in range(10)]
    manifest, grades, bundle_of = manifest_and_grades(-20.0)
    quality = score_quality(grades, manifest, bundle_of, iters=500)
    report = analyse(bundles, boot=3_000.0, quality=quality, iters=500)

    assert report.cost_gate_passes, "it really is much cheaper"
    assert not quality.non_inferior, "and the answers are much worse"
    assert report.observations()["M12_batching_quality_lb"] < -5.0
