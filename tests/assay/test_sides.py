"""The two sides of the bill.

`billable_units` fuses material sent with answer written, and an output token costs five
input tokens. One blended error number cannot say which of the two a form is actually
predicting, and that is the first question anyone asks of a cost model.
"""

import pytest

from assay.pricing import PROVISIONAL
from assay.sides import Side, retarget, score_sides, side_boot, side_units

from tests.assay.conftest import boot_estimate


def fittable(runs):
    return [r for r in runs if r.fittable]


def test_the_two_sides_add_up_to_the_billable_total(brick_runs, pricing):
    for run in fittable(brick_runs)[:25]:
        parts = side_units(run, Side.PROMPT, pricing) + side_units(run, Side.COMPLETION, pricing)
        assert parts == pytest.approx(run.billable_units, rel=1e-9)


def test_the_answer_side_is_priced_at_the_output_ratio(brick_runs, pricing):
    run = fittable(brick_runs)[0]
    assert side_units(run, Side.COMPLETION, pricing) == pytest.approx(
        pricing.ratio * run.completion_tokens
    )
    assert pricing.ratio == PROVISIONAL.output_per_mtok / PROVISIONAL.input_per_mtok


def test_retargeting_leaves_the_original_runs_untouched(brick_runs, pricing):
    runs = fittable(brick_runs)
    before = [r.billable_units for r in runs]
    retarget(runs, Side.PROMPT, pricing)
    assert [r.billable_units for r in runs] == before, "retarget must clone, not mutate"


def test_each_side_has_its_own_start_up_toll(brick_runs, pricing):
    runs = fittable(brick_runs)
    prompt = side_boot(runs, Side.PROMPT, pricing)
    answer = side_boot(runs, Side.COMPLETION, pricing)
    combined = side_boot(runs, Side.COMBINED, pricing)

    assert prompt > 0 and answer > 0
    assert combined == pytest.approx(prompt + answer, rel=0.02)
    assert combined == pytest.approx(boot_estimate(brick_runs), rel=0.02)


def test_a_side_toll_needs_a_null_probe_and_will_not_be_guessed(brick_runs, pricing):
    runs = [r for r in fittable(brick_runs) if r.tier.value != "null"]
    with pytest.raises(ValueError, match="null probe"):
        side_boot(runs, Side.PROMPT, pricing)


def test_the_report_carries_all_three_columns_or_none(brick_runs, vocab, pricing):
    report = score_sides(fittable(brick_runs), "bytes_per_brick", vocab, pricing)
    payload = report.as_dict()
    for key in ("prompt", "completion", "combined"):
        assert payload[key]["n_scored"] > 0
    assert 0.0 < payload["completion_share_of_bill_pct"] < 100.0
    text = report.render()
    assert "material (prompt)" in text and "answer (completion)" in text
    assert "No gate reads this table" in text


def test_the_simulator_cannot_exercise_the_split_and_the_suite_says_so(
    brick_runs, vocab, pricing
):
    """`MockAdapter` derives completion tokens as a fixed share of the same total.

    So its two sides are the same signal scaled, and they score alike no matter which form
    is used. That is a limitation of the simulator, not a property of agent runtimes -- on
    a real one, prompt tokens are mostly mounted material and completion tokens are the
    answer, and they need not move together at all. Asserting the limitation keeps anyone
    from reading a flat table here as evidence that the split does not matter.
    """
    runs = fittable(brick_runs)
    sides = score_sides(runs, "bytes_per_brick", vocab, pricing)
    assert sides.prompt.pooled.vwape == pytest.approx(
        sides.completion.pooled.vwape, rel=0.10
    ), "the mock splits proportionally; if these diverge the simulator has changed"


def test_the_material_side_is_not_assumed_to_be_the_easy_one(brick_runs, vocab, pricing):
    """`bytes` explains the material side well only when material drives cost.

    In the brick arm it does not -- the planted brick term moves both sides -- so a size
    baseline is weak on prompt tokens too. The diagnostic must report that rather than
    assume prompt tokens are trivially predictable.
    """
    runs = fittable(brick_runs)
    from_size = score_sides(runs, "bytes", vocab, pricing)
    from_bricks = score_sides(runs, "bytes_per_brick", vocab, pricing)
    assert from_bricks.prompt.pooled.vwape < from_size.prompt.pooled.vwape


def test_the_diagnostic_reuses_the_same_evaluation_path_as_the_real_score(
    brick_runs, vocab, pricing
):
    """The combined column must reproduce the ordinary cross-validation exactly.

    If it did not, the diagnostic would be scoring the model on a gentler path than the
    one that decides anything, and its columns could not be compared with the ladder.
    """
    from assay.select import cross_validate

    runs = fittable(brick_runs)
    boot = boot_estimate(brick_runs)
    direct = cross_validate(runs, "bytes_per_brick", vocab, boot)
    sides = score_sides(runs, "bytes_per_brick", vocab, pricing)

    assert sides.combined.n_scored == direct.n_scored
    assert sides.combined.pooled.vwape == pytest.approx(direct.pooled.vwape, rel=1e-6)


def test_sides_are_reported_for_the_null_arm_too(null_runs, vocab, pricing):
    """The negative control gets the same decomposition; a diagnostic that only runs on
    the flattering arm is not a diagnostic."""
    report = score_sides(fittable(null_runs), "bytes_per_brick", vocab, pricing)
    assert report.money_weighted_note
    assert report.completion.pooled.n > 0
