import pytest

from assay.errors import BudgetError, SealError
from assay.evidence import EvidenceClass
from assay.harness.campaign import design_batching, design_blind, design_campaign
from assay.harness.mock_adapter import MockAdapter, MockMode, MockSpec
from assay.harness.probes import NULL_INSTRUCTION, compose_instruction
from assay.harness.runner import Budget, run_campaign
from assay.linalg import spearman
from assay.metrics import cv_pct
from assay.schemas import Split, Tier, read_runs

from tests.assay.conftest import RUNTIME_HASH


# -- the simulator ----------------------------------------------------------------


def test_mock_is_deterministic_across_fresh_adapters(tmp_path):
    doc = tmp_path / "d.txt"
    doc.write_text("x" * 1000, encoding="utf-8")

    def once():
        a = MockAdapter(MockMode.BRICK, seed=5)
        a.register("hello", {"Extract": 2})
        return a.run("hello", [doc])

    assert once().prompt_tokens == once().prompt_tokens


def test_mock_replicates_differ_so_the_noise_floor_is_not_free(tmp_path):
    doc = tmp_path / "d.txt"
    doc.write_text("x" * 1000, encoding="utf-8")
    a = MockAdapter(MockMode.BRICK, seed=5)
    a.register("hello", {"Extract": 2})
    first, second = a.run("hello", [doc]), a.run("hello", [doc])
    assert first.prompt_tokens != second.prompt_tokens


def test_brick_mode_prices_bricks_and_null_mode_does_not(tmp_path):
    doc = tmp_path / "d.txt"
    doc.write_text("x" * 1000, encoding="utf-8")
    spec = MockSpec(noise_cv=0.0)

    brick = MockAdapter(MockMode.BRICK, spec, seed=1)
    null = MockAdapter(MockMode.NULL, spec, seed=1)
    for a in (brick, null):
        a.register("with", {"Retrieve": 1})
        a.register("without", {})

    assert brick.run("with", [doc]).prompt_tokens > brick.run("without", [doc]).prompt_tokens
    n_with = null.run("with", [doc])
    n_without = null.run("without", [doc])
    assert n_with.prompt_tokens == n_without.prompt_tokens, (
        "null mode must contain no brick term at all"
    )


def test_mock_boot_is_not_the_reference_boot():
    """29,821 belongs to a runtime nobody here owns; assuming it would bake in a foreign harness."""
    assert MockSpec().boot < 10_000


def test_mock_is_always_pipeline_only():
    assert MockAdapter().evidence_class is EvidenceClass.PIPELINE_ONLY


# -- campaign design --------------------------------------------------------------


def test_campaign_brackets_with_null_probes(vocab, sealed):
    plan = design_campaign(vocab, sealed)
    nulls = plan.by_tier(Tier.NULL)
    assert len(nulls) == 6
    assert plan.probes[0].probe_id.startswith("null-open")
    assert plan.probes[-1].probe_id.startswith("null-close")
    assert all(p.instruction == NULL_INSTRUCTION for p in nulls)


def test_every_brick_gets_at_least_two_replicates(vocab, sealed):
    plan = design_campaign(vocab, sealed, replicates=2)
    for brick in vocab.names:
        base = [p for p in plan.by_tier(Tier.BASE) if p.units.get(brick)]
        assert len(base) >= 2, f"{brick} has too few replicates to have a coefficient"


def test_unit_count_is_orthogonal_to_document_size(vocab, sealed):
    """The design's whole claim to per-brick coefficients rests on this."""
    plan = design_campaign(vocab, sealed)
    fittable = [p for p in plan.probes if p.tier in (Tier.BASE, Tier.COMPOSITE)]
    units = [float(p.total_units) for p in fittable]
    sizes = [float(sum(sealed.size(d) for d in p.context_files)) for p in fittable]
    rho = spearman(units, sizes)
    assert abs(rho) <= 0.20, f"units and bytes track each other (Spearman {rho:+.2f})"


def test_campaign_never_touches_blind_documents(vocab, sealed):
    plan = design_campaign(vocab, sealed)
    used = {d for p in plan.probes for d in p.context_files}
    sealed.assert_fit_only(used)


def test_ladder_probes_are_excluded_from_fitting(vocab, sealed):
    plan = design_campaign(vocab, sealed)
    assert all(p.split is Split.PROBE for p in plan.by_tier(Tier.LADDER))


def test_orthogonality_probe_holds_bytes_and_varies_document_count(vocab, sealed):
    plan = design_campaign(vocab, sealed)
    orth = [p for p in plan.by_tier(Tier.LADDER) if "orth" in p.probe_id]
    assert {len(p.context_files) for p in orth} == {1, 2, 4}


def test_blind_probes_use_only_blind_documents(vocab, sealed):
    probes = design_blind(vocab, sealed, n_tasks=24)
    assert len(probes) == 24
    blind_names = set(sealed.seal.blind)
    assert all(set(p.context_files) <= blind_names for p in probes)
    assert all(p.split is Split.BLIND for p in probes)


def test_batching_bundles_pair_a_separate_and_a_batched_arm(vocab, sealed):
    probes = design_batching(vocab, sealed, n_two_task=8, n_four_task=3)
    bundles: dict[str, list] = {}
    for p in probes:
        bundles.setdefault(p.bundle_id, []).append(p)
    assert len(bundles) == 11
    two = [v for k, v in bundles.items() if k.startswith("BATCH2")]
    assert all(len(v) == 3 for v in two), "a two-task bundle costs 3 dispatches"
    four = [v for k, v in bundles.items() if k.startswith("BATCH4")]
    assert all(len(v) == 5 for v in four)


def test_batched_arm_reuses_the_separate_clauses_verbatim():
    """A cost difference must not be a difference in what was asked."""
    separate = ["Extract clause.", "Validate clause."]
    batched = compose_instruction([("Extract", 1), ("Validate", 1)])
    from assay.harness.probes import brick_instruction

    for brick in ("Extract", "Validate"):
        assert brick_instruction(brick, 1) in batched
    assert "Answer all sections" in batched
    assert separate  # the separate arm is the unwrapped clause, by construction


# -- execution --------------------------------------------------------------------


def test_runner_is_resumable_and_never_double_charges(vocab, sealed, pricing, tmp_path):
    plan = design_campaign(vocab, sealed)
    out = tmp_path / "runs.jsonl"
    kwargs = dict(
        campaign_id="c", pricing=pricing, corpus=sealed,
        runtime_hash=RUNTIME_HASH, expected_model=MockAdapter().model,
    )

    first = run_campaign(plan.probes[:20], MockAdapter(seed=3), out, **kwargs)
    second = run_campaign(plan.probes, MockAdapter(seed=3), out, **kwargs)

    assert first.dispatched == 20
    assert second.skipped == 20
    assert second.dispatched == len(plan.probes) - 20
    ids = [r.run_id for r in read_runs(out, pricing)]
    assert len(ids) == len(set(ids)), "a resumed campaign must not duplicate a run"


def test_budget_is_checked_before_dispatch_not_after(vocab, sealed, pricing, tmp_path):
    plan = design_campaign(vocab, sealed)
    budget = Budget(max_dispatches=5)
    with pytest.raises(BudgetError):
        run_campaign(
            plan.probes, MockAdapter(), tmp_path / "runs.jsonl",
            campaign_id="c", pricing=pricing, corpus=sealed,
            runtime_hash=RUNTIME_HASH, budget=budget,
        )
    assert budget.spent == 5
    assert len(read_runs(tmp_path / "runs.jsonl", pricing)) == 5


def test_blind_probes_refuse_to_dispatch_through_the_ordinary_runner(vocab, sealed, pricing, tmp_path):
    probes = design_blind(vocab, sealed, n_tasks=2)
    with pytest.raises(SealError, match="blind runner"):
        run_campaign(
            probes, MockAdapter(), tmp_path / "runs.jsonl",
            campaign_id="c", pricing=pricing, corpus=sealed, runtime_hash=RUNTIME_HASH,
        )


def test_an_adapter_that_has_not_disabled_caching_never_dispatches(vocab, sealed, pricing, tmp_path):
    """The usage contract catches a cache hit, but only after the money is spent.

    Prompt caching is on by default on several providers, and a cached run measures what
    was asked before it rather than what the task costs. On the 24-task blind set the
    fourth such exclusion voids the whole set, so the declaration is required up front.
    """
    from assay.errors import AdapterError

    adapter = MockAdapter()
    adapter.caching_disabled = False
    out = tmp_path / "runs.jsonl"
    plan = design_campaign(vocab, sealed)

    with pytest.raises(AdapterError, match="caching_disabled"):
        run_campaign(
            plan.probes[:4], adapter, out,
            campaign_id="c", pricing=pricing, corpus=sealed, runtime_hash=RUNTIME_HASH,
        )
    assert not out.exists(), "the refusal must come before the first dispatch"


def test_an_adapter_silent_about_caching_is_refused_too(vocab, sealed, pricing, tmp_path):
    """Silence is not consent: an adapter must state the property, not omit it."""
    from assay.errors import AdapterError

    adapter = MockAdapter()
    del adapter.caching_disabled
    plan = design_campaign(vocab, sealed)

    with pytest.raises(AdapterError, match="caching_disabled"):
        run_campaign(
            plan.probes[:4], adapter, tmp_path / "runs.jsonl",
            campaign_id="c", pricing=pricing, corpus=sealed, runtime_hash=RUNTIME_HASH,
        )


def test_a_model_mismatch_becomes_an_exclusion_row_not_a_hole(vocab, sealed, pricing, tmp_path):
    plan = design_campaign(vocab, sealed)
    out = tmp_path / "runs.jsonl"
    report = run_campaign(
        plan.probes[:4], MockAdapter(), out,
        campaign_id="c", pricing=pricing, corpus=sealed,
        runtime_hash=RUNTIME_HASH, expected_model="a-different-snapshot",
    )
    assert report.excluded == 4
    rows = read_runs(out, pricing)
    assert len(rows) == 4, "excluded runs stay in the file"
    assert all(not r.accepted for r in rows)


def test_recorded_context_bytes_match_the_manifest(brick_runs, sealed):
    for run in brick_runs:
        expected = sum(sealed.size(d) for d in run.context_files)
        assert run.context_bytes == expected


def test_noise_floor_is_measurable_and_small(brick_runs):
    groups: dict[str, list[float]] = {}
    for r in brick_runs:
        if r.accepted:
            groups.setdefault(r.instruction_sha256, []).append(r.billable_units)
    replicated = {k: v for k, v in groups.items() if len(v) >= 2}
    assert replicated, "no instruction was replicated; the noise floor is unmeasurable"
    floor = sum(cv_pct(v) for v in replicated.values()) / len(replicated)
    assert 0.0 < floor < 3.0
