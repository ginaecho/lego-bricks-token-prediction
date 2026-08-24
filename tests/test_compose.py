"""Tests for the base task vocabulary, the composition model, and decomposition."""

from pathlib import Path

import pytest

from token_yield.compose import (
    CompositionModel, Run, batching_saving, default_runs_path, load_runs,
    mape, noise_floor, select_model,
)
from token_yield.decompose import (
    Decomposition, decompose_prompt, explain, heuristic_decompose,
    parse_decomposition, price, reconstruction_error,
)
from token_yield.tasks import ORDER, PRIMITIVES, TaskSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runs():
    return load_runs(default_runs_path())


@pytest.fixture(scope="module")
def model(runs):
    return select_model(runs)


# ── the vocabulary ───────────────────────────────────────────────────────

def test_every_primitive_is_named_for_the_work_not_the_framework():
    for slug, p in PRIMITIVES.items():
        assert p.name and p.name[0].isupper()
        assert p.slug == slug
        assert p.industry, f"{slug} must say where the task type is bought"
        assert p.blurb.endswith(".")


def test_order_covers_the_vocabulary_exactly():
    assert set(ORDER) == set(PRIMITIVES)
    assert len(ORDER) == len(PRIMITIVES)


def test_categories_map_onto_the_maintenance_taxonomy():
    cats = {p.category for p in PRIMITIVES.values()}
    assert {"corrective", "adaptive", "perfective"} <= cats


# ── composition algebra ──────────────────────────────────────────────────

def test_counts_sum_repeated_primitives():
    t = TaskSpec("x", (("review", 1), ("review", 2)))
    assert t.counts()["review"] == 3
    assert t.total_units == 3
    assert t.arity == 1, "the same primitive twice is still one kind of work"


def test_arity_counts_distinct_primitives():
    t = TaskSpec("x", (("review", 1), ("draft", 2), ("validate", 1)))
    assert t.arity == 3
    assert t.total_units == 4


def test_notation_is_readable():
    t = TaskSpec("x", (("review", 2), ("reconcile", 1)))
    assert t.notation() == "2xReview + Reconcile"


def test_null_task_asks_for_nothing():
    t = TaskSpec("null", ())
    assert t.total_units == 0
    assert "null" in t.prompt()
    assert "DONE" in t.prompt()


def test_a_composite_never_hands_one_primitive_anothers_material():
    """The bug this guards: Report inheriting Extract's field names."""
    t = TaskSpec("x", (("extract", 1), ("report", 1)),
                 context=(str(REPO_ROOT / "README.md"),),
                 targets=(("extract", ("invoice total",)),
                          ("report", ("revenue performance",))))
    prompt = t.prompt()
    assert "invoice total" in prompt
    assert "revenue performance" in prompt
    assert t.targets_for("extract") == ("invoice total",)
    assert t.targets_for("report") == ("revenue performance",)


def test_bare_targets_still_work_for_a_single_primitive():
    t = TaskSpec("x", (("extract", 2),), targets=("a", "b"))
    assert t.targets_for("extract") == ("a", "b")


def test_source_free_primitives_need_no_context():
    for slug in ("draft", "remediate"):
        t = TaskSpec("x", ((slug, 2),))
        assert t.prompt()
        assert t.context_bytes() == 0


# ── the measured campaign ────────────────────────────────────────────────

def test_campaign_is_present_and_measured(runs):
    assert len(runs) >= 30
    assert any(r.held_out for r in runs)
    assert any(r.total_units == 0 for r in runs), "a null probe must be measured"
    assert all(r.tokens > 0 for r in runs)


def test_the_null_probe_is_most_of_any_task(runs):
    """The headline finding: cost is dominated by starting an agent at all."""
    boot = min(r.tokens for r in runs if r.total_units == 0)
    biggest = max(r.tokens for r in runs)
    assert boot / biggest > 0.5


def test_noise_floor_is_small_but_real(runs):
    nf = noise_floor(runs)
    assert 0.0 < nf < 0.05


# ── model selection ──────────────────────────────────────────────────────

def test_selection_beats_the_constant_model(model):
    assert model.scores["constant"] > model.loo_mape


def test_selected_model_predicts_inside_a_few_percent(model):
    assert model.loo_mape < 0.05


def test_context_bytes_earn_their_place(model):
    """Review's cost is what it reads; a model blind to bytes should lose."""
    assert model.form.startswith("bytes")
    assert model.byte_slope() > 0.1


def test_byte_slope_is_in_the_measured_range(model):
    """Independently measured at ~0.41 tokens/byte on two different corpora."""
    assert 0.2 < model.byte_slope() < 0.6


def test_retrieve_is_the_most_expensive_primitive(model):
    """Searching costs tool calls; the fit should discover that unprompted."""
    marg = model.marginals()
    assert marg["retrieve"] == max(marg.values())


def test_ridge_does_not_swamp_the_intercept(model):
    """Regression guard: a globally-scaled ridge destroyed the intercept."""
    assert 20_000 < model.coef[0] < 60_000


def test_model_is_not_merely_memorising(model):
    assert model.in_sample_mape <= model.loo_mape


# ── held-out prediction ──────────────────────────────────────────────────

def test_held_out_tasks_are_priced_without_being_fitted(runs, model):
    held = [r for r in runs if r.held_out]
    assert held
    errs = [abs(model.predict_run(r) - r.tokens) / r.tokens for r in held]
    assert sum(errs) / len(errs) < 0.10


def test_an_unseen_arity_still_predicts(runs, model):
    """A four-way mix, when the fit only ever saw up to three."""
    quad = [r for r in runs if r.held_out and r.arity >= 4]
    assert quad, "the campaign must hold out an arity it never fitted"
    for r in quad:
        assert abs(model.predict_run(r) - r.tokens) / r.tokens < 0.15


# ── what composition buys ────────────────────────────────────────────────

def test_batching_beats_separate_agents(model):
    counts = {"review": 1, "remediate": 2, "validate": 2}
    batched, separate, saving = batching_saving(model, counts, 1235)
    assert batched < separate
    assert saving > 0.3, "paying start-up once is the whole point"


def test_batching_saving_grows_with_the_number_of_parts(model):
    _, _, two = batching_saving(model, {"review": 1, "draft": 1}, 1000)
    _, _, four = batching_saving(
        model, {"review": 1, "draft": 1, "validate": 1, "report": 1}, 1000)
    assert four > two


# ── decomposition: the autoencoder ───────────────────────────────────────

def test_decompose_prompt_offers_the_whole_vocabulary():
    p = decompose_prompt("read this and summarise it")
    for prim in PRIMITIVES.values():
        assert prim.name in p
        assert prim.slug in p
    assert "JSON" in p


def test_parse_reads_a_fenced_reply():
    reply = ('```json\n{"counts": {"review": 1, "validate": 2}, '
             '"rationale": "read then check"}\n```')
    d = parse_decomposition(reply, context_bytes=1235)
    assert d.counts["review"] == 1
    assert d.counts["validate"] == 2
    assert d.context_bytes == 1235
    assert d.source == "agent"


def test_parse_drops_slugs_outside_the_vocabulary():
    """Inventing a primitive would corrupt the model's feature vector."""
    d = parse_decomposition('{"counts": {"review": 1, "telepathy": 9}}')
    assert d.counts["review"] == 1
    assert "telepathy" not in d.counts
    assert set(d.counts) == set(ORDER)


def test_parse_survives_junk_values():
    d = parse_decomposition('{"counts": {"review": "two", "draft": -3}}')
    assert d.counts["review"] == 0
    assert d.counts["draft"] == 0


def test_parse_rejects_a_reply_with_no_json():
    with pytest.raises(ValueError):
        parse_decomposition("I would rather not.")


def test_heuristic_is_available_and_admits_what_it_is():
    d = heuristic_decompose("please reconcile these and draft a memo")
    assert d.source == "heuristic"
    assert d.counts["reconcile"] == 1
    assert d.counts["draft"] == 1


def test_round_trip_prices_a_request(model):
    d = parse_decomposition(
        '{"counts": {"review": 1, "extract": 3, "validate": 2}}',
        context_bytes=1235)
    assert d.notation() == "Review + 3xExtract + 2xValidate"
    assert price(d, model) > model.coef[0]


def test_reconstruction_error_is_measured_against_truth(model):
    d = parse_decomposition('{"counts": {"reconcile": 1, "report": 1}}',
                            context_bytes=3799)
    assert reconstruction_error(d, model, 34_968) < 0.05


def test_explain_shows_every_term(model):
    d = parse_decomposition('{"counts": {"review": 1, "validate": 2}}',
                            context_bytes=1235)
    text = explain(d, model)
    assert "agent start-up" in text
    assert "context" in text
    assert "Validate" in text
    assert "predicted tokens" in text


def test_empty_decomposition_prices_to_the_boot_cost(model):
    d = Decomposition({s: 0 for s in ORDER})
    assert d.is_empty()
    assert price(d, model) == pytest.approx(model.coef[0], rel=1e-6)


# ── recorded end-to-end cases ────────────────────────────────────────────

def test_plain_english_cases_round_trip(model):
    """Requests written as a person would write them, priced before running."""
    import json
    path = REPO_ROOT / "experiments" / "decompose_cases.jsonl"
    cases = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(cases) >= 3
    errs = []
    for c in cases:
        counts = {s: 0 for s in ORDER}
        counts.update(c["encoded"])
        d = Decomposition(counts, c["rationale"], c["context_bytes"])
        errs.append(reconstruction_error(d, model, c["actual_tokens"]))
    assert sum(errs) / len(errs) < 0.10


# ── plumbing ─────────────────────────────────────────────────────────────

def test_mape_is_symmetric_in_magnitude():
    assert mape([100.0], [110.0]) == pytest.approx(0.1)


def test_model_equation_reads_as_an_equation(model):
    eq = model.equation()
    assert eq.startswith("tokens = ")
    assert "context_bytes" in eq


def test_predict_accepts_a_bare_counts_dict(model):
    assert model.predict({"review": 1}, 0) > 0
    assert isinstance(model, CompositionModel)


def test_run_total_units():
    r = Run("l", "n", {"review": 2, "draft": 1}, 0, 2, 1, 0, False)
    assert r.total_units == 3
