"""Tests for the multi-outcome prototype: heads, lineage, encoding, economics.

The interesting tests here are not "does it return a number". They are:

* does each head recover the relationship the synthetic generator actually put
  in the data (the one check a real corpus could never give you), and
* does the thing refuse to look confident when it should not — a warning
  missing on an extrapolated estimate is a worse defect than a wrong number,
  because a wrong number with a caveat is still usable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from project_yield import linalg
from project_yield.corpus import load_engagements, summarise
from project_yield.economics import DEFAULT_RATES, compute
from project_yield.encode import (encode_prompt, heuristic_encode,
                                  parse_encoding)
from project_yield.features import FORMS, FeatureRow, build, names, width
from project_yield.lineage import LineageIndex
from project_yield.multihead import INTERVAL
from project_yield.outcomes import ORDER, OUTCOMES, STAFF_OUTCOMES
from project_yield.predict import Predictor
from project_yield.report import forecast_card, model_card, portfolio_table
from project_yield.usecase import GOALS, INDUSTRIES, UseCase

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return load_engagements()


@pytest.fixture(scope="module")
def predictor():
    return Predictor.from_defaults()


@pytest.fixture
def usecase():
    return UseCase(
        id="T1", title="Invoice intake", industry="manufacturing",
        goal="cost_reduction",
        counts={"review": 2, "extract": 9, "classify": 4, "validate": 3},
        context_bytes=20000, monthly_runs=12000,
    )


# ── linear algebra ───────────────────────────────────────────────────────

def test_leverage_sums_to_the_number_of_parameters():
    """A basic identity: trace(H) = k. If it fails, leave-one-out is wrong.

    Slightly under k rather than exactly k: the ridge term shrinks the trace,
    which is what a penalty is for. The tolerance is loose enough to allow that
    and far too tight to allow an actual error in the hat diagonal.
    """
    x = [[1.0, float(i), float(i * i)] for i in range(12)]
    assert sum(linalg.leverage(x)) == pytest.approx(3.0, abs=1e-3)


def test_press_identity_matches_brute_force_leave_one_out():
    """The closed-form deletion residual must equal actually refitting."""
    x = [[1.0, float(i), float((i * 7) % 5)] for i in range(15)]
    y = [3.0 + 2.0 * r[1] - 1.5 * r[2] + ((i * 13) % 7) * 0.3
         for i, r in enumerate(x)]
    coef = linalg.least_squares(x, y)
    h = linalg.leverage(x)
    for i in range(len(y)):
        rest_x = x[:i] + x[i + 1:]
        rest_y = y[:i] + y[i + 1:]
        refit = linalg.least_squares(rest_x, rest_y)
        brute = y[i] - sum(c * f for c, f in zip(refit, x[i]))
        fitted = sum(c * f for c, f in zip(coef, x[i]))
        closed = (y[i] - fitted) / (1.0 - h[i])
        assert closed == pytest.approx(brute, rel=1e-4)


def test_logistic_never_returns_certainty_on_separable_data():
    """Separable data must not produce a win probability of 1.0.

    A model that says a deal is certain is the most expensive kind of wrong,
    and separability is common in a corpus where big regulated programmes
    always land.
    """
    x = [[1.0, float(i)] for i in range(10)]
    y = [0.0] * 5 + [1.0] * 5
    beta = linalg.irls_logistic(x, y)
    probs = [linalg.sigmoid(sum(b * f for b, f in zip(beta, xi))) for xi in x]
    assert all(0.001 < p < 0.999 for p in probs), probs


# ── the corpus ───────────────────────────────────────────────────────────

def test_corpus_is_labelled_synthetic(corpus):
    """Nobody should be able to mistake the shipped corpus for evidence."""
    assert all(e.provenance == "synthetic" for e in corpus)
    assert "synthetic" in summarise(corpus)


def test_corpus_columns_are_validated(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "X", "industry": "manufacturing",
                               "goal": "cost_reduction", "counts": {}}) + "\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_engagements(str(bad))


def test_corpus_rejects_an_unknown_industry(tmp_path):
    good = load_engagements()[0]
    row = dict(id=good.id, title="", client="", industry="fintech",
               goal=good.goal, counts=good.counts, context_bytes=0,
               contract_value=1.0, won=True, architect_days=1.0,
               engineer_days=1.0, pm_days=1.0, calendar_days=1.0)
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unknown industry"):
        load_engagements(str(path))


def test_parents_always_precede_their_children(corpus):
    """Lineage must be acyclic and temporally coherent, or reuse is nonsense."""
    seen = set()
    for eng in corpus:
        if eng.parent_id:
            assert eng.parent_id in seen, f"{eng.id} precedes its parent"
        seen.add(eng.id)


# ── lineage ──────────────────────────────────────────────────────────────

def test_inherited_fraction_is_bounded_and_meaningful():
    root = UseCase("A", "root", counts={"extract": 10, "classify": 5})
    same = UseCase("B", "same", counts={"extract": 10, "classify": 5},
                   parent_id="A")
    fresh = UseCase("C", "fresh", counts={"draft": 8}, parent_id="A")
    idx = LineageIndex([root, same, fresh])
    assert idx.features_for(same).inherited_fraction == pytest.approx(1.0)
    assert idx.features_for(fresh).inherited_fraction == pytest.approx(0.0)
    assert idx.features_for(root).inherited_fraction == pytest.approx(0.0)


def test_siblings_are_inferred_from_a_shared_parent():
    """Naming one sibling must not be penalised versus naming all of them."""
    root = UseCase("A", "root", counts={"extract": 3})
    b = UseCase("B", "b", counts={"extract": 3}, parent_id="A")
    c = UseCase("C", "c", counts={"extract": 3}, parent_id="A")
    idx = LineageIndex([root, b, c])
    assert idx.features_for(b).sibling_count == 1
    assert {u.id for u in idx.siblings(b)} == {"C"}


def test_lineage_survives_a_cycle():
    a = UseCase("A", "a", counts={"extract": 1}, parent_id="B")
    b = UseCase("B", "b", counts={"extract": 1}, parent_id="A")
    idx = LineageIndex([a, b])
    assert idx.features_for(a).reuse_depth <= 4          # terminates


def test_a_missing_parent_is_not_silently_ignored(predictor):
    uc = UseCase("Z", "orphan", industry="retail", goal="cost_reduction",
                 counts={"extract": 5}, parent_id="NOT-A-REAL-ID")
    warnings = predictor.forecast(uc).warnings
    assert any("not in the library" in w for w in warnings)


# ── features ─────────────────────────────────────────────────────────────

def test_every_form_has_names_matching_its_width():
    row = FeatureRow({"extract": 3}, 100, "retail", "cost_reduction")
    for form in FORMS:
        assert len(build(form, row)) == width(form) == len(names(form))


def test_forms_are_ordered_from_simplest_to_richest():
    widths = [width(f) for f in FORMS]
    assert widths == sorted(widths)


def test_count_features_are_compressed_so_scope_does_not_explode():
    """log1p means a power law. Linear counts under a log link would make cost
    grow exponentially in scope, which is unusable at any real volume."""
    small = FeatureRow({"extract": 10}, 0, "retail", "cost_reduction")
    big = FeatureRow({"extract": 1000}, 0, "retail", "cost_reduction")
    # a hundredfold in scope must move the feature by less than threefold
    assert build("size", big)[1] / build("size", small)[1] < 3.0


# ── the heads ────────────────────────────────────────────────────────────

def test_every_head_beats_its_own_baseline_except_where_it_says_so(predictor):
    for slug in ORDER:
        head = predictor.heads[slug]
        assert head.loo_score >= 0
        if not head.beats_baseline:
            # allowed, but it must be surfaced rather than quietly shipped
            assert "does not beat" in predictor._warnings(
                UseCase("W", "w", industry="retail", goal="cost_reduction",
                        counts={"extract": 5}),
                predictor.feature_row(UseCase(
                    "W", "w", industry="retail", goal="cost_reduction",
                    counts={"extract": 5})))[-1]


def test_heads_recover_the_generator_reuse_effect(predictor):
    """The generator discounts engineering by reuse. The head must find it.

    This is the check a corpus of real engagements cannot give you: here the
    true relationship is known, so a failure is unambiguously the fitting code
    rather than a weak signal.
    """
    base = dict(industry="manufacturing", goal="cost_reduction",
                counts={"review": 2, "extract": 8, "classify": 4,
                        "validate": 3, "reconcile": 3}, context_bytes=26000)
    parent = UseCase("P", "parent", **base)
    child = UseCase("C", "child", parent_id="P", **base)
    predictor.index.add(parent)
    predictor.index.add(child)

    alone = predictor.forecast(UseCase("C2", "child alone", **base))
    linked = predictor.forecast(child)
    assert linked.value("engineer_days") < alone.value("engineer_days")
    assert linked.value("win_probability") > alone.value("win_probability")
    assert linked.value("calendar_days") < alone.value("calendar_days")


def test_holdout_agrees_with_cross_validation(predictor):
    """A gap between the two would mean the form selection fitted the corpus."""
    holdout = predictor.evaluate_holdout()
    assert holdout, "the corpus must keep a hold-out slice"
    for slug, (score, n) in holdout.items():
        assert n > 0
        cv = predictor.heads[slug].loo_score
        assert score < cv * 2.0 + 0.15, f"{slug}: {score:.3f} vs cv {cv:.3f}"


def test_intervals_bracket_the_point_estimate(predictor, usecase):
    for est in predictor.forecast(usecase).estimates.values():
        assert est.low <= est.value <= est.high
        assert est.low > 0 or OUTCOMES[est.outcome].binary


def test_win_probability_stays_a_probability(predictor):
    for units in (1, 5, 50, 500):
        uc = UseCase(f"P{units}", "scale", industry="public_sector",
                     goal="revenue_growth", counts={"extract": units})
        est = predictor.forecast(uc).estimates["win_probability"]
        assert 0.0 < est.low <= est.value <= est.high < 1.0


def test_interval_is_the_advertised_width():
    assert INTERVAL == pytest.approx(0.80)


# ── encoding ─────────────────────────────────────────────────────────────

def test_encode_prompt_offers_every_vocabulary_term():
    prompt = encode_prompt("anything")
    for term in list(INDUSTRIES) + list(GOALS):
        assert term in prompt
    assert "ONE end-to-end run" in prompt


def test_parse_encoding_drops_unknown_slugs_and_levels():
    uc = parse_encoding('{"counts": {"extract": 4, "teleport": 9}, '
                        '"industry": "atlantis", "goal": "world_peace", '
                        '"context_bytes": 100, "monthly_runs": 5}')
    assert uc.counts["extract"] == 4
    assert "teleport" not in uc.counts
    assert uc.industry in INDUSTRIES and uc.goal in GOALS
    assert uc.encoder == "agent"


def test_parse_encoding_tolerates_a_fenced_reply():
    uc = parse_encoding('Here you go:\n```json\n{"counts": {"draft": 2}}\n```')
    assert uc.counts["draft"] == 2


def test_parse_encoding_rejects_a_reply_with_no_json():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_encoding("I could not work out what this use case is.")


def test_heuristic_reads_throughput_as_volume_not_as_scope():
    """"400 claims a day" is 12,000 runs a month of one claim, not 400 per run.

    Counting it as both would multiply the scope by the volume — the single
    most expensive mistake the fallback encoder could make.
    """
    uc = heuristic_encode("Classify around 400 claims documents a day and "
                          "extract the policy number.")
    assert uc.monthly_runs == 12000
    assert uc.total_units < 10
    assert "throughput" in uc.rationale


def test_heuristic_reads_a_fixed_quantity_as_scope():
    uc = heuristic_encode("Read 12 supplier contracts and reconcile them.")
    assert uc.counts["review"] == 12
    assert uc.monthly_runs == 0


def test_heuristic_always_says_it_is_the_heuristic():
    uc = heuristic_encode("do something vague")
    assert uc.encoder == "heuristic"
    assert uc.total_units >= 1


# ── economics ────────────────────────────────────────────────────────────

def test_cost_is_committed_and_revenue_is_contingent():
    """The win probability must discount value only — never cost."""
    kwargs = dict(staff_days={"architect_days": 5, "engineer_days": 20,
                              "pm_days": 4}, tokens=40000)
    sure = compute(100000, 1.0, **kwargs)
    coin = compute(100000, 0.5, **kwargs)
    assert sure.delivery_cost == pytest.approx(coin.delivery_cost)
    assert coin.expected_margin < sure.expected_margin


def test_breakeven_win_rate_is_where_expected_margin_crosses_zero():
    e = compute(200000, 0.5, {"architect_days": 6, "engineer_days": 30,
                              "pm_days": 5}, 50000)
    at_breakeven = compute(200000, e.breakeven_win_rate,
                           {"architect_days": 6, "engineer_days": 30,
                            "pm_days": 5}, 50000)
    assert at_breakeven.expected_margin == pytest.approx(0.0, abs=1.0)


def test_run_rate_is_separate_from_build_cost():
    quiet = compute(100000, 0.6, {"engineer_days": 20}, 40000, monthly_runs=50)
    busy = compute(100000, 0.6, {"engineer_days": 20}, 40000,
                   monthly_runs=500000)
    assert quiet.delivery_cost == pytest.approx(busy.delivery_cost)
    assert not quiet.run_rate_dominates
    assert busy.run_rate_dominates


def test_rates_are_marked_as_placeholders():
    assert "PLACEHOLDER" in DEFAULT_RATES.source
    custom = DEFAULT_RATES.with_rates(engineer_day=999.0, source="internal")
    assert custom.engineer_day == 999.0 and custom.architect_day == \
        DEFAULT_RATES.architect_day


# ── the whole thing ──────────────────────────────────────────────────────

def test_forecast_always_warns_that_the_corpus_is_synthetic(predictor, usecase):
    warnings = predictor.forecast(usecase).warnings
    assert any("SYNTHETIC" in w for w in warnings)


def test_extrapolating_past_the_corpus_is_reported(predictor):
    huge = UseCase("H", "enormous", industry="retail", goal="cost_reduction",
                   counts={"extract": 100000})
    assert any("larger than anything in the corpus" in w
               for w in predictor.forecast(huge).warnings)


def test_an_unusual_brick_mix_is_reported(predictor):
    odd = UseCase("O", "all drafting", industry="retail",
                  goal="revenue_growth", counts={"draft": 40})
    warnings = predictor.forecast(odd).warnings
    assert any("extrapolating in shape" in w for w in warnings)


def test_the_token_head_is_the_measured_model_unchanged(predictor):
    from token_yield.compose import default_runs_path, load_runs, select_model
    assert predictor.token_model.form == select_model(
        load_runs(default_runs_path())).form


def test_forecast_serialises_to_json(predictor, usecase):
    payload = predictor.forecast(usecase).to_dict()
    assert json.loads(json.dumps(payload))["outcomes"].keys() == set(ORDER)
    assert payload["economics"]["total_staff_days"] > 0


def test_reports_render(predictor, usecase):
    f = predictor.forecast(usecase)
    assert usecase.title.upper() in forecast_card(f)
    assert "PLACEHOLDER" in forecast_card(f)
    assert "cross-validated" in model_card(predictor.heads,
                                           predictor.evaluate_holdout())
    assert "PORTFOLIO" in portfolio_table([f])


def test_staff_outcomes_are_all_priced(predictor, usecase):
    f = predictor.forecast(usecase)
    assert set(f.staff_days) == set(STAFF_OUTCOMES)
    assert f.economics.total_staff_days == pytest.approx(
        sum(f.staff_days.values()))


# ── the app, as a library ────────────────────────────────────────────────

def test_app_endpoints_return_json(predictor):
    from project_yield.app import YieldApp
    app = YieldApp(predictor)
    assert set(app.meta()["goals"]) == set(GOALS)
    encoded = app.encode({"description": "Read 5 contracts and check them."})
    assert encoded["counts"]["review"] == 5
    forecast = app.predict({"title": "t", "industry": "retail",
                            "goal": "cost_reduction",
                            "counts": {"extract": 6}, "context_bytes": 1000})
    assert forecast["outcomes"]["contract_value"]["value"] > 0
    model = app.model()
    assert model["token_model"]["provenance"] == "measured agent runs"


def test_app_refuses_an_empty_scope(predictor):
    from project_yield.app import YieldApp
    with pytest.raises(ValueError, match="no task bricks"):
        YieldApp(predictor).predict({"counts": {}})


def test_a_use_case_scoped_now_can_be_a_parent_next(predictor):
    from project_yield.app import YieldApp
    app = YieldApp(predictor)
    first = app.predict({"title": "phase 1", "industry": "energy",
                         "goal": "compliance_risk",
                         "counts": {"review": 4, "validate": 5}})
    uid = first["usecase"]["id"]
    second = app.predict({"title": "phase 2", "industry": "energy",
                          "goal": "compliance_risk",
                          "counts": {"review": 4, "validate": 5},
                          "parent_id": uid})
    assert second["lineage"]["reuse_depth"] == 1
    assert second["lineage"]["inherited_fraction"] == pytest.approx(1.0)


# ── the Azure seams ──────────────────────────────────────────────────────

def test_azure_imports_without_any_azure_library():
    import project_yield.azure as azure
    assert azure.foundry_encoder_if_configured.__doc__


def test_foundry_encoder_needs_configuration(monkeypatch):
    from project_yield.azure import (ENV_DEPLOYMENT, ENV_ENDPOINT,
                                     FoundryEncoder,
                                     foundry_encoder_if_configured)
    monkeypatch.delenv(ENV_ENDPOINT, raising=False)
    monkeypatch.delenv(ENV_DEPLOYMENT, raising=False)
    assert foundry_encoder_if_configured() is None
    with pytest.raises(ValueError, match=ENV_ENDPOINT):
        FoundryEncoder()


def test_foundry_encoder_builds_the_deployment_url(monkeypatch):
    from project_yield.azure import FoundryEncoder
    enc = FoundryEncoder(endpoint="https://x.services.ai.azure.com/",
                         deployment="gpt-4o", key="k")
    assert enc.url.startswith("https://x.services.ai.azure.com/openai/"
                              "deployments/gpt-4o/chat/completions")
    assert enc._headers()["api-key"] == "k"


def test_fabric_query_selects_exactly_the_required_columns():
    from project_yield.azure import REQUIRED_COLUMNS, FabricCorpus
    fabric = FabricCorpus(sql_endpoint="x.fabric.microsoft.com", database="wh")
    for column in REQUIRED_COLUMNS:
        assert column in fabric.query()
    assert "ActiveDirectoryDefault" in fabric.connection_string()


def test_generator_and_corpus_stay_in_step():
    """Regenerating must reproduce the committed corpus byte for byte."""
    import subprocess
    out = subprocess.run(
        ["python", "-m", "experiments.make_engagements"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    committed = (REPO_ROOT / "experiments" / "engagements.jsonl").read_text()
    assert out == committed, "run: make corpus"
