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
from project_yield.outcomes import ORDER, OUTCOMES
from project_yield.predict import Predictor
from project_yield.report import forecast_card, model_card, portfolio_table
from project_yield.roles import DEFAULT_ROSTER, load_roster
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
               contract_value=1.0, won=True, calendar_days=1.0,
               role_days={"software_engineer": 1.0})
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
    assert (linked.role("software_engineer").days_when_needed
            < alone.role("software_engineer").days_when_needed)
    assert linked.value("win_probability") > alone.value("win_probability")
    assert linked.value("calendar_days") < alone.value("calendar_days")


def test_holdout_carries_its_own_baseline(predictor):
    """A score is not interpretable without the do-nothing model beside it."""
    holdout = predictor.evaluate_holdout()
    assert holdout, "the corpus must keep a hold-out slice"
    for slug, score in holdout.items():
        assert score.n > 0
        assert score.baseline > 0
        assert score.metric in ("mape", "brier")


def test_the_money_heads_generalise_forward(predictor):
    """Contract value and duration must beat their base rate on unseen work.

    Deliberately not asserted for the win head: on the shipped corpus it has
    cross-validated skill and loses to its own base rate on the most recent
    engagements, which the model card and every forecast now say out loud. A
    test that demanded otherwise would be a test that the corpus flatters us.
    """
    holdout = predictor.evaluate_holdout()
    for slug in ("contract_value", "calendar_days"):
        assert holdout[slug].beats_baseline, (
            f"{slug}: {holdout[slug].score:.3f} vs base "
            f"{holdout[slug].baseline:.3f}")


def test_a_head_that_fails_forward_says_so(predictor, usecase):
    """Whenever cross-validation and the hold-out disagree, the forecast tells
    you — that disagreement is the most decision-relevant thing about a head."""
    holdout = predictor.evaluate_holdout()
    failing = [s for s in ORDER
               if s in holdout and not holdout[s].beats_baseline]
    warnings = predictor.forecast(usecase).warnings
    for slug in failing:
        assert any(OUTCOMES[slug].name.lower() in w and "base rate" in w
                   for w in warnings), slug


def test_intervals_bracket_the_point_estimate(predictor, usecase):
    forecast = predictor.forecast(usecase)
    for est in forecast.estimates.values():
        assert est.low <= est.value <= est.high
    for role in forecast.staffing:
        assert role.low <= role.days_when_needed <= role.high


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
    kwargs = dict(staff_days={"solution_architect": 5, "software_engineer": 20,
                              "project_manager": 4}, tokens=40000)
    sure = compute(100000, 1.0, **kwargs)
    coin = compute(100000, 0.5, **kwargs)
    assert sure.delivery_cost == pytest.approx(coin.delivery_cost)
    assert coin.expected_margin < sure.expected_margin


def test_breakeven_win_rate_is_where_expected_margin_crosses_zero():
    e = compute(200000, 0.5, {"solution_architect": 6, "software_engineer": 30,
                              "project_manager": 5}, 50000)
    at_breakeven = compute(200000, e.breakeven_win_rate,
                           {"solution_architect": 6, "software_engineer": 30,
                            "project_manager": 5}, 50000)
    assert at_breakeven.expected_margin == pytest.approx(0.0, abs=1.0)


def test_run_rate_is_separate_from_build_cost():
    quiet = compute(100000, 0.6, {"software_engineer": 20}, 40000,
                    monthly_runs=50)
    busy = compute(100000, 0.6, {"software_engineer": 20}, 40000,
                   monthly_runs=500000)
    assert quiet.delivery_cost == pytest.approx(busy.delivery_cost)
    assert not quiet.run_rate_dominates
    assert busy.run_rate_dominates


def test_rates_are_marked_as_placeholders():
    assert "PLACEHOLDER" in DEFAULT_RATES.source
    custom = DEFAULT_RATES.with_day_rates(software_engineer=999.0)
    assert custom.day_rate("software_engineer") == 999.0
    assert custom.day_rate("project_manager") == \
        DEFAULT_RATES.day_rate("project_manager")


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
    assert set(payload["outcomes"]) >= set(ORDER)
    assert json.loads(json.dumps(payload))["staffing"]
    assert payload["economics"]["total_staff_days"] > 0


def test_reports_render(predictor, usecase):
    f = predictor.forecast(usecase)
    assert usecase.title.upper() in forecast_card(f)
    assert "PLACEHOLDER" in forecast_card(f)
    assert "held out" in model_card(predictor.heads,
                                    predictor.evaluate_holdout())
    assert "PORTFOLIO" in portfolio_table([f])


def test_staff_outcomes_are_all_priced(predictor, usecase):
    f = predictor.forecast(usecase)
    assert set(f.staff_days) <= set(DEFAULT_ROSTER.slugs)
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


def test_encode_then_predict_is_the_single_button_path(predictor):
    """The page encodes and predicts on one press. Both halves must compose.

    The two-step version only ever made people press twice, so the button does
    the encode itself when the scope has not been set by hand — which means
    encode's output has to be a valid predict input without any editing.
    """
    from project_yield.app import YieldApp
    app = YieldApp(predictor)
    encoded = app.encode({"description": "Read 6 supplier contracts and "
                                         "reconcile the pricing terms."})
    forecast = app.predict(encoded)
    assert forecast["outcomes"]["contract_value"]["value"] > 0
    assert forecast["usecase"]["encoder"] == "heuristic"


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


# ── reading a folder of written descriptions ─────────────────────────────

def test_the_shipped_folder_loads_and_declares_its_lineage():
    from project_yield.casefiles import default_folder, read_folder
    cases = read_folder(default_folder())
    assert len(cases) >= 20
    assert all(c.title and not c.title.startswith("#") for c in cases)
    continuations = [c for c in cases if c.parent]
    assert continuations, "the folder must exercise lineage"
    ids = {c.uid for c in cases}
    for case in continuations:
        assert case.parent in ids
        # a continuation encoded before its parent is priced as greenfield
        assert [c.uid for c in cases].index(case.parent) < \
            [c.uid for c in cases].index(case.uid)


def test_readme_and_manifest_are_not_encoded_as_use_cases():
    from project_yield.casefiles import default_folder, read_folder
    names = {c.filename for c in read_folder(default_folder())}
    assert "README.md" not in names
    assert "manifest.jsonl" not in names


def test_manifest_naming_a_missing_file_is_an_error(tmp_path):
    from project_yield.casefiles import read_folder
    (tmp_path / "a.md").write_text("# A\nRead 3 contracts.")
    (tmp_path / "manifest.jsonl").write_text(
        '{"file": "a.md", "id": "A", "continues": "ghost.md"}\n')
    with pytest.raises(ValueError, match="not in the folder"):
        read_folder(str(tmp_path))


def test_a_folder_with_no_manifest_still_loads(tmp_path):
    from project_yield.casefiles import encode_folder
    (tmp_path / "one.md").write_text("# One\nRead 4 contracts and check them.")
    cases = encode_folder(str(tmp_path))
    assert len(cases) == 1 and cases[0].parent_id is None
    assert cases[0].counts["review"] == 4


def test_lineage_supplies_an_industry_the_prose_never_states(tmp_path):
    """A phase-two note that never repeats the sector is the normal case."""
    from project_yield.casefiles import encode_folder
    (tmp_path / "1.md").write_text(
        "# Phase one\nContoso is a retail group. Classify inbound tickets.")
    (tmp_path / "2.md").write_text(
        "# Phase two\nExtend the same triage to the European desks.")
    (tmp_path / "manifest.jsonl").write_text(
        '{"file": "1.md", "id": "P1"}\n'
        '{"file": "2.md", "id": "P2", "continues": "1.md"}\n')
    one, two = encode_folder(str(tmp_path))
    assert one.industry == "retail"
    assert two.industry == "retail"
    assert any("taken from P1" in a for a in two.assumptions)


def test_the_whole_folder_forecasts(predictor):
    from project_yield.casefiles import default_folder, encode_folder
    cases = encode_folder(default_folder())
    for case in cases:
        predictor.index.add(case)
    forecasts = [predictor.forecast(c) for c in cases]
    assert all(f.economics.contract_value > 0 for f in forecasts)
    # the folder is meant to span the space, not to be twenty of the same thing
    units = [c.total_units for c in cases]
    assert max(units) > 20 * min(units)
    wins = [f.value("win_probability") for f in forecasts]
    assert max(wins) - min(wins) > 0.25


def test_a_continuation_in_the_folder_beats_its_own_greenfield_twin(predictor):
    from project_yield.casefiles import default_folder, encode_folder
    cases = {c.id: c for c in encode_folder(default_folder())}
    child = next(c for c in cases.values() if c.parent_id)
    for case in cases.values():
        predictor.index.add(case)
    twin = UseCase.from_dict(dict(child.to_dict(), id=child.id + "-alone",
                                  parent_id=None))
    assert predictor.forecast(child).value("win_probability") > \
        predictor.forecast(twin).value("win_probability")


# ── the encoder reads prose, not just digits ─────────────────────────────

def test_spelled_out_quantities_are_read():
    """Scoping notes spell out small numbers. Missing them flattens every use
    case to the same size, which is the failure that makes a portfolio view
    useless."""
    uc = heuristic_encode("Read the invoice, extract nine header fields and "
                          "write three validation checks.")
    assert uc.counts["extract"] == 9
    assert uc.counts["validate"] == 3


def test_a_quantity_attaches_to_the_nearest_task_only():
    """One number, one brick — not nine of everything."""
    uc = heuristic_encode("Read the invoice, extract nine header fields, "
                          "then classify it.")
    assert uc.counts["extract"] == 9
    assert uc.counts["review"] == 1
    assert uc.counts["classify"] == 1


def test_a_duration_is_not_a_quantity_of_work():
    uc = heuristic_encode("Review the pack. It currently takes 48 hours and "
                          "three years of history are in scope.")
    assert uc.counts["review"] == 1


def test_a_defaulted_industry_is_recorded_as_an_assumption():
    uc = heuristic_encode("Somebody wants an assistant that finds things.")
    assert any("No industry is named" in a for a in uc.assumptions)


def test_a_stated_industry_produces_no_assumption_about_it():
    uc = heuristic_encode("Read 4 patient records at the hospital and "
                          "classify them to cut manual effort.")
    assert uc.industry == "healthcare"
    assert not any("industry" in a for a in uc.assumptions)


def test_encoder_assumptions_reach_the_forecast(predictor):
    uc = heuristic_encode("Somebody wants an assistant that finds things.",
                          uid="VAGUE")
    warnings = predictor.forecast(uc).warnings
    assert any("No industry is named" in w for w in warnings)


def test_a_markdown_heading_does_not_become_the_title():
    uc = heuristic_encode("# Northwind invoice intake\n\nRead 3 invoices.")
    assert uc.title == "Northwind invoice intake"


def test_provenance_survives_the_encode_then_predict_round_trip(predictor):
    """A keyword-derived estimate must not render like a hand-typed one.

    The page encodes and predicts in two calls, so the fact that a model never
    read the description is exactly the fact most easily lost in between.
    """
    from project_yield.app import YieldApp
    app = YieldApp(predictor)
    encoded = app.encode({"description": "Somebody wants an assistant that "
                                         "finds things for the team."})
    forecast = app.predict(dict(encoded, encoder="heuristic"))
    assert any("keyword match" in w for w in forecast["warnings"])
    assert any("No industry is named" in w for w in forecast["warnings"])


def test_overruling_the_encoder_retires_its_assumption(predictor):
    """Telling it the industry should stop it saying the industry was guessed."""
    from project_yield.app import YieldApp
    app = YieldApp(predictor)
    encoded = app.encode({"description": "Somebody wants an assistant that "
                                         "finds things for the team."})
    chosen = app.predict(dict(encoded, encoder="heuristic", industry="energy"))
    assert not any("No industry is named" in w for w in chosen["warnings"])
    assert any("keyword match" in w for w in chosen["warnings"])


def test_a_hand_specified_use_case_carries_no_encoder_assumptions(predictor):
    from project_yield.app import YieldApp
    forecast = YieldApp(predictor).predict({
        "title": "typed by hand", "industry": "energy",
        "goal": "compliance_risk", "counts": {"review": 4, "validate": 5}})
    assert not any("keyword match" in w for w in forecast["warnings"])


# ── the roster is data, not code ─────────────────────────────────────────

def test_the_roster_covers_the_roles_a_delivery_team_actually_has():
    names = {r.slug for r in DEFAULT_ROSTER}
    for expected in ("solution_architect", "consultant", "data_engineer",
                     "data_scientist", "security_expert", "project_manager"):
        assert expected in names, expected
    assert all(r.day_rate > 0 for r in DEFAULT_ROSTER)


def test_the_shipped_roles_json_matches_the_built_in_roster():
    """roles.json is the file a user edits. It must start as the default."""
    on_disk = load_roster(str(REPO_ROOT / "roles.json"))
    assert on_disk.slugs == DEFAULT_ROSTER.slugs
    assert on_disk.rates() == DEFAULT_ROSTER.rates()


def test_an_unknown_role_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="solution_architect"):
        DEFAULT_ROSTER["chief_vibes_officer"]
    with pytest.raises(ValueError, match="unknown role"):
        DEFAULT_ROSTER.validate(["chief_vibes_officer"])


def test_a_roster_role_with_no_history_gets_no_head_and_says_why(tmp_path):
    """Adding a role must never invent a number for it."""
    roles = tmp_path / "roles.json"
    roles.write_text(json.dumps({"roles": DEFAULT_ROSTER.to_list() + [
        {"slug": "mlops_engineer", "name": "MLOps engineer",
         "day_rate": 1350.0}]}))
    predictor = Predictor.from_defaults(roster=load_roster(str(roles)))
    assert "mlops_engineer_days" not in predictor.heads
    assert "mlops_engineer_days" in predictor.heads.unfitted


def test_naming_a_role_with_no_history_warns_rather_than_dropping_it(tmp_path):
    roles = tmp_path / "roles.json"
    roles.write_text(json.dumps({"roles": DEFAULT_ROSTER.to_list() + [
        {"slug": "mlops_engineer", "name": "MLOps engineer",
         "day_rate": 1350.0}]}))
    predictor = Predictor.from_defaults(roster=load_roster(str(roles)))
    uc = UseCase("M", "needs mlops", industry="retail", goal="cost_reduction",
                 counts={"extract": 6}, roles=["software_engineer",
                                               "mlops_engineer"])
    forecast = predictor.forecast(uc)
    assert any("MLOps engineer" in w and "no history" in w
               for w in forecast.warnings)
    assert forecast.role("mlops_engineer") is not None    # on the plan at zero


# ── presence and days are separate questions ─────────────────────────────

def test_days_are_days_when_needed_not_an_average_over_everyone(predictor,
                                                                usecase):
    """A role used half the time must not report half its days as its days."""
    for role in predictor.forecast(usecase).staffing:
        if not role.is_certain:
            assert role.days_when_needed > role.expected_days
            assert role.expected_days == pytest.approx(
                role.probability * role.days_when_needed)


def test_naming_the_team_overrides_the_base_rates(predictor):
    base = dict(industry="healthcare", goal="compliance_risk",
                counts={"review": 4, "extract": 6, "validate": 5},
                context_bytes=18000)
    inferred = predictor.forecast(UseCase("A", "inferred", **base))
    named = predictor.forecast(UseCase(
        "B", "named", roles=["solution_architect", "data_scientist",
                             "security_expert"], **base))
    assert {r.role.slug for r in named.staffing} == {
        "solution_architect", "data_scientist", "security_expert"}
    assert all(r.is_certain for r in named.staffing)
    assert len(inferred.staffing) > len(named.staffing)
    assert any("specified rather than predicted" in w for w in named.warnings)


def test_entered_days_win_over_any_prediction(predictor):
    uc = UseCase("E", "hand priced", industry="energy", goal="compliance_risk",
                 counts={"review": 5, "reconcile": 4},
                 role_days={"software_engineer": 42.0})
    role = predictor.forecast(uc).role("software_engineer")
    assert role.days_when_needed == pytest.approx(42.0)
    assert role.source == "entered" and role.is_certain


def test_a_presence_head_with_no_skill_is_replaced_by_the_base_rate(predictor):
    """Not merely flagged. For a role staffed on 97% of jobs there is nothing
    to predict, and the base rate is the better answer."""
    forecast = predictor.forecast(UseCase(
        "S", "s", industry="retail", goal="cost_reduction",
        counts={"extract": 8, "classify": 5}))
    sources = {r.source for r in forecast.staffing}
    assert sources <= {"predicted", "base rate", "assumed"}
    for role in forecast.staffing:
        if role.source == "base rate":
            head = predictor.heads[role.role.used_outcome]
            assert not head.beats_baseline
            assert role.probability == pytest.approx(head.baseline_value)


def test_cost_uses_expected_days_and_the_roster_rate(predictor, usecase):
    f = predictor.forecast(usecase)
    for role in f.staffing:
        assert role.cost == pytest.approx(
            role.expected_days * DEFAULT_ROSTER[role.role.slug].day_rate)
    assert f.economics.delivery_cost == pytest.approx(
        sum(r.cost for r in f.staffing) + f.economics.token_cost)


# ── the client-side impact ───────────────────────────────────────────────

def test_impact_scales_with_volume_not_with_scope():
    from project_yield.impact import compute
    counts = {"review": 1, "extract": 9, "validate": 3}
    quiet = compute(counts, 100, 500, 50000)
    busy = compute(counts, 100000, 500, 50000)
    assert quiet.minutes_per_run == busy.minutes_per_run
    assert busy.annual_benefit > 100 * quiet.annual_benefit * 0.99


def test_impact_is_not_quoted_without_a_run_rate():
    from project_yield.impact import compute
    i = compute({"review": 4}, 0, 0, 50000)
    assert not i.quoted and i.payback_months is None
    assert "run rate" in i.verdict


def test_impact_nets_off_what_it_costs_to_run():
    from project_yield.impact import compute
    cheap = compute({"extract": 5}, 10000, 1_000, 40000)
    dear = compute({"extract": 5}, 10000, 10_000_000, 40000)
    assert cheap.annual_net_benefit > dear.annual_net_benefit
    assert dear.payback_months is None          # running costs eat the benefit


def test_deflection_below_one_is_the_default():
    """Assuming automation removes all the handling is the commonest way a
    benefit case is overstated."""
    from project_yield.impact import DEFAULT_ASSUMPTIONS
    assert 0 < DEFAULT_ASSUMPTIONS.deflection < 1
    assert "PLACEHOLDER" in DEFAULT_ASSUMPTIONS.source


def test_the_impact_assumptions_are_stated_on_every_forecast(predictor,
                                                             usecase):
    warnings = predictor.forecast(usecase).warnings
    assert any("minutes on one of these by hand" in w for w in warnings)


def test_a_volume_use_case_is_worth_doing(predictor):
    """The shape the whole tool exists for: small per-run scope, large volume.

    If this comes out negative the prototype is not demonstrating anything —
    but it must come out positive because of the arithmetic, not because the
    arithmetic was arranged to.
    """
    uc = UseCase("V", "invoice intake", industry="manufacturing",
                 goal="cost_reduction",
                 counts={"review": 1, "extract": 9, "classify": 1,
                         "validate": 3, "remediate": 1},
                 context_bytes=20000, monthly_runs=24000)
    f = predictor.forecast(uc)
    assert f.impact.is_positive
    assert f.impact.fte_equivalent > 1
    assert f.impact.payback_months < 24
    assert f.economics.gross_margin > 0
