"""Marketplace contracts tested offline; synthetic labels are not measurements."""

import copy
import json
from pathlib import Path

import pytest

from token_yield.customer_decomposition import (
    SCOPING_BRICKS, content_hash, load_catalog, load_template, quote_features,
)
from token_yield.customer_models import fit_token_models, forecast_tokens
from token_yield.economics import Pricing
from token_yield.marketplace import check_selection_output, quote_selection, rate_quote, service_catalog


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    template = load_template(ROOT / "experiments" / "customer_requests" / "template.json")
    projects = load_catalog(ROOT / "experiments" / "customer_requests" / "catalog.json")["projects"][:4]
    runtime = {"deployment": "synthetic-model", "deployment_version": "test-v1",
               "template_sha256": content_hash(template), "execution_policy": "no_tools_no_retries"}
    rows = []
    for project in projects:
        for index, operations in enumerate([*(tuple([slug]) for slug in SCOPING_BRICKS), SCOPING_BRICKS]):
            quote = quote_features(project, template, operations)
            inp = quote["prompt_bytes"] // 4
            out = 50 * len(operations)
            rows.append({
                "call_id": f"{project['id']}-{index}", "project_id": project["id"], "split": "train",
                "status": "completed", "quote": quote, "rated_cost_usd": 0.01,
                "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                          "cached_tokens": 0, "reasoning_tokens": 0},
            })
    artifact = fit_token_models(rows, runtime)
    return projects[0], template, artifact, runtime


def selections_for(project, template, artifact, slugs=SCOPING_BRICKS):
    version = service_catalog(template, artifact)["services"][0]["version"]
    return [{"service_id": slug, "version": version,
             "quantity": 1 if slug == "report" else len(project["requirements"])} for slug in slugs]


def test_catalog_is_versioned_and_preserves_scoping_semantics(inputs):
    project, template, artifact, _ = inputs
    original = copy.deepcopy((template, artifact))
    catalog = service_catalog(template, artifact)
    assert [service["id"] for service in catalog["services"]] == list(SCOPING_BRICKS)
    for service, operation in zip(catalog["services"], template["operations"]):
        assert service["instruction"] == operation["instruction"]
        assert service["unit"] == operation["units"]
        assert service["depends_on"] == []
        assert service["human_review_required"]
        assert service["acceptance_checker"].endswith(".check_output")
        assert service["input_schema"]["additionalProperties"] is False
        assert service["output_schema"]["required"] == [service["id"]]
        assert service["observed_feature_ranges"] == artifact["support"]["feature_ranges"]
    assert "preserve" in catalog["services"][1]["instruction"]
    catalog["services"][0]["observed_feature_ranges"]["prompt_bytes"][0] = -99
    assert (template, artifact) == original
    updated = copy.deepcopy(template)
    updated["operations"][0]["instruction"] += " Revised."
    with pytest.raises(ValueError, match="template"):
        service_catalog(updated, artifact)
    newer = copy.deepcopy(artifact)
    newer["runtime"]["template_sha256"] = content_hash(updated)
    assert service_catalog(updated, newer)["services"][0]["version"] != catalog["services"][0]["version"]


@pytest.mark.parametrize("mode", ["separate", "batched"])
def test_selection_uses_existing_quote_features_and_models(inputs, mode):
    project, template, artifact, runtime = inputs
    selection = selections_for(project, template, artifact)
    original = copy.deepcopy(selection)
    result = quote_selection(project, template, artifact, runtime, selection, execution_mode=mode)
    assert result["status"] == "research_estimate"
    assert len(result["calls"]) == (4 if mode == "separate" else 1)
    for call in result["calls"]:
        expected = quote_features(project, template, tuple(call["operations"]))
        assert call["quote_features"] == expected
        assert call["point_estimate"] == forecast_tokens(artifact, expected, runtime)["point_estimate"]
    for channel in ("input_tokens", "output_tokens"):
        assert result["point_estimate"][channel] == pytest.approx(
            sum(call["point_estimate"][channel] for call in result["calls"]))
    assert result["point_estimate"]["total_tokens"] == pytest.approx(
        result["point_estimate"]["input_tokens"] + result["point_estimate"]["output_tokens"])
    assert result["calibrated_interval"] is None and result["production_recommendation"] is None
    assert selection == original
    assert quote_selection(project, template, artifact, runtime, list(reversed(selection)),
                           execution_mode=mode) == result
    assert quote_selection(project, template, json.loads(json.dumps(artifact)), runtime, selection,
                           execution_mode=mode) == result


def test_individual_service_and_unmeasured_batch(inputs):
    project, template, artifact, runtime = inputs
    selection = selections_for(project, template, artifact, ["extract"])
    individual = quote_selection(project, template, artifact, runtime, selection, execution_mode="separate")
    assert individual["status"] == "research_estimate"
    assert individual["calls"][0]["quote_features"]["counts"] == {
        "extract": len(project["requirements"]), "classify": 0, "plan": 0, "report": 0}
    partial = selections_for(project, template, artifact, ["extract", "report"])
    result = quote_selection(project, template, artifact, runtime, partial, execution_mode="batched")
    assert result["status"] == "unsupported" and result["point_estimate"] is None
    assert result["calls"][0]["point_estimate"] is None
    assert any("unmeasured_operation_combination" in reason for reason in result["reasons"])
    separate = quote_selection(project, template, artifact, runtime, partial, execution_mode="separate")
    assert separate["status"] == "research_estimate" and len(separate["calls"]) == 2


@pytest.mark.parametrize("defect", ["empty", "duplicate", "unknown", "version", "zero", "negative",
                                    "boolean", "fraction", "quantity", "extra", "missing"])
def test_selection_validation(inputs, defect):
    project, template, artifact, runtime = inputs
    selection = selections_for(project, template, artifact)
    if defect == "empty":
        selection = []
    elif defect == "duplicate":
        selection.append(copy.deepcopy(selection[0]))
    elif defect == "unknown":
        selection[0]["service_id"] = "retrieve"
    elif defect == "version":
        selection[0]["version"] = "invented-v2"
    elif defect == "extra":
        selection[0]["discount"] = 0.5
    elif defect == "missing":
        del selection[0]["quantity"]
    else:
        selection[0]["quantity"] = {"zero": 0, "negative": -1, "boolean": True,
                                    "fraction": 1.5, "quantity": 1}[defect]
    with pytest.raises((TypeError, ValueError)):
        quote_selection(project, template, artifact, runtime, selection, execution_mode="separate")


def test_runtime_input_and_execution_modes_fail_closed(inputs):
    project, template, artifact, runtime = inputs
    selection = selections_for(project, template, artifact)
    changed = {**runtime, "execution_policy": "tools_enabled"}
    result = quote_selection(project, template, artifact, changed, selection, execution_mode="separate")
    assert result["point_estimate"] is None and result["status"] == "unsupported"
    assert all("runtime_contract_mismatch" in reason for reason in result["reasons"])
    with pytest.raises(ValueError, match="execution_mode"):
        quote_selection(project, template, artifact, runtime, selection, execution_mode="sequential")
    oversized = copy.deepcopy(project)
    oversized["summary"] *= 1000
    result = quote_selection(oversized, template, artifact, runtime, selection, execution_mode="separate")
    assert result["point_estimate"] is None and any("outside_training_range" in r for r in result["reasons"])
    project["requirements"][0]["id"] = "R999"
    with pytest.raises(ValueError, match="requirement ids"):
        quote_selection(project, template, artifact, runtime, selection, execution_mode="separate")


def test_acceptance_reuses_grounded_checks_not_language_model_judgment(inputs):
    project, template, artifact, _ = inputs
    selection = selections_for(project, template, artifact, ["extract"])
    output = {"extract": [{"requirement_id": r["id"], "evidence": r["text"]}
                          for r in project["requirements"]]}
    result = check_selection_output(project, template, artifact, selection, json.dumps(output))
    assert result == {"contract_passed": True, "human_review_required": True, "errors": []}
    output["extract"][0]["evidence"] = "Invented evidence"
    assert not check_selection_output(project, template, artifact, selection,
                                      json.dumps(output))["contract_passed"]


@pytest.fixture
def quoting(inputs):
    project, template, artifact, runtime = inputs
    selections = selections_for(project, template, artifact)
    quote = quote_selection(project, template, artifact, runtime, selections, execution_mode="separate")
    card = {
        "id": "synthetic-test-rates", "version": "1", "currency": "USD",
        "runtime_sha256": quote["runtime_sha256"], "as_of": "2026-09-16",
        "rates": {"input_per_million": 2.5, "cached_input_per_million": 0.25,
                  "output_per_million": 15.0},
        "source": {"kind": "synthetic_test_fixture_not_current_market_rates"},
    }
    return quote, card


def test_rating_reuses_pricing_and_keeps_unknown_fees_unknown(quoting):
    quote, card = quoting
    original = copy.deepcopy((quote, card))
    result = rate_quote(quote, card)
    expected = Pricing(**card["rates"]).attempt_cost(
        input_tokens=quote["point_estimate"]["input_tokens"], cached_input_tokens=0,
        output_tokens=quote["point_estimate"]["output_tokens"])
    assert result["api_cost_usd"] == expected
    assert result["token_breakdown"]["cached_input_tokens"] == 0
    assert result["token_breakdown"]["uncached_input_tokens"] == quote["point_estimate"]["input_tokens"]
    assert result["proposed_selling_price_usd"] is None and result["estimated_cost_basis_usd"] is None
    assert result["tool_cost_usd"] is None and result["human_review_cost_usd"] is None
    assert len(result["missing_commercial_inputs"]) == 3
    assert result["rate_card_sha256"] == content_hash(card)
    assert result["quote_sha256"] == quote["quote_sha256"]
    assert result["upper_budget_bound"] is None and result["production_recommendation"] is None
    result["rate_card"]["rates"]["input_per_million"] = 999
    assert (quote, card) == original


def test_commercial_gross_margin_is_not_markup(quoting):
    quote, card = quoting
    result = rate_quote(quote, card, tool_cost_usd=2.0, human_review_cost_usd=3.0, gross_margin_fraction=0.25)
    assert result["estimated_cost_basis_usd"] == pytest.approx(result["api_cost_usd"] + 5)
    assert result["proposed_selling_price_usd"] == pytest.approx(result["estimated_cost_basis_usd"] / 0.75)
    assert result["gross_margin_usd"] / result["proposed_selling_price_usd"] == pytest.approx(0.25)
    assert result["missing_commercial_inputs"] == []
    zero = rate_quote(quote, card, tool_cost_usd=0, human_review_cost_usd=0, gross_margin_fraction=0)
    assert zero["proposed_selling_price_usd"] == zero["api_cost_usd"]
    assert zero["gross_margin_usd"] == 0


def test_repricing_changes_price_identity_not_token_estimates(quoting):
    quote, card = quoting
    original = copy.deepcopy(quote)
    first = rate_quote(quote, card)
    card["version"] = "2"
    card["rates"]["input_per_million"] *= 2
    card["rates"]["output_per_million"] *= 2
    second = rate_quote(quote, card)
    assert second["api_cost_usd"] == pytest.approx(first["api_cost_usd"] * 2)
    assert second["rated_quote_sha256"] != first["rated_quote_sha256"]
    assert second["model_sha256"] == first["model_sha256"]
    assert quote == original
    card["rates"]["cached_input_per_million"] = 0
    assert rate_quote(quote, card)["api_cost_usd"] == second["api_cost_usd"]


@pytest.mark.parametrize("bad", [-1, True, "5", float("nan"), float("inf")])
def test_invalid_rates_and_fees_are_rejected(quoting, bad):
    quote, card = quoting
    with pytest.raises((TypeError, ValueError)):
        rate_quote(quote, card, tool_cost_usd=bad)
    with pytest.raises((TypeError, ValueError)):
        rate_quote(quote, card, human_review_cost_usd=bad)
    with pytest.raises((TypeError, ValueError)):
        rate_quote(quote, card, gross_margin_fraction=bad)
    card["rates"]["input_per_million"] = bad
    with pytest.raises((TypeError, ValueError)):
        rate_quote(quote, card)


@pytest.mark.parametrize("field,value", [
    ("currency", "EUR"), ("version", ""), ("as_of", "not-a-date"), ("as_of", "20260916"),
    ("source", {}), ("rates", {"input_per_million": 2.5}),
])
def test_invalid_rate_card_metadata_rejected(quoting, field, value):
    quote, card = quoting
    card[field] = value
    with pytest.raises((TypeError, ValueError)):
        rate_quote(quote, card)


def test_rating_cannot_turn_unsupported_forecast_into_a_price(inputs, quoting):
    project, template, artifact, runtime = inputs
    _, card = quoting
    selection = selections_for(project, template, artifact, ["extract", "report"])
    quote = quote_selection(project, template, artifact, runtime, selection, execution_mode="batched")
    result = rate_quote(quote, card, tool_cost_usd=0, human_review_cost_usd=0, gross_margin_fraction=0)
    assert result["status"] == "unsupported"
    assert result["api_cost_usd"] is None and result["proposed_selling_price_usd"] is None
    assert result["token_breakdown"] is None


def test_rate_runtime_mismatch_and_quote_tampering_fail_closed(quoting):
    quote, card = quoting
    card["runtime_sha256"] = "different-runtime"
    result = rate_quote(quote, card)
    assert result["status"] == "unsupported" and result["api_cost_usd"] is None
    assert result["reasons"] == ["rate_card_runtime_mismatch"]
    quote["point_estimate"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        rate_quote(quote, card)


@pytest.mark.parametrize("margin", [1, 1.5])
def test_invalid_gross_margin_denominator_rejected(quoting, margin):
    with pytest.raises(ValueError, match="gross_margin"):
        rate_quote(*quoting, gross_margin_fraction=margin)


def test_quote_example_uses_frozen_inputs_and_persists_results(tmp_path, monkeypatch, capsys):
    from examples.marketplace_quote import main

    destination = tmp_path / "quote"
    monkeypatch.setattr("sys.argv", [
        "marketplace_quote", "--project-id", "paradigm-website-redesign",
        "--services", "extract", "report", "--execution-mode", "separate",
        "--recorded-runtime", "--historical-rates", "--run-dir", str(destination),
    ])
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "research_estimate" and summary["spent_usd"] == 0
    quote = json.loads((destination / "quote.json").read_text(encoding="utf-8"))
    rated = json.loads((destination / "rated_quote.json").read_text(encoding="utf-8"))
    assert len(quote["calls"]) == 2
    assert rated["api_cost_usd"] == summary["api_cost_usd"]
    assert rated["proposed_selling_price_usd"] is None
    assert rated["quote_sha256"] == quote["quote_sha256"]
    assert (destination / "catalog.json").exists() and (destination / "selections.json").exists()
    with pytest.raises(FileExistsError):
        main()
