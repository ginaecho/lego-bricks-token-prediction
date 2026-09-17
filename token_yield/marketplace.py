"""Offline, versioned scoping-service selection and research-only quoting.

Services use the original brief independently. This module neither dispatches
work nor treats a schema pass as human acceptance or delivery completion.
"""

from __future__ import annotations

import json
import math
from datetime import date

from .customer_decomposition import (
    CATALOG_VERSION, OWNERS, SCOPING_BRICKS, _keys, check_output, content_hash,
    canonical_json, quote_features, validate_project, validate_template,
)
from .customer_models import (
    _count, _identifier, _mapping, _number, _required, _runtime_contract,
    _token_channels, forecast_tokens,
)
from .economics import Pricing
from .tasks import ORDER


_CATALOG_SCHEMA = "marketplace-services-v1"
_QUOTE_SCHEMA = "marketplace-selection-quote-v1"
_RATE_FIELDS = {"input_per_million", "cached_input_per_million", "output_per_million"}


def _object_schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _input_schema() -> dict:
    text = {"type": "string", "minLength": 1}
    return _object_schema({
        "id": {**text, "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$"},
        "buyer": text, "title": text, "source_url": {**text, "format": "uri"},
        "summary": text, "split": {"enum": ["train", "holdout"]},
        "requirements": {
            "type": "array", "minItems": 1,
            "items": _object_schema({
                "id": {**text, "pattern": "^R[1-9][0-9]*$"},
                "text": text, "brick": {"enum": list(ORDER)},
                "owner": {"enum": sorted(OWNERS)},
            }),
        },
        "exclusions": {"type": "array", "minItems": 1, "uniqueItems": True, "items": text},
    })


def _output_schema(slug: str) -> dict:
    requirement_id = {"type": "string", "pattern": "^R[1-9][0-9]*$"}
    if slug == "report":
        artifact = _object_schema({
            "source_url": {"type": "string", "format": "uri"},
            "requirement_ids": {"type": "array", "minItems": 1, "items": requirement_id},
            "exclusions": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "delivery_status": {"const": "not_executed"},
            "summary": {"type": "string", "minLength": 40},
        })
    else:
        properties = {"requirement_id": requirement_id}
        if slug == "extract":
            properties["evidence"] = {"type": "string"}
        else:
            properties["owner"] = {"enum": sorted(OWNERS)}
            if slug == "classify":
                properties["brick"] = {"enum": list(ORDER)}
            else:
                properties.update(
                    prerequisite_status={"const": "unverified"},
                    review_required={"const": True},
                    proposed_artifact={"type": "string", "minLength": 20},
                )
        artifact = {"type": "array", "minItems": 1, "items": _object_schema(properties)}
    return _object_schema({slug: artifact})


def service_catalog(template: dict, artifact: dict) -> dict:
    """Describe services bound to the trained template, with observed support.

    Schemas describe structure. Existing validators additionally check HTTPS,
    whitespace, ordered coverage, grounded values, and trimmed text lengths.
    """
    validate_template(template)
    _token_channels(artifact)
    runtime = _runtime_contract(artifact["runtime"])
    digest = content_hash(template)
    if runtime.get("template_sha256") != digest:
        raise ValueError("service template differs from the trained model contract")
    support = _mapping(_required(artifact, "support"), "support")
    ranges = _mapping(_required(support, "feature_ranges"), "feature_ranges")
    version = f"{_CATALOG_SCHEMA}:{digest}"
    catalog = {
        "schema_version": _CATALOG_SCHEMA,
        "template_sha256": digest,
        "model_sha256": content_hash(artifact),
        "scope": template["scope"],
        "services": [
            {
                "id": operation["id"], "version": version,
                "unit": operation["units"],
                "quantity_rule": ("one_per_project" if operation["id"] == "report"
                                  else "all_supplied_requirements_once"),
                "depends_on": operation["depends_on"],
                "instruction": operation["instruction"],
                "input_schema_version": CATALOG_VERSION,
                "input_schema": _input_schema(),
                "output_schema": _output_schema(operation["id"]),
                "input_validator": "token_yield.customer_decomposition.validate_project",
                "acceptance_checker": "token_yield.customer_decomposition.check_output",
                "human_review_required": True,
                "max_output_tokens": template["max_output_tokens_per_operation"],
                "observed_feature_ranges": ranges,
                "support_limitation": support["interpretation"],
            }
            for operation in template["operations"]
        ],
    }
    return json.loads(canonical_json(catalog))


def _selection(project: dict, catalog: dict, selections: list[dict]) -> tuple[str, ...]:
    validate_project(project)
    if not isinstance(selections, list) or not selections:
        raise ValueError("selections must be a nonempty list")
    services = {service["id"]: service for service in catalog["services"]}
    seen = set()
    for selection in selections:
        _keys(selection, {"service_id", "version", "quantity"}, "selection")
        slug = _identifier(selection["service_id"], "service_id")
        version = _identifier(selection["version"], "version")
        quantity = _count(selection["quantity"], "quantity")
        if slug not in services:
            raise ValueError(f"unknown service_id: {slug}")
        if slug in seen:
            raise ValueError(f"duplicate service selection: {slug}")
        seen.add(slug)
        if version != services[slug]["version"]:
            raise ValueError(f"unsupported service version: {slug}")
        expected = 1 if slug == "report" else len(project["requirements"])
        if quantity != expected:
            raise ValueError(f"{slug} quantity must equal {expected} for the supplied project")
    return tuple(slug for slug in SCOPING_BRICKS if slug in seen)


def quote_selection(
    project: dict, template: dict, artifact: dict, runtime: dict,
    selections: list[dict], *, execution_mode: str,
) -> dict:
    """Quote explicit independent services; abstain on extrapolation or mismatch.

    Separate mode means fresh, independent requests over the original brief,
    never a pipeline that passes generated answers between services.
    """
    catalog = service_catalog(template, artifact)
    operations = _selection(project, catalog, selections)
    runtime = _runtime_contract(runtime)
    if execution_mode not in ("separate", "batched"):
        raise ValueError("execution_mode must be separate or batched; sequential handoffs are unsupported")
    batches = tuple((slug,) for slug in operations) if execution_mode == "separate" else (operations,)
    calls = []
    reasons = []
    for index, batch in enumerate(batches):
        features = quote_features(project, template, batch)
        forecast = forecast_tokens(artifact, features, runtime)
        if forecast["support"]["status"] != "within_observed_ranges":
            reasons.extend(f"call_{index}:{reason}" for reason in forecast["support"]["reasons"])
        calls.append({
            "operations": list(batch), "quote_features": features,
            "point_estimate": (forecast["point_estimate"]
                               if forecast["support"]["status"] == "within_observed_ranges" else None),
            "support": forecast["support"],
        })
    total = None
    if not reasons:
        total = {
            channel: _number(math.fsum(call["point_estimate"][channel] for call in calls), channel)
            for channel in ("input_tokens", "output_tokens")
        }
        total["total_tokens"] = _number(math.fsum(total.values()), "total_tokens")
    result = {
        "schema_version": _QUOTE_SCHEMA,
        "project_sha256": content_hash(project),
        "template_sha256": catalog["template_sha256"],
        "model_sha256": catalog["model_sha256"],
        "runtime_sha256": content_hash(runtime),
        "selections": sorted(json.loads(canonical_json(selections)), key=lambda item: operations.index(item["service_id"])),
        "execution_mode": execution_mode,
        "calls": calls, "point_estimate": total,
        "status": "unsupported" if reasons else "research_estimate",
        "reasons": reasons,
        "calibrated_interval": None, "upper_budget_bound": None,
        "production_recommendation": None,
        "human_review_required": True,
        "scope": artifact["scope"],
        "composition_basis": ("sum_of_independent_fresh_calls_no_handoffs" if execution_mode == "separate"
                              else "direct_prediction_of_selected_batch_no_fixed_discount"),
    }
    return {**result, "quote_sha256": content_hash(result)}


def check_selection_output(
    project: dict, template: dict, artifact: dict, selections: list[dict], output: str,
) -> dict:
    """Apply the existing grounded checks, without certifying human acceptance."""
    operations = _selection(project, service_catalog(template, artifact), selections)
    return check_output(project, operations, output)


def rate_quote(
    quote: dict, rate_card: dict, *, tool_cost_usd: float | None = None,
    human_review_cost_usd: float | None = None, gross_margin_fraction: float | None = None,
) -> dict:
    """Rate a frozen quote without retraining or assuming cache discounts.

    Fees and gross margin are explicit caller assumptions, not model outputs.
    Selling price remains unknown until every commercial input is supplied.
    Gross margin means (selling price - cost) / selling price, not markup.
    """
    quote = _mapping(quote, "quote")
    if quote.get("schema_version") != _QUOTE_SCHEMA:
        raise ValueError("unsupported marketplace quote schema")
    unsigned = {key: value for key, value in quote.items() if key != "quote_sha256"}
    if content_hash(unsigned) != _required(quote, "quote_sha256"):
        raise ValueError("quote hash mismatch")
    status = _required(quote, "status")
    if status not in ("research_estimate", "unsupported"):
        raise ValueError("unknown marketplace quote status")
    reasons = _required(quote, "reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
        raise ValueError("quote reasons must be a list of nonempty strings")
    point = _required(quote, "point_estimate")
    if (status == "unsupported" and (point is not None or not reasons)
            or status == "research_estimate" and (point is None or reasons)):
        raise ValueError("quote status, reasons, and point estimate are inconsistent")
    if point is not None:
        _keys(point, {"input_tokens", "output_tokens", "total_tokens"}, "point_estimate")
        for name, value in point.items():
            _number(value, name)
        if not math.isclose(point["total_tokens"], point["input_tokens"] + point["output_tokens"],
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
    _keys(rate_card, {"id", "version", "currency", "runtime_sha256", "as_of", "rates", "source"},
          "rate_card")
    for name in ("id", "version", "currency", "runtime_sha256", "as_of"):
        _identifier(rate_card[name], name)
    if rate_card["currency"] != "USD":
        raise ValueError("only USD rate cards are supported; no currency conversion is assumed")
    if date.fromisoformat(rate_card["as_of"]).isoformat() != rate_card["as_of"]:
        raise ValueError("rate card as_of must use YYYY-MM-DD")
    source = _mapping(rate_card["source"], "rate card source")
    if not source:
        raise ValueError("rate card source must include provenance")
    _keys(rate_card["rates"], _RATE_FIELDS, "rates")
    rates = {name: _number(value, name) for name, value in rate_card["rates"].items()}
    pricing = Pricing(**rates)
    terms = {
        "tool_cost_usd": tool_cost_usd, "human_review_cost_usd": human_review_cost_usd,
        "gross_margin_fraction": gross_margin_fraction,
    }
    for name, value in terms.items():
        if value is not None:
            _number(value, name)
    if gross_margin_fraction is not None and gross_margin_fraction >= 1:
        raise ValueError("gross_margin_fraction must be in [0, 1)")
    reasons = list(reasons)
    if rate_card["runtime_sha256"] != quote["runtime_sha256"]:
        reasons.append("rate_card_runtime_mismatch")
    api_cost = None
    token_breakdown = None
    if not reasons:
        token_breakdown = {
            "uncached_input_tokens": point["input_tokens"],
            "cached_input_tokens": 0,
            "output_tokens": point["output_tokens"],
        }
        api_cost = _number(pricing.attempt_cost(
            input_tokens=point["input_tokens"], cached_input_tokens=0,
            output_tokens=point["output_tokens"]), "api_cost_usd")
    cost_basis = selling_price = margin_amount = None
    if api_cost is not None and tool_cost_usd is not None and human_review_cost_usd is not None:
        cost_basis = _number(math.fsum((api_cost, tool_cost_usd, human_review_cost_usd)),
                             "estimated_cost_basis_usd")
        if gross_margin_fraction is not None:
            selling_price = _number(cost_basis / (1 - gross_margin_fraction), "proposed_selling_price_usd")
            margin_amount = _number(selling_price - cost_basis, "gross_margin_usd")
    result = {
        "schema_version": "marketplace-rated-quote-v1",
        "quote_sha256": quote["quote_sha256"],
        "model_sha256": quote["model_sha256"],
        "rate_card_sha256": content_hash(rate_card),
        "rate_card": json.loads(canonical_json(rate_card)),
        "status": "unsupported" if reasons else "research_estimate",
        "reasons": reasons, "currency": "USD",
        "token_breakdown": token_breakdown,
        "cache_assumption": "all_input_uncached_no_measured_cache_forecast",
        "api_cost_usd": api_cost,
        **terms,
        "commercial_basis": "caller_declared_fees_and_margin_not_measured_or_model_predicted",
        "missing_commercial_inputs": [name for name, value in terms.items() if value is None],
        "estimated_cost_basis_usd": cost_basis,
        "proposed_selling_price_usd": selling_price,
        "gross_margin_usd": margin_amount,
        "calibrated_interval": None, "upper_budget_bound": None,
        "production_recommendation": None,
        "pricing_limitation": "Caller-supplied as-of rates, not verified live prices, invoices, tax, or discounts.",
    }
    return {**result, "rated_quote_sha256": content_hash(result)}
