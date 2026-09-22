"""Deterministic interest matching; run: python implementation.py input.json.

Preferences/exclusions are attribute/value rules. Product attributes contain
scalars or nonempty lists of scalars. Strings match after stripping/casefolding;
numbers match numerically, and booleans never match numbers. Positive integer
preference weights are additive. Exclusions always take precedence.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path


class ValidationError(ValueError):
    """The input does not satisfy the documented schema."""


class ExplanationAdapterError(ValueError):
    """The injected explanation callable failed or returned ungrounded data."""


def _object(value, required, optional, path):
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected an object")
    if not required <= value.keys() or value.keys() - required - optional:
        raise ValidationError(
            f"{path}: required fields {sorted(required)}; "
            f"optional fields {sorted(optional)}"
        )


def _text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path}: expected a nonempty string")


def _scalar(value, path):
    if isinstance(value, str):
        _text(value, path)
    elif type(value) not in (int, float, bool):
        raise ValidationError(f"{path}: expected string, number, or boolean")
    elif type(value) is float and not math.isfinite(value):
        raise ValidationError(f"{path}: numbers must be finite")


def _list(value, path, maximum, nonempty=False):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValidationError(f"{path}: expected a list of at most {maximum} items")
    if nonempty and not value:
        raise ValidationError(f"{path}: must not be empty")


def _equal(left, right):
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    return type(left) in (int, float) and type(right) in (int, float) and left == right


def _values(value):
    return value if isinstance(value, list) else [value]


def _validate(data):
    _object(data, {"products", "preferences"},
            {"exclusions", "excluded_product_ids", "limit"}, "input")
    _list(data["products"], "products", 10000, nonempty=True)
    seen = set()
    for index, product in enumerate(data["products"]):
        path = f"products[{index}]"
        _object(product, {"id", "name", "attributes"}, set(), path)
        _text(product["id"], f"{path}.id")
        _text(product["name"], f"{path}.name")
        if product["id"] in seen:
            raise ValidationError(f"{path}: duplicate product ID {product['id']!r}")
        seen.add(product["id"])
        attributes = product["attributes"]
        if not isinstance(attributes, dict) or not 1 <= len(attributes) <= 100:
            raise ValidationError(f"{path}.attributes: expected 1..100 attributes")
        for key, value in attributes.items():
            _text(key, f"{path}.attribute key")
            if isinstance(value, list):
                _list(value, f"{path}.{key}", 100, nonempty=True)
            for item in _values(value):
                _scalar(item, f"{path}.{key}")
    for field in ("preferences", "exclusions"):
        rules = data.get(field, [])
        _list(rules, field, 100, nonempty=field == "preferences")
        prior = []
        for index, rule in enumerate(rules):
            path = f"{field}[{index}]"
            _object(rule, {"attribute", "value"},
                    {"weight"} if field == "preferences" else set(), path)
            _text(rule["attribute"], f"{path}.attribute")
            _scalar(rule["value"], f"{path}.value")
            weight = rule.get("weight", 1)
            if type(weight) is not int or not 1 <= weight <= 1000:
                raise ValidationError(f"{path}.weight: expected integer 1..1000")
            if any(old["attribute"] == rule["attribute"]
                   and _equal(old["value"], rule["value"]) for old in prior):
                raise ValidationError(f"{path}: duplicate attribute/value rule")
            prior.append(rule)
    excluded_ids = data.get("excluded_product_ids", [])
    _list(excluded_ids, "excluded_product_ids", 10000)
    for item in excluded_ids:
        _text(item, "excluded_product_ids item")
    limit = data.get("limit", 10)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValidationError("limit: expected integer 1..100")


def _matching_fact(product, rule):
    attributes = product["attributes"]
    if rule["attribute"] not in attributes:
        return None
    for value in _values(attributes[rule["attribute"]]):
        if _equal(value, rule["value"]):
            return {"attribute": rule["attribute"], "value": value}
    return None


def _render_fact(fact):
    return f"{fact['attribute']} = {json.dumps(fact['value'], ensure_ascii=False)}"


def _adapt_explanations(adapter, products, recommendations):
    """Invoke a synchronous callable with a JSON-compatible grounding envelope.

    The callable may call an LLM, but must return only structured selections of
    supplied facts: {"explanations": [{"product_id": "...", "facts":
    [{"attribute": "...", "value": <supplied scalar>}]}]}. Exactly one entry per
    recommended product is required. Free-form prose is deliberately disallowed:
    explanations are rendered locally so invented factual prose cannot leak out.
    """
    if not callable(adapter):
        raise ExplanationAdapterError("explanation_adapter must be callable")
    allowed = {product["id"]: product for product in products}
    envelope = {
        "instruction": (
            "Select supplied attribute facts explaining each recommendation. "
            "Return exactly one entry per supplied product. Do not add prose, "
            "IDs, attributes, or values absent from the supplied products."
        ),
        "products": products,
        "matches": [
            {"product_id": item["product_id"],
             "matched_preferences": item["matched_preferences"]}
            for item in recommendations
        ],
        "response_contract": {
            "explanations": [
                {"product_id": "supplied product ID",
                 "facts": [{"attribute": "supplied key", "value": "supplied scalar"}]}
            ]
        },
    }
    try:
        response = adapter(copy.deepcopy(envelope))
    except Exception as exc:
        raise ExplanationAdapterError(
            f"explanation callable failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        _object(response, {"explanations"}, set(), "adapter response")
        _list(response["explanations"], "adapter explanations", 100, nonempty=True)
        rendered = {}
        for item in response["explanations"]:
            _object(item, {"product_id", "facts"}, set(), "adapter explanation")
            product_id = item["product_id"]
            _text(product_id, "adapter product_id")
            if product_id not in allowed:
                raise ValidationError(f"unlisted product ID {product_id!r}")
            if product_id in rendered:
                raise ValidationError(f"duplicate product ID {product_id!r}")
            _list(item["facts"], "adapter facts", 100, nonempty=True)
            attributes = allowed[product_id]["attributes"]
            facts = []
            for fact in item["facts"]:
                _object(fact, {"attribute", "value"}, set(), "adapter fact")
                _text(fact["attribute"], "adapter fact.attribute")
                _scalar(fact["value"], "adapter fact.value")
                key = fact["attribute"]
                if key not in attributes or not any(
                    type(fact["value"]) is type(value) and fact["value"] == value
                    for value in _values(attributes[key])
                ):
                    raise ValidationError(
                        f"ungrounded fact for product {product_id!r}: {key!r}"
                    )
                text = _render_fact(fact)
                if text in facts:
                    raise ValidationError(f"duplicate fact for product {product_id!r}")
                facts.append(text)
            rendered[product_id] = "Supplied product facts: " + "; ".join(facts) + "."
        if rendered.keys() != allowed.keys():
            raise ValidationError("missing explanation for one or more recommended IDs")
    except ValidationError as exc:
        raise ExplanationAdapterError(f"invalid explanation response: {exc}") from exc
    for recommendation in recommendations:
        recommendation["adapter_explanation"] = rendered[recommendation["product_id"]]


def recommend(data, explanation_adapter=None):
    """Validate input and return ranked JSON-compatible recommendations.

    Scores sum matched preference weights, not probabilities. Ties use ascending
    exact product ID, independent of catalog order. Zero-score products are never
    returned. Any exclusion match (OR semantics), or excluded ID, removes a
    product before ranking. Missing attributes cannot match either kind of rule.
    """
    _validate(data)
    if explanation_adapter is not None and not callable(explanation_adapter):
        raise ExplanationAdapterError("explanation_adapter must be callable")
    data = copy.deepcopy(data)
    excluded_ids = set(data.get("excluded_product_ids", []))
    ranked = []
    excluded_count = 0
    for product in data["products"]:
        if product["id"] in excluded_ids or any(
            _matching_fact(product, rule) is not None
            for rule in data.get("exclusions", [])
        ):
            excluded_count += 1
            continue
        matches = []
        for rule in data["preferences"]:
            fact = _matching_fact(product, rule)
            if fact is not None:
                matches.append({
                    "attribute": fact["attribute"], "product_value": fact["value"],
                    "requested_value": rule["value"], "weight": rule.get("weight", 1),
                })
        if matches:
            ranked.append({
                "product_id": product["id"], "name": product["name"],
                "score": sum(match["weight"] for match in matches),
                "matched_preferences": matches,
                "explanation": "Matches stated preferences: " + "; ".join(
                    _render_fact({"attribute": match["attribute"],
                                  "value": match["product_value"]})
                    for match in matches
                ) + ".",
            })
    ranked.sort(key=lambda item: (-item["score"], item["product_id"]))
    total_matches = len(ranked)
    ranked = ranked[:data.get("limit", 10)]
    if ranked and explanation_adapter is not None:
        selected_ids = {item["product_id"] for item in ranked}
        _adapt_explanations(
            explanation_adapter,
            [product for product in data["products"] if product["id"] in selected_ids],
            ranked,
        )
    reason = None
    if not ranked:
        reason = ("all_products_excluded" if excluded_count == len(data["products"])
                  else "no_positive_preference_matches")
    return {
        "status": "ok" if ranked else "no_match",
        "reason": reason, "recommendations": ranked,
        "total_matches": total_matches, "excluded_count": excluded_count,
        "explanation_source": (
            "validated_adapter_facts" if ranked and explanation_adapter is not None
            else "deterministic_attribute_matches"
        ),
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValidationError(f"invalid JSON number {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="UTF-8 JSON input file; JSON is written to stdout")
    args = parser.parse_args(argv)
    try:
        with Path(args.input_file).open(encoding="utf-8-sig") as stream:
            data = json.load(stream, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        result = recommend(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
