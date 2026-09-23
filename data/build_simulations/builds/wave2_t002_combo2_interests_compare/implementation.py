"""Synthetic, deterministic interests -> comparison reference pipeline.

Run: python -B implementation.py example_input.json
Prices are USD; comparison metrics are price_usd, weight_g, battery_hours,
and interest. Missing numeric attributes receive zero utility, not imputed data.
Unknown fields and units are rejected. No network or provider integration.
"""

import copy
import json
import math
import sys
from decimal import Decimal
from pathlib import Path


class ValidationError(ValueError):
    pass


class Validation:
    """The shared contract layer for input, normalized products and handoffs."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @staticmethod
    def fields(value, required, optional=(), path="object"):
        Validation.require(isinstance(value, dict), f"{path}: expected object")
        Validation.require(
            set(required) <= value.keys() <= set(required) | set(optional),
            f"{path}: missing or unknown fields",
        )

    @staticmethod
    def text(value, path, canonical=False):
        Validation.require(isinstance(value, str) and bool(value.strip()),
                           f"{path}: expected nonempty string")
        return value.strip().casefold() if canonical else value.strip()

    @staticmethod
    def number(value, path, positive=False):
        Validation.require(type(value) in (int, float), f"{path}: expected number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        Validation.require(finite and (value > 0 if positive else value >= 0),
                           f"{path}: expected finite {'positive' if positive else 'nonnegative'} number")
        Validation.require(value <= 1_000_000_000, f"{path}: exceeds reference limit")
        return value

    @staticmethod
    def strings(value, path, canonical=False):
        Validation.require(isinstance(value, list), f"{path}: expected array")
        result = [Validation.text(item, path, canonical) for item in value]
        Validation.require(len(result) == len(set(result)), f"{path}: duplicate values")
        return result

    @staticmethod
    def weights(value, path, allowed=None, positive=False):
        Validation.require(isinstance(value, dict), f"{path}: expected object")
        result = {}
        for key, weight in value.items():
            key = Validation.text(key, path, True)
            Validation.require(key not in result, f"{path}: duplicate normalized key")
            Validation.require(allowed is None or key in allowed, f"{path}: unknown metric")
            result[key] = Validation.number(weight, path, positive)
        return result

    @staticmethod
    def money(value, path):
        Validation.number(value, path)
        cents = Decimal(str(value)) * 100
        Validation.require(cents == cents.to_integral_value(),
                           f"{path}: USD supports at most two decimal places")
        return float(cents / 100)

    @staticmethod
    def measurement(value, units, path):
        if value is None:
            return None
        Validation.fields(value, ("value", "unit"), path=path)
        amount = Validation.number(value["value"], path)
        unit = Validation.text(value["unit"], path, True)
        Validation.require(unit in units, f"{path}: unsupported unit")
        return float(amount * units[unit])

    @staticmethod
    def product(product, normalized=False):
        if normalized:
            Validation.fields(product, ("id", "name", "category", "tags", "price_usd",
                                        "weight_g", "battery_hours"), path="normalized product")
            for key in ("id", "name", "category"):
                Validation.text(product[key], key)
            Validation.strings(product["tags"], "tags")
            Validation.money(product["price_usd"], "price_usd")
            for key in ("weight_g", "battery_hours"):
                if product[key] is not None:
                    Validation.number(product[key], key)
            return product
        Validation.fields(product, ("id", "name", "category", "tags", "price"),
                          ("weight", "battery"), "product")
        Validation.fields(product["price"], ("amount", "currency"), path="price")
        Validation.require(product["price"]["currency"] == "USD", "price: only USD is supported")
        result = {
            "id": Validation.text(product["id"], "id"),
            "name": Validation.text(product["name"], "name"),
            "category": Validation.text(product["category"], "category", True),
            "tags": Validation.strings(product["tags"], "tags", True),
            "price_usd": Validation.money(product["price"]["amount"], "price.amount"),
            "weight_g": Validation.measurement(product.get("weight"), {"g": 1, "kg": 1000}, "weight"),
            "battery_hours": Validation.measurement(product.get("battery"), {"h": 1, "min": 1 / 60}, "battery"),
        }
        return Validation.product(result, normalized=True)

    @staticmethod
    def input(data):
        Validation.fields(data, ("schema_version", "fixture", "products", "preferences", "limit"))
        Validation.require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                           "schema_version: expected 1")
        Validation.fields(data["fixture"], ("synthetic", "label"), path="fixture")
        Validation.require(data["fixture"]["synthetic"] is True, "fixture must be synthetic")
        label = Validation.text(data["fixture"]["label"], "fixture.label")
        Validation.require(isinstance(data["products"], list) and len(data["products"]) <= 200,
                           "products: expected array of at most 200 products")
        products = [Validation.product(p) for p in data["products"]]
        Validation.require(len({p["id"] for p in products}) == len(products), "duplicate product id")
        Validation.require(type(data["limit"]) is int and 1 <= data["limit"] <= 50,
                           "limit: expected integer from 1 to 50")
        preferences = data["preferences"]
        Validation.fields(preferences, ("interests", "exclude_ids", "exclude_tags", "comparison_weights"),
                          ("max_price_usd",), "preferences")
        interests = Validation.weights(preferences["interests"], "interests", positive=True)
        weights = Validation.weights(preferences["comparison_weights"], "comparison_weights",
                                     {"interest", "price_usd", "weight_g", "battery_hours"})
        Validation.require(sum(weights.values()) > 0, "comparison_weights: need positive total")
        result = {
            "schema_version": 1,
            "fixture": {"synthetic": True, "label": label},
            "products": products,
            "limit": data["limit"],
            "preferences": {
                "interests": interests,
                "exclude_ids": Validation.strings(preferences["exclude_ids"], "exclude_ids"),
                "exclude_tags": Validation.strings(preferences["exclude_tags"], "exclude_tags", True),
                "comparison_weights": weights,
                "max_price_usd": None,
            },
        }
        if "max_price_usd" in preferences:
            result["preferences"]["max_price_usd"] = Validation.money(
                preferences["max_price_usd"], "max_price_usd")
        return result

    @staticmethod
    def handoff(stage, context):
        Validation.fields(stage, ("stage", "recommendations", "excluded", "eligible_count"),
                          path="interest handoff")
        Validation.require(stage["stage"] == "interests", "wrong handoff stage")
        Validation.require(isinstance(stage["recommendations"], list), "recommendations must be array")
        for entry in stage["recommendations"]:
            Validation.fields(entry, ("product", "interest_score", "matched_interests", "explanation"))
            Validation.product(entry["product"], normalized=True)
        # Re-derive the contract to reject injected, mutated or exclusion-bypassing handoffs.
        Validation.require(stage == _interest_result(context), "interest handoff failed grounding validation")
        return stage


def _exclusion_reasons(product, preferences):
    reasons = []
    if product["id"] in preferences["exclude_ids"]:
        reasons.append("Explicitly excluded product id: " + product["id"])
    blocked = sorted(set(product["tags"]) & set(preferences["exclude_tags"]))
    if blocked:
        reasons.append("Excluded tags: " + ", ".join(blocked))
    budget = preferences["max_price_usd"]
    if budget is not None and product["price_usd"] > budget:
        reasons.append(f"Price USD {product['price_usd']:.2f} exceeds budget USD {budget:.2f}")
    return reasons


def _interest_result(context):
    preferences = context["preferences"]
    candidates, excluded = [], []
    for product in context["products"]:
        reasons = _exclusion_reasons(product, preferences)
        if reasons:
            excluded.append({"product_id": product["id"], "reasons": reasons})
            continue
        evidence = set(product["tags"]) | {product["category"]}
        matches = sorted(evidence & preferences["interests"].keys())
        score = sum(preferences["interests"][key] for key in matches)
        explanation = [
            f"Interest '{key}' matches "
            + ("category" if key == product["category"] else "tag")
            + f"; preference weight {preferences['interests'][key]:g}"
            for key in matches
        ] or ["No configured interest matches this product's category or tags."]
        candidates.append({
            "product": copy.deepcopy(product),
            "interest_score": score,
            "matched_interests": matches,
            "explanation": explanation,
        })
    candidates.sort(key=lambda item: (-item["interest_score"], item["product"]["id"]))
    return {
        "stage": "interests",
        "recommendations": candidates[:context["limit"]],
        "excluded": sorted(excluded, key=lambda item: item["product_id"]),
        "eligible_count": len(candidates),
    }


def recommend(context):
    return Validation.handoff(_interest_result(context), context)


def compare(context, handoff):
    handoff = Validation.handoff(handoff, context)
    candidates = handoff["recommendations"]
    weights = context["preferences"]["comparison_weights"]
    metrics = ("interest", "price_usd", "weight_g", "battery_hours")
    values = {
        metric: {
            item["product"]["id"]: item["interest_score"] if metric == "interest"
            else item["product"][metric]
            for item in candidates
        }
        for metric in metrics
    }
    ranking = []
    total_weight = sum(weights.values())
    for item in candidates:
        product_id = item["product"]["id"]
        utilities, explanations = {}, []
        for metric in metrics:
            value = values[metric][product_id]
            available = [v for v in values[metric].values() if v is not None]
            if value is None:
                utility = 0.0
                explanation = f"{metric}: missing; utility 0 (no imputation)"
            else:
                low, high = min(available), max(available)
                if low == high:
                    utility = 1.0
                elif metric in ("price_usd", "weight_g"):
                    utility = (high - value) / (high - low)
                else:
                    utility = (value - low) / (high - low)
                explanation = (f"{metric}: value {value:g}, candidate range [{low:g}, {high:g}]; "
                               f"utility {utility:.6f}")
            utilities[metric] = utility
            if weights.get(metric, 0) > 0:
                explanations.append(explanation + f"; weight {weights[metric]:g}")
        score = round(sum(utilities[m] * weights.get(m, 0) for m in metrics) / total_weight, 6)
        ranking.append({
            "product_id": product_id,
            "score": score,
            "interest_score": item["interest_score"],
            "matched_interests": list(item["matched_interests"]),
            "utilities": utilities,
            "explanation": explanations,
        })
    ranking.sort(key=lambda item: (-item["score"], -item["interest_score"], item["product_id"]))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    comparison = {
        "stage": "compare",
        "candidate_ids": [item["product"]["id"] for item in candidates],
        "side_by_side": [
            {"attribute": metric,
             "unit": {"interest": "preference_points", "price_usd": "USD",
                      "weight_g": "g", "battery_hours": "h"}[metric],
             "direction": "lower" if metric in ("price_usd", "weight_g") else "higher",
             "values": values[metric]}
            for metric in metrics
        ],
        "ranking": ranking,
    }
    return comparison


def pipeline(data):
    context = Validation.input(data)
    interests = recommend(context)
    comparison = compare(context, interests)
    return {
        "schema_version": 1,
        "status": "ok",
        "fixture": context["fixture"],
        "interests": interests,
        "comparison": comparison,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        Validation.require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        result = pipeline(data)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except (ValidationError, OSError, ValueError, RecursionError) as error:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(error)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
