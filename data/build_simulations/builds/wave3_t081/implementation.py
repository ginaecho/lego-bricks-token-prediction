"""Deterministic personalized discovery; prices share one caller-defined currency.

Run: python -B implementation.py example_input.json
Only supplied catalog/profile data is used; no persistence or external providers.
"""

import json
import math
import sys


class ValidationError(ValueError):
    pass


def object_fields(value, path, allowed, required=()):
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(allowed)
    if missing or extra:
        raise ValidationError(
            f"{path}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}"
        )


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a nonblank string")


def number(value, path, low, high):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValidationError(f"{path} must be a finite number from {low} to {high}")


def strings(value, path):
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be an array")
    for item in value:
        text(item, path)
    if len(value) != len(set(value)):
        raise ValidationError(f"{path} cannot contain duplicates")


def validate(data):
    """One strict schema boundary shared by the Python entry point and CLI."""
    object_fields(data, "input", {"schema_version", "catalog", "customer", "options"},
                  {"schema_version", "catalog", "customer"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    catalog = data["catalog"]
    if not isinstance(catalog, list) or len(catalog) > 10000:
        raise ValidationError("catalog must be an array with at most 10000 products")
    ids = set()
    for product in catalog:
        fields = {"id", "name", "category", "tags", "price", "popularity", "in_stock"}
        object_fields(product, "product", fields, fields)
        for key in ("id", "name", "category"):
            text(product[key], f"product.{key}")
        if product["id"] in ids:
            raise ValidationError("catalog product ids must be unique")
        ids.add(product["id"])
        strings(product["tags"], "product.tags")
        number(product["price"], "product.price", 0, 1_000_000_000)
        number(product["popularity"], "product.popularity", 0, 1)
        if type(product["in_stock"]) is not bool:
            raise ValidationError("product.in_stock must be boolean")
    customer = data["customer"]
    object_fields(customer, "customer",
                  {"id", "preferred_categories", "preferred_tags", "blocked_categories",
                   "budget", "interactions"}, {"id"})
    text(customer["id"], "customer.id")
    for key in ("preferred_categories", "preferred_tags", "blocked_categories"):
        strings(customer.get(key, []), f"customer.{key}")
    if "budget" in customer:
        budget = customer["budget"]
        object_fields(budget, "budget", {"min", "max"}, {"min", "max"})
        for key in ("min", "max"):
            number(budget[key], f"budget.{key}", 0, 1_000_000_000)
        if budget["min"] > budget["max"]:
            raise ValidationError("budget.min cannot exceed budget.max")
    interactions = customer.get("interactions", [])
    if not isinstance(interactions, list) or len(interactions) > 10000:
        raise ValidationError("interactions must be an array of at most 10000 events")
    for event in interactions:
        object_fields(event, "interaction", {"product_id", "type"},
                      {"product_id", "type"})
        text(event["product_id"], "interaction.product_id")
        text(event["type"], "interaction.type")
        if event["product_id"] not in ids:
            raise ValidationError("interaction references an unknown product")
        if event["type"] not in {"view", "like", "purchase", "dislike"}:
            raise ValidationError("interaction.type must be view, like, purchase or dislike")
    options = data.get("options", {})
    object_fields(options, "options", {"limit", "exclude_purchased", "diversity"})
    limit = options.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValidationError("options.limit must be an integer from 1 to 100")
    if type(options.get("exclude_purchased", True)) is not bool:
        raise ValidationError("options.exclude_purchased must be boolean")
    number(options.get("diversity", 0.5), "options.diversity", 0, 1)
    return data


def recommend(data):
    validate(data)
    catalog, customer = data["catalog"], data["customer"]
    options = data.get("options", {})
    by_id = {product["id"]: product for product in catalog}
    categories, tags = {}, {}
    purchased, disliked = set(), set()
    # Deduplicate identical events: repeated views cannot dominate the profile.
    events = {(e["product_id"], e["type"]) for e in customer.get("interactions", [])}
    weights = {"view": 0.25, "like": 1.0, "purchase": 1.5, "dislike": -2.0}
    for product_id, kind in sorted(events):
        product = by_id[product_id]
        weight = weights[kind]
        category = product["category"]
        categories[category] = categories.get(category, 0) + weight
        for tag in product["tags"]:
            tags[tag] = tags.get(tag, 0) + weight
        if kind == "purchase":
            purchased.add(product_id)
        if kind == "dislike":
            disliked.add(product_id)
    preferred_categories = set(customer.get("preferred_categories", []))
    preferred_tags = set(customer.get("preferred_tags", []))
    blocked = set(customer.get("blocked_categories", []))
    budget = customer.get("budget")
    excluded = {"out_of_stock": 0, "blocked_category": 0, "outside_budget": 0,
                "disliked": 0, "purchased": 0}
    candidates = []
    for product in catalog:
        reason = None
        if not product["in_stock"]:
            reason = "out_of_stock"
        elif product["category"] in blocked:
            reason = "blocked_category"
        elif budget and not budget["min"] <= product["price"] <= budget["max"]:
            reason = "outside_budget"
        elif product["id"] in disliked:
            reason = "disliked"
        elif options.get("exclude_purchased", True) and product["id"] in purchased:
            reason = "purchased"
        if reason:
            excluded[reason] += 1
            continue
        product_tags = set(product["tags"])
        overlap = sorted(product_tags & preferred_tags)
        category_score = 3.0 if product["category"] in preferred_categories else 0.0
        tag_score = 2.0 * len(overlap) / max(1, len(preferred_tags))
        category_affinity = max(-2, min(2, categories.get(product["category"], 0)))
        tag_affinity = sum(max(-2, min(2, tags.get(tag, 0))) for tag in sorted(product_tags))
        history_score = category_affinity + tag_affinity / max(1, len(product_tags))
        components = {"category_preference": category_score, "tag_preference": tag_score,
                      "history_affinity": history_score, "popularity": product["popularity"]}
        reasons = []
        if category_score:
            reasons.append("Matches a preferred category")
        if overlap:
            reasons.append("Matches preferred tags: " + ", ".join(overlap))
        if history_score > 0:
            reasons.append("Similar to products with positive interactions")
        elif history_score < 0:
            reasons.append("Reduced rank due to negative interaction affinity")
        if not reasons:
            reasons.append("Catalog popularity fallback; no matching personal signals")
        candidates.append((product, components, reasons))
    eligible_count = len(candidates)
    recommendations, selected_categories = [], {}
    while candidates and len(recommendations) < options.get("limit", 5):
        def ranking(item):
            product, components, _ = item
            penalty = options.get("diversity", 0.5) * selected_categories.get(product["category"], 0)
            return -(sum(components.values()) - penalty), product["id"]

        candidate = min(candidates, key=ranking)
        candidates.remove(candidate)
        product, components, reasons = candidate
        penalty = options.get("diversity", 0.5) * selected_categories.get(product["category"], 0)
        components = dict(components, diversity_penalty=-penalty)
        if penalty:
            reasons = reasons + ["Adjusted for category variety in this recommendation list"]
        recommendations.append({
            "rank": len(recommendations) + 1, "product_id": product["id"],
            "name": product["name"], "category": product["category"], "price": product["price"],
            "score": round(sum(components.values()), 6),
            "score_components": {k: round(v, 6) for k, v in components.items()},
            "reasons": reasons,
        })
        selected_categories[product["category"]] = selected_categories.get(product["category"], 0) + 1
    return {
        "status": "ok", "schema_version": 1, "customer_id": customer["id"],
        "recommendations": recommendations,
        "summary": {
            "catalog_count": len(catalog), "eligible_count": eligible_count,
            "returned_count": len(recommendations), "excluded": excluded,
            "mode": "personalized" if preferred_categories or preferred_tags or events else "cold_start",
            "unique_interaction_count": len(events),
        },
    }


def reject_constant(value):
    raise ValidationError(f"Nonstandard JSON number: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = recommend(data)
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
