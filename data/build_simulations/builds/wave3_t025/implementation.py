"""Deterministic personalized discovery, Python standard library only.

All prices must use the same caller-defined currency. No data is persisted.
See build_manifest.json for the input/output contract and scoring policy.
"""

import json
import math
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def fail(message):
    raise ValidationError(message)


def object_fields(value, allowed, required, path):
    if not isinstance(value, dict):
        fail(f"{path} must be an object")
    if set(value) - set(allowed):
        fail(f"{path} contains unknown fields")
    if set(required) - set(value):
        fail(f"{path} is missing required fields")


def text(value, path):
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        fail(f"{path} must be a nonblank string of at most 500 characters")
    return value.strip()


def strings(value, path, normalize=False):
    if not isinstance(value, list) or len(value) > 1000:
        fail(f"{path} must be an array of at most 1000 strings")
    result = [text(item, path) for item in value]
    return set(item.casefold() for item in result) if normalize else set(result)


def number(value, path, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{path} must be a finite nonnegative number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (maximum is not None and value > maximum):
        fail(f"{path} is outside its permitted numeric range")
    return value


def validate(payload):
    """The single validation boundary used by both the Python API and CLI."""
    object_fields(payload, {"products", "customer", "limit"}, {"products"}, "input")
    raw_products = payload["products"]
    if not isinstance(raw_products, list) or len(raw_products) > 10000:
        fail("products must be an array with at most 10000 entries")
    limit = payload.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 50:
        fail("limit must be an integer from 1 through 50")
    products, ids = [], set()
    for index, raw in enumerate(raw_products):
        path = f"products[{index}]"
        object_fields(
            raw,
            {"id", "title", "category", "tags", "price", "popularity", "in_stock"},
            {"id", "title", "category", "price"},
            path,
        )
        product = {
            "id": text(raw["id"], path + ".id"),
            "title": text(raw["title"], path + ".title"),
            "category": text(raw["category"], path + ".category"),
            "tags": strings(raw.get("tags", []), path + ".tags", normalize=True),
            "price": number(raw["price"], path + ".price"),
            "popularity": number(raw.get("popularity", 0), path + ".popularity", 1),
            "in_stock": raw.get("in_stock", True),
        }
        if type(product["in_stock"]) is not bool:
            fail(path + ".in_stock must be a boolean")
        if product["id"] in ids:
            fail("product ids must be unique after trimming")
        ids.add(product["id"])
        products.append(product)
    raw_customer = payload.get("customer", {})
    object_fields(
        raw_customer,
        {"preferred_categories", "preferred_tags", "viewed_product_ids",
         "purchased_product_ids", "excluded_categories", "max_price"},
        set(),
        "customer",
    )
    customer = {}
    for key in ("preferred_categories", "preferred_tags", "excluded_categories",
                "viewed_product_ids", "purchased_product_ids"):
        customer[key] = strings(
            raw_customer.get(key, []), "customer." + key,
            normalize=not key.endswith("_ids"),
        )
    customer["max_price"] = (
        number(raw_customer["max_price"], "customer.max_price")
        if "max_price" in raw_customer else None
    )
    for key in ("viewed_product_ids", "purchased_product_ids"):
        if customer[key] - ids:
            fail(f"customer.{key} must reference catalog product ids")
    return products, customer, limit


def recommend(payload):
    products, customer, limit = validate(payload)
    history = {}
    for kind in ("viewed", "purchased"):
        entries = [p for p in products if p["id"] in customer[kind + "_product_ids"]]
        history[kind] = (
            {p["category"].casefold() for p in entries},
            set().union(*(p["tags"] for p in entries)),
        )
    personalized = any(customer[key] for key in (
        "preferred_categories", "preferred_tags", "viewed_product_ids",
        "purchased_product_ids",
    ))
    ranked = []
    filtered = {"out_of_stock": 0, "already_purchased": 0,
                "excluded_category": 0, "over_budget": 0}
    for product in products:
        category = product["category"].casefold()
        rejection = None
        if not product["in_stock"]:
            rejection = "out_of_stock"
        elif product["id"] in customer["purchased_product_ids"]:
            rejection = "already_purchased"
        elif category in customer["excluded_categories"]:
            rejection = "excluded_category"
        elif customer["max_price"] is not None and product["price"] > customer["max_price"]:
            rejection = "over_budget"
        if rejection:
            filtered[rejection] += 1
            continue
        score = product["popularity"]
        reasons = []
        if category in customer["preferred_categories"]:
            score += 5
            reasons.append("preferred_category")
        matched_tags = sorted(product["tags"] & customer["preferred_tags"])
        if matched_tags:
            score += 2 * len(matched_tags)
            reasons.append("preferred_tags: " + ", ".join(matched_tags))
        for kind, category_weight, tag_weight in (
            ("viewed", 1, 0.5), ("purchased", 2, 1)
        ):
            categories, tags = history[kind]
            if category in categories:
                score += category_weight
                reasons.append(kind + "_category_affinity")
            overlap = product["tags"] & tags
            if overlap:
                score += tag_weight * len(overlap)
                reasons.append(kind + "_tag_affinity")
        if not reasons:
            reasons.append("popularity_fallback")
        ranked.append((score, product, reasons))
    ranked.sort(key=lambda item: (
        -item[0], -item[1]["popularity"], item[1]["price"], item[1]["id"],
    ))
    return {
        "status": "ok",
        "mode": "personalized" if personalized else "cold_start",
        "recommendations": [
            {"product_id": p["id"], "title": p["title"], "category": p["category"],
             "price": p["price"], "score": round(score, 6), "reasons": reasons}
            for score, p, reasons in ranked[:limit]
        ],
        "summary": {
            "catalog_count": len(products), "eligible_count": len(ranked),
            "returned_count": min(limit, len(ranked)), "filtered": filtered,
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            fail("usage: python -B implementation.py INPUT.json")
        payload = json.loads(
            Path(args[0]).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: fail("nonfinite JSON number: " + value),
        )
        result = recommend(payload)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
