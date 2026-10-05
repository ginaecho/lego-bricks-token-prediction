"""Deterministic, dependency-free product search over a caller-supplied catalog.

Run: python -B implementation.py example_input.json
All prices in a request must use one caller-selected currency.
"""

import difflib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path


class ValidationError(ValueError):
    pass


STOP_WORDS = {"a", "an", "the", "for", "and", "of", "with", "please", "find", "me"}
SYNONYMS = {
    "sneaker": "shoe", "sneakers": "shoe", "shoes": "shoe",
    "trainers": "shoe", "trainer": "shoe",
    "notebook": "laptop", "notebooks": "laptop", "laptops": "laptop",
    "cordless": "wireless", "earphones": "headphone", "headphones": "headphone",
    "couches": "sofa", "couch": "sofa", "sofas": "sofa",
    "television": "tv", "televisions": "tv",
    "rucksack": "backpack", "rucksacks": "backpack", "backpacks": "backpack",
}


def tokens(text):
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return [SYNONYMS.get(t, t) for t in re.findall(r"[^\W_]+", normalized)
            if t not in STOP_WORDS]


def object_fields(value, allowed, required, location):
    if not isinstance(value, dict):
        raise ValidationError(f"{location} must be an object")
    if set(value) - allowed:
        raise ValidationError(f"{location} contains unknown fields")
    if required - set(value):
        raise ValidationError(f"{location} is missing required fields")


def text(value, location, maximum=200, blank=False):
    if not isinstance(value, str) or len(value) > maximum or (not blank and not value.strip()):
        raise ValidationError(f"{location} must be {'a nonempty' if not blank else 'a'} string of at most {maximum} characters")


def number(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{location} must be a finite nonnegative number")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError(f"{location} must be a finite nonnegative number")


def validate(payload):
    object_fields(payload, {"query", "products", "filters", "limit", "sort"},
                  {"query", "products"}, "input")
    text(payload["query"], "query", 300, blank=True)
    products = payload["products"]
    if not isinstance(products, list) or len(products) > 1000:
        raise ValidationError("products must be an array with at most 1000 items")
    identifiers = set()
    for index, product in enumerate(products):
        loc = f"products[{index}]"
        object_fields(product, {"id", "name", "description", "category", "tags", "price", "in_stock"},
                      {"id", "name", "category", "price", "in_stock"}, loc)
        for field in ("id", "name", "category"):
            text(product[field], f"{loc}.{field}")
        if product["id"] in identifiers:
            raise ValidationError("product ids must be unique")
        identifiers.add(product["id"])
        text(product.get("description", ""), f"{loc}.description", 2000, blank=True)
        tags = product.get("tags", [])
        if not isinstance(tags, list) or len(tags) > 30:
            raise ValidationError(f"{loc}.tags must be an array of at most 30 strings")
        for tag in tags:
            text(tag, f"{loc}.tags item", 100)
        number(product["price"], f"{loc}.price")
        if type(product["in_stock"]) is not bool:
            raise ValidationError(f"{loc}.in_stock must be boolean")
    filters = payload.get("filters", {})
    object_fields(filters, {"category", "min_price", "max_price", "in_stock"}, set(), "filters")
    if "category" in filters:
        text(filters["category"], "filters.category")
    for field in ("min_price", "max_price"):
        if field in filters:
            number(filters[field], f"filters.{field}")
    if filters.get("min_price", 0) > filters.get("max_price", float("inf")):
        raise ValidationError("min_price must not exceed max_price")
    if "in_stock" in filters and type(filters["in_stock"]) is not bool:
        raise ValidationError("filters.in_stock must be boolean")
    limit = payload.get("limit", 10)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValidationError("limit must be an integer from 1 to 100")
    if payload.get("sort", "relevance") not in ("relevance", "price_asc", "price_desc"):
        raise ValidationError("sort must be relevance, price_asc or price_desc")
    return payload


def permitted(product, filters):
    return (
        ("category" not in filters or product["category"].strip().casefold() == filters["category"].strip().casefold())
        and product["price"] >= filters.get("min_price", 0)
        and product["price"] <= filters.get("max_price", float("inf"))
        and ("in_stock" not in filters or product["in_stock"] == filters["in_stock"])
    )


def similarity(query_token, candidate):
    if query_token == candidate:
        return 1.0, "exact_or_synonym"
    if len(query_token) >= 3 and candidate.startswith(query_token):
        return 0.8, "prefix"
    if min(len(query_token), len(candidate)) >= 4:
        ratio = difflib.SequenceMatcher(None, query_token, candidate, autojunk=False).ratio()
        if ratio >= 0.8:
            return 0.65, "typo"
    return 0.0, ""


def search(payload):
    """Validate once, filter, rank, and return the shared JSON response envelope."""
    payload = validate(payload)
    query = list(dict.fromkeys(tokens(payload["query"])))
    filters = payload.get("filters", {})
    ranked = []
    browse = not payload["query"].strip()
    for product in payload["products"]:
        if not permitted(product, filters):
            continue
        fields = [
            (tokens(product["name"]), 3),
            (tokens(" ".join(product.get("tags", []))), 2),
            (tokens(product["category"]), 2),
            (tokens(product.get("description", "")), 1),
        ]
        score = 0
        matches = []
        for term in query:
            best = (0.0, "", "")
            for words, weight in fields:
                for word in words:
                    strength, kind = similarity(term, word)
                    candidate = (strength * weight, word, kind)
                    if candidate > best:
                        best = candidate
            if best[0]:
                score += best[0]
                matches.append({"query_token": term, "catalog_token": best[1], "kind": best[2]})
        # Require every meaningful query term; avoid irrelevant partial matches.
        if not browse and (not query or len(matches) != len(query)):
            continue
        ranked.append({"product": dict(product), "score": round(score, 4), "matches": matches})
    sort = payload.get("sort", "relevance")
    if sort == "relevance":
        key = lambda row: (-row["score"], row["product"]["id"])
    else:
        direction = 1 if sort == "price_asc" else -1
        key = lambda row: (direction * row["product"]["price"], -row["score"], row["product"]["id"])
    ranked.sort(key=key)
    return {
        "status": "ok",
        "query": payload["query"],
        "normalized_tokens": query,
        "mode": "browse" if browse else "search",
        "total_matches": len(ranked),
        "results": ranked[:payload.get("limit", 10)],
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"invalid JSON numeric constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        raw = Path(argv[0]).read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = search(payload)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        result = {"status": "error", "error": {"code": "invalid_input", "message": str(exc)}}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
