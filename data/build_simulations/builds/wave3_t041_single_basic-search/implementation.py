"""Deterministic synthetic-catalog product search, using Python's standard library.

Input: schema_version=1, fixture_label, query, products, optional limit/filters.
Output: status, schema_version, query, interpreted_terms, total_matches, results.
Unknown fields are rejected. Blank queries browse the filtered catalog by ID.
Prices are nonnegative numbers in one caller-selected currency (no conversion).
"""

import difflib
import json
import math
import re
import sys
import unicodedata


class ValidationError(ValueError):
    pass


STOP_WORDS = {"a", "an", "the", "for", "and", "or", "of", "to", "with",
              "i", "want", "need", "please", "find", "me", "some"}
SYNONYMS = (
    {"sofa", "couch"}, {"sneaker", "trainer"},
    {"laptop", "notebook"}, {"tv", "television"},
    {"headphone", "headset"}, {"bike", "bicycle"},
    {"rucksack", "backpack"}, {"cellphone", "smartphone"},
)


def tokens(text):
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    result = []
    for word in re.findall(r"[^\W_]+", normalized, re.UNICODE):
        if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us")):
            word = word[:-1]
        if word not in STOP_WORDS:
            result.append(word)
    return result


def check(condition, message):
    if not condition:
        raise ValidationError(message)


def object_fields(value, required, optional, path):
    check(isinstance(value, dict), f"{path} must be an object")
    check(not (set(value) - required - optional), f"{path} has unknown fields")
    check(required <= set(value), f"{path} is missing required fields")


def text(value, path, maximum=2000, allow_empty=False):
    check(isinstance(value, str), f"{path} must be a string")
    check(len(value) <= maximum and (allow_empty or bool(value.strip())),
          f"{path} is empty or too long")


def number(value, path):
    check(type(value) in (int, float), f"{path} must be a number")
    check(value >= 0 and value <= 10**12 and math.isfinite(value),
          f"{path} must be finite and between 0 and 1e12")


def validate(data):
    """The shared validation boundary for CLI and Python callers."""
    object_fields(data, {"schema_version", "fixture_label", "query", "products"},
                  {"filters", "limit"}, "input")
    check(type(data["schema_version"]) is int and data["schema_version"] == 1,
          "schema_version must be integer 1")
    text(data["fixture_label"], "fixture_label", 200)
    text(data["query"], "query", 500, allow_empty=True)
    limit = data.get("limit", 10)
    check(type(limit) is int and 1 <= limit <= 100, "limit must be integer 1..100")
    check(isinstance(data["products"], list) and len(data["products"]) <= 1000,
          "products must be an array of at most 1000 items")
    ids = set()
    for i, product in enumerate(data["products"]):
        path = f"products[{i}]"
        object_fields(product, {"id", "name", "description", "category", "price",
                                "in_stock", "tags"}, set(), path)
        for key in ("id", "name", "category"):
            text(product[key], f"{path}.{key}", 200)
        text(product["description"], f"{path}.description", allow_empty=True)
        check(product["id"] not in ids, "product IDs must be unique")
        ids.add(product["id"])
        number(product["price"], f"{path}.price")
        check(type(product["in_stock"]) is bool, f"{path}.in_stock must be boolean")
        tags = product["tags"]
        check(isinstance(tags, list) and len(tags) <= 50, f"{path}.tags must be an array of at most 50 strings")
        for tag in tags:
            text(tag, f"{path}.tags[]", 100)
    filters = data.get("filters", {})
    object_fields(filters, set(), {"category", "min_price", "max_price", "in_stock"},
                  "filters")
    if "category" in filters:
        text(filters["category"], "filters.category", 200)
    for key in ("min_price", "max_price"):
        if key in filters:
            number(filters[key], f"filters.{key}")
    if "in_stock" in filters:
        check(type(filters["in_stock"]) is bool, "filters.in_stock must be boolean")
    check(filters.get("min_price", 0) <= filters.get("max_price", 10**12),
          "min_price must not exceed max_price")
    return data


def search(data):
    validate(data)
    query_terms = sorted(set(tokens(data["query"])))
    filters = data.get("filters", {})
    candidates = []
    vocabulary = set()
    for product in data["products"]:
        if "category" in filters and product["category"].strip().casefold() != filters["category"].strip().casefold():
            continue
        if "in_stock" in filters and product["in_stock"] != filters["in_stock"]:
            continue
        if not filters.get("min_price", 0) <= product["price"] <= filters.get("max_price", 10**12):
            continue
        fields = {
            "name": (set(tokens(product["name"])), 4),
            "tags": (set(tokens(" ".join(product["tags"]))), 3),
            "category": (set(tokens(product["category"])), 2),
            "description": (set(tokens(product["description"])), 1),
        }
        for terms, _ in fields.values():
            vocabulary.update(terms)
        candidates.append((product, fields))

    interpretations = []
    for term in query_terms:
        alternatives = {term: ("exact", 100)}
        for group in SYNONYMS:
            if term in group:
                alternatives.update({word: ("synonym", 85) for word in group if word != term})
        # Never fuzzy-correct a known exact or synonym term.
        if not vocabulary.intersection(alternatives) and len(term) >= 4:
            nearby = difflib.get_close_matches(term, sorted(vocabulary), n=1, cutoff=0.80)
            if nearby:
                alternatives[nearby[0]] = ("typo", 70)
        interpretations.append((term, alternatives))

    results = []
    for product, fields in candidates:
        score = 0
        reasons = []
        for original, alternatives in interpretations:
            matches = []
            for field, (words, weight) in fields.items():
                for word, (kind, confidence) in alternatives.items():
                    if word in words:
                        matches.append((weight * confidence, field, word, kind))
            if not matches:
                break
            points, field, word, kind = max(matches)
            score += points
            reasons.append({"term": original, "matched": word, "kind": kind, "field": field})
        else:
            results.append({"product": dict(product), "score": score, "matches": reasons})
    results.sort(key=lambda result: (-result["score"], result["product"]["id"]))
    return {
        "status": "ok", "schema_version": 1, "fixture_label": data["fixture_label"],
        "query": data["query"],
        "interpreted_terms": [
            {"term": term, "alternatives": sorted(alternatives)}
            for term, alternatives in interpretations
        ],
        "total_matches": len(results), "results": results[:data.get("limit", 10)],
    }


def reject_constant(value):
    raise ValidationError(f"Non-finite JSON constant: {value}")


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
        check(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, parse_constant=reject_constant,
                             object_pairs_hook=unique_object)
        output = search(data)
        code = 0
    except (ValidationError, OSError, ValueError, RecursionError) as error:
        output = {"status": "error", "schema_version": 1, "error": str(error)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
