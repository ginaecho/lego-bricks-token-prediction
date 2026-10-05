"""Synthetic retail search reference. Standard library only; no live providers."""

import json
import math
import re
import sys
from collections import Counter
from decimal import Decimal


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), label="object"):
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(required) <= value.keys(), f"{label} has missing fields")
    require(value.keys() <= set(required) | set(optional),
            f"{label} has unsupported fields (reviews/endorsements are not supported)")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonblank text")


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum, f"invalid {label}")


def strings(value, label):
    require(isinstance(value, list), f"{label} must be a list")
    for entry in value:
        text(entry, label)


def validate(data):
    """One shared boundary for feed, profile, basket, search and event-log input."""
    fields(data, ("synthetic", "catalog", "customer", "basket", "clickstream", "search"),
           label="input")
    require(data["synthetic"] is True, "synthetic must be true for this reference")
    catalog = data["catalog"]
    require(isinstance(catalog, list) and len(catalog) <= 1000, "catalog must have 0..1000 products")
    by_sku = {}
    for product in catalog:
        fields(product, ("sku", "name", "description", "category", "tags", "price",
                         "currency", "stock"), label="product")
        for key in ("sku", "name", "description", "category"):
            text(product[key], key)
        strings(product["tags"], "tags")
        require(isinstance(product["price"], str) and
                re.fullmatch(r"(0|[1-9][0-9]{0,9})\.[0-9]{2}", product["price"]) is not None,
                "price must be a nonnegative decimal string with two fractional digits")
        require(isinstance(product["currency"], str) and
                re.fullmatch(r"[A-Z]{3}", product["currency"]) is not None,
                "currency must be three uppercase letters")
        integer(product["stock"], 0, 1000000, "stock")
        require(product["sku"] not in by_sku, "duplicate catalog SKU")
        by_sku[product["sku"]] = product
    customer = data["customer"]
    fields(customer, ("id", "consent", "preferences"), label="customer")
    text(customer["id"], "customer id")
    fields(customer["consent"], ("gdpr", "ccpa"), label="consent")
    require(all(type(v) is bool for v in customer["consent"].values()), "consent must be boolean")
    strings(customer["preferences"], "preferences")
    search = data["search"]
    fields(search, ("query",), ("limit", "personalize", "include_out_of_stock"), "search")
    text(search["query"], "query")
    require(len(search["query"]) <= 500, "query too long")
    integer(search.get("limit", 10), 1, 100, "limit")
    for key in ("personalize", "include_out_of_stock"):
        require(type(search.get(key, False)) is bool, f"{key} must be boolean")
    if search.get("personalize", False):
        require(all(customer["consent"].values()),
                "GDPR and CCPA consent required before personalization")
    require(isinstance(data["basket"], list), "basket must be a list")
    seen = set()
    for item in data["basket"]:
        fields(item, ("sku", "quantity"), label="basket item")
        text(item["sku"], "basket SKU")
        require(item["sku"] in by_sku, "unknown basket SKU")
        require(item["sku"] not in seen, "duplicate basket SKU")
        seen.add(item["sku"])
        integer(item["quantity"], 1, by_sku[item["sku"]]["stock"], "basket quantity")
    require(isinstance(data["clickstream"], list) and len(data["clickstream"]) <= 10000,
            "clickstream must have 0..10000 events")
    for event in data["clickstream"]:
        fields(event, ("customer_id", "sku", "event"), label="clickstream event")
        text(event["customer_id"], "event customer id")
        text(event["sku"], "event SKU")
        require(event["customer_id"] == customer["id"], "event belongs to another shopper")
        require(event["sku"] in by_sku, "unknown event SKU")
        require(event["event"] in ("view", "click", "add_to_basket"), "unsupported event type")
    return by_sku


SYNONYMS = {
    "sneakers": "shoe", "sneaker": "shoe", "trainers": "shoe", "shoes": "shoe",
    "jogging": "run", "running": "run", "runs": "run",
    "eco": "sustainable", "green": "sustainable", "recycled": "sustainable",
    "rucksack": "backpack", "rucksacks": "backpack", "backpacks": "backpack",
}


def tokens(value):
    return [SYNONYMS.get(word, word) for word in re.findall(r"[a-z0-9]+", value.lower())]


def product_text(product):
    return " ".join([product["sku"], product["name"], product["description"],
                     product["category"], *product["tags"]])


def embedding_scores(embedder, texts):
    """Injected callable: list[str] -> list[list[finite number]], one vector per text."""
    try:
        vectors = embedder(texts)
        require(isinstance(vectors, (list, tuple)) and len(vectors) == len(texts),
                "embedding count mismatch")
        dimension = None
        normalized = []
        for vector in vectors:
            require(isinstance(vector, (list, tuple)) and 0 < len(vector) <= 4096,
                    "invalid embedding vector")
            require(all(type(x) in (int, float) and math.isfinite(x) for x in vector),
                    "embeddings must contain finite numbers")
            dimension = len(vector) if dimension is None else dimension
            require(len(vector) == dimension, "embedding dimension mismatch")
            norm = math.hypot(*vector)
            require(math.isfinite(norm) and norm > 0, "embedding norm must be finite and positive")
            normalized.append([x / norm for x in vector])
        query = normalized[0]
        return [max(0.0, min(1.0, sum(a * b for a, b in zip(query, vector))))
                for vector in normalized[1:]]
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("embedding callable failed or returned invalid vectors") from exc


def search_products(data, embedder=None):
    by_sku = validate(data)
    search = data["search"]
    products = sorted(by_sku.values(), key=lambda product: product["sku"])
    query_terms = set(tokens(search["query"]))
    require(bool(query_terms), "query must contain searchable ASCII letters or digits")
    # Inverted index with field weights; synonym canonicalization is deliberately bounded.
    index = {}
    for product in products:
        weights = Counter()
        for key, weight in (("sku", 4), ("name", 3), ("category", 2), ("description", 1)):
            for term in set(tokens(product[key])):
                weights[term] += weight
        for term in set(tokens(" ".join(product["tags"]))):
            weights[term] += 2
        for term, weight in weights.items():
            index.setdefault(term, {})[product["sku"]] = weight
    base_scores = Counter()
    for term in sorted(query_terms):
        postings = index.get(term, {})
        idf = 1 + math.log((1 + len(products)) / (1 + len(postings)))
        for sku, weight in postings.items():
            base_scores[sku] += weight * idf / len(query_terms)
    similarities = ([0.0] * len(products) if embedder is None else
                    embedding_scores(embedder, [search["query"]] + [product_text(p) for p in products]))
    personalize = search.get("personalize", False)
    preference_terms = set()
    interest = Counter()
    if personalize:
        preference_terms = set(tokens(" ".join(data["customer"]["preferences"])))
        for event in data["clickstream"]:
            interest[event["sku"]] += {"view": 0.1, "click": 0.2, "add_to_basket": 0.3}[event["event"]]
        for item in data["basket"]:
            preference_terms.update(tokens(by_sku[item["sku"]]["category"]))
    results = []
    for product, similarity in zip(products, similarities):
        lexical = base_scores[product["sku"]]
        if lexical <= 0 and similarity <= 0:
            continue
        if product["stock"] == 0 and not search.get("include_out_of_stock", False):
            continue
        bonus = 0.0
        if personalize:
            bonus = 0.25 * len(preference_terms & set(tokens(product_text(product))))
            bonus += min(1.0, interest[product["sku"]])
        results.append({
            "sku": product["sku"], "name": product["name"],
            "price": product["price"], "currency": product["currency"], "stock": product["stock"],
            "score": round(lexical + 2 * similarity + bonus, 6),
            "matched_terms": sorted(query_terms & set(tokens(product_text(product)))),
        })
    results.sort(key=lambda result: (-result["score"], result["sku"]))
    basket = []
    for item in data["basket"]:
        product = by_sku[item["sku"]]
        basket.append({
            "sku": item["sku"], "quantity": item["quantity"], "unit_price": product["price"],
            "currency": product["currency"], "stock": product["stock"],
            "line_total": str(Decimal(product["price"]) * item["quantity"]),
        })
    return {
        "status": "ok", "synthetic": True, "query": search["query"],
        "personalized": personalize, "ranking": "lexical+injected-embedding" if embedder else "lexical",
        "total_matches": len(results), "results": results[:search.get("limit", 10)],
        "basket": basket,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonfinite JSON constant: {value}")


def main(argv=None):
    try:
        args = sys.argv[1:] if argv is None else argv
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = search_products(data)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
