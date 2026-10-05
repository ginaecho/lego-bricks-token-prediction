"""Synthetic retail reference: validated sentiment -> consent-aware product search.

Run: python -B implementation.py example_input.json
Money is represented in integer cents. No reviews, endorsements or external services.
"""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    pass


LEXICON = {
    "love": 2, "great": 2, "excellent": 3, "good": 1, "happy": 2,
    "bad": -1, "broken": -3, "hate": -2, "terrible": -3,
    "disappointed": -2, "late": -1, "unsafe": -3, "dangerous": -3,
}
SEVERITY = {"low": 0, "medium": 10, "high": 20, "critical": 30}
SYNONYMS = {
    "sneakers": "shoe", "trainers": "shoe", "shoes": "shoe",
    "kicks": "shoe", "jogging": "running", "coats": "jacket",
    "coat": "jacket", "raincoat": "jacket", "rucksack": "backpack",
    "bags": "backpack",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unsupported fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    require(len(value) <= 2000, path + " exceeds 2000 characters")


def integer(value, minimum, maximum, path):
    require(type(value) is int and minimum <= value <= maximum,
            path + " must be an integer in range")


def tokens(value):
    return re.findall(r"[a-z0-9]+", value.lower())


def canonical(value):
    return {SYNONYMS.get(word, word) for word in tokens(value)}


def validate_input(data):
    keys(data, ["schema_version", "synthetic", "catalog", "customer", "basket",
                "issues", "query", "clickstream"], "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "unsupported schema_version")
    require(data["synthetic"] is True, "this reference accepts labeled synthetic fixtures only")
    text(data["query"], "query")
    require(bool(tokens(data["query"])), "query must contain searchable words")
    catalog = data["catalog"]
    require(isinstance(catalog, list) and 0 < len(catalog) <= 1000, "catalog size must be 1..1000")
    by_sku = {}
    for product in catalog:
        keys(product, ["sku", "name", "description", "category", "price_cents", "currency",
                       "stock"], "product")
        for field in ["sku", "name", "description", "category"]:
            text(product[field], "product." + field)
        require(product["sku"] not in by_sku, "duplicate catalog SKU")
        integer(product["price_cents"], 0, 10**9, "price_cents")
        integer(product["stock"], 0, 10**6, "stock")
        require(product["currency"] in ("USD", "EUR", "GBP"), "unsupported currency")
        by_sku[product["sku"]] = product
    customer = data["customer"]
    keys(customer, ["id", "consent", "preferred_categories"], "customer")
    text(customer["id"], "customer.id")
    keys(customer["consent"], ["personalization"], "consent")
    require(type(customer["consent"]["personalization"]) is bool, "consent must be boolean")
    categories = customer["preferred_categories"]
    require(isinstance(categories, list), "preferred_categories must be an array")
    known_categories = {p["category"] for p in catalog}
    for category in categories:
        text(category, "preferred category")
        require(category in known_categories, "unknown preferred category")
    require(isinstance(data["basket"], list), "basket must be an array")
    basket_skus = set()
    for item in data["basket"]:
        keys(item, ["sku", "quantity"], "basket item")
        text(item["sku"], "basket SKU")
        require(item["sku"] in by_sku and item["sku"] not in basket_skus,
                "unknown or duplicate basket SKU")
        integer(item["quantity"], 1, 10**6, "basket quantity")
        basket_skus.add(item["sku"])
    require(isinstance(data["issues"], list) and len(data["issues"]) <= 1000,
            "issues must be an array of at most 1000 items")
    issue_ids = set()
    for issue in data["issues"]:
        keys(issue, ["id", "sku", "text", "severity"], "issue")
        for field in ("id", "sku", "text", "severity"):
            text(issue[field], "issue." + field)
        require(issue["id"] not in issue_ids, "duplicate issue ID")
        require(issue["sku"] in by_sku, "unknown issue SKU")
        require(issue["severity"] in SEVERITY, "invalid severity")
        issue_ids.add(issue["id"])
    require(isinstance(data["clickstream"], list) and len(data["clickstream"]) <= 10000,
            "clickstream must be an array of at most 10000 events")
    previous = -1
    for event in data["clickstream"]:
        keys(event, ["sequence", "customer_id", "action", "sku"], "clickstream event")
        integer(event["sequence"], 0, 10**12, "event sequence")
        require(event["sequence"] > previous, "clickstream must have increasing sequences")
        previous = event["sequence"]
        require(event["customer_id"] == customer["id"], "event customer mismatch")
        require(event["action"] in ("view", "click", "add_to_basket"), "unknown event action")
        text(event["sku"], "event SKU")
        require(event["sku"] in by_sku, "unknown event SKU")


def score_issue(issue, by_sku):
    words = tokens(issue["text"])
    evidence = []
    for index, word in enumerate(words):
        if word in LEXICON:
            negated = any(w in ("not", "never", "no") for w in words[max(0, index - 2):index])
            weight = LEXICON[word] * (-1 if negated else 1)
            evidence.append({"token": word, "position": index, "negated": negated, "weight": weight})
    score = sum(item["weight"] for item in evidence)
    # The bounded sentiment adjustment cannot override an adjacent severity band.
    priority = SEVERITY[issue["severity"]] + min(9, max(0, -score))
    return {
        "issue_id": issue["id"], "sku": issue["sku"],
        "category": by_sku[issue["sku"]]["category"], "severity": issue["severity"],
        "score": score, "sentiment": "negative" if score < 0 else "positive" if score > 0 else "neutral",
        "priority": priority, "evidence": evidence,
    }


def derive_insights(data):
    by_sku = {p["sku"]: p for p in data["catalog"]}
    issues = [score_issue(issue, by_sku) for issue in data["issues"]]
    issues.sort(key=lambda issue: (-issue["priority"], issue["issue_id"]))
    return {
        "issues": issues,
        "overall_score": sum(issue["score"] for issue in issues),
        "urgent": any(issue["severity"] in ("high", "critical") for issue in issues),
        "negative_categories": sorted({issue["category"] for issue in issues if issue["score"] < 0}),
    }


def validate(document, stage):
    """One envelope and validation entry point for every pipeline boundary."""
    fields = ["schema_version", "status", "input"]
    if stage in ("sentiment", "search"):
        fields.append("insights")
    if stage == "search":
        fields.append("search")
    require(stage in ("input", "sentiment", "search"), "unknown pipeline stage")
    keys(document, fields, stage)
    require(document["schema_version"] == 1 and type(document["schema_version"]) is int,
            "invalid envelope version")
    require(document["status"] == "ok", "invalid envelope status")
    validate_input(document["input"])
    if stage in ("sentiment", "search"):
        require(document["insights"] == derive_insights(document["input"]),
                "sentiment handoff failed validation")
    if stage == "search":
        require(document["search"] == derive_search(document["input"], document["insights"]),
                "search output failed catalog or ranking validation")
    return document


def sentiment_stage(document):
    validate(document, "input")
    output = copy.deepcopy(document)
    output["insights"] = derive_insights(output["input"])
    return validate(output, "sentiment")


def derive_search(data, insights):
    terms = canonical(data["query"])
    consent = data["customer"]["consent"]["personalization"]
    # Personal signals, including complaint-derived preferences, are gated together.
    preferred = set(data["customer"]["preferred_categories"]) if consent else set()
    clicked = {e["sku"] for e in data["clickstream"]} if consent else set()
    basket = {item["sku"] for item in data["basket"]} if consent else set()
    negative = set(insights["negative_categories"]) if consent else set()
    urgent = insights["urgent"] if consent else False
    results = []
    for product in data["catalog"]:
        title = canonical(product["name"])
        category = canonical(product["category"])
        description = canonical(product["description"])
        matched = terms & (title | category | description | canonical(product["sku"]))
        if not matched or (urgent and product["stock"] == 0):
            continue
        components = {
            "query": len(matched) * 10 + len(terms & title) * 3 + len(terms & category) * 2,
            "preferred_category": 2 if product["category"] in preferred else 0,
            "clickstream": 1 if product["sku"] in clicked else 0,
            "basket": 1 if product["sku"] in basket else 0,
            "issue_context": 3 if product["category"] in negative else 0,
        }
        results.append({
            "product": copy.deepcopy(product), "score": sum(components.values()),
            "matched_terms": sorted(matched), "score_components": components,
        })
    results.sort(key=lambda item: (-item["score"], item["product"]["sku"]))
    return {
        "query": data["query"], "normalized_terms": sorted(terms),
        "personalization_applied": consent, "urgent_in_stock_only": urgent,
        "source_issue_ids": [issue["issue_id"] for issue in insights["issues"]] if consent else [],
        "results": results,
        "message": "Catalog matches" if results else "No catalog matches; try a broader product term.",
    }


def search_stage(document):
    validate(document, "sentiment")
    output = copy.deepcopy(document)
    output["search"] = derive_search(output["input"], output["insights"])
    return validate(output, "search")


def run_pipeline(data):
    document = {"schema_version": 1, "status": "ok", "input": copy.deepcopy(data)}
    return search_stage(sentiment_stage(document))


def reject_constant(value):
    raise ValidationError("non-finite JSON numbers are unsupported")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
