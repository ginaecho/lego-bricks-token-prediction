"""Deterministic, offline support -> product-search reference pipeline.

All knowledge and catalog records are supplied by the caller. Answers cite
provided knowledge rather than inventing policies. No provider is required.
"""

import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def keys(value, expected, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(expected), label + " has missing or unknown fields")


def validate(value, kind, source=None):
    """One validation boundary for input and both internal stage contracts."""
    if kind == "input":
        keys(value, ("schema_version", "message", "filters", "knowledge", "products"), kind)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        require(text(value["message"]) and len(value["message"]) <= 4000,
                "message must be nonempty and at most 4000 characters")
        keys(value["filters"], ("category", "max_price", "limit"), "filters")
        f = value["filters"]
        require(f["category"] is None or text(f["category"]), "invalid category")
        require(f["max_price"] is None or number(f["max_price"]), "invalid max_price")
        require(type(f["limit"]) is int and 1 <= f["limit"] <= 20, "invalid limit")
        for collection, fields in (
            ("knowledge", ("id", "question", "answer", "keywords")),
            ("products", ("id", "name", "description", "category", "price", "in_stock", "tags")),
        ):
            require(isinstance(value[collection], list), collection + " must be a list")
            ids = set()
            for item in value[collection]:
                keys(item, fields, collection + " item")
                for field in ("id", "question", "answer") if collection == "knowledge" else (
                    "id", "name", "description", "category"
                ):
                    require(text(item[field]), "invalid " + field)
                require(item["id"] not in ids, "duplicate " + collection + " id")
                ids.add(item["id"])
                words = item["keywords"] if collection == "knowledge" else item["tags"]
                require(isinstance(words, list) and all(text(w) for w in words),
                        "keywords/tags must be lists of nonempty strings")
                if collection == "products":
                    require(number(item["price"]), "invalid price")
                    require(type(item["in_stock"]) is bool, "invalid in_stock")
    elif kind == "support":
        require(source is not None, "support validation requires source")
        keys(value, ("answer", "citations", "needs_human", "search_request"), kind)
        require(text(value["answer"]) and type(value["needs_human"]) is bool,
                "invalid support answer")
        require(isinstance(value["citations"], list)
                and all(text(c) for c in value["citations"]), "invalid citations")
        records = {k["id"]: k for k in source["knowledge"]}
        require(len(value["citations"]) == len(set(value["citations"]))
                and all(c in records for c in value["citations"]), "unknown/duplicate citation")
        if value["citations"]:
            require(value["answer"] == "\n\n".join(records[c]["answer"] for c in value["citations"])
                    and not value["needs_human"], "answer must exactly match cited knowledge")
        else:
            require(value["answer"] == FALLBACK and value["needs_human"],
                    "unsupported answers must use safe fallback")
        keys(value["search_request"], ("query", "filters"), "search_request")
        require(value["search_request"]["query"] == source["message"],
                "search query must preserve customer message")
        require(value["search_request"]["filters"] == source["filters"],
                "search filters must preserve customer constraints")
    elif kind == "search":
        keys(value, ("query", "filters", "results", "message"), kind)
        require(source is not None and value["query"] == source["query"]
                and value["filters"] == source["filters"], "search handoff mismatch")
        require(isinstance(value["results"], list)
                and len(value["results"]) <= source["filters"]["limit"], "invalid results")
        ids = set()
        for row in value["results"]:
            keys(row, ("id", "name", "price", "category", "score", "matched_terms"), "result")
            require(text(row["id"]) and row["id"] not in ids and text(row["name"])
                    and text(row["category"]) and number(row["price"]), "invalid result")
            ids.add(row["id"])
            require(type(row["score"]) is int and row["score"] > 0, "invalid score")
            require(isinstance(row["matched_terms"], list) and row["matched_terms"]
                    and all(text(t) for t in row["matched_terms"]), "invalid matched_terms")
        require(text(value["message"]), "invalid search message")
    else:
        raise ValidationError("unknown validation kind")
    return value


FALLBACK = ("I do not have a verified answer in the supplied knowledge. "
            "Please contact the support team for confirmation; I can still search the catalog.")
STOP = set("a an the i my me you your can could do does is are it and or to for of "
           "with about please want need looking find show what how any have".split())
ALIASES = {
    "sneakers": "shoe", "sneaker": "shoe", "shoes": "shoe", "trainers": "shoe",
    "jogging": "running", "jog": "running", "runs": "running",
    "waterproof": "rainproof", "rain": "rainproof",
    "returns": "return", "returning": "return", "refund": "return",
    "shipping": "delivery", "ship": "delivery", "ships": "delivery",
    "headphones": "headphone", "earphones": "headphone",
}


def terms(value):
    return {ALIASES.get(t, t) for t in re.findall(r"[^\W_]+", value.casefold())
            if t not in STOP}


def support_stage(data, responder=None):
    validate(data, "input")
    query = terms(data["message"])
    ranked = []
    for item in data["knowledge"]:
        keyword_terms = terms(" ".join(item["keywords"]))
        score = len(query & keyword_terms)
        # Require an explicit topic keyword, not incidental question-word overlap.
        if score:
            ranked.append((-score, item["id"], item))
    selected = [entry[2] for entry in sorted(ranked)[:2]]
    output = {
        "answer": "\n\n".join(item["answer"] for item in selected) if selected else FALLBACK,
        "citations": [item["id"] for item in selected],
        "needs_human": not bool(selected),
        "search_request": {
            "query": data["message"],
            "filters": dict(data["filters"]),
        },
    }
    if responder is not None:
        # Injection is an offline seam; isolate caller data from callable mutation.
        output = responder(json.loads(json.dumps(output)))
    return validate(output, "support", data)


def search_stage(support, data):
    validate(data, "input")
    validate(support, "support", data)
    request = support["search_request"]
    query = terms(request["query"])
    filters = request["filters"]
    results = []
    for product in data["products"]:
        if not product["in_stock"]:
            continue
        if (filters["category"] is not None
                and product["category"].casefold() != filters["category"].strip().casefold()):
            continue
        if filters["max_price"] is not None and product["price"] > filters["max_price"]:
            continue
        primary = terms(product["name"] + " " + " ".join(product["tags"]))
        secondary = terms(product["description"] + " " + product["category"])
        matched = query & (primary | secondary)
        if not matched:
            continue
        score = 3 * len(query & primary) + len(query & (secondary - primary))
        results.append({
            "id": product["id"], "name": product["name"], "price": product["price"],
            "category": product["category"], "score": score, "matched_terms": sorted(matched),
        })
    results.sort(key=lambda r: (-r["score"], r["price"], r["id"]))
    results = results[:filters["limit"]]
    return validate({
        "query": request["query"], "filters": dict(filters), "results": results,
        "message": "Matching in-stock products." if results else
                   "No matching in-stock products within these filters. Try broader terms or filters.",
    }, "search", request)


def run(data, responder=None):
    validate(data, "input")
    support = support_stage(data, responder)
    search = search_stage(support, data)
    return {"schema_version": 1, "status": "ok", "support": support, "search": search}


def reject_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run(data)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
