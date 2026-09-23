"""Synthetic, deterministic FAQ -> behavioral discovery reference pipeline.

Run: python -B implementation.py example_input.json
All timestamps must carry a timezone. Scores use days and a 30-day half-life.
"""

import json
import math
import re
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, name):
    require(isinstance(value, dict), name + " must be an object")
    require(set(value) == set(expected), name + " has missing or unknown fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")


def number(value, name, low, high):
    require(type(value) in (float, int) and math.isfinite(value)
            and low <= value <= high, name + " is outside its allowed range")


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO timestamp") from exc
    require(result.tzinfo is not None and result.utcoffset() is not None,
            "timestamps must include a timezone")
    return result


STOP = {"a", "an", "the", "is", "are", "to", "for", "of", "and", "in",
        "how", "can", "i", "do", "does", "my", "what", "with", "it"}


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOP


def validate(value, kind, context=None):
    """One shared validation boundary for input, FAQ handoff and final output."""
    if kind == "input":
        fields(value, ("schema_version", "fixture_label", "now", "question",
                       "knowledge_base", "products", "events", "limit"), "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "unsupported schema_version")
        text(value["fixture_label"], "fixture_label")
        text(value["question"], "question")
        now = timestamp(value["now"])
        require(type(value["limit"]) is int and 1 <= value["limit"] <= 100,
                "limit must be an integer in 1..100")
        for key in ("knowledge_base", "products", "events"):
            require(isinstance(value[key], list), key + " must be an array")
        product_ids = set()
        for p in value["products"]:
            fields(p, ("id", "name", "category", "popularity"), "product")
            for key in ("id", "name", "category"):
                text(p[key], "product." + key)
            number(p["popularity"], "popularity", 0, 1)
            require(p["id"] not in product_ids, "duplicate product id")
            product_ids.add(p["id"])
        article_ids = set()
        for a in value["knowledge_base"]:
            fields(a, ("id", "title", "answer", "product_ids"), "article")
            for key in ("id", "title", "answer"):
                text(a[key], "article." + key)
            require(a["id"] not in article_ids, "duplicate article id")
            article_ids.add(a["id"])
            require(isinstance(a["product_ids"], list), "product_ids must be an array")
            for product_id in a["product_ids"]:
                text(product_id, "article product id")
                require(product_id in product_ids, "unknown article product")
            require(len(set(a["product_ids"])) == len(a["product_ids"]),
                    "duplicate article product reference")
        for event in value["events"]:
            fields(event, ("product_id", "kind", "at"), "event")
            text(event["product_id"], "event product_id")
            require(event["product_id"] in product_ids, "unknown event product")
            require(event["kind"] in ("browse", "purchase"), "unsupported event kind")
            require(timestamp(event["at"]) <= now, "future event")
    elif kind == "faq":
        fields(value, ("status", "answer", "source_ids", "product_ids",
                       "confidence", "reason"), "faq")
        require(context is not None, "FAQ validation needs input context")
        # Recompute the exact extractive answer and citations, not just their shapes.
        require(value == _faq_result(context), "FAQ output is not grounded in its input")
    elif kind == "output":
        fields(value, ("schema_version", "status", "fixture_label", "faq",
                       "behavior"), "output")
        require(value["schema_version"] == 1 and value["status"] == "ok",
                "invalid output envelope")
        require(value["fixture_label"] == context["fixture_label"], "fixture label changed")
        validate(value["faq"], "faq", context)
        fields(value["behavior"], ("cold_start", "recommendations"), "behavior")
        require(value["behavior"] == _behavior_result(context, value["faq"]),
                "behavior output does not match validated FAQ and history")
    else:
        raise ValidationError("unknown validation schema")
    return value


def _faq_result(data):
    query = tokens(data["question"])
    candidates = []
    for article in data["knowledge_base"]:
        document = tokens(article["title"] + " " + article["answer"])
        coverage = len(query & document) / len(query) if query else 0.0
        candidates.append((coverage, article["id"], article))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates or candidates[0][0] < 0.5:
        return {"status": "abstained", "answer": None, "source_ids": [],
                "product_ids": [], "confidence": 0.0,
                "reason": "No article meets minimum query-token coverage of 0.5."}
    confidence, _, article = candidates[0]
    return {"status": "answered", "answer": article["answer"],
            "source_ids": [article["id"]], "product_ids": sorted(article["product_ids"]),
            "confidence": round(confidence, 8), "reason": "Extracted verbatim from cited article."}


def answer_faq(data):
    validate(data, "input")
    return validate(_faq_result(data), "faq", data)


def _behavior_result(data, faq):
    now = timestamp(data["now"])
    weights = {"browse": 1.0, "purchase": 3.0}
    history = {p["id"]: 0.0 for p in data["products"]}
    contributions = {p["id"]: [] for p in data["products"]}
    for event in data["events"]:
        age = (now - timestamp(event["at"])).total_seconds() / 86400
        contributions[event["product_id"]].append(weights[event["kind"]] * 2 ** (-age / 30))
    for product_id in history:
        history[product_id] = math.fsum(sorted(contributions[product_id]))
    recommendations = []
    for product in data["products"]:
        product_id = product["id"]
        components = {
            "popularity": round(0.1 * product["popularity"], 8),
            "history": round(history[product_id], 8),
            "faq": 1.0 if product_id in faq["product_ids"] else 0.0,
        }
        recommendations.append({
            "product_id": product_id, "name": product["name"],
            "score": round(math.fsum(components.values()), 8),
            "components": components,
            "faq_source_ids": list(faq["source_ids"]) if components["faq"] else [],
        })
    recommendations.sort(key=lambda p: (-p["score"], p["product_id"]))
    return {"cold_start": not bool(data["events"]),
            "recommendations": recommendations[:data["limit"]]}


def personalize(data, faq):
    validate(data, "input")
    validate(faq, "faq", data)
    return _behavior_result(data, faq)


def run_pipeline(data):
    validate(data, "input")
    faq = answer_faq(data)
    output = {"schema_version": 1, "status": "ok",
              "fixture_label": data["fixture_label"], "faq": faq,
              "behavior": personalize(data, faq)}
    return validate(output, "output", data)


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
