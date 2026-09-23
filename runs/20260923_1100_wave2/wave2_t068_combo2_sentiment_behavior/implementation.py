"""Deterministic synthetic customer-insight -> discovery reference pipeline.

Run: python -B implementation.py example_input.json
Only Python's standard library is used. Dates must have explicit UTC offsets.
Scores are explanations, not trained predictions. No provider calls are made.
"""

import copy
import json
import math
import re
import sys
from datetime import datetime


VERSION = "1.0"
POSITIVE = frozenset({"good", "great", "love", "excellent", "helpful", "happy"})
NEGATIVE = frozenset({"bad", "broken", "hate", "poor", "terrible", "slow"})
NEGATORS = frozenset({"not", "never", "no"})
SEVERITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}
EVENT_WEIGHT = {"view": 1.0, "purchase": 3.0}
HALF_LIFE_DAYS = 30.0


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def number(value, low, high, path):
    require(type(value) in (int, float), path + " must be numeric")
    require(math.isfinite(value) and low <= value <= high, path + " out of range")


def timestamp(value, path):
    text(value, path)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(path + " must be ISO 8601") from exc
    require(result.tzinfo is not None and result.utcoffset() is not None,
            path + " must have a timezone")
    return result


def unique_rows(rows, keys, path):
    require(isinstance(rows, list), path + " must be an array")
    ids = set()
    for row in rows:
        fields(row, keys, path + " item")
        text(row["id"], path + ".id")
        require(row["id"] not in ids, path + " contains a duplicate id")
        ids.add(row["id"])
    return ids


def validate_input(data):
    fields(data, ("schema_version", "synthetic", "as_of", "top_k", "customers",
                  "products", "feedback", "events"), "input")
    require(data["schema_version"] == VERSION, "unsupported schema_version")
    require(data["synthetic"] is True, "reference fixtures must be labeled synthetic")
    now = timestamp(data["as_of"], "as_of")
    require(type(data["top_k"]) is int and 1 <= data["top_k"] <= 100,
            "top_k must be an integer from 1 to 100")
    customers = unique_rows(data["customers"], ("id",), "customers")
    products = unique_rows(data["products"], ("id", "category", "popularity"), "products")
    for product in data["products"]:
        text(product["category"], "product.category")
        number(product["popularity"], 0, 1, "product.popularity")
    unique_rows(data["feedback"],
                ("id", "customer_id", "product_id", "text", "severity"), "feedback")
    unique_rows(data["events"],
                ("id", "customer_id", "product_id", "kind", "timestamp"), "events")
    for name in ("feedback", "events"):
        for row in data[name]:
            text(row["customer_id"], name + ".customer_id")
            text(row["product_id"], name + ".product_id")
            require(row["customer_id"] in customers, name + " has unknown customer")
            require(row["product_id"] in products, name + " has unknown product")
            if name == "feedback":
                require(isinstance(row["text"], str), "feedback.text must be text")
                require(isinstance(row["severity"], str) and row["severity"] in SEVERITY,
                        "invalid severity")
            else:
                require(isinstance(row["kind"], str) and row["kind"] in EVENT_WEIGHT,
                        "invalid event kind")
                require(timestamp(row["timestamp"], "event.timestamp") <= now,
                        "event.timestamp cannot be in the future")


def validate(envelope, phase):
    """Single validation boundary used by both stages and final serialization."""
    require(phase in ("input", "sentiment", "complete"), "invalid pipeline phase")
    fields(envelope, ("schema_version", "status", "input", "sentiment",
                      "personalization"), "envelope")
    require(envelope["schema_version"] == VERSION and envelope["status"] == "ok",
            "invalid envelope version or status")
    validate_input(envelope["input"])
    data = envelope["input"]
    if phase == "input":
        require(envelope["sentiment"] is None and envelope["personalization"] is None,
                "initial envelope must not contain stage results")
        return envelope
    feedback = {row["id"]: row for row in data["feedback"]}
    products = {row["id"]: row for row in data["products"]}
    rows = envelope["sentiment"]
    ids = unique_rows(rows, ("id", "customer_id", "product_id", "category", "severity",
                            "score", "label", "matches", "is_issue", "priority"),
                      "sentiment")
    require(ids == set(feedback), "sentiment must cover every feedback exactly once")
    for row in rows:
        original = feedback[row["id"]]
        for key in ("customer_id", "product_id", "severity"):
            require(row[key] == original[key], "sentiment changed feedback provenance")
        require(row["category"] == products[row["product_id"]]["category"],
                "sentiment category does not match catalog")
        number(row["score"], -1, 1, "sentiment.score")
        expected_label = "positive" if row["score"] > 0 else (
            "negative" if row["score"] < 0 else "neutral")
        require(row["label"] == expected_label, "sentiment label mismatch")
        require(isinstance(row["matches"], list), "matches must be an array")
        for match in row["matches"]:
            fields(match, ("token", "index", "negated", "contribution"), "match")
            text(match["token"], "match.token")
            require(type(match["index"]) is int and match["index"] >= 0,
                    "match.index must be a nonnegative integer")
            require(type(match["negated"]) is bool, "match.negated must be boolean")
            require(type(match["contribution"]) is int and
                    match["contribution"] in (-1, 1), "invalid match contribution")
        expected_score = round(sum(m["contribution"] for m in row["matches"]) /
                               len(row["matches"]), 6) if row["matches"] else 0.0
        require(row["score"] == expected_score, "sentiment score lacks evidence")
        issue = row["score"] < 0 or SEVERITY[row["severity"]] >= 3
        require(type(row["is_issue"]) is bool and row["is_issue"] == issue,
                "invalid issue flag")
        number(row["priority"], 0, 100, "sentiment.priority")
        require(row["priority"] == priority(row["score"], row["severity"]),
                "priority does not match severity and sentiment")
    require(rows == sorted(rows, key=lambda r: (-r["priority"], r["id"])),
            "issues must be ordered by priority then id")
    if phase == "sentiment":
        require(envelope["personalization"] is None, "premature personalization")
        return envelope
    results = envelope["personalization"]
    customer_ids = unique_rows(results, ("id", "cold_start", "recommendations"),
                               "personalization")
    require(customer_ids == {r["id"] for r in data["customers"]},
            "personalization must cover every customer")
    for result in results:
        require(type(result["cold_start"]) is bool, "cold_start must be boolean")
        recommendations = result["recommendations"]
        ranked_ids = unique_rows(recommendations,
                                 ("id", "category", "score", "base_score",
                                  "issue_penalty", "issue_ids", "item_affinity",
                                  "category_affinity", "popularity"),
                                 "recommendations")
        require(ranked_ids <= set(products), "recommendation references unknown product")
        require(len(recommendations) == min(data["top_k"], len(products)),
                "incorrect number of recommendations")
        for item in recommendations:
            product = products[item["id"]]
            require(item["category"] == product["category"] and
                    item["popularity"] == product["popularity"], "catalog mismatch")
            for key in ("score", "base_score", "item_affinity", "category_affinity",
                        "popularity"):
                number(item[key], 0, 1, "recommendation." + key)
            number(item["issue_penalty"], 0, 0.5, "recommendation.issue_penalty")
            applicable = [r for r in rows if r["customer_id"] == result["id"] and
                          r["category"] == item["category"] and r["is_issue"]]
            require(item["issue_ids"] == sorted(r["id"] for r in applicable),
                    "invalid issue propagation")
            penalty = max((r["priority"] for r in applicable), default=0) / 200
            require(item["issue_penalty"] == round(penalty, 6),
                    "invalid category penalty")
            require(item["score"] == round(max(0, item["base_score"] -
                                               item["issue_penalty"]), 6),
                    "invalid final ranking score")
        require(recommendations == sorted(
            recommendations, key=lambda r: (-r["score"], -r["popularity"], r["id"])),
            "recommendations must be sorted deterministically")
    return envelope


def priority(score, severity):
    if score >= 0 and SEVERITY[severity] < 3:
        return 0.0
    return round(SEVERITY[severity] * 20 + max(0, -score) * 20, 6)


def sentiment_stage(envelope):
    validate(envelope, "input")
    result = copy.deepcopy(envelope)
    products = {p["id"]: p for p in result["input"]["products"]}
    rows = []
    for feedback in result["input"]["feedback"]:
        # Negation scope is intentionally just the immediately preceding word.
        tokens = re.findall(r"[a-z]+", feedback["text"].lower())
        matches = []
        for index, token in enumerate(tokens):
            if token not in POSITIVE and token not in NEGATIVE:
                continue
            negated = index > 0 and tokens[index - 1] in NEGATORS
            contribution = (1 if token in POSITIVE else -1) * (-1 if negated else 1)
            matches.append({"token": token, "index": index, "negated": negated,
                            "contribution": contribution})
        score = round(sum(m["contribution"] for m in matches) / len(matches), 6) \
            if matches else 0.0
        rows.append({
            "id": feedback["id"], "customer_id": feedback["customer_id"],
            "product_id": feedback["product_id"],
            "category": products[feedback["product_id"]]["category"],
            "severity": feedback["severity"], "score": score,
            "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "matches": matches,
            "is_issue": score < 0 or SEVERITY[feedback["severity"]] >= 3,
            "priority": priority(score, feedback["severity"]),
        })
    result["sentiment"] = sorted(rows, key=lambda row: (-row["priority"], row["id"]))
    return validate(result, "sentiment")


def behavior_stage(envelope):
    validate(envelope, "sentiment")
    result = copy.deepcopy(envelope)
    data = result["input"]
    products = {p["id"]: p for p in data["products"]}
    now = timestamp(data["as_of"], "as_of")
    profiles = []
    for customer in sorted(data["customers"], key=lambda row: row["id"]):
        item_weights = {}
        category_weights = {}
        for event in sorted(data["events"], key=lambda row: row["id"]):
            if event["customer_id"] != customer["id"]:
                continue
            age = (now - timestamp(event["timestamp"], "event.timestamp")).total_seconds()
            weight = EVENT_WEIGHT[event["kind"]] * 2 ** (-age / 86400 / HALF_LIFE_DAYS)
            item_id = event["product_id"]
            category = products[item_id]["category"]
            item_weights[item_id] = item_weights.get(item_id, 0.0) + weight
            category_weights[category] = category_weights.get(category, 0.0) + weight
        item_max = max(item_weights.values(), default=0.0)
        category_max = max(category_weights.values(), default=0.0)
        cold_start = item_max == 0
        ranked = []
        for product in data["products"]:
            item_affinity = item_weights.get(product["id"], 0.0) / item_max \
                if item_max else 0.0
            category_affinity = category_weights.get(product["category"], 0.0) / \
                category_max if category_max else 0.0
            base = product["popularity"] if cold_start else (
                0.7 * item_affinity + 0.3 * category_affinity)
            issues = [row for row in result["sentiment"]
                      if row["customer_id"] == customer["id"] and row["is_issue"] and
                      row["category"] == product["category"]]
            penalty = round(max((row["priority"] for row in issues), default=0) / 200, 6)
            base = round(base, 6)
            ranked.append({
                "id": product["id"], "category": product["category"],
                "score": round(max(0.0, base - penalty), 6), "base_score": base,
                "issue_penalty": penalty, "issue_ids": sorted(row["id"] for row in issues),
                "item_affinity": round(item_affinity, 6),
                "category_affinity": round(category_affinity, 6),
                "popularity": product["popularity"],
            })
        ranked.sort(key=lambda item: (-item["score"], -item["popularity"], item["id"]))
        profiles.append({"id": customer["id"], "cold_start": cold_start,
                         "recommendations": ranked[:data["top_k"]]})
    result["personalization"] = profiles
    return validate(result, "complete")


def run_pipeline(data):
    initial = {"schema_version": VERSION, "status": "ok", "input": copy.deepcopy(data),
               "sentiment": None, "personalization": None}
    return behavior_stage(sentiment_stage(initial))


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as source:
            data = json.load(source, parse_constant=reject_constant,
                             object_pairs_hook=reject_duplicate_keys)
        result = run_pipeline(data)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
