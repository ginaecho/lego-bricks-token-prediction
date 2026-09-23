"""Deterministic comparison -> feedback reference pipeline (standard library only)."""

import copy
import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


ALIASES = {
    "price": "price", "cost": "price",
    "weight": "weight", "mass": "weight",
    "battery_hours": "battery_hours", "battery": "battery_hours",
}
UNITS = {
    "price": {"": 1, "usd": 1, "$": 1},
    "weight": {"": 1, "kg": 1, "g": 0.001},
    "battery_hours": {"": 1, "h": 1, "hours": 1, "min": 1 / 60},
}
THEMES = {
    "battery": {"battery", "charge", "charging"},
    "portability": {"light", "lightweight", "heavy", "portable"},
    "quality": {"quality", "broken", "durable", "reliable"},
    "value": {"price", "cheap", "expensive", "value"},
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value, label):
    require(type(value) in (int, float), label + " must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValidationError(label + " must be finite") from None
    require(math.isfinite(result), label + " must be finite")
    return result


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required fields: " + ", ".join(sorted(set(required) - value.keys())))
    require(value.keys() <= set(required) | set(optional), "Unknown fields")


def normalize(attributes):
    require(isinstance(attributes, dict) and bool(attributes), "attributes must be a nonempty object")
    result = {}
    for alias, raw in attributes.items():
        require(alias in ALIASES, "Unsupported attribute: " + str(alias))
        key = ALIASES[alias]
        require(key not in result, "Conflicting aliases for " + key)
        if isinstance(raw, str):
            match = re.fullmatch(r"\s*(\$)?\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", raw)
            require(match is not None, "Invalid measurement for " + key)
            prefix, magnitude, suffix = match.groups()
            require(not (prefix and suffix), "Ambiguous measurement")
            unit = (prefix or suffix).lower()
            require(unit in UNITS[key], "Unsupported unit for " + key)
            value = float(magnitude) * UNITS[key][unit]
        else:
            value = number(raw, key)
        require(math.isfinite(value) and value >= 0, key + " must be finite and nonnegative")
        result[key] = value
    return result


def canonical_words(value):
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def validate(payload, stage="input"):
    """One validation boundary shared by input, comparison and final handoff."""
    require(stage in {"input", "comparison", "final"}, "Invalid validation stage")
    required = {"schema_version", "data_label", "products", "preferences", "feedback"}
    if stage != "input":
        required |= {"status", "comparison"}
    if stage == "final":
        required.add("insights")
    fields(payload, required)
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1, "Unsupported schema_version")
    text(payload["data_label"], "data_label")
    products = payload["products"]
    require(isinstance(products, list) and bool(products), "products must be a nonempty array")
    ids = set()
    normalized = {}
    for product in products:
        fields(product, {"id", "name", "attributes"})
        pid = text(product["id"], "product id")
        text(product["name"], "product name")
        require(pid not in ids, "Duplicate product id")
        ids.add(pid)
        normalized[pid] = normalize(product["attributes"])
        if stage != "input":
            require(product["attributes"].keys() == normalized[pid].keys(), "Handoff attributes must use canonical names")
            for value in product["attributes"].values():
                number(value, "normalized attribute")
    preferences = payload["preferences"]
    require(isinstance(preferences, dict) and bool(preferences), "preferences must be a nonempty object")
    weights = []
    for key, preference in preferences.items():
        require(key in UNITS, "Preference must use a canonical attribute")
        fields(preference, {"weight", "direction"})
        weight = number(preference["weight"], "preference weight")
        require(weight >= 0, "Preference weight cannot be negative")
        weights.append(weight)
        require(preference["direction"] in ("min", "max"), "direction must be min or max")
        require(all(key in attrs for attrs in normalized.values()), "Every product must supply preference attribute " + key)
    require(any(weight > 0 for weight in weights), "At least one preference weight must be positive")
    feedback = payload["feedback"]
    require(isinstance(feedback, list), "feedback must be an array")
    feedback_ids = set()
    sources = {}
    for item in feedback:
        fields(item, {"id", "product_id", "text"})
        fid = text(item["id"], "feedback id")
        require(fid not in feedback_ids, "Duplicate feedback id")
        feedback_ids.add(fid)
        pid = text(item["product_id"], "feedback product_id")
        require(pid in ids, "Feedback references an unknown product")
        require(bool(canonical_words(text(item["text"], "feedback text"))), "Feedback must contain words")
        sources[fid] = item
    if stage != "input":
        require(payload["status"] == "ok", "Invalid status")
        rows = payload["comparison"]
        require(isinstance(rows, list) and len(rows) == len(products), "Incomplete comparison")
        seen = set()
        for rank, row in enumerate(rows, 1):
            fields(row, {"product_id", "rank", "score", "attributes"})
            pid = text(row["product_id"], "comparison product_id")
            require(pid in ids and pid not in seen, "Invalid comparison product")
            seen.add(pid)
            require(type(row["rank"]) is int and row["rank"] == rank, "Ranks must be consecutive")
            score = number(row["score"], "comparison score")
            require(0 <= score <= 1, "Score out of range")
            require(row["attributes"] == normalized[pid], "Comparison attributes lost provenance")
        require(rows == sorted(rows, key=lambda row: (-row["score"], row["product_id"])), "Comparison order is invalid")
    if stage == "final":
        insights = payload["insights"]
        require(isinstance(insights, list) and len(insights) == len(products), "Incomplete insights")
        for row, insight in zip(payload["comparison"], insights):
            fields(insight, {"product_id", "rank", "score", "raw_count", "unique_count", "themes"})
            for key in ("product_id", "rank", "score"):
                require(insight[key] == row[key], "Comparison metadata not propagated")
            expected = analyze_product(row, feedback)
            require(insight == expected, "Insights must match deduplicated sources and exact excerpts")
    return payload


def compare(payload):
    validate(payload)
    result = copy.deepcopy(payload)
    for product in result["products"]:
        product["attributes"] = normalize(product["attributes"])
    preferences = result["preferences"]
    # Scale first to avoid overflow when many large but finite weights are added.
    largest = max(float(item["weight"]) for item in preferences.values())
    scaled = {key: item["weight"] / largest for key, item in preferences.items()}
    total = sum(scaled.values())
    bounds = {
        key: (min(p["attributes"][key] for p in result["products"]),
              max(p["attributes"][key] for p in result["products"]))
        for key in preferences
    }
    rows = []
    for product in result["products"]:
        score = 0.0
        for key, preference in preferences.items():
            low, high = bounds[key]
            value = product["attributes"][key]
            utility = 1.0 if low == high else (value - low) / (high - low)
            if low != high and preference["direction"] == "min":
                utility = 1 - utility
            score += utility * (scaled[key] / total)
        rows.append({"product_id": product["id"], "score": round(min(1.0, max(0.0, score)), 12),
                     "attributes": copy.deepcopy(product["attributes"])})
    rows.sort(key=lambda item: (-item["score"], item["product_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    result.update(status="ok", comparison=rows)
    return validate(result, "comparison")


def analyze_product(row, feedback):
    records = [item for item in feedback if item["product_id"] == row["product_id"]]
    groups = {}
    for item in records:
        key = tuple(canonical_words(item["text"]))
        groups.setdefault(key, []).append(item)
    themes = {}
    for words, duplicates in groups.items():
        labels = [label for label, keywords in THEMES.items() if set(words) & keywords] or ["general"]
        for label in labels:
            theme = themes.setdefault(label, {"theme": label, "count": 0, "support": []})
            theme["count"] += 1
            theme["support"].append({
                "feedback_ids": [item["id"] for item in duplicates],
                "excerpts": [{"feedback_id": item["id"], "text": item["text"]} for item in duplicates],
            })
    return {
        "product_id": row["product_id"], "rank": row["rank"], "score": row["score"],
        "raw_count": len(records), "unique_count": len(groups),
        "themes": [themes[key] for key in sorted(themes)],
    }


def feedback_analysis(comparison):
    validate(comparison, "comparison")
    result = copy.deepcopy(comparison)
    result["insights"] = [analyze_product(row, result["feedback"]) for row in result["comparison"]]
    return validate(result, "final")


def run(payload):
    return feedback_analysis(compare(payload))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        raw = Path(argv[0]).read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=unique_object)
        output = run(payload)
        code = 0
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        output = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
