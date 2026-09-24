"""Deterministic synthetic interest discovery. Python standard library only."""

import json
import math
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(required <= value.keys(), f"{path} missing required fields")
    require(value.keys() <= required | optional, f"{path} contains unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonempty text")
    require(value == value.strip(), f"{path} must not have surrounding whitespace")


def text_list(value, path):
    require(isinstance(value, list), f"{path} must be an array")
    for entry in value:
        text(entry, path)
    require(len(value) == len(set(value)), f"{path} must contain unique values")


def number(value, low, high, path):
    require(type(value) in (int, float), f"{path} must be numeric, not boolean")
    require(low <= value <= high and math.isfinite(value),
            f"{path} must be finite and between {low} and {high}")


def validate(data):
    """One shared validation boundary used by both the API and CLI.

    Identifiers and interest labels are exact, case-sensitive strings.
    All fields are required; unknown fields are rejected.
    """
    fields(data, {"schema_version", "fixture_label", "preferences", "catalog", "limit"}, set(), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["fixture_label"] == "synthetic", "fixture_label must be synthetic")
    require(type(data["limit"]) is int and 0 <= data["limit"] <= 100,
            "limit must be an integer between 0 and 100")
    pref = data["preferences"]
    fields(pref, {"interests", "exclusions"}, set(), "preferences")
    require(isinstance(pref["interests"], dict), "interests must be an object")
    for interest, weight in pref["interests"].items():
        text(interest, "interest")
        number(weight, 0, 5, "interest weight")
    excluded = pref["exclusions"]
    fields(excluded, {"item_ids", "categories", "tags"}, set(), "exclusions")
    for key, values in excluded.items():
        text_list(values, f"exclusions.{key}")
    require(isinstance(data["catalog"], list), "catalog must be an array")
    seen = set()
    for item in data["catalog"]:
        fields(item, {"id", "title", "category", "tags", "popularity"}, set(), "catalog item")
        for key in ("id", "title", "category"):
            text(item[key], key)
        require(item["id"] not in seen, "catalog item ids must be unique")
        seen.add(item["id"])
        text_list(item["tags"], "tags")
        number(item["popularity"], 0, 1, "popularity")
    return data


def recommend(data):
    data = validate(data)
    preferences = data["preferences"]
    weights = {key: weight for key, weight in preferences["interests"].items() if weight > 0}
    total = math.fsum(weights.values())
    mode = "personalized" if weights else "popularity_fallback"
    exclusions = {key: set(value) for key, value in preferences["exclusions"].items()}
    candidates, excluded, unmatched = [], [], []
    for item in data["catalog"]:
        reasons = []
        if item["id"] in exclusions["item_ids"]:
            reasons.append({"kind": "item_id", "value": item["id"]})
        if item["category"] in exclusions["categories"]:
            reasons.append({"kind": "category", "value": item["category"]})
        reasons.extend({"kind": "tag", "value": tag}
                       for tag in sorted(set(item["tags"]) & exclusions["tags"]))
        if reasons:
            excluded.append({"item_id": item["id"], "reasons": reasons})
            continue
        matches = []
        for interest in sorted(weights):
            sources = []
            if interest == item["category"]:
                sources.append("category")
            if interest in item["tags"]:
                sources.append("tags")
            if sources:
                matches.append({"interest": interest, "weight": weights[interest],
                                "sources": sources})
        if weights and not matches:
            unmatched.append(item["id"])
            continue
        score = math.fsum(match["weight"] for match in matches) / total if weights else 0.0
        explanation = {
            "matched_interests": matches,
            "popularity": item["popularity"],
            "reason": ("Ranked by weighted category/tag matches; popularity breaks ties."
                       if weights else "No positive interests supplied; ranked by catalog popularity.")
        }
        candidates.append({"item_id": item["id"], "title": item["title"],
                           "score": score, "explanation": explanation})
    candidates.sort(key=lambda row: (-row["score"], -row["explanation"]["popularity"], row["item_id"]))
    selected = candidates[:data["limit"]]
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return {
        "status": "ok", "schema_version": 1, "fixture_label": "synthetic", "mode": mode,
        "recommendations": selected,
        "audit": {"catalog_count": len(data["catalog"]), "eligible_count": len(candidates),
                  "excluded": sorted(excluded, key=lambda row: row["item_id"]),
                  "unmatched_item_ids": sorted(unmatched),
                  "omitted_by_limit": len(candidates) - len(selected)}
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        raw = Path(argv[0]).read_text(encoding="utf-8")
        data = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = recommend(data)
        code = 0
    except (OSError, ValueError, RecursionError) as exc:
        result = {"status": "error", "error": {"type": "validation_or_file_error", "message": str(exc)}}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
