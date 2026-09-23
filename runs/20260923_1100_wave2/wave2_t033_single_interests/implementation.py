"""Deterministic synthetic interest recommendations; Python standard library only."""

import json
import sys
from pathlib import Path


class ValidationError(ValueError):
    """An input does not conform to the shared schema."""


def object_fields(value, required, path):
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    if set(value) != set(required):
        raise ValidationError(f"{path} must contain exactly: {', '.join(required)}")


def text(value, path, canonical=False):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValidationError(f"{path} must be a nonblank string of at most 200 characters")
    value = value.strip()
    return value.casefold() if canonical else value


def strings(value, path, canonical=False, maximum=100):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValidationError(f"{path} must be an array of at most {maximum} strings")
    result = [text(item, f"{path}[{i}]", canonical) for i, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ValidationError(f"{path} contains duplicate normalized values")
    return result


def integer(value, path, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{path} must be an integer from {minimum} to {maximum}")
    return value


def validate_input(data):
    """Validate once and return a fresh, canonical schema shared by all logic."""
    object_fields(data, ("fixture_label", "user", "catalog", "limit"), "input")
    label = text(data["fixture_label"], "fixture_label")
    if not label.casefold().startswith("synthetic"):
        raise ValidationError("fixture_label must start with 'synthetic'")
    object_fields(data["user"], ("interests", "exclusions"), "user")
    user = data["user"]
    if not isinstance(user["interests"], dict) or len(user["interests"]) > 100:
        raise ValidationError("user.interests must be an object with at most 100 entries")
    interests = {}
    for tag, weight in user["interests"].items():
        normalized = text(tag, "interest name", canonical=True)
        if normalized in interests:
            raise ValidationError("user.interests contains duplicate normalized names")
        interests[normalized] = integer(weight, f"interest[{normalized}]", 1, 100)
    object_fields(user["exclusions"], ("item_ids", "categories", "tags"), "exclusions")
    exclusions = {
        key: strings(value, f"exclusions.{key}", canonical=key != "item_ids",
                     maximum=1000 if key == "item_ids" else 100)
        for key, value in user["exclusions"].items()
    }
    if not isinstance(data["catalog"], list) or len(data["catalog"]) > 1000:
        raise ValidationError("catalog must be an array of at most 1000 items")
    catalog = []
    seen = set()
    for index, raw in enumerate(data["catalog"]):
        path = f"catalog[{index}]"
        object_fields(raw, ("id", "title", "category", "tags"), path)
        item = {
            "id": text(raw["id"], f"{path}.id"),
            "title": text(raw["title"], f"{path}.title"),
            "category": text(raw["category"], f"{path}.category", canonical=True),
            "tags": strings(raw["tags"], f"{path}.tags", canonical=True),
        }
        if item["id"] in seen:
            raise ValidationError("catalog item IDs must be unique")
        seen.add(item["id"])
        catalog.append(item)
    return {
        "fixture_label": label,
        "user": {"interests": interests, "exclusions": exclusions},
        "catalog": catalog,
        "limit": integer(data["limit"], "limit", 0, 100),
    }


def recommend(data):
    """Exclude first, rank tag matches by summed weight, explain only evidence."""
    data = validate_input(data)
    interests = data["user"]["interests"]
    exclusions = data["user"]["exclusions"]
    candidates, excluded = [], []
    unmatched = 0
    for item in data["catalog"]:
        reasons = []
        if item["id"] in exclusions["item_ids"]:
            reasons.append({"field": "id", "value": item["id"]})
        if item["category"] in exclusions["categories"]:
            reasons.append({"field": "category", "value": item["category"]})
        for tag in sorted(set(item["tags"]) & set(exclusions["tags"])):
            reasons.append({"field": "tags", "value": tag})
        if reasons:
            excluded.append({"item_id": item["id"], "reasons": reasons})
            continue
        matches = [
            {"tag": tag, "weight": interests[tag], "source": "catalog.tags"}
            for tag in sorted(set(item["tags"]) & set(interests))
        ]
        if not matches:
            unmatched += 1
            continue
        score = sum(match["weight"] for match in matches)
        evidence = ", ".join(f"{match['tag']} (weight {match['weight']})" for match in matches)
        candidates.append({
            "item_id": item["id"],
            "title": item["title"],
            "score": score,
            "matched_interests": matches,
            "explanation": f"Catalog tags match your interests: {evidence}. Total score: {score}.",
        })
    candidates.sort(key=lambda item: (-item["score"], item["item_id"]))
    recommendations = [
        {"rank": rank, **item}
        for rank, item in enumerate(candidates[:data["limit"]], 1)
    ]
    return {
        "status": "ok",
        "fixture_label": data["fixture_label"],
        "recommendations": recommendations,
        "audit": {
            "catalog_count": len(data["catalog"]),
            "eligible_match_count": len(candidates),
            "unmatched_count": unmatched,
            "excluded": sorted(excluded, key=lambda item: item["item_id"]),
            "returned_count": len(recommendations),
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"Non-finite JSON number is not allowed: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = recommend(data)
        code = 0
    except (ValueError, OSError, RecursionError) as exc:
        result = {"status": "error", "error": {"message": str(exc)}}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
