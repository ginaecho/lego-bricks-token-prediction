"""Deterministic behavioral discovery using synthetic-compatible JSON input.

Run: python -B implementation.py example_input.json
Scores combine item and category affinity with an optional popularity prior.
No wall clock, randomness, external dependencies, or provider calls are used.
"""

import json
import math
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def fields(value, required, optional, path):
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise ValidationError(
            f"{path}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}"
        )


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a nonempty string")
    return value


def number(value, path, minimum=0, maximum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValidationError(f"{path} must be finite in [{minimum}, {maximum}]")
    return float(value)


def timestamp(value, path):
    text(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{path} must be an ISO 8601 timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValidationError(f"{path} must contain a timezone")
    return parsed


def validate(payload):
    """One validation boundary for every discovery input and CLI request."""
    fields(payload, {"schema_version", "synthetic", "as_of", "user_id", "catalog", "events"},
           {"limit", "half_life_days", "popularity_weight"}, "input")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    if type(payload["synthetic"]) is not bool:
        raise ValidationError("synthetic must be boolean")
    user_id = text(payload["user_id"], "user_id")
    as_of = timestamp(payload["as_of"], "as_of")
    limit = payload.get("limit", 10)
    if type(limit) is not int or not 0 <= limit <= 1000:
        raise ValidationError("limit must be an integer in [0, 1000]")
    half_life = number(payload.get("half_life_days", 30), "half_life_days", 0.001, 36500)
    prior = number(payload.get("popularity_weight", 0.1), "popularity_weight")
    if not isinstance(payload["catalog"], list) or len(payload["catalog"]) > 10000:
        raise ValidationError("catalog must be a list with at most 10000 items")
    catalog = {}
    for index, item in enumerate(payload["catalog"]):
        path = f"catalog[{index}]"
        fields(item, {"item_id", "category", "popularity", "available"}, set(), path)
        item_id = text(item["item_id"], f"{path}.item_id")
        text(item["category"], f"{path}.category")
        number(item["popularity"], f"{path}.popularity")
        if type(item["available"]) is not bool:
            raise ValidationError(f"{path}.available must be boolean")
        if item_id in catalog:
            raise ValidationError(f"duplicate item_id: {item_id}")
        catalog[item_id] = dict(item)
    if not isinstance(payload["events"], list) or len(payload["events"]) > 100000:
        raise ValidationError("events must be a list with at most 100000 entries")
    events = []
    event_ids = set()
    for index, event in enumerate(payload["events"]):
        path = f"events[{index}]"
        fields(event, {"event_id", "user_id", "item_id", "kind", "timestamp"}, set(), path)
        event_id = text(event["event_id"], f"{path}.event_id")
        text(event["user_id"], f"{path}.user_id")
        item_id = text(event["item_id"], f"{path}.item_id")
        kind = text(event["kind"], f"{path}.kind")
        if event_id in event_ids:
            raise ValidationError(f"duplicate event_id: {event_id}")
        event_ids.add(event_id)
        if item_id not in catalog:
            raise ValidationError(f"{path}.item_id references unknown catalog item")
        if kind not in ("browse", "purchase"):
            raise ValidationError(f"{path}.kind must be browse or purchase")
        at = timestamp(event["timestamp"], f"{path}.timestamp")
        if at > as_of:
            raise ValidationError(f"{path}.timestamp must not be after as_of")
        events.append({**event, "_at": at})
    return user_id, as_of, limit, half_life, prior, catalog, events


def discover(payload):
    user_id, as_of, limit, half_life, prior, catalog, events = validate(payload)
    item_weights = {}
    category_weights = {}
    selected = sorted(
        (event for event in events if event["user_id"] == user_id),
        key=lambda event: event["event_id"],
    )
    for event in selected:
        age_days = (as_of - event["_at"]).total_seconds() / 86400
        weight = (3 if event["kind"] == "purchase" else 1) * 2 ** (-age_days / half_life)
        item_id = event["item_id"]
        category = catalog[item_id]["category"]
        item_weights[item_id] = item_weights.get(item_id, 0) + weight
        category_weights[category] = category_weights.get(category, 0) + weight
    total = math.fsum(item_weights.values())
    cold_start = total == 0
    ranked = []
    for item_id, item in catalog.items():
        if not item["available"]:
            continue
        item_affinity = 0 if cold_start else item_weights.get(item_id, 0) / total
        category_affinity = 0 if cold_start else category_weights.get(item["category"], 0) / total
        score = (
            item["popularity"] if cold_start
            else 0.7 * item_affinity + 0.3 * category_affinity + prior * item["popularity"]
        )
        ranked.append({
            "item_id": item_id,
            "score": score,
            "reason": "popularity_fallback" if cold_start else "behavioral_affinity",
            "components": {
                "item_affinity": item_affinity,
                "category_affinity": category_affinity,
                "popularity": item["popularity"],
            },
        })
    ranked.sort(key=lambda item: (-item["score"], item["item_id"]))
    recommendations = ranked[:limit]
    for recommendation in recommendations:
        recommendation["score"] = round(recommendation["score"], 12)
        recommendation["components"] = {
            key: round(value, 12) for key, value in recommendation["components"].items()
        }
    return {
        "schema_version": 1,
        "status": "ok",
        "synthetic": payload["synthetic"],
        "user_id": user_id,
        "as_of": payload["as_of"],
        "cold_start": cold_start,
        "history_events_used": len(selected),
        "recommendations": recommendations,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as source:
            payload = json.load(source, object_pairs_hook=unique_object)
        output = discover(payload)
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
