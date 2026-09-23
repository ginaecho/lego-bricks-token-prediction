"""Deterministic personalized discovery; Python standard library only."""

import datetime as dt
import json
import math
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def object_fields(value, required, optional, path):
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


def number(value, path, low, high, exclusive_low=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path} must be a number")
    if not low <= value <= high or (exclusive_low and value == low):
        raise ValidationError(f"{path} is outside its allowed range")
    return value


def timestamp(value, path):
    text(value, path)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.astimezone(dt.timezone.utc)
    except (ValueError, OverflowError):
        raise ValidationError(f"{path} must be a valid timezone-aware ISO 8601 timestamp")


def validate(payload):
    """The shared validation boundary for library and CLI input."""
    object_fields(
        payload, {"user_id", "now", "items", "events"},
        {"limit", "half_life_days", "exclude_purchased", "fixture_label"}, "input",
    )
    user_id = text(payload["user_id"], "user_id")
    now = timestamp(payload["now"], "now")
    if "fixture_label" in payload:
        text(payload["fixture_label"], "fixture_label")
    limit = payload.get("limit", 10)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 1000:
        raise ValidationError("limit must be an integer from 0 to 1000")
    half_life = number(payload.get("half_life_days", 30), "half_life_days", 0, 3650, True)
    exclude = payload.get("exclude_purchased", True)
    if not isinstance(exclude, bool):
        raise ValidationError("exclude_purchased must be a boolean")
    items = payload["items"]
    events = payload["events"]
    if not isinstance(items, list) or not isinstance(events, list):
        raise ValidationError("items and events must be arrays")
    catalog = {}
    for index, item in enumerate(items):
        path = f"items[{index}]"
        object_fields(item, {"id", "title", "category", "popularity"}, set(), path)
        for key in ("id", "title", "category"):
            text(item[key], f"{path}.{key}")
        number(item["popularity"], f"{path}.popularity", 0, 1)
        if item["id"] in catalog:
            raise ValidationError(f"{path}.id is duplicated")
        catalog[item["id"]] = dict(item)
    normalized_events = []
    for index, event in enumerate(events):
        path = f"events[{index}]"
        object_fields(event, {"user_id", "item_id", "type", "timestamp"}, set(), path)
        text(event["user_id"], f"{path}.user_id")
        text(event["item_id"], f"{path}.item_id")
        if event["item_id"] not in catalog:
            raise ValidationError(f"{path}.item_id is not in the catalog")
        if event["type"] not in ("browse", "purchase"):
            raise ValidationError(f"{path}.type must be browse or purchase")
        when = timestamp(event["timestamp"], f"{path}.timestamp")
        if when > now:
            raise ValidationError(f"{path}.timestamp cannot be in the future")
        normalized_events.append({**event, "timestamp": when})
    return user_id, now, catalog, normalized_events, limit, half_life, exclude


def personalize(payload):
    user_id, now, catalog, events, limit, half_life, exclude = validate(payload)
    own_events = [event for event in events if event["user_id"] == user_id]
    purchased = {event["item_id"] for event in own_events if event["type"] == "purchase"}
    item_weights = {}
    category_weights = {}
    for event in own_events:
        age_days = (now - event["timestamp"]).total_seconds() / 86400
        weight = (3.0 if event["type"] == "purchase" else 1.0) * 2.0 ** (-age_days / half_life)
        item_id = event["item_id"]
        category = catalog[item_id]["category"]
        item_weights.setdefault(item_id, []).append(weight)
        category_weights.setdefault(category, []).append(weight)
    # Sorting before summation also makes event order irrelevant numerically.
    direct = {key: math.fsum(sorted(values)) for key, values in item_weights.items()}
    categories = {key: math.fsum(sorted(values)) for key, values in category_weights.items()}
    total = math.fsum(sorted(direct.values()))
    strategy = "behavioral" if total > 0 else "cold_start"
    ranked = []
    for item_id, item in catalog.items():
        if exclude and item_id in purchased:
            continue
        category_affinity = categories.get(item["category"], 0) / total if total else 0.0
        item_affinity = direct.get(item_id, 0) / total if total else 0.0
        popularity = item["popularity"]
        score = (
            0.65 * category_affinity + 0.25 * item_affinity + 0.10 * popularity
            if total else popularity
        )
        ranked.append((score, {
            "item_id": item_id,
            "title": item["title"],
            "category": item["category"],
            "score": round(score, 8),
            "signals": {
                "category_affinity": round(category_affinity, 8),
                "item_affinity": round(item_affinity, 8),
                "popularity": popularity,
            },
        }))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]["item_id"]))
    return {
        "status": "ok",
        "user_id": user_id,
        "strategy": strategy,
        "matched_event_count": len(own_events),
        "excluded_purchase_count": len(purchased) if exclude else 0,
        "recommendations": [item for _, item in ranked[:limit]],
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        payload = json.loads(
            Path(args[0]).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object, parse_constant=reject_constant,
        )
        result = personalize(payload)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
