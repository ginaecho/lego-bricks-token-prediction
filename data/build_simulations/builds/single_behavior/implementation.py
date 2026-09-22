"""Deterministic behavioral ranking; no dependencies or provider integration."""

import argparse
import copy
import datetime as dt
import json
import math
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: expected nonempty string")
    return value


def object_fields(value, allowed, required, label):
    require(isinstance(value, dict), f"{label}: expected object")
    require(set(value) <= set(allowed), f"{label}: unknown fields")
    require(set(required) <= set(value), f"{label}: missing required fields")


def strings(value, label):
    require(isinstance(value, list), f"{label}: expected array")
    for item in value:
        text(item, label)
    require(len(value) == len(set(value)), f"{label}: duplicate values")
    return set(value)


def timestamp(value, label):
    text(value, label)
    require("T" in value, f"{label}: expected ISO timestamp with time")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.utcoffset() is not None, f"{label}: timezone required")
        return result.astimezone(dt.timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{label}: invalid timezone-aware timestamp") from exc


def finite(value, label, positive=False):
    require(type(value) in (int, float), f"{label}: expected number")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    require(valid, f"{label}: expected finite {'positive' if positive else 'nonnegative'} number")
    return value


def rank_products(data, explanation_callback=None):
    """Return ranking; callback selects grounded claim IDs, never arbitrary prose.

    Callback input: {product_id, score, claims: {claim_id: grounded_text}}.
    Callback output: {product_id, claim_ids: [one or more offered claim IDs]}.
    Invalid callback output raises ValidationError; callback exceptions propagate.
    """
    object_fields(
        data, ("as_of", "products", "events", "preferences", "exclusions", "half_life_days"),
        ("as_of", "products"), "input",
    )
    now = timestamp(data["as_of"], "as_of")
    half_life = finite(data.get("half_life_days", 30), "half_life_days", positive=True)
    require(isinstance(data["products"], list), "products: expected array")
    catalog = {}
    for product in data["products"]:
        object_fields(product, ("id", "name", "category", "tags"), ("id", "category", "tags"), "product")
        pid = text(product["id"], "product.id")
        require(pid not in catalog, f"duplicate product ID: {pid}")
        text(product["category"], "product.category")
        strings(product["tags"], "product.tags")
        if "name" in product:
            text(product["name"], "product.name")
        catalog[pid] = product

    preferences = data.get("preferences", {})
    object_fields(preferences, ("categories", "tags"), (), "preferences")
    preferred_categories = strings(preferences.get("categories", []), "preferences.categories")
    preferred_tags = strings(preferences.get("tags", []), "preferences.tags")
    exclusions = data.get("exclusions", {})
    object_fields(exclusions, ("product_ids", "categories", "tags"), (), "exclusions")
    excluded_ids = strings(exclusions.get("product_ids", []), "exclusions.product_ids")
    excluded_categories = strings(exclusions.get("categories", []), "exclusions.categories")
    excluded_tags = strings(exclusions.get("tags", []), "exclusions.tags")
    require(excluded_ids <= catalog.keys(), "exclusions: unknown product ID")
    eligible = {
        pid: product for pid, product in catalog.items()
        if pid not in excluded_ids
        and product["category"] not in excluded_categories
        and not excluded_tags.intersection(product["tags"])
    }
    events = data.get("events", [])
    require(isinstance(events, list), "events: expected array")
    event_ids, event_signatures = set(), set()
    validated_events = []
    for event in events:
        object_fields(
            event, ("id", "product_id", "type", "count", "timestamp"),
            ("id", "product_id", "type", "count", "timestamp"), "event",
        )
        eid = text(event["id"], "event.id")
        require(eid not in event_ids, f"duplicate event ID: {eid}")
        event_ids.add(eid)
        pid = text(event["product_id"], "event.product_id")
        require(pid in catalog, f"event: unknown product ID: {pid}")
        require(event["type"] in ("browse", "purchase"), "event: invalid type")
        require(type(event["count"]) is int, "event.count: expected integer")
        finite(event["count"], "event.count")
        when = timestamp(event["timestamp"], "event.timestamp")
        require(when <= now, "event: future timestamp")
        signature = (pid, event["type"], when, event["count"])
        require(signature not in event_signatures, "duplicate event content")
        event_signatures.add(signature)
        validated_events.append((when, eid, event))

    direct = {pid: 0.0 for pid in eligible}
    category_weights, tag_weights = {}, {}
    active_count = 0
    # Canonical accumulation order makes input order irrelevant.
    for when, _, event in sorted(validated_events):
        pid = event["product_id"]
        if pid not in eligible or event["count"] == 0:
            continue
        age_days = (now - when).total_seconds() / 86400
        weight = event["count"] * (4.0 if event["type"] == "purchase" else 1.0)
        weight *= 0.5 ** (age_days / half_life)
        finite(weight, "event weighted score")
        active_count += 1
        direct[pid] += weight
        product = eligible[pid]
        category = product["category"]
        category_weights[category] = category_weights.get(category, 0.0) + weight
        for tag in sorted(product["tags"]):
            tag_weights[tag] = tag_weights.get(tag, 0.0) + weight / max(1, len(product["tags"]))

    cold_start = active_count == 0
    ranked = []
    for pid, product in eligible.items():
        category = product["category"]
        matched_tags = sorted(preferred_tags.intersection(product["tags"]))
        components = {
            "direct_behavior": direct[pid],
            "category_affinity": 0.25 * category_weights.get(category, 0.0),
            "tag_affinity": 0.25 * sum(tag_weights.get(tag, 0.0) for tag in sorted(product["tags"])),
            "explicit_preferences": float(2 * (category in preferred_categories) + len(matched_tags)),
        }
        score = sum(components.values())
        finite(score, "total score")
        claims = {}
        if direct[pid] > 0:
            claims["direct_behavior"] = f"Recency-weighted activity on this product contributes {direct[pid]:.6f}."
        if components["category_affinity"] > 0:
            claims["category_affinity"] = f"Activity in category {category!r} contributes {components['category_affinity']:.6f}."
        if components["tag_affinity"] > 0:
            claims["tag_affinity"] = f"Activity on shared tags contributes {components['tag_affinity']:.6f}."
        if category in preferred_categories:
            claims["preferred_category"] = f"Explicit preference for category {category!r} contributes 2."
        if matched_tags:
            claims["preferred_tags"] = f"Explicit preferred tags {matched_tags!r} contribute {len(matched_tags)}."
        if cold_start:
            claims["cold_start"] = "No positive-count eligible history; using explicit preferences, then product ID."
        if score == 0:
            claims["no_signal"] = "No positive scoring signal; ties use ascending product ID."
        if not claims:
            claims["decayed_history"] = "Historical activity has decayed to zero; ties use ascending product ID."
        ranked.append({
            "product_id": pid, "score": score, "components": components,
            "claims": claims, "explanations": list(claims.values()),
            "explanation_source": "deterministic",
        })
    ranked.sort(key=lambda item: (-item["score"], item["product_id"]))
    if explanation_callback is not None:
        require(callable(explanation_callback), "explanation_callback: must be callable")
        for item in ranked:
            offered = item["claims"]
            payload = {"product_id": item["product_id"], "score": item["score"], "claims": copy.deepcopy(offered)}
            result = explanation_callback(payload)
            object_fields(result, ("product_id", "claim_ids"), ("product_id", "claim_ids"), "callback")
            require(result["product_id"] == item["product_id"], "callback: wrong product ID")
            selected = strings(result["claim_ids"], "callback.claim_ids")
            require(bool(selected) and selected <= offered.keys(), "callback: ungrounded or empty claims")
            item["explanations"] = [offered[key] for key in result["claim_ids"]]
            item["explanation_source"] = "injected_callback"
    return {
        "ranked_product_ids": [item["product_id"] for item in ranked],
        "cold_start": cold_start,
        "history_events_used": active_count,
        "excluded_product_ids": sorted(catalog.keys() - eligible.keys()),
        "ranking": ranked,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input JSON file (UTF-8)")
    parser.add_argument("--output", help="Output JSON file; defaults to stdout")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object)
        output = json.dumps(rank_products(data), indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as stream:
                stream.write(output + "\n")
        else:
            print(output)
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
