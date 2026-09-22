"""Interest recommendation -> comparison, using only Python's standard library.

Run: python -B implementation.py example_input.json
Optional Python API: build(payload, explanation_callback=callable).
Callbacks receive selected-product facts only and return {"claims": [fact, ...]}.
Each claim must exactly match a supplied fact; arbitrary generated prose is not
accepted because its factual grounding cannot be verified mechanically.
"""

import copy
import itertools
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def fail(message):
    raise ValidationError(message)


def obj(value, allowed, required, path):
    if not isinstance(value, dict):
        fail(f"{path}: expected object")
    if set(value) - set(allowed):
        fail(f"{path}: unknown fields {sorted(set(value) - set(allowed))}")
    if set(required) - set(value):
        fail(f"{path}: missing fields {sorted(set(required) - set(value))}")
    return value


def text(value, path, folded=False):
    if not isinstance(value, str) or not value.strip():
        fail(f"{path}: expected nonempty string")
    value = value.strip()
    return value.casefold() if folded else value


def number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{path}: expected finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        fail(f"{path}: expected finite number")
    return value


def strings(value, path, folded=False):
    if not isinstance(value, list):
        fail(f"{path}: expected array")
    result = [text(item, path, folded) for item in value]
    if len(set(result)) != len(result):
        fail(f"{path}: duplicate values after normalization")
    return result


def currency(value, path):
    value = text(value, path).upper()
    if not re.fullmatch(r"[A-Z]{3}", value):
        fail(f"{path}: expected three-letter currency")
    return value


def attribute(value, path):
    obj(value, ("value", "unit"), ("value",), path)
    scalar = value["value"]
    if scalar is not None and not isinstance(scalar, (str, bool, int, float)):
        fail(f"{path}.value: expected scalar or null")
    if isinstance(scalar, str):
        scalar = text(scalar, path)
    elif scalar is not None and not isinstance(scalar, bool):
        number(scalar, path)
    unit = value.get("unit")
    if unit is not None:
        unit = text(unit, path, folded=True)
    if unit is not None and (isinstance(scalar, (str, bool))):
        fail(f"{path}: units require numeric or missing values")
    return {"value": scalar, "unit": unit}


def normalized_map(value, path, convert):
    if not isinstance(value, dict):
        fail(f"{path}: expected object")
    result = {}
    for key, item in value.items():
        name = text(key, path, folded=True)
        if name in result:
            fail(f"{path}: duplicate normalized key {name}")
        result[name] = convert(item, f"{path}.{name}")
    return result


def normalize(payload):
    """The sole schema boundary shared by recommendation and comparison."""
    obj(payload, ("products", "preferences", "comparison_attributes", "candidate_limit"),
        ("products", "preferences", "comparison_attributes"), "input")
    if not isinstance(payload["products"], list):
        fail("products: expected array")
    products = []
    ids = set()
    for index, raw in enumerate(payload["products"]):
        path = f"products[{index}]"
        obj(raw, ("id", "name", "brand", "tags", "price", "currency", "attributes", "source"),
            ("id", "name", "brand", "tags", "price", "currency", "attributes"), path)
        product_id = text(raw["id"], f"{path}.id")
        if product_id in ids:
            fail(f"{path}: duplicate product ID {product_id}")
        ids.add(product_id)
        price = raw["price"]
        if price is not None and number(price, f"{path}.price") < 0:
            fail(f"{path}.price: cannot be negative")
        attrs = normalized_map(raw["attributes"], f"{path}.attributes", attribute)
        if "price" in attrs:
            fail(f"{path}.attributes: price is reserved for the common price field")
        products.append({
            "id": product_id,
            "name": text(raw["name"], f"{path}.name"),
            "brand": text(raw["brand"], f"{path}.brand", folded=True),
            "tags": strings(raw["tags"], f"{path}.tags", folded=True),
            "price": price,
            "currency": currency(raw["currency"], f"{path}.currency"),
            "attributes": attrs,
            "provenance": {
                "input_index": index,
                "source": text(raw["source"], f"{path}.source") if "source" in raw else None,
            },
        })
    prefs = payload["preferences"]
    obj(prefs, ("interests", "exclusions", "constraints", "attribute_preferences"),
        ("interests",), "preferences")

    def weight(value, path):
        if number(value, path) <= 0:
            fail(f"{path}: weight must be positive")
        return value

    interests = normalized_map(prefs["interests"], "preferences.interests", weight)
    if not interests:
        fail("preferences.interests: at least one interest is required")
    # fsum must remain finite even when individual input weights are finite.
    try:
        total = math.fsum(interests.values())
    except OverflowError:
        fail("preferences.interests: total weight exceeds numeric range")
    if not math.isfinite(total):
        fail("preferences.interests: total weight exceeds numeric range")
    exclusions = prefs.get("exclusions", {})
    obj(exclusions, ("product_ids", "brands", "tags"), (), "preferences.exclusions")
    exclusions = {
        key: strings(exclusions.get(key, []), f"preferences.exclusions.{key}",
                     folded=key != "product_ids")
        for key in ("product_ids", "brands", "tags")
    }
    constraints = prefs.get("constraints", {})
    obj(constraints, ("currency", "max_price", "required_tags"), (), "preferences.constraints")
    max_price = constraints.get("max_price")
    if max_price is not None and number(max_price, "constraints.max_price") < 0:
        fail("constraints.max_price: cannot be negative")
    constraints = {
        "currency": currency(constraints.get("currency", "USD"), "constraints.currency"),
        "max_price": max_price,
        "required_tags": strings(constraints.get("required_tags", []),
                                 "constraints.required_tags", folded=True),
    }
    attrs = strings(payload["comparison_attributes"], "comparison_attributes", folded=True)
    if not attrs:
        fail("comparison_attributes: at least one attribute is required")

    def direction(value, path):
        value = text(value, path, folded=True)
        if value not in ("higher", "lower"):
            fail(f"{path}: expected higher or lower")
        return value

    directions = normalized_map(prefs.get("attribute_preferences", {}),
                               "preferences.attribute_preferences", direction)
    if set(directions) - set(attrs):
        fail("attribute_preferences: directions must reference comparison_attributes")
    limit = payload.get("candidate_limit", 3)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        fail("candidate_limit: expected positive integer")
    return {
        "products": products,
        "preferences": {"interests": interests, "exclusions": exclusions,
                        "constraints": constraints, "attribute_preferences": directions},
        "comparison_attributes": attrs,
        "candidate_limit": limit,
    }


def recommend(products, preferences, limit):
    """Filter once, then rank. Selection carries normalized product records."""
    interests = preferences["interests"]
    exclusions = preferences["exclusions"]
    constraints = preferences["constraints"]
    ranked, rejected = [], []
    for product in products:
        reasons = []
        if product["id"] in exclusions["product_ids"]:
            reasons.append("excluded_product")
        if product["brand"] in exclusions["brands"]:
            reasons.append("excluded_brand")
        if set(product["tags"]) & set(exclusions["tags"]):
            reasons.append("excluded_tag")
        if product["currency"] != constraints["currency"]:
            reasons.append("currency_mismatch")
        if not set(constraints["required_tags"]) <= set(product["tags"]):
            reasons.append("missing_required_tags")
        if constraints["max_price"] is not None:
            if product["price"] is None:
                reasons.append("unknown_price")
            elif product["price"] > constraints["max_price"]:
                reasons.append("over_budget")
        matched = sorted(set(product["tags"]) & set(interests))
        if not matched:
            reasons.append("no_interest_match")
        if reasons:
            rejected.append({"product_id": product["id"], "reasons": reasons,
                             "provenance": product["provenance"]})
        else:
            ranked.append({
                "product": product,
                "score": math.fsum(interests[tag] for tag in matched),
                "matched_interests": [{"tag": tag, "weight": interests[tag]} for tag in matched],
            })
    ranked.sort(key=lambda item: (-item["score"], item["product"]["id"]))
    selected = ranked[:limit]
    return {
        "selected": selected,
        "selected_ids": [item["product"]["id"] for item in selected],
        "eligible_count": len(ranked),
        "not_selected_ids": [item["product"]["id"] for item in ranked[limit:]],
        "rejected": sorted(rejected, key=lambda item: item["product_id"]),
        "tie_policy": "score descending, then case-sensitive product ID ascending",
    }


def fact(product, name):
    value = ({"value": product["price"], "unit": product["currency"]}
             if name == "price" else
             product["attributes"].get(name, {"value": None, "unit": None}))
    return {"product_id": product["id"], "attribute": name, **value,
            "provenance": product["provenance"]}


def value_kind(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


def describe(item):
    suffix = f" {item['unit']}" if item["unit"] else ""
    return f"{item['product_id']}: {json.dumps(item['value'], ensure_ascii=False)}{suffix}"


def compare(selection, preferences, names):
    """Only the stage-one selection is visible here, never the full catalog."""
    products = [item["product"] for item in selection]
    comparisons, facts = [], []
    for name in names:
        entries = [fact(product, name) for product in products]
        facts.extend(entries)
        known = [entry for entry in entries if entry["value"] is not None]
        missing = [entry["product_id"] for entry in entries if entry["value"] is None]
        signatures = {(value_kind(entry["value"]), entry["unit"]) for entry in known}
        direction = preferences["attribute_preferences"].get(name)
        winners = []
        if len(known) < 2:
            status = "insufficient_data"
        elif len(signatures) > 1:
            status = "noncomparable"
        else:
            status = "comparable"
            if direction and value_kind(known[0]["value"]) == "number":
                best = (max if direction == "higher" else min)(e["value"] for e in known)
                winners = [e["product_id"] for e in known if e["value"] == best]
        rendered = "; ".join(describe(entry) for entry in known) or "no known values"
        explanation = f"{name}: {rendered}."
        if status == "noncomparable":
            explanation += " Values use different types or units; no conversion or ordering is inferred."
        elif status == "insufficient_data":
            explanation += " Fewer than two known values; no relative conclusion."
        elif winners:
            explanation += f" By the requested {direction} preference, best among known values: {', '.join(winners)}."
        else:
            explanation += " No directional numeric preference is applicable; no best product is inferred."
        if missing:
            explanation += f" Missing for {', '.join(missing)}; these products cannot be evaluated on this attribute."
        comparisons.append({"attribute": name, "status": status, "values": entries,
                            "missing_ids": missing, "preferred_ids": winners,
                            "explanation": explanation})
    tradeoffs = []
    for first, second in itertools.combinations(products, 2):
        advantages = {first["id"]: [], second["id"]: []}
        supporting_facts = []
        for item in comparisons:
            direction = preferences["attribute_preferences"].get(item["attribute"])
            pair = [entry for entry in item["values"]
                    if entry["product_id"] in advantages]
            if not direction or len(pair) != 2:
                continue
            left, right = pair
            if (left["value"] is None or right["value"] is None
                    or value_kind(left["value"]) != "number"
                    or value_kind(right["value"]) != "number"
                    or left["unit"] != right["unit"]
                    or left["value"] == right["value"]):
                continue
            left_wins = ((left["value"] > right["value"]) if direction == "higher"
                         else (left["value"] < right["value"]))
            winner = left if left_wins else right
            advantages[winner["product_id"]].append(item["attribute"])
            supporting_facts.extend(pair)
        if all(advantages.values()):
            tradeoffs.append({
                "product_ids": [first["id"], second["id"]],
                "advantages": advantages,
                "facts": supporting_facts,
                "explanation": (
                    f"{first['id']} is preferred on {', '.join(advantages[first['id']])}, "
                    f"while {second['id']} is preferred on {', '.join(advantages[second['id']])}, "
                    "under the supplied attribute directions. Neither wins every evaluated "
                    "attribute; missing or noncomparable values are not used."
                ),
            })
    return {"candidate_ids": [p["id"] for p in products], "attributes": comparisons,
            "facts": facts, "tradeoffs": tradeoffs,
            "scope": "Only selected candidates; preference winners are attribute-local, not overall recommendations."}


def grounded_callback(callback, selected_ids, facts):
    """Fail closed on unknown IDs, altered facts, duplicate claims or prose."""
    response = callback(copy.deepcopy({"candidate_ids": selected_ids, "facts": facts}))
    obj(response, ("claims",), ("claims",), "callback")
    claims = response["claims"]
    if not isinstance(claims, list):
        fail("callback.claims: expected array")
    allowed = {json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False)
               for item in facts}
    seen, explanations = set(), []
    for claim in claims:
        obj(claim, ("product_id", "attribute", "value", "unit", "provenance"),
            ("product_id", "attribute", "value", "unit", "provenance"), "callback.claim")
        if not isinstance(claim["product_id"], str) or claim["product_id"] not in selected_ids:
            fail("callback.claim: product ID was not selected")
        try:
            encoded = json.dumps(claim, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            fail("callback.claim: not a supplied JSON fact")
        if encoded not in allowed:
            fail("callback.claim: not an exact supplied fact")
        if encoded in seen:
            fail("callback.claim: duplicate fact")
        seen.add(encoded)
        explanations.append(f"{claim['attribute']}: {describe(claim)}.")
    return explanations


def build(payload, explanation_callback=None):
    normalized = normalize(payload)
    preferences = normalized["preferences"]
    recommendation = recommend(normalized["products"], preferences, normalized["candidate_limit"])
    comparison = compare(recommendation["selected"], preferences, normalized["comparison_attributes"])
    result = {"status": "ok" if recommendation["selected"] else "no_match",
              "preferences": preferences, "recommendation": recommendation,
              "comparison": comparison}
    if explanation_callback is not None:
        result["callback_explanations"] = grounded_callback(
            explanation_callback, recommendation["selected_ids"], comparison["facts"])
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON: duplicate object key {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            fail("usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object,
                                parse_constant=lambda value: fail(f"JSON: invalid constant {value}"))
        result = build(payload)
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
