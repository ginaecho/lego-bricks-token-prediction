"""Synthetic, deterministic ticket -> discovery -> comparison reference CLI.

Run: python -B implementation.py example_input.json
The strict schema is demonstrated by example_input.json. Only synthetic data
is accepted. No providers, dependencies, network calls, or persistence are used.
"""

import copy
import json
import math
import re
import sys


SCHEMA_VERSION = 1
PRIORITIES = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
UNITS = {
    "price": {"USD": 1.0},
    "weight": {"g": 1.0, "kg": 1000.0},
    "battery_life": {"h": 1.0, "min": 1.0 / 60.0},
}
CANONICAL_UNITS = {"price": "USD", "weight": "g", "battery_life": "h"}


class ValidationError(ValueError):
    """Invalid input or an invalid cross-stage handoff."""


def fail(path, message):
    raise ValidationError(f"{path}: {message}")


def object_fields(value, keys, path):
    if not isinstance(value, dict):
        fail(path, "expected an object")
    if set(value) != set(keys):
        fail(path, "expected exactly these fields: " + ", ".join(sorted(keys)))


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        fail(path, "expected a nonempty string")
    if value != value.strip():
        fail(path, "leading/trailing whitespace is not allowed")
    return value


def number(value, path, positive=False):
    if type(value) not in (int, float):
        fail(path, "expected a finite number, not a boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (value <= 0 if positive else value < 0):
        fail(path, "expected a finite " + ("positive" if positive else "nonnegative") + " number")
    return value


def array(value, path, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        fail(path, "expected " + ("a nonempty" if nonempty else "an") + " array")
    return value


def unique_strings(value, path, casefold=False, nonempty=False):
    array(value, path, nonempty)
    seen = set()
    for item in value:
        text(item, path)
        key = item.casefold() if casefold else item
        if key in seen:
            fail(path, "duplicate value: " + item)
        seen.add(key)
    return seen


def validate_input(data):
    """Shared strict validation; returns a copy and never mutates the caller."""
    object_fields(data, {
        "schema_version", "data_label", "ticket", "triage_policy",
        "preferences", "catalog",
    }, "input")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        fail("schema_version", "only integer version 1 is supported")
    if data["data_label"] != "synthetic":
        fail("data_label", "must be synthetic")
    ticket = data["ticket"]
    object_fields(ticket, {"id", "subject", "body"}, "ticket")
    for key in ticket:
        text(ticket[key], "ticket." + key)
    policy = data["triage_policy"]
    object_fields(policy, {
        "categories", "fallback_category", "priority_rules", "default_priority",
    }, "triage_policy")
    category_ids = set()
    for category in array(policy["categories"], "categories", nonempty=True):
        object_fields(category, {"id", "keywords", "route"}, "category")
        cid = text(category["id"], "category.id")
        if cid == "*" or cid in category_ids:
            fail("category.id", "must be unique and not '*'")
        category_ids.add(cid)
        unique_strings(category["keywords"], "category.keywords", casefold=True)
        object_fields(category["route"], {"team", "owner"}, "category.route")
        for key in category["route"]:
            text(category["route"][key], "category.route." + key)
    if text(policy["fallback_category"], "fallback_category") not in category_ids:
        fail("fallback_category", "unknown category")
    if text(policy["default_priority"], "default_priority") not in PRIORITIES:
        fail("default_priority", "unknown priority")
    for rule in array(policy["priority_rules"], "priority_rules"):
        object_fields(rule, {"keywords", "priority"}, "priority_rule")
        unique_strings(rule["keywords"], "priority_rule.keywords", casefold=True, nonempty=True)
        if text(rule["priority"], "priority_rule.priority") not in PRIORITIES:
            fail("priority_rule.priority", "unknown priority")
    prefs = data["preferences"]
    object_fields(prefs, {"interests", "exclusions", "attributes", "limit"}, "preferences")
    if type(prefs["limit"]) is not int or not 1 <= prefs["limit"] <= 50:
        fail("preferences.limit", "expected an integer from 1 to 50")
    tags = []
    for interest in array(prefs["interests"], "interests"):
        object_fields(interest, {"tag", "weight"}, "interest")
        tags.append(text(interest["tag"], "interest.tag"))
        number(interest["weight"], "interest.weight", positive=True)
    unique_strings(tags, "interests.tag", casefold=True)
    exclusions = prefs["exclusions"]
    object_fields(exclusions, {"product_ids", "tags"}, "exclusions")
    unique_strings(exclusions["product_ids"], "exclusions.product_ids")
    unique_strings(exclusions["tags"], "exclusions.tags", casefold=True)
    attributes = []
    for attr in array(prefs["attributes"], "preferences.attributes"):
        object_fields(attr, {"name", "direction", "weight"}, "preference_attribute")
        name = text(attr["name"], "preference_attribute.name")
        if name not in UNITS:
            fail("preference_attribute.name", "unsupported attribute")
        if attr["direction"] not in ("min", "max"):
            fail("preference_attribute.direction", "expected min or max")
        number(attr["weight"], "preference_attribute.weight", positive=True)
        attributes.append(name)
    unique_strings(attributes, "preferences.attributes.name")
    product_ids = []
    for product in array(data["catalog"], "catalog"):
        object_fields(product, {
            "id", "name", "tags", "support_categories", "attributes",
        }, "product")
        product_ids.append(text(product["id"], "product.id"))
        text(product["name"], "product.name")
        unique_strings(product["tags"], "product.tags", casefold=True)
        supported = unique_strings(product["support_categories"], "product.support_categories",
                                   nonempty=True)
        if not supported <= category_ids | {"*"}:
            fail("product.support_categories", "unknown category")
        if not isinstance(product["attributes"], dict):
            fail("product.attributes", "expected an object")
        for name, measurement in product["attributes"].items():
            if name not in UNITS:
                fail("product.attributes", "unsupported attribute: " + name)
            object_fields(measurement, {"value", "unit"}, "measurement")
            value = number(measurement["value"], "measurement.value")
            unit = text(measurement["unit"], "measurement.unit")
            if unit not in UNITS[name]:
                fail("measurement.unit", "unsupported unit for " + name)
            try:
                normalized = value * UNITS[name][unit]
            except OverflowError:
                normalized = float("inf")
            if not math.isfinite(normalized):
                fail("measurement.value", "normalized value is not finite")
    unique_strings(product_ids, "catalog.id")
    return copy.deepcopy(data)


def matches(keyword, content):
    return re.search(r"(?<!\w)" + re.escape(keyword.casefold()) + r"(?!\w)", content) is not None


def weighted_mean(pairs):
    """Scale weights before summing to avoid overflow for valid large weights."""
    if not pairs:
        return 0.0
    scale = max(weight for _, weight in pairs)
    denominator = math.fsum(weight / scale for _, weight in pairs)
    return math.fsum(score * (weight / scale) for score, weight in pairs) / denominator


def _triage_result(data):
    policy = data["triage_policy"]
    content = (data["ticket"]["subject"] + "\n" + data["ticket"]["body"]).casefold()
    candidates = []
    for index, category in enumerate(policy["categories"]):
        evidence = [keyword for keyword in category["keywords"] if matches(keyword, content)]
        candidates.append((len(evidence), -index, category, evidence))
    _, _, chosen, evidence = max(candidates, key=lambda item: item[:2])
    if not evidence:
        chosen = next(c for c in policy["categories"] if c["id"] == policy["fallback_category"])
    priority_evidence = []
    priority = policy["default_priority"]
    for rule in policy["priority_rules"]:
        hits = [word for word in rule["keywords"] if matches(word, content)]
        if hits:
            priority_evidence.append({"priority": rule["priority"], "keywords": hits})
            if PRIORITIES[rule["priority"]] > PRIORITIES[priority]:
                priority = rule["priority"]
    return {
        "category": chosen["id"], "priority": priority,
        "route": copy.deepcopy(chosen["route"]),
        "category_evidence": evidence, "used_fallback": not bool(evidence),
        "priority_evidence": priority_evidence,
    }


def _interest_result(data, triage):
    prefs = data["preferences"]
    excluded_ids = set(prefs["exclusions"]["product_ids"])
    excluded_tags = {tag.casefold() for tag in prefs["exclusions"]["tags"]}
    excluded, ineligible, ranked = [], [], []
    for product in data["catalog"]:
        tags = {tag.casefold() for tag in product["tags"]}
        reasons = []
        if product["id"] in excluded_ids:
            reasons.append("explicit_product_id")
        reasons.extend("excluded_tag:" + tag for tag in sorted(tags & excluded_tags))
        if reasons:
            excluded.append({"product_id": product["id"], "reasons": reasons})
            continue
        if triage["category"] not in product["support_categories"] and "*" not in product["support_categories"]:
            ineligible.append(product["id"])
            continue
        hits = [item for item in prefs["interests"] if item["tag"].casefold() in tags]
        score = weighted_mean([
            (1.0 if item["tag"].casefold() in tags else 0.0, item["weight"])
            for item in prefs["interests"]
        ])
        ranked.append({
            "product_id": product["id"], "interest_score": score,
            "matched_interests": copy.deepcopy(hits),
            "explanation": (
                [f"Catalog tag '{item['tag']}' matches preference (weight {item['weight']})." for item in hits]
                or ["No declared interest tag matches; retained as an eligible fallback."]
            ),
        })
    ranked.sort(key=lambda item: (-item["interest_score"], item["product_id"]))
    return {
        "category": triage["category"], "priority": triage["priority"],
        "accountable_route": copy.deepcopy(triage["route"]),
        "recommendations": ranked[:prefs["limit"]],
        "excluded": sorted(excluded, key=lambda item: item["product_id"]),
        "category_ineligible": sorted(ineligible),
        "eligible_count": len(ranked),
    }


def _comparison_result(data, interests):
    products = {p["id"]: p for p in data["catalog"]}
    attrs = data["preferences"]["attributes"]
    rows = []
    for item in interests["recommendations"]:
        product = products[item["product_id"]]
        values = {}
        for name in UNITS:
            measurement = product["attributes"].get(name)
            values[name] = None if measurement is None else measurement["value"] * UNITS[name][measurement["unit"]]
        rows.append({
            "product_id": product["id"], "name": product["name"],
            "normalized_attributes": values, "interest_score": item["interest_score"],
        })
    for row in rows:
        scores, evidence = [], []
        for attr in attrs:
            name = attr["name"]
            value = row["normalized_attributes"][name]
            observed = [r["normalized_attributes"][name] for r in rows
                        if r["normalized_attributes"][name] is not None]
            if value is None:
                score = 0.0
                explanation = "Missing catalog value; utility is 0."
            else:
                low, high = min(observed), max(observed)
                score = 1.0 if low == high else (
                    (high - value) / (high - low) if attr["direction"] == "min"
                    else (value - low) / (high - low)
                )
                explanation = f"Catalog value {value} {CANONICAL_UNITS[name]}; prefer {attr['direction']}."
            scores.append((score, attr["weight"]))
            evidence.append({
                "attribute": name, "direction": attr["direction"], "weight": attr["weight"],
                "normalized_value": value, "utility": score, "explanation": explanation,
            })
        row["preference_score"] = weighted_mean(scores) if attrs else row["interest_score"]
        row["preference_evidence"] = evidence
    rows.sort(key=lambda row: (-row["preference_score"], -row["interest_score"], row["product_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {
        "category": interests["category"], "priority": interests["priority"],
        "accountable_route": copy.deepcopy(interests["accountable_route"]),
        "attribute_units": dict(CANONICAL_UNITS),
        "ranking_basis": "weighted_attribute_utility" if attrs else "interest_score",
        "rows": rows,
    }


def _same_json(actual, expected):
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False


def validate_state(data, state, stage):
    """One shared handoff validator, including source-grounded semantic checks.

    Recomputing deterministic stage facts prevents forged scores, explanations,
    routing, and excluded or non-shortlisted products crossing a stage boundary.
    """
    data = validate_input(data)
    stages = ("triage", "interests", "comparison")
    if stage not in stages:
        fail("stage", "unknown stage")
    completed = stages[:stages.index(stage) + 1]
    object_fields(state, {"schema_version", "data_label", "status", "ticket_id"} | set(completed), "handoff")
    expected = {
        "schema_version": SCHEMA_VERSION, "data_label": "synthetic",
        "status": "ok", "ticket_id": data["ticket"]["id"],
        "triage": _triage_result(data),
    }
    if "interests" in completed:
        expected["interests"] = _interest_result(data, expected["triage"])
    if "comparison" in completed:
        expected["comparison"] = _comparison_result(data, expected["interests"])
    if not _same_json(state, expected):
        fail("handoff", "stage output does not match validated source facts")
    return copy.deepcopy(state)


def triage_stage(data):
    data = validate_input(data)
    state = {
        "schema_version": SCHEMA_VERSION, "data_label": "synthetic",
        "status": "ok", "ticket_id": data["ticket"]["id"],
        "triage": _triage_result(data),
    }
    return validate_state(data, state, "triage")


def interests_stage(data, previous):
    data = validate_input(data)
    state = validate_state(data, previous, "triage")
    state["interests"] = _interest_result(data, state["triage"])
    return validate_state(data, state, "interests")


def comparison_stage(data, previous):
    data = validate_input(data)
    state = validate_state(data, previous, "interests")
    state["comparison"] = _comparison_result(data, state["interests"])
    return validate_state(data, state, "comparison")


def run_pipeline(data):
    data = validate_input(data)
    return comparison_stage(data, interests_stage(data, triage_stage(data)))


def reject_constant(value):
    fail("JSON", "non-finite constant is not allowed: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("JSON", "duplicate object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            fail("arguments", "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA_VERSION, "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
