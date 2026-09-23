"""Synthetic reference pipeline: normalized comparison -> accountable ticket triage."""
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


ATTRIBUTES = {
    "price_usd": ("min", {"usd": 1, "$": 1}),
    "ram_gb": ("max", {"gb": 1, "mb": 1 / 1024}),
    "storage_gb": ("max", {"gb": 1, "tb": 1024}),
    "battery_hours": ("max", {"h": 1, "hours": 1, "min": 1 / 60}),
    "weight_kg": ("min", {"kg": 1, "g": 0.001}),
}
PRIORITIES = ("normal", "high", "urgent")


def require(condition, path, message):
    if not condition:
        raise ValidationError(f"{path}: {message}")


def obj(value, path, required, optional=()):
    require(isinstance(value, dict), path, "must be an object")
    require(set(required) <= value.keys(), path, "missing required fields")
    require(value.keys() <= set(required) | set(optional), path, "unknown fields")
    return value


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path, "must be nonempty text")
    return value.strip()


def number(value, path, positive=False):
    require(type(value) in (int, float), path, "must be a finite number")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    require(valid, path, "must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def strings(value, path):
    require(isinstance(value, list), path, "must be an array")
    result = [text(v, path).casefold() for v in value]
    require(len(set(result)) == len(result), path, "duplicate values")
    return result


def normalize(key, value, path):
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z$]+)\s*", value)
        require(match is not None, path, "expected a nonnegative quantity and unit")
        amount, unit = match.groups()
        units = ATTRIBUTES[key][1]
        require(unit.casefold() in units, path, "unsupported unit")
        value = float(amount) * units[unit.casefold()]
    return number(value, path)


def validate_input(data):
    obj(data, "input", ("schema_version", "fixture_label", "products", "preferences", "tickets", "routing"))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "schema_version", "expected 1")
    text(data["fixture_label"], "fixture_label")
    products = data["products"]
    require(isinstance(products, list) and len(products) > 0, "products", "must be a nonempty array")
    ids = set()
    for p in products:
        obj(p, "product", ("id", "name", "attributes", "features"))
        pid = text(p["id"], "product.id")
        require(pid == p["id"] and pid not in ids, "product.id", "must be unique and trimmed")
        ids.add(pid)
        text(p["name"], "product.name")
        obj(p["attributes"], "attributes", ("price_usd",), ATTRIBUTES)
        for key, value in p["attributes"].items():
            normalize(key, value, f"{pid}.{key}")
        strings(p["features"], "product.features")
    pref = obj(data["preferences"], "preferences", ("weights", "required_features"), ("max_price_usd",))
    weights = obj(pref["weights"], "weights", (), ATTRIBUTES)
    require(bool(weights), "weights", "must not be empty")
    for key, weight in weights.items():
        number(weight, f"weights.{key}", positive=True)
    number(sum(weights.values()), "weights.total", positive=True)
    strings(pref["required_features"], "required_features")
    if "max_price_usd" in pref:
        number(pref["max_price_usd"], "max_price_usd")
    require(isinstance(data["tickets"], list), "tickets", "must be an array")
    ticket_ids = set()
    for ticket in data["tickets"]:
        obj(ticket, "ticket", ("id", "text"), ("product_id",))
        tid = text(ticket["id"], "ticket.id")
        require(tid == ticket["id"] and tid not in ticket_ids, "ticket.id", "must be unique and trimmed")
        ticket_ids.add(tid)
        text(ticket["text"], "ticket.text")
        if "product_id" in ticket:
            require(isinstance(ticket["product_id"], str) and ticket["product_id"] in ids,
                    "ticket.product_id", "unknown product")
    routing = obj(data["routing"], "routing", ("rules", "fallback", "priority_keywords"))
    require(isinstance(routing["rules"], list), "routing.rules", "must be an array")
    categories = set()
    for rule in routing["rules"]:
        obj(rule, "rule", ("category", "keywords", "signals", "priority", "team", "owner"))
        category = text(rule["category"], "category")
        require(category not in categories, "category", "duplicate category")
        categories.add(category)
        keywords = strings(rule["keywords"], "keywords")
        signals = strings(rule["signals"], "signals")
        require(set(signals) <= {"over_budget", "missing_required_features", "missing_attributes"},
                "signals", "unknown comparison signal")
        require(bool(keywords or signals), "rule", "at least one matcher required")
        validate_assignment(rule)
    fallback = obj(routing["fallback"], "fallback", ("category", "priority", "team", "owner"))
    text(fallback["category"], "fallback.category")
    validate_assignment(fallback)
    obj(routing["priority_keywords"], "priority_keywords", ("high", "urgent"))
    for level in ("high", "urgent"):
        strings(routing["priority_keywords"][level], f"priority_keywords.{level}")
    return data


def validate_assignment(value):
    require(value["priority"] in PRIORITIES, "priority", "unknown priority")
    text(value["team"], "team")
    text(value["owner"], "owner")


def compare(data):
    pref = data["preferences"]
    products = []
    for p in data["products"]:
        attributes = {key: normalize(key, value, f"{p['id']}.{key}")
                      for key, value in p["attributes"].items()}
        features = strings(p["features"], "features")
        missing_features = sorted(set(strings(pref["required_features"], "required_features")) - set(features))
        missing_attributes = sorted(set(pref["weights"]) - attributes.keys())
        signals = []
        if attributes["price_usd"] > pref.get("max_price_usd", float("inf")):
            signals.append("over_budget")
        if missing_features:
            signals.append("missing_required_features")
        if missing_attributes:
            signals.append("missing_attributes")
        products.append({
            "id": p["id"], "name": p["name"], "attributes": attributes,
            "features": features, "signals": signals, "missing_features": missing_features,
            "missing_attributes": missing_attributes, "eligible": not signals,
        })
    total = sum(pref["weights"].values())
    for p in products:
        contributions = {}
        for key, weight in pref["weights"].items():
            values = [q["attributes"][key] for q in products if key in q["attributes"]]
            utility = 0.0
            if key in p["attributes"]:
                lo, hi = min(values), max(values)
                utility = 1.0 if hi == lo else (p["attributes"][key] - lo) / (hi - lo)
                if hi != lo and ATTRIBUTES[key][0] == "min":
                    utility = 1.0 - utility
            contributions[key] = utility * (weight / total)
        p["score"] = sum(contributions.values())
        p["score_contributions"] = contributions
    ranking = sorted(products, key=lambda p: (not p["eligible"], -p["score"], p["id"]))
    for rank, p in enumerate(ranking, 1):
        p["rank"] = rank
    return {
        "products": products,
        "ranking": [p["id"] for p in ranking],
        "recommended_product_id": next((p["id"] for p in ranking if p["eligible"]), None),
        "side_by_side": [
            {"attribute": key, "direction": ATTRIBUTES[key][0],
             "values": {p["id"]: p["attributes"].get(key) for p in products}}
            for key in ATTRIBUTES
        ],
    }


def validate_comparison(comparison, data):
    # Recompute the deterministic contract to reject any altered handoff fields.
    require(comparison == compare(data), "comparison", "invalid comparison handoff")
    return comparison


def matches(keyword, content):
    return re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", content, re.IGNORECASE) is not None


def triage(data, comparison):
    validate_comparison(comparison, data)
    indexed = {p["id"]: p for p in comparison["products"]}
    results = []
    for ticket in data["tickets"]:
        product_id = ticket.get("product_id", comparison["recommended_product_id"])
        product = indexed.get(product_id)
        signals = product["signals"] if product else []
        rule = data["routing"]["fallback"]
        reasons = ["fallback"]
        for candidate in data["routing"]["rules"]:
            keywords = [k for k in strings(candidate["keywords"], "keywords") if matches(k, ticket["text"])]
            propagated = sorted(set(strings(candidate["signals"], "signals")) & set(signals))
            if keywords or propagated:
                rule = candidate
                reasons = ["keyword:" + k for k in keywords] + ["comparison:" + s for s in propagated]
                break
        priority = rule["priority"]
        for level in ("high", "urgent"):
            hits = [k for k in strings(data["routing"]["priority_keywords"][level], "priority_keywords")
                    if matches(k, ticket["text"])]
            if hits:
                if PRIORITIES.index(level) > PRIORITIES.index(priority):
                    priority = level
                reasons.extend("priority_keyword:" + k for k in hits)
        results.append({
            "ticket_id": ticket["id"], "category": rule["category"], "priority": priority,
            "team": rule["team"], "owner": rule["owner"], "reasons": reasons,
            "comparison_context": {
                "product_id": product_id,
                "recommended_product_id": comparison["recommended_product_id"],
                "rank": product["rank"] if product else None,
                "score": product["score"] if product else None,
                "signals": list(signals),
                "missing_features": list(product["missing_features"]) if product else [],
            },
        })
    return results


def run_pipeline(data):
    validate_input(data)
    comparison = validate_comparison(compare(data), data)
    tickets = triage(data, comparison)
    for ticket in tickets:
        validate_assignment(ticket)
        require(ticket["comparison_context"]["product_id"] in
                {p["id"] for p in comparison["products"]} | {None},
                "triage.context", "unknown product")
    return {"status": "ok", "schema_version": 1, "fixture_label": data["fixture_label"],
            "comparison": comparison, "triage": tickets}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, key, "duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "arguments", "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=unique_object)
        result = run_pipeline(data)
        output = json.dumps(result, allow_nan=False, sort_keys=True)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
