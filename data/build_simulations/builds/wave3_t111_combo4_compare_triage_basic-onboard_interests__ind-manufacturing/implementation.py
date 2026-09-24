"""Synthetic manufacturing reference pipeline; no certification or live services."""
import copy
import csv
import io
import json
import math
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, name):
    require(isinstance(value, dict), name + " must be an object")
    return value


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def number(value, name, low=0, high=1e12):
    require(type(value) in (int, float) and math.isfinite(value)
            and low <= value <= high, name + " is out of range")
    return float(value)


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid timestamp") from exc
    require(result.utcoffset() is not None, "timestamp requires timezone")
    return result


def strings(value, name):
    require(isinstance(value, list), name + " must be a list")
    for item in value:
        text(item, name + " entry")
    require(len(value) == len(set(value)), name + " must be unique")
    return value


METRICS = {
    "price_usd": ("USD", "cost_usd", 1, 1e7),
    "lead_hours": ("h", "lead_days", 24, 8760),
    "vibration_reduction_pct": ("%", "reduction_fraction", 100, 100),
}
STAGES = ("compare", "triage", "onboard", "interests")
PRIORITIES = {"low": 0, "normal": 1, "high": 2, "critical": 3}


def validate_input(raw):
    """The sole input boundary: normalize attributes, units and shared entities."""
    data = copy.deepcopy(obj(raw, "input"))
    require(data.get("schema_version") == 1, "unsupported schema_version")
    require(data.get("synthetic") is True, "fixtures must be labeled synthetic")
    wo = obj(data.get("work_order"), "work_order")
    for key in ("id", "asset_id", "operation"):
        text(wo.get(key), "work_order." + key)
    require(type(wo.get("quantity")) is int and wo["quantity"] > 0, "invalid quantity")
    customer = obj(data.get("customer"), "customer")
    text(customer.get("id"), "customer.id")
    text(customer.get("name"), "customer.name")
    require(customer.get("experience") in ("new", "experienced"), "invalid experience")
    strings(customer.get("completed_steps"), "completed_steps")
    strings(customer.get("interests"), "interests")
    prefs = obj(data.get("preferences"), "preferences")
    weights = obj(prefs.get("weights"), "weights")
    require(set(weights) == set(METRICS), "weights must cover all comparison metrics")
    for key, value in weights.items():
        number(value, key + " weight", 0, 1)
    require(sum(weights.values()) > 0, "weights cannot all be zero")
    strings(prefs.get("excluded_product_ids"), "excluded_product_ids")
    strings(prefs.get("excluded_tags"), "excluded_tags")
    number(prefs.get("budget_usd"), "budget_usd", 0, 1e7)
    require(type(prefs.get("recommendation_limit")) is int
            and 1 <= prefs["recommendation_limit"] <= 20, "invalid recommendation_limit")
    specs = obj(data.get("sensor_specs"), "sensor_specs")
    require(set(specs) == {"temperature", "vibration"}, "missing sensor specifications")
    for metric, unit, maximum in (("temperature", "C", 150), ("vibration", "mm/s", 100)):
        spec = obj(specs[metric], metric)
        require(spec.get("unit") == unit, metric + " unit mismatch")
        nominal = number(spec.get("nominal"), "nominal", 0, maximum)
        tolerance = number(spec.get("tolerance"), "tolerance", 0, maximum)
        safety = number(spec.get("safety_upper"), "safety_upper", 0, maximum)
        require(nominal - tolerance >= 0 and nominal + tolerance < safety,
                "inconsistent sensor tolerance/safety bounds")
    source = text(data.get("telemetry_csv"), "telemetry_csv")
    fields = ["timestamp", "asset_id", "temperature", "temperature_unit",
              "vibration", "vibration_unit"]
    readings = []
    try:
        reader = csv.DictReader(io.StringIO(source), strict=True)
        require(reader.fieldnames == fields, "invalid sensor CSV headers")
        previous = None
        for row in reader:
            require(set(row) == set(fields) and None not in row.values(), "malformed sensor CSV row")
            when = timestamp(row["timestamp"])
            require(previous is None or when > previous, "sensor timestamps must increase")
            previous = when
            require(row["asset_id"] == wo["asset_id"], "sensor asset mismatch")
            reading = {"timestamp": row["timestamp"], "asset_id": row["asset_id"]}
            for metric, maximum in (("temperature", 150), ("vibration", 100)):
                require(row[metric + "_unit"] == specs[metric]["unit"], "sensor unit mismatch")
                reading[metric] = number(float(row[metric]), metric, 0, maximum)
            readings.append(reading)
    except (csv.Error, ValueError) as exc:
        raise ValidationError("invalid telemetry: " + str(exc)) from exc
    require(bool(readings), "at least one sensor reading required")
    inspections = data.get("quality_inspections")
    require(isinstance(inspections, list) and bool(inspections), "inspection records required")
    ids = set()
    for item in inspections:
        obj(item, "inspection")
        for key in ("id", "inspector", "reason", "characteristic"):
            text(item.get(key), "inspection." + key)
        require(item["id"] not in ids, "duplicate inspection ID")
        ids.add(item["id"])
        require(item.get("work_order_id") == wo["id"] and item.get("asset_id") == wo["asset_id"],
                "inspection traceability mismatch")
        timestamp(item.get("timestamp"))
        require(item.get("unit") == "mm", "inspection unit must be mm")
        nominal = number(item.get("nominal"), "inspection nominal", 0, 10000)
        tolerance = number(item.get("tolerance"), "inspection tolerance", 0, 1000)
        measured = number(item.get("measured"), "inspection measured", 0, 10000)
        require(nominal >= tolerance, "invalid inspection tolerance")
        require(item.get("decision") in ("accept", "reject", "review"), "invalid quality decision")
        require(item["decision"] != "accept" or abs(measured - nominal) <= tolerance + 1e-9,
                "accepted inspection outside tolerance")
    logs = data.get("maintenance_logs")
    require(isinstance(logs, list) and bool(logs), "invented maintenance logs required")
    for log in logs:
        obj(log, "maintenance log")
        require(log.get("asset_id") == wo["asset_id"], "maintenance asset mismatch")
        text(log.get("note"), "maintenance note")
        timestamp(log.get("timestamp"))
    products = data.get("products")
    require(isinstance(products, list) and bool(products), "products required")
    normalized = []
    ids = set()
    for product in products:
        obj(product, "product")
        pid = text(product.get("id"), "product.id")
        require(pid not in ids, "duplicate product ID")
        ids.add(pid)
        text(product.get("name"), "product.name")
        strings(product.get("tags"), "product tags")
        strings(product.get("compatible_assets"), "compatible_assets")
        attrs = obj(product.get("attributes"), "attributes")
        allowed = set(METRICS) | {v[1] for v in METRICS.values()}
        require(set(attrs) <= allowed, "unknown product attribute")
        values = {}
        for key, (unit, alias, factor, maximum) in METRICS.items():
            require((key in attrs) != (alias in attrs), "provide one form of " + key)
            chosen = key if key in attrs else alias
            entry = obj(attrs[chosen], chosen)
            expected = unit if chosen == key else {"cost_usd": "USD", "lead_days": "day",
                                                  "reduction_fraction": "fraction"}[alias]
            require(entry.get("unit") == expected, "product unit mismatch")
            value = number(entry.get("value"), chosen, 0, maximum if chosen == key else maximum / factor)
            values[key] = value if chosen == key else value * factor
        normalized.append({"id": pid, "name": product["name"], "tags": product["tags"],
                           "compatible_assets": product["compatible_assets"], "attributes": values})
    routing = obj(data.get("routing"), "routing")
    require(set(routing) == {"safety", "quality", "telemetry", "procurement"}, "routing categories required")
    for category, rule in routing.items():
        obj(rule, "routing rule")
        text(rule.get("owner"), "accountable owner")
        require(rule.get("priority") in PRIORITIES, "invalid priority")
        require(type(rule.get("human")) is bool, "human routing flag required")
    require(routing["safety"]["human"] and routing["safety"]["priority"] == "critical",
            "safety must route critical to a human")
    data["readings"] = readings
    data["normalized_products"] = normalized
    return data


def eligible(context, product):
    prefs = context["preferences"]
    return (product["id"] not in prefs["excluded_product_ids"]
            and not set(product["tags"]).intersection(prefs["excluded_tags"])
            and context["work_order"]["asset_id"] in product["compatible_assets"]
            and product["attributes"]["price_usd"] <= prefs["budget_usd"])


def validate_output(envelope, expected):
    """One shared boundary validator for all cumulative pipeline handoffs."""
    obj(envelope, "pipeline output")
    require(envelope.get("schema_version") == 1 and envelope.get("status") == "ok",
            "invalid output envelope")
    require(envelope.get("stage") == expected and expected in STAGES, "unexpected pipeline stage")
    context = validate_input(envelope.get("context"))
    require(context == envelope["context"], "context normalization mismatch")
    stages = obj(envelope.get("stages"), "stages")
    index = STAGES.index(expected)
    require(set(stages) == set(STAGES[:index + 1]), "missing or unexpected pipeline stage")
    # Recompute deterministic contracts rather than trusting mutable upstream claims.
    computed = {}
    for name in STAGES[:index + 1]:
        computed[name] = COMPUTE[name](context, computed)
        require(stages[name] == computed[name], "invalid " + name + " stage contract")
    return envelope


def compare_result(context, stages):
    products = context["normalized_products"]
    weights = context["preferences"]["weights"]
    viable = [p for p in products if eligible(context, p)]
    scores = {}
    for product in viable:
        total = 0
        for key in METRICS:
            values = [p["attributes"][key] for p in viable]
            low, high = min(values), max(values)
            value = product["attributes"][key]
            utility = 1 if low == high else (value - low) / (high - low)
            if key != "vibration_reduction_pct" and low != high:
                utility = 1 - utility
            total += utility * weights[key]
        scores[product["id"]] = round(total / sum(weights.values()), 8)
    ranked = sorted(viable, key=lambda p: (-scores[p["id"]], p["id"]))
    return {
        "work_order_id": context["work_order"]["id"],
        "columns": {key: unit[0] for key, unit in METRICS.items()},
        "side_by_side": [{"product_id": p["id"], **p["attributes"],
                         "eligible": eligible(context, p)} for p in products],
        "ranking": [{"product_id": p["id"], "score": scores[p["id"]]} for p in ranked],
        "selected_product_id": ranked[0]["id"] if ranked else None,
        "selection_status": "selected" if ranked else "no_eligible_product",
    }


def triage_result(context, stages):
    comparison = stages["compare"]
    evidence = []
    categories = {"procurement"}
    for reading in context["readings"]:
        for metric, spec in context["sensor_specs"].items():
            value = reading[metric]
            if value >= spec["safety_upper"]:
                categories.add("safety")
                evidence.append({"kind": "safety", "metric": metric, "value": value,
                                 "timestamp": reading["timestamp"], "unit": spec["unit"]})
            elif abs(value - spec["nominal"]) > spec["tolerance"]:
                categories.add("telemetry")
                evidence.append({"kind": "tolerance", "metric": metric, "value": value,
                                 "timestamp": reading["timestamp"], "unit": spec["unit"]})
    for inspection in context["quality_inspections"]:
        if inspection["decision"] != "accept":
            categories.add("quality")
    # Safety overrides all configurable category priorities.
    category = "safety" if "safety" in categories else sorted(
        categories, key=lambda c: (-PRIORITIES[context["routing"][c]["priority"]],
                                   ("quality", "telemetry", "procurement").index(c)))[0]
    rule = context["routing"][category]
    return {"ticket_id": "T-" + context["work_order"]["id"], "source_stage": "compare",
            "selected_product_id": comparison["selected_product_id"],
            "category": category, "detected_categories": sorted(categories),
            "priority": rule["priority"], "accountable_owner": rule["owner"],
            "requires_human": rule["human"], "safety_hold": "safety" in categories,
            "evidence": evidence,
            "quality_decision_trace": copy.deepcopy(context["quality_inspections"]),
            "work_order_id": comparison["work_order_id"], "asset_id": context["work_order"]["asset_id"]}


def onboard_result(context, stages):
    ticket = stages["triage"]
    customer = context["customer"]
    blocked = ticket["safety_hold"] or "quality" in ticket["detected_categories"]
    if ticket["safety_hold"]:
        step, action = "await_safety_review", "Do not start work; contact the accountable human safety owner."
    elif "quality" in ticket["detected_categories"]:
        step, action = "await_quality_review", "Request human review of the traced quality decision before work."
    else:
        options = [("confirm_profile", "Confirm your customer profile and preferences."),
                   ("review_work_order", "Review the work order and its unit/tolerance requirements."),
                   ("review_selection", "Review the selected product with your assigned owner."),
                   ("ready", "Your setup is complete; consult the work-order owner for the next authorized task.")]
        if customer["experience"] == "experienced":
            options = options[1:]
        step, action = next(((k, v) for k, v in options if k not in customer["completed_steps"]),
                            options[-1])
        if ticket["selected_product_id"] is None:
            step, action = "resolve_product_preferences", "Review exclusions and budget with your owner; no eligible product exists."
    return {"source_stage": "triage", "customer_id": customer["id"], "ticket_id": ticket["ticket_id"],
            "selected_product_id": ticket["selected_product_id"], "accountable_owner": ticket["accountable_owner"],
            "next_step": step, "instruction": customer["name"] + ": " + action,
            "blocked": blocked, "work_authorized": False,
            "discovery_allowed": not blocked, "interests": copy.deepcopy(customer["interests"])}


def interests_result(context, stages):
    onboarding = stages["onboard"]
    comparison = stages["compare"]
    if not onboarding["discovery_allowed"]:
        return {"source_stage": "onboard", "next_step": onboarding["next_step"],
                "status": "withheld_pending_review", "recommendations": []}
    scores = {item["product_id"]: item["score"] for item in comparison["ranking"]}
    rows = []
    for product in context["normalized_products"]:
        if eligible(context, product):
            matches = sorted(set(product["tags"]).intersection(onboarding["interests"]))
            affinity = len(matches) / max(1, len(onboarding["interests"]))
            score = round(0.7 * scores[product["id"]] + 0.3 * affinity, 8)
            rows.append({"product_id": product["id"], "score": score,
                         "explanation": {"matched_interests": matches,
                                         "comparison_score": scores[product["id"]],
                                         "price_usd": product["attributes"]["price_usd"],
                                         "compatible_asset": context["work_order"]["asset_id"],
                                         "onboarding_next_step": onboarding["next_step"]}})
    rows.sort(key=lambda row: (-row["score"], row["product_id"]))
    return {"source_stage": "onboard", "next_step": onboarding["next_step"],
            "status": "ranked" if rows else "no_eligible_product",
            "recommendations": rows[:context["preferences"]["recommendation_limit"]]}


COMPUTE = dict(zip(STAGES, (compare_result, triage_result, onboard_result, interests_result)))


def advance(previous, stage):
    require(stage in STAGES, "unknown stage")
    index = STAGES.index(stage)
    if index == 0:
        context, stages = validate_input(previous), {}
    else:
        validate_output(previous, STAGES[index - 1])
        context, stages = copy.deepcopy(previous["context"]), copy.deepcopy(previous["stages"])
    stages[stage] = COMPUTE[stage](context, stages)
    return validate_output({"schema_version": 1, "status": "ok", "stage": stage,
                            "context": context, "stages": stages}, stage)


def run_pipeline(data):
    result = data
    for stage in STAGES:
        result = advance(result, stage)
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream)
        output = run_pipeline(data)
        print(json.dumps(output, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
