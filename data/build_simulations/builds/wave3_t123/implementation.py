"""Synthetic manufacturing reference pipeline. No certification claims."""

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


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value.strip()


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite numeric")
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO timestamp") from exc
    require(parsed.tzinfo is not None, "timestamp requires timezone")
    return parsed


def rows(value, label, nonempty=False):
    require(isinstance(value, list), label + " must be a list")
    require(not nonempty or len(value) > 0, label + " must not be empty")
    require(all(isinstance(row, dict) for row in value), label + " entries must be objects")
    ids = [text(row.get("id"), label + " id") for row in value]
    require(len(ids) == len(set(ids)), label + " ids must be unique")
    return value


UNITS = {"temperature": "degC", "vibration": "mm/s", "pressure": "bar"}
STAGES = ["documents", "insights", "search", "onboard"]
ALIASES = {
    "temperature": {"temperature", "hot", "heat", "overheat", "cooling"},
    "vibration": {"vibration", "vibrating", "shake", "shaking", "bearing"},
    "quality": {"quality", "defect", "defects", "reject", "inspection"},
    "maintenance": {"maintenance", "service", "repair", "downtime"},
}


def terms(value):
    import re
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    for canonical, variants in ALIASES.items():
        if tokens & variants:
            tokens.add(canonical)
    return tokens


def validate_input(data):
    require(isinstance(data, dict), "input must be an object")
    require(data.get("schema_version") == "1.0", "unsupported schema_version")
    require(data.get("synthetic") is True, "input must be labeled synthetic")
    orders = rows(data.get("work_orders"), "work_orders", True)
    order_map = {order["id"]: order for order in orders}
    for order in orders:
        text(order.get("asset_id"), "asset_id")
        text(order.get("operation"), "operation")
        require(order.get("status") in ("planned", "running", "complete"), "invalid work order status")
        specs = order.get("specifications")
        require(isinstance(specs, dict) and bool(specs), "specifications required")
        for metric, spec in specs.items():
            require(metric in UNITS and isinstance(spec, dict), "unsupported metric")
            require(spec.get("unit") == UNITS[metric], "invalid specification unit")
            for key in ("min", "max", "safety_min", "safety_max"):
                number(spec.get(key), key)
            require(spec["safety_min"] <= spec["min"] < spec["max"] <= spec["safety_max"],
                    "invalid tolerance or safety bounds")
    for inspection in rows(data.get("inspections"), "inspections"):
        require(inspection.get("work_order_id") in order_map, "unknown inspection work order")
        order = order_map[inspection["work_order_id"]]
        require(inspection.get("metric") in order["specifications"], "unknown inspection metric")
        spec = order["specifications"][inspection["metric"]]
        require(inspection.get("unit") == spec["unit"], "inspection unit mismatch")
        number(inspection.get("value"), "inspection value")
        text(inspection.get("inspector"), "inspector")
        timestamp(inspection.get("timestamp"))
    for feedback in rows(data.get("feedback"), "feedback"):
        require(feedback.get("work_order_id") in order_map, "unknown feedback work order")
        text(feedback.get("text"), "feedback text")
        require(feedback.get("source") in ("customer", "maintenance_log", "operator"), "invalid feedback source")
    for product in rows(data.get("products"), "products"):
        text(product.get("name"), "product name")
        text(product.get("description"), "product description")
        require(isinstance(product.get("tags"), list) and all(isinstance(t, str) and t.strip()
                for t in product["tags"]), "product tags must be text list")
        require(isinstance(product.get("compatible_assets"), list) and all(isinstance(t, str) and t.strip()
                for t in product["compatible_assets"]), "compatible_assets must be text list")
    customer = data.get("customer")
    require(isinstance(customer, dict), "customer must be object")
    text(customer.get("id"), "customer id")
    text(customer.get("name"), "customer name")
    text(customer.get("query"), "customer query")
    require(customer.get("experience") in ("new", "experienced"), "invalid customer experience")
    require(customer.get("work_order_id") in order_map, "unknown customer work order")
    text(data.get("sensor_csv"), "sensor_csv")
    return data


def parse_readings(data):
    orders = {o["id"]: o for o in data["work_orders"]}
    stream = csv.DictReader(io.StringIO(data["sensor_csv"]))
    columns = ["timestamp", "asset_id", "work_order_id", "metric", "value", "unit"]
    require(stream.fieldnames == columns, "sensor CSV header mismatch")
    result, seen = [], set()
    for line, row in enumerate(stream, 2):
        require(None not in row and all(row.get(k) for k in columns), "malformed sensor row")
        moment = timestamp(row["timestamp"])
        require(row["work_order_id"] in orders, "unknown sensor work order")
        order = orders[row["work_order_id"]]
        require(row["asset_id"] == order["asset_id"], "sensor asset mismatch")
        require(row["metric"] in order["specifications"], "unknown sensor metric")
        require(row["unit"] == order["specifications"][row["metric"]]["unit"], "sensor unit mismatch")
        try:
            value = float(row["value"])
        except ValueError as exc:
            raise ValidationError("invalid sensor value") from exc
        number(value, "sensor value")
        key = (moment, row["work_order_id"], row["metric"])
        require(key not in seen, "duplicate sensor timestamp/metric/work order")
        seen.add(key)
        result.append(dict(row, id="reading-" + str(line - 1), value=value))
    return sorted(result, key=lambda r: (timestamp(r["timestamp"]), r["id"]))


def validate_state(state, expected_stage):
    """Shared boundary validator: revalidate source and derived trace references."""
    require(isinstance(state, dict), "state must be object")
    require(state.get("status") == "ok" and state.get("schema_version") == "1.0", "invalid state envelope")
    require(state.get("stage") == expected_stage, "unexpected pipeline stage")
    require(expected_stage in STAGES, "unknown stage")
    data = validate_input(state.get("source"))
    expected_readings = parse_readings(data)
    require(state.get("readings") == expected_readings, "reading provenance mismatch")
    orders = {o["id"]: o for o in data["work_orders"]}
    expected_decisions, expected_alerts = quality_records(data, expected_readings, orders)
    require(state.get("quality_decisions") == expected_decisions, "quality decision traceability mismatch")
    require(state.get("alerts") == expected_alerts, "safety escalation mismatch")
    level = STAGES.index(expected_stage)
    if level >= 1:
        require(state.get("themes") == derive_themes(state), "theme evidence mismatch")
    if level >= 2:
        require(state.get("search_results") == rank_products(state), "search handoff mismatch")
    if level >= 3:
        require(state.get("onboarding") == plan_onboarding(state), "onboarding handoff mismatch")
    return state


def quality_records(data, readings, orders):
    decisions, alerts = [], []
    for inspection in data["inspections"]:
        spec = orders[inspection["work_order_id"]]["specifications"][inspection["metric"]]
        decisions.append({
            "id": "decision-" + inspection["id"],
            "inspection_id": inspection["id"],
            "work_order_id": inspection["work_order_id"],
            "asset_id": orders[inspection["work_order_id"]]["asset_id"],
            "inspector": inspection["inspector"],
            "timestamp": inspection["timestamp"],
            "metric": inspection["metric"], "value": inspection["value"], "unit": spec["unit"],
            "tolerance": {"min": spec["min"], "max": spec["max"]},
            "outcome": "pass" if spec["min"] <= inspection["value"] <= spec["max"] else "fail",
            "rule": "inclusive-tolerance-v1",
        })
    for kind, records in (("telemetry", readings), ("inspection", data["inspections"])):
        for record in records:
            spec = orders[record["work_order_id"]]["specifications"][record["metric"]]
            if not spec["safety_min"] <= record["value"] <= spec["safety_max"]:
                alerts.append({
                    "id": "alert-" + kind + "-" + record["id"],
                    "source_type": kind, "source_id": record["id"],
                    "work_order_id": record["work_order_id"], "metric": record["metric"],
                    "severity": "safety-critical", "escalate_to": "human_safety_supervisor",
                    "status": "pending_human_review",
                })
    return decisions, alerts


def documents(data):
    data = copy.deepcopy(validate_input(data))
    readings = parse_readings(data)
    decisions, alerts = quality_records(data, readings, {o["id"]: o for o in data["work_orders"]})
    state = {"schema_version": "1.0", "status": "ok", "stage": "documents",
             "synthetic": True, "source": data, "readings": readings,
             "quality_decisions": decisions, "alerts": alerts}
    return validate_state(state, "documents")


def derive_themes(state):
    evidence = {}

    def add(theme, reference, work_order):
        bucket = evidence.setdefault((work_order, theme), set())
        bucket.add(reference)

    for feedback in state["source"]["feedback"]:
        found = terms(feedback["text"]) & set(ALIASES)
        for theme in found or {"general"}:
            add(theme, "feedback:" + feedback["id"], feedback["work_order_id"])
    for decision in state["quality_decisions"]:
        if decision["outcome"] == "fail":
            add("quality", decision["id"], decision["work_order_id"])
    orders = {o["id"]: o for o in state["source"]["work_orders"]}
    for reading in state["readings"]:
        spec = orders[reading["work_order_id"]]["specifications"][reading["metric"]]
        if not spec["min"] <= reading["value"] <= spec["max"]:
            add(reading["metric"], reading["id"], reading["work_order_id"])
    for alert in state["alerts"]:
        add(alert["metric"], alert["id"], alert["work_order_id"])
    return [{"id": "theme-" + str(index), "work_order_id": work_order, "theme": theme,
             "evidence": sorted(refs), "count": len(refs),
             "action": "Review " + theme + " evidence with the responsible operator"}
            for index, ((work_order, theme), refs) in enumerate(sorted(evidence.items()), 1)]


def insights(previous):
    validate_state(previous, "documents")
    state = copy.deepcopy(previous)
    state["themes"] = derive_themes(state)
    state["stage"] = "insights"
    return validate_state(state, "insights")


def rank_products(state):
    customer = state["source"]["customer"]
    order = next(o for o in state["source"]["work_orders"] if o["id"] == customer["work_order_id"])
    themes = [t for t in state["themes"] if t["work_order_id"] == order["id"]]
    query = terms(customer["query"])
    results = []
    for product in state["source"]["products"]:
        if order["asset_id"] not in product["compatible_assets"]:
            continue
        indexed = terms(" ".join([product["name"], product["description"]] + product["tags"]))
        query_matches = sorted(query & indexed)
        supporting = [t for t in themes if t["theme"] in indexed]
        score = 3 * len(query_matches) + sum(min(t["count"], 3) for t in supporting)
        if score:
            results.append({"product_id": product["id"], "name": product["name"], "score": score,
                            "query_matches": query_matches, "theme_ids": [t["id"] for t in supporting],
                            "work_order_id": order["id"]})
    return sorted(results, key=lambda r: (-r["score"], r["product_id"]))[:5]


def search(previous):
    validate_state(previous, "insights")
    state = copy.deepcopy(previous)
    state["search_results"] = rank_products(state)
    state["stage"] = "search"
    return validate_state(state, "search")


def plan_onboarding(state):
    customer = state["source"]["customer"]
    relevant_alerts = [a["id"] for a in state["alerts"]
                       if a["work_order_id"] == customer["work_order_id"]]
    decisions = [d["id"] for d in state["quality_decisions"]
                 if d["work_order_id"] == customer["work_order_id"] and d["outcome"] == "fail"]
    steps = []
    if relevant_alerts:
        steps.append({"action": "Contact human safety supervisor before machine operation",
                      "owner": "human_safety_supervisor", "references": relevant_alerts})
    if decisions:
        steps.append({"action": "Obtain human quality disposition for failed inspections",
                      "owner": "quality_reviewer", "references": decisions})
    if customer["experience"] == "new":
        steps.append({"action": "Complete operator orientation and unit/tolerance training",
                      "owner": customer["id"], "references": []})
    if state["search_results"]:
        product = state["search_results"][0]
        steps.append({"action": "Review recommended product with an application specialist",
                      "owner": customer["id"], "references": [product["product_id"]]})
    else:
        steps.append({"action": "Ask an application specialist to clarify requirements",
                      "owner": customer["id"], "references": []})
    return {"customer_id": customer["id"], "customer_name": customer["name"],
            "work_order_id": customer["work_order_id"],
            "operational_status": "blocked_pending_human_review" if relevant_alerts or decisions else "orientation_only",
            "recommended_product_id": state["search_results"][0]["product_id"] if state["search_results"] else None,
            "next_step": steps[0], "steps": steps,
            "notice": "Demonstrative traceability only; not ISO 9001 certification or authorization to operate."}


def onboard(previous):
    validate_state(previous, "search")
    state = copy.deepcopy(previous)
    state["onboarding"] = plan_onboarding(state)
    state["stage"] = "onboard"
    return validate_state(state, "onboard")


def run(data):
    return onboard(search(insights(documents(data))))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream)
        result = run(data)
    except (OSError, ValueError, TypeError, KeyError, csv.Error, OverflowError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
