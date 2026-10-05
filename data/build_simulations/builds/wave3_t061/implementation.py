"""Synthetic manufacturing workflow; demonstrative rules, not certification.

Usage: python -B implementation.py example_input.json
Python standard library only. No provider, network, storage, or actuator calls.
"""

import copy
import csv
import io
import json
import math
import sys
from datetime import datetime


SCHEMA_VERSION = 1
METRICS = {
    "temperature": ("C", -50, 300),
    "vibration": ("mm/s", 0, 100),
    "diameter": ("mm", 0.001, 1000),
}
CSV_COLUMNS = [
    "reading_id", "timestamp", "work_order_id", "asset_id", "metric", "value", "unit"
]
STAGES = ("documents", "onboarding", "insights")


class ValidationError(ValueError):
    """An input or handoff failed the shared schema."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_fields(value, fields, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(fields), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def number(value, path):
    require(
        isinstance(value, (float, int)) and not isinstance(value, bool)
        and math.isfinite(value), path + " must be a finite number"
    )


def timestamp(value, path):
    text(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
                path + " must include a timezone")
        return parsed
    except (ValueError, OverflowError) as exc:
        raise ValidationError(path + " must be an ISO 8601 timestamp with timezone") from exc


def records(value, id_field, path):
    require(isinstance(value, list), path + " must be an array")
    seen = set()
    for record in value:
        require(isinstance(record, dict), path + " entries must be objects")
        text(record.get(id_field), path + "." + id_field)
        require(record[id_field] not in seen, path + " contains duplicate IDs")
        seen.add(record[id_field])
    return value


def validate_measurement(measurement, order, path):
    object_fields(measurement, ("metric", "value", "unit"), path)
    metric = measurement["metric"]
    require(isinstance(metric, str) and metric in order["tolerances"],
            path + " metric has no work-order tolerance")
    unit, low, high = METRICS[metric]
    require(measurement["unit"] == unit, path + " unit must be " + unit)
    number(measurement["value"], path + ".value")
    require(low <= measurement["value"] <= high,
            path + " value is outside demonstrative physical bounds")


def out_of_tolerance(measurement, order):
    bounds = order["tolerances"][measurement["metric"]]
    return not bounds["min"] <= measurement["value"] <= bounds["max"]


def parse_csv(source):
    text(source, "telemetry_csv")
    require(len(source) <= 2_000_000, "telemetry_csv exceeds bounded reference limit")
    try:
        reader = csv.DictReader(io.StringIO(source), strict=True)
        require(reader.fieldnames == CSV_COLUMNS, "telemetry_csv header must match shared schema")
        result = []
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()),
                    "telemetry_csv row has missing or extra columns")
            row = {key: value.strip() for key, value in row.items()}
            try:
                row["value"] = float(row["value"])
            except ValueError as exc:
                raise ValidationError("telemetry_csv value must be numeric") from exc
            result.append(row)
        return result
    except csv.Error as exc:
        raise ValidationError("malformed telemetry_csv: " + str(exc)) from exc


def validate_common(state):
    require(type(state["schema_version"]) is int and state["schema_version"] == SCHEMA_VERSION,
            "unsupported schema_version")
    require(state["synthetic"] is True, "reference fixture must be labeled synthetic")
    text(state["fixture_label"], "fixture_label")
    require(state["industry"] == "Manufacturing", "industry must be Manufacturing")
    orders = {}
    customers = {}
    for order in records(state["work_orders"], "work_order_id", "work_orders"):
        object_fields(order, (
            "work_order_id", "customer_id", "customer_name", "asset_id", "lot_id",
            "product", "operator", "tolerances"
        ), "work_order")
        for key in set(order) - {"tolerances"}:
            text(order[key], "work_order." + key)
        customer = order["customer_id"]
        require(customer not in customers or customers[customer] == order["customer_name"],
                "customer_id has conflicting names")
        customers[customer] = order["customer_name"]
        tolerances = order["tolerances"]
        require(isinstance(tolerances, dict) and bool(tolerances), "tolerances must be nonempty")
        for metric, bounds in tolerances.items():
            require(metric in METRICS, "unsupported tolerance metric")
            object_fields(bounds, ("unit", "min", "max", "safety_max"), "tolerance")
            unit, physical_min, physical_max = METRICS[metric]
            require(bounds["unit"] == unit, "tolerance has wrong unit")
            for name in ("min", "max", "safety_max"):
                number(bounds[name], "tolerance." + name)
            require(physical_min <= bounds["min"] <= bounds["max"]
                    <= bounds["safety_max"] <= physical_max, "invalid tolerance interval")
        orders[order["work_order_id"]] = order
    require(bool(orders), "at least one work order is required")

    def linked(record):
        order_id = record.get("work_order_id")
        require(isinstance(order_id, str) and order_id in orders, "unknown work_order_id")
        return orders[order_id]

    previous = {}
    for reading in records(state["telemetry"], "reading_id", "telemetry"):
        object_fields(reading, CSV_COLUMNS, "reading")
        order = linked(reading)
        require(reading["asset_id"] == order["asset_id"], "reading asset does not match work order")
        require(reading["metric"] in ("temperature", "vibration"),
                "time-series sensor metric must be temperature or vibration")
        validate_measurement({key: reading[key] for key in ("metric", "value", "unit")},
                             order, "reading")
        at = timestamp(reading["timestamp"], "reading.timestamp")
        key = (reading["work_order_id"], reading["asset_id"], reading["metric"])
        require(key not in previous or at >= previous[key], "sensor series is not chronological")
        previous[key] = at
    for inspection in records(state["quality_inspections"], "inspection_id", "quality_inspections"):
        object_fields(inspection, (
            "inspection_id", "work_order_id", "lot_id", "measurement",
            "decision", "decision_by", "decided_at", "rationale"
        ), "inspection")
        order = linked(inspection)
        require(inspection["lot_id"] == order["lot_id"], "inspection lot traceability mismatch")
        text(inspection["decision_by"], "inspection.decision_by")
        text(inspection["rationale"], "inspection.rationale")
        timestamp(inspection["decided_at"], "inspection.decided_at")
        validate_measurement(inspection["measurement"], order, "inspection.measurement")
        expected = "reject" if out_of_tolerance(inspection["measurement"], order) else "accept"
        require(inspection["decision"] == expected,
                "inspection decision contradicts measured tolerance")
    for log in records(state["maintenance_logs"], "log_id", "maintenance_logs"):
        object_fields(log, (
            "log_id", "work_order_id", "asset_id", "timestamp", "text",
            "safety_critical", "status"
        ), "maintenance")
        order = linked(log)
        require(log["asset_id"] == order["asset_id"], "maintenance asset mismatch")
        timestamp(log["timestamp"], "maintenance.timestamp")
        text(log["text"], "maintenance.text")
        require(type(log["safety_critical"]) is bool, "safety_critical must be a boolean")
        require(log["status"] in ("open", "resolved"), "invalid maintenance status")
    for feedback in records(state["feedback"], "feedback_id", "feedback"):
        object_fields(feedback, (
            "feedback_id", "work_order_id", "timestamp", "text", "sentiment"
        ), "feedback")
        linked(feedback)
        timestamp(feedback["timestamp"], "feedback.timestamp")
        text(feedback["text"], "feedback.text")
        require(feedback["sentiment"] in ("positive", "neutral", "negative"),
                "invalid feedback sentiment")
    return orders


BASE_FIELDS = (
    "schema_version", "synthetic", "fixture_label", "industry", "work_orders",
    "telemetry", "quality_inspections", "maintenance_logs", "feedback"
)


def document_output(state):
    orders = {order["work_order_id"]: order for order in state["work_orders"]}
    alerts = []
    decisions = []

    def alert(source_type, source_id, order, severity, reason, observed_at, evidence):
        alerts.append({
            "alert_id": source_type + ":" + source_id,
            "source_type": source_type, "source_id": source_id,
            "work_order_id": order["work_order_id"], "customer_id": order["customer_id"],
            "asset_id": order["asset_id"], "lot_id": order["lot_id"],
            "severity": severity, "reason": reason, "observed_at": observed_at,
            "evidence": evidence, "human_required": severity == "critical",
        })

    for reading in state["telemetry"]:
        order = orders[reading["work_order_id"]]
        bounds = order["tolerances"][reading["metric"]]
        if out_of_tolerance(reading, order):
            severity = "critical" if reading["value"] > bounds["safety_max"] else "warning"
            alert("telemetry", reading["reading_id"], order, severity,
                  reading["metric"] + " outside tolerance", reading["timestamp"],
                  {"metric": reading["metric"], "value": reading["value"],
                   "unit": reading["unit"], "tolerance": copy.deepcopy(bounds)})
    for inspection in state["quality_inspections"]:
        order = orders[inspection["work_order_id"]]
        measurement = inspection["measurement"]
        bounds = order["tolerances"][measurement["metric"]]
        decision = copy.deepcopy(inspection)
        decision.update({
            "asset_id": order["asset_id"], "customer_id": order["customer_id"],
            "tolerance": copy.deepcopy(bounds),
        })
        decisions.append(decision)
        if inspection["decision"] == "reject":
            severity = "critical" if measurement["value"] > bounds["safety_max"] else "high"
            alert("inspection", inspection["inspection_id"], order, severity,
                  "quality inspection rejected", inspection["decided_at"], decision)
    for log in state["maintenance_logs"]:
        if log["status"] == "open":
            order = orders[log["work_order_id"]]
            alert("maintenance", log["log_id"], order,
                  "critical" if log["safety_critical"] else "warning",
                  log["text"], log["timestamp"], copy.deepcopy(log))
    return {
        "record_counts": {key: len(state[key]) for key in (
            "work_orders", "telemetry", "quality_inspections", "maintenance_logs", "feedback"
        )},
        "quality_decisions": decisions, "alerts": alerts,
        "notice": "Demonstrative ISO 9001-style decision traceability; not compliance certification.",
    }


def onboarding_output(state):
    tasks = []
    orders = {order["work_order_id"]: order for order in state["work_orders"]}
    rank = {"critical": 0, "high": 1, "warning": 2, "info": 3}
    for alert in state["documents"]["alerts"]:
        order = orders[alert["work_order_id"]]
        human = alert["human_required"]
        category = (
            "safety" if human else
            "quality" if alert["source_type"] == "inspection" else "maintenance"
        )
        tasks.append({
            "task_id": "review:" + alert["alert_id"],
            "customer_id": order["customer_id"], "work_order_id": order["work_order_id"],
            "asset_id": order["asset_id"], "source_alert_id": alert["alert_id"],
            "priority": alert["severity"], "category": category,
            "assignee_role": "human_safety_lead" if human else "human_operations_lead",
            "status": "awaiting_human" if human else "pending",
            "blocks_onboarding": human or alert["source_type"] == "inspection",
            "instruction": (
                "Escalate to the human safety lead; do not authorize production or restart. "
                if human else "Have the operations lead review the evidence. "
            ) + order["customer_name"] + ": " + alert["reason"],
        })
    for order in state["work_orders"]:
        tasks.append({
            "task_id": "orientation:" + order["work_order_id"],
            "customer_id": order["customer_id"], "work_order_id": order["work_order_id"],
            "asset_id": order["asset_id"], "source_alert_id": None,
            "priority": "info", "category": "training", "assignee_role": "customer_operator",
            "status": "pending", "blocks_onboarding": False,
            "instruction": order["customer_name"] + ": confirm operator " + order["operator"]
            + " understands " + order["asset_id"] + " work-order tolerances and lot "
            + order["lot_id"] + " quality records before starting " + order["product"] + ".",
        })
    tasks.sort(key=lambda task: (rank[task["priority"]], task["task_id"]))
    customers = []
    for customer_id in sorted({order["customer_id"] for order in orders.values()}):
        relevant = [task for task in tasks if task["customer_id"] == customer_id]
        customers.append({
            "customer_id": customer_id,
            "customer_name": orders[relevant[0]["work_order_id"]]["customer_name"],
            "status": "blocked_pending_human" if any(
                task["blocks_onboarding"] for task in relevant) else "ready_for_guided_setup",
            "next_step_id": relevant[0]["task_id"],
            "task_ids": [task["task_id"] for task in relevant],
        })
    return {"customers": customers, "tasks": tasks, "automatic_machine_actions": []}


def insights_output(state):
    keywords = {
        "safety": ("safety", "unsafe", "hazard", "guard", "injury"),
        "quality": ("quality", "defect", "reject", "scrap", "dimension"),
        "maintenance": ("maintenance", "vibration", "repair", "bearing", "downtime"),
        "training": ("training", "onboard", "instruction", "confus", "setup"),
    }
    recommendations = {
        "safety": "Human safety lead must triage safety evidence; no automatic clearance.",
        "quality": "Quality lead should review lot decisions and investigate recurring defects.",
        "maintenance": "Maintenance lead should inspect affected assets and plan follow-up.",
        "training": "Onboarding lead should clarify instructions and confirm operator readiness.",
        "other": "Customer success should review uncategorized feedback with the customer.",
    }
    groups = {}

    def group(category):
        return groups.setdefault(category, {
            "theme": category, "feedback_ids": [], "task_ids": [],
            "work_order_ids": set(), "customer_ids": set(), "negative_feedback_count": 0,
        })

    orders = {order["work_order_id"]: order for order in state["work_orders"]}
    for feedback in state["feedback"]:
        matches = [
            category for category, words in keywords.items()
            if any(word in feedback["text"].casefold() for word in words)
        ] or ["other"]
        for category in matches:
            item = group(category)
            item["feedback_ids"].append(feedback["feedback_id"])
            item["work_order_ids"].add(feedback["work_order_id"])
            item["customer_ids"].add(orders[feedback["work_order_id"]]["customer_id"])
            item["negative_feedback_count"] += feedback["sentiment"] == "negative"
    for task in state["onboarding"]["tasks"]:
        item = group(task["category"])
        item["task_ids"].append(task["task_id"])
        item["work_order_ids"].add(task["work_order_id"])
        item["customer_ids"].add(task["customer_id"])
    themes = []
    for category in sorted(groups):
        item = groups[category]
        for key in ("feedback_ids", "task_ids", "work_order_ids", "customer_ids"):
            item[key] = sorted(item[key])
        item["feedback_count"] = len(item["feedback_ids"])
        item["recommended_action"] = recommendations[category]
        item["human_review_required"] = category == "safety"
        themes.append(item)
    return {
        "themes": themes,
        "summary": {
            "unique_feedback_count": len(state["feedback"]),
            "theme_count": len(themes),
            "blocked_customer_count": sum(
                customer["status"] == "blocked_pending_human"
                for customer in state["onboarding"]["customers"]),
            "human_safety_escalations": sum(
                task["assignee_role"] == "human_safety_lead"
                for task in state["onboarding"]["tasks"]),
        },
        "method": "Deterministic multi-label keyword themes plus validated onboarding tasks; "
                  "counts are descriptive, not statistical or causal claims.",
    }


def validate(state, stage):
    """Single schema gateway for raw input and each complete stage handoff.

    Derived sections are recomputed to reject missing evidence, modified decisions,
    forged references, lost escalations, or unauthorized automatic actions.
    """
    require(stage == "input" or stage in STAGES, "unknown pipeline stage")
    base = set(BASE_FIELDS)
    if stage == "input":
        base.remove("telemetry")
        base.add("telemetry_csv")
        object_fields(state, base, "input")
        normalized = copy.deepcopy(state)
        normalized["telemetry"] = parse_csv(normalized.pop("telemetry_csv"))
        validate_common(normalized)
        return normalized
    index = STAGES.index(stage)
    object_fields(state, base | {"stage", "status"} | set(STAGES[:index + 1]), stage)
    require(state["stage"] == stage and state["status"] == "ok", "invalid stage envelope")
    validate_common(state)
    for name, builder in (
        ("documents", document_output), ("onboarding", onboarding_output), ("insights", insights_output)
    ):
        if STAGES.index(name) <= index:
            expected = builder(state)
            require(
                json.dumps(state[name], sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False),
                name + " handoff does not match validated evidence",
            )
    return state


def documents(raw):
    state = validate(raw, "input")
    state.update(stage="documents", status="ok")
    state["documents"] = document_output(state)
    return validate(state, "documents")


def onboard(document_state):
    validate(document_state, "documents")
    state = copy.deepcopy(document_state)
    state["stage"] = "onboarding"
    state["onboarding"] = onboarding_output(state)
    return validate(state, "onboarding")


def insights(onboarding_state):
    validate(onboarding_state, "onboarding")
    state = copy.deepcopy(onboarding_state)
    state["stage"] = "insights"
    state["insights"] = insights_output(state)
    return validate(state, "insights")


def run_pipeline(raw):
    return insights(onboard(documents(raw)))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as source:
            raw = json.load(source, object_pairs_hook=unique_object,
                            parse_constant=invalid_constant)
        result = run_pipeline(raw)
        output = json.dumps(result, ensure_ascii=True, allow_nan=False)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
