"""Synthetic manufacturing discovery reference; no compliance certification.

Run: python -B implementation.py example_input.json
Only Python's standard library is used. Input CSV is embedded, never a file path.
"""

import csv
import io
import json
import math
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


METRICS = {
    "temperature": ("C", -40, 250),
    "vibration": ("mm/s", 0, 100),
    "pressure": ("bar", 0, 300),
}
CSV_COLUMNS = [
    "reading_id", "work_order_id", "asset_id", "timestamp", "metric", "value", "unit"
]


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, name):
    require(isinstance(value, dict), name + " must be an object")
    return value


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite numeric")
    return value


def timestamp(value, name):
    text(value, name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(name + " must be ISO 8601") from exc
    require(result.tzinfo is not None and result.utcoffset() is not None,
            name + " must include timezone")
    return result


def sequence(value, name):
    require(isinstance(value, list), name + " must be an array")
    return value


def validate(data):
    """One validation boundary used by the CLI and the in-process API."""
    obj(data, "input")
    require(data.get("schema_version") == "1.0", "schema_version must be 1.0")
    require(data.get("synthetic") is True, "synthetic must be true")
    now = timestamp(data.get("as_of"), "as_of")
    user_id = text(data.get("user_id"), "user_id")
    half_life = number(data.get("half_life_days", 14), "half_life_days")
    require(0 < half_life <= 3650, "half_life_days must be in (0, 3650]")
    limit = data.get("limit", 10)
    require(type(limit) is int and 1 <= limit <= 100, "limit must be an integer in [1, 100]")
    orders = {}
    for order in sequence(data.get("work_orders"), "work_orders"):
        obj(order, "work order")
        oid = text(order.get("id"), "work order id")
        require(oid not in orders, "duplicate work order id")
        text(order.get("asset_id"), "asset_id")
        text(order.get("title"), "title")
        text(order.get("maintenance_log"), "invented maintenance_log")
        metric = order.get("metric")
        require(isinstance(metric, str) and metric in METRICS, "unsupported metric")
        unit, physical_min, physical_max = METRICS[metric]
        require(order.get("unit") == unit, "work order unit mismatch")
        low = number(order.get("tolerance_min"), "tolerance_min")
        high = number(order.get("tolerance_max"), "tolerance_max")
        safety_low = number(order.get("safety_min"), "safety_min")
        safety_high = number(order.get("safety_max"), "safety_max")
        require(physical_min <= safety_low <= low < high <= safety_high <= physical_max,
                "invalid tolerance/safety bounds")
        created = timestamp(order.get("created_at"), "created_at")
        due = timestamp(order.get("due_at"), "due_at")
        require(created <= now and created <= due, "invalid work order chronology")
        orders[oid] = dict(order, created_time=created, due_time=due)

    telemetry = data.get("telemetry_csv")
    require(isinstance(telemetry, str), "telemetry_csv must be text")
    readings = {}
    try:
        reader = csv.DictReader(io.StringIO(telemetry), strict=True)
        require(reader.fieldnames == CSV_COLUMNS, "invalid telemetry CSV header")
        for row in reader:
            require(set(row) == set(CSV_COLUMNS) and all(v is not None for v in row.values()),
                    "malformed telemetry CSV row")
            rid = text(row["reading_id"], "reading_id")
            require(rid not in readings, "duplicate reading_id")
            require(row["work_order_id"] in orders, "reading references unknown work order")
            order = orders[row["work_order_id"]]
            require(row["asset_id"] == order["asset_id"], "reading asset mismatch")
            require(row["metric"] == order["metric"] and row["unit"] == order["unit"],
                    "reading metric/unit mismatch")
            try:
                value = float(row["value"])
            except ValueError as exc:
                raise ValidationError("invalid telemetry value") from exc
            number(value, "telemetry value")
            _, physical_min, physical_max = METRICS[order["metric"]]
            require(physical_min <= value <= physical_max, "physically implausible telemetry")
            time = timestamp(row["timestamp"], "reading timestamp")
            require(order["created_time"] <= time <= now, "invalid reading chronology")
            readings[rid] = dict(row, value=value, time=time)
    except csv.Error as exc:
        raise ValidationError("malformed telemetry CSV") from exc

    inspections = {}
    for inspection in sequence(data.get("quality_inspections"), "quality_inspections"):
        obj(inspection, "quality inspection")
        iid = text(inspection.get("id"), "inspection id")
        require(iid not in inspections, "duplicate inspection id")
        oid = text(inspection.get("work_order_id"), "inspection work_order_id")
        require(oid in orders, "inspection references unknown work order")
        require(inspection.get("decision") in ("pass", "fail"), "invalid quality decision")
        text(inspection.get("decision_by"), "decision_by")
        text(inspection.get("reason"), "quality decision reason")
        time = timestamp(inspection.get("timestamp"), "inspection timestamp")
        require(orders[oid]["created_time"] <= time <= now, "invalid inspection chronology")
        sources = sequence(inspection.get("source_reading_ids"), "source_reading_ids")
        require(bool(sources), "quality decision requires source readings")
        for rid in sources:
            text(rid, "source reading id")
            require(rid in readings and readings[rid]["work_order_id"] == oid,
                    "invalid inspection traceability link")
            require(readings[rid]["time"] <= time, "inspection predates evidence")
        require(len(set(sources)) == len(sources), "duplicate inspection evidence")
        order = orders[oid]
        within = all(order["tolerance_min"] <= readings[rid]["value"] <= order["tolerance_max"]
                     for rid in sources)
        require((inspection["decision"] == "pass") == within,
                "quality decision contradicts tolerance evidence")
        inspections[iid] = dict(inspection, time=time)

    events = []
    event_ids = set()
    for event in sequence(data.get("events"), "events"):
        obj(event, "event")
        eid = text(event.get("id"), "event id")
        require(eid not in event_ids, "duplicate event id")
        event_ids.add(eid)
        text(event.get("user_id"), "event user_id")
        oid = text(event.get("work_order_id"), "event work_order_id")
        require(oid in orders, "event references unknown work order")
        require(event.get("action") in ("browse", "purchase"), "unsupported event action")
        time = timestamp(event.get("timestamp"), "event timestamp")
        require(orders[oid]["created_time"] <= time <= now, "invalid event chronology")
        events.append(dict(event, time=time))
    return now, user_id, half_life, limit, orders, readings, inspections, events


def run(data):
    now, user_id, half_life, limit, orders, readings, inspections, events = validate(data)
    own_events = [event for event in events if event["user_id"] == user_id]
    mode = "personalized" if own_events else "cold_start"
    alerts = []
    for rid, reading in sorted(readings.items()):
        order = orders[reading["work_order_id"]]
        if not order["safety_min"] <= reading["value"] <= order["safety_max"]:
            alerts.append({
                "reading_id": rid, "work_order_id": order["id"], "asset_id": order["asset_id"],
                "severity": "safety_critical", "escalate_to": "human",
                "state": "pending_human_review", "automatic_action_taken": False,
                "reason": "Reading outside configured safety bounds",
            })
    unsafe_orders = {alert["work_order_id"] for alert in alerts}
    ranked = []
    for oid, order in orders.items():
        score = 0.0
        contributions = []
        for event in own_events:
            source = orders[event["work_order_id"]]
            affinity = (1.0 if source["id"] == oid else
                        0.4 if source["asset_id"] == order["asset_id"] else
                        0.2 if source["metric"] == order["metric"] else 0.0)
            age_days = (now - event["time"]).total_seconds() / 86400
            weight = 3.0 if event["action"] == "purchase" else 1.0
            contribution = weight * affinity * 2 ** (-age_days / half_life)
            score += contribution
            if contribution:
                contributions.append({"event_id": event["id"], "contribution": round(contribution, 8)})
        records = sorted((i for i in inspections.values() if i["work_order_id"] == oid),
                         key=lambda i: (i["time"], i["id"]))
        latest = records[-1] if records else None
        trace = [
            {key: inspection[key] for key in
             ("id", "work_order_id", "decision", "decision_by", "reason", "timestamp", "source_reading_ids")}
            for inspection in records
        ]
        ranked.append({
            "work_order_id": oid, "asset_id": order["asset_id"], "title": order["title"],
            "score": round(score, 8), "requires_human_review": oid in unsafe_orders,
            "quality_status": latest["decision"] if latest else "uninspected",
            "quality_trace": trace, "event_contributions": contributions,
            "discovery_reason": "recency_weighted_affinity" if own_events else "earliest_due_first",
            "_score": score, "_due": order["due_time"],
        })
    # Safety cannot be hidden by personalization; alerts remain complete even under a result limit.
    ranked.sort(key=lambda row: (not row["requires_human_review"], -row["_score"],
                                 row["_due"], row["work_order_id"]))
    for row in ranked:
        del row["_score"]
        del row["_due"]
    return {
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "as_of": data["as_of"], "user_id": user_id, "mode": mode,
        "rankings": ranked[:limit], "alerts": alerts, "total_candidates": len(orders),
        "notice": "Demonstrative traceability and safety validation only; not ISO 9001 certification.",
    }


def reject_constant(value):
    raise ValidationError("Non-finite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run(data)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": "1.0", "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
