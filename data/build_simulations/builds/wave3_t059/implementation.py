"""Synthetic manufacturing discovery reference; no compliance certification.

Run: python -B implementation.py example_input.json
All decisions are advisory; safety escalations never release equipment or orders.
"""

import csv
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path


class ValidationError(ValueError):
    pass


SENSOR_COLUMNS = [
    "reading_id", "work_order_id", "asset_id", "timestamp", "metric",
    "value", "unit", "lower", "upper", "safety_critical",
]
# Demonstrative physical domains, not operating limits or certified thresholds.
METRICS = {
    "temperature": ("C", -50, 500),
    "vibration": ("mm/s", 0, 100),
    "pressure": ("bar", 0, 1000),
    "diameter": ("mm", 0, 10000),
    "torque": ("N*m", 0, 100000),
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(expected), location + " has missing or unknown fields")


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 2000,
            location + " must be a nonempty string of at most 2000 characters")
    require(value == value.strip(), location + " must not have surrounding whitespace")
    return value


def timestamp(value, location):
    text(value, location)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(location + " must be an ISO 8601 timestamp") from exc
    require(parsed.tzinfo is not None, location + " requires a timezone")


def number(value, location):
    require(type(value) in (int, float), location + " must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, location + " must be finite")
    return value


def strings(value, location):
    require(isinstance(value, list), location + " must be an array")
    for item in value:
        text(item, location + " item")
    require(len(set(value)) == len(value), location + " contains duplicates")


def measurement(record, location):
    metric = text(record["metric"], location + ".metric")
    require(metric in METRICS, location + " has unsupported metric")
    unit, minimum, maximum = METRICS[metric]
    require(record["unit"] == unit, location + " has incompatible unit")
    value = number(record["value"], location + ".value")
    lower = number(record["lower"], location + ".lower")
    upper = number(record["upper"], location + ".upper")
    require(minimum <= lower <= upper <= maximum, location + " has invalid tolerances")
    require(minimum <= value <= maximum, location + " is outside physical domain")
    require(type(record["safety_critical"]) is bool,
            location + ".safety_critical must be boolean")
    return not lower <= value <= upper


def validate(payload):
    """Shared validation boundary, including CSV normalization and all entity links."""
    fields(payload, ["schema_version", "synthetic", "fixture_seed", "preferences",
                     "work_orders", "sensor_csv", "quality_inspections"], "input")
    require(payload["schema_version"] == "1.0", "Unsupported schema_version")
    require(payload["synthetic"] is True, "Only clearly labeled synthetic data is accepted")
    require(type(payload["fixture_seed"]) is int, "fixture_seed must be an integer")
    prefs = payload["preferences"]
    fields(prefs, ["interests", "excluded_work_order_ids", "excluded_asset_ids",
                   "excluded_tags", "limit"], "preferences")
    require(isinstance(prefs["interests"], dict), "interests must be an object")
    for tag, weight in prefs["interests"].items():
        text(tag, "interest")
        require(type(weight) is int and 1 <= weight <= 10,
                "interest weights must be integers from 1 to 10")
    for key in ("excluded_work_order_ids", "excluded_asset_ids", "excluded_tags"):
        strings(prefs[key], key)
    require(type(prefs["limit"]) is int and 1 <= prefs["limit"] <= 50,
            "limit must be an integer from 1 to 50")
    require(isinstance(payload["work_orders"], list), "work_orders must be an array")
    orders = {}
    for order in payload["work_orders"]:
        fields(order, ["work_order_id", "asset_id", "title", "tags", "status",
                       "safety_critical", "maintenance_logs"], "work_order")
        for key in ("work_order_id", "asset_id", "title"):
            text(order[key], "work_order." + key)
        require(order["work_order_id"] not in orders, "Duplicate work_order_id")
        strings(order["tags"], "work_order.tags")
        require(order["status"] in ("open", "in_progress", "closed"), "Invalid order status")
        require(type(order["safety_critical"]) is bool, "Order safety_critical must be boolean")
        logs = order["maintenance_logs"]
        require(isinstance(logs, list) and logs, "At least one invented maintenance log is required")
        log_ids = set()
        for log in logs:
            fields(log, ["log_id", "timestamp", "note"], "maintenance_log")
            text(log["log_id"], "log_id")
            text(log["note"], "maintenance_log.note")
            timestamp(log["timestamp"], "maintenance_log.timestamp")
            require(log["log_id"] not in log_ids, "Duplicate maintenance log ID")
            log_ids.add(log["log_id"])
        orders[order["work_order_id"]] = order

    def linked(record, location):
        text(record["work_order_id"], location + ".work_order_id")
        text(record["asset_id"], location + ".asset_id")
        require(record["work_order_id"] in orders, location + " references unknown work order")
        require(record["asset_id"] == orders[record["work_order_id"]]["asset_id"],
                location + " asset does not match work order")
        timestamp(record["timestamp"], location + ".timestamp")

    raw_csv = payload["sensor_csv"]
    require(isinstance(raw_csv, str), "sensor_csv must be a CSV string")
    readings = []
    reading_ids = set()
    try:
        reader = csv.DictReader(io.StringIO(raw_csv), strict=True)
        require(reader.fieldnames == SENSOR_COLUMNS, "sensor_csv headers must match schema")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()),
                    "sensor_csv row has incorrect field count")
            text(row["reading_id"], "reading_id")
            require(row["reading_id"] not in reading_ids, "Duplicate reading_id")
            reading_ids.add(row["reading_id"])
            linked(row, "sensor")
            require(row["metric"] in ("temperature", "vibration", "pressure"),
                    "Unsupported sensor metric")
            require(row["safety_critical"] in ("true", "false"),
                    "CSV safety_critical must be true or false")
            row["safety_critical"] = row["safety_critical"] == "true"
            for key in ("value", "lower", "upper"):
                try:
                    row[key] = float(row[key])
                except ValueError as exc:
                    raise ValidationError("sensor." + key + " must be numeric") from exc
            measurement(row, "sensor")
            readings.append(row)
    except csv.Error as exc:
        raise ValidationError("Malformed sensor CSV") from exc
    require(isinstance(payload["quality_inspections"], list),
            "quality_inspections must be an array")
    inspection_ids = set()
    for record in payload["quality_inspections"]:
        fields(record, ["inspection_id", "work_order_id", "asset_id", "timestamp",
                        "inspector_id", "procedure_revision", "decision", "rationale",
                        "metric", "value", "unit", "lower", "upper", "safety_critical"],
               "quality_inspection")
        for key in ("inspection_id", "inspector_id", "procedure_revision", "rationale"):
            text(record[key], "quality_inspection." + key)
        require(record["inspection_id"] not in inspection_ids, "Duplicate inspection_id")
        inspection_ids.add(record["inspection_id"])
        linked(record, "quality_inspection")
        outside = measurement(record, "quality_inspection")
        require(record["decision"] == ("fail" if outside else "pass"),
                "Quality decision must agree with measured tolerance")
    for order_id in orders:
        require(any(r["work_order_id"] == order_id for r in readings),
                order_id + " requires sensor evidence")
        require(any(r["work_order_id"] == order_id for r in payload["quality_inspections"]),
                order_id + " requires traceable quality inspection evidence")
    return prefs, orders, readings, payload["quality_inspections"]


def recommend(payload):
    prefs, orders, readings, inspections = validate(payload)
    escalations = []
    for kind, records, id_field in (
        ("sensor", readings, "reading_id"),
        ("quality_inspection", inspections, "inspection_id"),
    ):
        for record in records:
            order = orders[record["work_order_id"]]
            outside = not record["lower"] <= record["value"] <= record["upper"]
            if outside and (record["safety_critical"] or order["safety_critical"]):
                escalations.append({
                    "work_order_id": record["work_order_id"],
                    "asset_id": record["asset_id"],
                    "evidence_type": kind,
                    "evidence_id": record[id_field],
                    "reason": "Safety-critical measurement outside supplied tolerance",
                    "route": "human_safety_reviewer",
                    "status": "pending_human_review",
                    "measurement": {k: record[k] for k in
                                    ("metric", "value", "unit", "lower", "upper", "timestamp")},
                })
    escalations.sort(key=lambda x: (x["work_order_id"], x["evidence_type"], x["evidence_id"]))
    candidates = []
    excluded = []
    for order_id, order in orders.items():
        reasons = []
        if order_id in prefs["excluded_work_order_ids"]:
            reasons.append("excluded_work_order")
        if order["asset_id"] in prefs["excluded_asset_ids"]:
            reasons.append("excluded_asset")
        if set(order["tags"]) & set(prefs["excluded_tags"]):
            reasons.append("excluded_tag")
        if order["status"] == "closed":
            reasons.append("closed_work_order")
        if reasons:
            excluded.append({"work_order_id": order_id, "reasons": reasons})
            continue
        matched = sorted(set(order["tags"]) & set(prefs["interests"]))
        if not matched:
            continue
        score = sum(prefs["interests"][tag] for tag in matched)
        evidence = sorted((r for r in readings if r["work_order_id"] == order_id),
                          key=lambda r: r["reading_id"])
        decisions = sorted((r for r in inspections if r["work_order_id"] == order_id),
                           key=lambda r: r["inspection_id"])
        candidates.append({
            "work_order_id": order_id,
            "asset_id": order["asset_id"],
            "title": order["title"],
            "score": score,
            "matched_interests": [{"tag": tag, "weight": prefs["interests"][tag]}
                                  for tag in matched],
            "explanation": "Matched declared tags: " + ", ".join(
                f"{tag} (+{prefs['interests'][tag]})" for tag in matched)
                + f". Weighted sum: {score}. Evidence is advisory, not authorization.",
            "sensor_evidence": evidence,
            "quality_decision_trace": decisions,
            "maintenance_log_ids": sorted(log["log_id"] for log in order["maintenance_logs"]),
            "requires_human_review": any(e["work_order_id"] == order_id for e in escalations),
        })
    candidates.sort(key=lambda c: (-c["score"], c["work_order_id"]))
    return {
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "recommendations": candidates[:prefs["limit"]],
        "eligible_count": len(candidates),
        "excluded": sorted(excluded, key=lambda x: x["work_order_id"]),
        "escalations": escalations,
        "notice": "Synthetic advisory reference. Demonstrative ISO 9001 traceability only; "
                  "no certification claim. Human escalation is a queue record, not a sent notification.",
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        content = Path(args[0]).read_text(encoding="utf-8")
        require(len(content) <= 2_000_000, "Input exceeds 2,000,000 character limit")
        result = recommend(json.loads(content, object_pairs_hook=unique_object))
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"schema_version": "1.0", "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
