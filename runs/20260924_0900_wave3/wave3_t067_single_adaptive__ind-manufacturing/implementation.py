"""Synthetic manufacturing onboarding reference; not a compliance certification."""

import copy
import csv
import io
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path


STEPS = ("safety", "units", "traceability", "inspection", "release")
METRICS = {
    "temperature": ("C", -20.0, 150.0, 85.0),
    "vibration": ("mm/s", 0.0, 50.0, 12.0),
}
CSV_FIELDS = [
    "reading_id", "timestamp", "asset_id", "metric", "value", "unit",
    "tolerance_min", "tolerance_max",
]
EXPLANATIONS = {
    "safety": "Recognize safety-critical temperature and vibration alerts and escalate to a human.",
    "units": "Verify measurement units and inclusive tolerances before interpreting readings.",
    "traceability": "Link each quality decision to its work order, asset, evidence, inspector and time.",
    "inspection": "Evaluate recorded measurements against tolerances and retain the decision rationale.",
    "release": "Review prerequisites and quality evidence; this guide does not authorize machine operation.",
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, path):
    require(isinstance(value, dict), path + " must be an object")
    return value


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    return value


def number(value, path):
    require(type(value) in (int, float), path + " must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, path + " must be finite")
    return value


def timestamp(value, path):
    text(value, path)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError(path + " must be an ISO 8601 timestamp") from None
    require(result.tzinfo is not None, path + " must include a timezone")
    return result


def tolerance(value, low, high, path):
    obj(value, path)
    minimum = number(value.get("min"), path + ".min")
    maximum = number(value.get("max"), path + ".max")
    require(low <= minimum <= maximum <= high, path + " is reversed or outside supported bounds")
    return minimum, maximum


def validate(data):
    """The single validation layer for the shared request/response envelope."""
    obj(data, "input")
    require(data.get("schema_version") == "1.0", "schema_version must be 1.0")
    require(data.get("synthetic") is True, "synthetic must be true")
    profile = obj(data.get("profile"), "profile")
    require(profile.get("experience") in ("novice", "intermediate", "expert"),
            "profile.experience is invalid")
    require(profile.get("preference") in ("concise", "detailed"), "profile.preference is invalid")
    completed = profile.get("completed_steps")
    require(isinstance(completed, list), "profile.completed_steps must be a list")
    require(all(isinstance(step, str) and step in STEPS for step in completed),
            "unknown completed step")
    require(len(set(completed)) == len(completed), "duplicate completed steps")
    require(set(completed) == set(STEPS[:len(completed)]),
            "completed steps must satisfy all earlier prerequisites")
    order = obj(data.get("work_order"), "work_order")
    for field in ("work_order_id", "asset_id", "product_id"):
        text(order.get(field), "work_order." + field)
    require(type(order.get("quantity")) is int and order["quantity"] > 0,
            "work_order.quantity must be a positive integer")
    require(order.get("status") in ("planned", "in_progress", "hold", "completed"),
            "work_order.status is invalid")
    created = timestamp(order.get("created_at"), "work_order.created_at")
    raw_csv = text(data.get("telemetry_csv"), "telemetry_csv")
    readings = []
    ids = set()
    previous = None
    try:
        reader = csv.DictReader(io.StringIO(raw_csv), strict=True)
        require(reader.fieldnames == CSV_FIELDS, "telemetry CSV header is invalid")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()),
                    "telemetry CSV row has an incorrect number of columns")
            rid = text(row["reading_id"], "reading_id")
            require(rid not in ids, "duplicate reading_id")
            ids.add(rid)
            when = timestamp(row["timestamp"], "reading.timestamp")
            require(when >= created and (previous is None or when >= previous),
                    "telemetry must be chronological and not precede the work order")
            previous = when
            require(row["asset_id"] == order["asset_id"], "telemetry asset does not match work order")
            require(row["metric"] in METRICS, "unsupported telemetry metric")
            unit, lower, upper, critical = METRICS[row["metric"]]
            require(row["unit"] == unit, "invalid telemetry unit")
            try:
                value, minimum, maximum = (
                    float(row[key]) for key in ("value", "tolerance_min", "tolerance_max"))
            except ValueError:
                raise ValidationError("telemetry values must be numeric") from None
            number(value, "reading.value")
            require(lower <= value <= upper, "reading outside supported physical bounds")
            tolerance({"min": minimum, "max": maximum}, lower, upper, "reading.tolerance")
            readings.append({
                "reading_id": rid, "timestamp": row["timestamp"], "asset_id": row["asset_id"],
                "metric": row["metric"], "value": value, "unit": unit,
                "tolerance": {"min": minimum, "max": maximum},
                "within_tolerance": minimum <= value <= maximum,
                "safety_critical": value >= critical,
            })
    except csv.Error as exc:
        raise ValidationError("malformed telemetry CSV: " + str(exc)) from None
    require(bool(readings), "at least one telemetry reading is required")
    by_id = {reading["reading_id"]: reading for reading in readings}
    inspections = data.get("quality_inspections")
    require(isinstance(inspections, list), "quality_inspections must be a list")
    inspection_ids = set()
    for record in inspections:
        obj(record, "inspection")
        for field in ("inspection_id", "inspector", "rationale", "evidence_reading_id"):
            text(record.get(field), "inspection." + field)
        require(record["inspection_id"] not in inspection_ids, "duplicate inspection_id")
        inspection_ids.add(record["inspection_id"])
        require(record.get("work_order_id") == order["work_order_id"], "inspection work order mismatch")
        require(record.get("asset_id") == order["asset_id"], "inspection asset mismatch")
        require(record.get("characteristic") == "shaft_diameter", "unsupported inspection characteristic")
        require(record.get("unit") == "mm", "inspection unit must be mm")
        value = number(record.get("value"), "inspection.value")
        require(0 < value <= 1000, "inspection dimension outside supported physical bounds")
        low, high = tolerance(record.get("tolerance"), 0, 1000, "inspection.tolerance")
        require(low > 0, "inspection tolerance minimum must be positive")
        expected = "pass" if low <= value <= high else "fail"
        require(record.get("decision") == expected, "quality decision contradicts measured tolerance")
        require(record["evidence_reading_id"] in by_id, "inspection evidence reading does not exist")
        inspected = timestamp(record.get("timestamp"), "inspection.timestamp")
        evidence_time = timestamp(by_id[record["evidence_reading_id"]]["timestamp"], "evidence.timestamp")
        require(inspected >= evidence_time, "inspection cannot precede its evidence")
    logs = data.get("maintenance_logs")
    require(isinstance(logs, list), "maintenance_logs must be a list")
    log_ids = set()
    for log in logs:
        obj(log, "maintenance_log")
        for field in ("log_id", "technician", "action"):
            text(log.get(field), "maintenance_log." + field)
        require(log["log_id"] not in log_ids, "duplicate maintenance log ID")
        log_ids.add(log["log_id"])
        require(log.get("synthetic") is True, "maintenance logs must be labeled synthetic")
        require(log.get("asset_id") == order["asset_id"], "maintenance asset mismatch")
        timestamp(log.get("timestamp"), "maintenance_log.timestamp")
    return readings


def onboard(data):
    readings = validate(data)
    response = copy.deepcopy(data)
    response["status"] = "ok"
    response["telemetry_readings"] = readings
    alerts = [{
        "reading_id": reading["reading_id"], "asset_id": reading["asset_id"],
        "severity": "safety_critical", "route": "human_safety_supervisor",
        "state": "pending_human_review", "automatic_clearance": False,
        "reason": "{} reached the fixed safety threshold".format(reading["metric"]),
    } for reading in readings if reading["safety_critical"]]
    blockers = []
    if alerts:
        blockers.append("safety_critical_alert_requires_human")
    if any(not reading["within_tolerance"] for reading in readings):
        blockers.append("telemetry_outside_tolerance")
    if not data["quality_inspections"]:
        blockers.append("quality_inspection_required")
    if any(record["decision"] == "fail" for record in data["quality_inspections"]):
        blockers.append("failed_quality_inspection_requires_human")
    if data["work_order"]["status"] == "hold":
        blockers.append("work_order_on_hold")
    profile = data["profile"]
    completed = set(profile["completed_steps"])
    first = readings[0]
    novice_examples = {
        "safety": "Example: 85 C requires a human safety supervisor even if the configured upper tolerance is 100 C.",
        "units": "Example: {} reports {} {}; compare with [{}, {}] in the same unit.".format(
            first["reading_id"], first["value"], first["unit"],
            first["tolerance"]["min"], first["tolerance"]["max"]),
        "traceability": "Example: retain work order {} and asset {} alongside the inspector, evidence ID and rationale.".format(
            data["work_order"]["work_order_id"], data["work_order"]["asset_id"]),
        "inspection": "Example: a 20.01 mm shaft passes inclusive limits [19.95, 20.05] mm; a 20.06 mm shaft fails.",
        "release": "Example: completed training plus a safety alert still means blocked pending human review.",
    }
    steps = []
    for index, step in enumerate(STEPS):
        prereqs = list(STEPS[:index])
        if step == "release" and blockers:
            state = "blocked"
        elif step in completed:
            state = "completed"
        elif all(prereq in completed for prereq in prereqs):
            state = "available"
        else:
            state = "locked"
        explanation = EXPLANATIONS[step]
        if profile["preference"] == "detailed":
            explanation += " Use the linked synthetic work order and evidence; retain each review result."
        steps.append({
            "step_id": step, "prerequisites": prereqs, "state": state,
            "explanation": explanation,
            "guidance": "worked_example" if profile["experience"] == "novice" else (
                "guided_checklist" if profile["experience"] == "intermediate" else "verification_checklist"),
            "learning_task": novice_examples[step] if profile["experience"] == "novice" else (
                "Review the explanation, check the linked evidence, then record completion of " + step + "."
                if profile["experience"] == "intermediate" else
                "Verify " + step + " against source evidence and record any exceptions."),
        })
    response["quality_decisions"] = [{
        "inspection_id": record["inspection_id"], "work_order_id": record["work_order_id"],
        "asset_id": record["asset_id"], "decision": record["decision"],
        "inspector": record["inspector"], "timestamp": record["timestamp"],
        "rationale": record["rationale"], "evidence_reading_id": record["evidence_reading_id"],
        "rule": "inclusive_dimension_tolerance_v1",
        "measurement": {"value": record["value"], "unit": record["unit"],
                        "tolerance": copy.deepcopy(record["tolerance"])},
    } for record in data["quality_inspections"]]
    response["safety_alerts"] = alerts
    response["onboarding"] = {
        "steps": steps, "blockers": blockers,
        "next_step": next((step["step_id"] for step in steps if step["state"] == "available"), None),
        "complete": all(step["state"] == "completed" for step in steps),
        "release_readiness": "blocked" if blockers else (
            "ready_for_human_review" if set(STEPS[:-1]).issubset(completed) else "prerequisites_pending"),
        "operational_authorization": False,
        "notice": "Synthetic demonstration of traceability principles; no ISO 9001 certification claim.",
    }
    validate(response)
    return response


def synthetic_fixture(seed=67):
    rng = random.Random(seed)
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    for index in range(6):
        metric = "temperature" if index % 2 == 0 else "vibration"
        value = round(rng.uniform(57, 63) if metric == "temperature" else rng.uniform(1.2, 2.4), 3)
        writer.writerow([
            "SYN-R-{:03d}".format(index), "2026-09-24T08:{:02d}:00Z".format(index),
            "SYN-MILL-067", metric, value, "C" if metric == "temperature" else "mm/s",
            40 if metric == "temperature" else 0, 75 if metric == "temperature" else 5,
        ])
    return {
        "schema_version": "1.0", "synthetic": True,
        "profile": {"experience": "novice", "preference": "detailed", "completed_steps": []},
        "work_order": {
            "work_order_id": "SYN-WO-2067", "asset_id": "SYN-MILL-067",
            "product_id": "SYN-SHAFT-A", "quantity": 40, "status": "in_progress",
            "created_at": "2026-09-24T07:00:00Z",
        },
        "telemetry_csv": stream.getvalue(),
        "quality_inspections": [{
            "inspection_id": "SYN-QI-067", "work_order_id": "SYN-WO-2067",
            "asset_id": "SYN-MILL-067", "inspector": "Fictitious Inspector Ada",
            "timestamp": "2026-09-24T08:10:00Z", "characteristic": "shaft_diameter",
            "value": 20.01, "unit": "mm", "tolerance": {"min": 19.95, "max": 20.05},
            "decision": "pass", "rationale": "Synthetic gauge result within inclusive drawing limits.",
            "evidence_reading_id": "SYN-R-005",
        }],
        "maintenance_logs": [{
            "log_id": "SYN-MAINT-067", "asset_id": "SYN-MILL-067", "synthetic": True,
            "timestamp": "2026-09-23T14:00:00Z", "technician": "Fictitious Technician Ren",
            "action": "Invented maintenance: lubricated spindle and checked coolant circulation.",
        }],
    }


def reject_constant(value):
    raise ValidationError("nonfinite JSON constant is forbidden: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = onboard(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
