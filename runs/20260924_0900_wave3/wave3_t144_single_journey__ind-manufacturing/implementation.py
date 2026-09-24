"""Synthetic manufacturing journey planner; not a compliance certification."""

import csv
import io
import json
import math
import random
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, name):
    require(isinstance(value, dict), name + " must be an object")
    require(set(value) == set(expected.split()), name + " has missing or unknown fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def number(value, name):
    require(type(value) in (int, float), name + " must be finite numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, name + " must be finite numeric")
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO timestamp") from exc
    require(result.tzinfo is not None and result.utcoffset() is not None,
            "timestamp must include timezone")
    return result


def validate(data):
    """Single input gate shared by discovery, journey planning and the CLI."""
    fields(data, "schema_version synthetic_fixture profile work_order telemetry_csv inspections maintenance_logs", "input")
    require(data["schema_version"] == "1.0", "unsupported schema_version")
    require(data["synthetic_fixture"] is True, "fixtures must be labeled synthetic")
    fields(data["profile"], "priority", "profile")
    require(data["profile"]["priority"] in ("quality", "maintenance", "production"),
            "invalid discovery priority")
    order = data["work_order"]
    fields(order, "id asset_id product_id state tolerance", "work_order")
    for key in ("id", "asset_id", "product_id"):
        text(order[key], key)
    require(order["state"] == "awaiting_review", "work order must await review")
    tolerance = order["tolerance"]
    fields(tolerance, "characteristic unit lower upper", "tolerance")
    require(tolerance["characteristic"] == "shaft_diameter" and tolerance["unit"] == "mm",
            "shaft diameter tolerance requires mm")
    lower, upper = (number(tolerance[k], k) for k in ("lower", "upper"))
    require(0 < lower < upper <= 1000, "invalid diameter tolerance bounds")
    require(isinstance(data["inspections"], list) and len(data["inspections"]) <= 1000,
            "inspections must be a bounded list")
    seen = set()
    previous = None
    for record in data["inspections"]:
        fields(record, "id work_order_id asset_id timestamp inspector_id characteristic value unit decision rationale",
               "inspection")
        for key in ("id", "inspector_id", "rationale"):
            text(record[key], key)
        require(record["id"] not in seen, "duplicate inspection ID")
        seen.add(record["id"])
        require(record["work_order_id"] == order["id"] and record["asset_id"] == order["asset_id"],
                "inspection traceability mismatch")
        moment = timestamp(record["timestamp"])
        require(previous is None or moment > previous, "inspections must be strictly chronological")
        previous = moment
        require(record["characteristic"] == tolerance["characteristic"] and record["unit"] == tolerance["unit"],
                "inspection unit/characteristic mismatch")
        measured = number(record["value"], "inspection value")
        require(0 < measured <= 1000, "implausible shaft diameter")
        expected = "pass" if lower <= measured <= upper else "fail"
        require(record["decision"] == expected, "quality decision contradicts tolerance")
    require(isinstance(data["maintenance_logs"], list) and len(data["maintenance_logs"]) <= 1000,
            "maintenance_logs must be a bounded list")
    seen = set()
    for log in data["maintenance_logs"]:
        fields(log, "id asset_id timestamp technician_id note", "maintenance log")
        for key in ("id", "technician_id", "note"):
            text(log[key], key)
        require(log["id"] not in seen, "duplicate maintenance ID")
        seen.add(log["id"])
        require(log["asset_id"] == order["asset_id"], "maintenance asset mismatch")
        timestamp(log["timestamp"])
    raw = text(data["telemetry_csv"], "telemetry_csv")
    require(len(raw) <= 200000, "telemetry CSV too large")
    reader = csv.DictReader(io.StringIO(raw), strict=True)
    readings, previous_by_metric = [], {}
    specifications = {"temperature": ("degC", -40, 200, 70, 85),
                      "vibration": ("mm/s", 0, 100, 7, 12)}
    try:
        require(reader.fieldnames == ["asset_id", "timestamp", "metric", "value", "unit"],
                "invalid telemetry CSV header")
        for row in reader:
            require(len(readings) < 1000, "too many telemetry rows")
            require(None not in row and all(v is not None for v in row.values()), "malformed CSV row")
            require(row["asset_id"] == order["asset_id"], "telemetry asset mismatch")
            require(row["metric"] in specifications, "unsupported telemetry metric")
            unit, minimum, maximum, warning, critical = specifications[row["metric"]]
            require(row["unit"] == unit, "invalid telemetry unit")
            try:
                value = float(row["value"])
            except ValueError as exc:
                raise ValidationError("invalid sensor number") from exc
            require(math.isfinite(value) and minimum <= value <= maximum, "implausible sensor value")
            moment = timestamp(row["timestamp"])
            prior = previous_by_metric.get(row["metric"])
            require(prior is None or moment > prior, "telemetry must increase in time per metric")
            previous_by_metric[row["metric"]] = moment
            readings.append(dict(row, value=value,
                                 severity="critical" if value >= critical else "warning" if value >= warning else "normal"))
    except csv.Error as exc:
        raise ValidationError("malformed telemetry CSV") from exc
    require(bool(readings), "telemetry cannot be empty")
    require(set(previous_by_metric) == set(specifications), "both temperature and vibration readings required")
    return readings


ACTIONS = {
    "escalate_to_human": ({"safety_alert"}, {"escalation_requested"}),
    "recommend_safety_hold": ({"escalation_requested"}, {"hold_recommended"}),
    "investigate_nonconformance": ({"quality_failed"}, {"investigation_planned"}),
    "schedule_reinspection": ({"investigation_planned"}, {"reinspection_planned"}),
    "schedule_inspection": ({"inspection_missing"}, {"inspection_scheduled"}),
    "assign_quality_reviewer": ({"inspection_scheduled"}, {"reviewer_assignment_planned"}),
    "review_maintenance_history": ({"quality_passed", "no_safety_alert"}, {"history_review_planned"}),
    "plan_preventive_maintenance": ({"history_review_planned"}, {"maintenance_planned"}),
    "review_quality_traceability": ({"quality_passed", "no_safety_alert"}, {"trace_review_planned"}),
    "request_human_release_review": ({"trace_review_planned"}, {"release_review_requested"}),
}


def validate_journey(action_ids, facts):
    require(isinstance(action_ids, list) and len(action_ids) == 2, "journey must contain two actions")
    available, steps = set(facts), []
    for index, action in enumerate(action_ids, 1):
        require(action in ACTIONS, "unknown journey action")
        prerequisites, effects = ACTIONS[action]
        require(prerequisites <= available, "unmet journey prerequisites: " + action)
        steps.append({"step": index, "action": action, "requires": sorted(prerequisites),
                      "planned_effects": sorted(effects), "depends_on_step": index - 1 if index > 1 else None})
        available.update(effects)
    return steps


def recommend(data):
    readings = validate(data)
    alerts = [r for r in readings if r["severity"] != "normal"]
    critical = any(r["severity"] == "critical" for r in alerts)
    latest = data["inspections"][-1] if data["inspections"] else None
    facts = {"safety_alert" if critical else "no_safety_alert"}
    facts.add("inspection_missing" if latest is None else "quality_passed" if latest["decision"] == "pass" else "quality_failed")
    if critical:
        pair = ["escalate_to_human", "recommend_safety_hold"]
        reason = "Safety-critical telemetry overrides all personalization; human intervention required."
    elif latest is None:
        pair = ["schedule_inspection", "assign_quality_reviewer"]
        reason = "A traceable inspection is required before any release review."
    elif latest["decision"] == "fail":
        pair = ["investigate_nonconformance", "schedule_reinspection"]
        reason = "The latest quality decision failed the supplied tolerance."
    elif alerts or data["profile"]["priority"] == "maintenance":
        pair = ["review_maintenance_history", "plan_preventive_maintenance"]
        reason = "Warning telemetry or the maintenance preference prioritizes asset care."
    else:
        pair = ["review_quality_traceability", "request_human_release_review"]
        reason = "Passing quality evidence permits a human release-review request, not automatic release."
    steps = validate_journey(pair, facts)
    return {
        "schema_version": "1.0", "status": "ok", "synthetic_fixture": True,
        "work_order_id": data["work_order"]["id"], "asset_id": data["work_order"]["asset_id"],
        "priority": data["profile"]["priority"], "initial_facts": sorted(facts),
        "recommendations": [{"rank": 1, "action": pair[0], "reason": reason,
                             "eligible_now": True}],
        "journey": {"validated": True, "execution": "plan_only", "steps": steps,
                    "prerequisite_semantics": "Step two is conditional on completing step one; planned effects are not observed facts."},
        "human_escalation": {"required": critical, "recipient_role": "plant_safety_supervisor" if critical else None,
                             "delivery_status": "not_sent"},
        "evidence": {"alerts": alerts, "quality_decisions": data["inspections"],
                     "selected_inspection_id": latest["id"] if latest else None,
                     "tolerance": data["work_order"]["tolerance"],
                     "maintenance_logs": data["maintenance_logs"]},
        "limitations": "Demonstrative ISO 9001-style decision traceability, not certification. No MES writes, physical control, or notifications.",
    }


def synthetic_fixture(seed=144):
    rng = random.Random(seed)
    rows = ["asset_id,timestamp,metric,value,unit"]
    for minute in range(3):
        rows.append(f"SYN-MILL-044,2026-09-24T08:0{minute}:00Z,temperature,{rng.uniform(48, 56):.2f},degC")
        rows.append(f"SYN-MILL-044,2026-09-24T08:0{minute}:00Z,vibration,{rng.uniform(1, 3):.2f},mm/s")
    return {
        "schema_version": "1.0", "synthetic_fixture": True, "profile": {"priority": "quality"},
        "work_order": {"id": "SYN-WO-144", "asset_id": "SYN-MILL-044", "product_id": "SYN-SHAFT-A",
                       "state": "awaiting_review",
                       "tolerance": {"characteristic": "shaft_diameter", "unit": "mm", "lower": 19.95, "upper": 20.05}},
        "telemetry_csv": "\n".join(rows) + "\n",
        "inspections": [{"id": "SYN-QI-001", "work_order_id": "SYN-WO-144", "asset_id": "SYN-MILL-044",
                         "timestamp": "2026-09-24T08:03:00Z", "inspector_id": "SYN-INSPECTOR-7",
                         "characteristic": "shaft_diameter", "value": 20.01, "unit": "mm", "decision": "pass",
                         "rationale": "Synthetic calibrated-gauge reading within declared drawing tolerance."}],
        "maintenance_logs": [{"id": "SYN-LOG-001", "asset_id": "SYN-MILL-044",
                              "timestamp": "2026-09-23T10:00:00Z", "technician_id": "SYN-TECH-2",
                              "note": "Invented fixture: spindle lubrication checked; no abnormal wear observed."}],
    }


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=reject_duplicates)
        output = recommend(data)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        output, code = {"schema_version": "1.0", "status": "error", "error": str(exc)}, 2
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
