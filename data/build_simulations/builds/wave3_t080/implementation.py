"""Synthetic manufacturing reference pipeline; no certification or live providers."""
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


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid timestamp") from exc
    require(result.tzinfo is not None, "timestamp must include timezone")
    return result


def fields(obj, keys, name):
    require(isinstance(obj, dict) and set(obj) == set(keys), name + " has invalid fields")


STEPS = (
    ("verify_work_order", ()),
    ("verify_telemetry", ("verify_work_order",)),
    ("review_inspections", ("verify_telemetry",)),
)
ACTIONS = STEPS + (
    ("review_maintenance", ("review_inspections",)),
    ("request_human_release", ("review_maintenance",)),
    ("monitor_next_cycle", ("request_human_release",)),
)
POSITIVE = {"good", "excellent", "stable", "resolved", "safe", "smooth"}
NEGATIVE = {"bad", "fault", "failed", "unsafe", "delay", "defect", "vibration"}
SEVERITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def validate_input(payload):
    fields(payload, ("schema_version", "synthetic", "work_order", "sensor",
                     "telemetry_csv", "inspections", "completed_steps", "maintenance_logs"), "input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "unsupported schema_version")
    require(payload["synthetic"] is True, "fixtures must be labeled synthetic")
    order = payload["work_order"]
    fields(order, ("id", "asset_id", "product", "quantity", "status"), "work_order")
    for key in ("id", "asset_id", "product"):
        text(order[key], key)
    require(type(order["quantity"]) is int and order["quantity"] > 0, "invalid quantity")
    require(order["status"] in ("planned", "in_progress"), "invalid work order status")
    sensor = payload["sensor"]
    fields(sensor, ("id", "metric", "unit", "tolerance_min", "tolerance_max",
                    "safety_min", "safety_max"), "sensor")
    text(sensor["id"], "sensor id")
    require(sensor["metric"] in ("temperature", "vibration"), "unsupported sensor metric")
    require(sensor["unit"] == {"temperature": "degC", "vibration": "mm/s"}[sensor["metric"]],
            "sensor unit does not match metric")
    for key in ("tolerance_min", "tolerance_max", "safety_min", "safety_max"):
        number(sensor[key], key)
    require(sensor["safety_min"] <= sensor["tolerance_min"] < sensor["tolerance_max"]
            <= sensor["safety_max"] and sensor["safety_min"] < sensor["safety_max"],
            "invalid sensor tolerance/safety bounds")
    require(sensor["safety_min"] >= (-273.15 if sensor["metric"] == "temperature" else 0),
            "physically impossible lower bound")
    text(payload["telemetry_csv"], "telemetry_csv")
    readings = []
    reader = csv.DictReader(io.StringIO(payload["telemetry_csv"]), strict=True)
    expected = ["timestamp", "work_order_id", "asset_id", "sensor_id", "value", "unit"]
    try:
        require(reader.fieldnames == expected, "invalid telemetry CSV header")
        previous = None
        for row in reader:
            require(set(row) == set(expected) and all(v is not None for v in row.values()),
                    "invalid telemetry row")
            instant = timestamp(row["timestamp"])
            require(previous is None or instant > previous, "telemetry must be strictly chronological")
            previous = instant
            require(row["work_order_id"] == order["id"] and row["asset_id"] == order["asset_id"]
                    and row["sensor_id"] == sensor["id"], "telemetry traceability mismatch")
            require(row["unit"] == sensor["unit"], "telemetry unit mismatch")
            try:
                value = float(row["value"])
            except (TypeError, ValueError) as exc:
                raise ValidationError("invalid telemetry numeric value") from exc
            number(value, "telemetry value")
            require(value >= (-273.15 if sensor["metric"] == "temperature" else 0),
                    "physically impossible reading")
            readings.append(dict(row, value=value))
    except csv.Error as exc:
        raise ValidationError("malformed telemetry CSV") from exc
    require(bool(readings), "telemetry cannot be empty")
    inspections = payload["inspections"]
    require(isinstance(inspections, list) and bool(inspections), "inspections cannot be empty")
    ids = set()
    for inspection in inspections:
        fields(inspection, ("id", "work_order_id", "asset_id", "characteristic", "unit",
                            "measured", "tolerance_min", "tolerance_max", "decision",
                            "decision_id", "inspector", "timestamp", "rationale"), "inspection")
        for key in ("id", "characteristic", "decision_id", "inspector", "rationale"):
            text(inspection[key], key)
        require(inspection["id"] not in ids and inspection["decision_id"] not in ids,
                "duplicate inspection/decision identifier")
        require(inspection["id"] != inspection["decision_id"], "decision id must be distinct")
        ids.update((inspection["id"], inspection["decision_id"]))
        require(inspection["work_order_id"] == order["id"] and inspection["asset_id"] == order["asset_id"],
                "inspection traceability mismatch")
        require(inspection["characteristic"] == "diameter" and inspection["unit"] == "mm",
                "inspection characteristic/unit mismatch")
        for key in ("measured", "tolerance_min", "tolerance_max"):
            number(inspection[key], key)
        require(0 < inspection["tolerance_min"] < inspection["tolerance_max"]
                and inspection["measured"] > 0, "invalid dimensional tolerance/measurement")
        expected_decision = ("pass" if inspection["tolerance_min"] <= inspection["measured"]
                             <= inspection["tolerance_max"] else "fail")
        require(inspection["decision"] == expected_decision, "quality decision contradicts tolerance")
        timestamp(inspection["timestamp"])
    completed = payload["completed_steps"]
    require(isinstance(completed, list) and all(isinstance(s, str) for s in completed),
            "completed_steps must be a list of names")
    require(len(set(completed)) == len(completed), "duplicate completed step")
    known = dict(STEPS)
    seen = set()
    for step in completed:
        require(step in known and set(known[step]) <= seen, "onboarding prerequisites not satisfied")
        seen.add(step)
    logs = payload["maintenance_logs"]
    require(isinstance(logs, list), "maintenance_logs must be a list")
    ids = set()
    for log in logs:
        fields(log, ("id", "work_order_id", "asset_id", "timestamp", "text", "severity",
                     "safety_critical"), "maintenance log")
        text(log["id"], "log id")
        require(log["id"] not in ids, "duplicate maintenance log id")
        ids.add(log["id"])
        require(log["work_order_id"] == order["id"] and log["asset_id"] == order["asset_id"],
                "maintenance log traceability mismatch")
        timestamp(log["timestamp"])
        text(log["text"], "maintenance text")
        require(isinstance(log["severity"], str) and log["severity"] in SEVERITY, "invalid severity")
        require(type(log["safety_critical"]) is bool, "safety_critical must be boolean")
    return readings


def _guided(data):
    completed = data["completed_steps"]
    return {
        "work_order_id": data["work_order"]["id"],
        "completed_steps": completed[:],
        "progress": {"completed": len(completed), "total": len(STEPS),
                     "fraction": len(completed) / len(STEPS)},
        "available_steps": [s for s, pre in STEPS if s not in completed and set(pre) <= set(completed)],
        "quality_decision_ids": [i["decision_id"] for i in data["inspections"]],
    }


def _journey(data, guided, readings):
    available = set(guided["completed_steps"])
    journey = []
    for _ in range(2):
        action, prerequisites = next((a, p) for a, p in ACTIONS
                                     if a not in available and set(p) <= available)
        journey.append({"action": action, "prerequisites": list(prerequisites),
                        "reason": "First unmet action whose prerequisites are satisfied"})
        available.add(action)
    sensor = data["sensor"]
    signals = []
    for index, reading in enumerate(readings):
        safety = not sensor["safety_min"] <= reading["value"] <= sensor["safety_max"]
        if safety or not sensor["tolerance_min"] <= reading["value"] <= sensor["tolerance_max"]:
            signals.append({"id": "telemetry:" + str(index), "source": "telemetry",
                            "source_id": reading["timestamp"], "text": "Sensor tolerance excursion",
                            "severity": "critical" if safety else "high", "safety_critical": safety,
                            "decision_id": None})
    for inspection in data["inspections"]:
        if inspection["decision"] == "fail":
            signals.append({"id": "inspection:" + inspection["id"], "source": "inspection",
                            "source_id": inspection["id"], "text": inspection["rationale"],
                            "severity": "high", "safety_critical": False,
                            "decision_id": inspection["decision_id"]})
    for log in data["maintenance_logs"]:
        signals.append({"id": "maintenance:" + log["id"], "source": "maintenance",
                        "source_id": log["id"], "text": log["text"], "severity": log["severity"],
                        "safety_critical": log["safety_critical"], "decision_id": None})
    return {"work_order_id": guided["work_order_id"], "based_on_progress": guided["progress"],
            "quality_decision_ids": guided["quality_decision_ids"], "steps": journey,
            "signals": signals, "execution": "recommendations_only"}


def score_sentiment(value):
    # Exact alphabetic token matching is intentionally bounded; no negation inference.
    tokens = "".join(c.lower() if c.isalpha() else " " for c in value).split()
    positive = [token for token in tokens if token in POSITIVE]
    negative = [token for token in tokens if token in NEGATIVE]
    score = (len(positive) - len(negative)) / max(1, len(positive) + len(negative))
    return {"score": score, "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "positive_matches": positive, "negative_matches": negative}


def _sentiment(journey):
    issues = []
    for signal in journey["signals"]:
        sentiment = score_sentiment(signal["text"])
        severity = "critical" if signal["safety_critical"] else signal["severity"]
        escalation = signal["safety_critical"] or severity == "critical"
        issues.append(dict(signal, severity=severity, sentiment=sentiment,
                           work_order_id=journey["work_order_id"],
                           priority_score=SEVERITY[severity] * 10 + max(0, -sentiment["score"]),
                           route="human_safety_review" if escalation else "quality_maintenance_queue",
                           human_required=escalation))
    issues.sort(key=lambda issue: (-int(issue["human_required"]), -issue["priority_score"], issue["id"]))
    return {"issues": issues, "human_review_required": any(i["human_required"] for i in issues),
            "quality_decision_ids": journey["quality_decision_ids"],
            "recommended_actions": [s["action"] for s in journey["steps"]],
            "scoring_policy": {
                "positive_lexicon": sorted(POSITIVE), "negative_lexicon": sorted(NEGATIVE),
                "sentiment": "(positive matches - negative matches) / max(1, total matches)",
                "priority": "severity rank * 10 + max(0, -sentiment); human escalation first, id breaks ties",
                "severity_ranks": SEVERITY,
                "limitations": "Exact-token English lexicon; no negation, sarcasm, or certification claims"}}


def validate_state(state, stage):
    """One shared boundary validator rejects missing, stale or altered stage handoffs."""
    stages = ["guided", "journey", "sentiment"]
    require(stage in stages, "unknown stage")
    keys = ["schema_version", "status", "synthetic", "input", "telemetry"] + stages[:stages.index(stage) + 1]
    fields(state, keys, "pipeline state")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1
            and state["status"] == "ok" and state["synthetic"] is True, "invalid state envelope")
    readings = validate_input(state["input"])
    require(state["telemetry"] == readings, "altered telemetry handoff")
    guided = _guided(state["input"])
    require(state["guided"] == guided, "invalid guided handoff")
    if stage != "guided":
        journey = _journey(state["input"], guided, readings)
        require(state["journey"] == journey, "invalid journey handoff")
    if stage == "sentiment":
        require(state["sentiment"] == _sentiment(journey), "invalid sentiment result")
    return copy.deepcopy(state)


def guided_setup(payload):
    readings = validate_input(payload)
    state = {"schema_version": 1, "status": "ok", "synthetic": True,
             "input": copy.deepcopy(payload), "telemetry": readings, "guided": _guided(payload)}
    return validate_state(state, "guided")


def recommend_journey(guided_state):
    state = validate_state(guided_state, "guided")
    state["journey"] = _journey(state["input"], state["guided"], state["telemetry"])
    return validate_state(state, "journey")


def prioritize_sentiment(journey_state):
    state = validate_state(journey_state, "journey")
    state["sentiment"] = _sentiment(state["journey"])
    return validate_state(state, "sentiment")


def run_pipeline(payload):
    return prioritize_sentiment(recommend_journey(guided_setup(payload)))


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
        with open(args[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object)
        result = run_pipeline(payload)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
