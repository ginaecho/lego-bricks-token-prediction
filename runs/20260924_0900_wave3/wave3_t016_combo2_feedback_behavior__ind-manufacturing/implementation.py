"""Synthetic manufacturing feedback -> personalized work-order discovery.

Standard-library reference implementation; never controls equipment or certifies
compliance. All timestamps must be timezone-aware. Rankings are advisory.
"""

import csv
import io
import json
import math
import re
import sys
from datetime import datetime


THEMES = {
    "thermal": ("hot", "heat", "temperature", "overheat"),
    "vibration": ("vibration", "vibrating", "rattle"),
    "quality": ("quality", "defect", "inspection", "tolerance"),
    "maintenance": ("maintenance", "lubrication", "service"),
    "safety": ("safety", "danger", "injury", "smoke", "guard"),
}
UNITS = {"temperature": "C", "vibration": "mm/s", "pressure": "kPa"}
LIMITS = {"temperature": (-40, 250), "vibration": (0, 100), "pressure": (0, 2000)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite numeric")
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("invalid timestamp") from None
    require(result.tzinfo is not None, "timestamp needs timezone")
    return result


def records(value, label):
    require(isinstance(value, list), label + " must be a list")
    seen = set()
    for row in value:
        require(isinstance(row, dict), label + " entries must be objects")
        key = text(row.get("id"), label + ".id")
        require(key not in seen, label + " duplicate id")
        seen.add(key)
    return value


def normalized(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def themes_for(value):
    words = set(normalized(value).split())
    matches = sorted(k for k, terms in THEMES.items() if words.intersection(terms))
    return matches or ["general"]


def readings(data):
    reader = csv.DictReader(io.StringIO(data["sensor_csv"]), strict=True)
    expected = ["id", "work_order_id", "asset_id", "timestamp", "metric", "value", "unit"]
    require(reader.fieldnames == expected, "sensor CSV headers invalid")
    rows = list(reader)
    seen = set()
    for row in rows:
        require(set(row) == set(expected) and all(v is not None for v in row.values()),
                "malformed sensor CSV row")
        require(row["id"] and row["id"] not in seen, "sensor id missing or duplicated")
        seen.add(row["id"])
        try:
            row["value"] = float(row["value"])
        except (TypeError, ValueError):
            raise ValueError("invalid sensor numeric value") from None
    return rows


def validate(data, stage="input"):
    """One validation boundary shared by all stages."""
    require(isinstance(data, dict), "payload must be an object")
    if stage != "input":
        require(stage in ("feedback", "result"), "unknown stage")
        require(data.get("schema_version") == 1, "unsupported envelope schema")
        source = validate(data.get("input"))
        analysis = data.get("feedback")
        require(isinstance(analysis, dict), "feedback handoff missing")
        # Exact deterministic reconstruction prevents unsupported excerpts,
        # dropped provenance and forged priorities at the stage boundary.
        require(analysis == analyze_feedback(source), "feedback handoff inconsistent with input")
        if stage == "result":
            require(data.get("status") == "ok", "invalid result status")
            require(data.get("behavior") == rank_behavior(source, analysis),
                    "behavior output inconsistent with validated handoff")
        return data
    require(data.get("schema_version") == 1, "schema_version must be 1")
    require(data.get("synthetic") is True, "fixtures must be labeled synthetic")
    now = timestamp(data.get("as_of"))
    text(data.get("user_id"), "user_id")
    orders = records(data.get("work_orders"), "work_orders")
    require(bool(orders), "at least one work order required")
    order_map = {o["id"]: o for o in orders}
    for order in orders:
        text(order.get("asset_id"), "asset_id")
        text(order.get("description"), "description")
        require(order.get("status") in ("open", "closed"), "invalid work order status")
        tags = order.get("tags")
        require(isinstance(tags, list) and tags and all(t in THEMES for t in tags),
                "work order tags must be known themes")
        specs = order.get("tolerances")
        require(isinstance(specs, dict) and specs, "tolerances required")
        for metric, spec in specs.items():
            require(metric in UNITS and isinstance(spec, dict), "unsupported metric")
            require(spec.get("unit") == UNITS[metric], "invalid tolerance unit")
            low = number(spec.get("min"), "tolerance min")
            high = number(spec.get("max"), "tolerance max")
            safety = number(spec.get("safety_max"), "safety threshold")
            lo, hi = LIMITS[metric]
            require(lo <= low <= high <= safety <= hi, "invalid tolerance bounds")
    inspections = records(data.get("quality_inspections"), "quality_inspections")
    inspection_map = {i["id"]: i for i in inspections}
    for item in inspections:
        require(item.get("work_order_id") in order_map, "unknown inspection work order")
        order = order_map[item["work_order_id"]]
        require(item.get("asset_id") == order["asset_id"], "inspection asset mismatch")
        require(timestamp(item.get("timestamp")) <= now, "future inspection")
        text(item.get("inspector_id"), "inspector_id")
        text(item.get("decision_reason"), "decision_reason")
        metric = item.get("metric")
        require(metric in order["tolerances"], "inspection metric missing tolerance")
        spec = order["tolerances"][metric]
        require(item.get("unit") == spec["unit"], "inspection unit mismatch")
        value = number(item.get("value"), "inspection value")
        lo, hi = LIMITS[metric]
        require(lo <= value <= hi, "implausible inspection")
        decision = "pass" if spec["min"] <= value <= spec["max"] else "fail"
        require(item.get("decision") == decision, "quality decision contradicts tolerance")
    text(data.get("sensor_csv"), "sensor_csv")
    for row in readings(data):
        require(row["work_order_id"] in order_map, "unknown sensor work order")
        order = order_map[row["work_order_id"]]
        require(row["asset_id"] == order["asset_id"], "sensor asset mismatch")
        require(timestamp(row["timestamp"]) <= now, "future sensor reading")
        metric = row["metric"]
        require(metric in order["tolerances"], "sensor metric missing tolerance")
        require(row["unit"] == UNITS[metric], "sensor unit mismatch")
        value = number(row["value"], "sensor value")
        lo, hi = LIMITS[metric]
        require(lo <= value <= hi, "implausible sensor value")
    for feedback in records(data.get("feedback"), "feedback"):
        require(feedback.get("work_order_id") in order_map, "unknown feedback work order")
        require(feedback.get("inspection_id") in inspection_map, "feedback inspection missing")
        require(inspection_map[feedback["inspection_id"]]["work_order_id"] ==
                feedback["work_order_id"], "feedback inspection mismatch")
        text(feedback.get("text"), "feedback text")
        require(bool(normalized(feedback["text"])), "feedback needs words")
        require(timestamp(feedback.get("timestamp")) <= now, "future feedback")
        require(type(feedback.get("safety_critical")) is bool, "safety_critical must be boolean")
    for event in records(data.get("events"), "events"):
        text(event.get("user_id"), "event user_id")
        require(event.get("work_order_id") in order_map, "unknown event work order")
        require(event.get("type") in ("browse", "purchase"), "invalid event type")
        require(timestamp(event.get("timestamp")) <= now, "future event")
    return data


def analyze_feedback(data):
    groups = {}
    for item in sorted(data["feedback"], key=lambda f: f["id"]):
        key = (item["work_order_id"], normalized(item["text"]))
        groups.setdefault(key, []).append(item)
    summaries = []
    for (order_id, _), members in sorted(groups.items()):
        summaries.append({
            "id": members[0]["id"],
            "work_order_id": order_id,
            "themes": themes_for(members[0]["text"]),
            "safety_critical": any(f["safety_critical"] for f in members) or
                              "safety" in themes_for(members[0]["text"]),
            "evidence": [{"feedback_id": f["id"], "inspection_id": f["inspection_id"],
                          "excerpt": f["text"]} for f in members],
        })
    alerts = []
    for group in summaries:
        if group["safety_critical"]:
            alerts.append({"source_type": "feedback", "source_id": group["id"],
                           "work_order_id": group["work_order_id"],
                           "action": "human_review_required"})
    orders = {o["id"]: o for o in data["work_orders"]}
    for kind, rows in (("sensor", readings(data)), ("inspection", data["quality_inspections"])):
        for row in rows:
            spec = orders[row["work_order_id"]]["tolerances"][row["metric"]]
            if row["value"] >= spec["safety_max"]:
                alerts.append({"source_type": kind, "source_id": row["id"],
                               "work_order_id": row["work_order_id"],
                               "action": "human_review_required"})
    trace = []
    for item in sorted(data["quality_inspections"], key=lambda x: x["id"]):
        trace.append({key: item[key] for key in (
            "id", "work_order_id", "asset_id", "timestamp", "inspector_id",
            "metric", "value", "unit", "decision", "decision_reason")})
        trace[-1]["applied_tolerance"] = dict(
            orders[item["work_order_id"]]["tolerances"][item["metric"]])
    return {"deduplicated": summaries, "raw_count": len(data["feedback"]),
            "unique_count": len(summaries), "quality_decision_trace": trace,
            "human_escalations": sorted(alerts, key=lambda a: (a["source_type"], a["source_id"]))}


def feedback_stage(data):
    source = validate(data)
    return validate({"schema_version": 1, "input": source,
                     "feedback": analyze_feedback(source)}, "feedback")


def rank_behavior(data, feedback):
    now = timestamp(data["as_of"])
    history = [e for e in data["events"] if e["user_id"] == data["user_id"]]
    orders = {o["id"]: o for o in data["work_orders"]}
    profile = {}
    direct = {}
    for event in sorted(history, key=lambda e: e["id"]):
        age_days = (now - timestamp(event["timestamp"])).total_seconds() / 86400
        weight = (3.0 if event["type"] == "purchase" else 1.0) * 2 ** (-age_days / 30)
        oid = event["work_order_id"]
        direct[oid] = direct.get(oid, 0.0) + weight
        for tag in set(orders[oid]["tags"]):
            profile[tag] = profile.get(tag, 0.0) + weight
    rankings = []
    for order in data["work_orders"]:
        if order["status"] != "open":
            continue
        groups = [g for g in feedback["deduplicated"] if g["work_order_id"] == order["id"]]
        theme_set = set(order["tags"])
        for group in groups:
            theme_set.update(group["themes"])
        affinity = sum(profile.get(tag, 0) for tag in sorted(theme_set))
        feedback_priority = 0.25 * len(groups)
        score = direct.get(order["id"], 0) + affinity + feedback_priority
        review = any(a["work_order_id"] == order["id"] for a in feedback["human_escalations"])
        rankings.append({
            "work_order_id": order["id"], "asset_id": order["asset_id"],
            "score": round(score, 8), "direct_recency_score": round(direct.get(order["id"], 0), 8),
            "theme_affinity_score": round(affinity, 8),
            "feedback_priority": feedback_priority,
            "feedback_group_ids": [g["id"] for g in groups],
            "supporting_feedback_ids": [e["feedback_id"] for g in groups for e in g["evidence"]],
            "themes": sorted(theme_set), "human_review_required": review,
            "automatic_execution_allowed": False,
        })
    rankings.sort(key=lambda r: (-r["score"], r["work_order_id"]))
    return {"user_id": data["user_id"], "cold_start": not history, "half_life_days": 30,
            "mode": "advisory_work_order_discovery", "rankings": rankings,
            "human_escalations": feedback["human_escalations"]}


def behavior_stage(envelope):
    validate(envelope, "feedback")
    result = {**envelope, "status": "ok",
              "behavior": rank_behavior(envelope["input"], envelope["feedback"])}
    return validate(result, "result")


def pipeline(data):
    return behavior_stage(feedback_stage(data))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle)
        result = pipeline(data)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, csv.Error, OverflowError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
