"""Synthetic manufacturing search -> onboarding reference; Python standard library.

Run: python -B implementation.py example_input.json
Optional embedding callable: callable(list[str]) -> list[list[finite float]].
It is injected only by Python callers; this module never contacts a provider.
"""

import copy
import csv
import io
import json
import math
import re
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def number(value, label):
    require(type(value) in (int, float), label + " must be a finite number")
    try:
        require(math.isfinite(value), label + " must be a finite number")
    except OverflowError as exc:
        raise ValidationError(label + " exceeds numeric range") from exc
    return float(value)


def timestamp(value, label):
    text(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(label + " must be ISO 8601") from exc
    require(parsed.tzinfo is not None, label + " must include a timezone")
    return parsed


def obj(value, label):
    require(isinstance(value, dict), label + " must be an object")
    return value


def array(value, label, nonempty=False):
    require(isinstance(value, list), label + " must be an array")
    require(not nonempty or len(value) > 0, label + " cannot be empty")
    return value


METRICS = {"temperature": "C", "vibration": "mm/s", "diameter": "mm"}
SYNONYMS = {
    "hot": "temperature", "heat": "temperature", "thermal": "temperature",
    "shaking": "vibration", "vibrating": "vibration", "oscillation": "vibration",
    "check": "inspection", "inspect": "inspection", "quality": "inspection",
    "bearings": "bearing", "servicing": "maintenance", "service": "maintenance",
}
CSV_FIELDS = ["reading_id", "work_order_id", "asset_id", "timestamp", "metric",
              "value", "unit", "lower", "upper", "safety_critical"]


def terms(value):
    return [SYNONYMS.get(t, t) for t in re.findall(r"[a-z0-9]+", value.lower())]


def measurement(metric, value, unit, lower, upper):
    require(isinstance(metric, str) and metric in METRICS, "unknown measurement metric")
    require(unit == METRICS[metric], "unit does not match metric " + metric)
    value, lower, upper = [number(v, "measurement") for v in (value, lower, upper)]
    require(lower <= upper, "lower tolerance exceeds upper tolerance")
    require(metric == "temperature" or min(value, lower, upper) >= 0,
            "non-temperature measurements cannot be negative")
    require(metric != "temperature" or min(value, lower, upper) >= -273.15,
            "temperature below absolute zero")
    return lower <= value <= upper


def unique(items, label):
    require(len(items) == len(set(items)), "duplicate " + label)


def validate(data, stage="input", source=None):
    """Single boundary validator for input and both pipeline output stages."""
    obj(data, stage)
    require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    if stage != "input":
        require(stage in ("search", "onboarding"), "unknown validation stage")
        require(source is not None, "stage validation requires source")
        require(data.get("status") == "ok" and data.get("stage") == stage,
                "invalid stage envelope")
        require(data.get("customer") == source["customer"], "customer handoff changed")
        require(data.get("query") == source["query"], "query handoff changed")
        require(data.get("synthetic") is True, "synthetic label missing")
        if stage == "search":
            records = {r["record_id"]: r for r in source["records"]}
            hits = array(data.get("hits"), "hits")
            require(len(hits) <= source["limit"], "too many hits")
            ids = []
            for hit in hits:
                obj(hit, "hit")
                record = obj(hit.get("record"), "hit record")
                record_id = text(record.get("record_id"), "record_id")
                require(record_id in records and record == records[record_id],
                        "search record provenance changed")
                score = number(hit.get("score"), "score")
                require(0 < score <= 1, "score out of range")
                ids.append(record_id)
            unique(ids, "hit")
            require(hits == sorted(hits, key=lambda h: (-h["score"], h["record"]["record_id"])),
                    "hits must be ranked deterministically")
            require(data.get("alerts") == source["alerts"], "safety alerts lost")
            require(data.get("quality_decisions") == source["quality_decisions"],
                    "quality traceability lost")
        else:
            require(data.get("search") == source, "search handoff changed")
            require(data.get("next_step") == next_step(source), "invalid personalized next step")
            require(data.get("human_escalations") == source["alerts"], "human escalation lost")
        return copy.deepcopy(data)

    require(data.get("synthetic") is True, "only clearly labeled synthetic fixtures accepted")
    customer = obj(data.get("customer"), "customer")
    for key in ("customer_id", "name", "goal"):
        text(customer.get(key), "customer." + key)
    require(customer.get("role") in ("operator", "quality_engineer", "planner"), "unsupported role")
    require(customer.get("experience") in ("new", "experienced"), "unsupported experience")
    query = text(data.get("query"), "query")
    limit = data.get("limit", 5)
    require(type(limit) is int and 1 <= limit <= 20, "limit must be integer 1..20")
    records, orders, alerts, decisions = [], {}, [], []
    for order in array(data.get("work_orders"), "work_orders", True):
        obj(order, "work order")
        for key in ("work_order_id", "asset_id", "product", "description"):
            text(order.get(key), key)
        oid = order["work_order_id"]
        require(oid not in orders, "duplicate work_order_id")
        require(order.get("status") in ("planned", "active", "held", "complete"),
                "invalid work-order status")
        require(type(order.get("quantity")) is int and order["quantity"] > 0,
                "quantity must be a positive integer")
        logs = array(order.get("maintenance_logs"), "maintenance_logs")
        for log in logs:
            obj(log, "maintenance log")
            timestamp(log.get("timestamp"), "maintenance timestamp")
            text(log.get("technician"), "technician")
            text(log.get("note"), "maintenance note")
        orders[oid] = order
        records.append({"record_id": "work_order:" + oid, "kind": "work_order",
                        "work_order_id": oid, "asset_id": order["asset_id"],
                        "text": " ".join([order["product"], order["description"]] +
                                        [log["note"] for log in logs]),
                        "data": order})

    def link(entry):
        oid = text(entry.get("work_order_id"), "work_order_id")
        require(oid in orders, "unknown work-order reference")
        require(entry.get("asset_id") == orders[oid]["asset_id"], "asset/work-order mismatch")
        return oid

    csv_text = text(data.get("telemetry_csv"), "telemetry_csv")
    try:
        reader = csv.DictReader(io.StringIO(csv_text), strict=True)
        require(reader.fieldnames == CSV_FIELDS, "telemetry CSV header mismatch")
        rows = list(reader)
    except csv.Error as exc:
        raise ValidationError("invalid telemetry CSV") from exc
    reading_ids, last_times = [], {}
    for row in rows:
        require(set(row) == set(CSV_FIELDS) and all(v is not None for v in row.values()),
                "telemetry row has missing or extra columns")
        rid = text(row["reading_id"], "reading_id")
        reading_ids.append(rid)
        oid = link(row)
        instant = timestamp(row["timestamp"], "telemetry timestamp")
        key = (row["asset_id"], row["metric"])
        require(key not in last_times or instant >= last_times[key],
                "telemetry must be chronological per asset and metric")
        last_times[key] = instant
        try:
            for field in ("value", "lower", "upper"):
                row[field] = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValidationError("telemetry measurement must be numeric") from exc
        require(row["safety_critical"] in ("true", "false"), "invalid safety_critical boolean")
        row["safety_critical"] = row["safety_critical"] == "true"
        within = measurement(*(row[k] for k in ("metric", "value", "unit", "lower", "upper")))
        record_id = "sensor:" + rid
        records.append({"record_id": record_id, "kind": "sensor",
                        "work_order_id": oid, "asset_id": row["asset_id"],
                        "text": row["metric"] + " telemetry " + orders[oid]["product"],
                        "data": dict(row)})
        if not within and row["safety_critical"]:
            alerts.append({"record_id": record_id, "work_order_id": oid,
                           "asset_id": row["asset_id"], "severity": "safety_critical",
                           "route": "human_safety_supervisor", "reason": "outside validated tolerance",
                           "evidence": dict(row)})
    unique(reading_ids, "reading_id")

    inspection_ids = []
    for inspection in array(data.get("quality_inspections"), "quality_inspections"):
        obj(inspection, "inspection")
        iid = text(inspection.get("inspection_id"), "inspection_id")
        inspection_ids.append(iid)
        oid = link(inspection)
        text(inspection.get("inspector"), "inspector")
        timestamp(inspection.get("timestamp"), "inspection timestamp")
        text(inspection.get("rationale"), "quality decision rationale")
        require(type(inspection.get("safety_critical")) is bool, "inspection safety_critical required")
        within = measurement(*(inspection.get(k) for k in
                               ("metric", "value", "unit", "lower", "upper")))
        require(inspection.get("decision") == ("pass" if within else "fail"),
                "quality decision disagrees with measurement evidence")
        record_id = "inspection:" + iid
        records.append({"record_id": record_id, "kind": "quality_inspection",
                        "work_order_id": oid, "asset_id": inspection["asset_id"],
                        "text": "quality inspection " + inspection["metric"] + " " +
                                inspection["rationale"] + " " + orders[oid]["product"],
                        "data": inspection})
        decisions.append({"record_id": record_id, "work_order_id": oid,
                          "asset_id": inspection["asset_id"], "inspector": inspection["inspector"],
                          "timestamp": inspection["timestamp"], "decision": inspection["decision"],
                          "rationale": inspection["rationale"], "evidence": inspection})
        if not within and inspection["safety_critical"]:
            alerts.append({"record_id": record_id, "work_order_id": oid,
                           "asset_id": inspection["asset_id"], "severity": "safety_critical",
                           "route": "human_safety_supervisor", "reason": "failed safety-critical inspection",
                           "evidence": inspection})
    unique(inspection_ids, "inspection_id")
    return {"schema_version": 1, "synthetic": True, "customer": customer, "query": query,
            "limit": limit, "records": records, "alerts": alerts, "quality_decisions": decisions}


class SearchIndex:
    """Local concept-normalized inverted index with optional injected embeddings."""

    def __init__(self, records):
        self.records = records
        self.postings = {}
        for position, record in enumerate(records):
            for term in set(terms(record["text"])):
                self.postings.setdefault(term, set()).add(position)

    def rank(self, query, embedding=None):
        query_terms = set(terms(query))
        scores = [0.0] * len(self.records)
        weights = {t: 1 + math.log((1 + len(scores)) /
                                  (1 + len(self.postings.get(t, ())))) for t in query_terms}
        total = sum(weights.values())
        for term, weight in weights.items():
            for position in self.postings.get(term, ()):
                scores[position] += weight / total
        if embedding is not None:
            require(callable(embedding), "embedding must be callable")
            texts = [query] + [r["text"] for r in self.records]
            try:
                vectors = embedding(texts)
            except Exception as exc:
                raise ValidationError("embedding callable failed") from exc
            array(vectors, "embedding output", True)
            require(len(vectors) == len(texts), "embedding output count mismatch")
            dimension = None
            normalized = []
            for vector in vectors:
                array(vector, "embedding vector", True)
                if dimension is None:
                    dimension = len(vector)
                require(len(vector) == dimension, "embedding dimension mismatch")
                vector = [number(v, "embedding component") for v in vector]
                norm = math.hypot(*vector)
                require(math.isfinite(norm) and norm > 0, "embedding must have finite nonzero norm")
                normalized.append([v / norm for v in vector])
            for i in range(len(scores)):
                cosine = sum(a * b for a, b in zip(normalized[0], normalized[i + 1]))
                scores[i] = 0.65 * scores[i] + 0.35 * max(0.0, min(1.0, cosine))
        hits = [{"record": record, "score": round(min(1.0, score), 8)}
                for record, score in zip(self.records, scores) if round(score, 8) > 0]
        return sorted(hits, key=lambda h: (-h["score"], h["record"]["record_id"]))


def search(source, embedding=None):
    result = {"schema_version": 1, "status": "ok", "stage": "search", "synthetic": True,
              "customer": source["customer"], "query": source["query"],
              "hits": SearchIndex(source["records"]).rank(source["query"], embedding)[:source["limit"]],
              "alerts": source["alerts"], "quality_decisions": source["quality_decisions"]}
    return validate(result, "search", source)


def next_step(source):
    customer = source["customer"]
    base = {"customer_id": customer["customer_id"], "goal": customer["goal"],
            "guidance_level": "guided" if customer["experience"] == "new" else "concise"}
    if source["alerts"]:
        alert = source["alerts"][0]
        base.update({"action": "contact_human_safety_supervisor",
                     "instruction": customer["name"] + ", request human safety review before proceeding.",
                     "owner": "human_safety_supervisor", "record_id": alert["record_id"],
                     "work_order_id": alert["work_order_id"], "requires_human": True})
    elif source["hits"]:
        record = source["hits"][0]["record"]
        action = {"operator": "review_work_order_and_safety_checklist",
                  "quality_engineer": "review_inspection_evidence",
                  "planner": "review_work_order_schedule"}[customer["role"]]
        base.update({"action": action, "instruction": customer["name"] + ", " +
                     action.replace("_", " ") + " for " + record["work_order_id"] +
                     " to support your goal: " + customer["goal"] + ".",
                     "owner": customer["role"], "record_id": record["record_id"],
                     "work_order_id": record["work_order_id"], "requires_human": False})
    else:
        base.update({"action": "refine_search", "instruction": customer["name"] +
                     ", search by product, measurement, or maintenance concern to support: " +
                     customer["goal"] + ".", "owner": customer["role"], "record_id": None,
                     "work_order_id": None, "requires_human": False})
    return base


def onboard(search_result):
    result = {"schema_version": 1, "status": "ok", "stage": "onboarding", "synthetic": True,
              "customer": search_result["customer"], "query": search_result["query"],
              "search": search_result, "next_step": next_step(search_result),
              "human_escalations": search_result["alerts"]}
    return validate(result, "onboarding", search_result)


def pipeline(data, embedding=None):
    # Copy at the boundary so an injected callable cannot mutate caller-owned evidence.
    try:
        data = json.loads(json.dumps(data, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValidationError("input must contain finite JSON values") from exc
    return onboard(search(validate(data), embedding))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle)
        result = pipeline(data)
    except (ValidationError, OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
