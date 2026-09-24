"""Synthetic manufacturing search reference; not ISO certification or a safety system.

run(payload, embed=None) is the shared entry point. An optional local callable receives
[query, *document_texts] and returns one finite, nonzero, same-width vector per text.
No provider is selected, imported, or contacted by this implementation.
"""

import csv
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


class ValidationError(ValueError):
    pass


METRICS = {
    "temperature": ("C", -50.0, 500.0),
    "vibration": ("mm/s", 0.0, 100.0),
    "pressure": ("bar", 0.0, 1000.0),
    "diameter": ("mm", 0.001, 10000.0),
}
CSV_FIELDS = [
    "id", "timestamp", "asset_id", "work_order_id", "metric", "value",
    "unit", "safety_critical",
]
SYNONYMS = {
    "hot": "temperature", "heat": "temperature", "thermal": "temperature",
    "overheating": "temperature", "shaking": "vibration", "oscillation": "vibration",
    "vibrating": "vibration", "defect": "failure", "defective": "failure",
    "rejected": "failure", "reject": "failure", "fail": "failure",
    "failed": "failure", "repair": "maintenance", "serviced": "maintenance",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(keys), location + " has missing or unknown fields")


def text(value, location, limit=2000):
    require(isinstance(value, str) and bool(value.strip()), location + " must be nonblank text")
    require(len(value) <= limit, location + " is too long")
    return value.strip()


def number(value, location):
    require(type(value) in (int, float), location + " must be a number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValidationError(location + " is not a finite number") from None
    require(math.isfinite(result), location + " must be finite")
    return result


def timestamp(value, location):
    text(value, location, 60)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError(location + " must be an ISO 8601 timestamp") from None
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            location + " must include a timezone")
    return parsed


def sequence(value, location, maximum):
    require(isinstance(value, list) and len(value) <= maximum,
            location + " must be a bounded list")


def metric_value(metric, unit, value, location):
    require(isinstance(metric, str) and metric in METRICS, location + ": unsupported metric")
    expected, lower, upper = METRICS[metric]
    require(unit == expected, location + ": expected unit " + expected)
    result = number(value, location)
    require(lower <= result <= upper, location + ": outside demonstrative physical bounds")
    return result


def tokens(value):
    result = []
    for word in re.findall(r"[a-z0-9]+", value.lower()):
        if len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        result.append(SYNONYMS.get(word, word))
    return result


def validate_and_normalize(payload):
    """All source formats converge on one document envelope and alert schema."""
    obj(payload, ["schema_version", "synthetic", "fixture_label", "work_orders",
                  "sensor_csv", "quality_inspections", "search"], "input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version must be 1")
    require(payload["synthetic"] is True, "synthetic must be true for this reference")
    text(payload["fixture_label"], "fixture_label")
    obj(payload["search"], ["query", "top_k"], "search")
    query = text(payload["search"]["query"], "search.query", 500)
    require(bool(tokens(query)), "query must contain searchable letters or numbers")
    top_k = payload["search"]["top_k"]
    require(type(top_k) is int and 1 <= top_k <= 100, "top_k must be an integer from 1 to 100")
    sequence(payload["work_orders"], "work_orders", 1000)
    sequence(payload["quality_inspections"], "quality_inspections", 1000)
    require(isinstance(payload["sensor_csv"], str) and len(payload["sensor_csv"]) <= 2000000,
            "sensor_csv must be text of at most 2000000 characters")

    orders, documents, alerts, ids = {}, [], [], set()

    def identity(record_id):
        text(record_id, "record id", 100)
        require(record_id not in ids, "duplicate record id: " + record_id)
        ids.add(record_id)

    def document(record_id, kind, order, content, traceability):
        documents.append({
            "id": record_id, "entity_type": kind, "work_order_id": order["id"],
            "asset_id": order["asset_id"], "synthetic": True, "text": content,
            "traceability": traceability,
        })

    def order_for(record):
        work_order_id = text(record["work_order_id"], "work_order_id", 100)
        require(work_order_id in orders, "unknown work_order_id: " + work_order_id)
        order = orders[work_order_id]
        require(record["asset_id"] == order["asset_id"], "asset/work-order mismatch")
        return order

    def measurement(record, order, location):
        metric = record["metric"]
        value = metric_value(metric, record["unit"], record["value"], location)
        require(metric in order["tolerances"], location + ": missing work-order tolerance")
        tolerance = order["tolerances"][metric]
        outside = not tolerance["min"] <= value <= tolerance["max"]
        return outside

    for order in payload["work_orders"]:
        obj(order, ["id", "asset_id", "product", "description", "status",
                    "tolerances", "maintenance_logs"], "work_order")
        identity(order["id"])
        for field in ("asset_id", "product", "description"):
            text(order[field], "work_order." + field)
        require(order["status"] in ("planned", "running", "complete", "held"), "invalid work-order status")
        require(isinstance(order["tolerances"], dict) and bool(order["tolerances"]),
                "tolerances must be a nonempty object")
        for metric, tolerance in order["tolerances"].items():
            obj(tolerance, ["unit", "min", "max"], "tolerance")
            lower = metric_value(metric, tolerance["unit"], tolerance["min"], "tolerance.min")
            upper = metric_value(metric, tolerance["unit"], tolerance["max"], "tolerance.max")
            require(lower <= upper, "tolerance.min must not exceed tolerance.max")
        sequence(order["maintenance_logs"], "maintenance_logs", 100)
        log_text = []
        for log in order["maintenance_logs"]:
            obj(log, ["timestamp", "technician", "note"], "maintenance_log")
            timestamp(log["timestamp"], "maintenance_log.timestamp")
            text(log["technician"], "maintenance_log.technician")
            log_text.append(text(log["note"], "maintenance_log.note"))
        orders[order["id"]] = order
        content = " ".join([order["id"], order["asset_id"], order["product"],
                            order["description"], order["status"], *log_text])
        document(order["id"], "work_order", order, content,
                 {"source_format": "ERP/MES JSON", "record_id": order["id"],
                  "tolerances": order["tolerances"], "maintenance_logs": order["maintenance_logs"]})

    try:
        reader = csv.DictReader(io.StringIO(payload["sensor_csv"]), strict=True)
        require(reader.fieldnames == CSV_FIELDS, "sensor_csv must have the exact documented header")
        for row_number, row in enumerate(reader, 2):
            require(row_number <= 10001, "sensor_csv exceeds 10000 readings")
            require(set(row) == set(CSV_FIELDS) and all(v is not None for v in row.values()),
                    "sensor_csv contains a row with the wrong number of columns")
            identity(row["id"])
            timestamp(row["timestamp"], "sensor.timestamp")
            order = order_for(row)
            try:
                row["value"] = float(row["value"])
            except ValueError:
                raise ValidationError("sensor.value must be numeric") from None
            outside = measurement(row, order, "sensor measurement")
            require(row["safety_critical"] in ("true", "false"), "safety_critical must be true or false")
            critical = row["safety_critical"] == "true"
            if critical or outside:
                alerts.append({
                    "record_id": row["id"], "work_order_id": order["id"], "asset_id": order["asset_id"],
                    "severity": "critical" if critical else "warning",
                    "reason": "safety-critical reading" if critical else "outside work-order tolerance",
                    "requires_human": True, "route": "human_safety_operator" if critical else "human_quality_reviewer",
                    "state": "pending_human_review", "automatic_action_taken": False,
                })
            content = " ".join(str(row[field]) for field in CSV_FIELDS)
            if critical:
                content += " safety critical alert human escalation"
            if outside:
                content += " outside tolerance failure"
            document(row["id"], "sensor_reading", order, content,
                     {"source_format": "time-series CSV", "record_id": row["id"],
                      "timestamp": row["timestamp"], "metric": row["metric"], "value": row["value"],
                      "unit": row["unit"], "tolerance": order["tolerances"][row["metric"]],
                      "outside_tolerance": outside, "safety_critical": critical})
    except csv.Error as exc:
        raise ValidationError("malformed sensor_csv: " + str(exc)) from None

    for inspection in payload["quality_inspections"]:
        obj(inspection, ["id", "work_order_id", "asset_id", "timestamp", "decision",
                         "decided_by", "rationale", "measurements"], "quality_inspection")
        identity(inspection["id"])
        timestamp(inspection["timestamp"], "inspection.timestamp")
        for field in ("decided_by", "rationale"):
            text(inspection[field], "inspection." + field)
        require(inspection["decision"] in ("pass", "fail", "rework"), "invalid quality decision")
        order = order_for(inspection)
        sequence(inspection["measurements"], "measurements", 100)
        require(bool(inspection["measurements"]), "quality decision requires measurement evidence")
        outside = False
        seen_metrics = set()
        for item in inspection["measurements"]:
            obj(item, ["metric", "value", "unit"], "inspection measurement")
            violation = measurement(item, order, "inspection measurement")
            require(item["metric"] not in seen_metrics, "duplicate inspection measurement metric")
            seen_metrics.add(item["metric"])
            outside = outside or violation
        require(not (outside and inspection["decision"] == "pass"),
                "pass decision contradicts out-of-tolerance measurement evidence")
        trace = {
            "source_format": "quality inspection JSON", "record_id": inspection["id"],
            "timestamp": inspection["timestamp"], "decided_by": inspection["decided_by"],
            "decision": inspection["decision"], "rationale": inspection["rationale"],
            "measurements": inspection["measurements"], "tolerances": order["tolerances"],
            "outside_tolerance": outside,
        }
        content = " ".join([
            inspection["id"], order["id"], order["asset_id"], "quality inspection",
            inspection["decision"], inspection["rationale"], inspection["decided_by"],
            json.dumps(inspection["measurements"], sort_keys=True),
        ])
        document(inspection["id"], "quality_inspection", order, content, trace)
    documents.sort(key=lambda doc: doc["id"])
    alerts.sort(key=lambda alert: alert["record_id"])
    return query, top_k, documents, alerts


class SearchIndex:
    """Inverted index with BM25 scoring and a small explicit domain synonym map."""

    def __init__(self, documents):
        self.documents = documents
        self.postings = defaultdict(dict)
        self.lengths = []
        for position, document in enumerate(documents):
            counts = Counter(tokens(document["text"]))
            self.lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self.postings[token][position] = count
        self.average_length = sum(self.lengths) / len(documents) if documents else 1.0

    def lexical_scores(self, query):
        scores = [0.0] * len(self.documents)
        for token in sorted(set(tokens(query))):
            postings = self.postings.get(token, {})
            idf = math.log(1 + (len(scores) - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings.items():
                denominator = frequency + 1.2 * (0.25 + 0.75 * self.lengths[position] / self.average_length)
                scores[position] += idf * frequency * 2.2 / denominator
        return [score / (1 + score) for score in scores]


def embedding_scores(embed, query, documents):
    require(callable(embed), "embed must be a callable")
    try:
        vectors = embed([query] + [document["text"] for document in documents])
    except Exception as exc:
        raise ValidationError("embedding callable failed: " + type(exc).__name__) from None
    require(isinstance(vectors, (list, tuple)) and len(vectors) == len(documents) + 1,
            "embedding must return one vector per supplied text")
    normalized, width = [], None
    for vector in vectors:
        require(isinstance(vector, (list, tuple)) and 1 <= len(vector) <= 4096,
                "embedding vectors must be nonempty lists or tuples (maximum 4096 dimensions)")
        width = len(vector) if width is None else width
        require(len(vector) == width, "embedding dimensions must match")
        values = [number(value, "embedding component") for value in vector]
        scale = max(abs(value) for value in values)
        require(scale > 0, "embedding vectors must have nonzero norm")
        scaled = [value / scale for value in values]
        norm = math.sqrt(sum(value * value for value in scaled))
        normalized.append([value / norm for value in scaled])
    return [max(0.0, min(1.0, sum(a * b for a, b in zip(normalized[0], vector))))
            for vector in normalized[1:]]


def run(payload, embed=None):
    query, top_k, documents, alerts = validate_and_normalize(payload)
    index = SearchIndex(documents)
    lexical = index.lexical_scores(query)
    semantic = embedding_scores(embed, query, documents) if embed is not None else None
    ranked = []
    for position, document in enumerate(documents):
        score = lexical[position]
        if semantic is not None:
            score = 0.7 * score + 0.3 * semantic[position]
        if score > 0:
            ranked.append((score, document))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    return {
        "status": "ok", "schema_version": 1, "synthetic": True,
        "fixture_label": payload["fixture_label"], "query": query,
        "results": [{**document, "score": round(score, 8)} for score, document in ranked[:top_k]],
        "alerts": alerts,
        "index": {"document_count": len(documents), "term_count": len(index.postings),
                  "matched_count": len(ranked),
                  "ranking": "BM25+synonyms+injected_cosine" if semantic is not None else "BM25+synonyms"},
        "constraint_notice": "Demonstrative traceability and safety routing only; not compliance certification.",
    }


def reject_constant(value):
    raise ValidationError("nonstandard JSON numeric constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with Path(argv[0]).open(encoding="utf-8") as source:
            payload = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run(payload)
    except (ValidationError, OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
