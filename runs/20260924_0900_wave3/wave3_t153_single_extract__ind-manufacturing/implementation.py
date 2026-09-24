"""Synthetic manufacturing extraction reference; standard library, no providers.

Input: schema_version=1, synthetic=true, documents containing work_order (JSON
text), telemetry (single-line-record CSV text), inspection (label: value text).
Output fields retain typed values and end-exclusive Unicode character spans
into the original document strings. JSON and CSV spans include quoting.
Missing/blank required fields produce status=incomplete, not invented values.
Malformed or contradictory supplied values produce a validation error.
"""

import csv
import json
import math
import re
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


# This shared schema drives extraction, typing, and missing-field reporting.
SCHEMA = {
    "work_order": {
        "work_order_id": "str", "asset_id": "str", "part_id": "str",
        "unit": "str", "target": "number", "tolerance": "number",
        "maintenance_log": "str",
    },
    "telemetry": {
        "timestamp": "time", "asset_id": "str", "temperature": "number",
        "temperature_unit": "str", "vibration": "number",
        "vibration_unit": "str",
    },
    "inspection": {
        "inspection_id": "str", "work_order_id": "str", "asset_id": "str",
        "part_id": "str", "timestamp": "time", "inspector_id": "str",
        "measured": "number", "unit": "str", "decision": "str",
        "decision_reason": "str",
    },
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def bad_constant(value):
        raise ValidationError("Non-finite JSON number: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def json_fields(text):
    """A flat ERP/MES object is intentional; nested data is out of scope."""
    data = strict_json(text)
    require(isinstance(data, dict), "work_order must contain a JSON object")
    require(set(data) <= set(SCHEMA["work_order"]), "Unknown work-order field")
    decoder = json.JSONDecoder()
    spans = {}
    pos = text.index("{") + 1
    for _ in data:
        while text[pos].isspace() or text[pos] == ",":
            pos += 1
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace():
            pos += 1
        pos += 1  # The full strict parse has already checked the colon.
        while text[pos].isspace():
            pos += 1
        start = pos
        _, pos = decoder.raw_decode(text, pos)
        spans[key] = (data[key], start, pos)
    return [spans]


def csv_fields(text):
    lines = text.splitlines(keepends=True)
    require(bool(lines), "Telemetry CSV needs a header")
    rows = []
    offset = 0
    header = None
    for line in lines:
        raw = line.rstrip("\r\n")
        require(bool(raw), "Blank CSV records are not supported")
        try:
            values = next(csv.reader([raw], strict=True))
        except csv.Error as exc:
            raise ValidationError("Malformed single-line CSV record") from exc
        # Preserve raw field bounds, including CSV quotes and escaped quotes.
        bounds = []
        start = 0
        while True:
            match = re.match(r'(?:"(?:[^"]|"")*"|[^",\r\n]*)(?=,|$)', raw[start:])
            require(match is not None, "Malformed CSV quoting")
            end = start + match.end()
            bounds.append((offset + start, offset + end))
            if end == len(raw):
                break
            start = end + 1
        require(len(values) == len(bounds), "Invalid CSV field boundaries")
        if header is None:
            header = values
            require(len(set(header)) == len(header), "Duplicate CSV header")
            require(set(header) <= set(SCHEMA["telemetry"]), "Unknown telemetry column")
        else:
            require(len(values) == len(header), "CSV row width mismatch")
            rows.append({key: (value, *bound)
                         for key, value, bound in zip(header, values, bounds)})
        offset += len(line)
    return rows


def inspection_fields(text):
    result = {}
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        if not raw.strip():
            offset += len(line)
            continue
        match = re.fullmatch(r"([a-z_]+):[ \t]*(.*?)[ \t]*", raw)
        require(match is not None, "Inspection must use field: value lines")
        key, value = match.group(1, 2)
        require(key in SCHEMA["inspection"], "Unknown inspection field: " + key)
        require(key not in result, "Duplicate inspection field: " + key)
        result[key] = (value, offset + match.start(2), offset + match.end(2))
        offset += len(line)
    return [result]


def typed(value, kind, location):
    if kind == "number":
        require(not isinstance(value, bool) and isinstance(value, (str, int, float)),
                location + " must be numeric")
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValidationError(location + " must be numeric") from exc
        require(math.isfinite(number), location + " must be finite")
        return number
    require(isinstance(value, str), location + " must be a string")
    require(value == value.strip(), location + " cannot have surrounding whitespace")
    if kind == "time":
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(location + " must be an ISO timestamp") from exc
        require(parsed.tzinfo is not None, location + " requires timezone")
    return value


def extract(payload):
    require(isinstance(payload, dict), "Input must be an object")
    require(set(payload) == {"schema_version", "synthetic", "documents"},
            "Input requires exactly schema_version, synthetic, documents")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "Unsupported schema_version")
    require(payload["synthetic"] is True, "Fixtures must be labeled synthetic=true")
    docs = payload["documents"]
    require(isinstance(docs, dict) and set(docs) == set(SCHEMA),
            "documents requires work_order, telemetry, inspection")
    require(all(isinstance(v, str) for v in docs.values()), "Documents must be strings")
    entities, missing = {}, []
    parsers = {"work_order": json_fields, "telemetry": csv_fields,
               "inspection": inspection_fields}
    for entity, schema in SCHEMA.items():
        entities[entity] = []
        raw_rows = parsers[entity](docs[entity])
        if not raw_rows:
            missing.append({"entity": entity, "record": None, "field": "*"})
        for index, raw in enumerate(raw_rows):
            row = {}
            for field, kind in schema.items():
                if field not in raw or raw[field][0] is None or raw[field][0] == "":
                    missing.append({"entity": entity, "record": index, "field": field})
                    continue
                value, start, end = raw[field]
                row[field] = {
                    "value": typed(value, kind, f"{entity}[{index}].{field}"),
                    "source_span": {"document": entity, "start": start, "end": end},
                }
            entities[entity].append(row)
    alerts, quality = validate_entities(entities)
    return {"status": "incomplete" if missing else "ok", "schema_version": 1,
            "synthetic": True, "entities": entities, "missing_fields": missing,
            "alerts": alerts, "quality_trace": quality,
            "validation_note": "Demonstrative traceability rules; not ISO 9001 certification."}


def validate_entities(entities):
    """Shared semantic validation for all three extracted entity types."""
    values = {name: [{k: v["value"] for k, v in row.items()} for row in rows]
              for name, rows in entities.items()}
    wo, inspection = values["work_order"][0], values["inspection"][0]
    for field in ("asset_id", "work_order_id", "part_id"):
        if field in wo and field in inspection:
            require(wo[field] == inspection[field], "Inspection traceability mismatch: " + field)
    for record in (wo, inspection):
        if "unit" in record:
            require(record["unit"] == "mm", "Dimensions must use mm")
    if "target" in wo:
        require(0 < wo["target"] <= 10000, "Target outside supported physical range")
    if "tolerance" in wo:
        require(0 <= wo["tolerance"] <= 100, "Tolerance must be between 0 and 100 mm")
    if "target" in wo and "tolerance" in wo:
        require(wo["tolerance"] < wo["target"], "Tolerance must be smaller than target")
    if "measured" in inspection:
        require(0 < inspection["measured"] <= 10000, "Measured dimension outside physical range")
    if "decision" in inspection:
        require(inspection["decision"] in ("pass", "fail"), "Decision must be pass or fail")
    expected = None
    if all(k in wo for k in ("target", "tolerance")) and "measured" in inspection:
        # Decimal arithmetic avoids binary floating-point boundary misclassification.
        from decimal import Decimal
        difference = abs(Decimal(str(inspection["measured"])) - Decimal(str(wo["target"])))
        expected = "pass" if difference <= Decimal(str(wo["tolerance"])) else "fail"
        if "decision" in inspection:
            require(inspection["decision"] == expected, "Quality decision contradicts tolerance")
    alerts = []
    last_time = None
    for index, row in enumerate(values["telemetry"]):
        if "asset_id" in row and "asset_id" in wo:
            require(row["asset_id"] == wo["asset_id"], "Telemetry asset mismatch")
        for field, unit in (("temperature_unit", "degC"), ("vibration_unit", "mm/s")):
            if field in row:
                require(row[field] == unit, "Unsupported unit: " + field)
        if "temperature" in row:
            require(-40 <= row["temperature"] <= 250, "Temperature outside physical range")
        if "vibration" in row:
            require(0 <= row["vibration"] <= 100, "Vibration outside physical range")
        if "timestamp" in row:
            time = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            require(last_time is None or time > last_time, "Telemetry timestamps must increase")
            last_time = time
        reasons = []
        if row.get("temperature", -40) >= 90:
            reasons.append("temperature_at_or_above_90_degC")
        if row.get("vibration", 0) > 8:
            reasons.append("vibration_above_8_mm_per_s")
        if reasons:
            alerts.append({"severity": "safety-critical", "record": index,
                           "reasons": reasons, "action": "escalate_to_human",
                           "human_review_required": True, "recipient": "plant_safety_operator",
                           "delivery_status": "pending_external_delivery",
                           "source_fields": [f"telemetry[{index}].{field}" for field in
                                             ("temperature", "vibration") if field in row]})
    complete = (set(inspection) == set(SCHEMA["inspection"])
                and set(wo) == set(SCHEMA["work_order"]))
    trace = {"complete": complete, "expected_decision": expected,
             "decision": inspection.get("decision"),
             "inspection_id": inspection.get("inspection_id"),
             "work_order_id": inspection.get("work_order_id"),
             "asset_id": inspection.get("asset_id"), "part_id": inspection.get("part_id"),
             "inspector_id": inspection.get("inspector_id"),
             "timestamp": inspection.get("timestamp"),
             "decision_reason": inspection.get("decision_reason"),
             "evidence": ["work_order[0].target", "work_order[0].tolerance",
                          "inspection[0].measured"],
             "release_authorized": False}
    return alerts, trace


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as handle:
            payload = strict_json(handle.read())
        result = extract(payload)
        code = 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        result, code = {"status": "error", "error": str(exc)}, 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
