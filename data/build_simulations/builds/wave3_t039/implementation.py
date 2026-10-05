"""Synthetic telecom reference pipeline; standard library, no external services.

Offsets are half-open Unicode character offsets into the original document.
CSV spans include CSV quoting; extracted values are decoded and privacy-filtered.
Pseudonyms are correlation identifiers, NOT anonymization or certification.
"""
import csv
import hashlib
import io
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


SCHEMA_VERSION = "1.0"
SCHEMAS = {
    "subscriber_account": {
        "plan": ("Plan", "text", True),
        "phone": ("Phone", "private", False),
        "imei": ("IMEI", "private", False),
        "feedback": ("Feedback", "text", False),
    },
    "network_fault_ticket": {
        "ticket_id": ("Ticket", "text", True),
        "feedback": ("Feedback", "text", True),
        "billing_adjustment": ("Billing adjustment", "number", False),
        "adjustment_reason": ("Adjustment reason", "text", False),
    },
    "call_data_usage_record": {
        "duration_seconds": ("duration_seconds", "number", True),
        "data_mb": ("data_mb", "number", True),
        "feedback": ("feedback", "text", False),
    },
}
THEMES = {
    "network_reliability": (
        r"\b(drop\w*|outage\w*|signal|coverage|disconnect\w*|slow|latency)\b",
        "Review fault clusters and prioritize affected network coverage.",
    ),
    "billing": (
        r"\b(bill\w*|charg\w*|refund\w*|overcharg\w*)\b",
        "Audit disputed charges and explain any reasoned adjustments.",
    ),
    "support_experience": (
        r"\b(agent\w*|support|wait\w*|helpful|rude)\b",
        "Review support response times and coach recurring service issues.",
    ),
    "plan_value": (
        r"\b(plan\w*|expensive|price\w*|allowance|cost\w*)\b",
        "Review plan fit and explain available allowances and prices.",
    ),
    "high_usage": ("", "Offer an allowance review; do not automatically change plans."),
    "other": ("", "Manually review uncategorized feedback."),
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required schema keys")
    require(value.keys() <= set(required) | set(optional), "Unknown schema keys")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def protect(value):
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[REDACTED]", value)
    value = re.sub(r"(?<!\w)\+?\d[\d ().-]{5,}\d(?!\w)", "[REDACTED]", value)
    return value


def validate(value, stage):
    """One shared validation boundary for input, extraction, and insights."""
    if stage == "input":
        keys(value, ("schema_version", "synthetic", "processing_residency",
                     "purpose", "records"))
        require(value["schema_version"] == SCHEMA_VERSION, "Unsupported schema version")
        require(value["synthetic"] is True, "Only labeled synthetic inputs are supported")
        require(value["processing_residency"] in ("EU", "US"), "Unsupported residency")
        require(value["purpose"] == "support_analytics", "Unsupported processing purpose")
        require(isinstance(value["records"], list), "records must be an array")
        require(len(value["records"]) <= 1000, "Too many records")
        for record in value["records"]:
            keys(record, ("kind", "subscriber_id", "residency", "privacy", "document"))
            require(record["kind"] in SCHEMAS, "Unsupported record kind")
            require(text(record["subscriber_id"]), "Missing subscriber identifier")
            require(record["residency"] == value["processing_residency"],
                    "Record residency does not match processing residency")
            keys(record["privacy"], ("lawful_basis", "opt_out"))
            require(record["privacy"]["lawful_basis"] in ("consent", "legitimate_interests"),
                    "Unsupported lawful basis")
            require(type(record["privacy"]["opt_out"]) is bool, "opt_out must be boolean")
            keys(record["document"], ("format", "text"))
            expected = "csv" if record["kind"] == "call_data_usage_record" else "transcript"
            require(record["document"]["format"] == expected, "Incorrect document format")
            require(isinstance(record["document"]["text"], str), "Document text must be a string")
            require(len(record["document"]["text"]) <= 100000, "Document too large")
        return value
    if stage == "extraction":
        keys(value, ("schema_version", "synthetic", "processing_residency",
                     "records", "excluded"))
        require(value["schema_version"] == SCHEMA_VERSION and value["synthetic"] is True,
                "Invalid extraction envelope")
        require(value["processing_residency"] in ("EU", "US"), "Unsupported residency")
        require(isinstance(value["records"], list) and isinstance(value["excluded"], list),
                "Invalid extraction collections")
        ids = set()
        for record in value["records"]:
            keys(record, ("record_id", "subscriber_ref", "kind", "residency",
                          "source_length", "fields", "missing_fields"))
            require(text(record["record_id"]) and record["record_id"] not in ids,
                    "Duplicate or invalid record identifier")
            ids.add(record["record_id"])
            require(isinstance(record["subscriber_ref"], str) and
                    re.fullmatch(r"[a-f0-9]{24}", record["subscriber_ref"]) is not None,
                    "Invalid subscriber pseudonym")
            require(record["kind"] in SCHEMAS, "Unsupported extracted kind")
            require(record["residency"] == value["processing_residency"], "Residency changed")
            require(type(record["source_length"]) is int and record["source_length"] >= 0,
                    "Invalid source length")
            schema = SCHEMAS[record["kind"]]
            require(isinstance(record["fields"], dict) and record["fields"].keys() <= schema.keys(),
                    "Invalid extracted fields")
            for name, field in record["fields"].items():
                keys(field, ("value", "span", "protected"))
                span = field["span"]
                require(isinstance(span, list) and len(span) == 2 and
                        all(type(n) is int for n in span) and
                        0 <= span[0] < span[1] <= record["source_length"], "Invalid source span")
                require(type(field["protected"]) is bool, "Invalid protection marker")
                kind = schema[name][1]
                if kind == "number":
                    require(number(field["value"]), "Invalid extracted number")
                    if name in ("duration_seconds", "data_mb"):
                        require(field["value"] >= 0, "Usage cannot be negative")
                else:
                    require(text(field["value"]), "Invalid extracted text")
                    require(protect(field["value"]) == field["value"], "Unprotected personal data")
                    if kind == "private":
                        require(field["value"] == "[REDACTED]" and field["protected"],
                                "Private fields must be redacted")
            missing = sorted(name for name, spec in schema.items()
                             if spec[2] and name not in record["fields"])
            require(record["missing_fields"] == missing, "Inconsistent missing-field report")
            fields = record["fields"]
            require("billing_adjustment" not in fields or "adjustment_reason" in fields,
                    "Billing adjustments require a stated reason")
        for excluded in value["excluded"]:
            keys(excluded, ("record_id", "residency", "reason"))
            require(text(excluded["record_id"]) and excluded["record_id"] not in ids,
                    "Duplicate or invalid excluded identifier")
            ids.add(excluded["record_id"])
            require(excluded["residency"] == value["processing_residency"], "Residency changed")
            require(excluded["reason"] == "privacy_opt_out", "Unknown exclusion reason")
        return value
    if stage == "insights":
        keys(value, ("status", "schema_version", "synthetic", "extraction", "insights"))
        require(value["status"] == "ok" and value["schema_version"] == SCHEMA_VERSION and
                value["synthetic"] is True, "Invalid output envelope")
        validate(value["extraction"], "extraction")
        require(value["insights"] == summarize(value["extraction"]),
                "Insights do not match validated extraction")
        return value
    raise ValidationError("Unknown validation stage")


def csv_cells(source):
    """Locate raw cells, supporting quoted commas, doubled quotes, and newlines."""
    rows, row, start, index, quoted = [], [], 0, 0, False
    while index < len(source):
        char = source[index]
        if char == '"':
            if quoted and index + 1 < len(source) and source[index + 1] == '"':
                index += 2
                continue
            quoted = not quoted
        if not quoted and char in ",\r\n":
            row.append((start, index))
            if char != ",":
                rows.append(row)
                row = []
                if char == "\r" and index + 1 < len(source) and source[index + 1] == "\n":
                    index += 1
            start = index + 1
        index += 1
    require(not quoted, "Malformed CSV quoting")
    if start < len(source) or row:
        row.append((start, len(source)))
        rows.append(row)
    return rows


def raw_fields(record):
    source = record["document"]["text"]
    schema = SCHEMAS[record["kind"]]
    found = {}
    if record["document"]["format"] == "csv":
        try:
            rows = list(csv.reader(io.StringIO(source, newline=""), strict=True))
        except csv.Error:
            raise ValidationError("Malformed CSV") from None
        spans = csv_cells(source)
        require(len(rows) == 2 and len(spans) == 2, "CSV requires header and exactly one usage row")
        header, data = rows
        require(len(header) == len(set(header)) and set(header) <= schema.keys(),
                "Unknown or duplicate CSV columns")
        require(len(data) == len(header) == len(spans[1]), "CSV row width mismatch")
        for name, raw, span in zip(header, data, spans[1]):
            if raw.strip():
                found[name] = (raw.strip(), list(span))
    else:
        for name, (label, _, _) in schema.items():
            matches = list(re.finditer(r"^[ \t]*" + re.escape(label) +
                                      r":[ \t]*([^\r\n]*)", source, re.I | re.M))
            require(len(matches) <= 1, "Ambiguous duplicate transcript field")
            if matches:
                match = matches[0]
                raw = match.group(1)
                trimmed = raw.strip()
                if trimmed:
                    start = match.start(1) + len(raw) - len(raw.lstrip())
                    found[name] = (trimmed, [start, start + len(trimmed)])
    return found


def extract(payload):
    validate(payload, "input")
    result = {"schema_version": SCHEMA_VERSION, "synthetic": True,
              "processing_residency": payload["processing_residency"],
              "records": [], "excluded": []}
    for index, record in enumerate(payload["records"], 1):
        record_id = f"r{index:04d}"
        if record["privacy"]["opt_out"]:
            result["excluded"].append({"record_id": record_id, "residency": record["residency"],
                                       "reason": "privacy_opt_out"})
            continue
        schema = SCHEMAS[record["kind"]]
        fields = {}
        for name, (raw, span) in raw_fields(record).items():
            kind = schema[name][1]
            if kind == "number":
                try:
                    value = float(raw)
                except ValueError:
                    raise ValidationError("Invalid numeric field") from None
            else:
                value = "[REDACTED]" if kind == "private" else protect(raw)
            fields[name] = {"value": value, "span": span,
                            "protected": kind == "private" or (kind == "text" and value != raw)}
        result["records"].append({
            "record_id": record_id,
            "subscriber_ref": hashlib.sha256(("telecom-demo:" + record["subscriber_id"])
                                             .encode("utf-8")).hexdigest()[:24],
            "kind": record["kind"], "residency": record["residency"],
            "source_length": len(record["document"]["text"]), "fields": fields,
            "missing_fields": sorted(name for name, spec in schema.items()
                                     if spec[2] and name not in fields),
        })
    return validate(result, "extraction")


def summarize(extraction):
    buckets = {}
    for record in extraction["records"]:
        fields = record["fields"]
        feedback = fields.get("feedback")
        matches = []
        if feedback:
            matches = [name for name, (pattern, _) in THEMES.items()
                       if pattern and re.search(pattern, feedback["value"], re.I)]
            if not matches:
                matches = ["other"]
        for theme in matches:
            buckets.setdefault(theme, []).append({
                "record_id": record["record_id"], "residency": record["residency"],
                "field": "feedback", "span": feedback["span"], "excerpt": feedback["value"],
            })
        usage = fields.get("data_mb")
        if usage and usage["value"] >= 10000:
            buckets.setdefault("high_usage", []).append({
                "record_id": record["record_id"], "residency": record["residency"],
                "field": "data_mb", "span": usage["span"], "excerpt": usage["value"],
            })
    themes = [{"theme": name, "record_count": len(evidence), "action": THEMES[name][1],
               "evidence": evidence} for name, evidence in buckets.items()]
    themes.sort(key=lambda item: (-item["record_count"], item["theme"]))
    return {
        "record_count": len(extraction["records"]),
        "subscriber_count": len({r["subscriber_ref"] for r in extraction["records"]}),
        "excluded_count": len(extraction["excluded"]),
        "incomplete_record_count": sum(bool(r["missing_fields"]) for r in extraction["records"]),
        "themes": themes,
        "limitations": "Deterministic keyword themes, not sentiment or causal diagnosis; "
                      "counts are record-level and themes may overlap.",
    }


def insights(extraction):
    validate(extraction, "extraction")
    return validate({"status": "ok", "schema_version": SCHEMA_VERSION, "synthetic": True,
                     "extraction": extraction, "insights": summarize(extraction)}, "insights")


def run(payload):
    return insights(extract(payload))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object)
        result = run(payload)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        # Never echo source data or filenames into public error responses.
        print(json.dumps({"status": "error", "message": "Input file or schema validation failed"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
