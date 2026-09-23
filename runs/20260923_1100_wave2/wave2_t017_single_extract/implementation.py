"""Deterministic extraction of labeled lines; offsets are Unicode code points.

Input: {"document": str, "schema": [{"name": str, "labels": [str],
        "type": "string"|"integer"|"number"|"date", "required": bool}]}.
Lines use a literal label followed by ':' or '='. First matching line wins.
Output always contains status; successful processing additionally returns fields,
missing_fields (including optional fields), and invalid_fields. Incomplete
required fields or any invalid typed values produce status "incomplete", exit 0.
Malformed input, schema, JSON, file access, or CLI usage produce "error", exit 2.
"""

import datetime
import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def validate(value, kind):
    """Shared validation boundary for request schemas and extracted values."""
    if kind == "request":
        if not isinstance(value, dict) or set(value) != {"document", "schema"}:
            raise ValidationError("Input must contain exactly document and schema")
        if not isinstance(value["document"], str):
            raise ValidationError("document must be a string")
        schema = value["schema"]
        if not isinstance(schema, list) or not schema:
            raise ValidationError("schema must be a nonempty list")
        names, labels = set(), set()
        for field in schema:
            if not isinstance(field, dict) or set(field) != {
                "name", "labels", "type", "required"
            }:
                raise ValidationError("Each field needs name, labels, type, required")
            name = field["name"]
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", name
            ) or name in names:
                raise ValidationError("Field names must be unique identifiers")
            names.add(name)
            if not isinstance(field["type"], str) or field["type"] not in {
                "string", "integer", "number", "date"
            }:
                raise ValidationError("Unsupported field type")
            if type(field["required"]) is not bool:
                raise ValidationError("required must be a boolean")
            if not isinstance(field["labels"], list) or not field["labels"]:
                raise ValidationError("labels must be a nonempty list")
            for label in field["labels"]:
                if (
                    not isinstance(label, str) or not label
                    or label != label.strip()
                    or any(c in label for c in "\r\n:=")
                ):
                    raise ValidationError("Labels must be nonblank single-line literals")
                if label.casefold() in labels:
                    raise ValidationError("Labels must be globally unique ignoring case")
                labels.add(label.casefold())
        return value
    if kind == "string":
        return value
    if kind == "integer":
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise ValidationError("Expected an integer")
        try:
            return int(value)
        except ValueError as exc:
            raise ValidationError("Integer exceeds runtime conversion limit") from exc
    if kind == "number":
        if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", value):
            raise ValidationError("Expected a finite decimal number")
        result = float(value)
        if not math.isfinite(result):
            raise ValidationError("Number exceeds finite float range")
        return result
    if kind == "date":
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValidationError("Expected an ISO YYYY-MM-DD date")
        try:
            return datetime.date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValidationError("Invalid calendar date") from exc
    raise ValidationError("Unsupported validation kind")


def extract(payload):
    payload = validate(payload, "request")
    document = payload["document"]
    fields, missing, invalid = {}, [], []
    # Keep line separators in the offset calculation; do not normalize source.
    lines = []
    offset = 0
    for raw_line in document.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        match = re.fullmatch(r"[ \t]*([^:=\r\n]+?)[ \t]*[:=](.*)", line)
        if match:
            text = match.group(2)
            start = offset + match.start(2) + len(text) - len(text.lstrip())
            source = text.strip()
            lines.append((match.group(1).strip().casefold(), source, start))
        offset += len(raw_line)
    for field in payload["schema"]:
        name = field["name"]
        labels = {label.casefold() for label in field["labels"]}
        found = next((line for line in lines if line[0] in labels), None)
        if found is None or not found[1]:
            fields[name] = None
            missing.append({"field": name, "required": field["required"]})
            continue
        _, source, start = found
        evidence = {"source_text": source, "span": {"start": start, "end": start + len(source)}}
        try:
            parsed = validate(source, field["type"])
        except ValidationError as exc:
            fields[name] = None
            invalid.append({"field": name, "message": str(exc), **evidence})
        else:
            fields[name] = {"value": parsed, **evidence}
    return {
        "status": "incomplete" if invalid or any(f["required"] for f in missing) else "ok",
        "fields": fields,
        "missing_fields": missing,
        "invalid_fields": invalid,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Nonstandard JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json")
        payload = json.loads(
            Path(args[0]).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        result = extract(payload)
    except (OSError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
