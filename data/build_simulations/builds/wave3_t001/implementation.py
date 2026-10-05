"""Deterministic document extraction, reshaping, and validation.

Usage: python -B implementation.py example_input.json
Documents contain inline CSV text or JSON arrays of objects. Field rules select
source columns, rename them, normalize strings, convert types, and apply checks.
No files are written and no network or provider is used.
"""

import csv
import io
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Nonfinite JSON number: " + value)


def load_json(text):
    return json.loads(text, object_pairs_hook=strict_object,
                      parse_constant=reject_constant)


def keys(value, allowed, required, location):
    require(isinstance(value, dict), location + " must be an object")
    require(not set(value) - set(allowed), location + " has unknown keys")
    require(set(required) <= set(value), location + " has missing keys")


def finite_number(value):
    return type(value) in (int, float) and (
        type(value) is int or math.isfinite(value))


def validate_input(payload):
    keys(payload, ("schema_version", "documents", "fields"),
         ("schema_version", "documents", "fields"), "input")
    require(type(payload["schema_version"]) is int and
            payload["schema_version"] == 1, "schema_version must be 1")
    require(isinstance(payload["documents"], list), "documents must be an array")
    require(isinstance(payload["fields"], list) and payload["fields"],
            "fields must be a nonempty array")
    ids, targets = set(), set()
    for doc in payload["documents"]:
        keys(doc, ("id", "format", "content"), ("id", "format", "content"), "document")
        require(isinstance(doc["id"], str) and doc["id"].strip(), "Invalid document id")
        require(doc["id"] not in ids, "Duplicate document id")
        ids.add(doc["id"])
        require(doc["format"] in ("csv", "json"), "format must be csv or json")
        require(isinstance(doc["content"], str), "document content must be text")
    for field in payload["fields"]:
        keys(field, ("source", "target", "type", "required", "transforms",
                     "min", "max", "pattern", "choices"),
             ("source", "target", "type"), "field")
        for name in ("source", "target"):
            require(isinstance(field[name], str) and field[name].strip(),
                    name + " must be a nonempty string")
        require(field["target"] not in targets, "Duplicate target field")
        targets.add(field["target"])
        require(field["type"] in ("string", "integer", "number", "boolean"),
                "Unsupported field type")
        require(type(field.get("required", False)) is bool, "required must be boolean")
        transforms = field.get("transforms", [])
        require(isinstance(transforms, list), "transforms must be an array")
        require(all(t in ("strip", "lower", "upper") for t in transforms),
                "Unsupported transform")
        require(not transforms or field["type"] == "string",
                "transforms apply only to string fields")
        for bound in ("min", "max"):
            if bound in field:
                require(field["type"] in ("number", "integer") and
                        finite_number(field[bound]), "Invalid numeric bound")
        if "min" in field and "max" in field:
            require(field["min"] <= field["max"], "min exceeds max")
        if "pattern" in field:
            require(field["type"] == "string" and isinstance(field["pattern"], str),
                    "pattern requires a string field and regex string")
            try:
                re.compile(field["pattern"])
            except re.error as exc:
                raise ValidationError("Invalid pattern: " + str(exc)) from exc
        if "choices" in field:
            require(isinstance(field["choices"], list) and field["choices"],
                    "choices must be a nonempty array")
            for choice in field["choices"]:
                converted = convert(choice, {"type": field["type"]})
                require(type(converted) is type(choice) and converted == choice,
                        "choices must use the field's output type")


def extract(document):
    if document["format"] == "json":
        rows = load_json(document["content"])
        require(isinstance(rows, list), "JSON document must contain an array")
        require(all(isinstance(row, dict) for row in rows),
                "JSON document rows must be objects")
        return rows
    reader = csv.reader(io.StringIO(document["content"], newline=""), strict=True)
    header = next(reader, None)
    require(header is not None and header and all(h.strip() for h in header),
            "CSV requires nonempty column headers")
    require(len(set(header)) == len(header), "Duplicate CSV header")
    rows = []
    for cells in reader:
        if not cells:
            continue
        require(len(cells) == len(header), "CSV row width differs from header")
        rows.append(dict(zip(header, cells)))
    return rows


def convert(value, field):
    kind = field["type"]
    if kind == "string":
        require(isinstance(value, str), "Expected string")
        for transform in field.get("transforms", []):
            value = getattr(value, transform)()
    elif kind == "integer":
        require(type(value) is int or
                (isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip())),
                "Expected integer")
        value = int(value)
    elif kind == "number":
        require(type(value) in (int, float) or isinstance(value, str), "Expected number")
        try:
            value = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValidationError("Expected finite number") from exc
        require(math.isfinite(value), "Expected finite number")
    else:
        if isinstance(value, str):
            normalized = value.strip().lower()
            require(normalized in ("true", "false"), "Expected true or false")
            value = normalized == "true"
        require(type(value) is bool, "Expected boolean")
    if "min" in field:
        require(value >= field["min"], "Value is below minimum")
    if "max" in field:
        require(value <= field["max"], "Value exceeds maximum")
    if "pattern" in field:
        require(re.fullmatch(field["pattern"], value) is not None, "Pattern mismatch")
    if "choices" in field:
        require(value in field["choices"], "Value is not an allowed choice")
    return value


def empty_result():
    return {"schema_version": 1, "status": "ok", "records": [], "issues": [],
            "summary": {"documents": 0, "rows_seen": 0, "rows_emitted": 0,
                        "rows_rejected": 0}}


def automate(payload):
    result = empty_result()

    def issue(document, row, field, message):
        result["status"] = "error"
        result["issues"].append({"document_id": document, "row": row,
                                 "field": field, "message": message})

    try:
        validate_input(payload)
    except (ValidationError, ValueError, TypeError, OverflowError) as exc:
        issue(None, None, None, str(exc))
        return result
    result["summary"]["documents"] = len(payload["documents"])
    for document in payload["documents"]:
        try:
            rows = extract(document)
        except (ValueError, csv.Error) as exc:
            issue(document["id"], None, None, str(exc))
            continue
        for index, row in enumerate(rows, 1):
            result["summary"]["rows_seen"] += 1
            data = {}
            valid = True
            for field in payload["fields"]:
                try:
                    value = row.get(field["source"])
                    missing = value is None or (isinstance(value, str) and not value.strip())
                    require(not missing or not field.get("required", False),
                            "Required value is missing")
                    data[field["target"]] = None if missing else convert(value, field)
                except (ValidationError, ValueError, OverflowError) as exc:
                    valid = False
                    issue(document["id"], index, field["target"], str(exc))
            if valid:
                result["records"].append({"document_id": document["id"],
                                          "row": index, "data": data})
            else:
                result["summary"]["rows_rejected"] += 1
    result["summary"]["rows_emitted"] = len(result["records"])
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8-sig") as handle:
            payload = load_json(handle.read())
        result = automate(payload)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        result = empty_result()
        result["status"] = "error"
        result["issues"].append({"document_id": None, "row": None,
                                 "field": None, "message": str(exc)})
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
