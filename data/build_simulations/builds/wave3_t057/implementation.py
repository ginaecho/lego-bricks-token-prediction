"""Deterministic synthetic-friendly document extraction, reshaping and checking.

Run: python -B implementation.py example_input.json
No files are modified; document contents are embedded in the input envelope.
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


def keys(value, allowed, required, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(not (set(value) - set(allowed)), f"{path} contains unknown keys")
    require(set(required) <= set(value), f"{path} is missing required keys")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return type(value) in (int, float) and (
        not isinstance(value, float) or math.isfinite(value)
    )


def validate(payload):
    keys(payload, ("schema_version", "fixture_label", "documents", "fields", "unique"),
         ("schema_version", "documents", "fields"), "input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version must be 1")
    if "fixture_label" in payload:
        require(text(payload["fixture_label"]), "fixture_label must be nonempty text")
    require(isinstance(payload["documents"], list), "documents must be an array")
    require(isinstance(payload["fields"], list) and payload["fields"],
            "fields must be a nonempty array")
    targets = set()
    for field in payload["fields"]:
        keys(field, ("source", "target", "type", "required", "default", "trim",
                     "min", "max", "pattern", "choices"),
             ("source", "target", "type"), "field")
        require(text(field["source"]) and text(field["target"]),
                "field source and target must be nonempty strings")
        require(field["target"] not in targets, "duplicate field target")
        targets.add(field["target"])
        require(field["type"] in ("string", "integer", "number", "boolean"),
                "unsupported field type")
        for flag in ("required", "trim"):
            if flag in field:
                require(type(field[flag]) is bool, f"{flag} must be boolean")
        for bound in ("min", "max"):
            if bound in field:
                require(field["type"] in ("integer", "number") and number(field[bound]),
                        f"{bound} requires a finite numeric bound and numeric field")
        if "min" in field and "max" in field:
            require(field["min"] <= field["max"], "min must not exceed max")
        if "pattern" in field:
            require(field["type"] == "string" and isinstance(field["pattern"], str),
                    "pattern requires a string field")
            try:
                re.compile(field["pattern"])
            except re.error as exc:
                raise ValidationError("invalid field pattern") from exc
        if "choices" in field:
            require(isinstance(field["choices"], list) and field["choices"],
                    "choices must be a nonempty array")
            for choice in field["choices"]:
                require(choice is not None, "choices cannot contain null")
                require(type(choice) is type(convert(choice, field)) and
                        choice == convert(choice, field),
                        "choices must already have the declared field type")
        if "default" in field and field["default"] is not None:
            try:
                convert(field["default"], field)
            except ValueError as exc:
                raise ValidationError("default cannot be converted to field type") from exc
    unique = payload.get("unique", [])
    require(isinstance(unique, list) and all(isinstance(x, str) for x in unique),
            "unique must be an array of target names")
    require(len(set(unique)) == len(unique) and set(unique) <= targets,
            "unique contains duplicate or unknown targets")
    ids = set()
    for doc in payload["documents"]:
        keys(doc, ("id", "format", "content"), ("id", "format", "content"), "document")
        require(text(doc["id"]) and doc["id"] not in ids,
                "document ids must be nonempty and unique")
        ids.add(doc["id"])
        require(doc["format"] in ("records", "csv", "key_value"), "unsupported format")
        if doc["format"] == "records":
            require(isinstance(doc["content"], list) and
                    all(isinstance(row, dict) and all(isinstance(k, str) for k in row)
                        for row in doc["content"]), "records content must be an array of objects")
        else:
            require(isinstance(doc["content"], str), "text document content must be a string")
    return payload


def convert(value, field):
    kind = field["type"]
    if isinstance(value, str) and field.get("trim", True):
        value = value.strip()
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError("expected text")
        return value
    if kind == "boolean":
        if type(value) is bool:
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError("expected true or false")
    if type(value) is bool:
        raise ValueError("boolean is not numeric")
    if kind == "integer":
        if type(value) is int:
            return value
        if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
            return int(value)
        raise ValueError("expected an integer")
    if not isinstance(value, (str, int, float)):
        raise ValueError("expected a number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError("expected a finite number")
    return result


def extract(doc):
    if doc["format"] == "records":
        return doc["content"]
    if doc["format"] == "key_value":
        row = {}
        for line in doc["content"].splitlines():
            if not line.strip():
                continue
            key, separator, value = line.partition(":")
            key = key.strip()
            require(separator and key and key not in row,
                    f"document {doc['id']}: invalid or duplicate key-value line")
            row[key] = value.strip()
        return [row] if row else []
    try:
        reader = csv.reader(io.StringIO(doc["content"], newline=""), strict=True)
        rows = [row for row in reader if row]
    except csv.Error as exc:
        raise ValidationError(f"document {doc['id']}: malformed CSV") from exc
    if not rows:
        return []
    headers = rows[0]
    require(all(text(h) for h in headers) and len(set(headers)) == len(headers),
            f"document {doc['id']}: CSV headers must be nonempty and unique")
    require(all(len(row) == len(headers) for row in rows[1:]),
            f"document {doc['id']}: CSV row width mismatch")
    return [dict(zip(headers, row)) for row in rows[1:]]


def automate(payload):
    payload = validate(payload)
    output, issues = [], []
    seen = {target: set() for target in payload.get("unique", [])}
    for doc in payload["documents"]:
        for index, source in enumerate(extract(doc), 1):
            shaped = {}
            start = len(issues)

            def issue(target, code, message):
                issues.append({"document_id": doc["id"], "row": index,
                               "field": target, "code": code, "message": message})

            for field in payload["fields"]:
                target = field["target"]
                raw = source.get(field["source"], field.get("default"))
                missing = raw is None or (
                    isinstance(raw, str) and not (
                        raw.strip() if field.get("trim", True) else raw))
                if missing:
                    shaped[target] = None
                    if field.get("required", False):
                        issue(target, "required", "required value is missing")
                    continue
                try:
                    value = convert(raw, field)
                except ValueError as exc:
                    shaped[target] = None
                    issue(target, "type", str(exc))
                    continue
                shaped[target] = value
                if "min" in field and value < field["min"]:
                    issue(target, "min", "value is below minimum")
                if "max" in field and value > field["max"]:
                    issue(target, "max", "value exceeds maximum")
                if "pattern" in field and re.fullmatch(field["pattern"], value) is None:
                    issue(target, "pattern", "value does not match pattern")
                if "choices" in field and value not in field["choices"]:
                    issue(target, "choices", "value is not an allowed choice")
                if target in seen:
                    if value in seen[target]:
                        issue(target, "unique", "duplicate value across documents")
                    seen[target].add(value)
            output.append({"document_id": doc["id"], "row": index,
                           "data": shaped, "valid": len(issues) == start})
    return {"schema_version": 1, "status": "needs_review" if issues else "ok",
            "fixture_label": payload.get("fixture_label"), "records": output,
            "issues": issues,
            "summary": {"documents": len(payload["documents"]), "records": len(output),
                        "valid_records": sum(row["valid"] for row in output),
                        "issues": len(issues)}}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=lambda value: (
                (_ for _ in ()).throw(ValidationError(f"nonfinite JSON value: {value}"))))
        result = automate(payload)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
