"""Deterministic labeled-line extraction; source offsets are Unicode code points."""

import datetime
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def normalize_label(value):
    return " ".join(value.split()).casefold()


class Validator:
    TYPES = {"string", "integer", "number", "boolean", "date"}

    @staticmethod
    def request(data):
        if not isinstance(data, dict) or set(data) != {"document", "schema"}:
            raise ValidationError("Input must contain exactly document and schema")
        if not isinstance(data["document"], str) or len(data["document"]) > 1_000_000:
            raise ValidationError("document must be a string of at most 1000000 characters")
        schema = data["schema"]
        if not isinstance(schema, list) or not 1 <= len(schema) <= 100:
            raise ValidationError("schema must contain 1 to 100 fields")
        names, labels = set(), set()
        for field in schema:
            if not isinstance(field, dict):
                raise ValidationError("Each schema field must be an object")
            if set(field) - {"name", "type", "required", "aliases"}:
                raise ValidationError("Unknown schema field property")
            if not {"name", "type", "required"} <= set(field):
                raise ValidationError("Fields require name, type and required")
            name = field["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
                raise ValidationError("Invalid field name")
            if name in names:
                raise ValidationError("Duplicate field name: " + name)
            names.add(name)
            if not isinstance(field["type"], str) or field["type"] not in Validator.TYPES:
                raise ValidationError("Unsupported field type")
            if type(field["required"]) is not bool:
                raise ValidationError("required must be a boolean")
            aliases = field.get("aliases", [])
            if not isinstance(aliases, list) or len(aliases) > 20:
                raise ValidationError("aliases must be a list of at most 20 labels")
            for label in [name] + aliases:
                if (not isinstance(label, str) or not normalize_label(label)
                        or len(label) > 100 or any(c in label for c in "\r\n:=")):
                    raise ValidationError("Invalid extraction label")
                normalized = normalize_label(label)
                if normalized in labels:
                    raise ValidationError("Duplicate or overlapping extraction labels")
                labels.add(normalized)
        return data

    @staticmethod
    def result(data, result):
        if result["status"] != "ok" or len(result["fields"]) != len(data["schema"]):
            raise ValidationError("Invalid result envelope")
        expected_missing = []
        for spec, field in zip(data["schema"], result["fields"]):
            if field["name"] != spec["name"] or field["type"] != spec["type"]:
                raise ValidationError("Result schema mismatch")
            state = field["state"]
            if state not in {"extracted", "missing", "invalid", "ambiguous"}:
                raise ValidationError("Invalid field state")
            for source in field["sources"]:
                start, end = source["start"], source["end"]
                if not 0 <= start <= end <= len(data["document"]):
                    raise ValidationError("Invalid source span")
                if data["document"][start:end] != source["text"]:
                    raise ValidationError("Source span does not match document")
            if state == "extracted":
                if len(field["sources"]) != 1:
                    raise ValidationError("Extracted field needs one source")
                expected = convert(field["sources"][0]["text"], spec["type"])
                if type(expected) is not type(field["value"]) or expected != field["value"]:
                    raise ValidationError("Value does not match declared type and source")
            elif field["value"] is not None:
                raise ValidationError("Unresolved field must have null value")
            if spec["required"] and state != "extracted":
                expected_missing.append({"name": spec["name"], "reason": state})
        if result["missing_fields"] != expected_missing:
            raise ValidationError("Invalid missing-field report")
        return result


def convert(text, kind):
    if kind == "string":
        return text
    if kind == "integer":
        if not re.fullmatch(r"[+-]?[0-9]+", text) or len(text) > 100:
            raise ValueError("Expected an integer of at most 100 characters")
        return int(text)
    if kind == "number":
        if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
            raise ValueError("Expected a decimal number")
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Number must be finite")
        return value
    if kind == "boolean":
        if text.casefold() not in {"true", "false"}:
            raise ValueError("Expected true or false")
        return text.casefold() == "true"
    if kind == "date":
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
            raise ValueError("Expected YYYY-MM-DD")
        return datetime.date.fromisoformat(text).isoformat()
    raise ValidationError("Unsupported field type")


def extract(data):
    data = Validator.request(data)
    lookup = {}
    matches = {field["name"]: [] for field in data["schema"]}
    for field in data["schema"]:
        for label in [field["name"]] + field.get("aliases", []):
            lookup[normalize_label(label)] = field["name"]
    offset = 0
    for line in data["document"].splitlines(keepends=True):
        content = line.rstrip("\r\n")
        match = re.match(r"([^:=]+)[:=](.*)\Z", content)
        if match:
            name = lookup.get(normalize_label(match.group(1)))
            if name:
                raw = match.group(2)
                text = raw.strip()
                start = offset + match.start(2) + len(raw) - len(raw.lstrip())
                matches[name].append({
                    "start": start, "end": start + len(text), "text": text,
                    "label": match.group(1).strip(),
                })
        offset += len(line)
    fields, missing = [], []
    for spec in data["schema"]:
        sources = matches[spec["name"]]
        item = {
            "name": spec["name"], "type": spec["type"], "required": spec["required"],
            "state": "missing", "value": None, "sources": sources, "message": None,
        }
        if len(sources) > 1:
            item.update(state="ambiguous", message="Multiple matching labeled lines")
        elif sources and sources[0]["text"]:
            try:
                item["value"] = convert(sources[0]["text"], spec["type"])
                item["state"] = "extracted"
            except ValueError as exc:
                item.update(state="invalid", message=str(exc))
        else:
            item["message"] = "Blank value" if sources else "No matching labeled line"
        fields.append(item)
        if spec["required"] and item["state"] != "extracted":
            missing.append({"name": spec["name"], "reason": item["state"]})
    return Validator.result(data, {"status": "ok", "fields": fields, "missing_fields": missing})


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
        # newline="" preserves CRLF for document spans in the decoded JSON string.
        with open(args[0], encoding="utf-8", newline="") as handle:
            data = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = extract(data)
        code = 0
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        result = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
