"""Synthetic, deterministic interest ranking -> grounded field extraction.

Usage: python -B implementation.py example_input.json
Tags and field labels are case-sensitive. Fields use the first exact line label
("Label: value"); spans are half-open Unicode character offsets in source text.
Missing required fields are reported, not fabricated or treated as fatal errors.
"""

import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown keys")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def strings(value, path):
    require(isinstance(value, list), path + " must be an array")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), path + " must contain unique values")


def validate(kind, value, context=None):
    """One validation entry point for input, ranking handoff and final output."""
    if kind == "input":
        keys(value, ("schema_version", "fixture_label", "preferences", "documents",
                     "extraction_schema"), "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        text(value["fixture_label"], "fixture_label")
        p = value["preferences"]
        keys(p, ("interests", "excluded_ids", "excluded_tags", "limit"), "preferences")
        require(isinstance(p["interests"], dict), "interests must be an object")
        for tag, weight in p["interests"].items():
            text(tag, "interest tag")
            require(type(weight) in (int, float) and 0 < weight <= 1_000_000,
                    "interest weights must be finite positive numbers <= 1000000")
        strings(p["excluded_ids"], "excluded_ids")
        strings(p["excluded_tags"], "excluded_tags")
        require(type(p["limit"]) is int and p["limit"] >= 0,
                "limit must be a nonnegative integer")
        require(isinstance(value["documents"], list), "documents must be an array")
        ids = []
        for doc in value["documents"]:
            keys(doc, ("id", "title", "tags", "text"), "document")
            text(doc["id"], "document.id")
            text(doc["title"], "document.title")
            strings(doc["tags"], "document.tags")
            require(isinstance(doc["text"], str), "document.text must be text")
            ids.append(doc["id"])
        require(len(ids) == len(set(ids)), "document IDs must be unique")
        schema = value["extraction_schema"]
        require(isinstance(schema, list) and bool(schema), "extraction_schema must be nonempty")
        names, labels = [], []
        for field in schema:
            keys(field, ("name", "label", "type", "required"), "field")
            text(field["name"], "field.name")
            text(field["label"], "field.label")
            require(not any(c in field["label"] for c in "\r\n:"),
                    "field.label cannot contain newlines or colon")
            require(field["label"] == field["label"].strip(),
                    "field.label cannot have surrounding whitespace")
            require(field["type"] in ("string", "integer"), "unsupported field type")
            require(type(field["required"]) is bool, "field.required must be boolean")
            names.append(field["name"])
            labels.append(field["label"])
        require(len(names) == len(set(names)), "field names must be unique")
        require(len(labels) == len(set(labels)), "field labels must be unique")
    elif kind == "handoff":
        require(context is not None, "handoff requires input context")
        require(value == _rank(context), "recommendation handoff is not grounded in input")
    elif kind == "output":
        require(context is not None, "output requires input context")
        keys(value, ("status", "schema_version", "fixture_label", "recommendations",
                     "extractions"), "output")
        require(value["status"] == "ok" and value["schema_version"] == 1,
                "invalid output envelope")
        require(value["fixture_label"] == context["fixture_label"], "fixture label changed")
        validate("handoff", value["recommendations"], context)
        expected = _extract(context, value["recommendations"])
        require(value["extractions"] == expected, "extraction output is not source-grounded")
    else:
        raise ValidationError("unknown validation kind")
    return value


def _rank(data):
    p = data["preferences"]
    result = []
    for doc in data["documents"]:
        if doc["id"] in p["excluded_ids"] or set(doc["tags"]) & set(p["excluded_tags"]):
            continue
        matches = [{"tag": tag, "weight": p["interests"][tag]}
                   for tag in sorted(doc["tags"]) if tag in p["interests"]]
        if not matches:
            continue
        score = sum(item["weight"] for item in matches)
        require(math.isfinite(score), "aggregate score must be finite")
        explanation = "Matched interests: " + ", ".join(
            f'{item["tag"]} (weight {item["weight"]})' for item in matches)
        result.append({"id": doc["id"], "title": doc["title"], "score": score,
                       "matched_interests": matches, "explanation": explanation})
    return sorted(result, key=lambda item: (-item["score"], item["id"]))[:p["limit"]]


def recommend(data):
    validate("input", data)
    return validate("handoff", _rank(data), data)


def _extract(data, recommendations):
    docs = {doc["id"]: doc for doc in data["documents"]}
    results = []
    for rank, recommendation in enumerate(recommendations, 1):
        doc = docs[recommendation["id"]]
        fields, missing, missing_required = [], [], []
        for spec in data["extraction_schema"]:
            pattern = r"(?m)^[ \t]*" + re.escape(spec["label"]) + r":[ \t]*(?P<value>[^\r\n]*)"
            match = re.search(pattern, doc["text"])
            raw = match.group("value").strip() if match else ""
            value, span = None, None
            if raw:
                value = raw
                if spec["type"] == "integer":
                    require(re.fullmatch(r"[+-]?[0-9]+", raw) is not None,
                            f'{doc["id"]}.{spec["name"]} must contain an integer')
                    try:
                        value = int(raw)
                    except ValueError as exc:
                        raise ValidationError("integer field exceeds runtime size limit") from exc
                start = match.start("value") + len(match.group("value")) - len(
                    match.group("value").lstrip())
                span = {"start": start, "end": start + len(raw), "text": raw}
            else:
                missing.append(spec["name"])
                if spec["required"]:
                    missing_required.append(spec["name"])
            fields.append({"name": spec["name"], "type": spec["type"],
                           "required": spec["required"], "value": value, "source_span": span})
        results.append({"document_id": doc["id"], "rank": rank, "fields": fields,
                        "missing_fields": missing, "missing_required_fields": missing_required})
    return results


def extract(data, recommendations):
    validate("input", data)
    validate("handoff", recommendations, data)
    return _extract(data, recommendations)


def run_pipeline(data):
    validate("input", data)
    recommendations = recommend(data)
    output = {"status": "ok", "schema_version": 1, "fixture_label": data["fixture_label"],
              "recommendations": recommendations,
              "extractions": extract(data, recommendations)}
    return validate("output", output, data)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_unique_object)
        output = run_pipeline(data)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
