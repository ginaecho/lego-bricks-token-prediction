"""Deterministic synthetic public-sector extraction; Python standard library only."""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


ENTITIES = {"citizen_service_request", "benefits_application", "policy_document"}
PUBLIC_FIELDS = {
    "case_number", "service_type", "benefit_type", "policy_id",
    "eligibility_age", "is_active", "deadline_days",
}
SENSITIVE = re.compile(
    r"name|address|email|phone|birth|ssn|national.?id|citizen.?id", re.I
)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact_keys(value, keys, description):
    require(isinstance(value, dict) and set(value) == set(keys),
            description + " has unsupported or missing keys.")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON object keys must be unique.")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("JSON numbers must be finite.")


def load_json(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def validate_input(data):
    exact_keys(data, {"schema_version", "synthetic", "purpose", "document", "fields"},
               "Input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Use schema version 1.")
    require(data["synthetic"] is True, "Only clearly labeled synthetic data is allowed.")
    require(data["purpose"] in ("service_delivery", "policy_review"),
            "Choose a supported use purpose.")
    doc = data["document"]
    exact_keys(doc, {"id", "entity", "format", "content"}, "Document")
    require(isinstance(doc["id"], str) and re.fullmatch(r"SYN-[A-Z0-9-]{1,60}", doc["id"]),
            "Document ID must be a synthetic SYN- identifier.")
    require(isinstance(doc["entity"], str) and doc["entity"] in ENTITIES,
            "Choose a supported public-sector entity.")
    require(doc["format"] in ("government_form_json", "policy_regulation_text"),
            "Choose a supported document format.")
    if doc["entity"] == "policy_document":
        require(doc["format"] == "policy_regulation_text"
                and data["purpose"] == "policy_review",
                "Policy documents need policy text and a policy-review purpose.")
    else:
        require(doc["format"] == "government_form_json"
                and data["purpose"] == "service_delivery",
                "Applications and requests need form JSON and a service-delivery purpose.")
    content = doc["content"]
    if doc["format"] == "government_form_json":
        require(isinstance(content, dict) and len(content) <= 100,
                "Form content must be an object with at most 100 scalar fields.")
        for key, value in content.items():
            require(isinstance(key, str) and len(key) <= 100,
                    "Form keys must be short text.")
            require(value is None or type(value) in (str, int, bool),
                    "Form values must be text, integers, booleans, or null.")
            require(not isinstance(value, str) or len(value) <= 10000,
                    "Form text is too long.")
    else:
        require(isinstance(content, str) and 0 < len(content) <= 100000,
                "Policy content must be nonempty text of at most 100000 characters.")
    fields = data["fields"]
    require(isinstance(fields, list) and 1 <= len(fields) <= 50,
            "Provide between 1 and 50 extraction fields.")
    names = set()
    for field in fields:
        exact_keys(field, {"name", "source", "type", "required", "pii"}, "Field")
        name = field["name"]
        require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,59}", name),
                "Field names must be short lower-case identifiers.")
        require(name not in names, "Field names must be unique.")
        names.add(name)
        source = field["source"]
        require(isinstance(source, str) and 0 < len(source) <= 100
                and not any(c in source for c in "\r\n:"),
                "Field sources must be short keys or single-line labels.")
        require(field["type"] in ("string", "integer", "boolean"),
                "Supported field types are string, integer, and boolean.")
        require(type(field["required"]) is bool and type(field["pii"]) is bool,
                "Required and PII settings must be booleans.")
        require(field["pii"] or (name in PUBLIC_FIELDS and not SENSITIVE.search(source)),
                "Personal or unrecognized fields must be marked as PII.")


def source_index(document):
    """Form spans refer to this exact sorted, indented JSON representation."""
    if document["format"] == "policy_regulation_text":
        return document["content"], None
    pieces = ["{\n"]
    indexed = {}
    items = sorted(document["content"].items())
    for index, (key, value) in enumerate(items):
        prefix = "  " + json.dumps(key, ensure_ascii=False) + ": "
        pieces.append(prefix)
        start = sum(map(len, pieces))
        encoded = json.dumps(value, ensure_ascii=False)
        pieces.append(encoded)
        indexed[key] = (value, start, start + len(encoded))
        pieces.append(",\n" if index < len(items) - 1 else "\n")
    pieces.append("}")
    return "".join(pieces), indexed


def locate(field, text, indexed):
    if indexed is not None:
        return indexed.get(field["source"])
    pattern = re.compile(r"^[ \t]*" + re.escape(field["source"])
                         + r"[ \t]*:[ \t]*(?P<value>[^\r\n]*)", re.I | re.M)
    matches = list(pattern.finditer(text))
    require(len(matches) <= 1, "A requested policy label occurs more than once.")
    if not matches:
        return None
    match = matches[0]
    value = match.group("value").rstrip()
    return value, match.start("value"), match.start("value") + len(value)


def convert(value, kind):
    if kind == "string":
        require(isinstance(value, str), "An extracted value does not match its field type.")
        return value.strip()
    if kind == "integer":
        if type(value) is int:
            return value
        require(isinstance(value, str) and re.fullmatch(r"-?(0|[1-9][0-9]*)", value.strip()),
                "An extracted value does not match its field type.")
        return int(value.strip())
    if type(value) is bool:
        return value
    require(isinstance(value, str) and value.strip().lower() in ("true", "false"),
            "An extracted value does not match its field type.")
    return value.strip().lower() == "true"


def extract(data):
    validate_input(data)
    document = data["document"]
    text, indexed = source_index(document)
    results, missing = [], []
    for field in data["fields"]:
        found = locate(field, text, indexed)
        absent = found is None or found[0] is None or (
            isinstance(found[0], str) and not found[0].strip())
        item = {"name": field["name"], "type": field["type"],
                "required": field["required"], "pii": field["pii"]}
        if absent:
            missing.append({"name": field["name"], "required": field["required"]})
            item.update(status="missing", value=None, source_span=None,
                        explanation="The source does not provide a value for this field.")
        else:
            value, start, end = found
            normalized = convert(value, field["type"])
            if not field["pii"]:
                # Public text is restricted to short codes, not arbitrary narratives.
                require(not isinstance(normalized, str) or
                        re.fullmatch(r"[A-Z][A-Z0-9_-]{0,39}", normalized),
                        "Public text fields must be short upper-case codes. Mark other text as PII.")
            item.update(status="redacted" if field["pii"] else "extracted",
                        value="[REDACTED]" if field["pii"] else normalized,
                        source_span={"document_id": document["id"], "start": start, "end": end},
                        explanation=("A value was found. Personal information is hidden."
                                     if field["pii"] else
                                     "The value was read from the requested source and checked for its type."))
        results.append(item)
    return {
        "schema_version": 1, "status": "incomplete" if any(m["required"] for m in missing) else "ok",
        "synthetic": True, "document_id": document["id"], "entity": document["entity"],
        "fields": results, "missing_fields": missing,
        "source_span_convention": {
            "unit": "Unicode code points", "interval": "start inclusive, end exclusive",
            "representation": ("Sorted form JSON with two-space indentation and literal Unicode; "
                               "scalar spans include JSON quotes."
                               if indexed is not None else "Original policy text."),
        },
        "review": {
            "decision_made": False,
            "explanation": "This report extracts facts only. A person must review any service or benefit decision.",
            "privacy": "Personal values and source text are not included. Keep the original in protected storage.",
            "accessibility": "Plain-language explanations are included. No color or visual layout is needed.",
            "notice": "Synthetic demonstration only. This is not a compliance certification.",
        },
    }


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Provide exactly one input JSON file.")
        path = Path(args[0])
        require(path.stat().st_size <= 1000000, "Input file is too large.")
        data = load_json(path.read_text(encoding="utf-8"))
        result = extract(data)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, OverflowError):
        # Never echo untrusted values, paths, or parser snippets into public errors.
        print(json.dumps({"schema_version": 1, "status": "error",
                          "message": "The input could not be read or did not meet the extraction rules."}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
