"""Synthetic reference pipeline: schema-driven extraction -> feedback analysis.

Patterns are trusted Python regular expressions with a named ``value`` group.
Only the first match is extracted; spans are half-open Python character offsets.
Run: python -B implementation.py example_input.json
"""

import json
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(expected), f"{path} must contain exactly {sorted(expected)}")


def text(value, path, maximum=20000):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonblank text")
    require(len(value) <= maximum, f"{path} exceeds {maximum} characters")


def array(value, path, maximum=100):
    require(isinstance(value, list), f"{path} must be an array")
    require(len(value) <= maximum, f"{path} exceeds {maximum} entries")


def validate_input(data):
    keys(data, {"schema_version", "synthetic", "schema", "documents", "feedback_config"}, "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be integer 1")
    require(type(data["synthetic"]) is bool, "synthetic must be boolean")
    keys(data["schema"], {"fields"}, "schema")
    fields = data["schema"]["fields"]
    array(fields, "schema.fields", 30)
    require(bool(fields), "schema.fields cannot be empty")
    names = set()
    for field in fields:
        keys(field, {"name", "pattern", "required"}, "field")
        text(field["name"], "field.name", 100)
        require(field["name"] not in names, "field names must be unique")
        names.add(field["name"])
        text(field["pattern"], "field.pattern", 500)
        require(type(field["required"]) is bool, "field.required must be boolean")
        try:
            pattern = re.compile(field["pattern"])
        except (re.error, OverflowError, RecursionError) as error:
            raise ValidationError(f"invalid field pattern: {error}") from error
        require("value" in pattern.groupindex, "field pattern must have a named value group")
    documents = data["documents"]
    array(documents, "documents")
    identifiers = set()
    for document in documents:
        keys(document, {"id", "text"}, "document")
        text(document["id"], "document.id", 100)
        require(document["id"] not in identifiers, "document ids must be unique")
        identifiers.add(document["id"])
        require(isinstance(document["text"], str), "document.text must be text")
        require(len(document["text"]) <= 20000, "document.text exceeds 20000 characters")
    config = data["feedback_config"]
    keys(config, {"text_field", "themes"}, "feedback_config")
    text(config["text_field"], "feedback_config.text_field", 100)
    require(config["text_field"] in names, "text_field must reference a schema field")
    array(config["themes"], "feedback_config.themes", 30)
    theme_names = set()
    for theme in config["themes"]:
        keys(theme, {"name", "keywords"}, "theme")
        text(theme["name"], "theme.name", 100)
        require(theme["name"] not in theme_names, "theme names must be unique")
        theme_names.add(theme["name"])
        array(theme["keywords"], "theme.keywords", 30)
        require(bool(theme["keywords"]), "theme.keywords cannot be empty")
        seen = set()
        for keyword in theme["keywords"]:
            text(keyword, "theme.keyword", 100)
            require(keyword == keyword.strip(), "theme keywords cannot have outer whitespace")
            require(keyword.casefold() not in seen, "theme keywords must be unique ignoring case")
            seen.add(keyword.casefold())
    return data


def _extract(data):
    records = []
    for document in data["documents"]:
        fields = {}
        missing = []
        for definition in data["schema"]["fields"]:
            match = re.search(definition["pattern"], document["text"])
            value = match.group("value") if match else None
            if value is None or not value.strip():
                fields[definition["name"]] = None
                missing.append({"field": definition["name"], "required": definition["required"]})
                continue
            start, end = match.span("value")
            left_trim = len(value) - len(value.lstrip())
            right_trim = len(value) - len(value.rstrip())
            fields[definition["name"]] = {
                "value": value.strip(),
                "source": {
                    "document_id": document["id"],
                    "start": start + left_trim,
                    "end": end - right_trim,
                },
            }
        records.append({
            "document_id": document["id"],
            "fields": fields,
            "missing_fields": missing,
            "complete": not any(item["required"] for item in missing),
        })
    return {"records": records}


def dedup_key(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def _analyze(data, extraction):
    groups = []
    skipped = []
    index = {}
    field_name = data["feedback_config"]["text_field"]
    eligible = {}
    for record in extraction["records"]:
        field = record["fields"][field_name]
        if not record["complete"]:
            reason = "missing_required_fields"
        elif field is None:
            reason = "missing_feedback"
        elif not dedup_key(field["value"]):
            reason = "no_feedback_words"
        else:
            reason = None
        if reason:
            skipped.append({"document_id": record["document_id"], "reason": reason})
            continue
        key = dedup_key(field["value"])
        eligible[record["document_id"]] = field
        if key not in index:
            group = {
                "id": f"feedback-{len(groups) + 1}",
                "dedup_key": key,
                "representative_text": field["value"],
                "document_ids": [],
            }
            index[key] = group
            groups.append(group)
        index[key]["document_ids"].append(record["document_id"])

    themes = []
    matched_groups = set()
    for definition in data["feedback_config"]["themes"]:
        supports = []
        group_ids = []
        for group in groups:
            group_supports = []
            for document_id in group["document_ids"]:
                field = eligible[document_id]
                for keyword in definition["keywords"]:
                    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
                    for match in re.finditer(pattern, field["value"], flags=re.IGNORECASE):
                        start = field["source"]["start"] + match.start()
                        end = field["source"]["start"] + match.end()
                        group_supports.append({
                            "group_id": group["id"],
                            "document_id": document_id,
                            "keyword": keyword,
                            "excerpt": match.group(),
                            "source": {"document_id": document_id, "start": start, "end": end},
                        })
            if group_supports:
                group_ids.append(group["id"])
                matched_groups.add(group["id"])
                supports.extend(group_supports)
        themes.append({
            "name": definition["name"],
            "unique_feedback_count": len(group_ids),
            "group_ids": group_ids,
            "supporting_excerpts": supports,
        })
    return {
        "groups": groups,
        "themes": themes,
        "skipped_documents": skipped,
        "unclassified_group_ids": [group["id"] for group in groups if group["id"] not in matched_groups],
        "counts": {
            "input_documents": len(extraction["records"]),
            "eligible_documents": len(eligible),
            "unique_feedback": len(groups),
            "duplicate_documents": len(eligible) - len(groups),
            "skipped_documents": len(skipped),
        },
    }


def same_json(actual, expected):
    """Canonical serialization also distinguishes booleans from integer offsets."""
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def validate(stage, payload, *, input_data=None, extraction=None):
    """Shared boundary validation, including deterministic semantic/provenance checks."""
    if stage == "input":
        return validate_input(payload)
    require(stage in {"extraction", "feedback", "output"}, "unknown validation stage")
    validate_input(input_data)
    expected_extraction = _extract(input_data)
    if stage == "extraction":
        require(same_json(payload, expected_extraction), "extraction does not match schema or source")
    elif stage == "feedback":
        require(same_json(extraction, expected_extraction), "invalid extraction handoff")
        require(same_json(payload, _analyze(input_data, extraction)),
                "feedback does not match validated extraction")
    else:
        expected = {
            "schema_version": 1,
            "status": "ok",
            "synthetic": input_data["synthetic"],
            "extraction": expected_extraction,
            "feedback": _analyze(input_data, expected_extraction),
        }
        require(same_json(payload, expected), "invalid pipeline output")
    return payload


def extract(data):
    validate("input", data)
    result = _extract(data)
    return validate("extraction", result, input_data=data)


def analyze_feedback(data, extraction):
    validate("extraction", extraction, input_data=data)
    result = _analyze(data, extraction)
    return validate("feedback", result, input_data=data, extraction=extraction)


def run_pipeline(data):
    extraction = extract(data)
    output = {
        "schema_version": 1,
        "status": "ok",
        "synthetic": data["synthetic"],
        "extraction": extraction,
        "feedback": analyze_feedback(data, extraction),
    }
    return validate("output", output, input_data=data)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        require(len(arguments) == 1, "usage: python -B implementation.py INPUT.json")
        with open(arguments[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
