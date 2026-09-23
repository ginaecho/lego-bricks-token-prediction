"""Synthetic document -> validated extraction -> grounded FAQ reference pipeline.

Input version 1: document, schema.fields, schema.question_field, knowledge_base,
and optional min_overlap (default 2). Labels are literal, case-sensitive,
line-anchored "Label: value" declarations. Spans use Python character offsets,
start inclusive and end exclusive. See example_input.json for the full schema.
"""

import json
import re
import string
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing keys: " + ", ".join(sorted(set(required) - set(value))))
    require(set(value) <= set(required) | set(optional), "Unknown object keys")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")


def placeholders(template):
    try:
        result = []
        for _, name, spec, conversion in string.Formatter().parse(template):
            if name is not None:
                require(bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)), "Invalid answer placeholder")
                require(not spec and conversion is None, "Formatting directives are unsupported")
                result.append(name)
        return result
    except ValueError as exc:
        raise ValidationError("Invalid answer template: " + str(exc)) from exc


def validate_input(data):
    keys(data, ("version", "document", "schema", "knowledge_base"), ("min_overlap",))
    require(type(data["version"]) is int and data["version"] == 1, "Unsupported version")
    require(isinstance(data["document"], str), "document must be text")
    keys(data["schema"], ("fields", "question_field"))
    fields = data["schema"]["fields"]
    require(isinstance(fields, list) and bool(fields), "fields must be a nonempty list")
    names, labels = set(), set()
    for field in fields:
        keys(field, ("name", "label", "type", "required"))
        text(field["name"], "name")
        require(bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field["name"])), "Invalid field name")
        text(field["label"], "label")
        require(field["label"] == field["label"].strip() and not any(c in field["label"] for c in "\r\n:"),
                "Labels must be trimmed single-line text without colons")
        require(field["type"] in ("text", "integer"), "Unsupported field type")
        require(type(field["required"]) is bool, "required must be boolean")
        require(field["name"] not in names and field["label"] not in labels, "Duplicate field name or label")
        names.add(field["name"])
        labels.add(field["label"])
    question = data["schema"]["question_field"]
    require(isinstance(question, str) and question in names, "Unknown question_field")
    require(next(f for f in fields if f["name"] == question)["type"] == "text",
            "question_field must have text type")
    threshold = data.get("min_overlap", 2)
    require(type(threshold) is int and threshold >= 1, "min_overlap must be a positive integer")
    kb = data["knowledge_base"]
    require(isinstance(kb, list), "knowledge_base must be a list")
    ids = set()
    for entry in kb:
        keys(entry, ("id", "question", "answer", "required_fields"))
        for name in ("id", "question", "answer"):
            text(entry[name], name)
        require(entry["id"] not in ids, "Duplicate knowledge id")
        ids.add(entry["id"])
        required = entry["required_fields"]
        require(isinstance(required, list) and all(isinstance(n, str) and n in names for n in required),
                "Unknown required_fields")
        require(len(required) == len(set(required)), "Duplicate required_fields")
        require(set(placeholders(entry["answer"])) <= names, "Unknown answer placeholder")
    return data


def extract(data):
    fields, errors, missing = {}, [], []
    for spec in data["schema"]["fields"]:
        name = spec["name"]
        pattern = r"^[ \t]*" + re.escape(spec["label"]) + r"[ \t]*:[ \t]*([^\r\n]*)"
        matches = list(re.finditer(pattern, data["document"], re.MULTILINE))
        item = {"value": None, "source": None, "state": "missing"}
        if len(matches) > 1:
            item["state"] = "invalid"
            errors.append({"field": name, "reason": "duplicate_label"})
        elif matches:
            match = matches[0]
            raw = match.group(1)
            value = raw.strip(" \t")
            if value:
                start = match.start(1) + len(raw) - len(raw.lstrip(" \t"))
                item["source"] = {"start": start, "end": start + len(value), "text": value}
                if spec["type"] == "integer" and not re.fullmatch(r"[+-]?[0-9]+", value):
                    item["state"] = "invalid"
                    errors.append({"field": name, "reason": "invalid_integer"})
                else:
                    try:
                        item.update(value=int(value) if spec["type"] == "integer" else value, state="present")
                    except ValueError:
                        item["state"] = "invalid"
                        errors.append({"field": name, "reason": "invalid_integer"})
        if item["state"] == "missing":
            missing.append(name)
        fields[name] = item
    result = {
        "fields": fields,
        "missing_fields": missing,
        "errors": errors,
        "complete": all(fields[f["name"]]["state"] == "present"
                        for f in data["schema"]["fields"] if f["required"]),
    }
    validate_extraction(data, result)
    return result


def validate_extraction(data, result):
    """The same field declarations validate extraction and the FAQ handoff."""
    keys(result, ("fields", "missing_fields", "errors", "complete"))
    require(isinstance(result["fields"], dict), "Invalid extracted fields")
    require(set(result["fields"]) == {s["name"] for s in data["schema"]["fields"]},
            "Extracted field set mismatch")
    expected_missing, invalid_names = [], []
    for spec in data["schema"]["fields"]:
        item = result["fields"][spec["name"]]
        keys(item, ("value", "source", "state"))
        require(item["state"] in ("present", "missing", "invalid"), "Invalid field state")
        if item["state"] == "missing":
            expected_missing.append(spec["name"])
        if item["state"] == "invalid":
            invalid_names.append(spec["name"])
        if item["state"] == "present":
            expected_type = int if spec["type"] == "integer" else str
            require(type(item["value"]) is expected_type, "Extracted value type mismatch")
            require(item["source"] is not None, "Present field needs source")
        else:
            require(item["value"] is None, "Unavailable value must be null")
        source = item["source"]
        if source is not None:
            keys(source, ("start", "end", "text"))
            start, end = source["start"], source["end"]
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(data["document"]),
                    "Invalid source offsets")
            require(data["document"][start:end] == source["text"], "Source text mismatch")
            if item["state"] == "present":
                require(str(item["value"]) == source["text"] if spec["type"] == "text"
                        else bool(re.fullmatch(r"[+-]?[0-9]+", source["text"]))
                        and int(source["text"]) == item["value"], "Value not grounded in source")
        if item["state"] == "missing":
            require(source is None, "Missing field cannot have a source")
    require(result["missing_fields"] == expected_missing, "Missing-field report mismatch")
    require(isinstance(result["errors"], list), "errors must be a list")
    for error in result["errors"]:
        keys(error, ("field", "reason"))
        require(error["reason"] in ("duplicate_label", "invalid_integer"), "Unknown extraction error")
    require([e["field"] for e in result["errors"]] == invalid_names, "Error report mismatch")
    complete = all(result["fields"][s["name"]]["state"] == "present"
                   for s in data["schema"]["fields"] if s["required"])
    require(type(result["complete"]) is bool and result["complete"] == complete, "Completeness mismatch")


STOP_WORDS = frozenset("a an the is are do does can i my me how what where when to of for in on please".split())


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)) - STOP_WORDS


def render(entry, extraction):
    needed = set(entry["required_fields"]) | set(placeholders(entry["answer"]))
    if any(extraction["fields"][name]["state"] != "present" for name in needed):
        return None
    return entry["answer"].format_map({name: extraction["fields"][name]["value"] for name in needed})


def answer_faq(data, extraction):
    validate_extraction(data, extraction)
    question = extraction["fields"][data["schema"]["question_field"]]
    result = {"status": "abstained", "answer": None, "reason": None, "citations": [], "retrieval": []}
    if not extraction["complete"]:
        result["reason"] = "required_extraction_unavailable"
    elif question["state"] != "present":
        result["reason"] = "question_unavailable"
    else:
        terms = tokens(question["value"])
        ranked = sorted(
            ({"id": e["id"], "score": len(terms & tokens(e["question"]))} for e in data["knowledge_base"]),
            key=lambda e: (-e["score"], e["id"]),
        )
        result["retrieval"] = ranked
        if not ranked or ranked[0]["score"] < data.get("min_overlap", 2):
            result["reason"] = "insufficient_evidence"
        elif len(ranked) > 1 and ranked[0]["score"] == ranked[1]["score"]:
            result["reason"] = "ambiguous_evidence"
        else:
            entry = next(e for e in data["knowledge_base"] if e["id"] == ranked[0]["id"])
            answer = render(entry, extraction)
            if answer is None:
                result["reason"] = "answer_fields_unavailable"
            else:
                result.update(status="answered", answer=answer, citations=[entry["id"]])
    validate_faq(data, extraction, result)
    return result


def validate_faq(data, extraction, result):
    keys(result, ("status", "answer", "reason", "citations", "retrieval"))
    require(result["status"] in ("answered", "abstained"), "Invalid FAQ status")
    if result["status"] == "answered":
        require(result["reason"] is None and len(result["citations"]) == 1, "Invalid answer metadata")
        entry = next((e for e in data["knowledge_base"] if e["id"] == result["citations"][0]), None)
        require(entry is not None and result["answer"] == render(entry, extraction), "Ungrounded answer")
        require(isinstance(result["answer"], str), "Answer must be text")
    else:
        require(result["answer"] is None and result["citations"] == [], "Abstention cannot contain an answer")
        require(result["reason"] in (
            "required_extraction_unavailable", "question_unavailable", "insufficient_evidence",
            "ambiguous_evidence", "answer_fields_unavailable"), "Invalid abstention reason")


def run_pipeline(data):
    validate_input(data)
    extraction = extract(data)
    faq = answer_faq(data, extraction)
    return {"version": 1, "status": "ok", "extraction": extraction, "faq": faq}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValidationError("Invalid JSON constant")))
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
