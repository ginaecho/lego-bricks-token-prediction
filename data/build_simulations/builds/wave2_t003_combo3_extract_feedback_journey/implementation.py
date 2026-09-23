"""Synthetic, deterministic extraction -> feedback -> journey reference pipeline.

Run: python -B implementation.py example_input.json
Spans are half-open Python Unicode character offsets in the original document.
Field labels match exactly at the start of a line, followed by a colon. Values
occupy that line only. Empty values are missing; duplicate labels are errors.
Missing required fields are reported, not invented or silently discarded.
"""

import json
import re
import sys
import unicodedata


VERSION = "1.0"
MAX_DOCUMENTS = 100
MAX_ACTIONS = 100


class ValidationError(ValueError):
    """Invalid input or a broken handoff invariant."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown keys")


def text(value, path):
    require(type(value) is str and bool(value.strip()), path + " must be nonblank text")


def array(value, path, maximum=10000):
    require(type(value) is list and len(value) <= maximum, path + " must be a bounded array")


def unique_strings(value, path, maximum=10000):
    array(value, path, maximum)
    for item in value:
        text(item, path + " item")
    require(len(set(value)) == len(value), path + " contains duplicates")


def normalized(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def _validate_input(data):
    keys(data, ("schema_version", "fixture_label", "documents", "field_schema",
                "themes", "actions", "profile"), "input")
    require(data["schema_version"] == VERSION, "unsupported schema_version")
    text(data["fixture_label"], "fixture_label")
    array(data["documents"], "documents", MAX_DOCUMENTS)
    ids = []
    for doc in data["documents"]:
        keys(doc, ("id", "text"), "document")
        text(doc["id"], "document.id")
        require(type(doc["text"]) is str and len(doc["text"]) <= 100000,
                "document.text must be a string of at most 100000 characters")
        ids.append(doc["id"])
    unique_strings(ids, "document ids")
    array(data["field_schema"], "field_schema", 50)
    names, labels = [], []
    for field in data["field_schema"]:
        keys(field, ("name", "label", "type", "required"), "field")
        text(field["name"], "field.name")
        text(field["label"], "field.label")
        require(field["label"] == field["label"].strip()
                and not any(c in field["label"] for c in ":\r\n"),
                "field.label must be trimmed and contain no colon or newline")
        require(field["type"] in ("string", "integer"), "unsupported field type")
        require(type(field["required"]) is bool, "field.required must be boolean")
        names.append(field["name"])
        labels.append(field["label"])
    unique_strings(names, "field names")
    unique_strings(labels, "field labels")
    schema = {field["name"]: field for field in data["field_schema"]}
    for name in ("customer", "feedback", "goal"):
        require(name in schema and schema[name]["type"] == "string",
                name + " must be declared as a string field")
    array(data["themes"], "themes", 100)
    theme_ids = []
    for theme in data["themes"]:
        keys(theme, ("id", "keywords"), "theme")
        text(theme["id"], "theme.id")
        unique_strings(theme["keywords"], "theme.keywords", 100)
        require(bool(theme["keywords"]), "theme needs at least one keyword")
        require(all(normalized(word) for word in theme["keywords"]),
                "keywords must contain word characters")
        theme_ids.append(theme["id"])
    unique_strings(theme_ids, "theme ids")
    array(data["actions"], "actions", MAX_ACTIONS)
    action_ids = []
    for action in data["actions"]:
        keys(action, ("id", "title", "prerequisites", "theme_ids"), "action")
        text(action["id"], "action.id")
        text(action["title"], "action.title")
        unique_strings(action["prerequisites"], "action.prerequisites", MAX_ACTIONS)
        unique_strings(action["theme_ids"], "action.theme_ids", 100)
        require(set(action["theme_ids"]) <= set(theme_ids), "unknown action theme")
        action_ids.append(action["id"])
    unique_strings(action_ids, "action ids")
    actions = {action["id"]: action for action in data["actions"]}
    for action in actions.values():
        require(set(action["prerequisites"]) <= set(actions), "unknown prerequisite")
    visiting, visited = set(), set()

    def visit(action_id):
        require(action_id not in visiting, "cyclic action prerequisites")
        if action_id in visited:
            return
        visiting.add(action_id)
        for dependency in actions[action_id]["prerequisites"]:
            visit(dependency)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in actions:
        visit(action_id)
    keys(data["profile"], ("completed_action_ids",), "profile")
    unique_strings(data["profile"]["completed_action_ids"], "completed_action_ids", MAX_ACTIONS)
    require(set(data["profile"]["completed_action_ids"]) <= set(actions),
            "unknown completed action")


def _extraction_records(data):
    records = []
    for doc in data["documents"]:
        fields, missing, missing_required = {}, [], []
        for spec in data["field_schema"]:
            matches = list(re.finditer(
                r"^" + re.escape(spec["label"]) + r":[ \t]*([^\r\n]*)",
                doc["text"], re.MULTILINE))
            require(len(matches) <= 1, "duplicate field label in document " + doc["id"])
            raw = matches[0].group(1) if matches else ""
            stripped = raw.strip()
            if not stripped:
                missing.append(spec["name"])
                if spec["required"]:
                    missing_required.append(spec["name"])
                continue
            start = matches[0].start(1) + len(raw) - len(raw.lstrip())
            end = start + len(stripped)
            value = stripped
            if spec["type"] == "integer":
                require(re.fullmatch(r"[+-]?[0-9]{1,18}", stripped) is not None,
                        "invalid integer for " + spec["name"] + " in " + doc["id"])
                value = int(stripped)
            fields[spec["name"]] = {
                "value": value,
                "source": {"document_id": doc["id"], "start": start, "end": end,
                           "text": stripped},
            }
        records.append({"document_id": doc["id"], "fields": fields,
                        "missing_fields": missing,
                        "missing_required_fields": missing_required})
    return records


def _feedback_records(extraction, data):
    groups, skipped = {}, []
    for doc in extraction["documents"]:
        field = doc["fields"].get("feedback")
        if field is None:
            skipped.append({"document_id": doc["document_id"], "reason": "missing_feedback"})
            continue
        value = field["value"]
        canonical = normalized(value)
        if not canonical:
            skipped.append({"document_id": doc["document_id"], "reason": "no_word_content"})
            continue
        if canonical not in groups:
            groups[canonical] = {
                "id": "F" + str(len(groups) + 1).zfill(4),
                "text": value, "normalized": canonical, "occurrences": [],
            }
        groups[canonical]["occurrences"].append({
            "source": dict(field["source"]),
            "customer": doc["fields"].get("customer", {}).get("value"),
            "goal": doc["fields"].get("goal", {}).get("value"),
        })
    items = list(groups.values())
    themes, matched = [], set()
    for theme in data["themes"]:
        supporting = [
            item for item in items
            if any(" " + normalized(keyword) + " " in " " + item["normalized"] + " "
                   for keyword in theme["keywords"])
        ]
        if not supporting:
            continue
        feedback_ids = [item["id"] for item in supporting]
        matched.update(feedback_ids)
        themes.append({
            "id": theme["id"], "feedback_ids": feedback_ids,
            "supporting_excerpts": [
                {"feedback_id": item["id"], **occurrence["source"]}
                for item in supporting for occurrence in item["occurrences"]
            ],
        })
    return {
        "schema_version": VERSION, "stage": "feedback", "items": items, "themes": themes,
        "unmatched_feedback_ids": [item["id"] for item in items if item["id"] not in matched],
        "skipped_documents": skipped,
        "duplicate_count": sum(len(item["occurrences"]) - 1 for item in items),
    }


def _journey_records(feedback, data):
    support = {theme["id"]: theme["feedback_ids"] for theme in feedback["themes"]}
    actions = {action["id"]: action for action in data["actions"]}
    completed = set(data["profile"]["completed_action_ids"])

    def recommendation(action_id):
        action = actions[action_id]
        themes = sorted(set(action["theme_ids"]) & set(support))
        evidence = sorted({item for theme in themes for item in support[theme]})
        return {"action_id": action_id, "title": action["title"],
                "prerequisites": list(action["prerequisites"]),
                "theme_ids": themes, "supporting_feedback_ids": evidence,
                "score": len(evidence)}

    def eligible(done):
        return sorted(action_id for action_id, action in actions.items()
                      if action_id not in done and set(action["prerequisites"]) <= done)

    next_actions = [recommendation(action_id) for action_id in eligible(completed)]
    next_actions.sort(key=lambda action: (-action["score"], action["action_id"]))
    candidates = []
    for first in next_actions:
        for second_id in eligible(completed | {first["action_id"]}):
            second = recommendation(second_id)
            candidates.append((first, second))
    candidates.sort(key=lambda pair: (
        -(pair[0]["score"] + pair[1]["score"]), -pair[0]["score"],
        pair[0]["action_id"], pair[1]["action_id"]))
    steps = []
    if candidates:
        steps = [{"step": index, **action}
                 for index, action in enumerate(candidates[0], 1)]
    return {
        "schema_version": VERSION, "stage": "journey",
        "status": "ready" if steps else "blocked", "next_actions": next_actions,
        "steps": steps,
        "reason": None if steps else "Fewer than two sequentially eligible uncompleted actions.",
        "ranking_policy": "Sum of per-step distinct supporting feedback counts; "
                          "ties by first-step score then action IDs. Zero-score fallback allowed.",
    }


def _same(actual, expected, path):
    """Type-strict recursive comparison also rejects fabricated or stale evidence."""
    require(type(actual) is type(expected), path + " has an invalid type")
    if type(expected) is dict:
        keys(actual, expected, path)
        for key in expected:
            _same(actual[key], expected[key], path + "." + key)
    elif type(expected) is list:
        require(len(actual) == len(expected), path + " has an invalid length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, path + "[" + str(index) + "]")
    else:
        require(actual == expected, path + " violates the validated handoff")


def validate(stage, payload, data=None, previous=None):
    """One validation boundary for input and all stage handoffs.

    In this bounded deterministic reference, output validation rederives expected
    records from the preceding validated stage. It checks all fields, references,
    spans, deduplication, ranking and prerequisite order, not just JSON shape.
    """
    if stage == "input":
        _validate_input(payload)
        return payload
    require(data is not None, "input context is required")
    _validate_input(data)
    if stage == "extraction":
        expected = {"schema_version": VERSION, "stage": "extraction",
                    "documents": _extraction_records(data)}
    elif stage == "feedback":
        require(previous is not None, "extraction handoff is required")
        validate("extraction", previous, data)
        expected = _feedback_records(previous, data)
    elif stage == "journey":
        require(type(previous) is tuple and len(previous) == 2,
                "extraction and feedback handoffs are required")
        extraction, feedback = previous
        validate("feedback", feedback, data, extraction)
        expected = _journey_records(feedback, data)
    elif stage == "result":
        keys(payload, ("schema_version", "status", "fixture_label", "extraction",
                       "feedback", "journey"), "result")
        _same(payload["schema_version"], VERSION, "result.schema_version")
        _same(payload["status"], "ok", "result.status")
        _same(payload["fixture_label"], data["fixture_label"], "result.fixture_label")
        validate("journey", payload["journey"], data,
                 (payload["extraction"], payload["feedback"]))
        return payload
    else:
        raise ValidationError("unknown validation stage")
    _same(payload, expected, stage)
    return payload


def extract(data):
    validate("input", data)
    result = {"schema_version": VERSION, "stage": "extraction",
              "documents": _extraction_records(data)}
    return validate("extraction", result, data)


def analyze_feedback(extraction, data):
    validate("extraction", extraction, data)
    return validate("feedback", _feedback_records(extraction, data), data, extraction)


def recommend_journey(feedback, extraction, data):
    validate("feedback", feedback, data, extraction)
    return validate("journey", _journey_records(feedback, data), data, (extraction, feedback))


def run_pipeline(data):
    extraction = extract(data)
    feedback = analyze_feedback(extraction, data)
    journey = recommend_journey(feedback, extraction, data)
    return validate("result", {
        "schema_version": VERSION, "status": "ok", "fixture_label": data["fixture_label"],
        "extraction": extraction, "feedback": feedback, "journey": journey,
    }, data)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError("nonstandard JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], "r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_json_object,
                             parse_constant=_invalid_constant)
        output = run_pipeline(data)
    except (ValidationError, OSError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
