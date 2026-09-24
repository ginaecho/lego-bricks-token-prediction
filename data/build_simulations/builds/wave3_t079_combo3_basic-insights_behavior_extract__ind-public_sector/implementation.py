"""Synthetic public-service pipeline. No eligibility decisions or external services."""

import copy
import datetime as dt
import json
import math
import re
import sys


VERSION = "1.0"
THEMES = {
    "waiting": (("wait", "delay", "slow"), "Explain the expected wait time."),
    "access": (("access", "screen reader", "language"), "Offer clear and accessible ways to get help."),
    "paperwork": (("form", "document", "paperwork"), "Explain which forms and documents are needed."),
}
NOTICE = "Synthetic demonstration only. Rankings do not decide benefit eligibility."


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def timestamp(value):
    require(text(value), "A time value is required.")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("Time values must use ISO 8601.") from None
    require(parsed.tzinfo is not None, "Time values must include a time zone.")
    return parsed


def exact(obj, keys):
    require(isinstance(obj, dict) and set(obj) == set(keys), "Object fields do not match the shared schema.")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def validate_document(doc):
    exact(doc, ("id", "title", "text", "tags"))
    require(text(doc["id"]) and re.fullmatch(r"SYN-P-[0-9]+", doc["id"]), "Use invented policy identifiers.")
    require(text(doc["title"]) and text(doc["text"]), "Policies need a title and text.")
    require(isinstance(doc["tags"], list) and all(isinstance(tag, str) and (tag in THEMES or tag == "other")
                                                for tag in doc["tags"])
            and len(doc["tags"]) == len(set(doc["tags"])), "Policy tags must be unique supported themes.")


def validate_fields(fields):
    require(isinstance(fields, list) and 0 < len(fields) <= 30, "Provide 1 to 30 extraction fields.")
    names, labels = set(), set()
    for field in fields:
        exact(field, ("name", "label", "type", "required"))
        require(text(field["name"]) and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", field["name"]),
                "Field names must be short lower-case identifiers.")
        require(text(field["label"]) and re.fullmatch(r"[A-Za-z][A-Za-z ]{0,59}", field["label"]),
                "Labels must use plain words.")
        require(field["name"] not in names and field["label"].casefold() not in labels,
                "Extraction names and labels must be unique.")
        names.add(field["name"])
        labels.add(field["label"].casefold())
        require(field["type"] in ("string", "integer", "date") and type(field["required"]) is bool,
                "Invalid extraction field definition.")


def validate(value, stage="input", private_values=()):
    """Single validation boundary used for input and every pipeline handoff."""
    require(isinstance(value, dict), "The payload must be an object.")
    require(value.get("schema_version") == VERSION, "Unsupported schema version.")
    require(value.get("synthetic") is True, "Only labeled synthetic fixtures are accepted.")
    if stage != "input":
        exact(value, ("schema_version", "synthetic", "status", "stage", "data", "notice"))
        require(value["status"] == "ok" and value["stage"] == stage, "Invalid stage envelope.")
        require(value["notice"] == NOTICE, "A plain-language demonstration notice is required.")
        data = value["data"]
        require(isinstance(data, dict), "Stage data must be an object.")
        require(not any(p.casefold() in item.casefold() for item in strings(value) for p in private_values),
                "Private data must not cross stage boundaries.")
        exact(data, {
            "insights": ("themes", "documents", "events", "now", "citizen_id", "extraction_schema"),
            "behavior": ("themes", "ranked_documents", "cold_start", "extraction_schema"),
            "extract": ("themes", "rankings", "cold_start", "extractions"),
        }[stage])
        require(isinstance(data["themes"], list), "Themes must be a list.")
        for theme in data["themes"]:
            exact(theme, ("name", "count", "action", "evidence"))
            require(theme["name"] in (*THEMES, "other") and text(theme["action"]), "Themes need plain-language actions.")
            require(isinstance(theme["evidence"], list), "Theme evidence must be a list.")
            require(type(theme["count"]) is int and theme["count"] == len(theme["evidence"]) > 0,
                    "Theme evidence must support its count.")
            for evidence in theme["evidence"]:
                exact(evidence, ("case_id", "text"))
                require(text(evidence["case_id"]) and text(evidence["text"]), "Evidence needs a case and feedback.")
        if stage in ("insights", "behavior"):
            validate_fields(data["extraction_schema"])
        if stage == "insights":
            require(isinstance(data["documents"], list), "Documents must be a list.")
            for doc in data["documents"]:
                validate_document(doc)
            ids = [doc["id"] for doc in data["documents"]]
            require(len(set(ids)) == len(ids), "Document identifiers must be unique.")
            require(isinstance(data["events"], list), "Events must be a list.")
            now = timestamp(data["now"])
            require(text(data["citizen_id"]), "Citizen identifier is required.")
            for event in data["events"]:
                exact(event, ("citizen_id", "document_id", "kind", "at"))
                require(text(event["citizen_id"]) and event["document_id"] in ids
                        and event["kind"] in ("browse", "purchase")
                        and timestamp(event["at"]) <= now, "Invalid activity handoff.")
        elif stage == "behavior":
            require(type(data["cold_start"]) is bool, "Cold-start state must be explicit.")
            require(isinstance(data["ranked_documents"], list), "Ranked documents must be a list.")
            for ranked in data["ranked_documents"]:
                exact(ranked, ("document", "score", "explanation"))
                validate_document(ranked["document"])
                require(type(ranked["score"]) in (int, float) and math.isfinite(ranked["score"])
                        and ranked["score"] >= 0 and text(ranked["explanation"]), "Rankings must be explained.")
        else:
            require(type(data["cold_start"]) is bool, "Cold-start state must be explicit.")
            require(isinstance(data["rankings"], list) and isinstance(data["extractions"], list),
                    "Extraction results must be lists.")
            for result in data["extractions"]:
                exact(result, ("document_id", "source_text", "fields", "missing_fields", "invalid_fields"))
                require(isinstance(result["fields"], dict), "Extracted fields must be an object.")
                for field in result["fields"].values():
                    exact(field, ("value", "source_span", "source_text"))
                    span = field["source_span"]
                    require(isinstance(span, list) and len(span) == 2
                            and all(type(n) is int for n in span)
                            and 0 <= span[0] < span[1] <= len(result["source_text"]),
                            "Source spans must be in bounds.")
                    require(result["source_text"][span[0]:span[1]] == field["source_text"],
                            "Source spans must match the privacy-safe source.")
        return value

    exact(value, ("schema_version", "synthetic", "now", "citizen_id", "personas",
                  "service_requests", "benefits_applications", "policy_documents",
                  "events", "extraction_schema"))
    now = timestamp(value["now"])
    for key in ("personas", "service_requests", "benefits_applications", "policy_documents", "events", "extraction_schema"):
        require(isinstance(value[key], list), "Collection fields must be lists.")
        require(len(value[key]) <= 1000, "Collections are limited to 1000 entries.")
    citizens = set()
    for persona in value["personas"]:
        exact(persona, ("id", "pii"))
        require(text(persona["id"]) and re.fullmatch(r"SYN-C-[0-9]+", persona["id"]), "Use invented citizen identifiers.")
        require(persona["id"] not in citizens, "Citizen identifiers must be unique.")
        citizens.add(persona["id"])
        exact(persona["pii"], ("name", "email", "address"))
        require(all(text(v) and len(v) >= 3 for v in persona["pii"].values()), "Synthetic personal fields are required.")
        require(persona["pii"]["email"].endswith(".invalid")
                and persona["pii"]["address"].startswith("SYNTHETIC "),
                "Use non-traceable synthetic contact details.")
    require(value["citizen_id"] in citizens, "The selected citizen must exist.")
    cases = set()
    for key in ("service_requests", "benefits_applications"):
        for case in value[key]:
            exact(case, ("case_id", "citizen_id", "feedback"))
            require(text(case["case_id"]) and re.fullmatch(r"SYN-[A-Z]+-[0-9]+", case["case_id"]),
                    "Use invented case numbers.")
            require(case["case_id"] not in cases, "Case numbers must be unique.")
            cases.add(case["case_id"])
            require(case["citizen_id"] in citizens and text(case["feedback"]), "Cases need a known citizen and feedback.")
    documents = set()
    for doc in value["policy_documents"]:
        validate_document(doc)
        require(doc["id"] not in documents, "Policy identifiers must be unique.")
        documents.add(doc["id"])
    for event in value["events"]:
        exact(event, ("citizen_id", "document_id", "kind", "at"))
        require(event["citizen_id"] in citizens and event["document_id"] in documents,
                "Events must refer to known citizens and policies.")
        require(event["kind"] in ("browse", "purchase"), "Event kind must be browse or purchase.")
        require(timestamp(event["at"]) <= now, "Future events are not accepted.")
    validate_fields(value["extraction_schema"])
    return value


def secrets(payload):
    return [v for persona in payload["personas"] for v in persona["pii"].values()]


def redact(value, private_values):
    for private in sorted(private_values, key=len, reverse=True):
        value = re.sub(re.escape(private), "[REDACTED]", value, flags=re.IGNORECASE)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]+\b", "[REDACTED]", value)
    return re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)",
                  lambda match: "[REDACTED]" if sum(c.isdigit() for c in match.group()) >= 10 else match.group(),
                  value)


def envelope(stage, data, private_values):
    return validate({"schema_version": VERSION, "synthetic": True, "status": "ok",
                     "stage": stage, "data": data, "notice": NOTICE}, stage, private_values)


def insights(payload):
    validate(payload)
    private = secrets(payload)
    groups = {}
    for case in payload["service_requests"] + payload["benefits_applications"]:
        feedback = redact(case["feedback"], private)
        found = [name for name, (words, _) in THEMES.items()
                 if any(re.search(r"\b" + re.escape(word) + r"\w*\b", feedback, re.I) for word in words)]
        for name in found or ["other"]:
            groups.setdefault(name, []).append({"case_id": case["case_id"], "text": feedback})
    themes = [{"name": name, "count": len(evidence), "evidence": evidence,
               "action": THEMES[name][1] if name in THEMES else "Review this feedback with the service team."}
              for name, evidence in sorted(groups.items())]
    documents = [{**doc, "text": redact(doc["text"], private), "title": redact(doc["title"], private)}
                 for doc in payload["policy_documents"]]
    return envelope("insights", {"themes": themes, "documents": documents,
                                "events": copy.deepcopy(payload["events"]), "now": payload["now"],
                                "citizen_id": payload["citizen_id"],
                                "extraction_schema": copy.deepcopy(payload["extraction_schema"])}, private)


def behavior(previous, private_values=()):
    data = validate(previous, "insights", private_values)["data"]
    now = timestamp(data["now"])
    events = [event for event in data["events"] if event["citizen_id"] == data["citizen_id"]]
    ranked = []
    for doc in data["documents"]:
        matching = [theme for theme in data["themes"] if theme["name"] in doc["tags"]]
        community = sum(theme["count"] for theme in matching)
        history = 0.0
        for event in events:
            if event["document_id"] == doc["id"]:
                days = (now - timestamp(event["at"])).total_seconds() / 86400
                history += (3 if event["kind"] == "purchase" else 1) * 2 ** (-days / 30)
        score = round(community + history, 6)
        explanation = (
            f"Feedback match: {community} points. Recent activity: {history:.6f} points. "
            "Activity weight halves every 30 days; a browse starts at 1 and a purchase at 3. "
            + ("No activity is known for this citizen; use feedback matches. " if not events else "")
            + "Ties use policy identifiers. This is a reading suggestion, not a benefits decision."
        )
        ranked.append({"document": copy.deepcopy(doc), "score": score, "explanation": explanation})
    ranked.sort(key=lambda item: (-item["score"], item["document"]["id"]))
    return envelope("behavior", {"themes": copy.deepcopy(data["themes"]), "ranked_documents": ranked,
                                "cold_start": not events, "extraction_schema": copy.deepcopy(data["extraction_schema"])},
                    private_values)


def extract(previous, private_values=()):
    data = validate(previous, "behavior", private_values)["data"]
    results = []
    for ranked in data["ranked_documents"]:
        doc = ranked["document"]
        source = doc["text"]
        fields, missing, invalid = {}, [], []
        for schema in data["extraction_schema"]:
            pattern = r"^[ \t]*" + re.escape(schema["label"]) + r"[ \t]*:[ \t]*([^\r\n]*?)[ \t]*\r?$"
            matches = list(re.finditer(pattern, source, re.I | re.M))
            if not matches or (len(matches) == 1 and not matches[0].group(1)):
                missing.append({"name": schema["name"], "required": schema["required"]})
                continue
            if len(matches) > 1:
                invalid.append({"name": schema["name"], "reason": "More than one value was found."})
                continue
            match = matches[0]
            raw = match.group(1)
            try:
                if "[REDACTED]" in raw:
                    raise ValueError
                if schema["type"] == "integer":
                    if not re.fullmatch(r"[+-]?[0-9]+", raw):
                        raise ValueError
                    converted = int(raw)
                elif schema["type"] == "date":
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
                        raise ValueError
                    converted = dt.date.fromisoformat(raw).isoformat()
                else:
                    converted = raw
            except ValueError:
                invalid.append({"name": schema["name"], "reason": "Value is private or does not match the requested type."})
                continue
            fields[schema["name"]] = {"value": converted, "source_span": list(match.span(1)), "source_text": raw}
        results.append({"document_id": doc["id"], "source_text": source, "fields": fields,
                        "missing_fields": missing, "invalid_fields": invalid})
    return envelope("extract", {"themes": copy.deepcopy(data["themes"]),
                               "rankings": [{"document_id": item["document"]["id"],
                                             "title": item["document"]["title"], "score": item["score"],
                                             "explanation": item["explanation"]} for item in data["ranked_documents"]],
                               "cold_start": data["cold_start"], "extractions": results}, private_values)


def run(payload):
    first = insights(payload)
    private = secrets(payload)
    return extract(behavior(first, private), private)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "Duplicate JSON keys are not accepted.")
        obj[key] = value
    return obj


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Provide one government form JSON file.")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object)
        result = run(payload)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError):
        print(json.dumps({"status": "error", "message": "Cannot process the file. Check the input schema and synthetic data."}))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
