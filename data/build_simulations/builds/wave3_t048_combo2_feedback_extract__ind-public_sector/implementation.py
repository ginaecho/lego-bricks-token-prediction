"""Synthetic public-sector feedback-to-extraction reference pipeline.

Standard library only. Spans are Python character offsets into the redacted
source text, never into original citizen records. No eligibility decisions.
"""

import copy
import json
import re
import sys
from pathlib import Path


VERSION = "1.0"
ENTITIES = {"citizen_service_request", "benefits_application", "policy_document"}
PII_KEYS = {"name", "address", "email", "phone", "national_id"}
THEMES = {
    "Service delay": ("wait", "delay", "slow"),
    "Clear information": ("confus", "unclear", "explain"),
    "Accessible service": ("accessib", "screen reader", "large print"),
}
NOTICE = (
    "Synthetic demonstration only. Personal details are removed before analysis. "
    "No benefit or service decision is made. A person must review these results."
)


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact_keys(value, keys, message):
    require(isinstance(value, dict) and set(value) == set(keys), message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def validate(value, stage):
    """One validation boundary used by input, feedback and extraction."""
    require(isinstance(value, dict), "Expected a JSON object.")
    if stage == "input":
        exact_keys(value, {"schema_version", "synthetic", "citizen_pii",
                           "documents", "fields"}, "Input keys do not match the schema.")
        require(value["schema_version"] == VERSION, "Unsupported schema version.")
        require(value["synthetic"] is True, "Only clearly labeled synthetic data is accepted.")
        require(isinstance(value["citizen_pii"], list), "citizen_pii must be a list.")
        for persona in value["citizen_pii"]:
            require(isinstance(persona, dict) and bool(persona)
                    and set(persona) <= PII_KEYS, "Invalid synthetic personal-detail declaration.")
            require(all(text(v) and len(v.strip()) >= 3 for v in persona.values()),
                    "Personal-detail values must contain at least three characters.")
        require(isinstance(value["documents"], list) and 0 < len(value["documents"]) <= 100,
                "Provide between 1 and 100 documents.")
        ids = set()
        for doc in value["documents"]:
            exact_keys(doc, {"id", "entity", "format", "content"}, "Invalid document keys.")
            require(text(doc["id"]) and re.fullmatch(r"doc-[0-9]{1,6}", doc["id"]),
                    "Document IDs must use doc- followed by digits.")
            require(doc["id"] not in ids, "Document IDs must be unique.")
            ids.add(doc["id"])
            require(isinstance(doc["entity"], str) and doc["entity"] in ENTITIES,
                    "Unknown document entity.")
            expected = "policy_text" if doc["entity"] == "policy_document" else "government_form_json"
            require(doc["format"] == expected, "The document format does not match its entity.")
            if expected == "policy_text":
                require(text(doc["content"]) and len(doc["content"]) <= 20000,
                        "Policy text must be nonempty and at most 20000 characters.")
            else:
                require(isinstance(doc["content"], dict) and 0 < len(doc["content"]) <= 100,
                        "A government form must contain 1 to 100 text fields.")
                for key, val in doc["content"].items():
                    require(text(key) and re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,59}", key),
                            "Invalid form field label.")
                    require(isinstance(val, str) and len(val) <= 2000
                            and "\n" not in val and "\r" not in val,
                            "Form values must be single-line strings of at most 2000 characters.")
        require(isinstance(value["fields"], list) and 0 < len(value["fields"]) <= 50,
                "Provide between 1 and 50 extraction fields.")
        names = set()
        for field in value["fields"]:
            exact_keys(field, {"name", "entity", "label", "required"}, "Invalid extraction field keys.")
            require(text(field["name"]) and re.fullmatch(r"[a-z][a-z0-9_]{0,59}", field["name"]),
                    "Invalid extraction field name.")
            require(field["name"] not in names, "Extraction field names must be unique.")
            names.add(field["name"])
            require(isinstance(field["entity"], str) and field["entity"] in ENTITIES,
                    "Unknown extraction field entity.")
            require(text(field["label"]) and re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,59}", field["label"]),
                    "Invalid extraction field label.")
            require(type(field["required"]) is bool, "required must be a boolean.")
        return value

    require(stage in {"feedback", "extraction"}, "Unknown validation stage.")
    exact_keys(value, {"schema_version", "synthetic", "status", "notice",
                       "fields", "feedback", "extraction"}, "Invalid pipeline envelope.")
    require(value["schema_version"] == VERSION and value["synthetic"] is True
            and value["status"] == "ok" and value["notice"] == NOTICE,
            "Invalid pipeline metadata.")
    feedback = value["feedback"]
    exact_keys(feedback, {"sources", "themes", "duplicate_count", "span_basis"}, "Invalid feedback output.")
    require(feedback["span_basis"] == "redacted source text; zero-based, end-exclusive characters",
            "Invalid span coordinate system.")
    require(isinstance(feedback["sources"], list) and feedback["sources"], "No analyzed sources.")
    by_id = {}
    aliases = []
    for source in feedback["sources"]:
        exact_keys(source, {"id", "entity", "text", "original_ids"}, "Invalid analyzed source.")
        require(source["entity"] in ENTITIES and text(source["text"]), "Invalid analyzed text.")
        require(isinstance(source["original_ids"], list) and source["original_ids"]
                and source["id"] == source["original_ids"][0], "Invalid duplicate trace.")
        require(all(isinstance(i, str) and re.fullmatch(r"doc-[0-9]{1,6}", i)
                    for i in source["original_ids"]), "Invalid source identifier.")
        aliases.extend(source["original_ids"])
        by_id[source["id"]] = source
    require(len(aliases) == len(set(aliases)), "Source identifiers overlap.")
    require(type(feedback["duplicate_count"]) is int
            and feedback["duplicate_count"] == len(aliases) - len(by_id), "Invalid duplicate count.")
    # Reuse input field rules rather than maintain a second field schema.
    validate({"schema_version": VERSION, "synthetic": True, "citizen_pii": [],
              "documents": [{"id": "doc-0", "entity": "policy_document",
                             "format": "policy_text", "content": "Schema check"}],
              "fields": value["fields"]}, "input")
    require(isinstance(feedback["themes"], list), "Invalid theme list.")
    seen = set()
    for theme in feedback["themes"]:
        exact_keys(theme, {"name", "explanation", "excerpts"}, "Invalid theme record.")
        require(theme["name"] in set(THEMES) | {"Other feedback"} and theme["name"] not in seen,
                "Invalid or repeated theme.")
        seen.add(theme["name"])
        require(text(theme["explanation"]) and isinstance(theme["excerpts"], list)
                and theme["excerpts"], "A theme needs a plain-language explanation and evidence.")
        for excerpt in theme["excerpts"]:
            exact_keys(excerpt, {"source_id", "start", "end", "text"}, "Invalid excerpt.")
            validate_span(excerpt, by_id)
    if stage == "feedback":
        require(value["extraction"] is None, "Feedback must precede extraction.")
    else:
        require(value["extraction"] == extraction_records(value),
                "Extraction does not match the validated feedback sources and schema.")
    return value


def validate_span(span, sources):
    require(span["source_id"] in sources, "An excerpt refers to an unknown source.")
    source_text = sources[span["source_id"]]["text"]
    require(type(span["start"]) is int and type(span["end"]) is int
            and 0 <= span["start"] < span["end"] <= len(source_text), "Invalid source span.")
    require(span["text"] == source_text[span["start"]:span["end"]], "Source span text does not match.")


def redact(raw, personas):
    values = sorted({v for p in personas for v in p.values()}, key=lambda v: (-len(v), v))
    for val in values:
        raw = re.sub(re.escape(val), "[REDACTED]", raw, flags=re.IGNORECASE)
    patterns = [
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"(?<!\w)(?:\+1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)",
    ]
    for pattern in patterns:
        raw = re.sub(pattern, "[REDACTED]", raw, flags=re.IGNORECASE)
    # Explicitly labeled PII is protected even when absent from the declaration.
    raw = re.sub(
        r"(?im)^((?:citizen name|name|address|email|phone|national id|national_id)\s*:)[^\n]*",
        r"\1 [REDACTED]", raw)
    return raw


def analyze_feedback(payload):
    validate(payload, "input")
    sources, lookup = [], {}
    for doc in payload["documents"]:
        content = doc["content"]
        raw = content if isinstance(content, str) else "\n".join(
            f"{key}: {val}" for key, val in sorted(content.items()))
        sanitized = redact(raw, payload["citizen_pii"])
        # Entity is part of the identity: unrelated kinds of public records never merge.
        key = (doc["entity"], " ".join(sanitized.casefold().split()))
        if key in lookup:
            lookup[key]["original_ids"].append(doc["id"])
        else:
            source = {"id": doc["id"], "entity": doc["entity"], "text": sanitized,
                      "original_ids": [doc["id"]]}
            sources.append(source)
            lookup[key] = source
    grouped = {}
    for source in sources:
        for match in re.finditer(r"[^\n]+", source["text"]):
            line = match.group()
            matched = [name for name, keywords in THEMES.items()
                       if any(re.search(r"\b" + re.escape(word), line, re.IGNORECASE)
                              for word in keywords)]
            # Unclassified form metadata is not treated as citizen sentiment.
            if not matched and line.lower().startswith("feedback:"):
                matched = ["Other feedback"]
            for name in matched:
                grouped.setdefault(name, []).append({
                    "source_id": source["id"], "start": match.start(),
                    "end": match.end(), "text": line})
    themes = [{"name": name,
               "explanation": ("These excerpts contain a matching topic word; "
                               "this is not a finding about the citizen.")
               if name != "Other feedback" else "This feedback does not match a listed topic.",
               "excerpts": excerpts} for name, excerpts in sorted(grouped.items())]
    output = {
        "schema_version": VERSION, "synthetic": True, "status": "ok", "notice": NOTICE,
        "fields": copy.deepcopy(payload["fields"]),
        "feedback": {"sources": sources, "themes": themes,
                     "duplicate_count": len(payload["documents"]) - len(sources),
                     "span_basis": "redacted source text; zero-based, end-exclusive characters"},
        "extraction": None,
    }
    return validate(output, "feedback")


def extraction_records(envelope):
    records = []
    for source in envelope["feedback"]["sources"]:
        fields = []
        for spec in envelope["fields"]:
            if spec["entity"] != source["entity"]:
                continue
            pattern = r"(?im)^" + re.escape(spec["label"]) + r"[ \t]*:[ \t]*([^\n]*)"
            matches = list(re.finditer(pattern, source["text"]))
            item = {"name": spec["name"], "required": spec["required"],
                    "value": None, "span": None, "reason": None}
            if not matches:
                item["reason"] = "not_found"
            elif len(matches) > 1:
                item["reason"] = "ambiguous"
            else:
                match = matches[0]
                raw = match.group(1)
                val = raw.strip()
                if not val:
                    item["reason"] = "empty"
                elif "[REDACTED]" in val:
                    item["reason"] = "protected_personal_details"
                else:
                    start = match.start(1) + len(raw) - len(raw.lstrip())
                    item["value"] = val
                    item["span"] = {"source_id": source["id"], "start": start,
                                    "end": start + len(val)}
            fields.append(item)
        records.append({
            "source_id": source["id"], "original_ids": list(source["original_ids"]),
            "entity": source["entity"],
            "themes": [t["name"] for t in envelope["feedback"]["themes"]
                       if any(e["source_id"] == source["id"] for e in t["excerpts"])],
            "fields": fields,
            "missing_fields": [f["name"] for f in fields if f["reason"]],
            "review_required": any(f["required"] and f["reason"] for f in fields),
            "explanation": "Values are copied from labeled lines. Missing or unclear values are not guessed.",
        })
    return records


def extract_fields(feedback):
    validate(feedback, "feedback")
    output = copy.deepcopy(feedback)
    output["extraction"] = extraction_records(output)
    return validate(output, "extraction")


def run_pipeline(payload):
    return extract_fields(analyze_feedback(payload))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        result = run_pipeline(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        # Never reflect untrusted file contents, paths or personal details in errors.
        message = str(exc) if isinstance(exc, ValidationError) else "The input file could not be read as JSON."
        print(json.dumps({"schema_version": VERSION, "status": "error", "message": message}))
        return 2
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
