"""Offline, synthetic-only healthcare evidence-to-support reference pipeline.

Run: python -B implementation.py example_input.json
This bounded de-identification demonstration is not HIPAA certification, a clinical
decision system, or a production PHI detector. Never supply real patient data.
"""

import copy
import json
import math
import re
import sys
from pathlib import Path


VERSION = "1.0"
STAGES = ("intake", "deep", "extract", "support", "sentiment")
POLICY = {
    "synthetic_only": True,
    "human_review_required": True,
    "autonomous_clinical_decisions": False,
    "deidentification": "Direct identifiers removed; known values redacted; residual pattern screening.",
    "limitation": "Demonstrative rules only; not HIPAA certification or comprehensive free-text PHI detection.",
}
IDENTIFIER_KEYS = {"name", "birthDate", "identifier", "telecom", "address"}
RESOURCE_PATHS = {
    "Condition": {"resourceType", "id", "subject.reference", "code.text", "clinicalStatus.text"},
    "Observation": {"resourceType", "id", "subject.reference", "code.text",
                    "valueQuantity.value", "valueQuantity.unit", "status"},
    "Claim": {"resourceType", "id", "patient.reference", "status", "use",
              "type.text", "priority.text"},
}
PHI_PATTERN = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{3}-\d{2}-\d{4}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"(?<!\w)(?:\+?1[- .]?)?\(?\d{3}\)?[- .]\d{3}[- .]\d{4}\b|"
    r"\b(?:MRN|medical record number|patient name|date of birth)\s*[:=]\s*(?!\[REDACTED\])\S+",
    re.I,
)
URGENT = ("chest pain", "can't breathe", "cannot breathe", "suicidal", "severe bleeding")
DEADLINES = ("deadline", "appeal due", "running out", "urgent")
LEXICON = {"angry": -3, "frustrated": -2, "terrible": -3, "bad": -1,
           "worried": -2, "delay": -1, "delayed": -1, "denied": -2,
           "helpful": 2, "great": 2, "thanks": 1, "thank": 1, "happy": 2}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact(obj, required, optional=()):
    require(isinstance(obj, dict), "Expected an object")
    require(set(required) <= obj.keys(), "Missing required object keys")
    require(obj.keys() <= set(required) | set(optional), "Unknown object keys")


def text(value, limit=10000):
    require(isinstance(value, str) and 0 < len(value) <= limit, "Expected bounded nonempty text")
    return value


def leaves(obj, prefix=""):
    require(isinstance(obj, dict), "Resource must contain objects and scalar leaves")
    for key, value in obj.items():
        require(isinstance(key, str), "Object keys must be strings")
        path = prefix + "." + key if prefix else key
        if isinstance(value, dict):
            require(bool(value), "Empty resource objects are unsupported")
            yield from leaves(value, path)
        else:
            require(isinstance(value, (str, int, float, bool)) and value is not None,
                    "Resource arrays/nulls are unsupported in this bounded schema")
            require(not isinstance(value, float) or math.isfinite(value), "Non-finite number")
            yield path, value


def all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_strings(item)


def clean(value, redactions):
    if isinstance(value, str):
        for token in sorted(redactions, key=len, reverse=True):
            value = re.sub(re.escape(token), lambda _: "[REDACTED]", value, flags=re.I)
        require(PHI_PATTERN.search(value) is None, "Possible residual identifier; use synthetic de-identified data")
        return value
    if isinstance(value, dict):
        return {k: clean(v, redactions) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v, redactions) for v in value]
    return value


def render(resource):
    return "\n".join(path + ": " + json.dumps(value, ensure_ascii=False, allow_nan=False)
                     for path, value in leaves(resource))


def validate_schema(schema):
    require(isinstance(schema, list) and 0 < len(schema) <= 30, "Expected 1-30 field definitions")
    names = set()
    for field in schema:
        exact(field, {"name", "type", "required", "sources"})
        require(re.fullmatch(r"[a-z][a-z0-9_]{0,49}", text(field["name"])) is not None,
                "Invalid field name")
        require(field["name"] not in names, "Duplicate field name")
        names.add(field["name"])
        require(field["type"] in ("string", "number", "boolean"), "Unsupported field type")
        require(type(field["required"]) is bool, "required must be boolean")
        require(isinstance(field["sources"], list) and 0 < len(field["sources"]) <= 10,
                "Expected field source selectors")
        for selector in field["sources"]:
            if "label" in selector:
                exact(selector, {"label"})
                require(re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,60}", text(selector["label"])) is not None,
                        "Invalid note label")
            else:
                exact(selector, {"resource_type", "path"})
                require(selector["resource_type"] in RESOURCE_PATHS, "Unsupported resource selector")
                require(selector["path"] in RESOURCE_PATHS[selector["resource_type"]] -
                        {"id", "resourceType", "subject.reference", "patient.reference"},
                        "Unsupported field path")


def audit(record, stage, record_id, changed_fields, action="derive"):
    record["audit"].append({
        "event_id": len(record["audit"]) + 1,
        "stage": stage,
        "record_id": record_id,
        "action": action,
        "changed_fields": changed_fields,
        "human_review_required": True,
    })


def intake(data):
    exact(data, {"schema_version", "synthetic", "patient", "documents", "field_schema", "request"})
    require(data["schema_version"] == VERSION, "Unsupported schema version")
    require(data["synthetic"] is True, "Only explicitly synthetic fixtures are accepted")
    patient = data["patient"]
    exact(patient, {"resourceType", "id"}, IDENTIFIER_KEYS | {"gender"})
    require(patient["resourceType"] == "Patient", "Expected FHIR-style Patient")
    original_id = text(patient["id"], 100)
    require(re.fullmatch(r"synthetic-patient-[a-z0-9-]+", original_id) is not None,
            "Patient ID must be explicitly synthetic")
    require(patient.get("gender", "unknown") in ("male", "female", "other", "unknown"),
            "Invalid patient gender")
    redactions = {s for k in IDENTIFIER_KEYS for s in all_strings(patient.get(k, [])) if s}
    require(all(len(s) >= 3 for s in redactions), "Identifier tokens must be at least 3 characters")
    safe_patient = {"resourceType": "Patient", "id": "patient-001",
                    "gender": patient.get("gender", "unknown")}
    request = data["request"]
    exact(request, {"message", "team_online"})
    text(request["message"])
    require(type(request["team_online"]) is bool, "team_online must be boolean")
    require(isinstance(data["documents"], list) and 0 < len(data["documents"]) <= 50,
            "Expected 1-50 documents")
    documents = []
    seen = set()
    for index, doc in enumerate(data["documents"], 1):
        exact(doc, {"id", "format"}, {"resource", "text"})
        require(re.fullmatch(r"synthetic-source-[a-z0-9-]+", text(doc["id"], 100)) is not None,
                "Source ID must be explicitly synthetic")
        require(doc["id"] not in seen, "Duplicate source ID")
        seen.add(doc["id"])
        source_id = "source-" + str(index).zfill(3)
        if doc["format"] == "fhir":
            exact(doc, {"id", "format", "resource"})
            resource = copy.deepcopy(doc["resource"])
            require(isinstance(resource, dict), "Expected resource object")
            kind = resource.get("resourceType")
            require(kind in RESOURCE_PATHS, "Unsupported FHIR-style resource")
            require(set(dict(leaves(resource))) <= RESOURCE_PATHS[kind], "Unsupported resource properties")
            require(isinstance(resource.get("id"), str), "Resource needs an ID")
            ref_key = "patient" if kind == "Claim" else "subject"
            require(resource.get(ref_key) == {"reference": "Patient/" + original_id},
                    "Resource references an unknown patient")
            resource["id"] = source_id
            resource[ref_key] = {"reference": "Patient/patient-001"}
            resource = clean(resource, redactions)
            documents.append({"id": source_id, "format": "fhir",
                              "resource": resource, "text": render(resource)})
        else:
            require(doc["format"] == "clinical_note", "Unsupported document format")
            exact(doc, {"id", "format", "text"})
            documents.append({"id": source_id, "format": "clinical_note",
                              "text": clean(text(doc["text"]), redactions)})
    validate_schema(data["field_schema"])
    schema = clean(copy.deepcopy(data["field_schema"]), redactions)
    message = request["message"].replace(original_id, "patient-001")
    for doc in documents:
        doc["text"] = doc["text"].replace(original_id, "patient-001")
    record = {"schema_version": VERSION, "status": "ok", "stage": "intake",
              "patient": safe_patient, "documents": documents, "field_schema": schema,
              "request": {"message": clean(message, redactions), "team_online": request["team_online"]},
              "policy": copy.deepcopy(POLICY), "audit": [], "payload": {}}
    audit(record, "intake", "patient-001", ["id", "direct_identifiers"], "deidentify")
    for doc in documents:
        audit(record, "intake", doc["id"], ["id", "content"], "deidentify")
    audit(record, "intake", "request", ["message", "team_online"], "deidentify")
    audit(record, "intake", "field_schema", ["definitions"], "validate")
    validate(record, "intake")
    return record


def convert(value, kind):
    if kind == "string":
        require(isinstance(value, str) and bool(value.strip()), "Expected nonempty string field")
        return value.strip()
    if kind == "boolean":
        if type(value) is bool:
            return value
        require(isinstance(value, str) and value.lower().strip() in ("true", "false"),
                "Expected boolean field")
        return value.lower().strip() == "true"
    require(not isinstance(value, bool) and isinstance(value, (str, int, float)), "Expected numeric field")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValidationError("Expected numeric field") from None
    require(math.isfinite(result), "Expected finite numeric field")
    return result


def canonical(value):
    return value.casefold() if isinstance(value, str) else json.dumps(value, sort_keys=True)


def research_payload(record):
    evidence = []
    for field in record["field_schema"]:
        seen_spans = set()
        for doc in record["documents"]:
            for selector in field["sources"]:
                if "label" in selector and doc["format"] == "clinical_note":
                    pattern = re.compile(r"^[ \t]*" + re.escape(selector["label"]) +
                                         r"[ \t]*:[ \t]*([^\r\n]+)", re.I | re.M)
                    matches = [(m.start(1), m.end(1), m.group(1)) for m in pattern.finditer(doc["text"])]
                elif (doc["format"] == "fhir" and
                      selector.get("resource_type") == doc["resource"]["resourceType"]):
                    path = selector["path"]
                    pattern = re.compile(r"^" + re.escape(path) + r": (.+)$", re.M)
                    matches = [(m.start(1), m.end(1), json.loads(m.group(1)))
                               for m in pattern.finditer(doc["text"])]
                else:
                    continue
                for start, end, value in matches:
                    key = (doc["id"], start, end)
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    evidence.append({"id": "e" + str(len(evidence) + 1).zfill(4),
                                     "field": field["name"], "source_id": doc["id"],
                                     "span": {"start": start, "end": end},
                                     "quote": doc["text"][start:end], "value": value})
    findings, disagreements, questions = [], [], []
    for field in record["field_schema"]:
        facts = [e for e in evidence if e["field"] == field["name"]]
        values, invalid = {}, []
        for fact in facts:
            try:
                value = convert(fact["value"], field["type"])
                values.setdefault(canonical(value), value)
            except ValidationError:
                invalid.append(fact["id"])
        finding = {"field": field["name"], "values": list(values.values()),
                   "evidence_ids": [f["id"] for f in facts], "invalid_evidence_ids": invalid}
        findings.append(finding)
        if len(values) > 1:
            disagreements.append({"field": field["name"], "values": list(values.values()),
                                  "evidence_ids": finding["evidence_ids"]})
            questions.append("Human review: reconcile conflicting " + field["name"] + ".")
        if not facts:
            questions.append("What is the missing " + field["name"] + "?")
        if invalid:
            questions.append("Human review: verify invalid " + field["name"] + " evidence.")
    return {"evidence": evidence, "findings": findings, "disagreements": disagreements,
            "unresolved_questions": questions, "sources_reviewed": [d["id"] for d in record["documents"]]}


def extraction_payload(record):
    research = record["payload"]["deep"]
    result, missing = {}, []
    for spec, finding in zip(record["field_schema"], research["findings"]):
        values = finding["values"]
        state = ("invalid" if finding["invalid_evidence_ids"] else
                 "missing" if not values else "conflict" if len(values) > 1 else "present")
        if state == "missing" and spec["required"]:
            missing.append(spec["name"])
        result[spec["name"]] = {"status": state, "type": spec["type"], "required": spec["required"],
                                "value": values[0] if state == "present" else None,
                                "candidates": values, "evidence_ids": finding["evidence_ids"]}
    return {"fields": result, "missing_required": missing,
            "unresolved_questions": research["unresolved_questions"],
            "human_review_required": True}


def signals(message, phrases):
    lower = message.lower()
    return [phrase for phrase in phrases if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lower)]


def support_payload(record):
    extracted = record["payload"]["extract"]
    request = record["request"]
    urgent = signals(request["message"], URGENT)
    clinical = bool(re.search(r"\b(dose|diagnos\w*|treat\w*|medicat\w*|symptom\w*|clinical)\b",
                              request["message"], re.I)) or bool(urgent)
    answer = ["I can help with record information and administrative next steps."]
    if not request["team_online"]:
        answer.append("The team is offline. This response does not contact them or create a ticket.")
    citations, issues = [], []
    for name, field in extracted["fields"].items():
        if field["status"] == "present":
            answer.append(name + ": " + str(field["value"]) + " (reported in the supplied synthetic records).")
            citations.append({"field": name, "evidence_ids": field["evidence_ids"]})
        elif field["status"] in ("conflict", "invalid") or field["required"]:
            answer.append(name + " is " + field["status"] + "; a human must verify it.")
            issues.append(name + ":" + field["status"])
    if urgent:
        answer.append("Your message contains a potential emergency signal. If this describes a current "
                      "emergency, contact local emergency services now; do not wait for this service.")
    if clinical:
        answer.append("I cannot diagnose, recommend treatment, change medication, or determine medical necessity. "
                      "Please ask a licensed clinician.")
    answer.append("A human must review this response. Contact your care or authorization team through "
                  "its established channel for corrections or authorization decisions; no approval is implied.")
    return {"answer": " ".join(answer), "citations": citations,
            "customer_message": request["message"], "team_online": request["team_online"],
            "issues": issues, "urgent_signals": urgent, "clinical_boundary_applied": clinical,
            "human_review_required": True, "action_taken": "none",
            "escalation_recommended": bool(urgent or clinical or issues or not request["team_online"])}


def sentiment_payload(record):
    support = record["payload"]["support"]
    message = support["customer_message"]
    tokens = list(re.finditer(r"\b[\w']+\b", message.lower()))
    matches = []
    for i, token in enumerate(tokens):
        word = token.group()
        if word not in LEXICON:
            continue
        preceding = tokens[max(0, i - 3):i]
        negated = any(t.group() in ("not", "no", "never", "isn't", "wasn't")
                      and not re.search(r"[.!?;,]", message[t.end():token.start()])
                      for t in preceding)
        matches.append({"token": word, "start": token.start(), "end": token.end(),
                        "base_weight": LEXICON[word], "negated": negated,
                        "weight": -LEXICON[word] if negated else LEXICON[word]})
    raw = sum(m["weight"] for m in matches)
    score = max(-10, min(10, raw))
    deadlines = signals(message, DEADLINES)
    reasons = []
    if support["urgent_signals"]:
        priority, severity = "P1", "potential_emergency"
        reasons.append("Potential emergency phrase; human triage, not a diagnosis")
    elif deadlines or support["issues"] or score <= -4:
        priority, severity = "P2", "time_sensitive_or_unresolved"
        if deadlines:
            reasons.append("Time-sensitive language")
        if support["issues"]:
            reasons.append("Unresolved extracted record fields")
        if score <= -4:
            reasons.append("Strong negative sentiment")
    else:
        priority, severity = "P3", "routine"
        reasons.append("No configured high-severity signal")
    return {"scored_text": message, "score": score, "unclamped_score": raw,
            "label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
            "matches": matches, "severity": severity, "priority": priority,
            "priority_reasons": reasons, "deadline_signals": deadlines,
            "urgent_signals": support["urgent_signals"], "issues": support["issues"],
            "human_review_required": True,
            "limitation": "Deterministic lexicon and conservative phrase rules; not clinical triage."}


BUILDERS = {"deep": research_payload, "extract": extraction_payload,
            "support": support_payload, "sentiment": sentiment_payload}


def validate(record, expected_stage):
    """One validation gate for every handoff, including deterministic provenance checks."""
    exact(record, {"schema_version", "status", "stage", "patient", "documents", "field_schema",
                   "request", "policy", "audit", "payload"})
    require(expected_stage in STAGES and record["stage"] == expected_stage, "Unexpected pipeline stage")
    require(record["schema_version"] == VERSION and record["status"] == "ok", "Invalid envelope")
    require(record["policy"] == POLICY, "Healthcare policy must be preserved")
    exact(record["patient"], {"resourceType", "id", "gender"})
    require(record["patient"]["resourceType"] == "Patient" and record["patient"]["id"] == "patient-001",
            "Patient must be de-identified")
    validate_schema(record["field_schema"])
    exact(record["request"], {"message", "team_online"})
    text(record["request"]["message"])
    require(type(record["request"]["team_online"]) is bool, "Invalid availability")
    require(isinstance(record["documents"], list) and 0 < len(record["documents"]) <= 50,
            "Invalid documents")
    for index, doc in enumerate(record["documents"], 1):
        require(doc["id"] == "source-" + str(index).zfill(3), "Invalid sanitized source ID")
        if doc["format"] == "fhir":
            exact(doc, {"id", "format", "resource", "text"})
            resource = doc["resource"]
            kind = resource.get("resourceType")
            require(kind in RESOURCE_PATHS, "Unsupported resource")
            require(set(dict(leaves(resource))) <= RESOURCE_PATHS[kind], "Unsupported resource fields")
            key = "patient" if kind == "Claim" else "subject"
            require(resource.get(key) == {"reference": "Patient/patient-001"}, "Invalid patient reference")
            require(resource.get("id") == doc["id"] and render(resource) == doc["text"],
                    "Resource rendering does not match")
        else:
            require(doc["format"] == "clinical_note", "Invalid source format")
            exact(doc, {"id", "format", "text"})
        text(doc["text"], 100000)
    require(clean(record, set()) == record, "Residual identifiers")
    count = STAGES.index(expected_stage)
    require(isinstance(record["payload"], dict) and
            set(record["payload"]) == set(STAGES[1:count + 1]), "Invalid stage payloads")
    for stage in STAGES[1:count + 1]:
        require(record["payload"][stage] == BUILDERS[stage](record),
                "Invalid " + stage + " result or provenance")
    expected_audit = []
    scratch = {"audit": expected_audit}
    audit(scratch, "intake", "patient-001", ["id", "direct_identifiers"], "deidentify")
    for doc in record["documents"]:
        audit(scratch, "intake", doc["id"], ["id", "content"], "deidentify")
    audit(scratch, "intake", "request", ["message", "team_online"], "deidentify")
    audit(scratch, "intake", "field_schema", ["definitions"], "validate")
    for stage in STAGES[1:count + 1]:
        audit(scratch, stage, "pipeline", ["stage", "payload." + stage])
    require(record["audit"] == expected_audit, "Incomplete or altered audit trail")
    return record


def advance(previous, stage):
    require(stage in BUILDERS, "Unknown stage")
    validate(previous, STAGES[STAGES.index(stage) - 1])
    result = copy.deepcopy(previous)
    result["payload"][stage] = BUILDERS[stage](result)
    result["stage"] = stage
    audit(result, stage, "pipeline", ["stage", "payload." + stage])
    return validate(result, stage)


def run(data):
    result = intake(data)
    for stage in STAGES[1:]:
        result = advance(result, stage)
    return result


def reject_constant(_):
    raise ValidationError("Non-finite JSON constants are forbidden")


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "Duplicate JSON key")
        obj[key] = value
    return obj


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 1_000_000, "Input exceeds 1 MB limit")
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                          object_pairs_hook=unique_object)
        result = run(data)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError,
            RecursionError, OverflowError):
        # Do not echo paths, input data, or exceptions that might contain identifiers.
        print(json.dumps({"status": "error", "error": "Invalid input, file, or pipeline validation failure."}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
