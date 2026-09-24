"""Synthetic-only administrative healthcare review pipeline; not clinical advice."""

import copy
import hashlib
import json
import re
import sys


BUILD_ID = "wave3_t064_combo3_review_triage_sentiment__ind-healthcare"
STAGES = ("review", "triage", "sentiment")
POSITIVE = {"helpful", "clear", "satisfied", "excellent", "thanks"}
NEGATIVE = {"delay", "delayed", "frustrated", "confusing", "denied", "poor"}
PHI = re.compile(
    r"\b(?:MRN|DOB|born|patient\s+name|birth\s*date|address)\b|"
    r"\b\d{3}-\d{2}-\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|"
    r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://",
    re.I,
)


class ValidationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names):
    need(isinstance(value, dict) and set(value) == set(names.split()),
         "Unexpected or missing schema fields")


def text(value, limit=4000):
    need(isinstance(value, str) and 0 < len(value.strip()) <= limit,
         "Expected bounded nonempty text")
    need(PHI.search(value) is None, "Possible patient identifier rejected")


def slug(value):
    need(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value),
         "Invalid administrative identifier")


def items(value, minimum=0, maximum=100):
    need(isinstance(value, list) and minimum <= len(value) <= maximum,
         "Invalid list size")


def strings(value, minimum=0):
    items(value, minimum)
    for entry in value:
        text(entry)
    need(len(set(value)) == len(value), "Duplicate text entries")


def tokens(value):
    return re.findall(r"[a-z0-9]+", value.lower())


def contains(haystack, needle):
    return (" " + " ".join(tokens(needle)) + " ") in (
        " " + " ".join(tokens(haystack)) + " "
    )


def validate_payload(payload):
    fields(payload, "id patient clinical_note prior_authorization")
    slug(payload["id"])
    patient = payload["patient"]
    fields(patient, "resourceType id synthetic diagnoses labs")
    need(patient["resourceType"] == "Patient" and patient["synthetic"] is True,
         "Only explicitly synthetic Patient resources are accepted")
    need(isinstance(patient["id"], str) and
         re.fullmatch(r"syn-patient-[0-9]+", patient["id"]), "Invalid synthetic patient ID")
    items(patient["diagnoses"])
    for diagnosis in patient["diagnoses"]:
        fields(diagnosis, "code label")
        text(diagnosis["code"], 50)
        text(diagnosis["label"], 200)
    items(patient["labs"])
    for lab in patient["labs"]:
        fields(lab, "code value unit")
        text(lab["code"], 50)
        text(lab["unit"], 50)
        need(type(lab["value"]) in (int, float) and
             -1000000 <= lab["value"] <= 1000000, "Invalid synthetic lab value")
    note = payload["clinical_note"]
    fields(note, "resourceType id synthetic text")
    need(note["resourceType"] == "DocumentReference" and note["synthetic"] is True,
         "Only synthetic clinical note text is accepted")
    slug(note["id"])
    text(note["text"])
    request = payload["prior_authorization"]
    fields(request, "resourceType id status procedure_code evidence")
    need(request["resourceType"] == "Claim" and request["status"] == "draft",
         "Only draft administrative prior authorization requests are accepted")
    slug(request["id"])
    text(request["procedure_code"], 100)
    strings(request["evidence"])


def validate_settings(settings):
    fields(settings, "requirements routing")
    items(settings["requirements"], 1)
    ids = []
    for requirement in settings["requirements"]:
        fields(requirement, "id description evidence_terms")
        slug(requirement["id"])
        text(requirement["description"])
        strings(requirement["evidence_terms"], 1)
        need(all(tokens(term) for term in requirement["evidence_terms"]),
             "Evidence terms must contain words")
        ids.append(requirement["id"])
    need(len(set(ids)) == len(ids), "Duplicate requirement IDs")
    routing = settings["routing"]
    fields(routing, "rules fallback urgent_keywords")
    items(routing["rules"])
    categories = []
    for rule in routing["rules"]:
        fields(rule, "category owner keywords")
        slug(rule["category"])
        slug(rule["owner"])
        strings(rule["keywords"], 1)
        categories.append(rule["category"])
    fields(routing["fallback"], "category owner")
    slug(routing["fallback"]["category"])
    slug(routing["fallback"]["owner"])
    need(len(categories) == len(set(categories)), "Duplicate routing categories")
    strings(routing["urgent_keywords"])
    need(all(tokens(term) for rule in routing["rules"] for term in rule["keywords"])
         and all(tokens(term) for term in routing["urgent_keywords"]),
         "Routing keywords must contain words")


def sources(payload):
    result = [("clinical_note.text", payload["clinical_note"]["text"])]
    result += [
        (f"prior_authorization.evidence[{i}]", value)
        for i, value in enumerate(payload["prior_authorization"]["evidence"])
    ]
    result += [
        (f"patient.diagnoses[{i}]", item["code"] + " " + item["label"])
        for i, item in enumerate(payload["patient"]["diagnoses"])
    ]
    result += [
        (f"patient.labs[{i}]", f'{item["code"]} {item["value"]} {item["unit"]}')
        for i, item in enumerate(payload["patient"]["labs"])
    ]
    return result


def review_result(payload, settings):
    checks = []
    for requirement in settings["requirements"]:
        evidence, missing = [], []
        for term in requirement["evidence_terms"]:
            matches = [path for path, value in sources(payload) if contains(value, term)]
            if matches:
                evidence.append({"term": term, "sources": matches})
            else:
                missing.append(term)
        checks.append({
            "requirement_id": requirement["id"],
            "description": requirement["description"],
            "status": "gap" if missing else "evidence_found",
            "evidence": evidence,
            "missing_terms": missing,
        })
    return {
        "checks": checks,
        "gap_ids": [check["requirement_id"] for check in checks if check["missing_terms"]],
        "assessment": "administrative_evidence_check_only",
    }


def triage_result(payload, review, settings):
    corpus = " ".join([payload["clinical_note"]["text"]] +
                      payload["prior_authorization"]["evidence"] +
                      [check["description"] for check in review["checks"]
                       if check["status"] == "gap"])
    routing = settings["routing"]
    selected = routing["fallback"]
    matched = []
    for rule in routing["rules"]:
        hits = [word for word in rule["keywords"] if contains(corpus, word)]
        if hits:
            selected, matched = rule, hits
            break
    urgent = [word for word in routing["urgent_keywords"] if contains(corpus, word)]
    priority = "urgent" if urgent else ("high" if review["gap_ids"] else "normal")
    return {
        "ticket_id": "ticket-" + payload["id"],
        "category": selected["category"],
        "owner": selected["owner"],
        "priority": priority,
        "gap_ids": review["gap_ids"][:],
        "matched_keywords": matched,
        "urgent_keywords": urgent,
        "routing_policy": "first_matching_rule_else_fallback",
        "decision": "human_administrative_review_required",
    }


def sentiment_result(payload, triage):
    words = tokens(payload["clinical_note"]["text"])
    contributions = []
    for index, word in enumerate(words):
        base = 1 if word in POSITIVE else (-1 if word in NEGATIVE else 0)
        if base:
            negated = index > 0 and words[index - 1] in {"not", "no", "never"}
            contributions.append({
                "token": word, "token_index": index, "negated": negated,
                "weight": -base if negated else base,
            })
    raw = sum(entry["weight"] for entry in contributions)
    score = round(raw / max(1, len(contributions)), 4)
    severity_base = {"normal": 10, "high": 50, "urgent": 80}[triage["priority"]]
    gap_bonus = min(10, len(triage["gap_ids"]) * 2)
    negative_bonus = round(max(0, -score) * 10)
    return {
        "score": score,
        "label": "positive" if score > 0 else ("negative" if score < 0 else "neutral"),
        "contributions": contributions,
        "method": "fixed_lexicon_with_immediate_preceding_negation",
        "administrative_severity": triage["priority"],
        "priority_score": severity_base + gap_bonus + negative_bonus,
        "priority_components": {
            "severity_base": severity_base, "gap_bonus": gap_bonus,
            "negative_bonus": negative_bonus,
        },
        "ticket_id": triage["ticket_id"],
        "owner": triage["owner"],
        "gap_ids": triage["gap_ids"][:],
    }


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def safety():
    return {
        "human_review_required": True,
        "autonomous_clinical_decision": False,
        "deidentification": "synthetic_only_identifier_rejection_demo",
    }


def calculated_stages(payload, settings):
    review = review_result(payload, settings)
    triage = triage_result(payload, review, settings)
    sentiment = sentiment_result(payload, triage)
    return {"review": review, "triage": triage, "sentiment": sentiment}


def snapshot(record, completed):
    return {
        "payload": record["payload"], "safety": record["safety"],
        "stages": {name: record["stages"][name] for name in STAGES[:completed]},
    }


def audit_entry(record, completed):
    return {
        "sequence": completed,
        "actor": "deterministic-administrative-pipeline",
        "action": "ingest" if completed == 0 else STAGES[completed - 1],
        "changed_paths": ["payload", "safety", "stages"] if completed == 0
        else ["stages." + STAGES[completed - 1]],
        "before_sha256": None if completed == 0 else digest(snapshot(record, completed - 1)),
        "after_sha256": digest(snapshot(record, completed)),
    }


def validate(document, phase="input"):
    """Single shared validation boundary for external input and every handoff."""
    need(phase in ("input", "ingested") + STAGES, "Unknown validation phase")
    if phase == "input":
        fields(document, "schema_version synthetic settings records")
    else:
        fields(document, "schema_version synthetic settings records phase status disclaimer")
        need(document["phase"] == phase and document["status"] == "ok",
             "Invalid output phase")
        need(document["disclaimer"] == DISCLAIMER, "Missing demonstration disclaimer")
    need(type(document["schema_version"]) is int and document["schema_version"] == 1,
         "Unsupported schema version")
    need(document["synthetic"] is True, "Input must be explicitly labeled synthetic")
    validate_settings(document["settings"])
    items(document["records"], 1, 100)
    seen = set()
    for record in document["records"]:
        payload = record if phase == "input" else record.get("payload") if isinstance(record, dict) else None
        validate_payload(payload)
        need(payload["id"] not in seen, "Duplicate record ID")
        seen.add(payload["id"])
        if phase == "input":
            continue
        fields(record, "payload safety stages audit")
        need(record["safety"] == safety(), "Human review and clinical safety flags required")
        completed = 0 if phase == "ingested" else STAGES.index(phase) + 1
        need(isinstance(record["stages"], dict) and
             set(record["stages"]) == set(STAGES[:completed]), "Invalid stage handoff")
        expected = calculated_stages(payload, document["settings"])
        for stage in STAGES[:completed]:
            need(record["stages"][stage] == expected[stage],
                 "Stage output does not match validated source evidence")
        need(record["audit"] == [audit_entry(record, n) for n in range(completed + 1)],
             "Record changes require a complete, valid audit chain")
    return document


DISCLAIMER = (
    "SYNTHETIC demonstration only. FHIR-style, not FHIR conformance. "
    "Identifier rejection illustrates HIPAA-oriented de-identification safeguards; "
    "it is not a complete HIPAA de-identification process or compliance certification. "
    "Free text must already be synthetic and de-identified. No diagnosis, treatment, "
    "medical urgency assessment, or authorization approval is performed. Human review required."
)


def ingest(document):
    validate(document)
    output = {
        "schema_version": 1, "synthetic": True,
        "settings": copy.deepcopy(document["settings"]), "records": [],
        "phase": "ingested", "status": "ok", "disclaimer": DISCLAIMER,
    }
    for payload in document["records"]:
        record = {"payload": copy.deepcopy(payload), "safety": safety(), "stages": {}, "audit": []}
        record["audit"].append(audit_entry(record, 0))
        output["records"].append(record)
    return validate(output, "ingested")


def advance(document, stage):
    need(stage in STAGES, "Unknown pipeline stage")
    index = STAGES.index(stage)
    previous = "ingested" if index == 0 else STAGES[index - 1]
    validate(document, previous)
    output = copy.deepcopy(document)
    for record in output["records"]:
        if stage == "review":
            value = review_result(record["payload"], output["settings"])
        elif stage == "triage":
            value = triage_result(record["payload"], record["stages"]["review"], output["settings"])
        else:
            value = sentiment_result(record["payload"], record["stages"]["triage"])
        record["stages"][stage] = value
        record["audit"].append(audit_entry(record, index + 1))
    output["phase"] = stage
    return validate(output, stage)


def run(document):
    output = ingest(document)
    for stage in STAGES:
        output = advance(output, stage)
    return output


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON numbers are forbidden")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        need(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as source:
            document = json.load(source, object_pairs_hook=unique_object,
                                 parse_constant=reject_constant)
        output = run(document)
    except (OSError, ValueError, TypeError, RecursionError, OverflowError):
        # Do not reflect raw input, identifiers, or file paths in error responses.
        print(json.dumps({"status": "error", "error": "Invalid input, schema, or unreadable file"}))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
