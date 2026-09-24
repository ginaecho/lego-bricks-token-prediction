"""Synthetic healthcare discovery -> extractive research reference CLI.

Only administrative/educational products are recommended, never treatments.
The strict synthetic-data profile demonstrates de-identification validation;
it is not a HIPAA certification or a general-purpose free-text PHI detector.
Offsets in citations are Python Unicode character offsets, end-exclusive.
"""

import copy
import hashlib
import json
import math
import re
import sys


SCHEMA = "healthcare-discovery-research/1"
BASE_KEYS = {
    "schema_version", "synthetic", "patient_record", "clinical_notes",
    "prior_authorization_requests", "catalog", "preferences",
}
GENERATED_KEYS = {"discovery", "research", "audit_trail"}
PHI_KEYS = {
    "name", "birthdate", "identifier", "mrn", "address", "telecom",
    "email", "ssn", "photo", "contact",
}
PHI_TEXT = re.compile(
    r"\b(?:MRN|DOB|SSN|patient name)\s*[:=]|\b\d{3}-\d{2}-\d{4}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|[\w.+-]+@[\w.-]+\.\w+|"
    r"\b(?:\+?\d[ -]*){10,}\b", re.I
)
SAFE_ID = re.compile(r"syn-[a-z0-9-]{1,50}\Z")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def shape(value, keys, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(keys), label + " has missing or unsupported fields")


def text(value, label, limit=12000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            label + " must be nonempty bounded text")
    require(not PHI_TEXT.search(value), label + " contains a prohibited identifier pattern")


def strings(values, label):
    require(isinstance(values, list) and len(values) <= 100, label + " must be a bounded list")
    for value in values:
        text(value, label, 120)
    require(len(set(values)) == len(values), label + " contains duplicates")


def phi_check(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower() not in PHI_KEYS, "Direct patient identifiers are prohibited")
            phi_check(child)
    elif isinstance(value, list):
        for child in value:
            phi_check(child)
    elif isinstance(value, str):
        require(not PHI_TEXT.search(value), "Prohibited identifier pattern")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def source_map(document):
    result = {r["id"]: r["text"] for r in document["clinical_notes"]}
    result.update({r["id"]: r["reasonText"] for r in document["prior_authorization_requests"]})
    return result


def fingerprint(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def expected_discovery(document):
    patient = document["patient_record"]
    interests = set(document["preferences"]["interests"])
    conditions = set(patient["conditions"])
    labs = {lab["code"] for lab in patient["labs"]}
    pending_auth = bool(document["prior_authorization_requests"])
    ranked = []
    for product in document["catalog"]:
        matches = sorted(set(product["tags"]) & (interests | conditions | labs))
        score = sum(3 if tag in interests else 2 if tag in conditions else 1 for tag in matches)
        if pending_auth and "authorization" in product["tags"]:
            score += 2
            matches = sorted(set(matches) | {"authorization"})
        ranked.append((score, product["id"], product, matches))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    records = []
    for score, _, product, matches in ranked[:document["preferences"]["limit"]]:
        records.append({
            "id": "rec-" + product["id"], "product_id": product["id"],
            "patient_reference": patient["id"], "score": score,
            "matched_tags": matches,
            "query_terms": sorted(set(product["tags"]) | set(matches)),
            "human_review_required": True, "autonomous_clinical_decision": False,
        })
    return records


def passages(raw):
    # Preserve original positions instead of joining or rewriting source text.
    for match in re.finditer(r"[^\n]+?(?:[.!?]+(?=\s|$)|(?=\n)|$)", raw):
        start, end = match.span()
        while start < end and raw[start].isspace():
            start += 1
        while end > start and raw[end - 1].isspace():
            end -= 1
        if start < end:
            yield start, end, raw[start:end]


def expected_research(document):
    findings = []
    for recommendation in document["discovery"]:
        query = tokens(" ".join(recommendation["query_terms"]))
        candidates = []
        for source_id, raw in source_map(document).items():
            for start, end, quote in passages(raw):
                overlap = len(query & tokens(quote))
                if overlap:
                    candidates.append((-overlap, source_id, start, end, quote))
        candidates.sort()
        for index, (_, source_id, start, end, quote) in enumerate(candidates[:2], 1):
            findings.append({
                "id": "finding-" + recommendation["id"] + "-" + str(index),
                "recommendation_reference": recommendation["id"],
                "patient_reference": recommendation["patient_reference"],
                "finding": quote,
                "citation": {"source_id": source_id, "start": start, "end": end, "quote": quote},
                "human_review_required": True, "autonomous_clinical_decision": False,
            })
    return findings


def expected_audit(document):
    events = []
    for stage, records in (("basic:recommend", document.get("discovery", [])),
                           ("normal", document.get("research", []))):
        for record in records:
            events.append({
                "sequence": len(events) + 1, "operation": "create",
                "stage": stage, "record_id": record["id"],
                "before_sha256": None, "after_sha256": fingerprint(record),
                "actor": "deterministic-reference-cli",
            })
    return events


def validate(document, phase="input"):
    """One validator for input, discovery handoff and final output."""
    require(phase in {"input", "discovery", "final"}, "Unknown schema phase")
    shape(document, BASE_KEYS if phase == "input" else BASE_KEYS | GENERATED_KEYS, "document")
    require(document["schema_version"] == SCHEMA, "Unsupported schema version")
    require(document["synthetic"] is True, "Only explicitly synthetic fixtures are accepted")
    phi_check({key: document[key] for key in BASE_KEYS})
    patient = document["patient_record"]
    shape(patient, {"resourceType", "id", "synthetic", "deidentified", "conditions", "labs"}, "patient")
    require(patient["resourceType"] == "Patient", "Expected FHIR-style Patient")
    require(patient["synthetic"] is True and patient["deidentified"] is True,
            "Patient must be synthetic and deidentified")
    strings(patient["conditions"], "conditions")
    ids = set()

    def register(identifier):
        require(isinstance(identifier, str) and SAFE_ID.fullmatch(identifier) is not None,
                "Resource IDs must be synthetic pseudonyms prefixed syn-")
        require(identifier not in ids, "Duplicate resource ID")
        ids.add(identifier)

    register(patient["id"])
    require(isinstance(patient["labs"], list) and len(patient["labs"]) <= 100, "Invalid labs")
    lab_codes = set()
    for lab in patient["labs"]:
        shape(lab, {"code", "value", "unit"}, "lab")
        text(lab["code"], "lab code", 120)
        text(lab["unit"], "lab unit", 40)
        require(lab["code"] not in lab_codes, "Duplicate lab code")
        lab_codes.add(lab["code"])
        require(type(lab["value"]) in (int, float) and math.isfinite(lab["value"])
                and lab["value"] >= 0, "Invalid synthetic lab value")
    for collection, resource_type, body_field in (
        ("clinical_notes", "DocumentReference", "text"),
        ("prior_authorization_requests", "Claim", "reasonText"),
    ):
        records = document[collection]
        require(isinstance(records, list) and len(records) <= 100, "Invalid resource collection")
        keys = {"resourceType", "id", "subject", body_field, "synthetic", "human_review_required"}
        if resource_type == "Claim":
            keys |= {"use", "status"}
        for record in records:
            shape(record, keys, resource_type)
            register(record["id"])
            require(record["resourceType"] == resource_type, "Unexpected resource type")
            require(record["subject"] == "Patient/" + patient["id"], "Wrong patient reference")
            require(record["synthetic"] is True and record["human_review_required"] is True,
                    "Synthetic marker and human review are mandatory")
            text(record[body_field], body_field)
            if resource_type == "Claim":
                require(record["use"] == "preauthorization" and record["status"] == "draft",
                        "Prior authorization must remain a draft for human review")
    require(isinstance(document["catalog"], list) and len(document["catalog"]) <= 100,
            "Invalid catalog")
    for product in document["catalog"]:
        shape(product, {"id", "title", "category", "tags"}, "product")
        register(product["id"])
        text(product["title"], "title", 200)
        require(product["category"] in {"education", "administrative-support"},
                "Clinical treatment products are not supported")
        strings(product["tags"], "product tags")
        require(bool(product["tags"]), "Product tags cannot be empty")
    shape(document["preferences"], {"interests", "limit"}, "preferences")
    strings(document["preferences"]["interests"], "interests")
    limit = document["preferences"]["limit"]
    require(type(limit) is int and 1 <= limit <= 10, "limit must be an integer from 1 to 10")
    if phase != "input":
        require(fingerprint(document["discovery"]) == fingerprint(expected_discovery(document)),
                "Invalid discovery handoff")
        expected = [] if phase == "discovery" else expected_research(document)
        require(fingerprint(document["research"]) == fingerprint(expected),
                "Invalid extractive findings or citations")
        require(fingerprint(document["audit_trail"]) == fingerprint(expected_audit(document)),
                "Invalid or missing audit trail")
    return document


def recommend(document):
    validate(document, "input")
    output = copy.deepcopy(document)
    output.update(discovery=expected_discovery(document), research=[], audit_trail=[])
    output["audit_trail"] = expected_audit(output)
    return validate(output, "discovery")


def research(document):
    validate(document, "discovery")
    output = copy.deepcopy(document)
    output["research"] = expected_research(output)
    output["audit_trail"] = expected_audit(output)
    return validate(output, "final")


def run_pipeline(document):
    return research(recommend(document))


def reject_constant(value):
    raise ValidationError("Non-finite JSON numbers are prohibited: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            document = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = {"status": "ok", "data": run_pipeline(document)}
        code = 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        # Never echo input content or paths in error output.
        result = {"status": "error", "error": type(exc).__name__,
                  "message": "Input/file validation failed; provide the documented synthetic schema."}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
