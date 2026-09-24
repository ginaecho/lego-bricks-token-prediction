"""Synthetic public-sector document review. Standard library; no providers.

Usage: python -B implementation.py example_input.json
Requirements are supplied explicitly, not inferred from policy prose. Evidence
matching is a literal, case-insensitive phrase check, not a truth assessment.
"""

import json
import re
import sys
from pathlib import Path

ENTITY_TYPES = {"citizen_service_request", "benefits_application", "policy_document"}
FORM_FIELDS = {"service_description", "household_size", "income_statement",
               "residency_statement", "public_summary"}
MAX_BYTES = 1_000_000


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_keys(value, keys):
    require(isinstance(value, dict) and set(value) == set(keys),
            "An object has missing or unsupported fields.")


def text(value, limit=10000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            "Text must be nonempty and within the size limit.")


def identifier(value, prefix):
    require(isinstance(value, str) and
            re.fullmatch(prefix + r"-[0-9]{3}", value) is not None,
            "Use synthetic identifiers with the required prefix and three digits.")


def sequence(value, allow_empty=False):
    require(isinstance(value, list) and len(value) <= 100 and
            (allow_empty or len(value) > 0), "A list has an invalid size or type.")


def unique(values):
    require(len(values) == len(set(values)), "Identifiers or list values must be unique.")


def normalize(value):
    return " ".join(value.casefold().split())


def validate(data):
    """One validation boundary shared by the Python API and CLI."""
    object_keys(data, {"schema_version", "synthetic_fixture", "policy", "documents"})
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Only schema version 1 is supported.")
    require(data["synthetic_fixture"] is True,
            "Only clearly labeled synthetic fixtures are accepted.")
    policy = data["policy"]
    object_keys(policy, {"id", "text", "requirements"})
    identifier(policy["id"], "P")
    text(policy["text"], 50000)
    sequence(policy["requirements"])
    requirements = {}
    for rule in policy["requirements"]:
        object_keys(rule, {"id", "applies_to", "source_phrase",
                           "required_fields", "evidence_terms"})
        identifier(rule["id"], "R")
        require(rule["id"] not in requirements, "Requirement identifiers must be unique.")
        requirements[rule["id"]] = rule
        sequence(rule["applies_to"])
        require(all(isinstance(v, str) and v in ENTITY_TYPES for v in rule["applies_to"]),
                "An entity type is not supported.")
        unique(rule["applies_to"])
        text(rule["source_phrase"])
        require(normalize(rule["source_phrase"]) in normalize(policy["text"]),
                "Every requirement must point to a phrase in the supplied policy.")
        sequence(rule["required_fields"], allow_empty=True)
        require(all(isinstance(v, str) and v in FORM_FIELDS for v in rule["required_fields"]),
                "A required form field is not supported.")
        unique(rule["required_fields"])
        sequence(rule["evidence_terms"])
        for term in rule["evidence_terms"]:
            text(term, 200)
        unique([normalize(term) for term in rule["evidence_terms"]])
    sequence(data["documents"])
    document_ids = []
    evidence_ids = []
    for document in data["documents"]:
        object_keys(document, {"id", "entity_type", "case_number", "citizen",
                               "form_fields", "evidence"})
        identifier(document["id"], "D")
        document_ids.append(document["id"])
        require(isinstance(document["entity_type"], str) and
                document["entity_type"] in ENTITY_TYPES, "An entity type is not supported.")
        require(isinstance(document["case_number"], str) and
                re.fullmatch(r"CASE-DEMO-[0-9]{4}", document["case_number"]) is not None,
                "Use an invented CASE-DEMO case number.")
        if document["citizen"] is not None:
            object_keys(document["citizen"], {"name", "address", "email"})
            for value in document["citizen"].values():
                text(value, 500)
        fields = document["form_fields"]
        require(isinstance(fields, dict) and set(fields).issubset(FORM_FIELDS),
                "Form fields must use supported names.")
        for value in fields.values():
            text(value)
        sequence(document["evidence"], allow_empty=True)
        for evidence in document["evidence"]:
            object_keys(evidence, {"id", "requirement_id", "text"})
            identifier(evidence["id"], "E")
            evidence_ids.append(evidence["id"])
            identifier(evidence["requirement_id"], "R")
            require(evidence["requirement_id"] in requirements,
                    "Evidence must reference a known requirement.")
            require(document["entity_type"] in
                    requirements[evidence["requirement_id"]]["applies_to"],
                    "Evidence references a requirement that does not apply.")
            text(evidence["text"])
    unique(document_ids)
    unique(evidence_ids)
    return data


def review(data):
    validate(data)
    reviews = []
    total_gaps = 0
    for document in data["documents"]:
        checks = []
        for index, rule in enumerate(data["policy"]["requirements"]):
            if document["entity_type"] not in rule["applies_to"]:
                continue
            candidates = [e for e in document["evidence"]
                          if e["requirement_id"] == rule["id"]]
            matches = [e["id"] for e in candidates
                       if all(normalize(term) in normalize(e["text"])
                              for term in rule["evidence_terms"])]
            missing_fields = [field for field in rule["required_fields"]
                              if field not in document["form_fields"]]
            gap = bool(missing_fields) or not matches
            total_gaps += int(gap)
            reasons = []
            if missing_fields:
                reasons.append("Add the missing form fields.")
            if not matches:
                reasons.append("Provide one evidence item containing all required phrases.")
            checks.append({
                "requirement_id": rule["id"],
                "policy_reference": {
                    "policy_id": data["policy"]["id"],
                    "requirement_path": "/policy/requirements/" + str(index),
                },
                "status": "gap" if gap else "supported",
                "missing_fields": missing_fields,
                "reviewed_evidence_ids": [e["id"] for e in candidates],
                "matched_evidence_ids": matches,
                "explanation": " ".join(reasons) if reasons else
                "The form fields are present. One evidence item has all required phrases.",
            })
        reviews.append({
            "document_id": document["id"],
            "entity_type": document["entity_type"],
            "status": ("not_assessed" if not checks else
                       "gaps_found" if any(c["status"] == "gap" for c in checks)
                       else "supported"),
            "checks": checks,
        })
    return {
        "schema_version": 1,
        "status": "ok",
        "synthetic_fixture": True,
        "review_type": "document_review",
        "summary": {"documents_reviewed": len(reviews), "gaps_found": total_gaps},
        "privacy": {
            "citizen_pii_included": False,
            "approach": "Names, addresses, emails, case numbers, and source text are not returned.",
        },
        "limitations": [
            "This is a synthetic demonstration, not a legal or compliance certification.",
            "Phrase matches do not prove facts or decide benefit eligibility.",
            "A person must review the evidence and any gaps before making a decision.",
            "Plain-language JSON is provided; Section 508 accessibility is not certified.",
        ],
        "reviews": reviews,
    }


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def reject_constant(_value):
    raise ValidationError("Non-finite JSON numbers are not allowed.")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Provide exactly one input JSON file.")
        with Path(args[0]).open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "Input file exceeds the size limit.")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
                          parse_constant=reject_constant)
        output = review(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError,
            RecursionError, ValueError):
        # Never reflect file paths, source text, or malformed values into errors.
        print(json.dumps({"schema_version": 1, "status": "error",
                          "message": "Input could not be read or validated. Check the schema and synthetic data."}))
        return 2
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
