"""Offline synthetic healthcare evidence synthesis, not a clinical decision system.

Run: python -B implementation.py example_input.json
Only the deliberately narrow synthetic schema is accepted. Identifier screening
is a demonstration, not HIPAA certification or a general free-text PHI detector.
"""

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path


MAX_BYTES = 1_000_000
NOTICE = (
    "Synthetic evidence review only; not a diagnosis, treatment recommendation, "
    "or authorization decision. Human review is required. Identifier screening "
    "is demonstrative and is not HIPAA certification."
)
IDENTIFIER = re.compile(
    r"\b(?:MRN|SSN|DOB|birth\s*date|date\s+of\s+birth|patient\s+name|"
    r"address|telephone|phone|email)\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b\d{3}[-. ]\d{2}[-. ]\d{4}\b|"
    r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b|"
    r"\b\d{7,}\b|https?://",
    re.IGNORECASE,
)


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(expected), location + " has missing or unsupported fields")


def text(value, location, maximum=10000):
    require(isinstance(value, str) and 0 < len(value) <= maximum and value.strip(),
            location + " must be nonempty bounded text")
    require(not any(ord(c) < 32 and c not in "\n\t\r" for c in value),
            location + " contains control characters")
    require(IDENTIFIER.search(value) is None,
            location + " contains a prohibited identifier pattern")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def source_text(record):
    resource = record["resource"]
    if record["kind"] == "clinical_note":
        return resource["text"]
    if record["kind"] == "prior_authorization_request":
        return resource["request_text"]
    return canonical(resource)


def summarize(document):
    results = []
    for question in document["research_questions"]:
        citations = []
        for record in document["records"]:
            for index, evidence in enumerate(record["evidence"]):
                if evidence["question"] == question:
                    citations.append({
                        "record_id": record["record_id"],
                        "evidence_index": index,
                        "stance": evidence["stance"],
                        "quote": evidence["quote"],
                    })
        stances = {item["stance"] for item in citations}
        reasons = []
        disagreement = "support" in stances and "oppose" in stances
        if not citations:
            finding = "no_evidence"
            reasons.append("No supplied document addresses this question.")
        elif disagreement:
            finding = "conflicting_evidence"
            reasons.append("Supporting and opposing evidence require human reconciliation.")
        elif stances == {"uncertain"}:
            finding = "uncertain_evidence"
        elif "support" in stances:
            finding = "supporting_evidence_recorded"
        else:
            finding = "opposing_evidence_recorded"
        if "uncertain" in stances:
            reasons.append("At least one source explicitly leaves the question uncertain.")
        source_count = len({item["record_id"] for item in citations})
        if citations and source_count < 2:
            reasons.append("Only one source document supplies evidence.")
        results.append({
            "question": question,
            "finding": finding,
            "disagreement": disagreement,
            "source_count": source_count,
            "citations": citations,
            "unresolved_questions": reasons,
            "human_review_required": True,
        })
    return {"questions": results, "notice": NOTICE, "clinical_decision": None}


def audit_for(document, synthesis):
    events = []
    for record in document["records"]:
        events.append({
            "sequence": len(events) + 1,
            "record_id": record["record_id"],
            "operation": "create",
            "actor": "offline-reference-builder",
            "before_sha256": None,
            "after_sha256": hashlib.sha256(canonical(record).encode("utf-8")).hexdigest(),
            "reason": "Register immutable validated synthetic source in this run.",
            "human_review_required": True,
        })
    events.append({
        "sequence": len(events) + 1,
        "record_id": "synthetic-research-report",
        "operation": "create",
        "actor": "offline-reference-builder",
        "before_sha256": None,
        "after_sha256": hashlib.sha256(canonical(synthesis).encode("utf-8")).hexdigest(),
        "reason": "Create evidence synthesis without changing source records.",
        "human_review_required": True,
    })
    return events


def validate_document(document, output=False):
    """One shared validation boundary for input and generated output."""
    common = {"schema_version", "synthetic", "human_review_required",
              "records", "research_questions"}
    expected = common | ({"status", "synthesis", "audit_trail",
                          "autonomous_clinical_decisions"} if output else set())
    keys(document, expected, "document")
    require(document["schema_version"] == "1.0", "Unsupported schema version")
    require(document["synthetic"] is True, "Only explicitly synthetic fixtures are accepted")
    require(document["human_review_required"] is True, "Human review cannot be disabled")
    questions = document["research_questions"]
    require(isinstance(questions, list) and 1 <= len(questions) <= 30,
            "research_questions must contain 1 to 30 questions")
    for question in questions:
        text(question, "research question", 500)
    require(len({q.casefold().strip() for q in questions}) == len(questions),
            "Research questions must be unique")
    records = document["records"]
    require(isinstance(records, list) and 1 <= len(records) <= 100,
            "records must contain 1 to 100 records")
    ids, patients, references = set(), [], []
    for record in records:
        keys(record, {"record_id", "kind", "resource", "evidence"}, "record")
        rid = record["record_id"]
        require(isinstance(rid, str) and
                re.fullmatch(r"synthetic-(patient|note|auth)-[0-9]{3}", rid) is not None,
                "Record IDs must be synthetic typed identifiers")
        require(rid not in ids, "Duplicate record ID")
        ids.add(rid)
        kind = record["kind"]
        require(isinstance(kind, str) and kind in
                {"patient_record", "clinical_note", "prior_authorization_request"},
                "Unsupported record kind")
        resource = record["resource"]
        base = {"resourceType", "id", "synthetic"}
        if kind == "patient_record":
            keys(resource, base | {"diagnoses", "labs"}, "patient resource")
            require(resource["resourceType"] == "Patient" and
                    rid.startswith("synthetic-patient-"), "Invalid Patient resource")
            patients.append(rid)
            diagnoses, labs = resource["diagnoses"], resource["labs"]
            require(isinstance(diagnoses, list) and len(diagnoses) <= 50,
                    "diagnoses must be a bounded list")
            for diagnosis in diagnoses:
                require(isinstance(diagnosis, str) and
                        re.fullmatch(r"[A-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?", diagnosis),
                        "diagnoses must use synthetic fixture diagnosis codes")
            require(isinstance(labs, list) and len(labs) <= 100,
                    "labs must be a bounded list")
            for lab in labs:
                keys(lab, {"code", "value", "unit"}, "lab")
                require(isinstance(lab["code"], str) and
                        re.fullmatch(r"[0-9]{1,5}-[0-9]", lab["code"]),
                        "Lab code must be LOINC-style")
                value = lab["value"]
                require(type(value) in (int, float) and 0 <= value <= 1_000_000 and
                        math.isfinite(value), "Lab value must be a finite bounded number")
                require(lab["unit"] in ("mg/dL", "mmol/L", "%", "g/dL"),
                        "Unsupported lab unit")
        elif kind == "clinical_note":
            keys(resource, base | {"subject", "text"}, "clinical note resource")
            require(resource["resourceType"] == "DocumentReference" and
                    rid.startswith("synthetic-note-"), "Invalid clinical note resource")
            keys(resource["subject"], {"reference"}, "note subject")
            references.append(resource["subject"]["reference"])
            text(resource["text"], "clinical note")
            require(resource["text"].startswith("SYNTHETIC FIXTURE:"),
                    "Clinical notes must be explicitly labeled synthetic")
        else:
            keys(resource, base | {"patient", "status", "purpose", "request_text"},
                 "authorization resource")
            require(resource["resourceType"] == "Claim" and
                    rid.startswith("synthetic-auth-"), "Invalid Claim resource")
            keys(resource["patient"], {"reference"}, "claim patient")
            references.append(resource["patient"]["reference"])
            require(resource["status"] == "draft" and
                    resource["purpose"] == "prior-authorization",
                    "Only draft prior authorization requests are accepted")
            text(resource["request_text"], "authorization text")
            require(resource["request_text"].startswith("SYNTHETIC FIXTURE:"),
                    "Authorization text must be explicitly labeled synthetic")
        require(resource["id"] == rid and resource["synthetic"] is True,
                "Resource identity or synthetic flag is invalid")
        evidence = record["evidence"]
        require(isinstance(evidence, list) and len(evidence) <= 100,
                "evidence must be a bounded list")
        seen = set()
        for item in evidence:
            keys(item, {"question", "stance", "quote"}, "evidence item")
            text(item["question"], "evidence question", 500)
            require(item["question"] in questions, "Evidence references an unknown question")
            require(isinstance(item["stance"], str) and
                    item["stance"] in ("support", "oppose", "uncertain"),
                    "Evidence stance is invalid")
            text(item["quote"], "evidence quotation", 2000)
            require(item["quote"] in source_text(record), "Evidence quotation is not in source")
            signature = canonical(item)
            require(signature not in seen, "Duplicate evidence item")
            seen.add(signature)
    require(len(patients) == 1, "Exactly one synthetic patient is required per research run")
    require(all(ref == "Patient/" + patients[0] for ref in references),
            "All records must reference the same supplied synthetic patient")
    if output:
        require(document["status"] == "ok", "Invalid output status")
        require(document["autonomous_clinical_decisions"] is False,
                "Autonomous clinical decisions are prohibited")
        expected_summary = summarize(document)
        require(document["synthesis"] == expected_summary, "Output synthesis is inconsistent")
        require(document["audit_trail"] == audit_for(document, expected_summary),
                "Every record creation must have an intact audit event")
    return document


def run(document):
    validate_document(document)
    result = copy.deepcopy(document)
    result.update(status="ok", autonomous_clinical_decisions=False)
    result["synthesis"] = summarize(result)
    result["audit_trail"] = audit_for(result, result["synthesis"])
    return validate_document(result, output=True)


def reject_constant(_):
    raise ValidationError("Nonfinite JSON numbers are prohibited")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object field")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "Input file exceeds size limit")
        document = json.loads(raw.decode("utf-8"), parse_constant=reject_constant,
                              object_pairs_hook=unique_object)
        result = run(document)
        print(canonical(result))
        return 0
    except (OSError, UnicodeError):
        error = "Input file cannot be read as UTF-8"
    except json.JSONDecodeError:
        error = "Invalid JSON input"
    except ValidationError as exc:
        error = str(exc)
    except (TypeError, ValueError, OverflowError, RecursionError):
        error = "Invalid input structure or numeric value"
    print(canonical({"status": "error", "error": error}))
    return 2


if __name__ == "__main__":
    sys.exit(main())
