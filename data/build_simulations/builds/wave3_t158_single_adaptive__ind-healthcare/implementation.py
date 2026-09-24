"""Synthetic-only guided onboarding; illustrative safeguards, not certification."""
import copy
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


PREREQUISITES = ("synthetic-data", "human-review")
EXPLANATIONS = {
    "synthetic-data": "Use only de-identified synthetic fixtures; never enter patient identifiers.",
    "human-review": "A qualified human must review every record; no clinical decision is automated.",
    "patient-record": "Inspect synthetic diagnoses and lab values without interpreting their clinical meaning.",
    "clinical-note": "Inspect the constrained synthetic note and its patient reference.",
    "prior-authorization": "Prepare the request for human review, never approve or deny it automatically.",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, label):
    require(type(value) is dict, label + " must be an object")
    require(set(value) == set(expected), label + " has missing or unsupported fields")


def identifier(value, prefix):
    require(type(value) is str and re.fullmatch(prefix + r"-[0-9]{3}", value),
            "Invalid synthetic identifier")


def canonical_note(patient):
    diagnoses = ", ".join(patient["diagnoses"]) or "none"
    labs = ", ".join(
        f'{lab["code"]}={lab["value"]} {lab["unit"]}' for lab in patient["labs"]
    ) or "none"
    return f"SYNTHETIC ONLY. Diagnoses: {diagnoses}. Labs: {labs}. Human review required."


def validate(data):
    """Single boundary validator shared by the CLI and in-process workflow."""
    fields(data, ("schema_version", "synthetic", "onboarding", "records"), "input")
    require(data["schema_version"] == "1.0", "Unsupported schema_version")
    require(data["synthetic"] is True, "Only explicitly synthetic fixtures are accepted")
    profile = data["onboarding"]
    fields(profile, ("experience", "detail", "format", "completed"), "onboarding")
    require(profile["experience"] in ("novice", "experienced"), "Invalid experience")
    require(profile["detail"] in ("guided", "concise"), "Invalid detail preference")
    require(profile["format"] in ("text", "checklist"), "Invalid format preference")
    completed = profile["completed"]
    require(type(completed) is list and all(type(x) is str for x in completed),
            "completed must be a list of prerequisite IDs")
    require(len(set(completed)) == len(completed), "Duplicate completion")
    require(all(x in PREREQUISITES for x in completed), "Unknown prerequisite")
    records = data["records"]
    fields(records, ("patient_record", "clinical_note", "prior_authorization_request"), "records")
    patient = records["patient_record"]
    fields(patient, ("resourceType", "id", "synthetic", "diagnoses", "labs"), "patient_record")
    require(patient["resourceType"] == "Patient", "Expected Patient resource")
    identifier(patient["id"], "anon")
    require(patient["synthetic"] is True, "Patient must be synthetic")
    diagnoses = patient["diagnoses"]
    require(type(diagnoses) is list and len(diagnoses) <= 20, "Invalid diagnoses")
    require(all(type(x) is str and re.fullmatch(r"D[0-9]{3}", x) for x in diagnoses),
            "Diagnoses must be synthetic codes")
    require(len(set(diagnoses)) == len(diagnoses), "Duplicate diagnosis")
    labs = patient["labs"]
    require(type(labs) is list and len(labs) <= 20, "Invalid labs")
    codes = []
    for lab in labs:
        fields(lab, ("code", "value", "unit"), "lab")
        require(type(lab["code"]) is str and re.fullmatch(r"L[0-9]{3}", lab["code"]),
                "Invalid synthetic lab code")
        require(type(lab["value"]) in (int, float) and math.isfinite(lab["value"])
                and abs(lab["value"]) <= 1000000, "Lab value must be a bounded finite number")
        require(lab["unit"] in ("mmol/L", "mg/dL", "g/L"), "Unsupported lab unit")
        codes.append(lab["code"])
    require(len(set(codes)) == len(codes), "Duplicate lab code")
    note = records["clinical_note"]
    fields(note, ("resourceType", "id", "subject", "text"), "clinical_note")
    require(note["resourceType"] == "DocumentReference", "Expected DocumentReference")
    identifier(note["id"], "note")
    reference = {"reference": "Patient/" + patient["id"]}
    require(note["subject"] == reference, "Note patient reference mismatch")
    # Constrained text avoids pretending arbitrary prose can be fully de-identified.
    require(note["text"] == canonical_note(patient),
            "Clinical note must match the canonical synthetic template; unrestricted prose is forbidden")
    request = records["prior_authorization_request"]
    fields(request, ("resourceType", "id", "patient", "use", "status", "service_code"),
           "prior_authorization_request")
    require(request["resourceType"] == "Claim", "Expected Claim resource")
    identifier(request["id"], "auth")
    require(request["patient"] == reference, "Authorization patient reference mismatch")
    require(request["use"] == "preauthorization" and request["status"] == "draft",
            "Only draft prior authorization requests are accepted")
    require(type(request["service_code"]) is str
            and re.fullmatch(r"S[0-9]{3}", request["service_code"]), "Invalid synthetic service code")
    return data


def run(data):
    validate(data)
    profile = data["onboarding"]
    completed = profile["completed"]
    missing = [key for key in PREREQUISITES if key not in completed]
    ready = not missing
    steps = []
    for key in (*PREREQUISITES, "patient-record", "clinical-note", "prior-authorization"):
        prerequisite = key in PREREQUISITES
        state = ("completed" if key in completed else "required") if prerequisite else (
            "available" if ready else "blocked")
        step = {
            "id": key, "state": state,
            "prerequisites": [] if prerequisite else list(PREREQUISITES),
            "explanation": EXPLANATIONS[key],
            "presentation": profile["format"],
        }
        if profile["experience"] == "novice" or profile["detail"] == "guided":
            step["guidance"] = "Read the explanation, inspect the synthetic example, then ask a human reviewer."
        steps.append(step)
    records = copy.deepcopy(data["records"])
    audit = []
    for key, record in records.items():
        workflow = {
            "human_review_required": True,
            "onboarding_status": "ready_for_human_review" if ready else "blocked",
            "clinical_decision": "not_performed",
        }
        record["workflow"] = workflow
        audit.append({
            "sequence": len(audit) + 1, "actor": "adaptive-onboarding",
            "resource_id": record["id"], "operation": "add",
            "path": "/records/" + key + "/workflow", "before": None,
            "after": copy.deepcopy(workflow), "reason": "Attach mandatory human-review routing metadata",
        })
    return {
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "human_review_required": True,
        "onboarding": {
            "experience": profile["experience"], "detail": profile["detail"],
            "format": profile["format"], "missing_prerequisites": missing,
            "state": "ready_for_human_review" if ready else "blocked", "steps": steps,
        },
        "records": records, "audit_trail": audit,
        "limitations": [
            "Synthetic Synthea-style fixtures, not real Synthea exports or full FHIR conformance.",
            "Demonstrative HIPAA-oriented identifier exclusion, not compliance certification.",
            "No diagnosis, treatment recommendation, eligibility determination, approval or denial.",
        ],
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object)
        result = run(data)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        # Do not reflect untrusted clinical text or file paths into error output.
        print(json.dumps({"status": "error", "error": "Invalid input, schema, or unreadable file"}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
