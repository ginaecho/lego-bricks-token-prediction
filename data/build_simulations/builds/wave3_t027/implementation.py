"""Synthetic-only guided healthcare onboarding; no clinical decision making."""

import copy
import json
import math
import re
import sys


STEPS = (
    "check_prerequisites",
    "register_patient",
    "attach_clinical_note",
    "prepare_prior_authorization",
    "request_human_review",
)
NOTE_TEXTS = {
    "SYNTHETIC: Training note only. No clinical decision recorded.",
    "SYNTHETIC: Fictitious follow-up note. Human review required.",
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unsupported fields")


def identifier(value, prefix, path):
    require(
        isinstance(value, str)
        and re.fullmatch(re.escape(prefix) + r"[0-9]{3}", value) is not None,
        path + " must be a synthetic identifier",
    )


def validate(data):
    """The single validation boundary used by the CLI and Python API.

    Closed field sets and controlled note text deliberately reject direct identifiers,
    arbitrary extensions, real demographics, and uncontrolled clinical narratives.
    This is a demonstration, not a HIPAA certification or general de-identification tool.
    """
    fields(data, ("schema_version", "synthetic_fixture", "prerequisites", "records", "actions"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "Unsupported schema_version")
    require(data["synthetic_fixture"] is True, "Only clearly labeled synthetic fixtures are accepted")
    prerequisites = data["prerequisites"]
    fields(prerequisites, ("deidentification_training_complete", "reviewer_available", "reviewer_id"), "prerequisites")
    for key in ("deidentification_training_complete", "reviewer_available"):
        require(type(prerequisites[key]) is bool, key + " must be boolean")
    identifier(prerequisites["reviewer_id"], "syn-reviewer-", "reviewer_id")

    records = data["records"]
    fields(records, ("patient_record", "clinical_note", "prior_authorization_request"), "records")
    patient = records["patient_record"]
    fields(patient, ("resourceType", "id", "synthetic", "conditions", "lab_values"), "patient_record")
    require(patient["resourceType"] == "Patient", "patient_record must be Patient")
    identifier(patient["id"], "syn-patient-", "patient_record.id")
    require(patient["synthetic"] is True, "patient_record must be synthetic")
    require(type(patient["conditions"]) is list and 0 < len(patient["conditions"]) <= 10, "Expected 1-10 conditions")
    for condition in patient["conditions"]:
        fields(condition, ("resourceType", "code"), "condition")
        require(condition["resourceType"] == "Condition", "Expected Condition")
        require(condition["code"] in ("DEMO-DX-001", "DEMO-DX-002"), "Only fictitious diagnosis codes are accepted")
    require(type(patient["lab_values"]) is list and 0 < len(patient["lab_values"]) <= 10, "Expected 1-10 lab values")
    for lab in patient["lab_values"]:
        fields(lab, ("resourceType", "code", "value", "unit"), "lab")
        require(lab["resourceType"] == "Observation" and lab["code"] == "DEMO-LAB-001", "Only synthetic observations are accepted")
        require(type(lab["value"]) in (int, float) and 0 <= lab["value"] <= 10000 and math.isfinite(lab["value"]), "Invalid lab value")
        require(lab["unit"] == "mg/dL", "Unsupported lab unit")

    note = records["clinical_note"]
    claim = records["prior_authorization_request"]
    fields(note, ("resourceType", "id", "subject", "synthetic", "text", "human_review_required"), "clinical_note")
    fields(claim, ("resourceType", "id", "patient", "synthetic", "service_code", "status", "human_review_required"), "prior_authorization_request")
    for record, resource_type, prefix, reference_key in (
        (note, "DocumentReference", "syn-note-", "subject"),
        (claim, "Claim", "syn-auth-", "patient"),
    ):
        require(record["resourceType"] == resource_type, "Unexpected resourceType")
        identifier(record["id"], prefix, resource_type + ".id")
        require(record["synthetic"] is True, "Every record must be synthetic")
        require(record["human_review_required"] is True, "Human review cannot be disabled")
        require(record[reference_key] == "Patient/" + patient["id"], "Patient reference mismatch")
    require(isinstance(note["text"], str) and note["text"] in NOTE_TEXTS, "Clinical note must use a controlled synthetic template; identifiers and clinical decisions are prohibited")
    require(claim["service_code"] == "DEMO-SERVICE-001", "Unsupported synthetic service")
    require(claim["status"] == "draft", "Autonomous authorization decisions are prohibited")

    require(type(data["actions"]) is list and len(data["actions"]) <= 100, "actions must contain at most 100 steps")
    for action in data["actions"]:
        require(isinstance(action, str) and action in STEPS, "Unknown onboarding action")
    return data


def run(data):
    validate(data)
    result = copy.deepcopy(data)
    active = {}
    completed = []
    audit = []

    def change_record(key, value, step):
        before = copy.deepcopy(active.get(key))
        active[key] = copy.deepcopy(value)
        audit.append({
            "sequence": len(audit) + 1,
            "actor": "synthetic-onboarding-system",
            "step": step,
            "record_id": value["id"],
            "operation": "create" if before is None else "update",
            "before": before,
            "after": copy.deepcopy(value),
        })

    for action in data["actions"]:
        if action in completed:
            continue
        require(action == STEPS[len(completed)], "Unmet prerequisite: next step is " + STEPS[len(completed)])
        if action == "check_prerequisites":
            require(data["prerequisites"]["deidentification_training_complete"], "De-identification training must be completed")
            require(data["prerequisites"]["reviewer_available"], "A human reviewer must be available")
        elif action == "register_patient":
            change_record("patient_record", data["records"]["patient_record"], action)
        elif action == "attach_clinical_note":
            change_record("clinical_note", data["records"]["clinical_note"], action)
        elif action == "prepare_prior_authorization":
            change_record("prior_authorization_request", data["records"]["prior_authorization_request"], action)
        elif action == "request_human_review":
            claim = copy.deepcopy(active["prior_authorization_request"])
            claim["status"] = "pending-human-review"
            change_record("prior_authorization_request", claim, action)
        completed.append(action)

    next_step = STEPS[len(completed)] if len(completed) < len(STEPS) else None
    result.update({
        "status": "ok",
        "onboarding": {
            "completed_steps": completed,
            "next_step": next_step,
            "progress_percent": len(completed) * 100 // len(STEPS),
            "setup_complete": next_step is None,
            "human_review_required": True,
            "reviewer_id": data["prerequisites"]["reviewer_id"],
            "review_status": "pending" if next_step is None else "not-requested",
            "autonomous_clinical_decisions": False,
        },
        "active_records": active,
        "audit_trail": audit,
    })
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON numbers are prohibited")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON fields are prohibited")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], "r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run(data)
        code = 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        # Do not echo submitted data or filesystem details into an error response.
        output = {"status": "error", "message": "Invalid input or unreadable file; check the shared schema and step prerequisites."}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
