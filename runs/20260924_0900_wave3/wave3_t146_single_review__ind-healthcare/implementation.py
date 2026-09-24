"""Synthetic-only document review; not clinical advice or HIPAA certification."""

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def check(condition, path, message):
    if not condition:
        raise ValidationError(f"{path}: {message}")


def fields(value, names, path):
    check(type(value) is dict, path, "expected object")
    check(set(value) == set(names.split()), path, "missing or unsupported fields")


def token(value, pattern, path):
    check(type(value) is str and re.fullmatch(pattern, value) is not None,
          path, "expected synthetic token")


def array(value, path, minimum=0):
    check(type(value) is list and minimum <= len(value) <= 100,
          path, "expected bounded array")


def unique(values, path):
    check(len(values) == len(set(values)), path, "duplicate identifier")


def validate(data):
    """One shared input validation layer, also used before reviewing any record.

    De-identification is implemented by excluding identifier-bearing fields and
    permitting only a closed synthetic note grammar, never arbitrary patient text.
    This intentionally does not implement a general HIPAA Safe Harbor detector.
    """
    fields(data, "schema_version synthetic_data records", "$")
    check(type(data["schema_version"]) is int and data["schema_version"] == 1,
          "$.schema_version", "expected version 1")
    check(data["synthetic_data"] is True, "$.synthetic_data", "synthetic data required")
    array(data["records"], "$.records", 1)
    record_ids, patient_ids, note_ids, claim_ids = [], [], [], []
    for i, record in enumerate(data["records"]):
        p = f"$.records[{i}]"
        fields(record, "record_id patient_record clinical_note prior_authorization_request", p)
        token(record["record_id"], r"REC-[0-9]{4}", p + ".record_id")
        record_ids.append(record["record_id"])
        patient = record["patient_record"]
        pp = p + ".patient_record"
        fields(patient, "resourceType id synthetic diagnoses labs", pp)
        check(patient["resourceType"] == "Patient", pp, "expected Patient")
        token(patient["id"], r"SYN-PAT-[0-9]{4}", pp + ".id")
        patient_ids.append(patient["id"])
        check(patient["synthetic"] is True, pp, "synthetic patient required")
        array(patient["diagnoses"], pp + ".diagnoses")
        dx_codes = []
        for j, dx in enumerate(patient["diagnoses"]):
            dp = f"{pp}.diagnoses[{j}]"
            fields(dx, "code", dp)
            token(dx["code"], r"SYN-DX-[A-Z0-9]{1,12}", dp + ".code")
            dx_codes.append(dx["code"])
        unique(dx_codes, pp + ".diagnoses")
        array(patient["labs"], pp + ".labs")
        lab_codes = []
        for j, lab in enumerate(patient["labs"]):
            lp = f"{pp}.labs[{j}]"
            fields(lab, "code value unit", lp)
            token(lab["code"], r"SYN-LAB-[A-Z0-9]{1,12}", lp + ".code")
            value = lab["value"]
            check(type(value) in (int, float) and -1e9 <= value <= 1e9
                  and math.isfinite(value), lp + ".value", "expected finite numeric value")
            check(lab["unit"] == "arb", lp + ".unit", "synthetic arbitrary units required")
            lab_codes.append(lab["code"])
        unique(lab_codes, pp + ".labs")
        note = record["clinical_note"]
        np = p + ".clinical_note"
        fields(note, "resourceType id subject text", np)
        check(note["resourceType"] == "DocumentReference", np, "expected DocumentReference")
        token(note["id"], r"SYN-NOTE-[0-9]{4}", np + ".id")
        note_ids.append(note["id"])
        reference = "Patient/" + patient["id"]
        fields(note["subject"], "reference", np + ".subject")
        check(note["subject"]["reference"] == reference, np, "patient reference mismatch")
        text = note["text"]
        check(type(text) is str and len(text) <= 4000, np + ".text", "expected bounded text")
        patterns = (
            r"Diagnosis: SYN-DX-[A-Z0-9]{1,12}",
            r"Lab: SYN-LAB-[A-Z0-9]{1,12} = -?[0-9]{1,6}(?:\.[0-9]{1,6})? arb",
            r"Documentation: synthetic encounter recorded\.",
        )
        for line in text.splitlines():
            check(any(re.fullmatch(pattern, line) for pattern in patterns),
                  np + ".text", "only closed synthetic note grammar is accepted")
        claim = record["prior_authorization_request"]
        cp = p + ".prior_authorization_request"
        fields(claim, "resourceType id use patient service_code requirements", cp)
        check(claim["resourceType"] == "Claim" and claim["use"] == "preauthorization",
              cp, "expected preauthorization Claim")
        token(claim["id"], r"SYN-PA-[0-9]{4}", cp + ".id")
        claim_ids.append(claim["id"])
        fields(claim["patient"], "reference", cp + ".patient")
        check(claim["patient"]["reference"] == reference, cp, "patient reference mismatch")
        token(claim["service_code"], r"SYN-SVC-[A-Z0-9]{1,12}", cp + ".service_code")
        array(claim["requirements"], cp + ".requirements", 1)
        requirement_ids = []
        for j, req in enumerate(claim["requirements"]):
            rp = f"{cp}.requirements[{j}]"
            fields(req, "id source code", rp)
            token(req["id"], r"REQ-[0-9]{3}", rp + ".id")
            check(type(req["source"]) is str and req["source"] in
                  ("diagnosis", "lab", "clinical_note"), rp + ".source", "unsupported source")
            pattern = {"diagnosis": r"SYN-DX-[A-Z0-9]{1,12}",
                       "lab": r"SYN-LAB-[A-Z0-9]{1,12}",
                       "clinical_note": r"documentation"}[req["source"]]
            token(req["code"], pattern, rp + ".code")
            requirement_ids.append(req["id"])
        unique(requirement_ids, cp + ".requirements")
    for name, values in (("record", record_ids), ("patient", patient_ids),
                         ("note", note_ids), ("claim", claim_ids)):
        unique(values, "$.records." + name)
    return data


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def review(data):
    validate(data)
    updated = copy.deepcopy(data["records"])
    audit = []
    for i, record in enumerate(updated):
        base = f"$.records[{i}]"
        checks = []
        for j, req in enumerate(record["prior_authorization_request"]["requirements"]):
            evidence = None
            if req["source"] in ("diagnosis", "lab"):
                collection = "diagnoses" if req["source"] == "diagnosis" else "labs"
                for k, item in enumerate(record["patient_record"][collection]):
                    if item["code"] == req["code"]:
                        evidence = f"{base}.patient_record.{collection}[{k}]"
                        break
            else:
                for k, line in enumerate(record["clinical_note"]["text"].splitlines()):
                    if line == "Documentation: synthetic encounter recorded.":
                        evidence = f"{base}.clinical_note.text#line={k + 1}"
                        break
            checks.append({
                "requirement_id": req["id"],
                "requirement_path": f"{base}.prior_authorization_request.requirements[{j}]",
                "source": req["source"], "expected_code": req["code"],
                "status": "evidence_present" if evidence else "gap",
                "evidence_path": evidence,
                "reason": "Documented evidence located." if evidence else
                          "Required documentation absent; human follow-up required.",
            })
        before = digest(record)
        record["review"] = {
            "documentation_status": "gaps_found" if any(c["status"] == "gap" for c in checks)
                                    else "evidence_present",
            "human_review_required": True,
            "clinical_decision": "not_performed",
            "authorization_decision": "not_performed",
            "checks": checks,
        }
        audit.append({
            "sequence": i + 1, "record_id": record["record_id"],
            "actor": "deterministic-document-review",
            "operation": "add", "path": f"{base}.review",
            "before_sha256": before, "after_sha256": digest(record),
            "value": copy.deepcopy(record["review"]),
        })
    return {
        "schema_version": 1, "synthetic_data": True, "status": "reviewed",
        "human_review_required": True,
        "notice": "Synthetic documentation checks only; no clinical or authorization decisions; "
                  "no HIPAA or other compliance certification.",
        "records": updated, "audit_trail": audit,
    }


def reject_constant(_):
    raise ValidationError("Non-finite JSON numbers are not accepted")


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "$", "duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        check(len(argv) == 1, "$", "usage: python -B implementation.py INPUT.json")
        path = Path(argv[0])
        check(path.stat().st_size <= 1_000_000, "$", "input exceeds 1 MB limit")
        data = json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=reject_constant, object_pairs_hook=object_pairs)
        output = review(data)
        code = 0
    except (OSError, ValueError, RecursionError) as exc:
        # File and JSON decoder details can contain sensitive input; never echo them.
        message = str(exc) if isinstance(exc, ValidationError) else "Cannot read valid input JSON"
        output = {"status": "error", "error": message}
        code = 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
