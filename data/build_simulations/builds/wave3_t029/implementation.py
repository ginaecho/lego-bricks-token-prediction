"""Synthetic, deterministic document-to-evidence reference pipeline; not medical advice."""
import copy
import hashlib
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required fields")
    require(value.keys() <= set(required) | set(optional), "Unsupported fields")


def text(value, label, limit=2000):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit,
            "Invalid " + label)


def code(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,39}", value),
            "Invalid clinical code")


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid numeric value")


def token(value):
    return "deid-" + hashlib.sha256(("synthetic-demo:" + value).encode()).hexdigest()[:20]


def fact(kind, clinical_code, value=None, unit=None):
    result = {"kind": kind, "code": clinical_code, "value": value, "unit": unit}
    validate_fact(result)
    return result


def validate_fact(value):
    keys(value, ("kind", "code", "value", "unit"))
    require(value["kind"] in ("diagnosis", "lab", "requested-service"), "Invalid fact kind")
    code(value["code"])
    if value["kind"] == "lab":
        number(value["value"])
        require(isinstance(value["unit"], str) and
                re.fullmatch(r"[A-Za-z%/0-9.^-]{1,20}", value["unit"]), "Invalid lab unit")
    else:
        require(value["value"] is None and value["unit"] is None, "Unexpected fact values")


def parse_note(note):
    """A bounded clinical-note grammar avoids copying free-text identifiers downstream."""
    text(note, "clinical note", 10000)
    facts = []
    for line in note.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("Diagnosis:", "Request:"):
            facts.append(fact("diagnosis" if parts[0] == "Diagnosis:" else
                              "requested-service", parts[1]))
        elif len(parts) == 4 and parts[0] == "Lab:":
            try:
                value = float(parts[2])
            except ValueError:
                raise ValidationError("Invalid lab number") from None
            facts.append(fact("lab", parts[1], value, parts[3]))
        else:
            raise ValidationError("Unsupported note line; identifiers and prose are not accepted")
    require(bool(facts), "Clinical note must contain facts")
    unique = []
    for item in facts:
        if item not in unique:
            unique.append(item)
    return unique


def validate_input(data):
    keys(data, ("schema_version", "synthetic", "resources", "research"))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Unsupported schema version")
    require(data["synthetic"] is True, "Only explicitly synthetic fixtures are accepted")
    require(isinstance(data["resources"], list) and 0 < len(data["resources"]) <= 1000,
            "Expected 1 to 1000 resources")
    seen, patients = set(), set()
    for resource in data["resources"]:
        require(isinstance(resource, dict), "Invalid resource")
        kind = resource.get("resourceType")
        fields = {
            "Patient": ("resourceType", "id"),
            "ClinicalNote": ("resourceType", "id", "subject", "text"),
            "PriorAuthorizationRequest": ("resourceType", "id", "subject", "serviceCode"),
        }
        require(kind in fields, "Unsupported resource type")
        keys(resource, fields[kind])
        prefix = {"Patient": "P", "ClinicalNote": "N", "PriorAuthorizationRequest": "A"}[kind]
        require(isinstance(resource["id"], str) and
                re.fullmatch(r"SYN-" + prefix + r"[1-9][0-9]{0,8}", resource["id"]),
                "Resource IDs must be synthetic tokens")
        require(resource["id"] not in seen, "Duplicate resource ID")
        seen.add(resource["id"])
        if kind == "Patient":
            patients.add(resource["id"])
        else:
            keys(resource["subject"], ("reference",))
            text(resource["subject"]["reference"], "subject reference", 40)
        if kind == "ClinicalNote":
            parse_note(resource["text"])
        elif kind == "PriorAuthorizationRequest":
            code(resource["serviceCode"])
    require(bool(patients), "At least one synthetic patient is required")
    for resource in data["resources"]:
        if resource["resourceType"] != "Patient":
            require(resource["subject"]["reference"] in
                    {"Patient/" + patient for patient in patients}, "Unresolved patient reference")
    research = data["research"]
    keys(research, ("question", "focus_codes", "sources"))
    text(research["question"], "research question")
    require(isinstance(research["focus_codes"], list), "Invalid focus codes")
    for item in research["focus_codes"]:
        code(item)
    require(isinstance(research["sources"], list) and len(research["sources"]) <= 1000,
            "Invalid source collection")
    seen_sources = set()
    for source in research["sources"]:
        keys(source, ("id", "synthetic", "text", "claims"))
        require(isinstance(source["id"], str) and re.fullmatch(r"SRC-[1-9][0-9]{0,8}", source["id"]),
                "Invalid source ID")
        require(source["id"] not in seen_sources, "Duplicate source ID")
        seen_sources.add(source["id"])
        require(source["synthetic"] is True, "Sources must be synthetic")
        text(source["text"], "source text", 10000)
        require(isinstance(source["claims"], list), "Invalid source claims")
        for claim in source["claims"]:
            keys(claim, ("code", "stance", "quote"))
            code(claim["code"])
            require(claim["stance"] in ("supports", "contradicts"), "Invalid evidence stance")
            text(claim["quote"], "quote")
            require(claim["quote"] in source["text"], "Evidence quote is absent from source")
            require(re.search(r"(?<![A-Z0-9.-])" + re.escape(claim["code"]) +
                              r"(?![A-Z0-9.-])", claim["quote"]) is not None,
                    "Evidence quote must contain exact clinical code")


def record_state(record):
    return {key: copy.deepcopy(value) for key, value in record.items() if key != "audit"}


def audit_change(record, stage, before):
    record["audit"].append({"sequence": len(record["audit"]) + 1, "stage": stage,
                            "before": before, "after": record_state(record)})


def validate_envelope(envelope, expected_stage):
    """Both stages use the same record schema, invariants, and audit replay checks."""
    keys(envelope, ("schema_version", "status", "stage", "synthetic", "human_review_required",
                    "clinical_decision", "records", "research", "limitations"))
    require(envelope["schema_version"] == 1 and envelope["status"] == "ok" and
            envelope["synthetic"] is True and envelope["stage"] == expected_stage,
            "Invalid pipeline envelope")
    require(envelope["human_review_required"] is True and
            envelope["clinical_decision"] == "not_performed", "Human review cannot be bypassed")
    require(isinstance(envelope["records"], list) and envelope["records"], "Missing records")
    ids = set()
    patients = {r.get("id") for r in envelope["records"]
                if isinstance(r, dict) and r.get("kind") == "patient"}
    for record in envelope["records"]:
        keys(record, ("id", "kind", "patient_ref", "facts", "evidence", "human_review_required", "audit"))
        require(isinstance(record["id"], str) and
                re.fullmatch(r"deid-[a-f0-9]{20}", record["id"]), "Non-deidentified record ID")
        require(record["id"] not in ids, "Duplicate normalized record")
        ids.add(record["id"])
        require(record["kind"] in ("patient", "clinical-note", "prior-authorization-request"),
                "Invalid record kind")
        require(record["patient_ref"] in patients and record["human_review_required"] is True,
                "Invalid patient linkage or missing review flag")
        require(isinstance(record["facts"], list), "Invalid facts")
        for item in record["facts"]:
            validate_fact(item)
        require(isinstance(record["evidence"], list), "Invalid evidence")
        fact_codes = {item["code"] for item in record["facts"]}
        for evidence in record["evidence"]:
            keys(evidence, ("code", "source_id", "stance", "quote_sha256"))
            require(evidence["code"] in fact_codes, "Evidence disconnected from document facts")
            require(re.fullmatch(r"SRC-[1-9][0-9]{0,8}", evidence["source_id"]) is not None,
                    "Invalid evidence source")
            require(evidence["stance"] in ("supports", "contradicts"), "Invalid evidence stance")
            require(re.fullmatch(r"[a-f0-9]{64}", evidence["quote_sha256"]) is not None,
                    "Invalid quote fingerprint")
        require(isinstance(record["audit"], list) and
                len(record["audit"]) == (1 if expected_stage == "documents" else 2), "Missing audit")
        previous = None
        for index, event in enumerate(record["audit"], 1):
            keys(event, ("sequence", "stage", "before", "after"))
            require(event["sequence"] == index and event["before"] == previous and
                    event["stage"] == ("documents" if index == 1 else "research"),
                    "Broken audit chain")
            previous = event["after"]
        require(previous == record_state(record), "Unaudited record change")
    if expected_stage == "documents":
        require(envelope["research"] is None and
                all(not r["evidence"] for r in envelope["records"]), "Premature research output")
    else:
        keys(envelope["research"], ("question_sha256", "findings", "disposition"))
        require(envelope["research"]["disposition"] == "human_review_only", "Autonomous decision")
    return envelope


def document_stage(data):
    validate_input(data)
    records = []
    for resource in data["resources"]:
        kind = resource["resourceType"]
        patient_id = resource["id"] if kind == "Patient" else resource["subject"]["reference"].split("/")[1]
        facts = parse_note(resource["text"]) if kind == "ClinicalNote" else (
            [fact("requested-service", resource["serviceCode"])] if kind == "PriorAuthorizationRequest" else [])
        record = {
            "id": token(resource["id"]),
            "kind": {"Patient": "patient", "ClinicalNote": "clinical-note",
                     "PriorAuthorizationRequest": "prior-authorization-request"}[kind],
            "patient_ref": token(patient_id), "facts": facts, "evidence": [],
            "human_review_required": True, "audit": [],
        }
        audit_change(record, "documents", None)
        records.append(record)
    result = {"schema_version": 1, "status": "ok", "stage": "documents", "synthetic": True,
              "human_review_required": True, "clinical_decision": "not_performed",
              "records": records, "research": None,
              "limitations": [
                  "Clearly labeled synthetic fixtures only; no real patient data.",
                  "Demonstrative HIPAA-oriented identifier exclusion, not certification or Safe Harbor verification.",
                  "Stable synthetic pseudonyms are linkable, not anonymization.",
                  "Source assertions are not verified medical truth; no treatment or authorization decision.",
                  "FHIR-style subset, not full HL7 FHIR conformance; restricted clinical-note grammar.",
                  "Audit snapshots cover pipeline record changes, not tamper-proof storage.",
              ]}
    return validate_envelope(result, "documents")


def research_stage(documents, research):
    validate_envelope(documents, "documents")
    # Validate research through the same input validator without re-reading raw documents.
    validate_input({"schema_version": 1, "synthetic": True,
                    "resources": [{"resourceType": "Patient", "id": "SYN-P1"}],
                    "research": research})
    result = copy.deepcopy(documents)
    focus = set(research["focus_codes"])
    observed = {f["code"] for r in result["records"] for f in r["facts"]}
    targets = focus or observed
    findings = []
    index = {}
    for clinical_code in sorted(targets):
        matches = []
        if clinical_code in observed:
            for source in research["sources"]:
                for claim in source["claims"]:
                    if claim["code"] == clinical_code:
                        evidence = {"code": clinical_code, "source_id": source["id"],
                                    "stance": claim["stance"],
                                    "quote_sha256": hashlib.sha256(claim["quote"].encode()).hexdigest()}
                        if evidence not in matches:
                            matches.append(evidence)
        index[clinical_code] = matches
        stances = {m["stance"] for m in matches}
        assessment = ("not_in_documents" if clinical_code not in observed else
                      "no_evidence" if not matches else "conflicting" if len(stances) > 1 else
                      "support_only" if "supports" in stances else "contradiction_only")
        findings.append({"code": clinical_code, "assessment": assessment,
                         "source_ids": sorted({m["source_id"] for m in matches}),
                         "human_review_required": True})
    for record in result["records"]:
        before = record_state(record)
        record["evidence"] = [copy.deepcopy(match)
                              for clinical_code in sorted({f["code"] for f in record["facts"]})
                              for match in index.get(clinical_code, [])]
        audit_change(record, "research", before)
    result["stage"] = "research"
    result["research"] = {
        "question_sha256": hashlib.sha256(research["question"].encode()).hexdigest(),
        "findings": findings, "disposition": "human_review_only",
    }
    return validate_envelope(result, "research")


def run_pipeline(data):
    return research_stage(document_stage(data), data["research"])


def reject_constant(value):
    raise ValidationError("Non-finite JSON number")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
        print(json.dumps(output, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        # Never echo raw patient input, file paths, or parser excerpts in diagnostics.
        print(json.dumps({"status": "error", "message": "Input file or schema validation failed."}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
