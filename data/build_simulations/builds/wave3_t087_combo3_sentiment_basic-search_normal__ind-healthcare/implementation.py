"""Synthetic, non-clinical healthcare information pipeline; standard library only.

FHIR-style resources, not a FHIR conformance implementation. The strict allowlist
and text checks demonstrate identifier exclusion, not certified HIPAA compliance.
Free text cannot be comprehensively de-identified with deterministic rules.
Only synthetic inputs are permitted; findings always require human review.
"""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected JSON object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            "Missing or unsupported fields")


def text(value):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 10000,
            "Expected nonempty bounded text")
    # Deliberately conservative demonstrations; arbitrary PHI detection is not claimed.
    require(not re.search(
        r"[\w.+-]+@[\w.-]+\.\w+|\b\d{3}[- ]\d{2}[- ]\d{4}\b|"
        r"\b\d{3}[- .]\d{3}[- .]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|"
        r"\b(?:MRN|DOB|born|birthdate|patient name|named|address)\b",
        value, re.I), "Potential patient identifier in text")
    return value


def identifier(value):
    require(isinstance(value, str) and
            re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value) is not None,
            "Invalid synthetic identifier")
    return value


def items(value, maximum=100):
    require(isinstance(value, list) and len(value) <= maximum, "Expected bounded list")
    return value


def unique(values):
    require(len(values) == len(set(values)), "Duplicate identifier")


def validate_input(data):
    fields(data, ("schema_version", "synthetic", "resources", "catalog", "sources"))
    require(data["schema_version"] == "1.0" and data["synthetic"] is True,
            "Only schema 1.0 explicitly synthetic fixtures are accepted")
    resources = items(data["resources"])
    require(resources, "At least one resource required")
    patients = set()
    for record in resources:
        require(isinstance(record, dict), "Resource must be object")
        kind = record.get("resourceType")
        if kind == "Patient":
            fields(record, ("resourceType", "id", "synthetic", "human_review_required"))
            require(isinstance(record["id"], str) and
                    re.fullmatch(r"anon-p[0-9]{3,6}", record["id"]),
                    "Patient ID must be a synthetic anonymous alias")
            patients.add(record["id"])
        elif kind == "ClinicalNote":
            fields(record, ("resourceType", "id", "synthetic", "human_review_required",
                            "subject", "text"))
            text(record["text"])
        elif kind == "PriorAuthorizationRequest":
            fields(record, ("resourceType", "id", "synthetic", "human_review_required",
                            "subject", "reason", "status"))
            require(record["status"] in ("draft", "pending", "denied", "approved"),
                    "Invalid prior authorization status")
            text(record["reason"])
        else:
            raise ValidationError("Unsupported resource type")
        identifier(record["id"])
        require(record["synthetic"] is True and record["human_review_required"] is True,
                "Synthetic and human review flags are mandatory")
    unique([r["id"] for r in resources])
    for record in resources:
        if "subject" in record:
            require(isinstance(record["subject"], str) and
                    record["subject"] in {"Patient/" + p for p in patients},
                    "Unknown anonymous patient reference")
    source_ids = []
    for source in items(data["sources"]):
        fields(source, ("id", "title", "passages"))
        source_ids.append(identifier(source["id"]))
        text(source["title"])
        passage_ids = []
        for passage in items(source["passages"]):
            fields(passage, ("id", "text"))
            passage_ids.append(identifier(passage["id"]))
            text(passage["text"])
        unique(passage_ids)
    unique(source_ids)
    product_ids = []
    for product in items(data["catalog"]):
        fields(product, ("id", "title", "description", "keywords", "source_ids"))
        product_ids.append(identifier(product["id"]))
        text(product["title"])
        text(product["description"])
        for keyword in items(product["keywords"]):
            text(keyword)
        for source_id in items(product["source_ids"]):
            identifier(source_id)
            require(source_id in source_ids, "Unknown catalog source")
        unique(product["source_ids"])
    unique(product_ids)
    return data


STOP = {"a", "an", "the", "is", "was", "and", "or", "to", "for", "of", "in",
        "with", "i", "my", "it", "has", "patient", "synthetic"}
SYNONYMS = {"coverage": "insurance", "insurer": "insurance", "auth": "authorization",
            "preauthorization": "authorization", "approval": "authorization",
            "rejected": "denied", "rejection": "denied", "waiting": "delay",
            "delayed": "delay", "refunds": "refund", "appointments": "appointment"}
POSITIVE = {"helpful", "happy", "excellent", "clear", "resolved", "thanks"}
NEGATIVE = {"angry", "frustrated", "confusing", "delay", "denied", "poor", "worried"}
SEVERITY = {"urgent": 3, "unsafe": 3, "emergency": 3, "denied": 2,
            "delay": 2, "confusing": 1}


def tokens(value):
    return [SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+", value.lower())]


def terms(value):
    return sorted(set(tokens(value)) - STOP)


def score_sentiment(value):
    words = tokens(value)
    evidence = []
    for index, word in enumerate(words):
        base = 1 if word in POSITIVE else -1 if word in NEGATIVE else 0
        if base:
            negated = any(w in {"not", "never", "no"} for w in words[max(0, index-2):index])
            evidence.append({"term": word, "weight": -base if negated else base,
                             "negated": negated})
    raw = sum(e["weight"] for e in evidence)
    severity_terms = sorted(set(words) & set(SEVERITY))
    severity = max((SEVERITY[w] for w in severity_terms), default=0)
    return {"score": raw, "label": "positive" if raw > 0 else "negative" if raw < 0 else "neutral",
            "evidence": evidence, "severity": severity, "severity_terms": severity_terms,
            "priority": severity * 100 + min(99, max(0, -raw)),
            "query_terms": terms(value),
            "human_review_required": True}


def record_text(record, resources):
    if record["resourceType"] == "Patient":
        return " ".join(r.get("text", r.get("reason", "")) for r in resources
                        if r.get("subject") == "Patient/" + record["id"])
    return record.get("text", record.get("reason", ""))


def search_insight(insight, catalog):
    query = set(insight["query_terms"])
    results = []
    for product in catalog:
        matched = sorted(query & set(terms(" ".join(
            [product["title"], product["description"]] + product["keywords"]))))
        if matched:
            results.append({"product_id": product["id"], "score": len(matched),
                            "matched_terms": matched, "source_ids": product["source_ids"][:]})
    results.sort(key=lambda r: (-r["score"], r["product_id"]))
    return {"query_terms": insight["query_terms"][:], "priority": insight["priority"],
            "results": results[:3], "human_review_required": True}


def extract_findings(search, sources):
    allowed = {s for r in search["results"] for s in r["source_ids"]}
    query = set(search["query_terms"])
    candidates = []
    for source in sources:
        if source["id"] not in allowed:
            continue
        for passage in source["passages"]:
            overlap = sorted(query & set(terms(passage["text"])))
            if overlap:
                candidates.append({"source_id": source["id"], "passage_id": passage["id"],
                                   "quote": passage["text"], "start": 0,
                                   "end": len(passage["text"]), "matched_terms": overlap})
    candidates.sort(key=lambda c: (-len(c["matched_terms"]), c["source_id"], c["passage_id"]))
    return {"findings": candidates[:3], "priority": search["priority"],
            "human_review_required": True, "clinical_decision": None,
            "evidence_status": "found" if candidates else "no_matching_evidence"}


STAGES = ("sentiment", "search", "research")


def expected_value(stage, row, data):
    if stage == "sentiment":
        return score_sentiment(record_text(row["resource"], data["resources"]))
    if stage == "search":
        return search_insight(row["sentiment"], data["catalog"])
    return extract_findings(row["search"], data["sources"])


def validate_envelope(envelope, stage):
    """One validation boundary for all stage handoffs, including audit replay."""
    require(stage in ("input",) + STAGES, "Unknown stage")
    fields(envelope, ("schema_version", "status", "stage", "human_review_required",
                      "clinical_decision", "input", "records", "audit_trail"))
    require(envelope["schema_version"] == "1.0" and envelope["status"] == "ok"
            and envelope["stage"] == stage, "Invalid envelope header")
    require(envelope["human_review_required"] is True and envelope["clinical_decision"] is None,
            "Human-only clinical decisions required")
    data = validate_input(envelope["input"])
    completed = () if stage == "input" else STAGES[:STAGES.index(stage)+1]
    rows = items(envelope["records"])
    require(len(rows) == len(data["resources"]), "Resource count changed")
    expected_audit = []
    for row, original in zip(rows, data["resources"]):
        fields(row, ("resource",) + completed)
        require(row["resource"] == original, "Clinical resource was modified")
        for name in completed:
            require(row[name] == expected_value(name, row, data),
                    "Invalid or altered stage output")
    for name in completed:
        for row in rows:
            expected_audit.append({"sequence": len(expected_audit)+1, "stage": name,
                                   "record_id": row["resource"]["id"],
                                   "operation": "add_enrichment", "path": name,
                                   "before": None, "after": row[name],
                                   "actor": "deterministic-reference",
                                   "human_review_required": True})
    require(envelope["audit_trail"] == expected_audit, "Missing or altered change audit")
    return envelope


def start(data):
    validate_input(data)
    return validate_envelope({
        "schema_version": "1.0", "status": "ok", "stage": "input",
        "human_review_required": True, "clinical_decision": None,
        "input": copy.deepcopy(data),
        "records": [{"resource": copy.deepcopy(r)} for r in data["resources"]],
        "audit_trail": []}, "input")


def advance(envelope, stage):
    require(stage in STAGES, "Unknown stage")
    previous = "input" if stage == "sentiment" else STAGES[STAGES.index(stage)-1]
    validate_envelope(envelope, previous)
    output = copy.deepcopy(envelope)
    output["stage"] = stage
    for row in output["records"]:
        value = expected_value(stage, row, output["input"])
        row[stage] = value
        output["audit_trail"].append({
            "sequence": len(output["audit_trail"])+1, "stage": stage,
            "record_id": row["resource"]["id"], "operation": "add_enrichment",
            "path": stage, "before": None, "after": copy.deepcopy(value),
            "actor": "deterministic-reference", "human_review_required": True})
    return validate_envelope(output, stage)


def run(data):
    envelope = start(data)
    for stage in STAGES:
        envelope = advance(envelope, stage)
    return envelope


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = handle.read(1000001)
        require(len(payload) <= 1000000, "Input exceeds size limit")
        data = json.loads(payload, object_pairs_hook=reject_duplicates,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValidationError("Non-finite JSON number")))
        result = run(data)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        # Never echo file paths or invalid input, which may contain identifiers.
        print(json.dumps({"status": "error", "message": "Input/file validation failed"}))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
