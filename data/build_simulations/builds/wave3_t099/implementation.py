"""Offline, synthetic-only healthcare research reference. No clinical decisions.

Run: python -B implementation.py example_input.json
Only fixture-backed retrieval is supported; URLs never cause network requests.
The narrow accepted resource vocabulary demonstrates identifier exclusion, not
HIPAA certification or general-purpose detection of identifiers in free text.
"""

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


BUILD_ID = "wave3_t099_single_web__ind-healthcare"
ALLOWED_HOSTS = frozenset({"guidance.synthetic.example"})
NOTES = frozenset({
    "SYNTHETIC: Routine follow-up. Human clinician review required.",
    "SYNTHETIC: Lab results recorded. Human clinician review required.",
    "SYNTHETIC: Prior authorization documentation pending human review.",
})
REVIEW = [
    {"url": "urn:demo:human-review-required", "valueBoolean": True},
    {"url": "urn:demo:clinical-decision", "valueCode": "not-performed"},
]


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected),
            label + ": unexpected or missing fields")


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def validate_url(url):
    require(type(url) is str and len(url) <= 240, "Invalid URL")
    try:
        parsed = urlsplit(url)
        require(parsed.scheme == "https" and parsed.netloc in ALLOWED_HOSTS,
                "URL origin is not allowlisted")
        require(not parsed.query and not parsed.fragment and
                re.fullmatch(r"/[a-zA-Z0-9/_-]+(?:\.txt)?", parsed.path),
                "URL must have a simple path and no query or fragment")
        require(url == "https://" + parsed.netloc + parsed.path,
                "URL must be canonical")
    except ValueError as exc:
        raise ValidationError("Invalid URL") from exc
    return url


def validate_record(record, processed=False):
    require(type(record) is dict, "Record must be an object")
    kind = record.get("resourceType")
    shapes = {
        "Patient": {"resourceType", "id", "synthetic", "ageBand",
                    "diagnosisCodes", "labs"},
        "DocumentReference": {"resourceType", "id", "synthetic", "subject", "text"},
        "Claim": {"resourceType", "id", "synthetic", "patient", "status",
                  "use", "serviceCode"},
    }
    require(type(kind) is str and kind in shapes, "Unsupported resourceType")
    keys(record, shapes[kind] | ({"extension"} if processed else set()), "Record")
    prefix = {"Patient": "patient", "DocumentReference": "note", "Claim": "auth"}[kind]
    require(type(record["id"]) is str and
            re.fullmatch(prefix + r"-synthetic-[0-9]{3}", record["id"]),
            "Only synthetic, non-linkable record IDs are accepted")
    require(record["synthetic"] is True, "Real patient data is prohibited")
    if processed:
        require(record["extension"] == REVIEW, "Human review safeguards required")
    if kind == "Patient":
        require(record["ageBand"] in ("adult", "older-adult"),
                "Only broad age bands are accepted; no birth dates")
        codes = record["diagnosisCodes"]
        require(type(codes) is list and 1 <= len(codes) <= 4 and
                all(type(c) is str and c in ("E11.9", "I10", "Z00.00") for c in codes),
                "Unsupported synthetic diagnosis codes")
        labs = record["labs"]
        require(type(labs) is list and 1 <= len(labs) <= 5, "Invalid lab list")
        for lab in labs:
            keys(lab, {"code", "value", "unit"}, "Lab")
            require(lab["code"] == "2345-7" and lab["unit"] == "mg/dL",
                    "Only the demonstration glucose lab is supported")
            require(type(lab["value"]) in (int, float) and
                    math.isfinite(lab["value"]) and 0 <= lab["value"] <= 1000,
                    "Lab value must be finite and within the demo range")
    elif kind == "DocumentReference":
        require(type(record["text"]) is str and record["text"] in NOTES,
                "Clinical notes must use an identifier-free synthetic template")
    else:
        require(record["status"] == "draft" and record["use"] == "preauthorization"
                and record["serviceCode"] == "DEMO-LAB",
                "Prior authorizations remain draft; no approval or denial")


def validate_records(records, processed=False):
    require(type(records) is list and len(records) == 3,
            "Exactly one patient, clinical note, and prior authorization required")
    for record in records:
        validate_record(record, processed)
    by_type = {r["resourceType"]: r for r in records}
    require(set(by_type) == {"Patient", "DocumentReference", "Claim"},
            "All three resource types required")
    reference = {"reference": "Patient/" + by_type["Patient"]["id"]}
    require(by_type["DocumentReference"]["subject"] == reference and
            by_type["Claim"]["patient"] == reference, "Broken patient reference")


def validate_request(request):
    keys(request, {"schema_version", "synthetic", "records", "research"}, "Input")
    require(request["schema_version"] == "1.0" and request["synthetic"] is True,
            "Synthetic schema version 1.0 required")
    validate_records(request["records"])
    research = request["research"]
    keys(research, {"urls", "fixtures"}, "Research")
    urls = research["urls"]
    require(type(urls) is list and 1 <= len(urls) <= 5, "Supply 1 to 5 URLs")
    for url in urls:
        validate_url(url)
    require(len(set(urls)) == len(urls), "Duplicate URLs are prohibited")
    fixtures = research["fixtures"]
    require(type(fixtures) is dict and set(fixtures) == set(urls),
            "Every URL requires exactly one local retrieval fixture")
    for fixture in fixtures.values():
        keys(fixture, {"content_type", "body", "source_label"}, "Fixture")
        require(fixture["content_type"] == "text/plain", "Only plain text is supported")
        require(fixture["source_label"] == "SYNTHETIC educational fixture",
                "Sources must be labeled synthetic")
        body = fixture["body"]
        require(type(body) is str and 0 < len(body) <= 5000 and body.strip(),
                "Source body must be nonempty and bounded")
        require(len(body.splitlines()) <= 20, "Source has too many lines")
        require(all(c in "\n\t" or ord(c) >= 32 for c in body),
                "Source contains control characters")


def validate_result(result, request):
    keys(result, {"schema_version", "synthetic", "status", "human_review_required",
                  "clinical_decision", "records", "sources", "findings", "audit"},
         "Output")
    require(result["schema_version"] == "1.0" and result["synthetic"] is True and
            result["status"] == "ok" and result["human_review_required"] is True and
            result["clinical_decision"] == "not-performed", "Unsafe output envelope")
    validate_records(result["records"], processed=True)
    require(len(result["audit"]) == len(request["records"]), "Incomplete audit")
    for index, (before, after, event) in enumerate(zip(
            request["records"], result["records"], result["audit"]), 1):
        expected = copy.deepcopy(before)
        expected["extension"] = copy.deepcopy(REVIEW)
        require(after == expected, "Untracked record change")
        require(event == {
            "sequence": index, "resource_id": before["id"],
            "actor": "deterministic-reference", "action": "add-human-review-flags",
            "changed_fields": ["extension"], "before": before, "after": after,
            "before_sha256": digest(before), "after_sha256": digest(after),
        }, "Invalid record audit")
    expected_sources, expected_findings = retrieve(request["research"])
    require(result["sources"] == expected_sources and
            result["findings"] == expected_findings, "Invalid finding provenance")


def retrieve(research):
    """Treat every source as inert text, never as instructions or medical advice."""
    sources, findings = [], []
    for url in research["urls"]:
        fixture = research["fixtures"][url]
        body = fixture["body"]
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        source_id = "source-" + str(len(sources) + 1)
        sources.append({
            "id": source_id, "requested_url": url, "resolved_url": url,
            "retrieval_mode": "offline-fixture", "content_type": "text/plain",
            "source_label": fixture["source_label"], "body": body,
            "body_sha256": body_hash,
        })
        offset = 0
        for line in body.splitlines(keepends=True):
            quote = line.rstrip("\r\n")
            if quote.strip():
                findings.append({
                    "id": "finding-" + str(len(findings) + 1),
                    "kind": "uninterpreted-source-excerpt", "source_id": source_id,
                    "url": url, "body_sha256": body_hash, "quote": quote,
                    "start_character": offset, "end_character": offset + len(quote),
                    "human_review_required": True, "clinical_decision": "not-performed",
                })
            offset += len(line)
    return sources, findings


def process(request):
    validate_request(request)
    records, audit = [], []
    for index, record in enumerate(request["records"], 1):
        after = copy.deepcopy(record)
        after["extension"] = copy.deepcopy(REVIEW)
        records.append(after)
        audit.append({
            "sequence": index, "resource_id": record["id"],
            "actor": "deterministic-reference", "action": "add-human-review-flags",
            "changed_fields": ["extension"], "before": copy.deepcopy(record),
            "after": copy.deepcopy(after), "before_sha256": digest(record),
            "after_sha256": digest(after),
        })
    sources, findings = retrieve(request["research"])
    result = {
        "schema_version": "1.0", "synthetic": True, "status": "ok",
        "human_review_required": True, "clinical_decision": "not-performed",
        "records": records, "sources": sources, "findings": findings, "audit": audit,
    }
    validate_result(result, request)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON fields are prohibited")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON number")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 100000, "Input file is too large")
        request = json.loads(path.read_text(encoding="utf-8"),
                             object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = process(request)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError,
            OverflowError, RecursionError):
        # Do not echo patient data, source bodies, or local paths on errors.
        print(json.dumps({"status": "error", "message":
                          "Invalid input or unreadable file; see the synthetic schema."}))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
