"""Offline synthetic web research -> document review reference pipeline.

URLs are exact fixture keys, never network requests. Evidence matching is literal
and case-sensitive; matches indicate textual support, not truth or certification.
"""

import copy
import hashlib
import json
import re
import sys
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, label):
    require(type(value) is dict, label + " must be an object")
    require(set(value) == set(expected), label + " has missing or unknown fields")


def text(value, label, limit=100_000):
    require(type(value) is str and bool(value.strip()), label + " must be nonblank text")
    require(len(value) <= limit, label + " exceeds size limit")
    require(not any(0xD800 <= ord(c) <= 0xDFFF for c in value),
            label + " contains invalid Unicode")


def strings(value, label, limit=100):
    require(type(value) is list and 0 < len(value) <= limit,
            label + " must be a nonempty bounded array")
    for item in value:
        text(item, label + " item", 2000)
    require(len(set(value)) == len(value), label + " contains duplicates")


def allowed_url(url, hosts):
    text(url, "URL", 2000)
    require(not any(c.isspace() or ord(c) < 32 for c in url)
            and "\\" not in url, "URL contains whitespace or backslash")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(parsed.scheme == "https" and parsed.hostname in hosts,
            "URL requires HTTPS and an exact allowlisted host")
    require(parsed.username is None and parsed.password is None
            and port in (None, 443) and not parsed.fragment,
            "URL credentials, non-HTTPS ports and fragments are prohibited")


def validate_request(request):
    fields(request, ["allowlisted_hosts", "urls", "fixtures", "document",
                     "requirements"], "request")
    hosts = request["allowlisted_hosts"]
    strings(hosts, "allowlisted_hosts")
    for host in hosts:
        require(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host) is not None,
                "Allowlisted hosts must be lowercase ASCII hostnames")
    strings(request["urls"], "urls", 30)
    for url in request["urls"]:
        allowed_url(url, hosts)
    fixtures = request["fixtures"]
    require(type(fixtures) is dict and len(fixtures) <= 100,
            "fixtures must be an object with at most 100 entries")
    for url, fixture in fixtures.items():
        allowed_url(url, hosts)
        require(type(fixture) is dict, "fixture must be an object")
        code = fixture.get("status_code")
        require(type(code) is int and code in (200, 301, 302, 404, 503),
                "Unsupported fixture status_code")
        if code == 200:
            fields(fixture, ["status_code", "title", "text"], "page fixture")
            text(fixture["title"], "fixture title", 2000)
            require(type(fixture["text"]) is str, "fixture text must be a string")
            if fixture["text"]:
                text(fixture["text"], "fixture text")
        elif code in (301, 302):
            fields(fixture, ["status_code", "location"], "redirect fixture")
            allowed_url(fixture["location"], hosts)
        else:
            fields(fixture, ["status_code"], "error fixture")
    fields(request["document"], ["id", "title", "text"], "document")
    for key in ("id", "title"):
        text(request["document"][key], "document " + key, 2000)
    require(type(request["document"]["text"]) is str, "document text must be a string")
    if request["document"]["text"]:
        text(request["document"]["text"], "document text")
    requirements = request["requirements"]
    require(type(requirements) is list and 0 < len(requirements) <= 100,
            "requirements must be a nonempty bounded array")
    ids = []
    for requirement in requirements:
        fields(requirement, ["id", "description", "document_terms", "evidence_terms"],
               "requirement")
        text(requirement["id"], "requirement id", 200)
        text(requirement["description"], "requirement description", 2000)
        strings(requirement["document_terms"], "document_terms", 20)
        strings(requirement["evidence_terms"], "evidence_terms", 20)
        ids.append(requirement["id"])
    require(len(set(ids)) == len(ids), "Duplicate requirement id")


def retrieve(request, url, source_id):
    source = {"id": source_id, "requested_url": url, "final_url": url,
              "redirect_chain": [], "status": "error", "error": None,
              "title": None, "text": None, "sha256": None}
    visited = set()
    while True:
        allowed_url(url, request["allowlisted_hosts"])
        source["final_url"] = url
        if url in visited:
            source["error"] = "redirect_loop"
            return source
        visited.add(url)
        source["redirect_chain"].append(url)
        fixture = request["fixtures"].get(url)
        if fixture is None:
            source["error"] = "fixture_missing"
            return source
        code = fixture["status_code"]
        if code in (301, 302):
            if len(source["redirect_chain"]) > 5:
                source["error"] = "redirect_limit"
                return source
            url = fixture["location"]
            continue
        if code != 200:
            source["error"] = "http_" + str(code)
            return source
        body = fixture["text"]
        source.update(status="retrieved", title=fixture["title"], text=body,
                      sha256=hashlib.sha256(body.encode("utf-8")).hexdigest())
        return source


def research(request):
    sources = [retrieve(request, url, "src-" + str(index))
               for index, url in enumerate(request["urls"], 1)]
    findings = []
    for requirement in request["requirements"]:
        for term in requirement["evidence_terms"]:
            for source in sources:
                if source["status"] != "retrieved":
                    continue
                offset = source["text"].find(term)
                if offset < 0:
                    continue
                findings.append({
                    "id": "finding-" + str(len(findings) + 1),
                    "requirement_id": requirement["id"], "term": term,
                    "source_id": source["id"], "url": source["final_url"],
                    "source_sha256": source["sha256"], "quote": term,
                    "start": offset, "end": offset + len(term)})
    return {"mode": "synthetic_fixture_retrieval", "sources": sources,
            "findings": findings}


def review(request, validated_research):
    body = request["document"]["text"]
    checks = []
    for requirement in request["requirements"]:
        findings = [f for f in validated_research["findings"]
                    if f["requirement_id"] == requirement["id"]]
        evidence_terms = {f["term"] for f in findings}
        document_matches = []
        missing_document = []
        for term in requirement["document_terms"]:
            offset = body.find(term)
            if offset < 0:
                missing_document.append(term)
            else:
                document_matches.append({"quote": term, "start": offset,
                                         "end": offset + len(term)})
        missing_evidence = [term for term in requirement["evidence_terms"]
                            if term not in evidence_terms]
        gaps = []
        for kind, terms in (("document", missing_document),
                            ("evidence", missing_evidence)):
            for term in terms:
                gaps.append({"kind": kind, "term": term,
                             "requirement_id": requirement["id"],
                             "checked_source_ids": [s["id"] for s in
                                                    validated_research["sources"]],
                             "document_id": request["document"]["id"]})
        checks.append({
            "requirement_id": requirement["id"],
            "description": requirement["description"],
            "status": "gap" if gaps else "textually_supported",
            "document_matches": document_matches,
            "finding_ids": [f["id"] for f in findings], "gaps": gaps})
    return {
        "document_id": request["document"]["id"],
        "document_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "checks": checks, "gap_count": sum(len(c["gaps"]) for c in checks),
        "conclusion": "gaps_found" if any(c["gaps"] for c in checks)
                      else "textual_checks_passed",
        "notice": "Synthetic textual review only; not certification, compliance "
                  "approval, verification of truth, or professional advice."}


def validate(envelope, phase):
    """Shared schema and deterministic integrity validation for all handoffs.

    Replay is safe here because retrieval only reads the input's fixture map.
    Exact structural comparison rejects altered quotes, hashes, IDs and gaps.
    """
    require(phase in ("input", "researched", "ok"), "Invalid validation phase")
    fields(envelope, ["schema_version", "synthetic", "status", "request",
                      "research", "review"], "envelope")
    require(type(envelope["schema_version"]) is int
            and envelope["schema_version"] == 1, "Unsupported schema_version")
    require(envelope["synthetic"] is True, "synthetic must be true")
    require(envelope["status"] == phase, "Unexpected pipeline status")
    validate_request(envelope["request"])
    if phase == "input":
        require(envelope["research"] is None and envelope["review"] is None,
                "Input must not contain precomputed stages")
        return
    expected_research = research(envelope["request"])
    require(canonical(envelope["research"]) == canonical(expected_research),
            "Research handoff failed provenance/schema validation")
    if phase == "researched":
        require(envelope["review"] is None, "Review must follow validated research")
    else:
        expected_review = review(envelope["request"], envelope["research"])
        require(canonical(envelope["review"]) == canonical(expected_review),
                "Review output failed evidence/schema validation")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def run_pipeline(payload):
    validate(payload, "input")
    envelope = copy.deepcopy(payload)
    envelope["research"] = research(envelope["request"])
    envelope["status"] = "researched"
    validate(envelope, "researched")
    envelope["review"] = review(envelope["request"], envelope["research"])
    envelope["status"] = "ok"
    validate(envelope, "ok")
    return envelope


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Nonfinite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], "r", encoding="utf-8-sig") as handle:
            raw = handle.read(2_000_001)
        require(len(raw) <= 2_000_000, "Input file exceeds size limit")
        payload = json.loads(raw, object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = run_pipeline(payload)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
