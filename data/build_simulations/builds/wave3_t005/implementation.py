"""Synthetic, offline support-to-service-search reference pipeline."""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    pass


SCHEMA_VERSION = "1.0"
FIXTURE_LABEL = "SYNTHETIC - no real citizens or cases"
STOP_WORDS = set("a an the i my me for to of and or is are can how do get need please help with want".split())
SYNONYMS = {
    "rent": "housing", "rental": "housing", "shelter": "housing",
    "groceries": "food", "grocery": "food", "meals": "food",
    "licence": "permit", "license": "permit", "permits": "permit",
    "applications": "application", "benefits": "benefit",
}
PII_PATTERNS = (
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)",
    r"\b\d+\s+[A-Z][A-Z .'-]*\s(?:street|road|avenue|lane|drive)\b",
)
NOTICE = "This is general guidance, not a benefits decision. A service team must review your application."


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(expected.split()), label + " has missing or unknown fields")


def text(value, label, limit=4000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            label + " must be nonempty bounded text")
    require(not any(ord(c) < 32 and c not in "\n\t" for c in value),
            label + " contains control characters")


def terms(value):
    return sorted({SYNONYMS.get(word, word) for word in re.findall(r"[a-z]+", value.lower())
                   if word not in STOP_WORDS and word != "redacted"})


def redact(value, citizen=None):
    for private in sorted((citizen or {}).values(), key=len, reverse=True):
        if private:
            value = re.sub(re.escape(private), "[redacted]", value, flags=re.I)
    for pattern in PII_PATTERNS:
        value = re.sub(pattern, "[redacted]", value, flags=re.I)
    return value


def public_text(value, label, citizen=None):
    text(value, label)
    require(redact(value, citizen) == value, label + " must not contain citizen PII")


def plain_text(value, label, citizen=None):
    public_text(value, label, citizen)
    require(all(len(sentence.split()) <= 28 for sentence in re.split(r"[.!?]", value)),
            label + " must use short sentences (at most 28 words)")
    require(not re.search(r"\b(hereinafter|aforementioned|pursuant|notwithstanding)\b", value, re.I),
            label + " must use plain language")


def policy_matches(envelope, query_terms):
    query = set(query_terms)
    return [policy for policy in envelope["policies"]
            if query.intersection(terms(policy["title"] + " " + policy["text"]))]


def expected_support(envelope):
    query_terms = terms(envelope["request"]["question"])
    matches = policy_matches(envelope, query_terms)
    return {
        "query_terms": query_terms,
        "answer": (" ".join(policy["plain_language_summary"] for policy in matches)
                   if matches else "I could not find guidance in the supplied policies. Please contact the service team."),
        "citations": [{"policy_id": p["id"], "evidence": p["text"]} for p in matches],
        "needs_human": not bool(matches),
        "notice": NOTICE,
        "explanation": "Guidance uses only supplied policies that share normalized words with your question.",
    }


def expected_search(envelope):
    support = envelope["support"]
    query = set(support["query_terms"])
    cited = {item["policy_id"] for item in support["citations"]}
    results = []
    for service in envelope["services"]:
        matched = sorted(query.intersection(terms(service["title"] + " " + service["description"])))
        linked = sorted(cited.intersection(service["policy_ids"]))
        if matched:
            results.append({
                "service_id": service["id"], "title": service["title"],
                "description": service["description"],
                "score": len(matched) * 2 + len(linked),
                "matched_terms": matched, "support_policy_ids": linked,
                "explanation": "Matched words: " + ", ".join(matched) + ".",
            })
    results.sort(key=lambda item: (-item["score"], item["service_id"]))
    return {
        "source_query_terms": list(support["query_terms"]),
        "source_policy_ids": sorted(cited),
        "results": results,
        "message": ("These services may help. Check the service details before applying."
                    if results else "No matching service was found. Please contact the service team."),
        "ranking_rule": "Two points per matched word, plus one per linked support policy. Ties use service ID.",
    }


def validate(envelope, phase):
    """One strict envelope schema; phase controls privacy and stage invariants."""
    require(phase in {"input", "supported", "ok"}, "Unknown validation phase")
    fields(envelope, "schema_version fixture_label status request benefits_application policies services support search",
           "Envelope")
    require(envelope["schema_version"] == SCHEMA_VERSION, "Unsupported schema_version")
    require(envelope["fixture_label"] == FIXTURE_LABEL, "Synthetic fixture label required")
    require(envelope["status"] == phase, "Unexpected pipeline status")
    request = envelope["request"]
    fields(request, "case_number kind question citizen", "Request")
    require(isinstance(request["case_number"], str) and
            re.fullmatch(r"SYN-CASE-\d{4}", request["case_number"]), "Synthetic case number required")
    require(request["kind"] in ("citizen_service_request", "benefits_application"),
            "Unknown request kind")
    text(request["question"], "Question", 1500)
    citizen = request["citizen"]
    if phase == "input":
        fields(citizen, "name email address", "Citizen")
        for key, value in citizen.items():
            text(value, "Citizen " + key, 200)
            require(len(value) >= 3, "Citizen fields must have at least three characters")
    else:
        require(citizen is None, "Citizen PII must be removed before support")
        public_text(request["question"], "Question")
    application = envelope["benefits_application"]
    fields(application, "application_id program status", "Benefits application")
    require(isinstance(application["application_id"], str) and
            re.fullmatch(r"SYN-APP-\d{4}", application["application_id"]),
            "Synthetic application ID required")
    public_text(application["program"], "Program", citizen)
    require(application["status"] in ("draft", "submitted", "under_review"),
            "Application status must not imply an eligibility decision")
    for collection in ("policies", "services"):
        require(isinstance(envelope[collection], list) and len(envelope[collection]) <= 50,
                collection + " must be a list with at most 50 entries")
    policy_ids = set()
    for policy in envelope["policies"]:
        fields(policy, "id title text plain_language_summary", "Policy")
        text(policy["id"], "Policy ID", 60)
        require(re.fullmatch(r"POL-[A-Z0-9-]+", policy["id"]), "Invalid policy ID")
        require(policy["id"] not in policy_ids, "Duplicate policy ID")
        policy_ids.add(policy["id"])
        public_text(policy["title"], "Policy title", citizen)
        public_text(policy["text"], "Policy text", citizen)
        plain_text(policy["plain_language_summary"], "Policy summary", citizen)
        require(policy["plain_language_summary"] in policy["text"],
                "Plain-language summary must be quoted from policy text")
    service_ids = set()
    for service in envelope["services"]:
        fields(service, "id title description policy_ids", "Service")
        text(service["id"], "Service ID", 60)
        require(re.fullmatch(r"SVC-[A-Z0-9-]+", service["id"]), "Invalid service ID")
        require(service["id"] not in service_ids, "Duplicate service ID")
        service_ids.add(service["id"])
        plain_text(service["title"], "Service title", citizen)
        plain_text(service["description"], "Service description", citizen)
        links = service["policy_ids"]
        require(isinstance(links, list) and all(isinstance(link, str) for link in links),
                "Service policy_ids must be a string list")
        require(len(links) == len(set(links)) and set(links) <= policy_ids,
                "Service policy references must be unique and known")
    if phase == "input":
        require(envelope["support"] is None and envelope["search"] is None,
                "Input cannot prepopulate pipeline outputs")
        return
    require(envelope["support"] == expected_support(envelope),
            "Support output failed grounding or handoff validation")
    plain_text(envelope["support"]["answer"], "Support answer")
    if phase == "supported":
        require(envelope["search"] is None, "Search must run after support")
    else:
        require(envelope["search"] == expected_search(envelope),
                "Search output failed ranking or handoff validation")


def support_stage(envelope):
    validate(envelope, "input")
    result = copy.deepcopy(envelope)
    result["request"]["question"] = redact(result["request"]["question"], result["request"]["citizen"])
    result["request"]["citizen"] = None
    result["status"] = "supported"
    result["support"] = expected_support(result)
    validate(result, "supported")
    return result


def search_stage(envelope):
    validate(envelope, "supported")
    result = copy.deepcopy(envelope)
    result["search"] = expected_search(result)
    result["status"] = "ok"
    validate(result, "ok")
    return result


def run_pipeline(envelope):
    return search_stage(support_stage(envelope))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], "r", encoding="utf-8") as stream:
            raw = stream.read(1_000_001)
        require(len(raw) <= 1_000_000, "Input file exceeds size limit")
        envelope = json.loads(raw, object_pairs_hook=unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValidationError("Invalid JSON number")))
        result = run_pipeline(envelope)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        # Never echo raw input, a path, or a parser exception into public output.
        print(json.dumps({"status": "error", "message": "Input could not be read or validated. Check the form and policy fields."}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
