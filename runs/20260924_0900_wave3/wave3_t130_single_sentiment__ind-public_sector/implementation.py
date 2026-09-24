"""Synthetic public-sector sentiment triage. Standard library; no providers.

Input is government-form JSON. Each record's text may be a service request,
benefits application narrative, or policy/regulation text. Scores are lexical,
not legal findings, eligibility decisions, or compliance certifications.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


POSITIVE = frozenset({"clear", "helpful", "resolved", "thanks", "accessible", "good"})
NEGATIVE = frozenset({"delay", "delayed", "denied", "confusing", "inaccessible",
                      "broken", "unsafe", "urgent", "eviction", "homeless"})
CRITICAL = frozenset({"unsafe", "eviction", "homeless"})
URGENT = frozenset({"urgent", "denied", "inaccessible"})
ENTITIES = {"citizen_service_request", "benefits_application", "policy_document"}
LEVELS = {"routine": 1, "urgent": 2, "critical": 3}
PII_PATTERNS = (
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}",
    r"\b\d{3}[- ]\d{2}[- ]\d{4}\b",
    r"\b(?:\+?\d[\d ().-]{6,}\d)\b",
    r"\b\d+\s+[a-z]+(?:\s+[a-z]+){0,3}\s+(?:street|st|road|rd|avenue|ave|lane|ln)\b",
    r"\b(?:my name is|date of birth|born on|passport|social security)\b",
)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys),
            label + " has missing or unsupported fields.")


def validate(payload):
    """Shared boundary for library and CLI; deliberately synthetic-only."""
    exact_keys(payload, {"schema_version", "synthetic", "records"}, "Input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version must be 1.")
    require(payload["synthetic"] is True, "Only clearly labeled synthetic data is allowed.")
    records = payload["records"]
    require(type(records) is list and 1 <= len(records) <= 100,
            "Provide between 1 and 100 records.")
    seen = set()
    for record in records:
        exact_keys(record, {"case_number", "entity", "persona", "address", "text",
                            "reported_severity"}, "Record")
        require(all(type(value) is str for value in record.values()),
                "All record fields must be strings.")
        case = record["case_number"]
        require(re.fullmatch(r"SYN-\d{4}", case) is not None,
                "Use invented case numbers in SYN-0000 format.")
        require(case not in seen, "Case numbers must be unique.")
        seen.add(case)
        require(record["entity"] in ENTITIES, "Unsupported entity.")
        require(re.fullmatch(r"Synthetic resident [A-Z]", record["persona"]) is not None,
                "Use a fabricated persona such as Synthetic resident A.")
        require(record["address"] == "FICTIONAL ADDRESS - NOT DELIVERABLE",
                "Use the non-traceable fixture address.")
        require(record["reported_severity"] in LEVELS, "Unsupported reported severity.")
        text = record["text"]
        require(1 <= len(text) <= 5000 and bool(text.strip()),
                "Text must contain 1 to 5000 characters and not be blank.")
        require(not any(re.search(pattern, text, re.I) for pattern in PII_PATTERNS),
                "Text contains a possible personal identifier. Remove it before retrying.")
        require(not any(ord(c) < 32 and c not in "\n\r\t" for c in text),
                "Text contains unsupported control characters.")
    return records


def analyze(payload):
    records = validate(payload)
    results = []
    for record in records:
        words = re.findall(r"[a-z]+", record["text"].lower())
        positive = sorted(word for word in words if word in POSITIVE)
        negative = sorted(word for word in words if word in NEGATIVE)
        score = len(positive) - len(negative)
        sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        critical = sorted(set(words) & CRITICAL)
        urgent = sorted(set(words) & URGENT)
        inferred = "critical" if critical else "urgent" if urgent else "routine"
        reported = record["reported_severity"]
        severity = max((reported, inferred), key=LEVELS.get)
        # Reserve nonoverlapping bands: severe cases always precede lower bands.
        priority = LEVELS[severity] * 100 + min(99, max(0, -score))
        results.append({
            "case_number": record["case_number"],
            "entity": record["entity"],
            "sentiment": {"label": sentiment, "score": score,
                          "positive_matches": positive, "negative_matches": negative},
            "severity": severity,
            "priority_score": priority,
            "explanation": {
                "sentiment_rule": "Score is positive word count minus negative word count.",
                "severity_rule": "Use the higher of reported severity and keyword severity.",
                "reported_severity": reported,
                "keyword_severity": inferred,
                "severity_keywords": critical if critical else urgent,
                "priority_rule": "Severity points plus negative score size, capped at 99.",
            },
            "next_step": "A staff member should review this item. No benefit decision was made.",
        })
    results.sort(key=lambda item: (-item["priority_score"], item["case_number"]))
    for rank, item in enumerate(results, 1):
        item["review_rank"] = rank
    return {
        "status": "ok", "schema_version": 1, "synthetic": True,
        "method": "lexical-triage-v1",
        "privacy": "Names, addresses, and source text are not included in results.",
        "limitations": [
            "Synthetic demonstration only. This is not a compliance certification.",
            "Personal identifier checks are limited. Do not use real citizen information.",
            "Word counts do not understand negation, sarcasm, or legal meaning.",
            "The ranking supports staff review, not eligibility or policy decisions.",
            "Plain-language JSON is not a Section 508 conformance assessment.",
        ],
        "results": results,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON must not contain duplicate fields.")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 1_000_000, "Input file exceeds the size limit.")
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        output = analyze(payload)
    except ValidationError as exc:
        output = {"status": "error", "message": str(exc)}
    except (OSError, ValueError, RecursionError):
        output = {"status": "error", "message": "Cannot read a valid UTF-8 JSON input file."}
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0 if output["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
