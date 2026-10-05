"""Synthetic-only public-service feedback analysis; Python standard library."""

import json
import re
import sys
from pathlib import Path


BUILD_ID = "wave3_t138_single_feedback__ind-public_sector"
ENTITY_TYPES = ("citizen_service_request", "benefits_application", "policy_document")
THEMES = {
    "access": {
        "label": "Getting help",
        "terms": ("accessible", "accessibility", "screen reader", "translation", "language"),
    },
    "clarity": {
        "label": "Clear information",
        "terms": ("unclear", "confusing", "plain language", "explain", "instructions"),
    },
    "delay": {
        "label": "Waiting for a response",
        "terms": ("wait", "waiting", "delay", "delayed", "slow"),
    },
    "positive": {
        "label": "Helpful service",
        "terms": ("helpful", "easy", "clear", "quick"),
    },
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def validate(data):
    """The single input boundary used by the API and CLI."""
    require(isinstance(data, dict), "Input must be a JSON object.")
    require(set(data) == {"schema_version", "synthetic", "records"},
            "Input must contain only schema_version, synthetic, and records.")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1.")
    require(data["synthetic"] is True, "Only clearly labeled synthetic data is accepted.")
    records = data["records"]
    require(isinstance(records, list) and 1 <= len(records) <= 200,
            "Provide between 1 and 200 records.")
    for record in records:
        require(isinstance(record, dict), "Each record must be an object.")
        require({"entity_type", "text"} <= set(record) and
                set(record) <= {"entity_type", "text", "citizen", "case_number"},
                "A record has missing or unsupported fields.")
        require(isinstance(record["entity_type"], str) and
                record["entity_type"] in ENTITY_TYPES, "Unsupported entity_type.")
        text = record["text"]
        require(isinstance(text, str) and 1 <= len(text) <= 10000 and text.strip(),
                "Record text must contain 1 to 10000 characters and not be blank.")
        require(not any(ord(c) < 32 and c not in "\n\r\t" for c in text),
                "Record text contains unsupported control characters.")
        if "case_number" in record:
            require(isinstance(record["case_number"], str) and
                    1 <= len(record["case_number"].strip()) <= 100,
                    "case_number must be nonblank text of at most 100 characters.")
        if "citizen" in record:
            person = record["citizen"]
            require(isinstance(person, dict) and set(person) == {"name", "address"},
                    "citizen must contain only name and address.")
            for value in person.values():
                require(isinstance(value, str) and 1 <= len(value.strip()) <= 200,
                        "Citizen values must be nonblank text of at most 200 characters.")
    return records


def private_spans(text, secrets):
    spans = []
    for secret in secrets:
        spans.extend((m.start(), m.end()) for m in
                     re.finditer(re.escape(secret), text, flags=re.IGNORECASE))
    for pattern in (
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)",
    ):
        spans.extend((m.start(), m.end()) for m in
                     re.finditer(pattern, text, flags=re.IGNORECASE))
    return spans


def overlaps(start, end, spans):
    return any(start < right and end > left for left, right in spans)


def normalized(text, spans):
    # Mask in place to keep source coordinates for the evidence step.
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return " ".join(re.findall(r"\w+", "".join(chars).casefold()))


def analyze(data):
    records = validate(data)
    secrets = []
    for record in records:
        if "case_number" in record:
            secrets.append(record["case_number"])
        if "citizen" in record:
            for value in record["citizen"].values():
                secrets.append(value)
            # Names are also masked when only one name component appears.
            secrets.extend(record["citizen"]["name"].split())

    groups = []
    seen = {}
    for index, record in enumerate(records, 1):
        ref = "record_{:04d}".format(index)
        spans = private_spans(record["text"], secrets)
        key = normalized(record["text"], spans)
        # Empty masked text must not merge unrelated private-only records.
        key = key if key else ("private-only", index)
        if key in seen:
            groups[seen[key]]["record_refs"].append(ref)
            continue
        seen[key] = len(groups)
        groups.append({"record_refs": [ref], "text": record["text"], "spans": spans})

    theme_results = []
    matched_groups = set()
    for theme_id, rule in THEMES.items():
        evidence = []
        for group_index, group in enumerate(groups):
            pattern = r"(?<!\w)(?:" + "|".join(
                re.escape(term) for term in sorted(rule["terms"], key=len, reverse=True)
            ) + r")(?!\w)"
            matches = [m for m in re.finditer(pattern, group["text"], re.IGNORECASE)
                       if not overlaps(m.start(), m.end(), group["spans"])]
            if matches:
                match = matches[0]
                evidence.append({
                    "record_ref": group["record_refs"][0],
                    "excerpt": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "reason": "The text contains a listed service-language term.",
                })
                matched_groups.add(group_index)
        if evidence:
            theme_results.append({
                "theme_id": theme_id,
                "label": rule["label"],
                "unique_feedback_count": len(evidence),
                "supporting_excerpts": evidence,
            })

    return {
        "status": "ok",
        "schema_version": 1,
        "synthetic": True,
        "summary": {
            "input_records": len(records),
            "unique_feedback": len(groups),
            "duplicate_records": len(records) - len(groups),
            "unclassified_unique_feedback": len(groups) - len(matched_groups),
        },
        "sources": [
            {"record_ref": "record_{:04d}".format(i),
             "entity_type": record["entity_type"]}
            for i, record in enumerate(records, 1)
        ],
        "deduplication": [
            {"canonical_record_ref": group["record_refs"][0],
             "record_refs": group["record_refs"],
             "reason": "Same words after private values, case, spacing, and punctuation are removed."}
            for group in groups
        ],
        "themes": theme_results,
        "review": {
            "method": "Count each unique feedback item once per matching theme.",
            "rules": {key: list(value["terms"]) for key, value in THEMES.items()},
            "trace": "Record numbers follow input order. Excerpt positions are zero-based; end is excluded.",
            "privacy": "Only approved service-language excerpts leave this tool. Names, addresses, case numbers, and full text are not output.",
            "limits": "Synthetic demonstration, not compliance certification. A term match is not a sentiment or benefits decision. A person should review results.",
            "deduplication_limit": "Repeated text can come from different people. Near-duplicates are not merged. Keep source input in controlled storage for review.",
        },
    }


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON contains a repeated field.")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py input.json")
        path = Path(args[0])
        require(path.stat().st_size <= 4_000_000, "Input file exceeds the size limit.")
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=reject_duplicate_keys,
                             parse_constant=lambda _: (_ for _ in ()).throw(
                                 ValidationError("JSON numbers must be finite.")))
        result = analyze(data)
    except ValidationError as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 2
    except (OSError, UnicodeError, ValueError, RecursionError):
        print(json.dumps({"status": "error", "message": "Cannot read a valid input JSON file."}))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
