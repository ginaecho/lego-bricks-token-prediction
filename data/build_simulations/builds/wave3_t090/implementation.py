"""Bounded deterministic document evidence review; not a certification tool."""

import json
import re
import sys
from pathlib import Path


SCHEMA_VERSION = "1.0"
MAX_INPUT_BYTES = 2_000_000
DISCLAIMER = (
    "Literal evidence coverage only; not a determination of correctness, "
    "compliance, or certification. Human review remains necessary."
)


class ValidationError(ValueError):
    def __init__(self, path, message):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def require(condition, path, message):
    if not condition:
        raise ValidationError(path, message)


def object_fields(value, required, optional, path):
    require(isinstance(value, dict), path, "must be an object")
    require(set(required) <= value.keys(), path, "missing required fields")
    require(value.keys() <= set(required) | set(optional), path, "unknown fields")


def text(value, path, limit):
    require(isinstance(value, str), path, "must be a string")
    require(bool(value.strip()), path, "must not be blank")
    require(len(value) <= limit, path, f"must be at most {limit} characters")
    require(not any(0xD800 <= ord(char) <= 0xDFFF for char in value),
            path, "must not contain unpaired Unicode surrogates")


def array(value, path, minimum, maximum):
    require(isinstance(value, list), path, "must be an array")
    require(minimum <= len(value) <= maximum, path,
            f"must contain {minimum} to {maximum} items")


def validate_input(payload):
    """The shared validation boundary for both Python and CLI callers."""
    object_fields(payload, ("schema_version", "dataset", "requirements", "documents"),
                  (), "$")
    require(payload["schema_version"] == SCHEMA_VERSION, "$.schema_version",
            "must equal '1.0'")
    dataset = payload["dataset"]
    object_fields(dataset, ("label", "synthetic"), (), "$.dataset")
    text(dataset["label"], "$.dataset.label", 200)
    require(type(dataset["synthetic"]) is bool, "$.dataset.synthetic",
            "must be a boolean")
    array(payload["requirements"], "$.requirements", 1, 100)
    array(payload["documents"], "$.documents", 0, 100)
    for collection in ("requirements", "documents"):
        seen = set()
        for index, item in enumerate(payload[collection]):
            path = f"$.{collection}[{index}]"
            if collection == "requirements":
                object_fields(item, ("id", "text", "evidence_terms"),
                              ("min_sources",), path)
            else:
                object_fields(item, ("id", "title", "content"), (), path)
            text(item["id"], path + ".id", 100)
            require(item["id"] not in seen, path + ".id", "duplicate ID")
            seen.add(item["id"])
            if collection == "documents":
                text(item["title"], path + ".title", 300)
                require(isinstance(item["content"], str), path + ".content",
                        "must be a string")
                # Empty documents are valid and cannot support any requirement.
                if item["content"]:
                    text(item["content"], path + ".content", 50_000)
                continue
            text(item["text"], path + ".text", 2000)
            terms = item["evidence_terms"]
            array(terms, path + ".evidence_terms", 1, 30)
            seen_terms = set()
            for term_index, term in enumerate(terms):
                term_path = f"{path}.evidence_terms[{term_index}]"
                text(term, term_path, 200)
                require(term == term.strip(), term_path,
                        "leading or trailing whitespace is not allowed")
                require(term.casefold() not in seen_terms, term_path,
                        "duplicate case-insensitive term")
                seen_terms.add(term.casefold())
            count = item.get("min_sources", 1)
            require(type(count) is int and 1 <= count <= 100,
                    path + ".min_sources", "must be an integer from 1 to 100")
    require(sum(len(doc["content"]) for doc in payload["documents"]) <= 500_000,
            "$.documents", "total content must be at most 500000 characters")
    return payload


def review(payload):
    validate_input(payload)
    findings = []
    for requirement in payload["requirements"]:
        terms = requirement["evidence_terms"]
        evidence = []
        supporting_ids = []
        for document in payload["documents"]:
            matches = []
            missing = []
            for term in terms:
                match = re.search(re.escape(term), document["content"], re.IGNORECASE)
                if match is None:
                    missing.append(term)
                else:
                    matches.append({
                        "term": term, "quote": match.group(0),
                        "start": match.start(), "end": match.end(),
                    })
            complete = not missing
            if complete:
                supporting_ids.append(document["id"])
            evidence.append({
                "document_id": document["id"],
                "document_title": document["title"],
                "complete": complete, "matches": matches, "missing_terms": missing,
            })
        minimum = requirement.get("min_sources", 1)
        deficit = max(0, minimum - len(supporting_ids))
        findings.append({
            "requirement_id": requirement["id"],
            "requirement_text": requirement["text"],
            "status": "gap" if deficit else "supported",
            "required_sources": minimum,
            "supporting_document_ids": supporting_ids,
            "evidence": evidence,
            "gap": None if not deficit else {
                "code": "insufficient_complete_sources",
                "additional_sources_needed": deficit,
                "message": (
                    f"Need {deficit} additional document(s) containing every "
                    "required literal term; inspect per-document missing_terms."
                ),
            },
        })
    supported = sum(item["status"] == "supported" for item in findings)
    return {
        "schema_version": SCHEMA_VERSION, "status": "ok",
        "dataset": dict(payload["dataset"]),
        "review_method": "case_insensitive_literal_substring",
        "disclaimer": DISCLAIMER,
        "summary": {"requirements": len(findings), "supported": supported,
                    "gaps": len(findings) - supported},
        "findings": findings,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "$", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("$", f"non-finite JSON constant: {value}")


def load_input(filename):
    with Path(filename).open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    require(len(data) <= MAX_INPUT_BYTES, "$", "input exceeds 2000000 bytes")
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        require(len(arguments) == 1, "$", "usage: implementation.py INPUT.json")
        result = review(load_input(arguments[0]))
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as error:
        result = {
            "schema_version": SCHEMA_VERSION, "status": "error",
            "errors": [{"path": getattr(error, "path", "$"),
                        "message": getattr(error, "message", str(error))}],
        }
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
