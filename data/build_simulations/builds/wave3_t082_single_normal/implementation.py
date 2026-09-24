"""Deterministic passage research; Python standard library only."""

import json
import re
import sys
from pathlib import Path


SCHEMA_VERSION = "research.v1"
MAX_INPUT_BYTES = 2_000_000


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def tokens(text):
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def validate_input(value):
    """One validation boundary shared by the CLI and Python API."""
    require(isinstance(value, dict), "Input must be an object")
    required = {"schema_version", "dataset_label", "query", "sources"}
    require(required <= value.keys(), "Missing required input fields")
    require(value.keys() <= required | {"top_k"}, "Unknown input fields")
    require(value["schema_version"] == SCHEMA_VERSION, "Unsupported schema_version")
    for key, limit in (("dataset_label", 200), ("query", 2000)):
        require(isinstance(value[key], str), key + " must be a string")
        require(0 < len(value[key].strip()) and len(value[key]) <= limit,
                key + " must be nonblank and within its length limit")
    require(bool(tokens(value["query"])), "query must contain a word or number")
    top_k = value.get("top_k", 5)
    require(type(top_k) is int and 1 <= top_k <= 50, "top_k must be an integer from 1 to 50")
    sources = value["sources"]
    require(isinstance(sources, list) and 1 <= len(sources) <= 100,
            "sources must contain 1 to 100 objects")
    identifiers = set()
    total_length = 0
    for source in sources:
        require(isinstance(source, dict), "Each source must be an object")
        require(source.keys() == {"id", "title", "text"}, "Source fields must be id, title, text")
        for key, limit in (("id", 200), ("title", 500), ("text", 100_000)):
            require(isinstance(source[key], str), "Source " + key + " must be a string")
            require(len(source[key]) <= limit, "Source " + key + " is too long")
            if key != "text":
                require(bool(source[key].strip()), "Source " + key + " must be nonblank")
        require(source["id"] not in identifiers, "Duplicate source id")
        identifiers.add(source["id"])
        total_length += len(source["text"])
    require(total_length <= 1_000_000, "Combined source text is too long")
    return value


def passages(text):
    """Each nonblank physical line is a passage; offsets index original Python text."""
    for match in re.finditer(r"[^\r\n]+", text):
        raw = match.group()
        quote = raw.strip()
        if quote:
            start = match.start() + len(raw) - len(raw.lstrip())
            yield start, start + len(quote), quote


def research(request):
    request = validate_input(request)
    query_terms = tokens(request["query"])
    candidates = []
    passage_count = 0
    for source_index, source in enumerate(request["sources"]):
        for start, end, quote in passages(source["text"]):
            passage_count += 1
            matched_terms = sorted(query_terms & tokens(quote))
            if matched_terms:
                candidates.append({
                    "source_index": source_index,
                    "matched_terms": matched_terms,
                    "citation": {
                        "source_id": source["id"],
                        "source_title": source["title"],
                        "start": start,
                        "end": end,
                        "quote": quote,
                    },
                })
    candidates.sort(key=lambda item: (
        -len(item["matched_terms"]), item["source_index"], item["citation"]["start"]))
    findings = []
    for rank, candidate in enumerate(candidates[:request.get("top_k", 5)], start=1):
        findings.append({
            "rank": rank,
            "text": candidate["citation"]["quote"],
            "score": round(len(candidate["matched_terms"]) / len(query_terms), 6),
            "matched_terms": candidate["matched_terms"],
            "citation": candidate["citation"],
        })
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "dataset_label": request["dataset_label"],
        "query": request["query"],
        "findings": findings,
        "retrieval": {
            "source_count": len(request["sources"]),
            "passage_count": passage_count,
            "matched_passage_count": len(candidates),
            "returned_count": len(findings),
            "no_matches": not candidates,
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Nonstandard JSON constant: " + value)


def load_request(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    require(len(raw) <= MAX_INPUT_BYTES, "Input file exceeds 2000000 bytes")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py input.json")
        result = research(load_request(args[0]))
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "schema_version": SCHEMA_VERSION,
                          "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
