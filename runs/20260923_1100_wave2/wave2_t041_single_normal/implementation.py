"""Deterministic local passage research. Offsets are Python Unicode character indices."""

import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def validate(value, kind):
    """The shared validation boundary for requests and extracted evidence."""
    if kind == "request":
        if not isinstance(value, dict):
            raise ValidationError("Input must be an object")
        if set(value) - {"query", "sources", "max_findings"}:
            raise ValidationError("Unknown input fields")
        if not isinstance(value.get("query"), str) or not value["query"].strip():
            raise ValidationError("query must be a nonempty string")
        limit = value.get("max_findings", 5)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValidationError("max_findings must be an integer from 1 to 50")
        sources = value.get("sources")
        if not isinstance(sources, list) or len(sources) > 100:
            raise ValidationError("sources must be a list with at most 100 items")
        seen = set()
        for source in sources:
            if not isinstance(source, dict) or set(source) != {"id", "title", "text"}:
                raise ValidationError("Each source needs exactly id, title, text")
            for key in ("id", "title", "text"):
                if not isinstance(source[key], str):
                    raise ValidationError("Source fields must be strings")
            if not source["id"].strip() or not source["title"].strip():
                raise ValidationError("Source id and title must not be blank")
            if source["id"] in seen:
                raise ValidationError("Source ids must be unique")
            seen.add(source["id"])
            if len(source["text"]) > 100000:
                raise ValidationError("Source text exceeds 100000 characters")
        if len(value["query"]) > 10000:
            raise ValidationError("query exceeds 10000 characters")
        return {"query": value["query"], "sources": sources, "max_findings": limit}
    if kind == "evidence":
        finding, source = value
        citation = finding["citation"]
        start, end = citation["start"], citation["end"]
        if not (type(start) is int and type(end) is int
                and 0 <= start < end <= len(source["text"])
                and source["text"][start:end] == finding["text"]
                and citation["source_id"] == source["id"]
                and citation["source_title"] == source["title"]
                and math.isfinite(finding["score"])):
            raise ValidationError("Extracted citation does not match its source")
        return finding
    raise ValidationError("Unknown validation kind")


def terms(text):
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def passages(text):
    """Split at sentence punctuation followed by whitespace, or at line breaks.

    This intentionally simple splitter is not a linguistic sentence parser.
    Quotes remain exact substrings, even with abbreviations or unusual punctuation.
    """
    cursor = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+|[\r\n]+", text):
        yield from trimmed_span(text, cursor, boundary.start())
        cursor = boundary.end()
    yield from trimmed_span(text, cursor, len(text))


def trimmed_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        yield start, end


def research(payload):
    request = validate(payload, "request")
    query_terms = terms(request["query"])
    candidates = []
    for source_index, source in enumerate(request["sources"]):
        for start, end in passages(source["text"]):
            text = source["text"][start:end]
            matches = sorted(query_terms & terms(text))
            if not matches:
                continue
            finding = {
                "text": text,
                "score": round(len(matches) / len(query_terms), 6),
                "matched_terms": matches,
                "citation": {
                    "source_id": source["id"],
                    "source_title": source["title"],
                    "start": start,
                    "end": end,
                },
            }
            validate((finding, source), "evidence")
            candidates.append((-len(matches), source_index, start, finding))
    candidates.sort(key=lambda item: item[:3])
    findings = [item[3] for item in candidates[:request["max_findings"]]]
    return {
        "status": "ok",
        "query": request["query"],
        "findings": findings,
        "summary": {
            "sources_examined": len(request["sources"]),
            "matching_passages": len(candidates),
            "findings_returned": len(findings),
            "no_matches": not findings,
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Invalid JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("r", encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        result = research(payload)
    except (OSError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
