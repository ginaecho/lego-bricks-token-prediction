"""Synthetic, offline extraction -> research reference pipeline.

Input envelope: version=1, fixture_label, documents[{id,text}],
fields[{name,label,required}], queries[{id,text,fields}], top_k.
Output preserves that envelope and adds status, extraction and findings.
Labels match whole line prefixes (case-insensitive) before a colon. Every
nonempty matching value is retained, including repeated labels. All spans
are half-open Python Unicode character offsets into the original document.
Research ranks nonempty lines by 2*query-term overlap + field-term overlap.
Field terms come exclusively from validated extracted values, globally across
documents. Ties follow document order then source offset. Findings are quotes,
not generated summaries, and lexical relevance does not establish truth.
"""

import copy
import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(expected), f"{path} has missing or unknown keys")


def text(value, path, limit=1000, blank=False):
    require(isinstance(value, str), f"{path} must be a string")
    require(len(value) <= limit, f"{path} is too long")
    require(blank or bool(value.strip()), f"{path} must not be blank")


def sequence(value, path, maximum, allow_empty=False):
    require(isinstance(value, list), f"{path} must be an array")
    require((allow_empty or len(value) > 0) and len(value) <= maximum,
            f"{path} has invalid length")


def unique(values, path):
    require(len(values) == len(set(values)), f"{path} contains duplicates")


def same(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def lines(document):
    offset = 0
    for raw in document["text"].splitlines(keepends=True):
        stripped = raw.strip()
        if stripped:
            start = offset + len(raw) - len(raw.lstrip())
            yield stripped, start, start + len(stripped)
        offset += len(raw)


def citation(document, start, end):
    return {"document_id": document["id"], "start": start, "end": end,
            "quote": document["text"][start:end]}


def tokens(value):
    return set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _extract(state):
    results = []
    for document in state["documents"]:
        matches = {field["name"]: [] for field in state["fields"]}
        for line, start, _ in lines(document):
            label, separator, rest = line.partition(":")
            if not separator or not rest.strip():
                continue
            for field in state["fields"]:
                if label.strip().casefold() == field["label"].casefold():
                    value = rest.strip()
                    value_start = start + len(label) + 1 + len(rest) - len(rest.lstrip())
                    matches[field["name"]].append({
                        "value": value,
                        "source": citation(document, value_start, value_start + len(value))
                    })
        missing = [{"name": field["name"], "required": field["required"]}
                   for field in state["fields"] if not matches[field["name"]]]
        results.append({"document_id": document["id"], "fields": matches,
                        "missing_fields": missing,
                        "complete": not any(item["required"] for item in missing)})
    return results


def _research(state):
    results = []
    for query in state["queries"]:
        evidence = []
        for record in state["extraction"]:
            for field in query["fields"]:
                for match in record["fields"][field]:
                    evidence.append({"field": field, "value": match["value"],
                                     "source": copy.deepcopy(match["source"])})
        expansion = set()
        for item in evidence:
            expansion.update(tokens(item["value"]))
        query_terms = tokens(query["text"])
        ranked = []
        for order, document in enumerate(state["documents"]):
            for passage, start, end in lines(document):
                terms = tokens(passage)
                direct = sorted(terms & query_terms)
                expanded = sorted(terms & expansion)
                score = 2 * len(direct) + len(expanded)
                if score:
                    ranked.append((score, order, start, {
                        "text": passage, "score": score,
                        "query_terms": direct, "field_terms": expanded,
                        "source": citation(document, start, end)
                    }))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        hits = [item[3] for item in ranked[:state["top_k"]]]
        results.append({"query_id": query["id"], "expanded_from": evidence,
                        "status": "found" if hits else "no_evidence",
                        "passages": hits})
    return results


def validate(state, phase="input"):
    """Single validation boundary for input, extraction handoff, and final output."""
    require(phase in ("input", "extracted", "researched"), "unknown validation phase")
    expected = {"version", "fixture_label", "documents", "fields", "queries", "top_k"}
    if phase != "input":
        expected |= {"extraction"}
    if phase == "researched":
        expected |= {"status", "findings"}
    keys(state, expected, "envelope")
    require(type(state["version"]) is int and state["version"] == 1,
            "version must be integer 1")
    text(state["fixture_label"], "fixture_label")
    require(type(state["top_k"]) is int and 1 <= state["top_k"] <= 20,
            "top_k must be an integer from 1 to 20")
    sequence(state["documents"], "documents", 100)
    for document in state["documents"]:
        keys(document, {"id", "text"}, "document")
        text(document["id"], "document.id", 100)
        text(document["text"], "document.text", 100000, blank=True)
    unique([d["id"] for d in state["documents"]], "document ids")
    require(sum(len(d["text"]) for d in state["documents"]) <= 1000000,
            "combined document text is too long")
    sequence(state["fields"], "fields", 50)
    for field in state["fields"]:
        keys(field, {"name", "label", "required"}, "field")
        text(field["name"], "field.name", 100)
        text(field["label"], "field.label", 100)
        require(field["label"] == field["label"].strip()
                and ":" not in field["label"]
                and len(field["label"].splitlines()) == 1,
                "field.label must be a trimmed single-line label without a colon")
        require(type(field["required"]) is bool, "field.required must be boolean")
    names = [field["name"] for field in state["fields"]]
    unique(names, "field names")
    unique([field["label"].casefold() for field in state["fields"]], "field labels")
    sequence(state["queries"], "queries", 30)
    for query in state["queries"]:
        keys(query, {"id", "text", "fields"}, "query")
        text(query["id"], "query.id", 100)
        text(query["text"], "query.text")
        require(bool(tokens(query["text"])), "query.text must contain a word")
        sequence(query["fields"], "query.fields", 50, allow_empty=True)
        for name in query["fields"]:
            text(name, "query field", 100)
            require(name in names, "query references an unknown field")
        unique(query["fields"], "query.fields")
    unique([query["id"] for query in state["queries"]], "query ids")
    if phase != "input":
        require(same(state["extraction"], _extract(state)),
                "extraction handoff differs from canonical source-grounded extraction")
    if phase == "researched":
        require(state["status"] == "ok", "result status must be ok")
        require(same(state["findings"], _research(state)),
                "findings differ from canonical cited retrieval")
    return state


def extract(state):
    validate(state)
    result = copy.deepcopy(state)
    result["extraction"] = _extract(state)
    return validate(result, "extracted")


def research(extracted):
    validate(extracted, "extracted")
    result = copy.deepcopy(extracted)
    result["findings"] = _research(result)
    result["status"] = "ok"
    return validate(result, "researched")


def run_pipeline(state):
    return research(extract(state))


def reject_constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle, parse_constant=reject_constant,
                             object_pairs_hook=object_pairs)
        result = run_pipeline(data)
    except (ValidationError, OSError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
