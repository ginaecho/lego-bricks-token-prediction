"""Bounded, source-grounded extraction using only the Python standard library."""

import datetime
import json
import math
import multiprocessing
import re
import sys
from pathlib import Path


MAX_INPUT_BYTES = 1_048_576
MAX_DOCUMENTS = 16
MAX_FIELDS = 32
MAX_DOCUMENT_CHARS = 32_768
MAX_TOTAL_CHARS = 131_072
MAX_PATTERN_CHARS = 512
MAX_VALUE_CHARS = 1_024
MAX_CANDIDATES = 2_048
REGEX_TIMEOUT_SECONDS = 2.0
TYPES = {"string", "integer", "number", "boolean", "date"}


class ExtractionError(ValueError):
    """Invalid input or an exceeded extraction resource limit."""


def _keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ExtractionError("Expected an object")
    if set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ExtractionError("Unexpected or missing object keys")


def _identifier(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ExtractionError("IDs must be nonempty strings of at most 64 characters")


def _validate(payload):
    _keys(payload, ("schema", "documents"), ("fixture_label",))
    if "fixture_label" in payload and not isinstance(payload["fixture_label"], str):
        raise ExtractionError("fixture_label must be a string")
    _keys(payload["schema"], ("fields",))
    fields = payload["schema"]["fields"]
    documents = payload["documents"]
    if not isinstance(fields, list) or not 1 <= len(fields) <= MAX_FIELDS:
        raise ExtractionError("schema.fields must contain 1..32 fields")
    if not isinstance(documents, list) or not 1 <= len(documents) <= MAX_DOCUMENTS:
        raise ExtractionError("documents must contain 1..16 documents")
    ids = set()
    for field in fields:
        _keys(field, ("id", "type", "required", "pattern"))
        _identifier(field["id"])
        if field["id"] in ids:
            raise ExtractionError("Duplicate field ID")
        ids.add(field["id"])
        if not isinstance(field["type"], str) or field["type"] not in TYPES:
            raise ExtractionError("Unsupported field type")
        if type(field["required"]) is not bool:
            raise ExtractionError("required must be a boolean")
        if not isinstance(field["pattern"], str) or not 1 <= len(field["pattern"]) <= MAX_PATTERN_CHARS:
            raise ExtractionError("pattern must contain 1..512 characters")
    ids = set()
    total = 0
    for document in documents:
        _keys(document, ("id", "text"))
        _identifier(document["id"])
        if document["id"] in ids:
            raise ExtractionError("Duplicate document ID")
        ids.add(document["id"])
        if not isinstance(document["text"], str) or len(document["text"]) > MAX_DOCUMENT_CHARS:
            raise ExtractionError("Document text must be a string of at most 32768 characters")
        total += len(document["text"])
    if total > MAX_TOTAL_CHARS:
        raise ExtractionError("Total document text exceeds 131072 characters")
    return fields, documents


def _regex_worker(connection, fields, documents):
    try:
        patterns = []
        for field in fields:
            pattern = re.compile(field["pattern"])
            if pattern.groups != 1 or pattern.groupindex != {"value": 1}:
                raise ExtractionError("Each pattern must have exactly one capture: (?P<value>...)")
            patterns.append((field["id"], pattern))
        candidates = []
        for document in documents:
            for field_id, pattern in patterns:
                for match in pattern.finditer(document["text"]):
                    start, end = match.span("value")
                    if start < 0:
                        continue
                    if len(candidates) >= MAX_CANDIDATES:
                        raise ExtractionError("Regex candidate limit exceeded")
                    candidates.append((field_id, document["id"], start, end))
        connection.send({"candidates": candidates})
    except Exception as exc:
        connection.send({"error": f"Regex extraction failed: {type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _regex_candidates(fields, documents):
    # A separate process bounds both regex compilation and catastrophic backtracking.
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_regex_worker, args=(send, fields, documents))
    started = False
    try:
        process.start()
        started = True
        send.close()
        if not receive.poll(REGEX_TIMEOUT_SECONDS):
            raise ExtractionError("Regex execution exceeded its time limit")
        try:
            result = receive.recv()
        except EOFError as exc:
            raise ExtractionError("Regex worker exited without results") from exc
        if "error" in result:
            raise ExtractionError(result["error"])
        return result["candidates"]
    finally:
        receive.close()
        send.close()
        if started:
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()


def _convert(text, field_type):
    if not text or len(text) > MAX_VALUE_CHARS:
        raise ExtractionError("Value must contain 1..1024 characters")
    if field_type == "string":
        return text
    if field_type == "integer":
        if not re.fullmatch(r"[+-]?[0-9]+", text):
            raise ExtractionError("Invalid integer")
        return int(text)
    if field_type == "number":
        if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?)(?:[eE][+-]?[0-9]+)?", text):
            raise ExtractionError("Invalid number")
        value = float(text)
        if not math.isfinite(value):
            raise ExtractionError("Number must be finite")
        return value
    if field_type == "boolean":
        if text.lower() not in ("true", "false"):
            raise ExtractionError("Invalid boolean; expected true or false")
        return text.lower() == "true"
    if field_type == "date":
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
            raise ExtractionError("Invalid date; expected YYYY-MM-DD")
        try:
            return datetime.date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise ExtractionError("Invalid calendar date") from exc
    raise ExtractionError("Unsupported field type")


def extract(payload, extractor=None):
    """Extract patterns, optionally supplemented by a trusted synchronous callback.

    Callback(document_copy, field_copies) returns a list of objects with exactly
    field_id, start, end, value. Spans must ground the typed value in that document.
    Invalid callback output rejects the entire request, never silently dropping it.
    """
    fields, documents = _validate(payload)
    if extractor is not None and not callable(extractor):
        raise ExtractionError("extractor must be callable")
    field_map = {field["id"]: field for field in fields}
    document_map = {document["id"]: document for document in documents}
    candidates = [
        (*candidate, "regex", None)
        for candidate in _regex_candidates(fields, documents)
    ]
    if extractor is not None:
        for document in documents:
            try:
                supplied = extractor(dict(document), [dict(field) for field in fields])
            except Exception as exc:
                raise ExtractionError("Injected extractor failed") from exc
            if not isinstance(supplied, list):
                raise ExtractionError("Injected extractor must return a list")
            if len(candidates) + len(supplied) > MAX_CANDIDATES:
                raise ExtractionError("Combined candidate limit exceeded")
            for item in supplied:
                _keys(item, ("field_id", "start", "end", "value"))
                if not isinstance(item["field_id"], str) or item["field_id"] not in field_map:
                    raise ExtractionError("Injected extractor returned an unknown field")
                start, end = item["start"], item["end"]
                if type(start) is not int or type(end) is not int:
                    raise ExtractionError("Source span offsets must be integers")
                if not 0 <= start < end <= len(document["text"]):
                    raise ExtractionError("Source span is out of bounds or empty")
                expected = _convert(document["text"][start:end], field_map[item["field_id"]]["type"])
                if type(item["value"]) is not type(expected) or item["value"] != expected:
                    raise ExtractionError("Injected value is not exactly grounded with the declared type")
                candidates.append((item["field_id"], document["id"], start, end, "callback", expected))
    results = {
        field["id"]: {
            "id": field["id"], "type": field["type"], "required": field["required"],
            "status": "missing", "values": [],
        }
        for field in fields
    }
    rejected = []
    for field_id, document_id, start, end, method, supplied_value in candidates:
        text = document_map[document_id]["text"][start:end]
        source = {
            "document_id": document_id, "start": start, "end": end,
            "text": text, "method": method,
        }
        try:
            value = _convert(text, field_map[field_id]["type"]) if method == "regex" else supplied_value
        except ExtractionError as exc:
            rejected.append({"field_id": field_id, "source": source, "reason": str(exc)})
            continue
        values = results[field_id]["values"]
        existing = next((entry for entry in values if entry["value"] == value), None)
        if existing is None:
            values.append({"value": value, "sources": [source]})
        elif source not in existing["sources"]:
            existing["sources"].append(source)
    missing_required = []
    conflicts = []
    for field in fields:
        result = results[field["id"]]
        count = len(result["values"])
        result["status"] = "missing" if count == 0 else "resolved" if count == 1 else "conflict"
        if count == 0 and field["required"]:
            missing_required.append(field["id"])
        if count > 1:
            conflicts.append(field["id"])
    return {
        "fields": list(results.values()),
        "missing_required": missing_required,
        "conflicts": conflicts,
        "rejected_candidates": rejected,
    }


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ExtractionError(f"Nonstandard JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ExtractionError("Usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("rb") as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ExtractionError("Input JSON exceeds 1048576 bytes")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant)
        output = extract(payload)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, OSError, RecursionError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
