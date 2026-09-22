"""Offline feedback grouping with auditable, verbatim source evidence."""

import argparse
import copy
import json
import re
import sys
import unicodedata


class ValidationError(ValueError):
    """The input or injected callback violated the public contract."""


ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
DEFAULT_THEMES = [
    {"id": "performance", "keywords": ["fast", "slow", "performance"]},
    {"id": "support", "keywords": ["support", "helpdesk"]},
    {"id": "usability", "keywords": ["easy", "confusing", "usability"]},
]


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _object(value, required, optional=(), where="object"):
    if not isinstance(value, dict):
        raise ValidationError(f"{where} must be an object")
    if set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValidationError(f"{where} has missing or unknown fields")


def _identifier(value, where):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValidationError(f"{where} must be a 1-128 character identifier")


def _list(value, where, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValidationError(f"{where} must be a list of at most {maximum} items")


def _text(value, where, maximum, nonempty=False):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValidationError(f"{where} must be a string of at most {maximum} characters")
    if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValidationError(f"{where} contains an unpaired Unicode surrogate")
    if nonempty and not normalize(value):
        raise ValidationError(f"{where} must not be blank")


def _validate(payload):
    _object(payload, ("records",), ("themes",), "input")
    _list(payload["records"], "records", 10000)
    records = {}
    total = 0
    for record in payload["records"]:
        _object(record, ("id", "text"), where="record")
        _identifier(record["id"], "record.id")
        _text(record["text"], "record.text", 100000)
        if record["id"] in records:
            raise ValidationError("record IDs must be unique")
        records[record["id"]] = record["text"]
        total += len(record["text"])
    if total > 1000000:
        raise ValidationError("total feedback exceeds 1000000 characters")
    supplied = payload.get("themes", DEFAULT_THEMES)
    _list(supplied, "themes", 100)
    themes = {}
    for theme in supplied:
        _object(theme, ("id", "keywords"), where="theme")
        _identifier(theme["id"], "theme.id")
        if theme["id"] in themes:
            raise ValidationError("theme IDs must be unique")
        _list(theme["keywords"], "theme.keywords", 100)
        if not theme["keywords"]:
            raise ValidationError("each theme needs at least one keyword")
        for keyword in theme["keywords"]:
            _text(keyword, "keyword", 200, nonempty=True)
        themes[theme["id"]] = sorted(set(map(normalize, theme["keywords"])))
    return dict(sorted(records.items())), dict(sorted(themes.items()))


def analyze(payload, callback=None):
    """Return deterministic groups; optional callback adds span-backed assignments.

    Callback input: {records: [{id,text}], themes: [{id,keywords}]}.
    Callback output: {assignments: [{record_id,theme_id,start,end}],
                      syntheses: [{theme_id,text,supports:
                                    [{record_id,start,end}]}]}.
    Both output lists are required. Synthesis text must equal its supporting
    excerpts joined by a newline, making synthesis extractive, not generative.
    Offsets are zero-based Python Unicode code-point indices, end-exclusive.
    """
    records, themes = _validate(payload)
    groups = {}
    empty_ids = []
    for record_id, text in records.items():
        key = normalize(text)
        if key:
            groups.setdefault(key, []).append(record_id)
        else:
            empty_ids.append(record_id)
    canonical = {
        record_id: ids[0] for ids in groups.values() for record_id in ids
    }
    evidence = {theme_id: set() for theme_id in themes}
    for normalized_text, ids in groups.items():
        for theme_id, keywords in themes.items():
            if any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)",
                             normalized_text) for word in keywords):
                evidence[theme_id].update(
                    (record_id, 0, len(records[record_id])) for record_id in ids
                )

    def span(value):
        _object(value, ("record_id", "start", "end"), where="source span")
        record_id = value["record_id"]
        _identifier(record_id, "source record_id")
        if record_id not in records:
            raise ValidationError("source span references an unknown record ID")
        start, end = value["start"], value["end"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(records[record_id])):
            raise ValidationError("source span has invalid offsets")
        if record_id not in canonical or not normalize(records[record_id][start:end]):
            raise ValidationError("source span must contain nonempty feedback")
        return record_id, start, end

    def known_theme(theme_id):
        _identifier(theme_id, "theme_id")
        if theme_id not in themes:
            raise ValidationError("callback references an unknown theme ID")

    def rendered(source):
        record_id, start, end = source
        return {
            "feedback_id": canonical[record_id], "record_id": record_id,
            "start": start, "end": end, "excerpt": records[record_id][start:end],
        }

    syntheses = {theme_id: [] for theme_id in themes}
    if callback is not None:
        if not callable(callback):
            raise ValidationError("callback must be callable")
        context = {
            "records": [{"id": key, "text": value} for key, value in records.items()],
            "themes": [{"id": key, "keywords": value} for key, value in themes.items()],
        }
        try:
            proposals = callback(copy.deepcopy(context))
        except Exception as exc:
            raise ValidationError("callback failed") from exc
        _object(proposals, ("assignments", "syntheses"), where="callback result")
        _list(proposals["assignments"], "assignments", 10000)
        _list(proposals["syntheses"], "syntheses", 1000)
        for assignment in proposals["assignments"]:
            _object(assignment, ("record_id", "theme_id", "start", "end"),
                    where="assignment")
            known_theme(assignment["theme_id"])
            source = span({key: assignment[key] for key in ("record_id", "start", "end")})
            evidence[assignment["theme_id"]].add(source)
        for synthesis in proposals["syntheses"]:
            _object(synthesis, ("theme_id", "text", "supports"), where="synthesis")
            theme_id = synthesis["theme_id"]
            known_theme(theme_id)
            _text(synthesis["text"], "synthesis.text", 1000000, nonempty=True)
            _list(synthesis["supports"], "synthesis.supports", 1000)
            if not synthesis["supports"]:
                raise ValidationError("synthesis requires source spans")
            sources = [span(value) for value in synthesis["supports"]]
            if len(set(sources)) != len(sources):
                raise ValidationError("synthesis has duplicate source spans")
            for record_id, start, end in sources:
                if not any(rid == record_id and a <= start and end <= b
                           for rid, a, b in evidence[theme_id]):
                    raise ValidationError("synthesis span lacks a matching theme assignment")
            expected = "\n".join(records[rid][a:b] for rid, a, b in sources)
            if synthesis["text"] != expected:
                raise ValidationError("synthesis text must exactly join its source excerpts")
            item = {"text": expected, "supports": [rendered(source) for source in sources]}
            if item not in syntheses[theme_id]:
                syntheses[theme_id].append(item)

    assigned = set()
    output_themes = []
    for theme_id, keywords in themes.items():
        sources = sorted(evidence[theme_id])
        group_ids = {canonical[rid] for rid, _, _ in sources}
        assigned.update(group_ids)
        output_themes.append({
            "id": theme_id, "keywords": keywords,
            "status": "matched" if sources else "no_match",
            "distinct_feedback_count": len(group_ids),
            "supporting_record_count": len({rid for rid, _, _ in sources}),
            "evidence": [rendered(source) for source in sources],
            "syntheses": sorted(syntheses[theme_id],
                                key=lambda item: json.dumps(item, sort_keys=True)),
        })
    unmatched = [
        {"feedback_id": ids[0],
         "evidence": [rendered((rid, 0, len(records[rid]))) for rid in ids]}
        for ids in groups.values() if ids[0] not in assigned
    ]
    return {
        "schema_version": 1,
        "status": ("empty_feedback" if not groups else
                   "no_themes" if not themes else
                   "no_match" if not assigned else "ok"),
        "counts": {
            "input_records": len(records), "empty_records": len(empty_ids),
            "distinct_feedback": len(groups),
            "duplicate_records": len(canonical) - len(groups),
            "matched_feedback": len(assigned),
            "unmatched_feedback": len(unmatched),
        },
        "empty_record_ids": empty_ids,
        "duplicate_groups": sorted(
            [{"feedback_id": ids[0], "record_ids": ids}
             for ids in groups.values() if len(ids) > 1],
            key=lambda item: item["feedback_id"]),
        "themes": output_themes,
        "unmatched": sorted(unmatched, key=lambda item: item["feedback_id"]),
    }


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValidationError(f"nonfinite JSON value is forbidden: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="UTF-8 JSON file, or - for stdin")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            raw = sys.stdin.read(8000001)
        else:
            with open(args.input, encoding="utf-8") as stream:
                raw = stream.read(8000001)
        if len(raw) > 8000000:
            raise ValidationError("JSON document exceeds 8000000 characters")
        payload = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
        result = analyze(payload)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True,
                         indent=2 if args.pretty else None, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
