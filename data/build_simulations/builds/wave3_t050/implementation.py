"""Deterministic feedback analysis; Python standard library, no providers.

Input v1: {schema_version: 1, dataset_label: str,
           feedback: [{id: str, text: str}]}.
Duplicates share NFKC/casefolded word tokens; punctuation and whitespace
are ignored. This is lexical equivalence, not semantic deduplication.
Output groups retain every original source; theme counts use unique groups.
Excerpt offsets are half-open Python character indices into original text.
"""

import json
from pathlib import Path
import re
import sys
import unicodedata


THEMES = {
    "delivery": frozenset(("delivery", "shipping", "shipment", "arrived", "late")),
    "price": frozenset(("price", "cost", "expensive", "cheap", "refund")),
    "product_quality": frozenset(("quality", "broken", "durable", "defective", "damaged")),
    "support": frozenset(("support", "service", "agent", "helpful", "unresponsive")),
    "usability": frozenset(("easy", "difficult", "confusing", "interface", "usable")),
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(expected), location + " has missing or unknown fields")


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def validate(data, kind="input"):
    """Shared input and output contract enforcement, including evidence links."""
    require(kind in ("input", "output"), "unknown validation kind")
    if kind == "input":
        keys(data, ("schema_version", "dataset_label", "feedback"), "input")
        require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                "schema_version must be integer 1")
        label = data["dataset_label"]
        require(isinstance(label, str) and 0 < len(label.strip()) <= 200,
                "dataset_label must be a nonempty string of at most 200 characters")
        items = data["feedback"]
        require(isinstance(items, list) and len(items) <= 1000,
                "feedback must be an array of at most 1000 records")
        seen = set()
        for index, item in enumerate(items):
            keys(item, ("id", "text"), "feedback[%d]" % index)
            identifier, text = item["id"], item["text"]
            require(isinstance(identifier, str) and 0 < len(identifier) <= 128
                    and identifier == identifier.strip(), "id must be a trimmed nonempty string")
            require(identifier not in seen, "feedback IDs must be unique")
            seen.add(identifier)
            require(isinstance(text, str) and 0 < len(text) <= 10000,
                    "text must be a nonempty string of at most 10000 characters")
            require(bool(normalized(text)), "text must contain word characters")
        return data

    keys(data, ("schema_version", "status", "dataset_label", "summary", "groups", "themes"),
         "output")
    require(data["schema_version"] == 1 and data["status"] == "ok", "invalid output version/status")
    require(isinstance(data["groups"], list) and isinstance(data["themes"], list),
            "groups and themes must be arrays")
    source_records, group_sources = {}, {}
    for group in data["groups"]:
        keys(group, ("group_id", "canonical_id", "normalized_text", "sources"), "group")
        gid = group["group_id"]
        require(isinstance(gid, str) and gid not in group_sources, "invalid group ID")
        require(isinstance(group["sources"], list) and bool(group["sources"]), "empty group")
        group_sources[gid] = set()
        for source in group["sources"]:
            keys(source, ("id", "text"), "source")
            require(isinstance(source["id"], str) and source["id"] not in source_records,
                    "invalid or repeated source ID")
            require(isinstance(source["text"], str), "invalid source text")
            require(normalized(source["text"]) == group["normalized_text"],
                    "group normalization mismatch")
            source_records[source["id"]] = source
            group_sources[gid].add(source["id"])
        require(group["canonical_id"] == group["sources"][0]["id"], "invalid canonical ID")
    validate({"schema_version": data["schema_version"], "dataset_label": data["dataset_label"],
              "feedback": list(source_records.values())})
    keys(data["summary"], ("input_count", "unique_count", "duplicate_count"), "summary")
    expected = {"input_count": len(source_records), "unique_count": len(group_sources),
                "duplicate_count": len(source_records) - len(group_sources)}
    require(all(type(v) is int for v in data["summary"].values())
            and data["summary"] == expected, "summary count mismatch")
    theme_ids, covered = set(), set()
    for theme in data["themes"]:
        keys(theme, ("theme_id", "unique_feedback_count", "source_count", "group_ids", "excerpts"),
             "theme")
        tid = theme["theme_id"]
        require(isinstance(tid, str) and tid in (*THEMES, "other") and tid not in theme_ids,
                "invalid or repeated theme")
        theme_ids.add(tid)
        gids = theme["group_ids"]
        require(isinstance(gids, list) and bool(gids)
                and all(isinstance(g, str) and g in group_sources for g in gids)
                and len(set(gids)) == len(gids), "invalid theme group IDs")
        require(type(theme["unique_feedback_count"]) is int
                and theme["unique_feedback_count"] == len(gids), "theme count mismatch")
        expected_sources = set().union(*(group_sources[g] for g in gids))
        require(type(theme["source_count"]) is int
                and theme["source_count"] == len(expected_sources), "source count mismatch")
        require(isinstance(theme["excerpts"], list), "excerpts must be an array")
        excerpt_sources = set()
        for excerpt in theme["excerpts"]:
            keys(excerpt, ("group_id", "source_id", "start", "end", "text"), "excerpt")
            gid, sid = excerpt["group_id"], excerpt["source_id"]
            require(isinstance(gid, str) and gid in gids and isinstance(sid, str)
                    and sid in group_sources[gid], "excerpt source link mismatch")
            start, end = excerpt["start"], excerpt["end"]
            text = source_records[sid]["text"]
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(text),
                    "invalid excerpt offsets")
            require(excerpt["text"] == text[start:end], "excerpt is not an exact source slice")
            excerpt_sources.add(sid)
        require(excerpt_sources == expected_sources, "missing supporting sources")
        covered.update(gids)
    require(covered == set(group_sources), "every group must have a theme")
    return data


def source_matches(text):
    words = set(normalized(text).split())
    result = {theme: (0, len(text)) for theme, terms in THEMES.items() if words & terms}
    found = set()
    for match in re.finditer(r"\w+", text):
        word = unicodedata.normalize("NFKC", match.group()).casefold()
        for theme, terms in THEMES.items():
            if theme in result and word in terms and theme not in found:
                result[theme] = (max(0, match.start() - 40), min(len(text), match.end() + 40))
                found.add(theme)
    return result or {"other": (0, min(len(text), 120))}


def analyze(data):
    validate(data)
    groups, by_text = [], {}
    for source in data["feedback"]:
        norm = normalized(source["text"])
        if norm not in by_text:
            group = {"group_id": "g%04d" % (len(groups) + 1),
                     "canonical_id": source["id"], "normalized_text": norm, "sources": []}
            by_text[norm] = group
            groups.append(group)
        by_text[norm]["sources"].append(dict(source))
    themes = {}
    for group in groups:
        for source in group["sources"]:
            for tid, (start, end) in source_matches(source["text"]).items():
                theme = themes.setdefault(tid, {"theme_id": tid, "unique_feedback_count": 0,
                                               "source_count": 0, "group_ids": [], "excerpts": []})
                if group["group_id"] not in theme["group_ids"]:
                    theme["group_ids"].append(group["group_id"])
                    theme["unique_feedback_count"] += 1
                theme["source_count"] += 1
                theme["excerpts"].append({"group_id": group["group_id"], "source_id": source["id"],
                                          "start": start, "end": end, "text": source["text"][start:end]})
    count = len(data["feedback"])
    result = {"schema_version": 1, "status": "ok", "dataset_label": data["dataset_label"],
              "summary": {"input_count": count, "unique_count": len(groups),
                          "duplicate_count": count - len(groups)},
              "groups": groups, "themes": [themes[tid] for tid in sorted(themes)]}
    return validate(result, "output")


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_text(encoding="utf-8-sig")
        data = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = analyze(data)
    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error",
                          "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
