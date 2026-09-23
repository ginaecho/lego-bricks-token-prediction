"""Deterministic synthetic-feedback reference CLI; Python standard library only."""

import json
from pathlib import Path
import re
import sys
import unicodedata


SCHEMA_VERSION = 1
THEME_WORDS = {
    "delivery": frozenset(("delivery", "shipping", "shipment", "arrived", "late")),
    "price": frozenset(("price", "expensive", "cheap", "cost", "affordable")),
    "quality": frozenset(("quality", "broken", "durable", "defective", "damaged")),
    "support": frozenset(("support", "service", "refund", "helpful", "response")),
    "usability": frozenset(("easy", "difficult", "confusing", "usable", "usability")),
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, location):
    require(isinstance(value, dict), f"{location} must be an object")
    require(set(value) == set(expected), f"{location} requires keys {sorted(expected)}")


def normalize(text):
    """Casefold, compatibility-normalize, and collapse punctuation/whitespace."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def classify(normalized):
    tokens = set(normalized.split())
    return [name for name, words in THEME_WORDS.items() if tokens & words] or ["other"]


def validate(document, kind="input", original=None):
    """Shared validation boundary for CLI input and analysis output."""
    if kind == "input":
        keys(document, ("schema_version", "dataset_label", "feedback"), "input")
        require(type(document["schema_version"]) is int
                and document["schema_version"] == SCHEMA_VERSION,
                "schema_version must be integer 1")
        require(document["dataset_label"] == "synthetic", "dataset_label must be synthetic")
        rows = document["feedback"]
        require(isinstance(rows, list) and len(rows) <= 10000,
                "feedback must be a list of at most 10000 records")
        seen = set()
        for i, row in enumerate(rows):
            keys(row, ("id", "text"), f"feedback[{i}]")
            identifier, text = row["id"], row["text"]
            require(isinstance(identifier, str) and 0 < len(identifier) <= 128
                    and identifier == identifier.strip(), "id must be a nonblank trimmed string")
            require(identifier not in seen, f"duplicate feedback id: {identifier}")
            seen.add(identifier)
            require(isinstance(text, str) and 0 < len(text) <= 4000,
                    "text must be a string of 1 to 4000 characters")
            require(bool(normalize(text)), "text must contain a word character")
        return document
    require(kind == "output", "unknown validation kind")
    validate(original)
    keys(document, ("schema_version", "dataset_label", "status", "summary", "groups", "themes"),
         "output")
    require(type(document["schema_version"]) is int and document["schema_version"] == 1
            and document["dataset_label"] == "synthetic" and document["status"] == "ok",
            "invalid output envelope")
    rows = {row["id"]: row for row in original["feedback"]}
    groups = document["groups"]
    require(isinstance(groups, list), "groups must be a list")
    seen, normalized_seen, by_id = [], set(), {}
    for group in groups:
        keys(group, ("canonical_id", "member_ids", "normalized_text", "themes"), "group")
        members = group["member_ids"]
        require(isinstance(members, list) and bool(members)
                and all(isinstance(mid, str) and mid in rows for mid in members),
                "group members must reference input")
        canonical = group["canonical_id"]
        require(canonical == members[0], "canonical must be first member")
        normalized = normalize(rows[canonical]["text"])
        require(group["normalized_text"] == normalized and normalized not in normalized_seen,
                "invalid or repeated normalized group")
        require(all(normalize(rows[mid]["text"]) == normalized for mid in members),
                "group contains nonduplicate feedback")
        require(group["themes"] == classify(normalized), "invalid group themes")
        normalized_seen.add(normalized)
        seen.extend(members)
        by_id[canonical] = group
    require(len(seen) == len(set(seen)) and set(seen) == set(rows),
            "groups must partition input exactly")
    summary = document["summary"]
    keys(summary, ("input_count", "unique_count", "duplicate_count"), "summary")
    require(all(type(value) is int for value in summary.values()), "counts must be integers")
    require(summary == {"input_count": len(rows), "unique_count": len(groups),
                        "duplicate_count": len(rows) - len(groups)}, "invalid counts")
    themes = document["themes"]
    require(isinstance(themes, list), "themes must be a list")
    expected_names = [name for name in (*THEME_WORDS, "other")
                      if any(name in group["themes"] for group in groups)]
    require(all(isinstance(theme, dict) for theme in themes), "themes must be objects")
    require([theme.get("name") for theme in themes] == expected_names, "invalid theme ordering")
    for theme in themes:
        keys(theme, ("name", "unique_count", "original_count", "evidence"), "theme")
        matching = [group for group in groups if theme["name"] in group["themes"]]
        require(type(theme["unique_count"]) is int and type(theme["original_count"]) is int
                and theme["unique_count"] == len(matching)
                and theme["original_count"] == sum(len(g["member_ids"]) for g in matching),
                "invalid theme counts")
        evidence = theme["evidence"]
        require(isinstance(evidence, list) and len(evidence) == len(matching),
                "missing theme evidence")
        for item, group in zip(evidence, matching):
            keys(item, ("feedback_id", "start", "end", "excerpt"), "evidence")
            text = rows[group["canonical_id"]]["text"]
            require(item["feedback_id"] == group["canonical_id"]
                    and type(item["start"]) is int and item["start"] == 0
                    and type(item["end"]) is int and item["end"] == len(text)
                    and item["excerpt"] == text, "excerpt must trace to original feedback")
    return document


def analyze(document):
    validate(document)
    grouped = {}
    originals = {}
    for row in document["feedback"]:
        normalized = normalize(row["text"])
        if normalized not in grouped:
            grouped[normalized] = {
                "canonical_id": row["id"], "member_ids": [],
                "normalized_text": normalized, "themes": classify(normalized),
            }
            originals[row["id"]] = row["text"]
        grouped[normalized]["member_ids"].append(row["id"])
    groups = list(grouped.values())
    themes = []
    for name in (*THEME_WORDS, "other"):
        matching = [group for group in groups if name in group["themes"]]
        if matching:
            themes.append({
                "name": name,
                "unique_count": len(matching),
                "original_count": sum(len(group["member_ids"]) for group in matching),
                "evidence": [
                    {"feedback_id": group["canonical_id"], "start": 0,
                     "end": len(originals[group["canonical_id"]]),
                     "excerpt": originals[group["canonical_id"]]}
                    for group in matching
                ],
            })
    result = {
        "schema_version": 1, "dataset_label": "synthetic", "status": "ok",
        "summary": {"input_count": len(document["feedback"]), "unique_count": len(groups),
                    "duplicate_count": len(document["feedback"]) - len(groups)},
        "groups": groups, "themes": themes,
    }
    return validate(result, "output", document)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError(f"invalid JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        text = Path(argv[0]).read_text(encoding="utf-8")
        document = json.loads(text, object_pairs_hook=unique_object,
                              parse_constant=invalid_constant)
        result = analyze(document)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)},
                         ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
