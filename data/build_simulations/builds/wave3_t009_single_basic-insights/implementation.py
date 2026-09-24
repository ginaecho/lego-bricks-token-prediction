"""Deterministic customer-feedback insights; Python standard library only.

Run: python -B implementation.py example_input.json
Matching is case-insensitive, whole-token/phrase based, and English-oriented.
Sentiment and theme assignments are heuristics, not established customer intent.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "1.0"
MAX_FILE_BYTES = 2_000_000
DEFAULT_THEMES = [
    {"id": "quality", "name": "Product quality",
     "keywords": ["broken", "bug", "crash", "reliable", "quality", "defective"],
     "action": "Review quality-related evidence and reproduce recurring defects."},
    {"id": "usability", "name": "Ease of use",
     "keywords": ["confusing", "easy", "difficult", "navigation", "intuitive"],
     "action": "Test the reported workflows with customers and simplify friction points."},
    {"id": "support", "name": "Customer support",
     "keywords": ["support", "service", "agent", "response"],
     "action": "Audit support conversations and response-time expectations."},
    {"id": "pricing", "name": "Pricing and value",
     "keywords": ["price", "pricing", "expensive", "cheap", "cost", "value"],
     "action": "Review perceived value and clarify pricing against cited concerns."},
    {"id": "delivery", "name": "Delivery and fulfillment",
     "keywords": ["delivery", "shipping", "shipment", "late", "arrived"],
     "action": "Check fulfillment delays and improve delivery communication."},
    {"id": "requests", "name": "Feature requests",
     "keywords": ["wish", "please add", "feature", "missing", "would like"],
     "action": "Validate the requested capabilities with customers before prioritizing."},
]
POSITIVE = {"good", "great", "excellent", "love", "easy", "helpful", "fast",
            "reliable", "intuitive", "happy", "affordable"}
NEGATIVE = {"bad", "poor", "terrible", "hate", "broken", "confusing", "difficult",
            "expensive", "slow", "late", "defective", "crash", "frustrating"}
NEGATORS = {"not", "never", "no", "isn't", "wasn't", "don't", "didn't"}


class ValidationError(ValueError):
    """An input does not conform to the shared schema."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, path, maximum=5000):
    require(isinstance(value, str), f"{path} must be a string")
    require(bool(value.strip()) and len(value) <= maximum,
            f"{path} must contain 1..{maximum} characters and not be blank")
    return value.strip()


def object_keys(value, required, optional, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(required <= value.keys(), f"{path} is missing required fields")
    require(value.keys() <= required | optional, f"{path} has unknown fields")


def tokens(value):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", value.casefold(), re.UNICODE)


def validate_input(payload):
    """All public entry points share this validation and normalization layer."""
    object_keys(payload, {"schema_version", "feedback"},
                {"data_label", "themes"}, "input")
    require(payload["schema_version"] == SCHEMA_VERSION,
            "schema_version must be '1.0'")
    label = text(payload.get("data_label", "unspecified"), "data_label", 200)
    rows = payload["feedback"]
    require(isinstance(rows, list) and len(rows) <= 10000,
            "feedback must be an array with at most 10000 entries")
    normalized = []
    ids = set()
    for i, row in enumerate(rows):
        path = f"feedback[{i}]"
        object_keys(row, {"id", "text"}, {"source", "customer_id"}, path)
        item = {key: text(value, f"{path}.{key}", 5000 if key == "text" else 200)
                for key, value in row.items()}
        require(item["id"] not in ids, f"{path}.id must be unique")
        ids.add(item["id"])
        item.setdefault("source", "unspecified")
        normalized.append(item)
    themes = payload.get("themes", DEFAULT_THEMES)
    require(isinstance(themes, list) and 1 <= len(themes) <= 50,
            "themes must contain 1..50 theme definitions")
    normalized_themes = []
    theme_ids = set()
    for i, theme in enumerate(themes):
        path = f"themes[{i}]"
        object_keys(theme, {"id", "name", "keywords", "action"}, set(), path)
        result = {key: text(theme[key], f"{path}.{key}", 1000 if key == "action" else 200)
                  for key in ("id", "name", "action")}
        require(result["id"] != "other" and result["id"] not in theme_ids,
                f"{path}.id must be unique and cannot be 'other'")
        theme_ids.add(result["id"])
        keywords = theme["keywords"]
        require(isinstance(keywords, list) and 1 <= len(keywords) <= 50,
                f"{path}.keywords must contain 1..50 strings")
        result["keywords"] = []
        for j, keyword in enumerate(keywords):
            keyword = text(keyword, f"{path}.keywords[{j}]", 200)
            require(bool(tokens(keyword)), f"{path}.keywords[{j}] needs word tokens")
            result["keywords"].append(keyword)
        normalized_themes.append(result)
    return label, normalized, normalized_themes


def contains_phrase(words, phrase):
    size = len(phrase)
    return any(words[i:i + size] == phrase for i in range(len(words) - size + 1))


def sentiment(words):
    score = 0
    for index, word in enumerate(words):
        weight = int(word in POSITIVE) - int(word in NEGATIVE)
        # Deliberately bounded heuristic: only the immediately preceding token.
        if index and words[index - 1] in NEGATORS:
            weight = -weight
        score += weight
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def analyze(payload):
    label, rows, definitions = validate_input(payload)
    definitions = definitions + [{
        "id": "other", "name": "Uncategorized feedback", "keywords": [],
        "action": "Read uncategorized feedback and refine the theme dictionary."
    }]
    groups = {theme["id"]: [] for theme in definitions}
    assignments = []
    totals = Counter()
    for row in sorted(rows, key=lambda item: item["id"]):
        words = tokens(row["text"])
        feeling = sentiment(words)
        matched = sorted(theme["id"] for theme in definitions
                         if any(contains_phrase(words, tokens(keyword))
                                for keyword in theme["keywords"]))
        matched = matched or ["other"]
        totals[feeling] += 1
        assignments.append({"feedback_id": row["id"], "theme_ids": matched,
                            "sentiment": feeling})
        for theme_id in matched:
            groups[theme_id].append((row, feeling))
    results = []
    for theme in definitions:
        members = groups[theme["id"]]
        if not members:
            continue
        counts = Counter(feeling for _, feeling in members)
        score = 3 * counts["negative"] + counts["neutral"] + counts["positive"]
        results.append({
            "id": theme["id"], "name": theme["name"],
            "feedback_count": len(members),
            "share_of_feedback": round(len(members) / len(rows), 4),
            "sentiment_counts": {key: counts[key] for key in
                                 ("negative", "neutral", "positive")},
            "priority_score": score,
            "priority_reason": "3 × negative + neutral + positive feedback counts",
            "recommended_action": theme["action"],
            "evidence": [{"feedback_id": row["id"], "text": row["text"],
                          "source": row["source"]} for row, _ in members[:3]],
        })
    results.sort(key=lambda item: (-item["priority_score"], -item["feedback_count"],
                                   item["id"]))
    return {
        "schema_version": SCHEMA_VERSION, "status": "ok", "data_label": label,
        "summary": {
            "total_feedback": len(rows),
            "theme_count": len(results),
            "sentiment_counts": {key: totals[key] for key in
                                 ("negative", "neutral", "positive")},
            "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        },
        "themes": results, "assignments": assignments,
        "limitations": [
            "English-oriented keyword and sentiment heuristics; no semantic inference.",
            "Themes overlap, so theme counts and shares need not sum to the total.",
            "Counts represent feedback records, not unique customers; duplicate text is retained.",
            "Evidence shows at most three records per theme, ordered by feedback ID.",
            "Sentiment is record-level; mixed opinions can cancel and sarcasm is not detected.",
            "Recommended actions are review suggestions, not verified root causes.",
        ],
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"Non-standard JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        require(len(raw) <= MAX_FILE_BYTES, "Input file exceeds 2000000 bytes")
        payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = analyze(payload)
        code = 0
    except (ValidationError, OSError, ValueError, RecursionError) as error:
        result = {"schema_version": SCHEMA_VERSION, "status": "error",
                  "error": {"type": "validation_or_file_error", "message": str(error)}}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
