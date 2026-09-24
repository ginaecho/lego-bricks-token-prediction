"""Deterministic, keyword-based customer insights reference CLI (stdlib only)."""

import json
import re
import sys
from collections import Counter
from pathlib import Path


DEFAULT_RULES = [
    {"name": "usability", "keywords": ["confusing", "difficult", "easy", "navigation"],
     "action": "Review the cited workflows and test a simpler customer journey."},
    {"name": "reliability", "keywords": ["crash", "broken", "bug", "error"],
     "action": "Reproduce cited failures and prioritize fixes by affected customers."},
    {"name": "performance", "keywords": ["slow", "fast", "loading", "latency"],
     "action": "Measure response times for cited workflows and remove bottlenecks."},
    {"name": "pricing", "keywords": ["price", "expensive", "cost", "affordable"],
     "action": "Review perceived value and test clearer pricing communication."},
    {"name": "support", "keywords": ["support", "helpful", "response", "service"],
     "action": "Review support examples and improve response quality and timeliness."},
]
NEGATIVE = {"confusing", "difficult", "crash", "broken", "bug", "error", "slow",
            "expensive", "bad", "poor", "hate", "frustrating"}
POSITIVE = {"easy", "fast", "affordable", "helpful", "great", "love", "excellent"}


class ValidationError(ValueError):
    pass


def validate(value, kind="input"):
    """Single validation entry point for the shared request schema."""
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    def text(v, label, maximum=10000):
        require(isinstance(v, str) and bool(v.strip()) and len(v) <= maximum,
                f"{label} must be a nonempty string of at most {maximum} characters")

    require(kind == "input", "unsupported schema kind")
    require(isinstance(value, dict), "input must be an object")
    require(set(value) <= {"schema_version", "dataset_label", "synthetic",
                           "feedback", "theme_rules", "max_evidence"}, "unknown input field")
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
            "schema_version must be 1")
    text(value.get("dataset_label"), "dataset_label", 200)
    require(type(value.get("synthetic")) is bool, "synthetic must be a boolean")
    feedback = value.get("feedback")
    require(isinstance(feedback, list) and len(feedback) <= 10000,
            "feedback must be a list with at most 10000 items")
    seen = set()
    for i, item in enumerate(feedback):
        require(isinstance(item, dict), f"feedback[{i}] must be an object")
        require(set(item) <= {"id", "text", "source", "rating"},
                f"feedback[{i}] has unknown fields")
        for key in ("id", "text", "source"):
            text(item.get(key), f"feedback[{i}].{key}", 10000 if key == "text" else 200)
        require(item["id"] not in seen, "feedback ids must be unique")
        seen.add(item["id"])
        if "rating" in item:
            require(type(item["rating"]) is int and 1 <= item["rating"] <= 5,
                    f"feedback[{i}].rating must be an integer from 1 to 5")
    limit = value.get("max_evidence", 3)
    require(type(limit) is int and 1 <= limit <= 20,
            "max_evidence must be an integer from 1 to 20")
    rules = value.get("theme_rules", DEFAULT_RULES)
    require(isinstance(rules, list) and 1 <= len(rules) <= 100,
            "theme_rules must contain 1 to 100 rules")
    names = set()
    for rule in rules:
        require(isinstance(rule, dict) and set(rule) == {"name", "keywords", "action"},
                "each theme rule requires exactly name, keywords and action")
        text(rule["name"], "theme name", 100)
        name = rule["name"].strip().casefold()
        require(name not in names and name != "uncategorized",
                "theme names must be unique (case-insensitive); uncategorized is reserved")
        names.add(name)
        text(rule["action"], "theme action", 1000)
        require(isinstance(rule["keywords"], list) and 1 <= len(rule["keywords"]) <= 100,
                "keywords must contain 1 to 100 phrases")
        for keyword in rule["keywords"]:
            text(keyword, "keyword", 100)
            require(bool(re.search(r"\w", keyword)), "keywords must contain a word character")
    return value


def normalize(text):
    return " ".join(text.casefold().split())


def matches(text, keyword):
    return re.search(r"(?<!\w)" + re.escape(normalize(keyword)) + r"(?!\w)", text) is not None


def sentiment(item):
    if "rating" in item:
        return "negative" if item["rating"] < 3 else "positive" if item["rating"] > 3 else "neutral"
    words = set(re.findall(r"\w+", item["text"].casefold()))
    score = len(words & POSITIVE) - len(words & NEGATIVE)
    return "negative" if score < 0 else "positive" if score > 0 else "neutral"


def analyze(request):
    request = validate(request)
    rules = request.get("theme_rules", DEFAULT_RULES)
    rule_map = {rule["name"].strip(): rule for rule in rules}
    groups = {}
    assignments = []
    totals = Counter()
    for item in sorted(request["feedback"], key=lambda entry: entry["id"]):
        body = normalize(item["text"])
        names = sorted(name for name, rule in rule_map.items()
                       if any(matches(body, keyword) for keyword in rule["keywords"]))
        names = names or ["uncategorized"]
        tone = sentiment(item)
        totals[tone] += 1
        assignments.append({"feedback_id": item["id"], "themes": names, "sentiment": tone})
        for name in names:
            groups.setdefault(name, []).append((item, tone))
    themes = []
    count = len(request["feedback"])
    for name, entries in groups.items():
        tones = Counter(tone for _, tone in entries)
        score = len(entries) + 2 * tones["negative"]
        action = (rule_map[name]["action"] if name in rule_map else
                  "Read uncategorized feedback and add a theme rule if a pattern recurs.")
        if name != "uncategorized" and not tones["negative"]:
            action = "Preserve strengths and validate opportunities: " + action
        themes.append({
            "name": name,
            "feedback_count": len(entries),
            "share_of_feedback": round(len(entries) / count, 4),
            "sentiment_counts": {key: tones[key] for key in ("negative", "neutral", "positive")},
            "source_counts": dict(sorted(Counter(item["source"] for item, _ in entries).items())),
            "priority_score": score,
            "recommended_action": action,
            "feedback_ids": [item["id"] for item, _ in entries],
            "evidence": [{"feedback_id": item["id"], "text": item["text"], "source": item["source"]}
                         for item, _ in sorted(entries, key=lambda pair:
                                               (pair[1] != "negative", pair[0]["id"]))
                         [:request.get("max_evidence", 3)]],
        })
    themes.sort(key=lambda theme: (-theme["priority_score"], theme["name"]))
    return {
        "status": "ok", "schema_version": 1,
        "dataset_label": request["dataset_label"], "synthetic": request["synthetic"],
        "summary": {"feedback_count": count, "theme_count": len(themes),
                    "sentiment_counts": {key: totals[key] for key in ("negative", "neutral", "positive")},
                    "uncategorized_count": len(groups.get("uncategorized", []))},
        "themes": themes, "assignments": assignments,
        "method": {"theme_detection": "case-insensitive whole-word/phrase rules; multiple themes allowed",
                   "sentiment": "rating first; otherwise unique positive minus negative lexicon words",
                   "priority": "feedback_count + 2 * negative_count",
                   "limitations": ["No semantic inference, stemming, sarcasm or negation handling.",
                                   "Themes can overlap; shares may sum above one.",
                                   "Priority is a heuristic, not statistical evidence.",
                                   "Repeated text with different ids counts separately."]},
    }


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        request = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                             parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = analyze(request)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
