"""Deterministic customer-insights -> document-automation reference pipeline."""

import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path


VERSION = "1.0"
SOURCES = {
    "theme", "mentions", "customer_count", "positive", "neutral", "negative",
    "action", "priority", "evidence_ids",
}
DEFAULT_RULES = [
    {"name": "Reliability", "keywords": ["bug", "crash", "broken", "error"],
     "action": "Investigate recurring failures and verify fixes."},
    {"name": "Usability", "keywords": ["confusing", "difficult", "easy", "navigation"],
     "action": "Review the reported workflows with customers."},
    {"name": "Pricing", "keywords": ["price", "expensive", "cost", "affordable"],
     "action": "Review pricing clarity and perceived value."},
    {"name": "Support", "keywords": ["support", "response", "helpful"],
     "action": "Review support response quality and follow-up."},
]
DEFAULT_COLUMNS = [
    {"label": "Theme", "source": "theme", "transform": "identity"},
    {"label": "Mentions", "source": "mentions", "transform": "identity"},
    {"label": "Priority", "source": "priority", "transform": "identity"},
    {"label": "Action", "source": "action", "transform": "identity"},
    {"label": "Evidence", "source": "evidence_ids", "transform": "identity"},
]


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing fields: " + ", ".join(
        sorted(set(required) - set(value))))
    require(set(value) <= set(required) | set(optional), "Unknown fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    require(not any(ord(c) < 32 and c not in "\n\r\t" for c in value),
            name + " contains unsupported control characters")


def count(value, name):
    require(type(value) is int and value >= 0, name + " must be a nonnegative integer")


def tokens(value):
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def matches(message, keyword):
    haystack, needle = tokens(message), tokens(keyword)
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def sentiment(record):
    if "rating" in record:
        return "negative" if record["rating"] <= 2 else (
            "positive" if record["rating"] >= 4 else "neutral")
    words = set(tokens(record["text"]))
    score = len(words & {"good", "great", "love", "easy", "helpful", "excellent"})
    score -= len(words & {"bad", "hate", "broken", "confusing", "slow", "expensive"})
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def validate_input(data):
    keys(data, {"schema_version", "fixture_label", "feedback"}, {"theme_rules", "document"})
    require(data["schema_version"] == VERSION, "Unsupported schema_version")
    text(data["fixture_label"], "fixture_label")
    require(isinstance(data["feedback"], list), "feedback must be an array")
    seen = set()
    for record in data["feedback"]:
        keys(record, {"id", "customer", "text"}, {"rating"})
        for field in ("id", "customer", "text"):
            text(record[field], field)
        require(record["id"] not in seen, "Duplicate feedback id")
        seen.add(record["id"])
        if "rating" in record:
            require(type(record["rating"]) is int and 1 <= record["rating"] <= 5,
                    "rating must be an integer from 1 to 5")
    rules = data.get("theme_rules", DEFAULT_RULES)
    require(isinstance(rules, list), "theme_rules must be an array")
    names = set()
    for rule in rules:
        keys(rule, {"name", "keywords", "action"})
        text(rule["name"], "theme name")
        text(rule["action"], "action")
        normalized = rule["name"].strip().casefold()
        require(normalized not in names and normalized != "other",
                "Theme names must be unique; Other is reserved")
        names.add(normalized)
        require(isinstance(rule["keywords"], list) and len(rule["keywords"]) > 0,
                "keywords must be a nonempty array")
        for keyword in rule["keywords"]:
            text(keyword, "keyword")
            require(bool(tokens(keyword)), "keyword must contain a word")
    config = data.get("document", {})
    validate_config(config)
    return data


def validate_config(config):
    keys(config, set(), {"title", "format", "columns", "min_mentions"})
    text(config.get("title", "Customer feedback action report"), "title")
    require(config.get("format", "csv") in ("csv", "json"), "format must be csv or json")
    count(config.get("min_mentions", 1), "min_mentions")
    columns = config.get("columns", DEFAULT_COLUMNS)
    require(isinstance(columns, list) and bool(columns), "columns must be nonempty")
    labels = set()
    for column in columns:
        keys(column, {"label", "source"}, {"transform"})
        text(column["label"], "column label")
        text(column["source"], "column source")
        require(column["label"] not in labels, "Duplicate column label")
        labels.add(column["label"])
        require(column["source"] in SOURCES, "Unknown column source")
        transform = column.get("transform", "identity")
        require(transform in ("identity", "upper", "lower"), "Unknown transform")
        require(transform == "identity" or column["source"] in {
            "theme", "action", "priority", "evidence_ids"},
            "Text transforms require a text source")
    return config


def validate_insights(data):
    keys(data, {"feedback_count", "themes"})
    count(data["feedback_count"], "feedback_count")
    require(isinstance(data["themes"], list), "themes must be an array")
    names, records = set(), {}
    for theme in data["themes"]:
        keys(theme, {"theme", "mentions", "customer_count", "sentiments",
                     "action", "priority", "evidence"})
        text(theme["theme"], "theme")
        text(theme["action"], "action")
        require(theme["theme"] not in names, "Duplicate theme")
        names.add(theme["theme"])
        require(theme["priority"] in ("high", "medium", "low"), "Invalid priority")
        count(theme["mentions"], "mentions")
        count(theme["customer_count"], "customer_count")
        keys(theme["sentiments"], {"positive", "neutral", "negative"})
        for value in theme["sentiments"].values():
            count(value, "sentiment count")
        require(isinstance(theme["evidence"], list) and bool(theme["evidence"]),
                "Theme evidence must be nonempty")
        ids, customers, sentiments = set(), set(), Counter()
        for evidence in theme["evidence"]:
            keys(evidence, {"id", "customer", "text", "sentiment"})
            for field in ("id", "customer", "text"):
                text(evidence[field], field)
            require(evidence["sentiment"] in ("positive", "neutral", "negative"),
                    "Invalid evidence sentiment")
            require(evidence["id"] not in ids, "Duplicate evidence in theme")
            ids.add(evidence["id"])
            customers.add(evidence["customer"])
            sentiments[evidence["sentiment"]] += 1
            require(evidence["id"] not in records or records[evidence["id"]] == evidence,
                    "Inconsistent cross-theme evidence")
            records[evidence["id"]] = evidence
        require(theme["mentions"] == len(ids), "mentions does not match evidence")
        require(theme["customer_count"] == len(customers), "customer_count mismatch")
        require(all(theme["sentiments"][s] == sentiments[s]
                    for s in ("positive", "neutral", "negative")), "sentiment counts mismatch")
        require(theme["priority"] == priority(theme["sentiments"]), "priority mismatch")
    require(len(records) == data["feedback_count"], "feedback_count does not match evidence")
    return data


def priority(sentiments):
    return "high" if sentiments["negative"] > 0 else (
        "medium" if sentiments["neutral"] > 0 else "low")


def customer_insights(data):
    validate("input", data)
    groups = {}
    for record in data["feedback"]:
        rules = [r for r in data.get("theme_rules", DEFAULT_RULES)
                 if any(matches(record["text"], k) for k in r["keywords"])]
        if not rules:
            rules = [{"name": "Other", "action": "Review uncategorized feedback and refine themes."}]
        for rule in rules:
            theme = groups.setdefault(rule["name"], {
                "theme": rule["name"], "action": rule["action"], "evidence": []})
            theme["evidence"].append({
                "id": record["id"], "customer": record["customer"],
                "text": record["text"], "sentiment": sentiment(record)})
    result = {"feedback_count": len(data["feedback"]), "themes": []}
    for name in sorted(groups, key=lambda n: (n.casefold(), n)):
        theme = groups[name]
        theme["evidence"].sort(key=lambda e: e["id"])
        counts = Counter(e["sentiment"] for e in theme["evidence"])
        theme.update(
            mentions=len(theme["evidence"]),
            customer_count=len({e["customer"] for e in theme["evidence"]}),
            sentiments={s: counts[s] for s in ("positive", "neutral", "negative")})
        theme["priority"] = priority(theme["sentiments"])
        result["themes"].append(theme)
    return validate("insights", result)


def document_rows(insights, config):
    rows = []
    for theme in insights["themes"]:
        if theme["mentions"] < config.get("min_mentions", 1):
            continue
        source = {key: theme[key] for key in
                  ("theme", "mentions", "customer_count", "action", "priority")}
        source.update(theme["sentiments"])
        source["evidence_ids"] = ", ".join(e["id"] for e in theme["evidence"])
        row = {}
        for column in config.get("columns", DEFAULT_COLUMNS):
            value = source[column["source"]]
            transform = column.get("transform", "identity")
            row[column["label"]] = (
                value.upper() if transform == "upper" else
                value.lower() if transform == "lower" else value)
        rows.append(row)
    return rows


def serialize_document(rows, config):
    if config.get("format", "csv") == "json":
        return json.dumps(rows, ensure_ascii=False, allow_nan=False, sort_keys=True)
    stream = io.StringIO(newline="")
    labels = [c["label"] for c in config.get("columns", DEFAULT_COLUMNS)]
    writer = csv.writer(stream, lineterminator="\n")

    def safe_cell(value):
        # Neutralize spreadsheet formulas, including whitespace-prefixed formulas.
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    writer.writerow([safe_cell(label) for label in labels])
    for row in rows:
        writer.writerow([safe_cell(row[label]) for label in labels])
    return stream.getvalue()


def validate_documents(data, insights, config):
    validate("insights", insights)
    validate_config(config)
    keys(data, {"title", "format", "rows", "content", "checks"})
    require(data["title"] == config.get("title", "Customer feedback action report"),
            "Document title mismatch")
    require(data["format"] == config.get("format", "csv"), "Document format mismatch")
    require(data["rows"] == document_rows(insights, config),
            "Document rows do not match validated insights")
    require(data["content"] == serialize_document(data["rows"], config),
            "Document content does not match rows")
    keys(data["checks"], {"row_count", "source_feedback_count", "included_theme_count",
                         "excluded_theme_count"})
    for field, value in data["checks"].items():
        count(value, field)
    expected = {
        "row_count": len(data["rows"]),
        "source_feedback_count": insights["feedback_count"],
        "included_theme_count": len(data["rows"]),
        "excluded_theme_count": len(insights["themes"]) - len(data["rows"]),
    }
    require(data["checks"] == expected, "Document checks mismatch")
    return data


def validate(kind, value, *, insights=None, config=None):
    """Single public validation entry point used at every pipeline boundary."""
    if kind == "input":
        return validate_input(value)
    if kind == "insights":
        return validate_insights(value)
    if kind == "documents":
        return validate_documents(value, insights, config)
    raise ValidationError("Unknown schema kind")


def document_automation(insights, config):
    validate("insights", insights)
    validate_config(config)
    rows = document_rows(insights, config)
    result = {
        "title": config.get("title", "Customer feedback action report"),
        "format": config.get("format", "csv"),
        "rows": rows,
        "content": serialize_document(rows, config),
        "checks": {
            "row_count": len(rows), "source_feedback_count": insights["feedback_count"],
            "included_theme_count": len(rows),
            "excluded_theme_count": len(insights["themes"]) - len(rows),
        },
    }
    return validate("documents", result, insights=insights, config=config)


def run_pipeline(data):
    validate("input", data)
    insights = customer_insights(data)
    documents = document_automation(insights, data.get("document", {}))
    return {"schema_version": VERSION, "status": "ok",
            "fixture_label": data["fixture_label"],
            "insights": insights, "documents": documents}


def reject_constant(value):
    raise ValidationError("Non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py input.json")
        with Path(argv[0]).open(encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant,
                             object_pairs_hook=unique_object)
        result = run_pipeline(data)
        exit_code = 0
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        result = {"schema_version": VERSION, "status": "error",
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
        exit_code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
