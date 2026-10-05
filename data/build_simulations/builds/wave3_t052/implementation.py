"""Synthetic-friendly, deterministic insights -> documents -> research CLI.

No providers or dependencies. Evidence matching is lexical, not fact verification.
Run: python -B implementation.py example_input.json
"""

import json
import re
import sys
from collections import Counter


VERSION = "1.0"
STOPWORDS = set("a an the is are was were to of for and or in on with do does "
                "should we what how which our customers customer".split())
NEGATIVE = {"slow", "broken", "confusing", "expensive", "bad", "failed", "difficult"}
POSITIVE = {"fast", "easy", "great", "helpful", "good", "love", "excellent"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOPWORDS


def keys(value, required, optional, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(required) <= value.keys(), path + " is missing required keys")
    require(value.keys() <= set(required) | set(optional), path + " has unknown keys")


def rows(value, path):
    require(isinstance(value, list), path + " must be an array")


def identifiers(items, path):
    seen = set()
    for item in items:
        require(isinstance(item, dict), path + " entries must be objects")
        text(item.get("id"), path + ".id")
        require(item["id"] not in seen, path + " contains duplicate ids")
        seen.add(item["id"])
    return seen


def validate(value, stage):
    """One boundary validator used for raw input and every stage handoff."""
    if stage == "input":
        keys(value, ["schema_version", "fixture_label", "feedback", "documents",
                     "questions", "theme_rules", "required_fields"], [], "input")
        require(value["schema_version"] == VERSION, "unsupported schema_version")
        text(value["fixture_label"], "fixture_label")
        for name in ("feedback", "documents", "questions"):
            rows(value[name], name)
            identifiers(value[name], name)
        for item in value["feedback"]:
            keys(item, ["id", "text"], [], "feedback")
            text(item["text"], "feedback.text")
        for item in value["documents"]:
            keys(item, ["id", "text", "feedback_ids"], [], "document")
            text(item["text"], "document.text")
            rows(item["feedback_ids"], "document.feedback_ids")
            require(all(isinstance(x, str) for x in item["feedback_ids"]),
                    "document feedback references must be strings")
            require(len(set(item["feedback_ids"])) == len(item["feedback_ids"]),
                    "duplicate document feedback reference")
            require(set(item["feedback_ids"]) <= {f["id"] for f in value["feedback"]},
                    "unknown feedback reference")
        for item in value["questions"]:
            keys(item, ["id", "text"], [], "question")
            text(item["text"], "question.text")
        require(isinstance(value["theme_rules"], dict), "theme_rules must be an object")
        require("other" not in value["theme_rules"], "'other' is a reserved theme")
        for name, words in value["theme_rules"].items():
            text(name, "theme name")
            rows(words, "theme keywords")
            require(bool(words), "theme keywords must not be empty")
            for word in words:
                text(word, "theme keyword")
                require(bool(tokens(word)), "theme keyword must contain searchable tokens")
        rows(value["required_fields"], "required_fields")
        for field in value["required_fields"]:
            text(field, "required field")
            require(re.fullmatch(r"[a-z][a-z0-9_]*", field) is not None,
                    "required fields must be normalized snake_case")
        require(len(set(value["required_fields"])) == len(value["required_fields"]),
                "duplicate required field")
        return value

    require(stage in {"insights", "documents", "research"}, "unknown stage")
    keys(value, ["schema_version", "status", "stage", "data"], [], stage)
    require(value["schema_version"] == VERSION and value["status"] == "ok"
            and value["stage"] == stage, "invalid stage envelope")
    data = value["data"]
    fields = {
        "insights": ["context", "feedback", "themes"],
        "documents": ["context", "feedback", "themes", "records"],
        "research": ["context", "feedback", "themes", "records", "answers"],
    }
    keys(data, fields[stage], [], stage + ".data")
    validate(data["context"], "input")
    # Recompute deterministic structures so malformed or stale handoffs cannot pass.
    expected_feedback, expected_themes = classify(data["context"])
    require(data["feedback"] == expected_feedback and data["themes"] == expected_themes,
            "invalid insight provenance or aggregation")
    if stage in {"documents", "research"}:
        require(data["records"] == reshape(data), "invalid document provenance or extraction")
    if stage == "research":
        require(data["answers"] == answer_questions(data), "invalid research evidence")
    return value


def envelope(stage, data):
    return validate({"schema_version": VERSION, "status": "ok",
                     "stage": stage, "data": data}, stage)


def classify(context):
    feedback = []
    grouped = {}
    for item in context["feedback"]:
        words = tokens(item["text"])
        themes = sorted(name for name, rules in context["theme_rules"].items()
                        if any(tokens(rule) <= words for rule in rules)) or ["other"]
        score = len(words & POSITIVE) - len(words & NEGATIVE)
        sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        feedback.append({**item, "themes": themes, "sentiment": sentiment})
        for theme in themes:
            grouped.setdefault(theme, []).append(feedback[-1])
    themes = []
    for name, items in grouped.items():
        counts = Counter(item["sentiment"] for item in items)
        themes.append({
            "name": name, "count": len(items),
            "feedback_ids": [item["id"] for item in items],
            "sentiment_counts": {s: counts[s] for s in ("positive", "neutral", "negative")},
            "priority": "review" if counts["negative"] else "monitor",
            "suggested_action": ("Investigate reported friction in " if counts["negative"]
                                 else "Monitor feedback about ") + name,
        })
    themes.sort(key=lambda t: (-t["sentiment_counts"]["negative"], -t["count"], t["name"]))
    return feedback, themes


def insights(raw):
    context = validate(raw, "input")
    # JSON copy prevents later input mutation from silently changing provenance.
    context = json.loads(json.dumps(context))
    feedback, themes = classify(context)
    return envelope("insights", {"context": context, "feedback": feedback, "themes": themes})


def extract_fields(body):
    fields = {}
    issues = []
    for line in body.splitlines():
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9 _-]*)\s*:\s*(.*?)\s*", line)
        if not match:
            continue
        name = re.sub(r"[\s-]+", "_", match[1].strip().lower())
        if name in fields:
            issues.append("duplicate:" + name)
        else:
            fields[name] = match[2]
    return fields, issues


def reshape(data):
    by_id = {item["id"]: item for item in data["feedback"]}
    records = []
    for item in data["context"]["documents"]:
        fields, issues = extract_fields(item["text"])
        issues.extend("missing:" + name for name in data["context"]["required_fields"]
                      if not fields.get(name))
        related = [by_id[key] for key in item["feedback_ids"]]
        records.append({
            "id": item["id"], "text": item["text"], "fields": fields,
            "feedback_ids": list(item["feedback_ids"]),
            "themes": sorted({theme for row in related for theme in row["themes"]}),
            "insight_negative_count": sum(row["sentiment"] == "negative" for row in related),
            "check_status": "needs_review" if issues else "valid",
            "issues": issues,
        })
    return records


def documents(previous):
    data = validate(previous, "insights")["data"]
    return envelope("documents", {**data, "records": reshape(data)})


def answer_questions(data):
    answers = []
    for question in data["context"]["questions"]:
        terms = tokens(question["text"])
        evidence = []
        for record in data["records"]:
            if record["check_status"] != "valid":
                continue
            overlap = terms & tokens(record["text"])
            if overlap:
                evidence.append({
                    "document_id": record["id"], "excerpt": record["text"],
                    "matched_terms": sorted(overlap), "score": len(overlap),
                    "themes": list(record["themes"]),
                    "feedback_ids": list(record["feedback_ids"]),
                    "insight_negative_count": record["insight_negative_count"],
                })
        evidence.sort(key=lambda row: (-row["score"], row["document_id"]))
        themes = sorted({t for row in evidence for t in row["themes"]})
        answers.append({
            "id": question["id"], "question": question["text"],
            "evidence_status": "evidence_found" if evidence else "insufficient_evidence",
            "summary": (f"{len(evidence)} checked document(s) contain matching terms; "
                        "review cited excerpts before deciding." if evidence else
                        "No checked document contains matching terms; collect more evidence."),
            "evidence": evidence, "related_themes": themes,
            "suggested_actions": [theme["suggested_action"] for theme in data["themes"]
                                  if theme["name"] in themes],
            "limitations": ["Lexical matches do not establish truth, causality, or consensus.",
                            "Sentiment and themes are keyword heuristics.",
                            "Unchecked documents are excluded, not repaired or verified."],
        })
    return answers


def research(previous):
    data = validate(previous, "documents")["data"]
    return envelope("research", {**data, "answers": answer_questions(data)})


def run(raw):
    return research(documents(insights(raw)))


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            raw = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run(raw)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
