"""Deterministic synthetic feedback-to-support pipeline; Python standard library."""

import json
import re
import sys
import unicodedata
from pathlib import Path


SCHEMA_VERSION = "1.0"
THEMES = {
    "delivery": {"delivery", "shipping", "shipment", "late", "tracking", "arrive"},
    "refunds": {"refund", "refunds", "return", "returns", "reimbursement"},
    "product": {"broken", "defect", "defective", "quality", "damaged"},
    "support": {"support", "help", "agent", "response", "contact"},
}
STOPWORDS = {
    "a", "an", "the", "i", "my", "me", "we", "our", "you", "your", "is",
    "are", "was", "were", "be", "to", "of", "for", "and", "or", "in", "on",
    "it", "this", "that", "can", "could", "do", "does", "how", "what", "please",
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def normalized(text):
    # Ignore punctuation/case/spacing, not wording: no fuzzy semantic merging.
    return " ".join(words(text)) or unicodedata.normalize("NFKC", text).casefold().strip()


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), location + " must be nonblank text")
    require(len(value) <= 10000, location + " exceeds 10000 characters")


def keys(value, expected, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(expected), location + " has missing or unknown fields")


def records(value, expected, location):
    require(isinstance(value, list) and len(value) <= 1000, location + " must be a list of at most 1000 items")
    seen = set()
    for item in value:
        keys(item, expected, location + " item")
        text(item["id"], location + ".id")
        require(item["id"] not in seen, location + " contains duplicate IDs")
        seen.add(item["id"])
    return seen


def validate(stage, value, context=None):
    """One validation boundary for the input and both cross-stage schemas."""
    if stage == "input":
        keys(value, {"schema_version", "data_label", "team_online", "feedback", "knowledge_base", "questions"}, stage)
        require(value["schema_version"] == SCHEMA_VERSION, "Unsupported schema_version")
        require(value["data_label"] == "synthetic", "data_label must be synthetic")
        require(type(value["team_online"]) is bool, "team_online must be a boolean")
        records(value["feedback"], {"id", "text"}, "feedback")
        records(value["questions"], {"id", "text"}, "questions")
        records(value["knowledge_base"], {"id", "title", "text", "keywords"}, "knowledge_base")
        for collection in ("feedback", "questions", "knowledge_base"):
            for item in value[collection]:
                text(item["text"], collection + ".text")
                if collection == "knowledge_base":
                    text(item["title"], "knowledge_base.title")
                    require(isinstance(item["keywords"], list) and len(item["keywords"]) <= 100,
                            "keywords must be a list of at most 100 strings")
                    for keyword in item["keywords"]:
                        text(keyword, "keyword")
    elif stage == "feedback":
        keys(value, {"source_count", "unique_count", "duplicate_count", "groups", "themes"}, stage)
        source = {item["id"]: item["text"] for item in context["feedback"]}
        group_ids = records(value["groups"], {"id", "source_ids", "text", "themes"}, "groups")
        seen_sources = []
        for group in value["groups"]:
            require(isinstance(group["source_ids"], list) and bool(group["source_ids"]), "Empty source group")
            require(all(isinstance(i, str) and i in source for i in group["source_ids"]), "Unknown feedback source")
            require(group["id"] == group["source_ids"][0], "Canonical ID must be first source")
            require(group["text"] == source[group["id"]], "Canonical text must preserve source")
            require(all(normalized(source[i]) == normalized(group["text"]) for i in group["source_ids"]),
                    "Non-equivalent feedback merged")
            require(group["themes"] == classify(group["text"]), "Invalid group classification")
            seen_sources.extend(group["source_ids"])
        require(sorted(seen_sources) == sorted(source), "Every feedback source must occur exactly once")
        require(len({normalized(g["text"]) for g in value["groups"]}) == len(group_ids), "Undeduplicated groups")
        require(type(value["source_count"]) is int and value["source_count"] == len(source), "Invalid source count")
        require(type(value["unique_count"]) is int and value["unique_count"] == len(group_ids), "Invalid unique count")
        require(type(value["duplicate_count"]) is int and value["duplicate_count"] == len(source) - len(group_ids),
                "Invalid duplicate count")
        require(value["themes"] == summarize_themes(value["groups"]), "Invalid theme counts or supporting excerpts")
    elif stage == "support":
        keys(value, {"team_online", "responses"}, stage)
        raw, analysis = context
        require(type(value["team_online"]) is bool and value["team_online"] == raw["team_online"], "Invalid availability")
        records(value["responses"],
                {"id", "question", "status", "answer", "citations", "related_themes", "supporting_feedback", "handoff"},
                "responses")
        # Re-derive the deterministic contract, checking grounding and propagation together.
        expected = [answer_question(q, raw, analysis) for q in raw["questions"]]
        require(value["responses"] == expected, "Support response violates grounding or feedback handoff")
    else:
        raise ValidationError("Unknown validation stage")
    return value


def classify(content):
    tokens = set(words(content))
    return sorted(name for name, keywords in THEMES.items() if tokens & keywords) or ["general"]


def summarize_themes(groups):
    result = []
    for theme in sorted({t for group in groups for t in group["themes"]}):
        selected = [group for group in groups if theme in group["themes"]]
        result.append({
            "name": theme,
            "unique_feedback_count": len(selected),
            "source_feedback_count": sum(len(g["source_ids"]) for g in selected),
            "excerpts": [
                {"group_id": g["id"], "source_ids": list(g["source_ids"]), "excerpt": g["text"][:240]}
                for g in selected
            ],
        })
    return result


def analyze_feedback(raw):
    groups = {}
    for item in raw["feedback"]:
        key = normalized(item["text"])
        if key in groups:
            groups[key]["source_ids"].append(item["id"])
        else:
            groups[key] = {"id": item["id"], "source_ids": [item["id"]],
                           "text": item["text"], "themes": classify(item["text"])}
    unique = list(groups.values())
    return {
        "source_count": len(raw["feedback"]),
        "unique_count": len(unique),
        "duplicate_count": len(raw["feedback"]) - len(unique),
        "groups": unique,
        "themes": summarize_themes(unique),
    }


def answer_question(question, raw, analysis):
    tokens = set(words(question["text"])) - STOPWORDS
    ranked = []
    for article in raw["knowledge_base"]:
        # Explicit author-supplied keywords are the retrieval gate; body words are not.
        keyword_tokens = set(words(" ".join(article["keywords"]))) - STOPWORDS
        score = len(tokens & keyword_tokens)
        if score:
            ranked.append((-score, article["id"], article))
    chosen = [entry[2] for entry in sorted(ranked)[:2]]
    question_themes = set(classify(question["text"])) - {"general"}
    group_map = {group["id"]: group for group in analysis["groups"]}
    related = []
    evidence = {}
    for theme in analysis["themes"]:
        excerpts = [
            excerpt for excerpt in theme["excerpts"]
            if theme["name"] in question_themes
            or tokens & (set(words(group_map[excerpt["group_id"]]["text"])) - STOPWORDS)
        ]
        if excerpts:
            related.append(theme["name"])
            for excerpt in excerpts:
                evidence[excerpt["group_id"]] = excerpt
    citations = [{"id": article["id"], "title": article["title"], "excerpt": article["text"]} for article in chosen]
    if chosen:
        answer = "Here is the relevant published guidance:\n" + "\n".join(
            f'[{article["id"]}] {article["text"]}' for article in chosen)
        status = "grounded_guidance"
        handoff = None
    else:
        answer = ("I don't have published guidance matching this question. "
                  "Please contact support with the issue details; do not send passwords or payment details.")
        status = "needs_human"
        handoff = {"reason": "No matching published guidance", "created": False}
    if not raw["team_online"]:
        answer += "\nThe team is currently offline. No response time is promised."
    if related:
        answer += "\nRelated customer feedback is listed separately; it is not verified policy."
    return {
        "id": question["id"], "question": question["text"], "status": status, "answer": answer,
        "citations": citations, "related_themes": related, "supporting_feedback": list(evidence.values()),
        "handoff": handoff,
    }


def provide_support(raw, analysis):
    validate("feedback", analysis, raw)
    return {"team_online": raw["team_online"],
            "responses": [answer_question(q, raw, analysis) for q in raw["questions"]]}


def run_pipeline(raw):
    validate("input", raw)
    analysis = validate("feedback", analyze_feedback(raw), raw)
    support = validate("support", provide_support(raw, analysis), (raw, analysis))
    return {"schema_version": SCHEMA_VERSION, "data_label": raw["data_label"], "status": "ok",
            "feedback_analysis": analysis, "support": support}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open(encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=unique_object,
                            parse_constant=lambda value: reject_constant(value))
        output = run_pipeline(raw)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True))
    return 0


def reject_constant(value):
    raise ValidationError("Non-finite JSON number is not allowed: " + value)


if __name__ == "__main__":
    sys.exit(main())
