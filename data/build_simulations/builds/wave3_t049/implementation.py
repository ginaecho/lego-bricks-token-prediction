"""Offline, deterministic customer support backed only by caller-provided articles.

Run: python -B implementation.py example_input.json
No services are called and no conversations or handoffs are persisted.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


STOP_WORDS = frozenset(
    "a an and are as at be can could do does for from how i in is it me my "
    "of on or please the to what when where with would you your".split()
)
MAX_FILE_BYTES = 1_000_000


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name, limit):
    require(isinstance(value, str), f"{name} must be a string")
    require(bool(value.strip()), f"{name} must not be blank")
    require(len(value) <= limit, f"{name} exceeds {limit} characters")


def validate(value, kind="input", source=None):
    """One validation boundary for requests and responses."""
    require(isinstance(value, dict), f"{kind} must be an object")
    if kind == "input":
        required = {"request_id", "message", "knowledge_base", "team_online"}
        allowed = required | {"consent_contact", "fixture_label"}
        require(required <= value.keys(), "Missing required input fields")
        require(value.keys() <= allowed, "Unknown input fields")
        text(value["request_id"], "request_id", 100)
        text(value["message"], "message", 4000)
        require(type(value["team_online"]) is bool, "team_online must be boolean")
        require(type(value.get("consent_contact", False)) is bool,
                "consent_contact must be boolean")
        if "fixture_label" in value:
            text(value["fixture_label"], "fixture_label", 200)
        articles = value["knowledge_base"]
        require(isinstance(articles, list), "knowledge_base must be an array")
        require(len(articles) <= 100, "knowledge_base supports at most 100 articles")
        identifiers = set()
        for index, article in enumerate(articles):
            require(isinstance(article, dict), f"Article {index} must be an object")
            require(set(article) == {"id", "title", "keywords", "answer"},
                    f"Article {index} has missing or unknown fields")
            for field, limit in (("id", 100), ("title", 200), ("answer", 8000)):
                text(article[field], f"Article {index}.{field}", limit)
            require(article["id"] == article["id"].strip(),
                    "Article IDs must not have surrounding whitespace")
            require(article["id"] not in identifiers, "Duplicate article id")
            identifiers.add(article["id"])
            keywords = article["keywords"]
            require(isinstance(keywords, list) and len(keywords) <= 30,
                    "keywords must be an array of at most 30 strings")
            for keyword in keywords:
                text(keyword, "keyword", 100)
        return value

    require(kind == "output", "Unknown schema kind")
    require(set(value) == {
        "status", "request_id", "disposition", "answer", "citations",
        "team_online", "handoff", "notice",
    }, "Invalid output fields")
    require(value["status"] == "ok", "Invalid success status")
    text(value["answer"], "answer", 9000)
    require(value["disposition"] in {"answered", "needs_clarification", "human_requested"},
            "Invalid disposition")
    require(value["handoff"] in {"not_needed", "consent_required", "suggested"},
            "Invalid handoff")
    require(isinstance(value["notice"], str), "notice must be a string")
    require(source is not None, "Output validation requires source input")
    require(value["request_id"] == source["request_id"], "Request identity mismatch")
    require(type(value["team_online"]) is bool
            and value["team_online"] == source["team_online"], "Availability mismatch")
    citations = value["citations"]
    require(isinstance(citations, list), "citations must be an array")
    by_id = {article["id"]: article for article in source["knowledge_base"]}
    if value["disposition"] == "answered":
        require(len(citations) == 1, "Answers require one citation")
        citation = citations[0]
        require(isinstance(citation, dict) and set(citation) == {"id", "title"},
                "Invalid citation")
        require(isinstance(citation["id"], str) and citation["id"] in by_id,
                "Unknown citation")
        article = by_id[citation["id"]]
        require(citation["title"] == article["title"], "Citation title mismatch")
        require(value["answer"] == article["answer"], "Answer must exactly quote its source")
        require(value["handoff"] == "not_needed", "Answered requests need no handoff")
    else:
        require(not citations, "Fallbacks must not claim citations")
        expected = "suggested" if source.get("consent_contact", False) else "consent_required"
        require(value["handoff"] == expected, "Handoff must respect consent")
    return value


def tokens(message):
    return set(re.findall(r"[^\W_]+", message.casefold(), flags=re.UNICODE)) - STOP_WORDS


def support(payload):
    request = validate(payload)
    query = tokens(request["message"])
    human_requested = bool(re.search(
        r"\b(?:speak|talk|connect)\s+(?:me\s+)?(?:to|with)\s+(?:a\s+)?"
        r"(?:human|person|agent|representative)\b|\b(?:human|live)\s+agent\b",
        request["message"], flags=re.IGNORECASE,
    ))
    ranked = []
    for article in request["knowledge_base"]:
        searchable = tokens(article["title"] + " " + " ".join(article["keywords"]))
        overlap = len(query & searchable)
        coverage = overlap / len(query) if query else 0.0
        if overlap and coverage >= 0.5:
            ranked.append((coverage, overlap, article["id"], article))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    ambiguous = len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]
    selected = ranked[0][3] if ranked and not ambiguous and not human_requested else None
    if selected:
        disposition = "answered"
        answer = selected["answer"]
        citations = [{"id": selected["id"], "title": selected["title"]}]
        handoff = "not_needed"
    else:
        disposition = "human_requested" if human_requested else "needs_clarification"
        if human_requested:
            answer = "You asked for a human. Please use your organization's support channel."
        elif ambiguous:
            answer = "More than one help article may apply. Please clarify the product or topic."
        else:
            answer = ("I don't have enough information in the supplied knowledge base to answer "
                      "confidently. Please describe the product, issue, and any error message. "
                      "Do not share passwords or payment details.")
        citations = []
        handoff = "suggested" if request.get("consent_contact", False) else "consent_required"
    notice = ("The support team is online." if request["team_online"]
              else "The support team is offline; knowledge-base help remains available.")
    if handoff != "not_needed":
        notice += (" Contact consent is required before sharing this conversation."
                   if handoff == "consent_required"
                   else " A human follow-up is suggested.")
        notice += " No ticket has been created or sent, and no response time is promised."
    result = {
        "status": "ok",
        "request_id": request["request_id"],
        "disposition": disposition,
        "answer": answer,
        "citations": citations,
        "team_online": request["team_online"],
        "handoff": handoff,
        "notice": notice,
    }
    return validate(result, "output", request)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"Non-finite JSON value is not allowed: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
        require(len(raw) <= MAX_FILE_BYTES, "Input file exceeds 1000000 bytes")
        payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = support(payload)
    except (OSError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": {
            "code": "invalid_input_or_file", "message": str(exc),
        }}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
