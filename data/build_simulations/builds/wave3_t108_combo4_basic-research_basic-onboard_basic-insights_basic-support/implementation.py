"""Offline synthetic reference pipeline. Run with one input JSON path."""

import copy
import json
import re
import sys
from pathlib import Path


ORDER = ("research", "onboard", "insights", "support")
STOPWORDS = set(
    "a an the to for of on in is are be and or how what can do does i my "
    "we our you your with please should want need about it this that".split()
)
THEMES = {
    "billing": {"bill", "billing", "price", "pricing", "payment", "refund"},
    "onboarding": {"onboard", "onboarding", "setup", "start", "getting"},
    "reliability": {"crash", "broken", "error", "outage", "slow"},
    "usability": {"confusing", "difficult", "navigation", "interface", "usability"},
    "features": {"feature", "integration", "export", "missing"},
}
NEGATIVE = {"broken", "confusing", "difficult", "crash", "slow", "missing", "bad"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(expected), f"{path} has missing or unknown fields")


def text(value, path, maximum=2000):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonblank text")
    require(len(value) <= maximum, f"{path} is too long")


def records(value, expected, path, minimum=0, maximum_text=2000):
    require(isinstance(value, list) and minimum <= len(value) <= 100,
            f"{path} must be a list of {minimum}..100 records")
    ids = set()
    for index, item in enumerate(value):
        fields(item, expected, f"{path}[{index}]")
        for key in expected:
            text(item[key], f"{path}[{index}].{key}", maximum_text)
        require(item["id"] not in ids, f"{path} contains duplicate ids")
        ids.add(item["id"])


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOPWORDS


def theme_for(value):
    words = tokens(value)
    scores = [(len(words & hints), name) for name, hints in THEMES.items()]
    score, name = sorted(scores, key=lambda pair: (-pair[0], pair[1]))[0]
    return name if score else "other"


def source_matches(question, sources):
    ranked = []
    for source in sources:
        score = len(tokens(question) & tokens(source["title"] + " " + source["content"]))
        if score:
            ranked.append((score, source["id"], source))
    return [source for _, _, source in sorted(ranked, key=lambda row: (-row[0], row[1]))[:3]]


def validate(document, completed=0):
    """One validation boundary for input and every accumulated stage envelope."""
    require(type(completed) is int and 0 <= completed <= 4, "invalid stage boundary")
    fields(document, ("schema_version", "status", "synthetic", "input", "stages"), "document")
    require(document["schema_version"] == "1.0", "unsupported schema_version")
    require(document["status"] == "ok", "status must be ok")
    require(type(document["synthetic"]) is bool, "synthetic must be boolean")
    data = document["input"]
    fields(data, ("questions", "sources", "customer", "feedback", "tickets"), "input")
    records(data["questions"], ("id", "text"), "questions", 1)
    records(data["sources"], ("id", "title", "content"), "sources")
    fields(data["customer"], ("id", "name", "goal", "experience"), "customer")
    for key, value in data["customer"].items():
        text(value, f"customer.{key}")
    require(data["customer"]["experience"] in ("new", "experienced"), "invalid experience")
    records(data["feedback"], ("id", "customer_id", "text"), "feedback")
    records(data["tickets"], ("id", "customer_id", "question"), "tickets")
    customer_id = data["customer"]["id"]
    for item in data["feedback"] + data["tickets"]:
        require(item["customer_id"] == customer_id, "unknown customer_id")
    stages = document["stages"]
    fields(stages, ORDER[:completed], "stages")
    sources = {source["id"]: source for source in data["sources"]}
    question_ids = [question["id"] for question in data["questions"]]
    evidence_ids = set()
    if completed >= 1:
        research_output = stages["research"]
        fields(research_output, ("findings", "source_count"), "research")
        require(type(research_output["source_count"]) is int
                and research_output["source_count"] == len(sources), "invalid source_count")
        findings = research_output["findings"]
        require(isinstance(findings, list) and len(findings) == len(question_ids),
                "findings must cover every question")
        for question, finding in zip(data["questions"], findings):
            fields(finding, ("question_id", "status", "evidence", "decision"), "finding")
            require(finding["question_id"] == question["id"], "invalid finding question reference")
            evidence = finding["evidence"]
            require(isinstance(evidence, list), "evidence must be a list")
            expected_sources = source_matches(question["text"], data["sources"])
            require(len(evidence) == len(expected_sources), "invalid evidence coverage")
            for item, source in zip(evidence, expected_sources):
                fields(item, ("source_id", "excerpt"), "evidence")
                require(item["source_id"] == source["id"] and item["excerpt"] == source["content"],
                        "evidence must quote the matched source exactly")
                evidence_ids.add(item["source_id"])
            require(finding["status"] == ("supported" if evidence else "insufficient_evidence"),
                    "invalid finding status")
            expected_decision = ("Review cited evidence before deciding." if evidence
                                 else "Collect a source that addresses this question.")
            require(finding["decision"] == expected_decision, "invalid research decision")
    if completed >= 2:
        onboard_output = stages["onboard"]
        fields(onboard_output, ("customer_id", "goal", "research_question_id", "source_ids",
                                "steps", "next_step_id", "needs_assistance"), "onboard")
        require(onboard_output["customer_id"] == customer_id
                and onboard_output["goal"] == data["customer"]["goal"], "invalid onboarding customer")
        require(onboard_output["research_question_id"] in question_ids, "unknown research question")
        expected_question = sorted(data["questions"], key=lambda q: (
            -len(tokens(q["text"]) & tokens(data["customer"]["goal"])), q["id"]))[0]
        require(onboard_output["research_question_id"] == expected_question["id"],
                "onboarding question must match the customer goal")
        selected = next(f for f in stages["research"]["findings"]
                        if f["question_id"] == onboard_output["research_question_id"])
        require(onboard_output["source_ids"] == [e["source_id"] for e in selected["evidence"]],
                "onboarding evidence must propagate from research")
        expected_assistance = data["customer"]["experience"] == "new" or not selected["evidence"]
        require(type(onboard_output["needs_assistance"]) is bool
                and onboard_output["needs_assistance"] == expected_assistance, "invalid assistance flag")
        records(onboard_output["steps"], ("id", "action"), "onboarding steps", 1, 12000)
        require(len(onboard_output["steps"]) == 2
                and [step["id"] for step in onboard_output["steps"]] == ["review", "try"],
                "invalid onboarding steps")
        review = ("Review sources " + ", ".join(onboard_output["source_ids"])
                  if onboard_output["source_ids"]
                  else "Ask your onboarding contact for verified guidance")
        require(onboard_output["steps"][0]["action"] == f"{data['customer']['name']}: {review}."
                and onboard_output["steps"][1]["action"] ==
                "Try a small, reversible task toward: " + data["customer"]["goal"],
                "invalid personalized onboarding actions")
        require(onboard_output["next_step_id"] == "review", "invalid next step")
    if completed >= 3:
        insights_output = stages["insights"]
        fields(insights_output, ("customer_id", "onboarding_next_step_id", "themes",
                                 "feedback_count"), "insights")
        require(insights_output["customer_id"] == customer_id, "invalid insights customer")
        require(insights_output["onboarding_next_step_id"] == stages["onboard"]["next_step_id"],
                "insights must consume onboarding next step")
        require(type(insights_output["feedback_count"]) is int
                and insights_output["feedback_count"] == len(data["feedback"]), "invalid feedback count")
        require(isinstance(insights_output["themes"], list), "themes must be a list")
        expected_groups = {}
        for item in data["feedback"]:
            expected_groups.setdefault(theme_for(item["text"]), []).append(item)
        require(len(insights_output["themes"]) == len(expected_groups), "invalid theme coverage")
        for theme, name in zip(insights_output["themes"], sorted(expected_groups)):
            fields(theme, ("name", "feedback_ids", "count", "negative_count", "action",
                           "onboarding_step_id"), "theme")
            group = expected_groups[name]
            require(theme["name"] == name and theme["feedback_ids"] == [f["id"] for f in group],
                    "invalid feedback attribution")
            require(type(theme["count"]) is int and theme["count"] == len(group), "invalid theme count")
            negatives = sum(bool(tokens(f["text"]) & NEGATIVE) for f in group)
            require(type(theme["negative_count"]) is int and theme["negative_count"] == negatives,
                    "invalid negative count")
            expected_step = stages["onboard"]["next_step_id"] if name in ("onboarding", "usability") else None
            require(theme["onboarding_step_id"] == expected_step, "invalid theme onboarding reference")
            expected_action = ("Clarify onboarding step " + stages["onboard"]["next_step_id"]
                               if expected_step else "Review " + name + " feedback with the responsible team")
            require(theme["action"] == expected_action, "invalid theme action")
    if completed >= 4:
        support_output = stages["support"]
        fields(support_output, ("customer_id", "responses"), "support")
        require(support_output["customer_id"] == customer_id, "invalid support customer")
        responses = support_output["responses"]
        require(isinstance(responses, list) and len(responses) == len(data["tickets"]),
                "responses must cover all tickets")
        available = [s for s in data["sources"] if s["id"] in evidence_ids]
        known_themes = {t["name"] for t in stages["insights"]["themes"]}
        for ticket, response in zip(data["tickets"], responses):
            fields(response, ("ticket_id", "status", "answer", "citations", "theme",
                              "next_step_id", "follow_up"), "response")
            require(response["ticket_id"] == ticket["id"], "invalid ticket reference")
            matching = source_matches(ticket["question"], available)
            require(response["citations"] == [s["id"] for s in matching], "invalid support citations")
            expected_answer = "\n\n".join(s["content"] for s in matching) if matching else (
                "I do not have evidence to answer this question. A human review is required.")
            require(response["answer"] == expected_answer, "support answer is not grounded")
            require(response["status"] == ("answered" if matching else "escalated"),
                    "invalid support disposition")
            ticket_theme = theme_for(ticket["question"])
            require(response["theme"] == (ticket_theme if ticket_theme in known_themes else None),
                    "invalid insights theme propagation")
            require(response["next_step_id"] == stages["insights"]["onboarding_next_step_id"],
                    "invalid support next step")
            known = next((t for t in stages["insights"]["themes"] if t["name"] == ticket_theme), None)
            expected_follow_up = (known["action"] if known else
                                  "Continue onboarding step " + stages["insights"]["onboarding_next_step_id"])
            require(response["follow_up"] == expected_follow_up, "invalid support follow-up")
    return document


def complete(document, index, output):
    result = copy.deepcopy(document)
    result["stages"][ORDER[index]] = output
    return validate(result, index + 1)


def research(document):
    validate(document, 0)
    data = document["input"]
    findings = []
    for question in data["questions"]:
        sources = source_matches(question["text"], data["sources"])
        findings.append({
            "question_id": question["id"],
            "status": "supported" if sources else "insufficient_evidence",
            "evidence": [{"source_id": s["id"], "excerpt": s["content"]} for s in sources],
            "decision": ("Review cited evidence before deciding." if sources
                         else "Collect a source that addresses this question."),
        })
    return complete(document, 0, {"findings": findings, "source_count": len(data["sources"])})


def onboard(document):
    validate(document, 1)
    data = document["input"]
    customer = data["customer"]
    question = sorted(data["questions"], key=lambda q: (
        -len(tokens(q["text"]) & tokens(customer["goal"])), q["id"]))[0]
    finding = next(f for f in document["stages"]["research"]["findings"]
                   if f["question_id"] == question["id"])
    references = [e["source_id"] for e in finding["evidence"]]
    review_action = ("Review sources " + ", ".join(references) if references
                     else "Ask your onboarding contact for verified guidance")
    return complete(document, 1, {
        "customer_id": customer["id"],
        "goal": customer["goal"],
        "research_question_id": question["id"],
        "source_ids": references,
        "steps": [{"id": "review", "action": f"{customer['name']}: {review_action}."},
                  {"id": "try", "action": "Try a small, reversible task toward: " + customer["goal"]}],
        "next_step_id": "review",
        "needs_assistance": customer["experience"] == "new" or not references,
    })


def insights(document):
    validate(document, 2)
    prior = document["stages"]["onboard"]
    groups = {}
    for feedback in document["input"]["feedback"]:
        groups.setdefault(theme_for(feedback["text"]), []).append(feedback)
    themes = []
    for name, group in sorted(groups.items()):
        onboarding_related = name in ("onboarding", "usability")
        themes.append({
            "name": name,
            "feedback_ids": [item["id"] for item in group],
            "count": len(group),
            "negative_count": sum(bool(tokens(item["text"]) & NEGATIVE) for item in group),
            "action": ("Clarify onboarding step " + prior["next_step_id"] if onboarding_related
                       else "Review " + name + " feedback with the responsible team"),
            "onboarding_step_id": prior["next_step_id"] if onboarding_related else None,
        })
    return complete(document, 2, {
        "customer_id": prior["customer_id"],
        "onboarding_next_step_id": prior["next_step_id"],
        "feedback_count": len(document["input"]["feedback"]),
        "themes": themes,
    })


def support(document):
    validate(document, 3)
    prior = document["stages"]["insights"]
    evidence_ids = {e["source_id"] for f in document["stages"]["research"]["findings"]
                    for e in f["evidence"]}
    available = [s for s in document["input"]["sources"] if s["id"] in evidence_ids]
    themes = {theme["name"]: theme for theme in prior["themes"]}
    responses = []
    for ticket in document["input"]["tickets"]:
        matches = source_matches(ticket["question"], available)
        theme = theme_for(ticket["question"])
        known = themes.get(theme)
        responses.append({
            "ticket_id": ticket["id"],
            "status": "answered" if matches else "escalated",
            "answer": "\n\n".join(s["content"] for s in matches) if matches else (
                "I do not have evidence to answer this question. A human review is required."),
            "citations": [s["id"] for s in matches],
            "theme": theme if known else None,
            "next_step_id": prior["onboarding_next_step_id"],
            "follow_up": known["action"] if known else "Continue onboarding step " + prior["onboarding_next_step_id"],
        })
    return complete(document, 3, {"customer_id": prior["customer_id"], "responses": responses})


def run_pipeline(document):
    current = document
    for stage in (research, onboard, insights, support):
        current = stage(current)
    return current


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_text(encoding="utf-8")
        document = json.loads(raw, object_pairs_hook=unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValidationError("non-finite JSON number")))
        result = run_pipeline(document)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
