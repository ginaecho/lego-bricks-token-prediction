"""Synthetic, deterministic onboarding -> insights -> evidence CLI.

Usage: python -B implementation.py example_input.json
Only the Python standard library is required. Research synthesizes supplied
structured claims, not external facts; citations are traceable but not verified.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, path, message):
    if not condition:
        raise ValidationError(f"{path}: {message}")


def obj(value, path, keys):
    require(isinstance(value, dict), path, "must be an object")
    require(set(value) == set(keys), path,
            "expected exactly fields " + ", ".join(sorted(keys)))


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path,
            "must be a nonempty string")


def array(value, path):
    require(isinstance(value, list), path, "must be an array")


def unique_rows(rows, path, fields):
    array(rows, path)
    ids = set()
    for index, row in enumerate(rows):
        where = f"{path}[{index}]"
        obj(row, where, fields)
        text(row["id"], where + ".id")
        require(row["id"] not in ids, where + ".id", "duplicate identifier")
        ids.add(row["id"])
    return ids


def strings(values, path):
    array(values, path)
    for value in values:
        text(value, path)
    require(len(set(values)) == len(values), path, "duplicate reference")


SEVERITIES = {"low": 10, "medium": 40, "high": 70, "critical": 100}
POSITIONS = ("support", "oppose", "neutral")
LEXICON = {
    "good": 1, "great": 1, "excellent": 1, "helpful": 1, "easy": 1,
    "love": 1, "happy": 1, "reliable": 1,
    "bad": -1, "broken": -1, "confusing": -1, "slow": -1,
    "hate": -1, "awful": -1, "frustrating": -1, "unhappy": -1,
}


def validate_input(data):
    obj(data, "$", ("schema_version", "synthetic", "prerequisites", "steps",
                    "feedback", "documents", "research_questions"))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "$.schema_version", "must be integer 1")
    require(data["synthetic"] is True, "$.synthetic",
            "this reference implementation requires synthetic fixtures")
    prerequisites = data["prerequisites"]
    require(isinstance(prerequisites, dict), "$.prerequisites", "must be an object")
    for key, value in prerequisites.items():
        text(key, "$.prerequisites key")
        require(type(value) is bool, "$.prerequisites." + key, "must be boolean")
    ids = unique_rows(data["steps"], "$.steps",
                      ("id", "title", "prerequisites", "depends_on", "completed"))
    for step in data["steps"]:
        where = "$.steps." + step["id"]
        text(step["title"], where + ".title")
        strings(step["prerequisites"], where + ".prerequisites")
        strings(step["depends_on"], where + ".depends_on")
        require(set(step["prerequisites"]) <= set(prerequisites), where,
                "unknown prerequisite")
        require(set(step["depends_on"]) <= ids, where, "unknown step dependency")
        require(type(step["completed"]) is bool, where + ".completed",
                "must be boolean")
    steps = {step["id"]: step for step in data["steps"]}
    # Iterative topological validation also supports large, non-recursive chains.
    remaining = set(steps)
    visited = set()
    while remaining:
        ready = {sid for sid in remaining
                 if set(steps[sid]["depends_on"]) <= visited}
        require(bool(ready), "$.steps", "dependency cycle")
        remaining -= ready
        visited |= ready
    unique_rows(data["feedback"], "$.feedback",
                ("id", "step_id", "topic", "text", "severity"))
    for row in data["feedback"]:
        for field in ("step_id", "topic", "text", "severity"):
            text(row[field], "$.feedback." + field)
        require(row["step_id"] in ids, "$.feedback.step_id", "unknown step")
        require(row["severity"] in SEVERITIES, "$.feedback.severity",
                "expected low, medium, high or critical")
    unique_rows(data["documents"], "$.documents", ("id", "title", "claims"))
    for document in data["documents"]:
        text(document["title"], "$.documents.title")
        array(document["claims"], "$.documents.claims")
        for claim in document["claims"]:
            obj(claim, "$.documents.claims[]", ("topic", "position", "evidence"))
            for key in claim:
                text(claim[key], "$.documents.claims[]." + key)
            require(claim["position"] in POSITIONS,
                    "$.documents.claims[].position",
                    "expected support, oppose or neutral")
    unique_rows(data["research_questions"], "$.research_questions",
                ("id", "topic", "question"))
    topics = {row["topic"] for row in data["feedback"]}
    for question in data["research_questions"]:
        text(question["topic"], "$.research_questions.topic")
        text(question["question"], "$.research_questions.question")
        require(question["topic"] in topics, "$.research_questions.topic",
                "must refer to a feedback topic")
    return data


def validate_handoff(state, stage):
    obj(state, "handoff", ("schema_version", "synthetic", "stage", "input",
                          "guided", "sentiment"))
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "handoff.schema_version", "must be integer 1")
    require(state["synthetic"] is True, "handoff.synthetic", "must be true")
    require(state["stage"] == stage, "handoff.stage", f"expected {stage}")
    validate_input(state["input"])
    # Recompute deterministic contracts to reject altered derived data, including
    # fabricated completion, scores, ranks, or dropped issues at either seam.
    expected_guided = guided_result(state["input"])
    require(state["guided"] == expected_guided, "handoff.guided",
            "does not match validated onboarding")
    expected_sentiment = (sentiment_result(state["input"], expected_guided)
                          if stage == "sentiment" else None)
    require(state["sentiment"] == expected_sentiment, "handoff.sentiment",
            "does not match validated insights")
    return state


def guided_result(data):
    completed = {step["id"] for step in data["steps"] if step["completed"]}
    results = []
    for step in data["steps"]:
        missing = sorted(key for key in step["prerequisites"]
                         if not data["prerequisites"][key])
        waiting = sorted(set(step["depends_on"]) - completed)
        require(not step["completed"] or not (missing or waiting),
                "$.steps." + step["id"] + ".completed",
                "cannot complete a blocked step")
        status = ("complete" if step["completed"] else
                  "blocked" if missing or waiting else "ready")
        results.append({"id": step["id"], "title": step["title"], "status": status,
                        "missing_prerequisites": missing, "waiting_for": waiting})
    total = len(results)
    return {
        "steps": results,
        "completed_step_ids": sorted(completed),
        "progress": {"completed": len(completed), "total": total,
                     "fraction": round(len(completed) / total, 4) if total else 1.0},
        "next_step_ids": [row["id"] for row in results if row["status"] == "ready"],
    }


def guided_stage(data):
    validate_input(data)
    state = {"schema_version": 1, "synthetic": True, "stage": "guided",
             "input": data, "guided": guided_result(data), "sentiment": None}
    return validate_handoff(state, "guided")


def score_sentiment(value):
    tokens = re.findall(r"[a-z]+|[.!?;,]", value.lower())
    matches = []
    recent = []
    for index, token in enumerate(tokens):
        if token in ".!?;,":
            recent = []
            continue
        if token in LEXICON:
            negated = sum(word in ("not", "never", "no")
                          for word in recent[-3:]) % 2 == 1
            weight = LEXICON[token] * (-1 if negated else 1)
            matches.append({"token": token, "token_index": index,
                            "negated": negated, "contribution": weight})
        recent.append(token)
    score = round(sum(match["contribution"] for match in matches) / len(matches), 4) \
        if matches else 0.0
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return {"score": score, "label": label, "matches": matches,
            "coverage": "lexicon_matches" if matches else "no_lexicon_matches"}


def sentiment_result(data, guided):
    completed = set(guided["completed_step_ids"])
    issues = []
    for row in data["feedback"]:
        require(row["step_id"] in completed, "$.feedback." + row["id"],
                "feedback must reference a completed onboarding step")
        sentiment = score_sentiment(row["text"])
        base = SEVERITIES[row["severity"]]
        adjustment = round(max(0, -sentiment["score"]) * 20)
        issues.append({**row, "sentiment": sentiment,
                       "priority_score": base + adjustment,
                       "priority_explanation": {
                           "severity_base": base, "negative_sentiment_bonus": adjustment}})
    issues.sort(key=lambda issue: (-issue["priority_score"], issue["id"]))
    for rank, issue in enumerate(issues, 1):
        issue["rank"] = rank
    return {
        "source_completed_step_ids": guided["completed_step_ids"][:],
        "method": "lexicon mean; three-token negation window; punctuation resets",
        "priority_method": "severity base + round(max(0, -sentiment) * 20); ties by id",
        "lexicon": dict(LEXICON),
        "issues": issues,
    }


def sentiment_stage(state):
    validate_handoff(state, "guided")
    result = {**state, "stage": "sentiment",
              "sentiment": sentiment_result(state["input"], state["guided"])}
    return validate_handoff(result, "sentiment")


def deep_stage(state):
    validate_handoff(state, "sentiment")
    data = state["input"]
    by_topic = {}
    for issue in state["sentiment"]["issues"]:
        by_topic.setdefault(issue["topic"], []).append(issue)
    findings = []
    for topic, issues in by_topic.items():
        evidence = []
        for document in data["documents"]:
            for index, claim in enumerate(document["claims"]):
                if claim["topic"] == topic:
                    evidence.append({
                        "document_id": document["id"], "title": document["title"],
                        "claim_index": index, "position": claim["position"],
                        "quote": claim["evidence"],
                    })
        positions = {item["position"] for item in evidence}
        disagreement = "support" in positions and "oppose" in positions
        document_ids = sorted({item["document_id"] for item in evidence})
        counts = {position: len({item["document_id"] for item in evidence
                                 if item["position"] == position})
                  for position in POSITIONS}
        unresolved = []
        if not evidence:
            unresolved.append("No supplied evidence for this topic.")
        if disagreement:
            unresolved.append("Supporting and opposing claims require reconciliation.")
        if len(document_ids) == 1:
            unresolved.append("Only one document; independent corroboration needed.")
        questions = []
        for question in data["research_questions"]:
            if question["topic"] == topic:
                questions.append({**question, "status": "needs_human_review"})
                unresolved.append(question["question"])
        agreement = ("disagreement" if disagreement else
                     "support" if "support" in positions else
                     "oppose" if "oppose" in positions else
                     "neutral" if evidence else "no_evidence")
        findings.append({
            "topic": topic, "priority_score": max(row["priority_score"] for row in issues),
            "source_issue_ids": [row["id"] for row in issues],
            "source_step_ids": sorted({row["step_id"] for row in issues}),
            "source_sentiment_labels": {row["id"]: row["sentiment"]["label"] for row in issues},
            "agreement": agreement, "disagreement": disagreement,
            "document_ids": document_ids, "document_counts_by_position": counts,
            "synthesis": (
                f"{len(document_ids)} supplied document(s): "
                f"{counts['support']} supporting, {counts['oppose']} opposing, "
                f"{counts['neutral']} neutral. Classification: {agreement}. "
                "Counts describe evidence, not truth or source independence."),
            "evidence": evidence, "questions": questions,
            "unresolved_questions": unresolved,
        })
    findings.sort(key=lambda row: (-row["priority_score"], row["topic"]))
    return {
        "schema_version": 1, "status": "ok", "synthetic": True,
        "guided": state["guided"], "sentiment": state["sentiment"],
        "deep": {
            "method": "exact-topic structured-claim synthesis; document-level stance counts",
            "limitations": [
                "Synthetic supplied evidence only; no factual verification or external research.",
                "Sentiment is English lexicon-based, not a contextual language model.",
                "Topic matching is exact and case-sensitive.",
                "Free-text questions remain unresolved pending human review.",
            ],
            "findings": findings,
            "unused_document_ids": sorted(
                document["id"] for document in data["documents"]
                if not any(claim["topic"] in by_topic for claim in document["claims"])),
        },
    }


def run_pipeline(data):
    return deep_stage(sentiment_stage(guided_stage(data)))


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "$", "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "CLI", "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=strict_object,
                             parse_constant=reject_constant)
        output = run_pipeline(data)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        output = {"schema_version": 1, "status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
