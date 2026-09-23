"""Deterministic synthetic document-to-customer-insight reference pipeline.

Run: python -B implementation.py example_input.json
Offsets are zero-based, end-exclusive Python Unicode character offsets.
No model, network, or third-party dependencies are used.
"""

import copy
import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


STAGES = ("input", "extract", "deep", "guided", "sentiment")
LEXICON = {
    "good": 1, "great": 2, "easy": 1, "helpful": 1, "love": 2,
    "excellent": 2, "bad": -1, "broken": -2, "confusing": -1,
    "frustrating": -2, "hate": -2, "slow": -1, "terrible": -2,
}
SEVERITY = {"low": 10, "medium": 30, "high": 60, "critical": 90}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, where):
    require(type(value) is dict, f"{where}: expected object")
    require(set(value) == set(keys), f"{where}: expected keys {sorted(keys)}")


def text(value, where, allow_empty=False):
    require(isinstance(value, str), f"{where}: expected string")
    require(allow_empty or bool(value.strip()), f"{where}: empty string")
    require(len(value) <= 20000, f"{where}: string exceeds 20000 characters")


def array(value, where, minimum=0, maximum=100):
    require(type(value) is list and minimum <= len(value) <= maximum,
            f"{where}: expected list of size {minimum}..{maximum}")


def names(items, where):
    result = []
    for item in items:
        text(item, where)
        require(item not in result, f"{where}: duplicate {item}")
        result.append(item)
    return set(result)


def validate_input(data):
    obj(data, ("schema_version", "synthetic", "documents", "fields", "steps", "feedback"),
        "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be integer 1")
    require(data["synthetic"] is True, "synthetic must be true")
    array(data["documents"], "documents", 1)
    for doc in data["documents"]:
        obj(doc, ("id", "text"), "document")
        text(doc["id"], "document.id")
        text(doc["text"], "document.text", allow_empty=True)
    names([d["id"] for d in data["documents"]], "document ids")
    array(data["fields"], "fields", 1, 30)
    for field in data["fields"]:
        obj(field, ("name", "pattern", "required"), "field")
        text(field["name"], "field.name")
        text(field["pattern"], "field.pattern")
        require(type(field["required"]) is bool, "field.required: expected boolean")
        try:
            pattern = re.compile(field["pattern"])
        except (re.error, OverflowError, RecursionError) as exc:
            raise ValidationError(f"field.pattern: {exc}") from exc
        require("value" in pattern.groupindex, "field.pattern needs named group 'value'")
    field_names = names([f["name"] for f in data["fields"]], "field names")
    array(data["steps"], "steps", 1)
    for step in data["steps"]:
        obj(step, ("id", "title", "requires_fields", "prerequisites", "complete"), "step")
        text(step["id"], "step.id")
        text(step["title"], "step.title")
        require(type(step["complete"]) is bool, "step.complete: expected boolean")
        array(step["requires_fields"], "step.requires_fields", maximum=30)
        array(step["prerequisites"], "step.prerequisites")
        require(names(step["requires_fields"], "required fields") <= field_names,
                "step references unknown field")
        names(step["prerequisites"], "prerequisites")
    step_ids = names([s["id"] for s in data["steps"]], "step ids")
    for step in data["steps"]:
        require(set(step["prerequisites"]) <= step_ids, "unknown prerequisite")
    ordered_steps(data["steps"])
    array(data["feedback"], "feedback")
    for feedback in data["feedback"]:
        obj(feedback, ("id", "step_id", "text", "severity"), "feedback")
        text(feedback["id"], "feedback.id")
        text(feedback["step_id"], "feedback.step_id")
        text(feedback["text"], "feedback.text", allow_empty=True)
        text(feedback["severity"], "feedback.severity")
        require(feedback["step_id"] in step_ids, "feedback references unknown step")
        require(feedback["severity"] in SEVERITY, "unknown severity")
    names([f["id"] for f in data["feedback"]], "feedback ids")


def ordered_steps(steps):
    remaining = list(steps)
    result, done = [], set()
    while remaining:
        ready = [s for s in remaining if set(s["prerequisites"]) <= done]
        require(bool(ready), "prerequisite cycle")
        for step in ready:
            result.append(step)
            done.add(step["id"])
            remaining.remove(step)
    return result


def build_extraction(data):
    documents = []
    for doc in data["documents"]:
        fields, missing = {}, []
        for field in data["fields"]:
            matches = []
            for match in re.finditer(field["pattern"], doc["text"]):
                value = match.group("value")
                if value is None or not value.strip():
                    continue
                start, end = match.span("value")
                matches.append({
                    "value": value, "document_id": doc["id"],
                    "span": {"start": start, "end": end},
                })
            fields[field["name"]] = matches
            if not matches:
                missing.append({"field": field["name"], "required": field["required"]})
        documents.append({"document_id": doc["id"], "fields": fields, "missing": missing})
    return {"documents": documents}


def normalize(value):
    return " ".join(value.casefold().split())


def build_research(data, extraction):
    topics, questions, disagreements = [], [], []
    for field in data["fields"]:
        name = field["name"]
        groups, missing_docs = {}, []
        for doc in extraction["documents"]:
            evidence = doc["fields"][name]
            if not evidence:
                missing_docs.append(doc["document_id"])
            for source in evidence:
                key = normalize(source["value"])
                groups.setdefault(key, []).append(copy.deepcopy(source))
        claims = [{"normalized_value": value, "evidence": sources,
                   "supporting_documents": sorted({s["document_id"] for s in sources})}
                  for value, sources in sorted(groups.items())]
        status = "missing" if not claims else "disputed" if len(claims) > 1 else "consensus"
        topic = {
            "field": name, "status": status, "claims": claims,
            "missing_documents": missing_docs,
            "usable": status == "consensus" and (not field["required"] or not missing_docs),
        }
        topics.append(topic)
        if status == "disputed":
            disagreements.append({"field": name, "values": sorted(groups)})
            questions.append({"field": name, "kind": "disagreement",
                              "question": f"Which value for {name} is authoritative?"})
        if missing_docs:
            questions.append({
                "field": name, "kind": "missing_evidence", "document_ids": missing_docs,
                "required": field["required"],
                "question": f"Can the missing documents supply {name}?",
            })
    return {"topics": topics, "disagreements": disagreements,
            "unresolved_questions": questions,
            "summary": {
                "document_count": len(extraction["documents"]),
                "consensus_fields": [t["field"] for t in topics if t["status"] == "consensus"],
                "disputed_fields": [t["field"] for t in topics if t["status"] == "disputed"],
            }}


def build_onboarding(data, research):
    topics = {t["field"]: t for t in research["topics"]}
    results = {}
    for step in ordered_steps(data["steps"]):
        blocked_fields = [f for f in step["requires_fields"] if not topics[f]["usable"]]
        blocked_steps = [p for p in step["prerequisites"]
                         if results[p]["status"] != "completed"]
        blocked = bool(blocked_fields or blocked_steps)
        status = "blocked" if blocked else "completed" if step["complete"] else "ready"
        results[step["id"]] = {
            "id": step["id"], "title": step["title"], "status": status,
            "completion_requested": step["complete"],
            "blocked_fields": blocked_fields, "blocked_prerequisites": blocked_steps,
            "evidence": {f: copy.deepcopy(topics[f]["claims"]) for f in step["requires_fields"]},
            "unresolved_questions": [copy.deepcopy(q) for q in research["unresolved_questions"]
                                     if q["field"] in step["requires_fields"]],
        }
    steps = list(results.values())
    completed = sum(s["status"] == "completed" for s in steps)
    return {
        "steps": steps,
        "progress": {"completed": completed, "total": len(steps),
                     "fraction": completed / len(steps)},
        "feedback": [dict(copy.deepcopy(f), step_status=results[f["step_id"]]["status"])
                     for f in data["feedback"]],
    }


def score_sentiment(content):
    tokens = list(re.finditer(r"\b[a-z]+\b", content, flags=re.IGNORECASE))
    contributions = []
    for index, token in enumerate(tokens):
        word = token.group().casefold()
        if word not in LEXICON:
            continue
        # Only immediately preceding negation within the same punctuation-free phrase.
        previous = tokens[index - 1] if index else None
        negated = bool(previous and previous.group().casefold() in {"not", "no", "never"}
                       and content[previous.end():token.start()].isspace())
        weight = -LEXICON[word] if negated else LEXICON[word]
        contributions.append({"token": word, "base_weight": LEXICON[word],
                              "negated": negated, "contribution": weight})
    raw = sum(c["contribution"] for c in contributions)
    score = max(-5, min(5, raw))
    return {"score": score, "raw_score": raw,
            "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "contributions": contributions}


def build_insights(onboarding):
    issues = []
    for feedback in onboarding["feedback"]:
        sentiment = score_sentiment(feedback["text"])
        components = {
            "severity": SEVERITY[feedback["severity"]],
            "negative_sentiment": 3 * max(0, -sentiment["score"]),
            "onboarding": {"blocked": 10, "ready": 5, "completed": 0}[feedback["step_status"]],
        }
        priority = min(100, sum(components.values()))
        issues.append({
            "id": feedback["id"], "step_id": feedback["step_id"],
            "text": feedback["text"], "severity": feedback["severity"],
            "step_status": feedback["step_status"], "sentiment": sentiment,
            "priority": priority, "priority_components": components,
        })
    issues.sort(key=lambda issue: (-issue["priority"], -SEVERITY[issue["severity"]], issue["id"]))
    for rank, issue in enumerate(issues, 1):
        issue["rank"] = rank
    return {"issues": issues, "policy": {
        "lexicon": dict(LEXICON), "severity_weights": dict(SEVERITY),
        "sentiment_range": [-5, 5], "negative_sentiment_multiplier": 3,
        "onboarding_weights": {"blocked": 10, "ready": 5, "completed": 0},
        "priority_cap": 100, "tie_break": "severity descending, id ascending",
        "negation": "immediately preceding not/no/never separated only by whitespace",
        "limitations": "English lexicon only; no sarcasm, intent, or contextual language inference",
    }}


def validate_state(state, expected_stage):
    """One validation boundary for input and every exact, provenance-bearing handoff.

    Recomputing this bounded deterministic schema rejects forged spans, stale research,
    bypassed prerequisites, and modified scores rather than trusting structural shape.
    """
    require(expected_stage in STAGES, "unknown expected stage")
    index = STAGES.index(expected_stage)
    outputs = ("extraction", "research", "onboarding", "insights")
    obj(state, ("schema_version", "status", "stage", "input") + outputs[:index], "state")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "state.schema_version must be integer 1")
    require(state["status"] == "ok" and state["stage"] == expected_stage,
            f"expected successful {expected_stage} state")
    validate_input(state["input"])
    expected = {}
    if index >= 1:
        expected["extraction"] = build_extraction(state["input"])
    if index >= 2:
        expected["research"] = build_research(state["input"], expected["extraction"])
    if index >= 3:
        expected["onboarding"] = build_onboarding(state["input"], expected["research"])
    if index >= 4:
        expected["insights"] = build_insights(expected["onboarding"])
    for key, value in expected.items():
        # Canonical JSON comparison distinguishes booleans from integers as well.
        try:
            actual_json = json.dumps(state[key], sort_keys=True, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"invalid {key}") from exc
        require(actual_json == json.dumps(value, sort_keys=True, allow_nan=False),
                f"invalid or stale {key} handoff")


def advance(state, previous, following, key, builder):
    validate_state(state, previous)
    result = copy.deepcopy(state)
    result[key] = builder(state)
    result["stage"] = following
    validate_state(result, following)
    return result


def extract(state):
    return advance(state, "input", "extract", "extraction",
                   lambda s: build_extraction(s["input"]))


def deep(state):
    return advance(state, "extract", "deep", "research",
                   lambda s: build_research(s["input"], s["extraction"]))


def guided(state):
    return advance(state, "deep", "guided", "onboarding",
                   lambda s: build_onboarding(s["input"], s["research"]))


def sentiment(state):
    return advance(state, "guided", "sentiment", "insights",
                   lambda s: build_insights(s["onboarding"]))


def run_pipeline(data):
    state = {"schema_version": 1, "status": "ok", "stage": "input",
             "input": copy.deepcopy(data)}
    return sentiment(guided(deep(extract(state))))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"non-finite JSON number: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
