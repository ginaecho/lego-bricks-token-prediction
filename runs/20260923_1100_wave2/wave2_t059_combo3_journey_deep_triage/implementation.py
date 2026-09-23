"""Synthetic, deterministic journey -> research -> accountable support pipeline."""

import json
import sys
from itertools import product


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(names.split()), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def texts(value, path, nonempty=False):
    require(isinstance(value, list), path + " must be an array")
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), path + " contains duplicates")
    require(not nonempty or bool(value), path + " must not be empty")


def records(value, path):
    require(isinstance(value, list), path + " must be an array")
    require(all(isinstance(item, dict) for item in value), path + " must contain objects")


def route(value, path):
    fields(value, "category priority owner", path)
    text(value["category"], path + ".category")
    text(value["owner"], path + ".owner")
    require(value["priority"] in ("low", "normal", "high", "urgent"),
            path + ".priority is invalid")


def validate_input(data):
    fields(data, "schema_version synthetic profile actions documents routing", "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "reference inputs must be labeled synthetic")
    profile = data["profile"]
    fields(profile, "completed interests", "profile")
    texts(profile["completed"], "profile.completed")
    texts(profile["interests"], "profile.interests", True)
    records(data["actions"], "actions")
    action_ids = set()
    for action in data["actions"]:
        fields(action, "id title requires topics", "action")
        text(action["id"], "action.id")
        text(action["title"], "action.title")
        texts(action["requires"], "action.requires")
        texts(action["topics"], "action.topics", True)
        require(action["id"] not in action_ids, "duplicate action id")
        action_ids.add(action["id"])
    known = action_ids | set(profile["completed"])
    for action in data["actions"]:
        require(set(action["requires"]) <= known, "unknown prerequisite")
        require(action["id"] not in action["requires"], "self prerequisite")
    # Topological removal catches cycles even in already-completed actions.
    pending = {a["id"]: set(a["requires"]) & action_ids for a in data["actions"]}
    while pending:
        ready = {key for key, deps in pending.items() if not deps}
        require(bool(ready), "cyclic prerequisites")
        pending = {key: deps - ready for key, deps in pending.items() if key not in ready}
    records(data["documents"], "documents")
    document_ids = set()
    for doc in data["documents"]:
        fields(doc, "id title text claims", "document")
        for key in ("id", "title", "text"):
            text(doc[key], "document." + key)
        require(doc["id"] not in document_ids, "duplicate document id")
        document_ids.add(doc["id"])
        records(doc["claims"], "document.claims")
        claimed_topics = set()
        for claim in doc["claims"]:
            fields(claim, "topic position quote", "claim")
            for key in ("topic", "position", "quote"):
                text(claim[key], "claim." + key)
            require(claim["quote"] in doc["text"], "claim quote is not in source text")
            require(claim["topic"] not in claimed_topics, "duplicate topic in document")
            claimed_topics.add(claim["topic"])
    routing = data["routing"]
    fields(routing, "rules default", "routing")
    route(routing["default"], "routing.default")
    records(routing["rules"], "routing.rules")
    for rule in routing["rules"]:
        fields(rule, "kind topics route", "routing rule")
        require(rule["kind"] in ("conflict", "evidence_gap", "corroboration_needed"),
                "unknown issue kind")
        texts(rule["topics"], "routing rule.topics")
        route(rule["route"], "routing rule.route")
    return data


def _journey(data):
    completed = set(data["profile"]["completed"])
    interests = set(data["profile"]["interests"])
    remaining = [a for a in data["actions"] if a["id"] not in completed]
    score = lambda a: len(set(a["topics"]) & interests)
    ready = sorted((a for a in remaining if set(a["requires"]) <= completed),
                   key=lambda a: (-score(a), a["id"]))
    pairs = [(a, b) for a, b in product(ready, remaining)
             if a["id"] != b["id"] and set(b["requires"]) <= completed | {a["id"]}]
    require(bool(pairs), "no valid two-step journey is available")
    first, second = min(pairs, key=lambda pair: (
        -sum(score(a) for a in pair), -score(pair[0]), pair[0]["id"], pair[1]["id"]))
    state = set(completed)
    steps = []
    for index, action in enumerate((first, second), 1):
        steps.append({
            "step": index, "action_id": action["id"], "title": action["title"],
            "requires": sorted(action["requires"]),
            "satisfied_by": sorted(set(action["requires"]) & state),
            "topics": sorted(action["topics"]), "interest_score": score(action),
        })
        state.add(action["id"])
    return {
        "next_actions": [{"action_id": a["id"], "interest_score": score(a)} for a in ready],
        "steps": steps,
        "research_topics": sorted(set(first["topics"]) | set(second["topics"])),
    }


def _research(data, journey):
    findings, issues = [], []
    for index, topic in enumerate(journey["research_topics"], 1):
        evidence = sorted(
            [{"document_id": doc["id"], "position": claim["position"], "quote": claim["quote"]}
             for doc in data["documents"] for claim in doc["claims"] if claim["topic"] == topic],
            key=lambda item: item["document_id"])
        positions = sorted({item["position"] for item in evidence})
        if not evidence:
            status, kind = "unanswered", "evidence_gap"
            summary = "No source evidence is available."
            question = "What evidence answers the question about " + topic + "?"
        elif len(positions) > 1:
            status, kind = "disputed", "conflict"
            summary = "Sources disagree: " + "; ".join(positions)
            question = "Which position about " + topic + " is supported, and why do sources disagree?"
        elif len(evidence) == 1:
            status, kind = "single_source", "corroboration_needed"
            summary = "One source reports: " + positions[0]
            question = "Can an independent source corroborate " + topic + "?"
        else:
            status, kind = "consensus", None
            summary = "Sources agree: " + positions[0]
            question = None
        action_ids = [step["action_id"] for step in journey["steps"] if topic in step["topics"]]
        findings.append({
            "topic": topic, "status": status, "summary": summary, "evidence": evidence,
            "positions": positions, "journey_action_ids": action_ids,
            "unresolved_questions": [question] if question else [],
        })
        if kind:
            issues.append({
                "id": "issue-" + str(index), "topic": topic, "kind": kind,
                "question": question, "evidence": evidence,
                "journey_action_ids": action_ids,
            })
    return {"topics": list(journey["research_topics"]), "findings": findings, "issues": issues}


def _triage(data, research):
    tickets = []
    for issue in research["issues"]:
        selected = data["routing"]["default"]
        matched_rule = "default"
        for index, rule in enumerate(data["routing"]["rules"]):
            if rule["kind"] == issue["kind"] and (
                    not rule["topics"] or issue["topic"] in rule["topics"]):
                selected, matched_rule = rule["route"], index
                break
        tickets.append({
            "id": "ticket-" + issue["id"], "issue_id": issue["id"],
            "topic": issue["topic"], "kind": issue["kind"],
            "subject": issue["question"], "category": selected["category"],
            "priority": selected["priority"], "owner": selected["owner"],
            "matched_rule": matched_rule,
            "journey_action_ids": list(issue["journey_action_ids"]),
            "evidence": [dict(item) for item in issue["evidence"]],
            "status": "open",
        })
    return {"tickets": tickets, "ticket_count": len(tickets)}


def validate(stage, payload, data=None, journey=None, research=None):
    """Single validation boundary; derived stages must exactly match their provenance."""
    if stage == "input":
        return validate_input(payload)
    validate_input(data)
    expected_journey = _journey(data)
    if stage == "journey":
        expected = expected_journey
    elif stage in ("research", "triage"):
        require(_same(journey, expected_journey), "invalid journey handoff")
        expected_research = _research(data, journey)
        if stage == "research":
            expected = expected_research
        else:
            require(_same(research, expected_research), "invalid research handoff")
            expected = _triage(data, research)
    else:
        raise ValidationError("unknown validation stage")
    require(_same(payload, expected), "invalid " + stage + " output")
    return payload


def _same(left, right):
    # JSON comparison distinguishes booleans from integers, unlike Python equality.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def recommend(data):
    validate("input", data)
    return validate("journey", _journey(data), data)


def research(data, journey):
    validate("journey", journey, data)
    return validate("research", _research(data, journey), data, journey)


def triage(data, journey, findings):
    validate("research", findings, data, journey)
    return validate("triage", _triage(data, findings), data, journey, findings)


def run(data):
    validate("input", data)
    journey = recommend(data)
    findings = research(data, journey)
    tickets = triage(data, journey, findings)
    return {"schema_version": 1, "synthetic": True, "status": "ok",
            "journey": journey, "research": findings, "triage": tickets}


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
            data = json.load(handle, object_pairs_hook=unique_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 ValidationError("non-finite JSON number: " + value)))
        result = run(data)
    except (ValidationError, OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
