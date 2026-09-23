"""Deterministic synthetic customer-insights -> discovery reference pipeline.

Usage: python -B implementation.py example_input.json
Severity always dominates sentiment when ranking issues. A ready journey has
exactly two sequentially feasible actions; exhausted paths are explicitly
unavailable, never padded with repeats or actions with unmet prerequisites.
"""

import json
import re
import sys
from pathlib import Path


VERSION = 1
LEXICON = {
    "good": 1, "great": 2, "love": 2, "helpful": 1, "excellent": 2,
    "bad": -1, "broken": -2, "hate": -2, "slow": -1, "awful": -2,
}
NEGATORS = {"not", "never", "no"}
ACTIONS = ("acknowledge", "triage", "investigate", "resolve", "confirm", "monitor")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, name):
    require(isinstance(value, dict), f"{name} must be an object")
    require(set(value) == set(expected), f"{name} has missing or unknown fields")


def same_schema_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_schema_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_schema_value(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def score(text):
    """Exact word matches; only an immediately preceding negator flips a term."""
    tokens = re.findall(r"[a-z]+|[.!?,;:]", text.lower())
    evidence = []
    for index, token in enumerate(tokens):
        if token in LEXICON:
            negated = index > 0 and tokens[index - 1] in NEGATORS
            weight = LEXICON[token] * (-1 if negated else 1)
            evidence.append({
                "token": token, "token_index": index,
                "weight": weight, "negated": negated,
            })
    total = sum(item["weight"] for item in evidence)
    scale = sum(abs(item["weight"]) for item in evidence)
    normalized = round(total / scale, 6) if scale else 0.0
    return normalized, evidence


def expected_sentiment(source):
    issues = []
    for feedback in source["feedback"]:
        value, evidence = score(feedback["text"])
        issues.append({
            "issue_id": feedback["id"],
            "severity": feedback["severity"],
            "sentiment": value,
            "label": "negative" if value < 0 else "positive" if value > 0 else "neutral",
            "evidence": evidence,
            "priority": feedback["severity"] * 100 + round(max(0, -value) * 20),
        })
    issues.sort(key=lambda item: (-item["priority"], item["issue_id"]))
    return {"schema_version": VERSION, "issues": issues}


def action_path(issue):
    if issue["severity"] >= 4:
        return list(ACTIONS)
    return [action for action in ACTIONS if action != "triage"]


def expected_journey(sentiment, completed):
    issues = sentiment["issues"]
    if not issues:
        return {
            "schema_version": VERSION, "status": "unavailable",
            "reason": "no_issues", "issue_id": None, "source_priority": None,
            "next_actions": [], "steps": [],
        }
    issue = issues[0]
    path = action_path(issue)
    done = set(completed)
    remaining = [action for action in path if action not in done]
    steps = []
    # The path is a prerequisite chain; completion histories must be prefixes.
    if len(remaining) >= 2:
        for number, action in enumerate(remaining[:2], 1):
            index = path.index(action)
            steps.append({
                "step": number, "action": action, "issue_id": issue["issue_id"],
                "requires": [] if index == 0 else [path[index - 1]],
            })
    return {
        "schema_version": VERSION, "status": "ready" if steps else "unavailable",
        "reason": None if steps else "fewer_than_two_remaining",
        "issue_id": issue["issue_id"], "source_priority": issue["priority"],
        "next_actions": remaining[:1], "steps": steps,
    }


def validate(kind, payload, source=None):
    """One validation boundary shared by input, stage handoffs, and output."""
    if kind == "input":
        keys(payload, ("schema_version", "synthetic", "feedback", "completed_actions"), kind)
        require(type(payload["schema_version"]) is int and payload["schema_version"] == VERSION,
                "schema_version must be integer 1")
        require(payload["synthetic"] is True, "this reference accepts synthetic data only")
        require(isinstance(payload["feedback"], list), "feedback must be an array")
        require(len(payload["feedback"]) <= 1000, "at most 1000 feedback entries")
        seen = set()
        for item in payload["feedback"]:
            keys(item, ("id", "text", "severity"), "feedback entry")
            identifier = item["id"]
            require(isinstance(identifier, str) and 0 < len(identifier) <= 100
                    and identifier == identifier.strip(), "id must be a nonblank trimmed string")
            require(identifier not in seen, "feedback ids must be unique")
            seen.add(identifier)
            require(isinstance(item["text"], str) and 0 < len(item["text"].strip())
                    and len(item["text"]) <= 10000, "text must contain 1..10000 characters")
            require(type(item["severity"]) is int and 1 <= item["severity"] <= 5,
                    "severity must be an integer from 1 to 5")
        completed = payload["completed_actions"]
        require(isinstance(completed, list), "completed_actions must be an array")
        require(all(isinstance(action, str) and action in ACTIONS for action in completed),
                "unknown completed action")
        require(len(set(completed)) == len(completed), "duplicate completed action")
        insights = expected_sentiment(payload)
        if insights["issues"]:
            path = action_path(insights["issues"][0])
            require(set(completed) == set(path[:len(completed)]),
                    "completed_actions must form a prerequisite-complete prefix for the top issue")
        else:
            require(not completed, "completed actions require an issue")
    elif kind == "sentiment":
        validate("input", source)
        keys(payload, ("schema_version", "issues"), kind)
        require(same_schema_value(payload, expected_sentiment(source)),
                "invalid sentiment evidence, ranking, or schema")
    elif kind == "journey":
        insights, original = source
        validate("sentiment", insights, original)
        keys(payload, ("schema_version", "status", "reason", "issue_id",
                       "source_priority", "next_actions", "steps"), kind)
        require(same_schema_value(payload, expected_journey(insights, original["completed_actions"])),
                "journey violates handoff or two-step prerequisite constraints")
    elif kind == "output":
        keys(payload, ("schema_version", "status", "synthetic", "sentiment", "journey"), kind)
        require(type(payload["schema_version"]) is int and payload["schema_version"] == VERSION
                and payload["status"] == "ok" and payload["synthetic"] is True,
                "invalid output envelope")
        validate("sentiment", payload["sentiment"], source)
        validate("journey", payload["journey"], (payload["sentiment"], source))
    else:
        raise ValidationError("unknown validation kind")
    return payload


def sentiment_stage(source):
    validate("input", source)
    return validate("sentiment", expected_sentiment(source), source)


def journey_stage(insights, source):
    validate("sentiment", insights, source)
    result = expected_journey(insights, source["completed_actions"])
    return validate("journey", result, (insights, source))


def run_pipeline(source):
    insights = sentiment_stage(source)
    journey = journey_stage(insights, source)
    return validate("output", {
        "schema_version": VERSION, "status": "ok", "synthetic": True,
        "sentiment": insights, "journey": journey,
    }, source)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        source = json.loads(
            Path(argv[0]).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object, parse_constant=reject_constant,
        )
        result = run_pipeline(source)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        result = {"schema_version": VERSION, "status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
