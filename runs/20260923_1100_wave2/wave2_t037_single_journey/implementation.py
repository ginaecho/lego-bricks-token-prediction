"""Synthetic personalized discovery reference CLI; Python standard library only."""

import json
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(expected), f"{path} must have exactly: {', '.join(expected)}")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonblank text")


def strings(value, path):
    require(isinstance(value, list), f"{path} must be a list")
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), f"{path} must not contain duplicates")


def integer(value, minimum, maximum, path):
    require(type(value) is int and minimum <= value <= maximum,
            f"{path} must be an integer in [{minimum}, {maximum}]")


def validate_input(data):
    """One validation layer enforces schema and prerequisite semantics."""
    fields(data, ("schema_version", "synthetic", "actions", "profile", "limit"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "synthetic must be true for this reference")
    integer(data["limit"], 1, 10, "limit")
    require(isinstance(data["actions"], list) and len(data["actions"]) <= 100,
            "actions must be a list of at most 100 actions")
    actions = {}
    for item in data["actions"]:
        fields(item, ("id", "title", "prerequisites", "tags", "minutes"), "action")
        text(item["id"], "action.id")
        text(item["title"], "action.title")
        strings(item["prerequisites"], "action.prerequisites")
        strings(item["tags"], "action.tags")
        integer(item["minutes"], 1, 1440, "action.minutes")
        require(item["id"] not in actions, f"duplicate action id: {item['id']}")
        actions[item["id"]] = item
    for item in actions.values():
        require(set(item["prerequisites"]) <= actions.keys(),
                f"unknown prerequisite for {item['id']}")
    # Iterative topological validation avoids recursion limits.
    resolved = set()
    while len(resolved) < len(actions):
        ready = {key for key, item in actions.items()
                 if key not in resolved and set(item["prerequisites"]) <= resolved}
        require(bool(ready), "prerequisite graph contains a cycle")
        resolved.update(ready)
    profile = data["profile"]
    fields(profile, ("completed", "goals", "available_minutes"), "profile")
    strings(profile["completed"], "profile.completed")
    strings(profile["goals"], "profile.goals")
    integer(profile["available_minutes"], 0, 2880, "profile.available_minutes")
    completed = set(profile["completed"])
    require(completed <= actions.keys(), "profile.completed contains unknown actions")
    for key in completed:
        require(set(actions[key]["prerequisites"]) <= completed,
                f"completed action {key} has incomplete prerequisites")
    return actions


def validate_journey(steps, actions, completed, budget):
    """Replay the handoff: each step sees only earlier completed actions."""
    require(len(steps) == 2, "journey must contain exactly two steps")
    done = set(completed)
    elapsed = 0
    for key in steps:
        require(key in actions and key not in done, "journey action must be new and known")
        item = actions[key]
        require(set(item["prerequisites"]) <= done, "journey prerequisite is unsatisfied")
        elapsed += item["minutes"]
        done.add(key)
    require(elapsed <= budget, "journey exceeds available_minutes")
    return elapsed


def recommend(data):
    actions = validate_input(data)
    profile = data["profile"]
    completed = set(profile["completed"])
    goals = set(profile["goals"])
    budget = profile["available_minutes"]

    def matches(key):
        return sorted(goals & set(actions[key]["tags"]))

    def available(key, done):
        return key not in done and set(actions[key]["prerequisites"]) <= done

    def card(key):
        item = actions[key]
        matched = matches(key)
        return {
            "id": key,
            "title": item["title"],
            "minutes": item["minutes"],
            "matched_goals": matched,
            "reason": "Matches stated goals" if matched else "Eligible discovery action",
        }

    eligible = [key for key in actions
                if available(key, completed) and actions[key]["minutes"] <= budget]
    eligible.sort(key=lambda key: (-len(matches(key)), actions[key]["minutes"], key))
    recommendations = [card(key) for key in eligible[:data["limit"]]]
    pairs = []
    for first in eligible:
        after_first = completed | {first}
        for second in sorted(actions):
            if not available(second, after_first):
                continue
            total = actions[first]["minutes"] + actions[second]["minutes"]
            if total > budget:
                continue
            coverage = len(set(matches(first)) | set(matches(second)))
            relevance = len(matches(first)) + len(matches(second))
            pairs.append(((-coverage, -relevance, total, first, second), first, second))
    if pairs:
        _, first, second = min(pairs)
        total = validate_journey([first, second], actions, completed, budget)
        steps = []
        done = set(completed)
        for position, key in enumerate((first, second), 1):
            steps.append(dict(card(key), step=position,
                              completed_before=sorted(done),
                              prerequisites=sorted(actions[key]["prerequisites"])))
            done.add(key)
        journey = {"status": "ready", "steps": steps, "total_minutes": total,
                   "validated": True, "reason": None}
    else:
        journey = {"status": "unavailable", "steps": [], "total_minutes": 0,
                   "validated": False,
                   "reason": "No valid two-step journey fits the remaining actions and time budget"}
    blocked = []
    for key in sorted(actions):
        if key in completed:
            continue
        missing = sorted(set(actions[key]["prerequisites"]) - completed)
        too_long = actions[key]["minutes"] > budget
        if missing or too_long:
            blocked.append({"id": key, "missing_prerequisites": missing,
                            "exceeds_budget": too_long})
    return {"schema_version": 1, "status": "ok", "synthetic": True,
            "recommendations": recommendations, "journey": journey, "blocked": blocked}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = recommend(data)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)},
                         ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
