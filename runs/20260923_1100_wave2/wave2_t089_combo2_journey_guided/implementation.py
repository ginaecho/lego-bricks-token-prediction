"""Synthetic, deterministic journey-to-onboarding reference pipeline.

Run: python -B implementation.py example_input.json
Only Python's standard library is used; no providers or persistence are needed.
"""

import json
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def strings(value, path, nonempty=False):
    require(isinstance(value, list), path + " must be an array")
    for item in value:
        text(item, path + " item")
    require(len(set(value)) == len(value), path + " contains duplicates")
    require(not nonempty or bool(value), path + " must not be empty")


def validate_input(data):
    fields(data, ("schema_version", "fixture_label", "profile", "actions", "progress_events"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be integer 1")
    require(data["fixture_label"] == "synthetic", "fixture_label must be synthetic")
    profile = data["profile"]
    fields(profile, ("goal", "interests", "completed"), "profile")
    text(profile["goal"], "profile.goal")
    strings(profile["interests"], "profile.interests")
    strings(profile["completed"], "profile.completed")
    require(isinstance(data["actions"], list) and bool(data["actions"]), "actions must be a nonempty array")
    catalog = {}
    for action in data["actions"]:
        fields(action, ("id", "title", "prerequisites", "tags", "goals"), "action")
        text(action["id"], "action.id")
        text(action["title"], "action.title")
        for key in ("prerequisites", "tags", "goals"):
            strings(action[key], "action." + key)
        require(action["id"] not in catalog, "duplicate action id")
        catalog[action["id"]] = action
    for action in catalog.values():
        require(set(action["prerequisites"]) <= catalog.keys(), "unknown prerequisite")
        require(action["id"] not in action["prerequisites"], "self prerequisite")
    # Iterative topological validation also handles deep catalogs without recursion.
    resolved = set()
    while len(resolved) < len(catalog):
        ready = {key for key, action in catalog.items()
                 if key not in resolved and set(action["prerequisites"]) <= resolved}
        require(bool(ready), "prerequisite cycle")
        resolved.update(ready)
    completed = set(profile["completed"])
    require(completed <= catalog.keys(), "unknown completed action")
    for key in completed:
        require(set(catalog[key]["prerequisites"]) <= completed,
                "initial completed actions must include their prerequisites")
    require(isinstance(data["progress_events"], list), "progress_events must be an array")
    for event in data["progress_events"]:
        fields(event, ("action_id", "status"), "progress event")
        text(event["action_id"], "event.action_id")
        require(event["status"] in ("in_progress", "completed"), "invalid event status")
    return catalog


def relevance(action, profile):
    return (10 if profile["goal"] in action["goals"] else 0) + 3 * len(
        set(action["tags"]) & set(profile["interests"]))


def step_for(action, profile):
    return {
        "id": action["id"],
        "title": action["title"],
        "prerequisites": list(action["prerequisites"]),
        "relevance_score": relevance(action, profile),
    }


def validate_journey(plan, data, catalog):
    """The same contract is enforced by both the producer and consumer."""
    fields(plan, ("schema_version", "fixture_label", "goal", "initial_completed", "steps"), "journey")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1,
            "journey schema mismatch")
    require(plan["fixture_label"] == data["fixture_label"], "journey fixture mismatch")
    require(plan["goal"] == data["profile"]["goal"], "journey goal mismatch")
    require(plan["initial_completed"] == sorted(data["profile"]["completed"]),
            "journey completion context mismatch")
    require(isinstance(plan["steps"], list) and len(plan["steps"]) == 2,
            "journey must have exactly two steps")
    completed = set(plan["initial_completed"])
    for step in plan["steps"]:
        fields(step, ("id", "title", "prerequisites", "relevance_score"), "journey step")
        text(step["id"], "journey step id")
        require(step["id"] in catalog, "unknown journey action")
        action = catalog[step["id"]]
        require(type(step["relevance_score"]) is int, "journey score must be an integer")
        require(step == step_for(action, data["profile"]), "journey step does not match catalog")
        require(step["id"] not in completed, "journey repeats a completed action")
        require(set(step["prerequisites"]) <= completed, "journey prerequisites not satisfied")
        completed.add(step["id"])
    require(plan["goal"] in catalog[plan["steps"][-1]["id"]]["goals"],
            "journey must end with a goal-relevant action")
    return plan


def recommend(data):
    catalog = validate_input(data)
    profile = data["profile"]
    completed = set(profile["completed"])
    candidates = []
    for first in catalog.values():
        if first["id"] in completed or not set(first["prerequisites"]) <= completed:
            continue
        after_first = completed | {first["id"]}
        for second in catalog.values():
            if second["id"] in after_first or not set(second["prerequisites"]) <= after_first:
                continue
            if profile["goal"] not in second["goals"]:
                continue
            candidates.append((first, second))
    require(bool(candidates), "no feasible two-step journey for this goal")
    first, second = min(
        candidates,
        key=lambda pair: (-sum(relevance(action, profile) for action in pair),
                          pair[0]["id"], pair[1]["id"]),
    )
    plan = {
        "schema_version": 1,
        "fixture_label": data["fixture_label"],
        "goal": profile["goal"],
        "initial_completed": sorted(completed),
        "steps": [step_for(first, profile), step_for(second, profile)],
    }
    return validate_journey(plan, data, catalog)


def guided_setup(data, plan):
    catalog = validate_input(data)
    validate_journey(plan, data, catalog)
    steps = {step["id"]: step for step in plan["steps"]}
    completed = set(plan["initial_completed"])
    active = set()
    for event in data["progress_events"]:
        key = event["action_id"]
        require(key in steps, "progress event is outside the selected journey")
        require(key not in completed, "completed steps cannot be updated again")
        require(set(steps[key]["prerequisites"]) <= completed,
                "progress event has unmet prerequisites")
        if event["status"] == "in_progress":
            require(key not in active, "step is already in progress")
            active.add(key)
        else:
            completed.add(key)
            active.discard(key)
    result_steps = []
    for step in plan["steps"]:
        key = step["id"]
        if key in completed:
            status = "completed"
        elif key in active:
            status = "in_progress"
        elif set(step["prerequisites"]) <= completed:
            status = "ready"
        else:
            status = "blocked"
        result_steps.append({**step, "prerequisites": list(step["prerequisites"]), "status": status})
    count = sum(step["id"] in completed for step in plan["steps"])
    return {
        "schema_version": plan["schema_version"],
        "fixture_label": plan["fixture_label"],
        "goal": plan["goal"],
        "steps": result_steps,
        "completed_ids": sorted(completed),
        "progress": {"completed": count, "total": 2, "percent": count * 50},
    }


def run_pipeline(data):
    journey = recommend(data)
    guided = guided_setup(data, journey)
    return {"status": "ok", "schema_version": 1, "fixture_label": "synthetic",
            "journey": journey, "guided": guided}


def reject_constant(value):
    raise ValidationError("nonstandard JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
