"""Synthetic personalized discovery reference; standard library, no providers."""

import json
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, label):
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == set(names), f"{label} must have exactly: {', '.join(names)}")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")
    require(value == value.strip(), f"{label} must not have surrounding whitespace")


def strings(value, label):
    require(isinstance(value, list), f"{label} must be an array")
    for item in value:
        text(item, label)
    require(len(value) == len(set(value)), f"{label} must not contain duplicates")


def validate_input(data):
    fields(data, ["schema_version", "synthetic", "profile", "actions"], "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be integer 1")
    require(data["synthetic"] is True, "synthetic must be true")
    profile = data["profile"]
    fields(profile, ["interests", "completed_actions"], "profile")
    strings(profile["interests"], "interests")
    strings(profile["completed_actions"], "completed_actions")
    require(isinstance(data["actions"], list), "actions must be an array")
    require(len(data["actions"]) <= 100, "at most 100 actions are supported")
    catalog = {}
    for action in data["actions"]:
        fields(action, ["id", "title", "topics", "prerequisites"], "action")
        text(action["id"], "action.id")
        text(action["title"], "action.title")
        strings(action["topics"], "action.topics")
        strings(action["prerequisites"], "action.prerequisites")
        require(action["id"] not in catalog, "duplicate action id")
        catalog[action["id"]] = action
    for action in catalog.values():
        require(set(action["prerequisites"]) <= catalog.keys(), "unknown prerequisite")
    completed = set(profile["completed_actions"])
    require(completed <= catalog.keys(), "unknown completed action")
    for action_id in completed:
        require(set(catalog[action_id]["prerequisites"]) <= completed,
                "completed history must include prerequisites")
    # Kahn-style traversal also catches disconnected cycles, without recursion.
    visited = set()
    while True:
        reachable = {key for key, action in catalog.items()
                     if set(action["prerequisites"]) <= visited}
        if reachable == visited:
            break
        visited = reachable
    require(visited == catalog.keys(), "prerequisite graph must be acyclic")
    return catalog


def eligible(catalog, completed):
    return [action for key, action in catalog.items()
            if key not in completed and set(action["prerequisites"]) <= completed]


def score(action, interests):
    return len(set(action["topics"]) & interests)


def describe(action, interests):
    matched = sorted(set(action["topics"]) & interests)
    return {"action_id": action["id"], "title": action["title"],
            "score": len(matched), "matched_interests": matched}


def validate_output(result, data):
    """Validate the shared envelope and handoff against the same source catalog."""
    catalog = validate_input(data)
    fields(result, ["status", "schema_version", "synthetic", "recommendations", "journey"],
           "result")
    require(result["status"] == "ok", "result status must be ok")
    require(type(result["schema_version"]) is int and result["schema_version"] == 1,
            "result schema_version must be integer 1")
    require(result["synthetic"] is True, "result must be synthetic")
    interests = set(data["profile"]["interests"])
    completed = set(data["profile"]["completed_actions"])
    expected = sorted(eligible(catalog, completed),
                      key=lambda action: (-score(action, interests), action["id"]))
    require(result["recommendations"] == [describe(a, interests) for a in expected],
            "recommendations must be ranked eligible actions")
    journey = result["journey"]
    fields(journey, ["status", "steps", "reason"], "journey")
    require(isinstance(journey["steps"], list), "steps must be an array")
    feasible = any(eligible(catalog, completed | {a["id"]}) for a in expected)
    if not feasible:
        require(journey == {"status": "unavailable", "steps": [],
                            "reason": "No prerequisite-valid two-step journey exists."},
                "unavailable journey must have no partial steps")
        return
    require(journey["status"] == "ready" and journey["reason"] is None,
            "feasible journey must be ready")
    require(len(journey["steps"]) == 2, "ready journey must have exactly two steps")
    for index, step in enumerate(journey["steps"], 1):
        fields(step, ["position", "action_id", "title", "score", "matched_interests",
                      "prerequisites"], "step")
        require(type(step["position"]) is int and step["position"] == index,
                "incorrect step position")
        text(step["action_id"], "step.action_id")
        require(step["action_id"] in catalog, "unknown journey action")
        action = catalog[step["action_id"]]
        require(action in eligible(catalog, completed), "journey prerequisite violation")
        expected_step = dict(describe(action, interests), position=index,
                             prerequisites=sorted(action["prerequisites"]))
        require(step == expected_step, "step metadata differs from catalog")
        completed.add(action["id"])


def recommend(data):
    catalog = validate_input(data)
    interests = set(data["profile"]["interests"])
    completed = set(data["profile"]["completed_actions"])
    firsts = sorted(eligible(catalog, completed),
                    key=lambda action: (-score(action, interests), action["id"]))
    pairs = [(first, second) for first in firsts
             for second in eligible(catalog, completed | {first["id"]})]
    journey = {"status": "unavailable", "steps": [],
               "reason": "No prerequisite-valid two-step journey exists."}
    if pairs:
        # Maximize total interest overlap, then first-step overlap, then stable IDs.
        first, second = min(
            pairs, key=lambda pair: (-(score(pair[0], interests) + score(pair[1], interests)),
                                     -score(pair[0], interests), pair[0]["id"], pair[1]["id"]))
        journey = {"status": "ready", "reason": None,
                   "steps": [dict(describe(action, interests), position=index,
                                  prerequisites=sorted(action["prerequisites"]))
                             for index, action in enumerate((first, second), 1)]}
    result = {"status": "ok", "schema_version": 1, "synthetic": True,
              "recommendations": [describe(action, interests) for action in firsts],
              "journey": journey}
    validate_output(result, data)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"invalid JSON constant: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_text(encoding="utf-8")
        data = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = recommend(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
