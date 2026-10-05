"""Deterministic synthetic guided onboarding; Python standard library only."""

import json
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


class Validator:
    """Shared validation for the request, definitions, and action payloads."""

    @staticmethod
    def object(value, required, optional=(), path="input"):
        if not isinstance(value, dict):
            raise ValidationError(f"{path} must be an object")
        missing = set(required) - value.keys()
        extra = value.keys() - set(required) - set(optional)
        if missing or extra:
            raise ValidationError(
                f"{path}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}"
            )
        return value

    @staticmethod
    def text(value, path):
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{path} must be nonblank text")
        return value

    @staticmethod
    def array(value, path):
        if not isinstance(value, list):
            raise ValidationError(f"{path} must be an array")
        return value

    @classmethod
    def names(cls, value, path):
        result = cls.array(value, path)
        for item in result:
            cls.text(item, path)
        if len(set(result)) != len(result):
            raise ValidationError(f"{path} must not contain duplicates")
        return result


def validate_request(request):
    v = Validator
    v.object(request, ("schema_version", "fixture_label", "steps", "actions"))
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    v.text(request["fixture_label"], "fixture_label")
    steps = v.array(request["steps"], "steps")
    index = {}
    for number, step in enumerate(steps):
        path = f"steps[{number}]"
        v.object(step, ("id", "title", "prerequisites", "required_fields"), path=path)
        v.text(step["id"], path + ".id")
        v.text(step["title"], path + ".title")
        v.names(step["prerequisites"], path + ".prerequisites")
        v.names(step["required_fields"], path + ".required_fields")
        if step["id"] in index:
            raise ValidationError(f"duplicate step id: {step['id']}")
        index[step["id"]] = step
    for step in steps:
        for prerequisite in step["prerequisites"]:
            if prerequisite not in index:
                raise ValidationError(f"unknown prerequisite: {prerequisite}")
    # Iterative topological validation also handles graphs deeper than recursion limits.
    remaining = set(index)
    resolved = set()
    while remaining:
        ready = {name for name in remaining
                 if set(index[name]["prerequisites"]) <= resolved}
        if not ready:
            raise ValidationError("prerequisite graph contains a cycle")
        resolved.update(ready)
        remaining.difference_update(ready)
    for number, action in enumerate(v.array(request["actions"], "actions")):
        path = f"actions[{number}]"
        v.object(action, ("step_id", "operation"), ("answers",), path)
        v.text(action["step_id"], path + ".step_id")
        if action["step_id"] not in index:
            raise ValidationError(f"{path}: unknown step")
        v.text(action["operation"], path + ".operation")
        if action["operation"] not in ("start", "complete"):
            raise ValidationError(f"{path}: operation must be start or complete")
        fields = (index[action["step_id"]]["required_fields"]
                  if action["operation"] == "complete" else ())
        answers = v.object(action.get("answers", {}), fields, path=path + ".answers")
        for key, value in answers.items():
            v.text(value, path + ".answers." + key)
    return index


def run(request):
    """Replay actions atomically; invalid requests raise without mutating input."""
    index = validate_request(request)
    states = {name: "pending" for name in index}
    evidence = {name: {} for name in index}
    events = []
    for number, action in enumerate(request["actions"]):
        name = action["step_id"]
        operation = action["operation"]
        blocked = [p for p in index[name]["prerequisites"] if states[p] != "completed"]
        if blocked:
            raise ValidationError(f"actions[{number}]: prerequisites incomplete: {blocked}")
        expected = "pending" if operation == "start" else "in_progress"
        if states[name] != expected:
            raise ValidationError(
                f"actions[{number}]: {operation} requires {expected}, got {states[name]}"
            )
        states[name] = "in_progress" if operation == "start" else "completed"
        if operation == "complete":
            evidence[name] = dict(action.get("answers", {}))
        events.append({"sequence": number + 1, "step_id": name,
                       "operation": operation, "state": states[name]})
    completed = sum(state == "completed" for state in states.values())
    total = len(states)
    results = []
    for name, step in index.items():
        blocked = [p for p in step["prerequisites"] if states[p] != "completed"]
        results.append({
            "id": name, "title": step["title"], "state": states[name],
            "blocked_by": blocked,
            "can_start": states[name] == "pending" and not blocked,
            "required_fields": list(step["required_fields"]),
            "answers": evidence[name],
        })
    return {
        "schema_version": 1, "status": "ok", "fixture_label": request["fixture_label"],
        "steps": results, "events": events,
        "progress": {
            "total": total, "completed": completed,
            "in_progress": sum(state == "in_progress" for state in states.values()),
            "pending": sum(state == "pending" for state in states.values()),
            "percent_complete": round(100 * completed / total, 2) if total else 100.0,
            "finished": completed == total,
        },
        "next_steps": [step["id"] for step in results
                       if step["can_start"] or step["state"] == "in_progress"],
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonfinite JSON number: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_text(encoding="utf-8")
        request = json.loads(raw, object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = run(request)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error",
                          "error": {"code": "invalid_input", "message": str(exc)}}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
