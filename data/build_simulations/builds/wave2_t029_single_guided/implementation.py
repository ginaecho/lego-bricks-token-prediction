"""Synthetic guided onboarding reference; Python standard library only.

Input: schema_version=1, fixture_label, steps, optional initial_completed/actions.
Steps: id, title, prerequisites (IDs), fields mapping answer names to
{type: string|integer|boolean, required: bool (default true), choices?: list}.
Actions: {step_id, answers}. Completion requires all prerequisites and valid
answers. initial_completed is a trusted caller-owned checkpoint, not proof of
identity or permission. Unknown properties are rejected at every schema level.
No state is written; reuse output.state.completed as input.initial_completed.
"""

import json
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_keys(value, required, optional, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(required <= value.keys(), f"{path}: missing required properties")
    require(value.keys() <= required | optional, f"{path}: unknown properties")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()),
            f"{path}: expected nonempty string")


def identifiers(value, path):
    require(isinstance(value, list), f"{path}: expected array")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), f"{path}: duplicate identifiers")


def answer_value(value, spec, path):
    kind = spec["type"]
    valid = ((kind == "string" and isinstance(value, str) and bool(value.strip()))
             or (kind == "integer" and type(value) is int)
             or (kind == "boolean" and type(value) is bool))
    require(valid, f"{path}: expected {kind}")
    if "choices" in spec:
        require(value in spec["choices"], f"{path}: value not in choices")


def validate(data):
    object_keys(data, {"schema_version", "fixture_label", "steps"},
                {"initial_completed", "actions"}, "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version: expected integer 1")
    text(data["fixture_label"], "fixture_label")
    steps = data["steps"]
    require(isinstance(steps, list) and len(steps) > 0,
            "steps: expected nonempty array")
    by_id = {}
    for step in steps:
        object_keys(step, {"id", "title", "prerequisites", "fields"}, set(), "step")
        text(step["id"], "step.id")
        text(step["title"], "step.title")
        require(step["id"] not in by_id, "steps: duplicate id")
        by_id[step["id"]] = step
        identifiers(step["prerequisites"], "step.prerequisites")
        require(isinstance(step["fields"], dict), "step.fields: expected object")
        for name, spec in step["fields"].items():
            text(name, "field name")
            object_keys(spec, {"type"}, {"required", "choices"}, f"field {name}")
            require(spec["type"] in ("string", "integer", "boolean"),
                    f"field {name}: unsupported type")
            require(type(spec.get("required", True)) is bool,
                    f"field {name}: required must be boolean")
            if "choices" in spec:
                choices = spec["choices"]
                require(isinstance(choices, list) and len(choices) > 0,
                        f"field {name}: choices must be nonempty array")
                for choice in choices:
                    answer_value(choice, {"type": spec["type"]}, f"field {name}.choices")
                require(len(choices) == len(set(choices)),
                        f"field {name}: duplicate choices")
    for step in steps:
        require(set(step["prerequisites"]) <= by_id.keys(),
                f"step {step['id']}: unknown prerequisite")
    remaining = set(by_id)
    reached = set()
    while remaining:
        ready = {sid for sid in remaining
                 if set(by_id[sid]["prerequisites"]) <= reached}
        require(bool(ready), "steps: prerequisite cycle")
        reached.update(ready)
        remaining.difference_update(ready)
    initial = data.get("initial_completed", [])
    identifiers(initial, "initial_completed")
    require(set(initial) <= by_id.keys(), "initial_completed: unknown step")
    for sid in initial:
        require(set(by_id[sid]["prerequisites"]) <= set(initial),
                f"initial_completed: unmet prerequisite for {sid}")
    actions = data.get("actions", [])
    require(isinstance(actions, list), "actions: expected array")
    for action in actions:
        object_keys(action, {"step_id", "answers"}, set(), "action")
        text(action["step_id"], "action.step_id")
        require(action["step_id"] in by_id, "action: unknown step")
        require(isinstance(action["answers"], dict), "action.answers: expected object")
        fields = by_id[action["step_id"]]["fields"]
        require(action["answers"].keys() <= fields.keys(), "answers: unknown field")
        for name, spec in fields.items():
            require(not spec.get("required", True) or name in action["answers"],
                    f"answers: missing required field {name}")
            if name in action["answers"]:
                answer_value(action["answers"][name], spec, f"answers.{name}")
    return by_id


def run(data):
    """Validate the whole request, then apply ordered completions atomically."""
    by_id = validate(data)
    completed = set(data.get("initial_completed", []))
    events = []
    for index, action in enumerate(data.get("actions", [])):
        sid = action["step_id"]
        require(sid not in completed, f"action {index}: step {sid} already completed")
        require(set(by_id[sid]["prerequisites"]) <= completed,
                f"action {index}: step {sid} has unmet prerequisites")
        completed.add(sid)
        events.append({"action_index": index, "step_id": sid, "status": "completed"})
    states = []
    for sid, step in by_id.items():
        missing = [p for p in step["prerequisites"] if p not in completed]
        status = "completed" if sid in completed else "locked" if missing else "available"
        states.append({"id": sid, "title": step["title"], "status": status,
                       "missing_prerequisites": missing,
                       "required_fields": [name for name, spec in step["fields"].items()
                                           if spec.get("required", True)]})
    available = [step["id"] for step in states if step["status"] == "available"]
    return {
        "schema_version": 1, "status": "ok", "fixture_label": data["fixture_label"],
        "state": {"completed": [sid for sid in by_id if sid in completed]},
        "progress": {"completed": len(completed), "total": len(by_id),
                     "percent": round(100 * len(completed) / len(by_id), 2),
                     "finished": len(completed) == len(by_id)},
        "steps": states, "next_steps": available, "events": events,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON: duplicate property {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"JSON: nonstandard constant {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
