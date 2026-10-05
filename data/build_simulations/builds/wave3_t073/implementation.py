"""Deterministic, standard-library guided onboarding reference CLI."""

import json
import sys


class ValidationError(ValueError):
    pass


DEFAULT_STEPS = [
    {"id": "setup", "title": "Set up your profile",
     "instruction": "Confirm your preferred name and account preferences.",
     "requires": [], "completed": False},
    {"id": "first_action", "title": "Complete your first meaningful action",
     "instruction": "Choose one small action that advances your stated goal and complete it.",
     "requires": ["setup"], "completed": False},
    {"id": "review", "title": "Review your first result",
     "instruction": "Check the outcome and choose what to try next.",
     "requires": ["first_action"], "completed": False},
]


def object_fields(value, required, optional, path):
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValidationError(f"{path} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValidationError(f"{path} unknown fields: {', '.join(sorted(unknown))}")


def text(value, path):
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise ValidationError(f"{path} must be nonblank text of at most 1000 characters")
    return value.strip()


def validate(payload):
    """The shared validation boundary; returns a normalized independent value."""
    object_fields(payload, {"schema_version", "customer"}, {"steps", "fixture_label"}, "input")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    customer = payload["customer"]
    object_fields(customer, {"name", "goal", "experience"}, set(), "customer")
    customer = {key: text(customer[key], f"customer.{key}") for key in customer}
    if customer["experience"] not in {"beginner", "experienced"}:
        raise ValidationError("customer.experience must be beginner or experienced")
    label = payload.get("fixture_label")
    if "fixture_label" in payload:
        label = text(label, "fixture_label")
    raw_steps = payload.get("steps", DEFAULT_STEPS)
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 100:
        raise ValidationError("steps must be a list containing 1 to 100 steps")
    steps = []
    seen = set()
    for index, step in enumerate(raw_steps):
        path = f"steps[{index}]"
        object_fields(step, {"id", "title", "instruction", "requires", "completed"}, set(), path)
        normalized = {key: text(step[key], f"{path}.{key}")
                      for key in ("id", "title", "instruction")}
        if normalized["id"] in seen:
            raise ValidationError("step IDs must be unique")
        seen.add(normalized["id"])
        if type(step["completed"]) is not bool:
            raise ValidationError(f"{path}.completed must be boolean")
        if not isinstance(step["requires"], list):
            raise ValidationError(f"{path}.requires must be a list")
        requires = [text(item, f"{path}.requires") for item in step["requires"]]
        if len(set(requires)) != len(requires):
            raise ValidationError(f"{path}.requires contains duplicates")
        normalized.update(requires=requires, completed=step["completed"])
        steps.append(normalized)
    by_id = {step["id"]: step for step in steps}
    for step in steps:
        if any(item not in by_id for item in step["requires"]):
            raise ValidationError(f"step {step['id']} has an unknown prerequisite")
        if step["completed"] and any(not by_id[item]["completed"] for item in step["requires"]):
            raise ValidationError(f"completed step {step['id']} has incomplete prerequisites")
    resolved = set()
    while len(resolved) < len(steps):
        ready = {step["id"] for step in steps
                 if step["id"] not in resolved and set(step["requires"]) <= resolved}
        if not ready:
            raise ValidationError("step prerequisites must not contain cycles")
        resolved.update(ready)
    return {"schema_version": 1, "customer": customer, "steps": steps, "fixture_label": label}


def onboard(payload):
    data = validate(payload)
    customer, steps = data["customer"], data["steps"]
    done = {step["id"] for step in steps if step["completed"]}
    eligible = [step for step in steps
                if not step["completed"] and set(step["requires"]) <= done]
    next_step = eligible[0] if eligible else None
    result = {
        "schema_version": 1,
        "status": "ok",
        "fixture_label": data["fixture_label"],
        "customer": customer,
        "onboarding_status": "complete" if next_step is None else "in_progress",
        "progress": {"completed": len(done), "total": len(steps),
                     "percent": round(100 * len(done) / len(steps), 2)},
        "next_step": None,
        "message": f"{customer['name']}, your onboarding for '{customer['goal']}' is complete.",
    }
    if next_step:
        guidance = (
            "Take this one step at a time; finish this action before moving on."
            if customer["experience"] == "beginner"
            else "Use your existing workflow to complete this action."
        )
        result["next_step"] = {
            "id": next_step["id"], "title": next_step["title"],
            "instruction": next_step["instruction"],
            "guidance": guidance,
            "reason": f"This is the first unfinished step with all prerequisites met for your goal: {customer['goal']}.",
        }
        result["message"] = (
            f"{customer['name']}, to work toward '{customer['goal']}', "
            f"your next step is: {next_step['title']}."
        )
    return result


def reject_constant(value):
    raise ValidationError(f"invalid JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            payload = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = onboard(payload)
    except (OSError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
