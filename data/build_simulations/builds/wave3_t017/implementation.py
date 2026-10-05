"""Deterministic, synthetic-fixture-friendly guided onboarding reference CLI."""

import json
import sys
from pathlib import Path


STEPS = {
    "verify_email": ("Verify your email", "Open your welcome email and follow the verification link.", 2),
    "complete_profile": ("Complete your profile", "Add your display name and workspace details in Settings > Profile.", 3),
    "create_project": ("Create your first project", "Open Projects, choose New project, and give it a name.", 5),
    "invite_teammate": ("Invite your first teammate", "Open Members, choose Invite, and enter a teammate's email.", 3),
    "take_tour": ("Take the product tour", "Open Help > Product tour and follow the introductory walkthrough.", 4),
}
GOALS = {
    "launch_project": ("verify_email", "complete_profile", "create_project"),
    "collaborate": ("verify_email", "complete_profile", "invite_teammate"),
    "explore_product": ("verify_email", "complete_profile", "take_tour"),
}


class ValidationError(ValueError):
    pass


def fields(value, required, location):
    if not isinstance(value, dict):
        raise ValidationError(f"{location} must be an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required)
    if missing or extra:
        raise ValidationError(
            f"{location}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}"
        )


def text(value, location):
    if not isinstance(value, str) or not value.strip() or len(value) > 120:
        raise ValidationError(f"{location} must be nonblank text of at most 120 characters")
    return value.strip()


def validate_input(payload):
    """Single validation boundary shared by direct callers and the CLI."""
    fields(payload, ("schema_version", "synthetic_fixture", "customer", "goal", "completed_steps"), "input")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    if type(payload["synthetic_fixture"]) is not bool:
        raise ValidationError("synthetic_fixture must be boolean")
    customer = payload["customer"]
    fields(customer, ("id", "name", "experience"), "customer")
    customer_id = text(customer["id"], "customer.id")
    name = text(customer["name"], "customer.name")
    experience = customer["experience"]
    if not isinstance(experience, str) or experience not in ("beginner", "experienced"):
        raise ValidationError("customer.experience must be beginner or experienced")
    goal = payload["goal"]
    if not isinstance(goal, str) or goal not in GOALS:
        raise ValidationError(f"goal must be one of {', '.join(GOALS)}")
    completed = payload["completed_steps"]
    if not isinstance(completed, list) or any(not isinstance(item, str) for item in completed):
        raise ValidationError("completed_steps must be an array of step IDs")
    if len(completed) != len(set(completed)):
        raise ValidationError("completed_steps must not contain duplicates")
    plan = GOALS[goal]
    if any(item not in plan for item in completed):
        raise ValidationError("completed_steps contains a step outside this goal's plan")
    # Completion order is irrelevant, but dependent steps cannot precede prerequisites.
    if set(completed) != set(plan[:len(completed)]):
        raise ValidationError("completed_steps must satisfy the plan's prerequisites")
    return {
        "schema_version": 1,
        "synthetic_fixture": payload["synthetic_fixture"],
        "customer": {"id": customer_id, "name": name, "experience": experience},
        "goal": goal,
        "completed_steps": list(plan[:len(completed)]),
    }


def onboard(payload):
    data = validate_input(payload)
    customer = data["customer"]
    sequence = GOALS[data["goal"]]
    count = len(data["completed_steps"])
    pending = sequence[count:]
    next_step = None
    if pending:
        step_id = pending[0]
        title, instructions, minutes = STEPS[step_id]
        guidance = (
            "Go one step at a time; use Help if you get stuck."
            if customer["experience"] == "beginner"
            else "You can use the direct navigation path above."
        )
        next_step = {
            "id": step_id,
            "title": title,
            "instructions": f"{customer['name']}, {instructions[0].lower()}{instructions[1:]}",
            "guidance": guidance,
            "estimated_minutes": minutes,
            "reason": f"This is the first unfinished prerequisite in your {data['goal']} plan.",
        }
    return {
        "schema_version": 1,
        "synthetic_fixture": data["synthetic_fixture"],
        "status": "ready" if pending else "completed",
        "customer_id": customer["id"],
        "goal": data["goal"],
        "progress": {
            "completed": count,
            "total": len(sequence),
            "percent": round(count / len(sequence) * 100),
        },
        "plan": [
            {"id": step, "title": STEPS[step][0], "completed": index < count}
            for index, step in enumerate(sequence)
        ],
        "next_step": next_step,
        "message": (
            f"{customer['name']}, your next step is: {next_step['title']}."
            if next_step
            else f"{customer['name']}, onboarding is complete. Continue from your workspace dashboard."
        ),
    }


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"Nonstandard JSON number: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)
        result = onboard(payload)
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
