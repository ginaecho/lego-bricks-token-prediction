"""Deterministic guided onboarding reference implementation; standard library only."""

import json
import sys
from pathlib import Path


CATALOG = {
    "orientation": ("Understand the storefront", (), "Learn how products, prices and checkout connect."),
    "catalog": ("Create a product listing", (), "A listing gives shoppers something to evaluate."),
    "pricing": ("Set a sustainable price", ("catalog",), "Price a listed product before offering it for sale."),
    "payments": ("Configure test checkout", (), "Verify the checkout path before accepting orders."),
    "publish": ("Review and publish the store", ("pricing", "payments"), "Publish only after listings, pricing and checkout are ready."),
    "analytics": ("Read synthetic sales metrics", ("publish",), "Use a launched store's metrics to identify opportunities."),
    "optimization": ("Plan one sales experiment", ("analytics",), "Choose an experiment based on measured evidence."),
}
GOALS = {"launch_store": "publish", "improve_sales": "optimization"}


class ValidationError(ValueError):
    def __init__(self, field, message):
        self.field = field
        super().__init__(message)


def require(condition, field, message):
    if not condition:
        raise ValidationError(field, message)


def fields(value, expected, field):
    require(isinstance(value, dict), field, "Must be an object.")
    require(set(value) == set(expected), field,
            "Expected exactly these fields: " + ", ".join(sorted(expected)))


def choice(value, options, field):
    require(isinstance(value, str) and value in options, field,
            "Must be one of: " + ", ".join(sorted(options)))


def unique_choices(value, options, field, nonempty=False):
    require(isinstance(value, list), field, "Must be an array.")
    require(not nonempty or bool(value), field, "Must contain at least one item.")
    for index, item in enumerate(value):
        choice(item, options, f"{field}[{index}]")
    require(len(set(value)) == len(value), field, "Duplicate items are not allowed.")


def dependencies(step, experience):
    deps = list(CATALOG[step][1])
    if experience == "novice" and step in ("catalog", "payments"):
        deps.insert(0, "orientation")
    return deps


def validate_input(payload):
    """The single shared validation boundary for direct callers and the CLI."""
    fields(payload, ("schema_version", "fixture", "experience", "preferences",
                     "goals", "completed_steps"), "$")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version", "Must be integer 1.")
    fields(payload["fixture"], ("kind", "id"), "fixture")
    require(payload["fixture"]["kind"] == "synthetic", "fixture.kind",
            "Only clearly labeled synthetic fixtures are supported.")
    identifier = payload["fixture"]["id"]
    require(isinstance(identifier, str) and identifier.startswith("synthetic-")
            and 10 < len(identifier) <= 100, "fixture.id",
            "Must start with synthetic- and contain a suffix (maximum 100 characters).")
    choice(payload["experience"], ("novice", "intermediate", "expert"), "experience")
    fields(payload["preferences"], ("format", "pace"), "preferences")
    choice(payload["preferences"]["format"], ("explanation", "practice"),
           "preferences.format")
    choice(payload["preferences"]["pace"], ("focused", "batch"), "preferences.pace")
    unique_choices(payload["goals"], GOALS, "goals", nonempty=True)
    unique_choices(payload["completed_steps"], CATALOG, "completed_steps")
    completed = set(payload["completed_steps"])
    for step in payload["completed_steps"]:
        missing = set(dependencies(step, payload["experience"])) - completed
        require(not missing, "completed_steps",
                f"Completed step {step} requires completed prerequisites: "
                + ", ".join(sorted(missing)))
    return payload


def onboarding(payload):
    payload = validate_input(payload)
    experience = payload["experience"]
    preferences = payload["preferences"]
    completed = set(payload["completed_steps"])
    required = set()
    reasons = {}

    def include(step, goal):
        required.add(step)
        reasons.setdefault(step, set()).add(goal)
        for prerequisite in dependencies(step, experience):
            include(prerequisite, goal)

    for goal in sorted(payload["goals"]):
        include(GOALS[goal], goal)

    guidance_levels = {"novice": "guided", "intermediate": "standard", "expert": "concise"}
    steps = []
    for identifier, (title, _, explanation) in CATALOG.items():
        if identifier not in required:
            continue
        prerequisites = dependencies(identifier, experience)
        blocked_by = [item for item in prerequisites if item not in completed]
        state = ("complete" if identifier in completed else
                 "blocked" if blocked_by else "ready")
        guidance = f"{title}."
        if experience != "expert":
            guidance += " " + explanation
        if experience == "novice":
            guidance += " Work through one decision at a time and review before continuing."
        if preferences["format"] == "practice":
            guidance += " Practice with a synthetic example, then check the result."
        else:
            guidance += " Read the rationale and review a synthetic example."
        steps.append({
            "id": identifier, "title": title, "state": state,
            "prerequisites": prerequisites, "blocked_by": blocked_by,
            "serves_goals": sorted(reasons[identifier]),
            "explanation": explanation, "guidance": guidance,
            "guidance_level": guidance_levels[experience],
        })
    ready = [step["id"] for step in steps if step["state"] == "ready"]
    done = sum(step["state"] == "complete" for step in steps)
    limit = 1 if preferences["pace"] == "focused" else 3
    return {
        "schema_version": 1, "fixture": dict(payload["fixture"]),
        "experience": experience, "preferences": dict(preferences),
        "goals": sorted(payload["goals"]), "steps": steps,
        "next_actions": ready[:limit],
        "progress": {"completed": done, "total": len(steps),
                     "percent": round(100 * done / len(steps), 2)},
        "onboarding_complete": done == len(steps),
        "adaptation_explanation": (
            f"{experience} experience uses {guidance_levels[experience]} guidance; "
            f"{preferences['format']} preference controls learning activity; "
            f"{preferences['pace']} pace offers at most {limit} ready action(s). "
            "Novices must complete orientation before listings or checkout. "
            "Experience never bypasses operational prerequisites."
        ),
    }


def response(payload):
    try:
        return {"status": "ok", "data": onboarding(payload), "errors": []}
    except ValidationError as error:
        return {"status": "error", "data": None,
                "errors": [{"field": error.field, "message": str(error)}]}


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant is not supported: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "$cli", "Usage: python -B implementation.py INPUT.json")
        payload = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                             parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = response(payload)
    except (OSError, ValueError, RecursionError) as error:
        result = {"status": "error", "data": None,
                  "errors": [{"field": getattr(error, "field", "$file"),
                              "message": str(error)}]}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
