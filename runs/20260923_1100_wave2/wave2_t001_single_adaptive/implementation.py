"""Synthetic, deterministic adaptive onboarding reference CLI (standard library only)."""

import json
import sys
from pathlib import Path


BUILD_ID = "wave2_t001_single_adaptive"
EXPERIENCES = ("beginner", "intermediate", "advanced")
PREFERENCES = ("guided", "concise")
GOALS = ("first_sale", "manage_catalog")
STEPS = (
    {
        "id": "account",
        "title": "Create a seller account",
        "requires": (),
        "why": "An account provides the identity that owns your shop.",
        "action": "Create and verify your synthetic seller account.",
        "help": "Choose a shop email, create a password, and confirm the verification message.",
    },
    {
        "id": "seller_profile",
        "title": "Complete your seller profile",
        "requires": ("account",),
        "why": "A seller profile identifies the shop to buyers.",
        "action": "Add a shop name and a short shop description.",
        "help": "Use a recognizable name and explain what your shop sells.",
    },
    {
        "id": "payout",
        "title": "Configure a payout method",
        "requires": ("seller_profile",),
        "why": "A payout method is required before this synthetic shop can accept a sale.",
        "action": "Select a synthetic payout method.",
        "help": "Use the fixture payout option only; never enter real financial information.",
    },
    {
        "id": "catalog",
        "title": "Set up your catalog",
        "requires": ("seller_profile",),
        "why": "A catalog organizes products and their stock levels.",
        "action": "Create a product category and choose an inventory tracking convention.",
        "help": "Start with one category and assign a unique stock code to each product.",
    },
    {
        "id": "listing",
        "title": "Create your first listing",
        "requires": ("catalog",),
        "why": "A listing gives buyers the product details they need to decide.",
        "action": "Add a synthetic product title, description, price, and stock quantity.",
        "help": "Preview the listing and check that the description matches the fixture product.",
    },
    {
        "id": "publish",
        "title": "Publish your first listing",
        "requires": ("payout", "listing"),
        "why": "Publishing makes the prepared listing available in this simulated shop.",
        "action": "Review the listing and confirm simulated publication.",
        "help": "Check the price, stock, and payout setup before confirming.",
    },
)
STEP_MAP = {step["id"]: step for step in STEPS}
TARGETS = {"first_sale": "publish", "manage_catalog": "catalog"}


class ValidationError(ValueError):
    def __init__(self, message, path="$", code="validation_error"):
        super().__init__(message)
        self.path = path
        self.code = code


def require(condition, message, path="$"):
    if not condition:
        raise ValidationError(message, path)


def validate_input(value):
    """Single shared validation boundary for both the Python API and CLI."""
    require(type(value) is dict, "Input must be an object.")
    required = {"schema_version", "synthetic", "profile", "goal"}
    allowed = required | {"completed_steps"}
    require(required <= value.keys(), "Missing fields: " + ", ".join(sorted(required - value.keys())))
    require(value.keys() <= allowed, "Unknown fields: " + ", ".join(sorted(value.keys() - allowed)))
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "schema_version must be integer 1.", "$.schema_version")
    require(value["synthetic"] is True, "synthetic must be true.", "$.synthetic")
    profile = value["profile"]
    require(type(profile) is dict, "profile must be an object.", "$.profile")
    require(set(profile) == {"experience", "preference"},
            "profile must contain exactly experience and preference.", "$.profile")
    require(type(profile["experience"]) is str and profile["experience"] in EXPERIENCES,
            "experience must be beginner, intermediate, or advanced.", "$.profile.experience")
    require(type(profile["preference"]) is str and profile["preference"] in PREFERENCES,
            "preference must be guided or concise.", "$.profile.preference")
    require(type(value["goal"]) is str and value["goal"] in GOALS,
            "goal must be first_sale or manage_catalog.", "$.goal")
    completed = value.get("completed_steps", [])
    require(type(completed) is list, "completed_steps must be an array.", "$.completed_steps")
    for index, step_id in enumerate(completed):
        require(type(step_id) is str and step_id in STEP_MAP,
                "Unknown step ID.", "$.completed_steps[" + str(index) + "]")
    require(len(set(completed)) == len(completed),
            "completed_steps must not contain duplicates.", "$.completed_steps")
    completed_set = set(completed)
    for step_id in completed:
        missing = set(STEP_MAP[step_id]["requires"]) - completed_set
        require(not missing, "Completed step " + step_id + " is missing prerequisites: "
                + ", ".join(sorted(missing)), "$.completed_steps")
    return {
        "schema_version": 1,
        "synthetic": True,
        "profile": dict(profile),
        "goal": value["goal"],
        "completed_steps": [step["id"] for step in STEPS if step["id"] in completed_set],
    }


def prerequisite_closure(step_id):
    result = {step_id}
    for prerequisite in STEP_MAP[step_id]["requires"]:
        result.update(prerequisite_closure(prerequisite))
    return result


def onboard(value):
    request = validate_input(value)
    completed = set(request["completed_steps"])
    required = prerequisite_closure(TARGETS[request["goal"]])
    profile = request["profile"]
    experience = profile["experience"]
    preference = profile["preference"]
    detail = "expanded" if experience == "beginner" else "standard" if experience == "intermediate" else "brief"
    plan = []
    for step in STEPS:
        if step["id"] not in required or step["id"] in completed:
            continue
        unmet = [item for item in step["requires"] if item not in completed]
        instructions = [step["action"]]
        if preference == "guided" or experience == "beginner":
            instructions.append(step["help"])
        if experience == "beginner" and preference == "guided":
            instructions.append("Finish this step before moving to any step that depends on it.")
        plan.append({
            "id": step["id"],
            "title": step["title"],
            "prerequisites": list(step["requires"]),
            "unmet_prerequisites": unmet,
            "state": "blocked" if unmet else "ready",
            "explanation": step["why"],
            "instructions": instructions,
        })
    ready = [step["id"] for step in plan if step["state"] == "ready"]
    completed_count = len(required & completed)
    return {
        "schema_version": 1,
        "status": "ok",
        "synthetic": True,
        "onboarding": {
            "goal": request["goal"],
            "profile": profile,
            "presentation": {
                "mode": "walkthrough" if preference == "guided" else "checklist",
                "detail": detail,
                "explanation": (
                    "Experience adjusts instructional detail; preference selects presentation. "
                    "Prerequisites are never skipped based on experience."
                ),
            },
            "completed_steps": request["completed_steps"],
            "progress": {"completed": completed_count, "total": len(required)},
            "complete": not plan,
            "next_step": ready[0] if ready else None,
            "ready_steps": ready,
            "steps": plan,
        },
    }


def reject_constant(value):
    raise ValidationError("Non-finite JSON numbers are not supported.", code="invalid_json")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate JSON key: " + key, code="invalid_json")
        result[key] = value
    return result


def error_result(error):
    return {
        "schema_version": 1,
        "status": "error",
        "error": {"code": error.code, "path": error.path, "message": str(error)},
    }


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json", code="usage_error")
        try:
            text = Path(args[0]).read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValidationError("Cannot read input file: " + str(exc), code="file_error") from exc
        try:
            value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except (ValueError, RecursionError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("Invalid JSON: " + str(exc), code="invalid_json") from exc
        result = onboard(value)
    except ValidationError as exc:
        print(json.dumps(error_result(exc), ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
