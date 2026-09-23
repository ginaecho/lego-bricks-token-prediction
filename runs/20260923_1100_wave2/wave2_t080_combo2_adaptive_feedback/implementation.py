"""Synthetic, deterministic adaptive-onboarding -> feedback CLI (standard library).

Run: python -B implementation.py example_input.json
Feedback is deduplicated within an onboarding step, never across unrelated steps.
Every supporting excerpt preserves its exact source text and identifier.
"""

import copy
import json
import re
import sys
import unicodedata


class ValidationError(ValueError):
    """A document or a pipeline handoff violates the shared contract."""


def obj(**fields):
    return {"type": "object", "fields": fields}


def arr(item):
    return {"type": "array", "item": item}


def enum(*values):
    return {"type": "string", "enum": values}


TEXT = {"type": "string"}
COUNT = {"type": "integer"}
BOOL = {"type": "boolean"}
VERSION = {"type": "integer", "enum": (1,)}
STEP_ID = enum("account", "listings", "analytics")
PROFILE = obj(
    experience=enum("beginner", "intermediate", "advanced"),
    preference=enum("guided", "concise"),
    goals=arr(enum("start", "sell", "track")),
    completed_steps=arr(STEP_ID),
)
SOURCE = obj(id=TEXT, step_id=STEP_ID, text=TEXT)
INPUT = obj(schema_version=VERSION, synthetic=BOOL, profile=PROFILE,
            feedback=arr(SOURCE))
STEP = obj(id=STEP_ID, prerequisites=arr(STEP_ID),
           status=enum("completed", "planned"), explanation=TEXT,
           instructions=arr(TEXT))
ONBOARDING = obj(profile=PROFILE, steps=arr(STEP), source_feedback=arr(SOURCE))
GROUP = obj(id=TEXT, step_id=STEP_ID, step_status=enum("completed", "planned"),
            canonical_text=TEXT, source_ids=arr(TEXT), excerpts=arr(SOURCE))
THEME = obj(name=TEXT, group_ids=arr(TEXT), supporting_excerpts=arr(SOURCE),
            recommendation=TEXT)
FEEDBACK = obj(source_count=COUNT, unique_count=COUNT, duplicate_count=COUNT,
               groups=arr(GROUP), themes=arr(THEME))
OUTPUT = obj(schema_version=VERSION, status=enum("ok"), synthetic=BOOL,
             onboarding=ONBOARDING, feedback=FEEDBACK)

CATALOG = {
    "account": {
        "prerequisites": [],
        "instructions": ["Create your seller account.", "Set your contact preferences.",
                         "Review and save the account settings."],
        "why": "An account supplies the identity and preferences used by later steps.",
    },
    "listings": {
        "prerequisites": ["account"],
        "instructions": ["Create a synthetic product listing.", "Set a sample price.",
                         "Preview the listing before saving."],
        "why": "Listings require an account and enable your selling goal.",
    },
    "analytics": {
        "prerequisites": ["listings"],
        "instructions": ["Open the sample sales dashboard.", "Choose a reporting period.",
                         "Compare the sample sales totals."],
        "why": "Analytics requires listings so that sales measurements have context.",
    },
}
GOALS = {"start": "account", "sell": "listings", "track": "analytics"}
KEYWORDS = {
    "setup": {"account", "setup", "login", "register"},
    "navigation": {"find", "navigation", "menu", "dashboard"},
    "pricing": {"price", "pricing", "cost", "expensive"},
    "learning": {"confusing", "instructions", "help", "explanation"},
    "performance": {"slow", "loading", "timeout"},
}


def validate(value, schema, path="$"):
    """One strict structural validator is shared by input, handoffs and output."""
    kind = schema["type"]
    expected = {"object": dict, "array": list, "string": str,
                "integer": int, "boolean": bool}[kind]
    if type(value) is not expected:
        raise ValidationError(f"{path}: expected {kind}")
    if kind == "object":
        fields = schema["fields"]
        if set(value) != set(fields):
            raise ValidationError(f"{path}: expected fields {sorted(fields)}")
        for key, child in fields.items():
            validate(value[key], child, f"{path}.{key}")
    elif kind == "array":
        for index, child in enumerate(value):
            validate(child, schema["item"], f"{path}[{index}]")
    elif kind == "string" and not value.strip():
        raise ValidationError(f"{path}: empty string")
    elif kind == "integer" and value < 0:
        raise ValidationError(f"{path}: must be nonnegative")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: unsupported value")


def unique(values, label):
    if len(values) != len(set(values)):
        raise ValidationError(f"{label}: duplicate values")


def canonical(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def validate_profile(profile):
    validate(profile, PROFILE, "$.profile")
    if not profile["goals"]:
        raise ValidationError("$.profile.goals: at least one goal is required")
    unique(profile["goals"], "goals")
    unique(profile["completed_steps"], "completed_steps")
    completed = set(profile["completed_steps"])
    for step in completed:
        if not set(CATALOG[step]["prerequisites"]) <= completed:
            raise ValidationError(f"completed step {step}: missing prerequisite")


def validate_sources(sources):
    validate(sources, arr(SOURCE), "$.feedback")
    unique([source["id"] for source in sources], "feedback IDs")
    for source in sources:
        if not canonical(source["text"]):
            raise ValidationError(f"feedback {source['id']}: text has no word characters")


def required_steps(profile):
    required = set(profile["completed_steps"])

    def include(step):
        required.add(step)
        for prerequisite in CATALOG[step]["prerequisites"]:
            include(prerequisite)

    for goal in profile["goals"]:
        include(GOALS[goal])
    return [step for step in CATALOG if step in required]


def validate_onboarding(handoff):
    validate(handoff, ONBOARDING, "$.onboarding")
    profile = handoff["profile"]
    validate_profile(profile)
    validate_sources(handoff["source_feedback"])
    steps = handoff["steps"]
    if [step["id"] for step in steps] != required_steps(profile):
        raise ValidationError("onboarding: incorrect prerequisite order or missing steps")
    for step in steps:
        expected_status = ("completed" if step["id"] in profile["completed_steps"]
                           else "planned")
        if step["prerequisites"] != CATALOG[step["id"]]["prerequisites"]:
            raise ValidationError("onboarding: incorrect prerequisites")
        if step["status"] != expected_status:
            raise ValidationError("onboarding: inconsistent completed status")
        if not step["instructions"]:
            raise ValidationError("onboarding: missing instructions")
    selected = {step["id"] for step in steps}
    for source in handoff["source_feedback"]:
        if source["step_id"] not in selected:
            raise ValidationError(f"feedback {source['id']}: step is outside onboarding")


def adaptive_onboarding(document):
    validate(document, INPUT)
    if not document["synthetic"]:
        raise ValidationError("this reference implementation requires synthetic data")
    profile = copy.deepcopy(document["profile"])
    validate_profile(profile)
    validate_sources(document["feedback"])
    steps = []
    for step_id in required_steps(profile):
        definition = CATALOG[step_id]
        completed = step_id in profile["completed_steps"]
        count = (1 if profile["preference"] == "concise"
                 or profile["experience"] == "advanced" else
                 3 if profile["experience"] == "beginner" else 2)
        explanation = (
            f"{definition['why']} "
            f"Tailored for {profile['experience']} experience and "
            f"{profile['preference']} guidance."
        )
        if completed:
            explanation += " Already completed; no repeat action is required."
        steps.append({
            "id": step_id,
            "prerequisites": list(definition["prerequisites"]),
            "status": "completed" if completed else "planned",
            "explanation": explanation,
            "instructions": (["Review this completed step only if needed."] if completed
                             else definition["instructions"][:count]),
        })
    handoff = {"profile": profile, "steps": steps,
               "source_feedback": copy.deepcopy(document["feedback"])}
    validate_onboarding(handoff)
    return handoff


def analyze_feedback(handoff):
    """Consume only the validated stage-one handoff, not the original input."""
    validate_onboarding(handoff)
    sources = handoff["source_feedback"]
    steps = {step["id"]: step for step in handoff["steps"]}
    by_key = {}
    for source in sources:
        normalized = canonical(source["text"])
        key = (source["step_id"], normalized)
        if key not in by_key:
            by_key[key] = {
                "id": f"group-{len(by_key) + 1:03d}",
                "step_id": source["step_id"],
                "step_status": steps[source["step_id"]]["status"],
                "canonical_text": normalized, "source_ids": [], "excerpts": [],
            }
        group = by_key[key]
        group["source_ids"].append(source["id"])
        group["excerpts"].append(copy.deepcopy(source))
    groups = list(by_key.values())
    buckets = {}
    for group in groups:
        words = set(group["canonical_text"].split())
        names = [name for name, words_for_theme in KEYWORDS.items()
                 if words & words_for_theme] or ["other"]
        for name in names:
            buckets.setdefault(name, []).append(group)
    themes = []
    profile = handoff["profile"]
    for name in sorted(buckets):
        members = buckets[name]
        targets = [step["id"] for step in handoff["steps"]
                   if any(group["step_id"] == step["id"] for group in members)]
        mode = ("step-by-step guidance" if profile["preference"] == "guided"
                and profile["experience"] != "advanced" else "a concise checklist")
        themes.append({
            "name": name,
            "group_ids": [group["id"] for group in members],
            "supporting_excerpts": [copy.deepcopy(excerpt) for group in members
                                    for excerpt in group["excerpts"]],
            "recommendation": f"Review {', '.join(targets)} using {mode}; "
                              "treat this keyword theme as a suggestion, not sentiment.",
        })
    result = {
        "source_count": len(sources), "unique_count": len(groups),
        "duplicate_count": len(sources) - len(groups),
        "groups": groups, "themes": themes,
    }
    validate(result, FEEDBACK, "$.feedback")
    return result


def run_pipeline(document):
    handoff = adaptive_onboarding(document)
    feedback = analyze_feedback(handoff)
    result = {"schema_version": 1, "status": "ok", "synthetic": True,
              "onboarding": handoff, "feedback": feedback}
    validate(result, OUTPUT)
    return result


def reject_constant(value):
    raise ValidationError(f"invalid JSON constant: {value}")


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as stream:
            document = json.load(stream, parse_constant=reject_constant,
                                 object_pairs_hook=strict_object)
        result = run_pipeline(document)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
