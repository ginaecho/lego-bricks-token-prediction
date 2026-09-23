"""Synthetic, deterministic onboarding -> support triage -> discovery reference.

Usage: python -B implementation.py example_input.json
Input and output use the same versioned envelope. Every stage validates the
entire preceding state; generated fields are checked against their deterministic
contract, so callers cannot forge completed prerequisites or remove exclusions.
No providers, persistence, third-party packages, or network access are used.
"""

import copy
import json
import sys


BUILD_ID = "wave2_t035_combo3_adaptive_triage_interests"
EXPERIENCES = ("beginner", "intermediate", "expert")
PRIORITIES = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
STAGES = ("input", "adaptive", "triage", "interests")
STEPS = {
    "basics": ([], "Learn marketplace navigation", "Choose a product and inspect its details."),
    "privacy": ([], "Set privacy preferences", "Review sharing and notification controls."),
    "workspace": (["basics", "privacy"], "Prepare your workspace", "Create a workspace after reviewing privacy."),
    "automation": (["workspace"], "Configure automation", "Create a sample rule and inspect its effect."),
}


class ValidationError(ValueError):
    """An invalid request or stage handoff."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown fields")


def text(value, path):
    require(type(value) is str and bool(value.strip()), path + " must be nonblank text")
    require(value == value.strip(), path + " must not have outer whitespace")


def strings(value, path):
    require(type(value) is list, path + " must be a list")
    for item in value:
        text(item, path + " item")
    require(len(set(value)) == len(value), path + " must be unique")


def validate_request(request):
    fields(request, ("profile", "tickets", "routing", "catalog", "limit"), "request")
    profile = request["profile"]
    fields(profile, ("experience", "presentation", "interests", "excluded_tags", "completed_steps"), "profile")
    require(profile["experience"] in EXPERIENCES, "Invalid experience")
    require(profile["presentation"] in ("guided", "concise"), "Invalid presentation")
    require(type(profile["interests"]) is dict, "interests must be an object")
    for tag, weight in profile["interests"].items():
        text(tag, "interest tag")
        require(type(weight) is int and 1 <= weight <= 5, "Interest weights must be integers 1..5")
    strings(profile["excluded_tags"], "excluded_tags")
    strings(profile["completed_steps"], "completed_steps")
    completed = set(profile["completed_steps"])
    require(completed <= set(STEPS), "Unknown completed step")
    for step in completed:
        require(set(STEPS[step][0]) <= completed, "Completed step has unmet prerequisites: " + step)

    routing = request["routing"]
    fields(routing, ("default_category", "default_owner", "default_priority", "onboarding_owner", "rules"), "routing")
    for key in ("default_category", "default_owner", "onboarding_owner"):
        text(routing[key], key)
    require(type(routing["default_priority"]) is str and routing["default_priority"] in PRIORITIES, "Invalid default priority")
    require(type(routing["rules"]) is list, "rules must be a list")
    rule_ids = []
    for rule in routing["rules"]:
        fields(rule, ("id", "keywords", "category", "priority", "owner", "prerequisites", "avoid_tags"), "rule")
        for key in ("id", "category", "owner"):
            text(rule[key], "rule " + key)
        strings(rule["keywords"], "keywords")
        require(bool(rule["keywords"]), "A rule needs keywords")
        require(type(rule["priority"]) is str and rule["priority"] in PRIORITIES, "Invalid rule priority")
        strings(rule["prerequisites"], "prerequisites")
        require(set(rule["prerequisites"]) <= set(STEPS), "Unknown rule prerequisite")
        strings(rule["avoid_tags"], "avoid_tags")
        rule_ids.append(rule["id"])
    require(len(rule_ids) == len(set(rule_ids)), "Duplicate rule ID")

    require(type(request["tickets"]) is list, "tickets must be a list")
    ticket_ids = []
    for ticket in request["tickets"]:
        fields(ticket, ("id", "text", "status"), "ticket")
        text(ticket["id"], "ticket id")
        text(ticket["text"], "ticket text")
        require(ticket["status"] in ("open", "resolved"), "Invalid ticket status")
        ticket_ids.append(ticket["id"])
    require(len(ticket_ids) == len(set(ticket_ids)), "Duplicate ticket ID")
    require(type(request["catalog"]) is list, "catalog must be a list")
    item_ids = []
    for item in request["catalog"]:
        fields(item, ("id", "title", "tags"), "catalog item")
        text(item["id"], "item id")
        text(item["title"], "item title")
        strings(item["tags"], "item tags")
        require(bool(item["tags"]), "Catalog items need tags")
        item_ids.append(item["id"])
    require(len(item_ids) == len(set(item_ids)), "Duplicate catalog ID")
    require(type(request["limit"]) is int and 0 <= request["limit"] <= 100, "limit must be an integer 0..100")


def _adaptive(request):
    profile = request["profile"]
    completed = profile["completed_steps"]
    targets = ["basics", "privacy", "workspace"]
    if profile["experience"] != "beginner":
        targets.append("automation")
    planned = set(completed)
    plan = []
    for step in targets:
        if step in completed:
            continue
        prerequisites, title, instruction = STEPS[step]
        require(set(prerequisites) <= planned, "Internal prerequisite ordering error")
        explanation = (
            f"{title}: included for {profile['experience']} experience. "
            + ("Complete prerequisite steps first: " + ", ".join(prerequisites) + "."
               if prerequisites else "No prerequisites.")
        )
        if profile["presentation"] == "guided":
            explanation += " " + instruction
        plan.append({
            "id": step,
            "prerequisites": list(prerequisites),
            "available_now": set(prerequisites) <= set(completed),
            "explanation": explanation,
        })
        planned.add(step)
    return {
        "experience": profile["experience"],
        "presentation": profile["presentation"],
        "completed_steps": list(completed),
        "interests": dict(profile["interests"]),
        "excluded_tags": list(profile["excluded_tags"]),
        "plan": plan,
    }


def _triage(request, onboarding):
    config = request["routing"]
    tickets = []
    suppressed = {}
    for ticket in request["tickets"]:
        matches = []
        for index, rule in enumerate(config["rules"]):
            keywords = [word for word in rule["keywords"] if word.casefold() in ticket["text"].casefold()]
            if keywords:
                matches.append((PRIORITIES[rule["priority"]], -index, rule, keywords))
        if matches:
            _, _, rule, keywords = max(matches, key=lambda match: match[:2])
            category, priority, owner = rule["category"], rule["priority"], rule["owner"]
            missing = [step for step in rule["prerequisites"] if step not in onboarding["completed_steps"]]
            rule_id = rule["id"]
            explanation = f"Rule {rule_id} matched keywords: {', '.join(keywords)}."
        else:
            rule, keywords, missing, rule_id = None, [], [], None
            category, priority, owner = config["default_category"], config["default_priority"], config["default_owner"]
            explanation = "No keyword rule matched; configured default route applies."
        blocked = ticket["status"] == "open" and bool(missing)
        assigned_owner = config["onboarding_owner"] if blocked else owner
        if blocked:
            explanation += " Onboarding owner accountable until completed: " + ", ".join(missing) + "."
        explanation += f" Assigned owner: {assigned_owner}; category owner: {owner}."
        # All matching open-ticket rules contribute exclusions, not only the winner.
        # Otherwise a higher-priority route could accidentally erase a restriction.
        if ticket["status"] == "open":
            for _, _, matched_rule, _ in matches:
                for tag in matched_rule["avoid_tags"]:
                    suppressed.setdefault(tag, set()).add(ticket["id"])
        tickets.append({
            "id": ticket["id"], "status": ticket["status"], "category": category,
            "priority": priority, "owner": assigned_owner, "category_owner": owner,
            "rule_id": rule_id, "matched_keywords": keywords,
            "missing_prerequisites": missing, "onboarding_blocked": blocked,
            "explanation": explanation,
        })
    return {
        "tickets": tickets,
        "interests": dict(onboarding["interests"]),
        "excluded_tags": list(onboarding["excluded_tags"]),
        "suppressed_tags": {tag: sorted(ids) for tag, ids in sorted(suppressed.items())},
    }


def _interests(request, triage):
    preferences = triage["interests"]
    exclusions = set(triage["excluded_tags"]) | set(triage["suppressed_tags"])
    ranked, omitted = [], []
    for item in request["catalog"]:
        blocked = sorted(set(item["tags"]) & exclusions)
        if blocked:
            omitted.append({
                "id": item["id"], "excluded_tags": blocked,
                "ticket_ids": sorted({tid for tag in blocked for tid in triage["suppressed_tags"].get(tag, [])}),
                "explanation": "Excluded because item tags intersect user or open-ticket exclusions.",
            })
            continue
        matches = {tag: preferences[tag] for tag in sorted(item["tags"]) if tag in preferences}
        score = sum(matches.values())
        if not score:
            continue
        ranked.append({
            "id": item["id"], "title": item["title"], "score": score,
            "matched_interests": matches,
            "explanation": "Preference score = " + " + ".join(f"{tag}:{weight}" for tag, weight in matches.items()) + f" = {score}.",
        })
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return {
        "items": ranked[:request["limit"]],
        "excluded_items": omitted,
        "effective_excluded_tags": sorted(exclusions),
        "explanation": "Ranked by summed explicit interest weights; ties use item ID. Unmatched items are not recommended.",
    }


def validate_state(state, expected_stage=None):
    """One shared validation boundary for input and all cross-stage handoffs."""
    require(type(state) is dict, "Envelope must be an object")
    stage = state.get("stage")
    require(stage in STAGES, "Unknown stage")
    if expected_stage is not None:
        require(stage == expected_stage, "Expected stage " + expected_stage)
    index = STAGES.index(stage)
    keys = ["schema_version", "synthetic", "status", "stage", "request"]
    keys += ["onboarding", "triage", "recommendations"][:index]
    fields(state, keys, "envelope")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1, "Unsupported schema version")
    require(state["synthetic"] is True, "Fixture must be explicitly synthetic")
    require(state["status"] == "ok", "State status must be ok")
    validate_request(state["request"])
    # Canonical JSON equality also distinguishes booleans from integer scores.
    expected = {}
    if index >= 1:
        expected["onboarding"] = _adaptive(state["request"])
    if index >= 2:
        expected["triage"] = _triage(state["request"], expected["onboarding"])
    if index >= 3:
        expected["recommendations"] = _interests(state["request"], expected["triage"])
    for name, value in expected.items():
        require(
            json.dumps(state[name], sort_keys=True, allow_nan=False) == json.dumps(value, sort_keys=True),
            "Invalid or inconsistent " + name + " handoff",
        )
    return state


def adaptive(state):
    validate_state(state, "input")
    result = copy.deepcopy(state)
    result.update(stage="adaptive", onboarding=_adaptive(result["request"]))
    return validate_state(result, "adaptive")


def triage(state):
    validate_state(state, "adaptive")
    result = copy.deepcopy(state)
    result.update(stage="triage", triage=_triage(result["request"], result["onboarding"]))
    return validate_state(result, "triage")


def interests(state):
    validate_state(state, "triage")
    result = copy.deepcopy(state)
    result.update(stage="interests", recommendations=_interests(result["request"], result["triage"]))
    return validate_state(result, "interests")


def run_pipeline(state):
    return interests(triage(adaptive(state)))


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def reject_constant(value):
    raise ValidationError("Non-finite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as source:
            state = json.load(source, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(state)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
