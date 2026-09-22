"""Deterministic, fictional-reference adaptive onboarding; Python standard library only."""

import copy
import json
import math
import sys


SKILLS = {"novice": 0, "intermediate": 1, "advanced": 2}


class ValidationError(ValueError):
    pass


def fail(message):
    raise ValidationError(message)


def obj(value, name, allowed):
    if not isinstance(value, dict):
        fail(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        fail(f"{name}: unknown fields {sorted(unknown)}")
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        fail(f"{name} must be a nonempty string")
    return value


def strings(value, name):
    if not isinstance(value, list):
        fail(f"{name} must be an array")
    for item in value:
        text(item, name)
    if len(set(value)) != len(value):
        fail(f"{name} contains duplicates")
    return value


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{name} must be a finite nonnegative number")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0:
        fail(f"{name} must be a finite nonnegative number")
    return value


def boolean(value, name):
    if not isinstance(value, bool):
        fail(f"{name} must be a boolean")
    return value


def choice(value, name, options):
    if not isinstance(value, str) or value not in options:
        fail(f"{name} must be one of {list(options)}")
    return value


def validate(data):
    obj(data, "input", {"fixture_label", "user", "steps", "progress"})
    if "fixture_label" in data:
        text(data["fixture_label"], "fixture_label")
    user = obj(data.get("user"), "user",
               {"skill", "experience_years", "preferences", "accessibility"})
    skill = choice(user.get("skill"), "user.skill", SKILLS)
    years = number(user.get("experience_years"), "user.experience_years")
    prefs = obj(user.get("preferences", {}), "preferences",
                {"pace", "detail", "preferred_formats", "skip_optional"})
    pace = choice(prefs.get("pace", "self_paced"), "pace",
                  ("self_paced", "short_sessions", "intensive"))
    detail = choice(prefs.get("detail", "auto"), "detail",
                    ("auto", "guided", "standard", "concise"))
    preferred = strings(prefs.get("preferred_formats", []), "preferred_formats")
    skip_optional = boolean(prefs.get("skip_optional", False), "skip_optional")
    access = obj(user.get("accessibility", {}), "accessibility",
                 {"needs", "forbidden_features", "avoid_formats"})
    needs = strings(access.get("needs", []), "accessibility.needs")
    forbidden = strings(access.get("forbidden_features", []), "forbidden_features")
    avoided = strings(access.get("avoid_formats", []), "avoid_formats")
    if set(needs) & set(forbidden):
        fail("Conflicting constraints: needed and forbidden accessibility features overlap")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        fail("steps must be an array")
    steps = {}
    for raw in raw_steps:
        obj(raw, "step", {"id", "title", "prerequisites", "required", "optional",
                          "redundant_if", "formats", "accessibility_support",
                          "accessibility_conflicts", "min_skill",
                          "min_experience_years", "wording_options"})
        sid = text(raw.get("id"), "step.id")
        if sid in steps:
            fail(f"Duplicate step ID: {sid}")
        title = text(raw.get("title"), f"{sid}.title")
        optional = boolean(raw.get("optional", False), f"{sid}.optional")
        required = boolean(raw.get("required", not optional), f"{sid}.required")
        if required and optional:
            fail(f"{sid}: required and optional conflict")
        redundant = raw.get("redundant_if")
        if "redundant_if" in raw:
            obj(redundant, f"{sid}.redundant_if", {"min_skill", "min_experience_years"})
            if not redundant:
                fail(f"{sid}: redundant_if must contain a condition")
            if "min_skill" in redundant:
                choice(redundant["min_skill"], f"{sid}.redundant_if.min_skill", SKILLS)
            if "min_experience_years" in redundant:
                number(redundant["min_experience_years"], f"{sid}.redundant_if.min_experience_years")
        if not required and not optional and redundant is None:
            fail(f"{sid}: nonrequired steps must be explicitly optional or conditionally redundant")
        formats = strings(raw.get("formats", ["text"]), f"{sid}.formats")
        if not formats:
            fail(f"{sid}: formats cannot be empty")
        support = strings(raw.get("accessibility_support", []), f"{sid}.accessibility_support")
        conflicts = strings(raw.get("accessibility_conflicts", []), f"{sid}.accessibility_conflicts")
        if set(support) & set(conflicts):
            fail(f"{sid}: supported and conflicting accessibility features overlap")
        steps[sid] = {
            "id": sid, "title": title, "optional": optional, "required": required,
            "redundant_if": redundant,
            "prerequisites": strings(raw.get("prerequisites", []), f"{sid}.prerequisites"),
            "formats": formats, "accessibility_support": support,
            "accessibility_conflicts": conflicts,
            "min_skill": choice(raw.get("min_skill", "novice"), f"{sid}.min_skill", SKILLS),
            "min_experience_years": number(raw.get("min_experience_years", 0),
                                          f"{sid}.min_experience_years"),
            "wording_options": strings(raw.get("wording_options", []), f"{sid}.wording_options"),
        }
    for sid, step in steps.items():
        missing = set(step["prerequisites"]) - set(steps)
        if missing:
            fail(f"{sid}: unknown prerequisite IDs {sorted(missing)}")
    # Iterative ordering also handles deep prerequisite chains without recursion.
    remaining = set(steps)
    order = []
    while remaining:
        ready = [sid for sid in steps if sid in remaining
                 and not (set(steps[sid]["prerequisites"]) & remaining)]
        if not ready:
            fail(f"Prerequisite cycle among IDs {sorted(remaining)}")
        order.extend(ready)
        remaining.difference_update(ready)
    progress = obj(data.get("progress", {}), "progress", {"completed"})
    completed = set(strings(progress.get("completed", []), "progress.completed"))
    if completed - set(steps):
        fail(f"Unknown completed IDs {sorted(completed - set(steps))}")
    for sid in completed:
        if not set(steps[sid]["prerequisites"]) <= completed:
            fail(f"Inconsistent progress: {sid} is completed before its prerequisites")
    return steps, order, completed, {
        "skill": skill, "experience_years": years, "pace": pace, "detail": detail,
        "preferred_formats": preferred, "skip_optional": skip_optional,
        "needs": needs, "forbidden_features": forbidden, "avoid_formats": avoided,
    }


def condition_matches(condition, user):
    return (("min_skill" not in condition
             or SKILLS[user["skill"]] >= SKILLS[condition["min_skill"]])
            and ("min_experience_years" not in condition
                 or user["experience_years"] >= condition["min_experience_years"]))


def adapt(data, wording_callback=None):
    """Return a plan without side effects.

    Optional callback receives a deep-copied list of {id, conditions,
    allowed_wordings}. It must return a list in the same order containing
    exactly {id, conditions, wording}. Wording must be one of the source's
    approved strings. This intentionally permits selection, not free-form
    generation: arbitrary natural language cannot be semantically verified.
    """
    steps, order, completed, user = validate(data)
    candidates = {}
    for sid in order:
        step = steps[sid]
        if sid in completed or step["required"]:
            continue
        if step["redundant_if"] and condition_matches(step["redundant_if"], user):
            candidates[sid] = {"code": "explicit_redundancy",
                               "condition": step["redundant_if"],
                               "message": "Source-declared redundancy condition matches this user."}
        elif step["optional"] and user["skip_optional"]:
            candidates[sid] = {"code": "optional_preference",
                               "message": "Explicitly optional; user requested skipping optional steps."}

    retained = set(steps) - completed - set(candidates)
    # Every retained step protects its entire unfinished prerequisite closure,
    # even if that prerequisite is optional or redundant for this user.
    pending = list(retained)
    protected = set()
    while pending:
        sid = pending.pop()
        for prerequisite in steps[sid]["prerequisites"]:
            if prerequisite in completed:
                continue
            if prerequisite not in retained:
                retained.add(prerequisite)
                protected.add(prerequisite)
                pending.append(prerequisite)
    skipped = [{"id": sid, "reason": candidates[sid]} for sid in order
               if sid in candidates and sid not in retained]
    blocked = {}
    plan = []
    eligible = []
    detail = user["detail"]
    if detail == "auto":
        detail = {"novice": "guided", "intermediate": "standard", "advanced": "concise"}[user["skill"]]
    for sid in order:
        if sid not in retained:
            continue
        step = steps[sid]
        reasons = []
        if SKILLS[user["skill"]] < SKILLS[step["min_skill"]]:
            reasons.append({"code": "skill_requirement", "required": step["min_skill"],
                            "actual": user["skill"]})
        if user["experience_years"] < step["min_experience_years"]:
            reasons.append({"code": "experience_requirement",
                            "required_years": step["min_experience_years"],
                            "actual_years": user["experience_years"]})
        missing_support = sorted(set(user["needs"]) - set(step["accessibility_support"]))
        if missing_support:
            reasons.append({"code": "accessibility_not_supported", "features": missing_support})
        conflict = sorted((set(user["needs"]) & set(step["accessibility_conflicts"]))
                          | (set(user["forbidden_features"]) & set(step["accessibility_support"])))
        if conflict:
            reasons.append({"code": "accessibility_conflict", "features": conflict})
        formats = [fmt for fmt in step["formats"] if fmt not in user["avoid_formats"]]
        if not formats:
            reasons.append({"code": "no_accessible_format", "avoided_formats": user["avoid_formats"]})
        unavailable = [pid for pid in step["prerequisites"] if pid in blocked]
        if unavailable:
            reasons.append({"code": "blocked_prerequisites", "ids": unavailable})
        if reasons:
            blocked[sid] = {"id": sid, "reasons": reasons}
            continue
        preferred = [fmt for fmt in user["preferred_formats"] if fmt in formats]
        selected_format = (preferred or formats)[0]
        waiting = [pid for pid in step["prerequisites"] if pid not in completed]
        explanations = [
            f"User skill is {user['skill']}; instructional detail is {detail} "
            f"({'skill-derived' if user['detail'] == 'auto' else 'explicit preference'}).",
            f"Reported experience {user['experience_years']} years satisfies the "
            f"source minimum of {step['min_experience_years']} years.",
            f"Format {selected_format} selected "
            f"({'preferred accessible format' if preferred else 'accessible source fallback'}).",
            f"Pace is {user['pace']}; this is presentation metadata, not generated instruction.",
            f"Declared accessibility needs satisfied: {user['needs']}.",
        ]
        if sid in protected:
            explanations.append("Retained despite a skip condition because a retained step requires it.")
        elif step["required"]:
            explanations.append("Required step; personalization cannot skip it.")
        if waiting:
            explanations.append(f"Planned after unfinished prerequisites: {waiting}.")
        else:
            eligible.append(sid)
            explanations.append("Eligible now: all prerequisites are completed.")
        plan.append({
            "id": sid, "title": step["title"], "wording": step["title"],
            "prerequisites": step["prerequisites"], "eligible_now": not waiting,
            "waiting_for": waiting, "format": selected_format,
            "detail": detail, "pace": user["pace"], "explanations": explanations,
            "conditions": {
                "min_skill": step["min_skill"],
                "min_experience_years": step["min_experience_years"],
                "accessibility_support": step["accessibility_support"],
                "accessibility_conflicts": step["accessibility_conflicts"],
                "formats": step["formats"],
                "required": step["required"], "optional": step["optional"],
                "redundant_if": step["redundant_if"],
                "prerequisites": step["prerequisites"],
            },
        })

    if wording_callback is not None:
        if not callable(wording_callback):
            fail("wording_callback must be callable")
        request = [{"id": item["id"], "conditions": item["conditions"],
                    "allowed_wordings": list(dict.fromkeys(
                        [item["title"]] + steps[item["id"]]["wording_options"]))}
                   for item in plan]
        try:
            response = wording_callback(copy.deepcopy(request))
        except Exception as exc:
            raise ValidationError("Wording callback failed; no plan was accepted") from exc
        if not isinstance(response, list) or len(response) != len(request):
            fail("Wording callback must preserve all planned IDs")
        for expected, actual, item in zip(request, response, plan):
            obj(actual, "callback item", {"id", "conditions", "wording"})
            if set(actual) != {"id", "conditions", "wording"}:
                fail("Wording callback must return exactly id, conditions, wording")
            if actual["id"] != expected["id"]:
                fail("Wording callback must preserve planned IDs and order")
            # Canonical JSON distinguishes booleans from numbers, unlike Python equality.
            try:
                unchanged = json.dumps(actual["conditions"], sort_keys=True, allow_nan=False) == json.dumps(
                    expected["conditions"], sort_keys=True, allow_nan=False)
            except (TypeError, ValueError):
                unchanged = False
            if not unchanged:
                fail("Wording callback changed source-supported conditions")
            if actual["wording"] not in expected["allowed_wordings"]:
                fail("Wording callback supplied unsupported wording")
            item["wording"] = actual["wording"]
    required_blocked = [sid for sid in order if sid in blocked and steps[sid]["required"]]
    return copy.deepcopy({
        "fixture_label": data.get("fixture_label"),
        "eligible_step_ids": eligible, "plan": plan,
        "blocked": list(blocked.values()), "skipped": skipped,
        "completed_step_ids": [sid for sid in order if sid in completed],
        "required_plan_feasible": not required_blocked,
        "blocked_required_step_ids": required_blocked,
        "notes": [
            "Eligibility is based on reported completed progress, not merely earlier plan entries.",
            "Blocked steps are not skipped; unmet requirements need external resolution.",
            "Accessibility support and redundancy are source declarations, not independently verified.",
        ],
    })


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            fail("Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object,
                             parse_constant=lambda value: fail(f"Invalid JSON number: {value}"))
        result = adapt(data)
    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
