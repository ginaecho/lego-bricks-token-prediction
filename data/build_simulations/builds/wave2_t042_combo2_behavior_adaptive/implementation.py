"""Deterministic synthetic discovery -> guided onboarding reference pipeline.

Run: python -B implementation.py example_input.json
Only Python's standard library is used. All timestamps require time zones.
"""

import json
import math
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, fields, where):
    require(isinstance(value, dict), where + " must be an object")
    require(set(value) == set(fields), where + " has missing or unknown fields")


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()), where + " must be nonempty text")


def strings(value, where):
    require(isinstance(value, list), where + " must be a list")
    for entry in value:
        text(entry, where)
    require(len(value) == len(set(value)), where + " must contain unique values")


def number(value, where):
    require(type(value) in (int, float), where + " must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and value >= 0, where + " must be finite and nonnegative")


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "timestamp must include a time zone")
    return parsed


def validate_profile(profile):
    obj(profile, ("experience", "interests", "completed_steps"), "profile")
    require(profile["experience"] in ("beginner", "intermediate", "expert"),
            "unknown experience")
    strings(profile["interests"], "interests")
    strings(profile["completed_steps"], "completed_steps")


def validate(value, kind):
    """Shared boundary validation for the request and both stage outputs."""
    if kind == "input":
        obj(value, ("schema_version", "synthetic", "now", "profile", "catalog",
                    "events", "steps", "limit"), "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        require(value["synthetic"] is True, "fixture must be labeled synthetic")
        now = timestamp(value["now"])
        validate_profile(value["profile"])
        require(type(value["limit"]) is int and 1 <= value["limit"] <= 100,
                "limit must be an integer from 1 to 100")
        for field in ("catalog", "events", "steps"):
            require(isinstance(value[field], list), field + " must be a list")
        ids = set()
        for item in value["catalog"]:
            obj(item, ("id", "title", "categories", "popularity"), "catalog item")
            text(item["id"], "item id")
            text(item["title"], "item title")
            strings(item["categories"], "categories")
            number(item["popularity"], "popularity")
            require(item["popularity"] <= 1, "popularity must be at most 1")
            require(item["id"] not in ids, "duplicate item id")
            ids.add(item["id"])
        for event in value["events"]:
            obj(event, ("item_id", "kind", "at"), "event")
            text(event["item_id"], "event item_id")
            require(event["item_id"] in ids, "event references unknown item")
            require(event["kind"] in ("browse", "purchase"), "unknown event kind")
            require(timestamp(event["at"]) <= now, "future events are not allowed")
        steps = {}
        for step in value["steps"]:
            obj(step, ("id", "title", "categories", "experiences", "prerequisites"), "step")
            text(step["id"], "step id")
            text(step["title"], "step title")
            strings(step["categories"], "step categories")
            strings(step["experiences"], "step experiences")
            strings(step["prerequisites"], "prerequisites")
            require(bool(step["experiences"]) and
                    set(step["experiences"]) <= {"beginner", "intermediate", "expert"},
                    "invalid step experiences")
            require(step["id"] not in steps, "duplicate step id")
            steps[step["id"]] = step
        require(set(value["profile"]["completed_steps"]) <= set(steps),
                "unknown completed step")
        for step in steps.values():
            require(set(step["prerequisites"]) <= set(steps), "unknown prerequisite")
        # Iterative topological validation avoids recursion-depth failures.
        remaining = set(steps)
        resolved = set()
        while remaining:
            ready = {sid for sid in remaining
                     if set(steps[sid]["prerequisites"]) <= resolved}
            require(bool(ready), "cyclic prerequisites")
            remaining -= ready
            resolved |= ready
    elif kind == "behavior":
        obj(value, ("profile", "cold_start", "ranking", "category_weights"), "behavior")
        validate_profile(value["profile"])
        require(type(value["cold_start"]) is bool, "cold_start must be boolean")
        require(isinstance(value["ranking"], list), "ranking must be a list")
        require(isinstance(value["category_weights"], dict), "category_weights must be an object")
        ids = set()
        for row in value["ranking"]:
            obj(row, ("item_id", "score", "categories", "explanation"), "ranking row")
            text(row["item_id"], "ranked item")
            require(row["item_id"] not in ids, "duplicate ranked item")
            ids.add(row["item_id"])
            number(row["score"], "score")
            strings(row["categories"], "ranked categories")
            text(row["explanation"], "ranking explanation")
        for category, weight in value["category_weights"].items():
            text(category, "weighted category")
            number(weight, "category weight")
    elif kind == "adaptive":
        obj(value, ("experience", "source_cold_start", "focus_categories",
                    "completed_steps", "plan"), "adaptive")
        require(value["experience"] in ("beginner", "intermediate", "expert"),
                "unknown experience")
        require(type(value["source_cold_start"]) is bool, "invalid source_cold_start")
        strings(value["focus_categories"], "focus categories")
        strings(value["completed_steps"], "completed steps")
        require(isinstance(value["plan"], list), "plan must be a list")
        available = set(value["completed_steps"])
        for row in value["plan"]:
            obj(row, ("step_id", "title", "prerequisites", "explanation"), "plan row")
            text(row["step_id"], "planned step")
            text(row["title"], "planned title")
            text(row["explanation"], "plan explanation")
            strings(row["prerequisites"], "planned prerequisites")
            require(row["step_id"] not in available, "duplicate or completed planned step")
            require(set(row["prerequisites"]) <= available, "unsatisfied planned prerequisite")
            available.add(row["step_id"])
    else:
        raise ValidationError("unknown schema kind")
    return value


def personalize(request):
    validate(request, "input")
    now = timestamp(request["now"])
    catalog = {item["id"]: item for item in request["catalog"]}
    signals = {sid: 0.0 for sid in catalog}
    category_signals = {}
    # A purchase counts three times a browse; signals halve every 30 days.
    for event in request["events"]:
        days = (now - timestamp(event["at"])).total_seconds() / 86400
        weight = (3 if event["kind"] == "purchase" else 1) * 2 ** (-days / 30)
        signals[event["item_id"]] += weight
        for category in catalog[event["item_id"]]["categories"]:
            category_signals[category] = category_signals.get(category, 0.0) + weight
    cold = not any(signals.values())
    interests = set(request["profile"]["interests"])
    ranking = []
    for item in catalog.values():
        affinity = sum(category_signals.get(c, 0.0) for c in item["categories"])
        preference = len(interests.intersection(item["categories"]))
        score = signals[item["id"]] + 0.25 * affinity + preference + 0.1 * item["popularity"]
        ranking.append({
            "item_id": item["id"], "score": round(score, 8),
            "categories": list(item["categories"]),
            "explanation": (
                ("Cold start: preferences and popularity" if cold else
                 "Recency-weighted browse/purchase and category affinity")
                + f"; direct={signals[item['id']]:.4f}, affinity={affinity:.4f}, "
                + f"preference_matches={preference}, popularity={item['popularity']}"
            ),
        })
    ranking.sort(key=lambda row: (-row["score"], row["item_id"]))
    ranking = ranking[:request["limit"]]
    category_weights = {}
    for row in ranking:
        for category in row["categories"]:
            category_weights[category] = category_weights.get(category, 0.0) + row["score"]
    # Explicit interests still guide onboarding when the catalog is empty.
    for category in interests:
        category_weights[category] = category_weights.get(category, 0.0) + 1.0
    return validate({
        "profile": {key: list(v) if isinstance(v, list) else v
                    for key, v in request["profile"].items()},
        "cold_start": cold, "ranking": ranking,
        "category_weights": {key: round(category_weights[key], 8)
                             for key in sorted(category_weights)},
    }, "behavior")


def onboard(behavior, steps):
    validate(behavior, "behavior")
    profile = behavior["profile"]
    weights = behavior["category_weights"]
    focus = sorted(weights, key=lambda category: (-weights[category], category))
    completed = set(profile["completed_steps"])
    lookup = {step["id"]: step for step in steps}
    selected = {
        sid for sid, step in lookup.items()
        if sid not in completed and profile["experience"] in step["experiences"]
        and (not step["categories"] or set(step["categories"]).intersection(focus))
    }
    targets = set(selected)
    pending = list(selected)
    while pending:
        sid = pending.pop()
        for prerequisite in lookup[sid]["prerequisites"]:
            if prerequisite not in selected and prerequisite not in completed:
                selected.add(prerequisite)
                pending.append(prerequisite)
    plan = []
    available = set(completed)
    while selected:
        ready = [sid for sid in selected
                 if set(lookup[sid]["prerequisites"]) <= available]
        require(bool(ready), "onboarding prerequisites cannot be resolved")
        ready.sort(key=lambda sid: (
            -sum(weights.get(c, 0.0) for c in lookup[sid]["categories"]), sid))
        sid = ready[0]
        step = lookup[sid]
        matching = sorted(set(step["categories"]).intersection(focus))
        explanation = (
            f"Matches {profile['experience']} experience; "
            + ("discovery focus: " + ", ".join(matching) if matching else "general guidance")
            if sid in targets else "Required prerequisite for selected guidance"
        )
        plan.append({
            "step_id": sid, "title": step["title"],
            "prerequisites": list(step["prerequisites"]),
            "explanation": explanation + "; prerequisites completed or ordered earlier",
        })
        available.add(sid)
        selected.remove(sid)
    return validate({
        "experience": profile["experience"],
        "source_cold_start": behavior["cold_start"],
        "focus_categories": focus,
        "completed_steps": list(profile["completed_steps"]), "plan": plan,
    }, "adaptive")


def run_pipeline(request):
    validate(request, "input")
    behavior = personalize(request)
    adaptive = onboard(behavior, request["steps"])
    return {"schema_version": 1, "synthetic": True, "status": "ok",
            "behavior": behavior, "adaptive": adaptive}


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            request = json.load(handle, parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
        result = run_pipeline(request)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
