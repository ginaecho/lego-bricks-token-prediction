"""Deterministic synthetic journey -> triage -> guided setup reference CLI.

Public API: run_pipeline(input_envelope), validate(envelope, through="input").
All stages use the same versioned envelope. Validation rejects unknown fields,
invalid dependency graphs, unaccountable routes, and forged stage handoffs.
"""

import copy
import json
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


PRIORITIES = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
ROUTE_FIELDS = {"category", "priority", "team", "owner", "setup_actions"}
PHASES = ("input", "journey", "triage", "guided")


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 2000,
            path + " must be nonblank text of at most 2000 characters")


def strings(value, path, limit=200):
    require(isinstance(value, list) and len(value) <= limit,
            path + " must be a bounded list")
    for item in value:
        text(item, path + " item")
    require(len(value) == len(set(value)), path + " contains duplicates")


def route_valid(route, ids, path, rule=False):
    fields(route, ROUTE_FIELDS | ({"keywords", "source"} if rule else set()), path)
    for key in ("category", "priority", "team", "owner"):
        text(route[key], path + "." + key)
    require(route["priority"] in PRIORITIES, path + " has invalid priority")
    strings(route["setup_actions"], path + ".setup_actions")
    require(set(route["setup_actions"]) <= ids, path + " has unknown setup actions")
    if rule:
        strings(route["keywords"], path + ".keywords", 50)
        require(bool(route["keywords"]), path + " requires keywords")
        require(route["source"] in ("ticket", "journey", "either"),
                path + " has invalid source")


def validate_request(request):
    fields(request, {"user", "actions", "ticket", "routing", "setup_events"}, "request")
    user = request["user"]
    fields(user, {"id", "goals", "completed"}, "user")
    text(user["id"], "user.id")
    strings(user["goals"], "user.goals")
    strings(user["completed"], "user.completed")
    actions = request["actions"]
    require(isinstance(actions, list) and 1 <= len(actions) <= 200,
            "actions must contain 1 to 200 items")
    ids = set()
    for action in actions:
        fields(action, {"id", "title", "goals", "prerequisites", "weight"}, "action")
        text(action["id"], "action.id")
        text(action["title"], "action.title")
        strings(action["goals"], "action.goals")
        strings(action["prerequisites"], "action.prerequisites")
        require(type(action["weight"]) is int and 0 <= action["weight"] <= 1000,
                "action.weight must be an integer from 0 to 1000")
        require(action["id"] not in ids, "duplicate action id")
        ids.add(action["id"])
    for action in actions:
        require(set(action["prerequisites"]) <= ids, "unknown prerequisite")
    reached = set()
    while len(reached) < len(ids):
        ready = {a["id"] for a in actions
                 if a["id"] not in reached and set(a["prerequisites"]) <= reached}
        require(bool(ready), "action dependency cycle")
        reached.update(ready)
    completed = set(user["completed"])
    require(completed <= ids, "unknown completed action")
    for action in actions:
        if action["id"] in completed:
            require(set(action["prerequisites"]) <= completed,
                    "completed actions must include their prerequisites")
    fields(request["ticket"], {"id", "text"}, "ticket")
    text(request["ticket"]["id"], "ticket.id")
    text(request["ticket"]["text"], "ticket.text")
    routing = request["routing"]
    fields(routing, {"rules", "default"}, "routing")
    require(isinstance(routing["rules"], list) and len(routing["rules"]) <= 100,
            "routing.rules must be a list with at most 100 rules")
    route_valid(routing["default"], ids, "routing.default")
    for rule in routing["rules"]:
        route_valid(rule, ids, "routing.rule", rule=True)
    strings(request["setup_events"], "setup_events")
    require(set(request["setup_events"]) <= ids, "unknown setup event action")


def _journey(request):
    actions = request["actions"]
    goals = set(request["user"]["goals"])
    completed = set(request["user"]["completed"])
    scores = {a["id"]: 1001 * len(goals & set(a["goals"])) + a["weight"]
              for a in actions}
    ranked = sorted(actions, key=lambda a: (-scores[a["id"]], a["id"]))
    eligible = [a for a in ranked if a["id"] not in completed
                and set(a["prerequisites"]) <= completed]
    pairs = [(first, second) for first in eligible for second in ranked
             if second["id"] not in completed | {first["id"]}
             and set(second["prerequisites"]) <= completed | {first["id"]}]
    require(bool(pairs), "a valid two-step journey is not available")
    first, second = pairs[0]
    return {
        "user_id": request["user"]["id"],
        "eligible_next_actions": [a["id"] for a in eligible],
        "steps": [{"action_id": a["id"], "score": scores[a["id"]],
                  "matched_goals": sorted(goals & set(a["goals"]))}
                 for a in (first, second)],
        "second_step_condition": "complete the first step before starting the second",
    }


def _triage(request, journey):
    ids = [step["action_id"] for step in journey["steps"]]
    catalog = {a["id"]: a for a in request["actions"]}
    context = " ".join(catalog[i]["title"] + " " + " ".join(catalog[i]["goals"])
                       for i in ids).casefold()
    ticket = request["ticket"]["text"].casefold()
    matches = []
    for index, rule in enumerate(request["routing"]["rules"]):
        haystack = {"ticket": ticket, "journey": context,
                    "either": ticket + " " + context}[rule["source"]]
        if any(keyword.casefold() in haystack for keyword in rule["keywords"]):
            matches.append((index, rule))
    if matches:
        index, rule = min(matches, key=lambda item: (-PRIORITIES[item[1]["priority"]],
                                                    item[0]))
    else:
        index, rule = None, request["routing"]["default"]
    return {
        "ticket_id": request["ticket"]["id"], "user_id": journey["user_id"],
        "journey_action_ids": ids,
        "matched_rule": index,
        "route": {key: copy.deepcopy(rule[key]) for key in sorted(ROUTE_FIELDS)},
    }


def _guided(request, triage):
    catalog = {a["id"]: a for a in request["actions"]}
    done = set(request["user"]["completed"])
    required = set(triage["journey_action_ids"]) | set(triage["route"]["setup_actions"])
    pending = list(required)
    while pending:
        for prerequisite in catalog[pending.pop()]["prerequisites"]:
            if prerequisite not in required:
                required.add(prerequisite)
                pending.append(prerequisite)
    ordered = []
    remaining = set(required)
    while remaining:
        ready = sorted(i for i in remaining
                       if set(catalog[i]["prerequisites"]) <= set(ordered))
        require(bool(ready), "setup has unsatisfied dependencies")
        ordered.extend(ready)
        remaining.difference_update(ready)
    for event in request["setup_events"]:
        require(event in required, "setup event is outside the routed plan")
        require(event not in done, "setup event repeats a completed action")
        require(set(catalog[event]["prerequisites"]) <= done,
                "setup event prerequisites are incomplete: " + event)
        done.add(event)
    steps = []
    for action_id in ordered:
        missing = sorted(set(catalog[action_id]["prerequisites"]) - done)
        state = "complete" if action_id in done else ("blocked" if missing else "ready")
        steps.append({"action_id": action_id, "state": state,
                      "missing_prerequisites": missing, "owner": triage["route"]["owner"],
                      "team": triage["route"]["team"]})
    count = len(required & done)
    return {
        "ticket_id": triage["ticket_id"], "user_id": triage["user_id"],
        "category": triage["route"]["category"], "priority": triage["route"]["priority"],
        "journey_action_ids": list(triage["journey_action_ids"]),
        "steps": steps, "next_actions": [s["action_id"] for s in steps if s["state"] == "ready"],
        "progress": {"completed": count, "total": len(required),
                     "percent": round(100 * count / len(required), 2)},
        "status": "complete" if count == len(required) else "in_progress",
    }


def validate(envelope, through="input"):
    """Validate schema plus canonical stage invariants before every handoff."""
    require(through in PHASES, "unknown validation phase")
    base = {"schema_version", "fixture_label", "request"}
    fields(envelope, base if through == "input" else base | {"status", "stages"}, "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "schema_version must be integer 1")
    text(envelope["fixture_label"], "fixture_label")
    require(envelope["fixture_label"].startswith("synthetic:"),
            "fixture_label must start with synthetic:")
    validate_request(envelope["request"])
    if through == "input":
        return
    require(envelope["status"] == ("ok" if through == "guided" else "processing"),
            "invalid envelope status")
    phase = PHASES.index(through)
    fields(envelope["stages"], PHASES[1:phase + 1], "stages")
    request = envelope["request"]
    expected = {"journey": _journey(request)}
    if phase >= 2:
        expected["triage"] = _triage(request, expected["journey"])
    if phase >= 3:
        expected["guided"] = _guided(request, expected["triage"])
    # JSON comparison also distinguishes booleans from integer IDs/scores.
    require(json.dumps(envelope["stages"], sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False),
            "stage output violates schema or deterministic handoff invariants")


def run_pipeline(envelope):
    validate(envelope)
    result = copy.deepcopy(envelope)
    request = result["request"]
    result.update(status="processing", stages={"journey": _journey(request)})
    validate(result, "journey")
    result["stages"]["triage"] = _triage(request, result["stages"]["journey"])
    validate(result, "triage")
    result["stages"]["guided"] = _guided(request, result["stages"]["triage"])
    result["status"] = "ok"
    validate(result, "guided")
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_text(encoding="utf-8")
        data = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
