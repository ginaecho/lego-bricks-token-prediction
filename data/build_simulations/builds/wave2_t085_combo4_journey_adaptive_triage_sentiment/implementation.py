"""Synthetic, deterministic journey -> onboarding -> triage -> sentiment CLI.

Run: python -B implementation.py example_input.json
The versioned envelope is validated before and after every stage. Plans simulate
completion; they never mark the user's actions as actually completed.
"""

import copy
import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    """A schema, consistency, or feasibility failure."""


STAGE_NAMES = ("journey", "adaptive", "triage", "sentiment")
BASE_KEYS = {"schema_version", "fixture", "profile", "actions", "tickets", "config"}
SEVERITIES = {"low": 4, "medium": 3, "high": 2, "critical": 1}
EXPERIENCES = {
    "beginner": "Follow the detailed walkthrough and verify each checkpoint.",
    "intermediate": "Use the guided checklist and verify the final result.",
    "expert": "Use the quick-start checklist and verify the final result.",
}
FORMATS = {
    "text": "Read the step-by-step instructions.",
    "video": "Follow the narrated demonstration.",
    "hands_on": "Practice the task in a synthetic exercise.",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def shape(value, keys, path):
    require(type(value) is dict, f"{path}: expected object")
    require(set(value) == set(keys), f"{path}: expected exactly {sorted(keys)}")


def text(value, path, limit=200):
    require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty string")
    require(len(value) <= limit, f"{path}: string too long (maximum {limit})")


def integer(value, low, high, path):
    require(type(value) is int and low <= value <= high, f"{path}: expected integer {low}..{high}")


def strings(value, path, maximum=200):
    require(type(value) is list and len(value) <= maximum, f"{path}: expected bounded list")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), f"{path}: duplicate values")


def choice(value, options, path):
    text(value, path)
    require(value in options, f"{path}: unsupported value {value!r}")


def tokens(value):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", value.casefold(), flags=re.UNICODE)


def phrase_present(words, phrase):
    wanted = tokens(phrase)
    return any(words[i:i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))


def _validate_base(state):
    integer(state["schema_version"], 1, 1, "schema_version")
    require(state["fixture"] == "synthetic", "fixture: must be labeled synthetic")
    profile = state["profile"]
    shape(profile, {"id", "experience", "preference", "interests", "completed"}, "profile")
    text(profile["id"], "profile.id")
    choice(profile["experience"], EXPERIENCES, "profile.experience")
    choice(profile["preference"], FORMATS, "profile.preference")
    strings(profile["interests"], "profile.interests")
    strings(profile["completed"], "profile.completed")

    config = state["config"]
    shape(config, {"routes", "rules", "default_category", "sentiment_lexicon"}, "config")
    routes = config["routes"]
    require(type(routes) is dict and 0 < len(routes) <= 100, "config.routes: expected 1..100 routes")
    for category, route in routes.items():
        text(category, "route category")
        shape(route, {"owner", "queue", "base_priority"}, f"route.{category}")
        text(route["owner"], "route.owner")
        text(route["queue"], "route.queue")
        integer(route["base_priority"], 1, 4, "route.base_priority")
    choice(config["default_category"], routes, "default_category")
    require(type(config["rules"]) is list and len(config["rules"]) <= 100, "rules: expected bounded list")
    for rule in config["rules"]:
        shape(rule, {"category", "keywords"}, "rule")
        choice(rule["category"], routes, "rule.category")
        strings(rule["keywords"], "rule.keywords", 100)
        require(bool(rule["keywords"]), "rule.keywords: must not be empty")
        for keyword in rule["keywords"]:
            require(bool(tokens(keyword)), "rule keyword: needs at least one word")
    lexicon = config["sentiment_lexicon"]
    require(type(lexicon) is dict and len(lexicon) <= 200, "sentiment_lexicon: expected bounded object")
    for word, weight in lexicon.items():
        text(word, "lexicon word")
        require(tokens(word) == [word], "lexicon word: must be a single lowercase token")
        integer(weight, -5, 5, "lexicon weight")

    actions = state["actions"]
    require(type(actions) is list and len(actions) <= 200, "actions: expected bounded list")
    ids = set()
    for action in actions:
        shape(action, {"id", "title", "prerequisites", "tags", "support_category"}, "action")
        text(action["id"], "action.id")
        require(action["id"] not in ids, "actions: duplicate id")
        ids.add(action["id"])
        text(action["title"], "action.title")
        strings(action["prerequisites"], "action.prerequisites")
        strings(action["tags"], "action.tags")
        choice(action["support_category"], routes, "action.support_category")
    require(set(profile["completed"]) <= ids, "profile.completed: unknown action")
    for action in actions:
        require(set(action["prerequisites"]) <= ids, "action.prerequisites: unknown action")
    resolved = set()
    while len(resolved) < len(ids):
        ready = {
            a["id"] for a in actions
            if a["id"] not in resolved and set(a["prerequisites"]) <= resolved
        }
        require(bool(ready), "actions: prerequisite cycle")
        resolved.update(ready)
    completed = set(profile["completed"])
    for action in actions:
        if action["id"] in completed:
            require(set(action["prerequisites"]) <= completed, "completed action lacks completed prerequisites")

    tickets = state["tickets"]
    require(type(tickets) is list and len(tickets) <= 1000, "tickets: expected bounded list")
    ticket_ids = set()
    for ticket in tickets:
        shape(ticket, {"id", "text", "severity", "action_id", "blocked"}, "ticket")
        text(ticket["id"], "ticket.id")
        require(ticket["id"] not in ticket_ids, "tickets: duplicate id")
        ticket_ids.add(ticket["id"])
        text(ticket["text"], "ticket.text", 10000)
        choice(ticket["severity"], SEVERITIES, "ticket.severity")
        require(type(ticket["blocked"]) is bool, "ticket.blocked: expected boolean")
        if ticket["action_id"] is not None:
            choice(ticket["action_id"], ids, "ticket.action_id")


def _journey(state):
    profile = state["profile"]
    completed = set(profile["completed"])
    interests = set(profile["interests"])

    def ranking(action):
        return (-len(interests.intersection(action["tags"])), action["id"])

    def ready(done):
        return sorted(
            (a for a in state["actions"] if a["id"] not in done and set(a["prerequisites"]) <= done),
            key=ranking,
        )

    available = ready(completed)
    selected = None
    for first in available:
        following = ready(completed | {first["id"]})
        if following:
            selected = [first, following[0]]
            break
    require(selected is not None, "journey: no feasible two-step journey remains")
    next_actions = [
        {"action_id": a["id"], "interest_matches": sorted(interests.intersection(a["tags"]))}
        for a in available
    ]
    steps = []
    for index, action in enumerate(selected, 1):
        matches = sorted(interests.intersection(action["tags"]))
        steps.append({
            "position": index,
            "action_id": action["id"],
            "prerequisites": sorted(action["prerequisites"]),
            "assumed_completed_before": sorted(completed),
            "reason": f"Prerequisites satisfied; interest matches: {', '.join(matches) or 'none'}; ties use action id.",
        })
        completed.add(action["id"])
    return {"profile_id": profile["id"], "planning_only": True, "next_actions": next_actions, "steps": steps}


def _adaptive(state):
    journey = state["stages"]["journey"]
    profile = state["profile"]
    actions = {a["id"]: a for a in state["actions"]}
    steps = []
    for step in journey["steps"]:
        action = actions[step["action_id"]]
        steps.append({
            "position": step["position"],
            "action_id": step["action_id"],
            "prerequisites": step["prerequisites"],
            "prerequisites_met": set(step["prerequisites"]) <= set(step["assumed_completed_before"]),
            "assumed_completed_before": step["assumed_completed_before"],
            "format": profile["preference"],
            "guidance": f"{action['title']}: {EXPERIENCES[profile['experience']]} {FORMATS[profile['preference']]}",
            "explanation": (
                f"{profile['experience']} experience determines detail; "
                f"{profile['preference']} preference determines format. "
                "Later steps require completing earlier planned prerequisites."
            ),
            "support_category": action["support_category"],
        })
    return {
        "profile_id": journey["profile_id"],
        "planning_only": journey["planning_only"],
        "experience": profile["experience"],
        "steps": steps,
    }


def _triage(state):
    onboarding = {s["action_id"]: s for s in state["stages"]["adaptive"]["steps"]}
    config = state["config"]
    results = []
    for ticket in state["tickets"]:
        step = onboarding.get(ticket["action_id"])
        matched = []
        category = step["support_category"] if step else config["default_category"]
        source = "onboarding" if step else "default"
        words = tokens(ticket["text"])
        for rule in config["rules"]:
            matched = [keyword for keyword in rule["keywords"] if phrase_present(words, keyword)]
            if matched:
                category, source = rule["category"], "keyword"
                break
        route = config["routes"][category]
        priority = min(route["base_priority"], SEVERITIES[ticket["severity"]])
        boosted = bool(step and ticket["blocked"])
        if boosted:
            priority = max(1, priority - 1)
        results.append({
            "ticket_id": ticket["id"],
            "text": ticket["text"],
            "action_id": ticket["action_id"],
            "severity": ticket["severity"],
            "category": category,
            "category_source": source,
            "matched_keywords": matched,
            "priority": priority,
            "owner": route["owner"],
            "queue": route["queue"],
            "onboarding_step": step["position"] if step else None,
            "onboarding_format": step["format"] if step else None,
            "onboarding_guidance": step["guidance"] if step else None,
            "blocked_onboarding_boost": boosted,
            "reason": "Minimum of route/severity priority; blocked planned step raises urgency one level.",
        })
    return {"profile_id": state["stages"]["adaptive"]["profile_id"], "tickets": results}


def _sentiment(state):
    lexicon = state["config"]["sentiment_lexicon"]
    issues = []
    for ticket in state["stages"]["triage"]["tickets"]:
        words = tokens(ticket["text"])
        contributions = []
        for index, word in enumerate(words):
            if word in lexicon:
                negated = index > 0 and words[index - 1] in {"not", "never", "no"}
                weight = lexicon[word]
                contributions.append({
                    "token": word,
                    "index": index,
                    "base_weight": weight,
                    "negated": negated,
                    "contribution": -weight if negated else weight,
                })
        score = sum(c["contribution"] for c in contributions)
        negative_boost = score <= -3
        priority = min(ticket["priority"], 2) if negative_boost else ticket["priority"]
        severity_weight = 5 - SEVERITIES[ticket["severity"]]
        negativity = min(5, max(0, -score))
        urgency = (5 - priority) * 100 + severity_weight * 10 + negativity
        issues.append({
            "ticket_id": ticket["ticket_id"],
            "category": ticket["category"],
            "owner": ticket["owner"],
            "queue": ticket["queue"],
            "action_id": ticket["action_id"],
            "onboarding_step": ticket["onboarding_step"],
            "severity": ticket["severity"],
            "triage_priority": ticket["priority"],
            "priority": priority,
            "urgency": urgency,
            "sentiment": {
                "score": score,
                "label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
                "contributions": contributions,
            },
            "negative_escalation": negative_boost and priority < ticket["priority"],
            "reason": "Score <= -3 sets priority to at least P2; positive sentiment never lowers urgency.",
        })
    issues.sort(key=lambda issue: (-issue["urgency"], issue["ticket_id"]))
    for rank, issue in enumerate(issues, 1):
        issue["rank"] = rank
    return {
        "issues": issues,
        "scoring": "Sum configured token weights; immediately preceding not/never/no inverts a weight.",
        "ordering": "(5-priority)*100 + (5-severity_priority)*10 + min(5,max(0,-sentiment_score)); descending, then ticket id.",
    }


BUILDERS = (_journey, _adaptive, _triage, _sentiment)


def validate(state, completed_stages):
    """Shared schema + canonical semantic validator for every pipeline boundary.

    Recomputing bounded pure stage values verifies not just types but prerequisite
    feasibility, ordering, accountable routing, and exact cross-stage provenance.
    """
    integer(completed_stages, 0, 4, "completed_stages")
    shape(state, BASE_KEYS | {"status", "stages"}, "envelope")
    require(state["status"] == "ok", "envelope.status: expected ok")
    _validate_base(state)
    shape(state["stages"], STAGE_NAMES[:completed_stages], "stages")
    for index in range(completed_stages):
        name = STAGE_NAMES[index]
        expected = BUILDERS[index](state)
        # JSON canonicalization also distinguishes bools from integers.
        try:
            actual_json = json.dumps(state["stages"][name], sort_keys=True, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValidationError(f"stages.{name}: invalid JSON value") from error
        require(actual_json == json.dumps(expected, sort_keys=True, allow_nan=False),
                f"stages.{name}: invalid or inconsistent stage output")
    return state


def advance(state, stage):
    """Consume one validated envelope and return a new validated envelope."""
    choice(stage, STAGE_NAMES, "stage")
    index = STAGE_NAMES.index(stage)
    validate(state, index)
    result = copy.deepcopy(state)
    result["stages"][stage] = BUILDERS[index](result)
    validate(result, index + 1)
    return result


def run_pipeline(payload):
    shape(payload, BASE_KEYS, "input")
    state = copy.deepcopy(payload)
    state.update({"status": "ok", "stages": {}})
    validate(state, 0)
    for stage in STAGE_NAMES:
        state = advance(state, stage)
    return state


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON: duplicate object key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError(f"JSON: non-finite number {value}")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        raw = Path(args[0]).read_text(encoding="utf-8-sig")
        payload = json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=_invalid_constant)
        output = run_pipeline(payload)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
