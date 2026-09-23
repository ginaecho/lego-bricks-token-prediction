"""Synthetic, deterministic marketplace journey-to-support reference pipeline.

Usage: python -B implementation.py example_input.json
Only Python's standard library is used; no provider or network access occurs.
"""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    pass


STAGES = ("journey", "adaptive", "triage", "sentiment")
LEVELS = ("low", "medium", "high", "critical")
POSITIVE = {"good": 1, "great": 2, "love": 2, "helpful": 1, "excellent": 2}
NEGATIVE = {"bad": -1, "broken": -2, "hate": -2, "frustrated": -2,
            "terrible": -2, "slow": -1, "confusing": -1}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()),
            name + " must be a nonempty string")


def strings(value, name):
    require(isinstance(value, list), name + " must be a list")
    for item in value:
        text(item, name + " item")
    require(len(set(value)) == len(value), name + " must not contain duplicates")


def fields(value, required, name):
    require(isinstance(value, dict), name + " must be an object")
    require(set(value) == set(required), name + " has missing or unknown fields")


def validate_input(data):
    fields(data, ("schema_version", "synthetic", "profile", "actions",
                  "tickets", "routing"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "synthetic must be true")
    p = data["profile"]
    fields(p, ("experience", "preferences", "completed"), "profile")
    require(p["experience"] in ("beginner", "intermediate", "expert"),
            "invalid experience")
    strings(p["preferences"], "preferences")
    strings(p["completed"], "completed")
    require(isinstance(data["actions"], list) and len(data["actions"]) >= 2,
            "at least two actions are required")
    ids = set()
    for action in data["actions"]:
        fields(action, ("id", "title", "prerequisites", "tags", "format",
                        "difficulty"), "action")
        text(action["id"], "action id")
        text(action["title"], "action title")
        require(action["id"] not in ids, "duplicate action id")
        ids.add(action["id"])
        strings(action["prerequisites"], "prerequisites")
        strings(action["tags"], "tags")
        require(action["format"] in ("video", "text", "interactive"),
                "invalid action format")
        require(type(action["difficulty"]) is int and 1 <= action["difficulty"] <= 3,
                "difficulty must be an integer in 1..3")
    require(set(p["completed"]) <= ids, "unknown completed action")
    for a in data["actions"]:
        require(set(a["prerequisites"]) <= ids, "unknown prerequisite")
    resolved = set()
    while len(resolved) < len(ids):
        newly = {a["id"] for a in data["actions"] if a["id"] not in resolved
                 and set(a["prerequisites"]) <= resolved}
        require(bool(newly), "cyclic action prerequisites")
        resolved.update(newly)
    require(isinstance(data["tickets"], list), "tickets must be a list")
    ticket_ids = set()
    for ticket in data["tickets"]:
        fields(ticket, ("id", "text", "action_id", "severity"), "ticket")
        text(ticket["id"], "ticket id")
        text(ticket["text"], "ticket text")
        require(ticket["id"] not in ticket_ids, "duplicate ticket id")
        ticket_ids.add(ticket["id"])
        require(isinstance(ticket["action_id"], str) and ticket["action_id"] in ids,
                "ticket references unknown action")
        require(ticket["severity"] in LEVELS, "invalid ticket severity")
    r = data["routing"]
    fields(r, ("rules", "fallback"), "routing")
    require(isinstance(r["rules"], list), "routing rules must be a list")
    categories = set()
    for rule in r["rules"] + [r["fallback"]]:
        is_fallback = rule is r["fallback"]
        fields(rule, ("category", "owner", "priority") if is_fallback else
               ("category", "owner", "priority", "keywords"), "routing rule")
        text(rule["category"], "category")
        text(rule["owner"], "accountable owner")
        require(rule["category"] not in categories, "duplicate routing category")
        categories.add(rule["category"])
        require(rule["priority"] in LEVELS, "invalid routing priority")
        if not is_fallback:
            strings(rule["keywords"], "keywords")
            require(bool(rule["keywords"]), "rule keywords cannot be empty")
            require(all(re.fullmatch(r"[A-Za-z]+", k) for k in rule["keywords"]),
                    "keywords must be single alphabetic words")
    return data


def validate(envelope, expected_stage):
    """Single validation boundary for input and all accumulated stage outputs."""
    fields(envelope, ("schema_version", "status", "stage", "input", "results"),
           "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "invalid envelope schema")
    require(envelope["status"] == "ok", "invalid envelope status")
    require(envelope["stage"] == expected_stage, "unexpected pipeline stage")
    data = validate_input(envelope["input"])
    require(isinstance(envelope["results"], dict), "results must be an object")
    count = 0 if expected_stage == "input" else STAGES.index(expected_stage) + 1
    require(set(envelope["results"]) == set(STAGES[:count]), "invalid stage history")
    if count:
        journey = envelope["results"]["journey"]
        fields(journey, ("next_actions", "steps"), "journey")
        expected = recommend(data)
        require(journey == expected, "invalid journey output or prerequisite order")
    if count >= 2:
        adaptive = envelope["results"]["adaptive"]
        require(adaptive == onboard(data, envelope["results"]["journey"]),
                "invalid adaptive output")
    if count >= 3:
        require(envelope["results"]["triage"] ==
                route(data, envelope["results"]["adaptive"]),
                "invalid triage output")
    if count == 4:
        require(envelope["results"]["sentiment"] ==
                prioritize(envelope["results"]["triage"]),
                "invalid sentiment output")
    return envelope


def score_action(action, profile):
    target = {"beginner": 1, "intermediate": 2, "expert": 3}[profile["experience"]]
    matched = sorted(set(profile["preferences"]) &
                     (set(action["tags"]) | {action["format"]}))
    score = 3 * len(matched) - abs(target - action["difficulty"])
    explanation = ("Preference matches: " + (", ".join(matched) or "none") +
                   "; experience target: " + str(target) +
                   "; difficulty: " + str(action["difficulty"]) + ".")
    return score, explanation


def recommend(data):
    profile = data["profile"]
    completed = set(profile["completed"])
    actions = data["actions"]

    def eligible(done):
        available = [a for a in actions if a["id"] not in done
                     and set(a["prerequisites"]) <= done]
        return sorted(available, key=lambda a: (-score_action(a, profile)[0], a["id"]))

    def entry(action, done):
        score, explanation = score_action(action, profile)
        return {"action_id": action["id"], "score": score,
                "satisfied_prerequisites": sorted(set(action["prerequisites"]) & done),
                "explanation": explanation}

    next_actions = [entry(a, completed) for a in eligible(completed)]
    steps = []
    for _ in range(2):
        choices = eligible(completed)
        require(bool(choices), "cannot construct a valid two-step journey")
        selected = choices[0]
        steps.append(entry(selected, completed))
        completed.add(selected["id"])
    return {"next_actions": next_actions, "steps": steps}


def onboard(data, journey):
    actions = {a["id"]: a for a in data["actions"]}
    experience = data["profile"]["experience"]
    depth = {"beginner": "guided", "intermediate": "standard", "expert": "fast-track"}[
        experience]
    completed = set(data["profile"]["completed"])
    modules = []
    for step in journey["steps"]:
        a = actions[step["action_id"]]
        prerequisites = sorted(a["prerequisites"])
        require(set(prerequisites) <= completed, "unmet onboarding prerequisite")
        preferred = [f for f in data["profile"]["preferences"]
                     if f in ("video", "text", "interactive")]
        delivery = preferred[0] if preferred else a["format"]
        modules.append({
            "action_id": a["id"], "title": a["title"], "depth": depth,
            "format": delivery, "prerequisites": prerequisites,
            "explanation": f"{depth} for {experience}; format {delivery} "
                           f"from {'preference' if preferred else 'action default'}; "
                           "prerequisites satisfied by completed actions or earlier modules.",
        })
        completed.add(a["id"])
    return {"modules": modules, "experience": experience}


def tokens(value):
    return re.findall(r"[a-z]+", value.lower())


def higher(*values):
    return max(values, key=LEVELS.index)


def route(data, adaptive):
    modules = {m["action_id"]: m for m in adaptive["modules"]}
    records = []
    for ticket in data["tickets"]:
        words = set(tokens(ticket["text"]))
        matches = [(rule, sorted(words & {k.lower() for k in rule["keywords"]}))
                   for rule in data["routing"]["rules"]]
        matches = [(rule, hits) for rule, hits in matches if hits]
        # Most keyword matches wins; configured rule order breaks ties.
        rule, hits = max(matches, key=lambda pair: len(pair[1])) if matches else (
            data["routing"]["fallback"], [])
        module = modules.get(ticket["action_id"])
        priority = higher(rule["priority"], ticket["severity"],
                          "medium" if module else "low")
        records.append({
            "ticket_id": ticket["id"], "text": ticket["text"],
            "action_id": ticket["action_id"], "severity": ticket["severity"],
            "category": rule["category"], "owner": rule["owner"],
            "priority": priority, "matched_keywords": hits,
            "onboarding": copy.deepcopy(module),
            "explanation": ("Matched keywords: " + (", ".join(hits) or "fallback") +
                            "; priority is maximum of routing, severity and active "
                            "onboarding floor; owner is accountable."),
        })
    return {"tickets": records}


def sentiment(text_value):
    words = tokens(text_value)
    contributions = []
    lexicon = {**POSITIVE, **NEGATIVE}
    for i, word in enumerate(words):
        if word not in lexicon:
            continue
        value = lexicon[word]
        negated = i > 0 and words[i - 1] in ("not", "never", "no")
        if negated:
            value = -value
        contributions.append({"word": word, "position": i, "value": value,
                              "negated": negated})
    score = sum(c["value"] for c in contributions)
    return {"score": score,
            "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "contributions": contributions}


def prioritize(triage):
    issues = []
    for routed in triage["tickets"]:
        analysis = sentiment(routed["text"])
        # Negative wording can raise urgency one level; never lowers severity.
        floor = max(LEVELS.index(routed["priority"]), LEVELS.index(routed["severity"]))
        boost = int(analysis["score"] <= -2)
        rank = min(3, floor + boost)
        issues.append({
            **copy.deepcopy(routed), "sentiment": analysis,
            "final_priority": LEVELS[rank], "priority_rank": rank,
            "priority_explanation": f"Baseline rank {floor}; negative sentiment "
                                    f"boost {boost}; capped at critical.",
        })
    issues.sort(key=lambda i: (-i["priority_rank"], i["sentiment"]["score"], i["ticket_id"]))
    return {"issues": issues, "scoring_policy": {
        "positive": POSITIVE.copy(), "negative": NEGATIVE.copy(),
        "negation": "Immediately preceding not/never/no reverses one scored word.",
        "escalation": "Score <= -2 raises priority one level; severity is a floor.",
        "limitations": "English token lexicon only; no sarcasm or semantic inference.",
    }}


def advance(envelope, stage):
    require(stage in STAGES, "unknown stage")
    index = STAGES.index(stage)
    previous = "input" if index == 0 else STAGES[index - 1]
    validate(envelope, previous)
    result = copy.deepcopy(envelope)
    data = result["input"]
    history = result["results"]
    if stage == "journey":
        output = recommend(data)
    elif stage == "adaptive":
        output = onboard(data, history["journey"])
    elif stage == "triage":
        output = route(data, history["adaptive"])
    else:
        output = prioritize(history["triage"])
    history[stage] = output
    result["stage"] = stage
    return validate(result, stage)


def run(data):
    envelope = {"schema_version": 1, "status": "ok", "stage": "input",
                "input": copy.deepcopy(data), "results": {}}
    validate(envelope, "input")
    for stage in STAGES:
        envelope = advance(envelope, stage)
    return envelope


def reject_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON object key: " + key)
        obj[key] = value
    return obj


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant,
                             object_pairs_hook=unique_object)
        output = run(data)
        code = 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        output = {"schema_version": 1, "status": "error", "error": str(error)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
