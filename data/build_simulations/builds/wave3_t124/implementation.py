"""Deterministic synthetic support-to-discovery reference pipeline (stdlib only)."""
import copy
import json
import re
import sys


class ValidationError(ValueError):
    pass


ORDER = ("triage", "feedback", "adaptive", "recommend")
PRIORITIES = ("low", "normal", "high", "urgent")


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def strings(value, path):
    require(isinstance(value, list), path + " must be an array")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), path + " contains duplicates")


def records(value, path):
    require(isinstance(value, list), path + " must be an array")
    ids = []
    for record in value:
        require(isinstance(record, dict), path + " entries must be objects")
        text(record.get("id"), path + ".id")
        ids.append(record["id"])
    require(len(ids) == len(set(ids)), path + " IDs must be unique")


def number(value, path):
    require(type(value) in (int, float) and 0 <= value < float("inf"),
            path + " must be a finite nonnegative number")


def normalize(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def matches(value, keyword):
    return (" " + normalize(keyword) + " ") in (" " + normalize(value) + " ")


def validate_request(data):
    obj(data, ("schema_version", "fixture_label", "tickets", "profile", "catalog", "config"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "unsupported schema_version")
    text(data["fixture_label"], "fixture_label")
    records(data["tickets"], "tickets")
    for ticket in data["tickets"]:
        obj(ticket, ("id", "text"), "ticket")
        text(ticket["text"], "ticket.text")
        require(bool(normalize(ticket["text"])), "ticket must contain words")
    profile = data["profile"]
    obj(profile, ("experience", "preferences", "completed_steps", "budget", "excluded_products"), "profile")
    require(profile["experience"] in ("beginner", "intermediate", "expert"), "invalid experience")
    for key in ("preferences", "completed_steps", "excluded_products"):
        strings(profile[key], "profile." + key)
    number(profile["budget"], "budget")
    config = data["config"]
    obj(config, ("categories", "default_category", "priority_keywords", "themes", "onboarding_steps", "recommendation_limit"), "config")
    categories = config["categories"]
    require(isinstance(categories, dict) and bool(categories), "categories must be a nonempty object")
    for category, rule in categories.items():
        text(category, "category")
        obj(rule, ("keywords", "owner", "priority"), "category rule")
        strings(rule["keywords"], "category.keywords")
        for keyword in rule["keywords"]:
            require(bool(normalize(keyword)), "keywords must contain words")
        text(rule["owner"], "category.owner")
        require(rule["priority"] in PRIORITIES, "invalid category priority")
    require(isinstance(config["default_category"], str) and config["default_category"] in categories,
            "unknown default_category")
    obj(config["priority_keywords"], PRIORITIES, "priority_keywords")
    for keywords in config["priority_keywords"].values():
        strings(keywords, "priority keywords")
        require(all(normalize(k) for k in keywords), "priority keywords must contain words")
    require(isinstance(config["themes"], dict), "themes must be an object")
    for theme, keywords in config["themes"].items():
        text(theme, "theme")
        strings(keywords, "theme keywords")
        require(all(normalize(k) for k in keywords), "theme keywords must contain words")
    steps = config["onboarding_steps"]
    records(steps, "onboarding_steps")
    step_ids = {s["id"] for s in steps}
    for step in steps:
        obj(step, ("id", "title", "experiences", "tags", "prerequisites"), "onboarding step")
        text(step["title"], "step.title")
        for key in ("experiences", "tags", "prerequisites"):
            strings(step[key], "step." + key)
        require(set(step["experiences"]) <= {"beginner", "intermediate", "expert"}, "invalid step experience")
        require(set(step["prerequisites"]) <= step_ids, "unknown prerequisite")
    visited, active = set(), set()
    by_id = {s["id"]: s for s in steps}

    def visit(step_id):
        require(step_id not in active, "onboarding prerequisite cycle")
        if step_id in visited:
            return
        active.add(step_id)
        for parent in by_id[step_id]["prerequisites"]:
            visit(parent)
        active.remove(step_id)
        visited.add(step_id)

    for step_id in sorted(step_ids):
        visit(step_id)
    require(set(profile["completed_steps"]) <= step_ids, "unknown completed step")
    for step_id in profile["completed_steps"]:
        require(set(by_id[step_id]["prerequisites"]) <= set(profile["completed_steps"]),
                "completed steps must include prerequisites")
    records(data["catalog"], "catalog")
    for product in data["catalog"]:
        obj(product, ("id", "name", "tags", "price", "available", "requires_steps"), "product")
        text(product["name"], "product.name")
        strings(product["tags"], "product.tags")
        strings(product["requires_steps"], "product.requires_steps")
        require(set(product["requires_steps"]) <= step_ids, "unknown product prerequisite")
        number(product["price"], "product.price")
        require(type(product["available"]) is bool, "product.available must be boolean")
    require(set(profile["excluded_products"]) <= {p["id"] for p in data["catalog"]}, "unknown excluded product")
    limit = config["recommendation_limit"]
    require(type(limit) is int and 1 <= limit <= 100, "recommendation_limit must be 1..100")
    return data


def triage_result(state):
    request = state["request"]
    config = request["config"]
    result = []
    for ticket in request["tickets"]:
        scores = {name: sum(matches(ticket["text"], k) for k in rule["keywords"])
                  for name, rule in config["categories"].items()}
        best = sorted(scores, key=lambda name: (-scores[name], name))[0]
        category = best if scores[best] else config["default_category"]
        rule = config["categories"][category]
        rank = PRIORITIES.index(rule["priority"])
        for priority, keywords in config["priority_keywords"].items():
            if any(matches(ticket["text"], k) for k in keywords):
                rank = max(rank, PRIORITIES.index(priority))
        result.append({"ticket_id": ticket["id"], "text": ticket["text"], "category": category,
                       "priority": PRIORITIES[rank], "owner": rule["owner"],
                       "reason": "keyword matches: " + str(scores[category])})
    return {"tickets": result}


def feedback_result(state):
    groups = {}
    config = state["request"]["config"]
    for ticket in state["stages"]["triage"]["tickets"]:
        key = normalize(ticket["text"])
        groups.setdefault(key, []).append(ticket)
    entries = []
    for index, key in enumerate(sorted(groups), 1):
        tickets = groups[key]
        themes = sorted(theme for theme, keywords in config["themes"].items()
                        if any(matches(key, keyword) for keyword in keywords))
        if not themes:
            themes = sorted({t["category"] for t in tickets})
        entries.append({"id": "feedback-" + str(index), "themes": themes,
                        "ticket_ids": sorted(t["ticket_id"] for t in tickets),
                        "owners": sorted({t["owner"] for t in tickets}),
                        "priority": max((t["priority"] for t in tickets), key=PRIORITIES.index),
                        "excerpts": [{"ticket_id": t["ticket_id"], "text": t["text"][:240]}
                                     for t in sorted(tickets, key=lambda t: t["ticket_id"])]})
    themes = []
    for theme in sorted({theme for entry in entries for theme in entry["themes"]}):
        supporting = [entry for entry in entries if theme in entry["themes"]]
        themes.append({"name": theme, "unique_feedback_count": len(supporting),
                       "feedback_ids": [entry["id"] for entry in supporting]})
    return {"entries": entries, "themes": themes}


def adaptive_result(state):
    request = state["request"]
    profile = request["profile"]
    feedback = state["stages"]["feedback"]
    themes = {t["name"] for t in feedback["themes"]}
    interests = themes | set(profile["preferences"])
    steps = {s["id"]: s for s in request["config"]["onboarding_steps"]}
    completed = set(profile["completed_steps"])
    reasons = {}
    for step_id, step in steps.items():
        why = []
        if profile["experience"] in step["experiences"]:
            why.append("experience: " + profile["experience"])
        for tag in sorted(set(step["tags"]) & interests):
            why.append(("feedback theme: " if tag in themes else "preference: ") + tag)
        if why and step_id not in completed:
            reasons[step_id] = why
    plan, emitted = [], set()

    def include(step_id, prerequisite_for=None):
        if step_id in completed or step_id in emitted:
            return
        step = steps[step_id]
        for parent in sorted(step["prerequisites"]):
            include(parent, step_id)
        plan.append({"step_id": step_id, "title": step["title"],
                     "prerequisites": step["prerequisites"],
                     "reasons": reasons.get(step_id, ["prerequisite for: " + str(prerequisite_for)])})
        emitted.add(step_id)

    for step_id in sorted(reasons):
        include(step_id)
    return {"plan": plan, "interests": sorted(interests),
            "completed_steps": sorted(completed),
            "feedback_ids": [entry["id"] for entry in feedback["entries"]],
            "budget": profile["budget"], "excluded_products": sorted(profile["excluded_products"])}


def recommend_result(state):
    adaptive = state["stages"]["adaptive"]
    feedback = state["stages"]["feedback"]
    weighted_themes = {t["name"]: t["unique_feedback_count"] for t in feedback["themes"]}
    preferences = set(state["request"]["profile"]["preferences"])
    ranked, excluded = [], []
    for product in state["request"]["catalog"]:
        reasons = []
        if not product["available"]:
            reasons.append("unavailable")
        if product["price"] > adaptive["budget"]:
            reasons.append("over budget")
        if product["id"] in adaptive["excluded_products"]:
            reasons.append("explicit exclusion")
        if not set(product["requires_steps"]) <= set(adaptive["completed_steps"]):
            reasons.append("onboarding prerequisites not completed")
        if reasons:
            excluded.append({"product_id": product["id"], "reasons": reasons})
            continue
        tags = sorted(set(product["tags"]) & set(adaptive["interests"]))
        score = sum(2 * (tag in preferences) + weighted_themes.get(tag, 0) for tag in tags)
        ranked.append({"product_id": product["id"], "name": product["name"], "price": product["price"],
                       "score": score, "matched_interests": tags,
                       "reasons": ["matches interest: " + tag for tag in tags] or ["budget-safe general discovery"],
                       "feedback_ids": [entry["id"] for entry in feedback["entries"]
                                        if set(entry["themes"]) & set(tags)]})
    ranked.sort(key=lambda p: (-p["score"], p["price"], p["product_id"]))
    return {"products": ranked[:state["request"]["config"]["recommendation_limit"]],
            "excluded": sorted(excluded, key=lambda p: p["product_id"]),
            "strategy": "interest overlap, then lower price, then product ID"}


BUILDERS = (triage_result, feedback_result, adaptive_result, recommend_result)


def validate_state(state, completed):
    """One boundary validator; recomputation validates shape, provenance and semantics."""
    obj(state, ("schema_version", "status", "request", "stages"), "state")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1, "invalid state version")
    require(state["status"] == ("ok" if completed == 4 else "in_progress"), "invalid state status")
    validate_request(state["request"])
    require(isinstance(state["stages"], dict) and list(state["stages"]) == list(ORDER[:completed]),
            "invalid stage sequence")
    for name, builder in zip(ORDER[:completed], BUILDERS):
        expected = builder(state)
        require(json.dumps(state["stages"][name], sort_keys=True, allow_nan=False) ==
                json.dumps(expected, sort_keys=True, allow_nan=False), "invalid " + name + " output")
    return state


def advance(state, stage):
    require(stage in ORDER, "unknown stage")
    index = ORDER.index(stage)
    validate_state(state, index)
    result = copy.deepcopy(state)
    result["stages"][stage] = BUILDERS[index](result)
    result["status"] = "ok" if index == 3 else "in_progress"
    return validate_state(result, index + 1)


def run(data):
    validate_request(data)
    state = {"schema_version": 1, "status": "in_progress", "request": copy.deepcopy(data), "stages": {}}
    for stage in ORDER:
        state = advance(state, stage)
    return state


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=unique_object)
        result = run(data)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
