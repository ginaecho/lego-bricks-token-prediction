"""Synthetic, offline journey -> evidence research -> accountable triage.

Usage: python -B implementation.py example_input.json
Only structured claims are synthesized; stance is supplied evidence, not verified fact.
"""

import copy
import json
import sys


VERSION = "1.0"
PRIORITIES = ("low", "normal", "high", "urgent")
REASONS = ("disagreement", "missing_evidence", "uncertain", "limited_sources", "resolved")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def record(value, keys, path):
    require(type(value) is dict, f"{path}: expected object")
    require(set(value) == set(keys.split()), f"{path}: expected fields {keys}")


def text(value, path):
    require(type(value) is str and bool(value.strip()) and len(value) <= 2000,
            f"{path}: expected nonblank string of at most 2000 characters")
    require(value == value.strip(), f"{path}: surrounding whitespace is not allowed")


def array(value, path, minimum=0, maximum=100):
    require(type(value) is list and minimum <= len(value) <= maximum,
            f"{path}: expected list with {minimum}..{maximum} entries")


def strings(value, path, minimum=0):
    array(value, path, minimum)
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), f"{path}: duplicate entries")


def _validate_request(data):
    record(data, "schema_version synthetic request_id user actions documents routing", "request")
    require(data["schema_version"] == VERSION, "request: unsupported schema_version")
    require(data["synthetic"] is True, "request: synthetic must be true")
    text(data["request_id"], "request_id")
    user = data["user"]
    record(user, "id goals completed_actions", "user")
    text(user["id"], "user.id")
    strings(user["goals"], "user.goals", 1)
    strings(user["completed_actions"], "user.completed_actions")
    array(data["actions"], "actions", 2)
    actions = {}
    for action in data["actions"]:
        record(action, "id title goals prerequisites research_topics", "action")
        text(action["id"], "action.id")
        text(action["title"], "action.title")
        strings(action["goals"], "action.goals", 1)
        strings(action["prerequisites"], "action.prerequisites")
        strings(action["research_topics"], "action.research_topics", 1)
        require(action["id"] not in actions, "actions: duplicate id")
        actions[action["id"]] = action
    for action in actions.values():
        require(set(action["prerequisites"]) <= actions.keys(),
                "action: unknown prerequisite")
    completed = set(user["completed_actions"])
    require(completed <= actions.keys(), "user: unknown completed action")
    for action_id in completed:
        require(set(actions[action_id]["prerequisites"]) <= completed,
                "user: completed actions must include their prerequisites")
    visited, visiting = set(), set()

    def visit(action_id):
        require(action_id not in visiting, "actions: prerequisite cycle")
        if action_id in visited:
            return
        visiting.add(action_id)
        for prerequisite in actions[action_id]["prerequisites"]:
            visit(prerequisite)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in actions:
        visit(action_id)
    array(data["documents"], "documents")
    document_ids = set()
    for doc in data["documents"]:
        record(doc, "id title claims", "document")
        text(doc["id"], "document.id")
        text(doc["title"], "document.title")
        require(doc["id"] not in document_ids, "documents: duplicate id")
        document_ids.add(doc["id"])
        array(doc["claims"], "document.claims")
        topics = set()
        for claim in doc["claims"]:
            record(claim, "topic stance text", "claim")
            text(claim["topic"], "claim.topic")
            text(claim["text"], "claim.text")
            require(claim["stance"] in ("supports", "opposes", "uncertain"),
                    "claim: invalid stance")
            require(claim["topic"] not in topics, "document: duplicate claim topic")
            topics.add(claim["topic"])
    routing = data["routing"]
    record(routing, "categories fallback_category priorities", "routing")
    array(routing["categories"], "routing.categories", 1)
    category_ids = set()
    for category in routing["categories"]:
        record(category, "id topics owner queue", "category")
        for field in ("id", "owner", "queue"):
            text(category[field], "category." + field)
        strings(category["topics"], "category.topics")
        require(category["id"] not in category_ids, "categories: duplicate id")
        category_ids.add(category["id"])
    text(routing["fallback_category"], "routing.fallback_category")
    require(routing["fallback_category"] in category_ids, "routing: unknown fallback category")
    record(routing["priorities"], " ".join(REASONS), "routing.priorities")
    for priority in routing["priorities"].values():
        require(priority in PRIORITIES, "routing: invalid priority")


def _matches(action, request):
    return [goal for goal in request["user"]["goals"] if goal in action["goals"]]


def _score(action, request):
    goals = request["user"]["goals"]
    return sum(len(goals) - index for index, goal in enumerate(goals)
               if goal in action["goals"])


def _make_journey(request):
    actions = {action["id"]: action for action in request["actions"]}
    completed = set(request["user"]["completed_actions"])
    available = [action for action in actions.values()
                 if action["id"] not in completed
                 and set(action["prerequisites"]) <= completed]
    available.sort(key=lambda action: (-_score(action, request), action["id"]))
    pairs = []
    for first in available:
        for second in actions.values():
            if second["id"] in completed | {first["id"]}:
                continue
            if set(second["prerequisites"]) <= completed | {first["id"]}:
                total = _score(first, request) + _score(second, request)
                if total:
                    pairs.append((first, second))
    require(bool(pairs), "journey: no goal-relevant feasible two-step journey")
    first, second = min(
        pairs,
        key=lambda pair: (-sum(_score(action, request) for action in pair),
                          -_score(pair[0], request), pair[0]["id"], pair[1]["id"]),
    )
    steps = [{
        "action_id": action["id"], "title": action["title"],
        "prerequisites": list(action["prerequisites"]),
        "matching_goals": _matches(action, request),
        "research_topics": list(action["research_topics"]),
    } for action in (first, second)]
    return {
        "user_id": request["user"]["id"],
        "next_actions": [{"action_id": action["id"], "matching_goals": _matches(action, request)}
                         for action in available],
        "steps": steps,
        "research_topics": sorted({topic for step in steps for topic in step["research_topics"]}),
    }


def _make_research(request, journey):
    reports = []
    for topic in journey["research_topics"]:
        evidence = []
        for doc in sorted(request["documents"], key=lambda item: item["id"]):
            for claim in doc["claims"]:
                if claim["topic"] == topic:
                    evidence.append({
                        "document_id": doc["id"], "document_title": doc["title"],
                        "stance": claim["stance"], "text": claim["text"],
                    })
        positions = {stance: [item["document_id"] for item in evidence if item["stance"] == stance]
                     for stance in ("supports", "opposes", "uncertain")}
        reasons, questions = [], []
        if positions["supports"] and positions["opposes"]:
            reasons.append("disagreement")
            questions.append(f"Which conflicting position on '{topic}' is better supported?")
        if not evidence:
            reasons.append("missing_evidence")
            questions.append(f"What evidence is available for '{topic}'?")
        if positions["uncertain"]:
            reasons.append("uncertain")
            questions.append(f"What would resolve the uncertain evidence on '{topic}'?")
        if len(evidence) == 1:
            reasons.append("limited_sources")
            questions.append(f"Can an independent document corroborate '{topic}'?")
        if not reasons:
            reasons.append("resolved")
        if not evidence:
            synthesis = f"No supplied document addresses '{topic}'; no conclusion is supported."
        else:
            counts = ", ".join(f"{len(positions[stance])} {stance}"
                               for stance in ("supports", "opposes", "uncertain"))
            synthesis = (f"Supplied evidence for '{topic}': {counts} across "
                         f"{len(evidence)} documents. "
                         "Stances are reported, not verified; document counts do not establish truth.")
        reports.append({
            "topic": topic,
            "action_ids": [step["action_id"] for step in journey["steps"]
                           if topic in step["research_topics"]],
            "evidence": evidence,
            "source_count": len(evidence),
            "synthesis": synthesis,
            "disagreement": ({"supports": positions["supports"], "opposes": positions["opposes"]}
                             if "disagreement" in reasons else None),
            "reason_codes": reasons,
            "unresolved_questions": questions,
        })
    return {
        "journey_action_ids": [step["action_id"] for step in journey["steps"]],
        "document_ids": sorted({item["document_id"] for report in reports for item in report["evidence"]}),
        "topics": reports,
    }


def _make_triage(request, research):
    routing = request["routing"]
    categories = routing["categories"]
    fallback = next(category for category in categories
                    if category["id"] == routing["fallback_category"])
    tickets = []
    for index, report in enumerate(research["topics"], 1):
        category = next((category for category in categories if report["topic"] in category["topics"]),
                        fallback)
        priority = max((routing["priorities"][reason] for reason in report["reason_codes"]),
                       key=PRIORITIES.index)
        tickets.append({
            "id": f"{request['request_id']}:ticket:{index}",
            "topic": report["topic"],
            "action_ids": list(report["action_ids"]),
            "document_ids": [item["document_id"] for item in report["evidence"]],
            "category": category["id"],
            "priority": priority,
            "owner": category["owner"],
            "queue": category["queue"],
            "reason_codes": list(report["reason_codes"]),
            "questions": list(report["unresolved_questions"]),
            "summary": report["synthesis"],
        })
    return {"tickets": tickets}


def _equal(actual, expected, path):
    # Strict recursive comparison rejects bool-as-int and extra/missing fields.
    require(type(actual) is type(expected), f"{path}: invalid type")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{path}: invalid fields")
        for key in expected:
            _equal(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{path}: invalid length")
        for index, (item, reference) in enumerate(zip(actual, expected)):
            _equal(item, reference, f"{path}[{index}]")
    else:
        require(actual == expected, f"{path}: value violates the deterministic contract")


def validate(stage, value, request=None, journey=None, research=None):
    """Shared validation boundary, including exact provenance and routing rules.

    Stage schemas are the canonical deterministic records generated above.
    Recomputing these small records rejects omissions, additions, stale handoffs,
    invented quotations, incorrect priorities, and altered owner assignments.
    """
    if stage == "request":
        _validate_request(value)
        return value
    require(stage in ("journey", "research", "triage", "output"), "unknown validation stage")
    validate("request", request)
    if stage == "journey":
        expected = _make_journey(request)
    else:
        validate("journey", journey, request=request)
        if stage == "research":
            expected = _make_research(request, journey)
        else:
            validate("research", research, request=request, journey=journey)
            expected = _make_triage(request, research)
            if stage == "output":
                expected = {
                    "schema_version": VERSION, "synthetic": True,
                    "request_id": request["request_id"], "status": "ok",
                    "journey": journey, "research": research, "triage": expected,
                }
    _equal(value, expected, stage)
    return value


def recommend_journey(request):
    validate("request", request)
    return validate("journey", _make_journey(request), request=request)


def deep_research(request, journey):
    validate("journey", journey, request=request)
    return validate("research", _make_research(request, journey),
                    request=request, journey=journey)


def triage_tickets(request, journey, research):
    validate("research", research, request=request, journey=journey)
    return validate("triage", _make_triage(request, research),
                    request=request, journey=journey, research=research)


def run_pipeline(data):
    request = copy.deepcopy(validate("request", data))
    journey = recommend_journey(request)
    research = deep_research(request, journey)
    triage = triage_tickets(request, journey, research)
    result = {
        "schema_version": VERSION, "synthetic": True,
        "request_id": request["request_id"], "status": "ok",
        "journey": journey, "research": research, "triage": triage,
    }
    return validate("output", result, request=request, journey=journey, research=research)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON: duplicate field {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValidationError("JSON: non-finite number " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], "r", encoding="utf-8-sig") as handle:
            data = json.load(handle, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        result = run_pipeline(data)
    except (ValidationError, ValueError, OSError, RecursionError) as exc:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(exc)},
                         ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
