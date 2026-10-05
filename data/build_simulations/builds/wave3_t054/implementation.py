"""Synthetic, deterministic research -> journey -> accountable triage reference CLI.

Run: python -B implementation.py example_input.json
Only Python's standard library is used. Evidence is lexical, not fact verification.
"""

import copy
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 10000,
            name + " must be a nonblank string of at most 10000 characters")


def fields(value, names, name):
    require(isinstance(value, dict) and set(value) == set(names.split()),
            name + " has missing or unknown fields")


def strings(value, name):
    require(isinstance(value, list) and len(value) <= 100, name + " must be a bounded list")
    for item in value:
        text(item, name + " item")
    require(len(set(value)) == len(value), name + " must not contain duplicates")


STOP = {"a", "an", "the", "to", "for", "of", "and", "is", "in", "how", "do", "i", "can"}
PRIORITIES = ["low", "normal", "high", "urgent"]


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold())) - STOP


def validate(stage, value, document=None, previous=None):
    """One shared structural and semantic validation boundary for all handoffs."""
    if stage == "input":
        fields(value, "schema_version fixture_label question sources actions completed ticket routing", stage)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "unsupported schema_version")
        text(value["fixture_label"], "fixture_label")
        text(value["question"], "question")
        require(bool(tokens(value["question"])), "question needs searchable terms")
        for collection in ("sources", "actions"):
            require(isinstance(value[collection], list) and len(value[collection]) <= 100,
                    collection + " must be a list of at most 100 items")
        source_ids = set()
        for source in value["sources"]:
            fields(source, "id title text reliability", "source")
            for key in ("id", "title", "text"):
                text(source[key], "source." + key)
            reliability = source["reliability"]
            require(type(reliability) in (int, float) and math.isfinite(reliability)
                    and 0 <= reliability <= 1, "reliability must be finite and between 0 and 1")
            require(source["id"] not in source_ids, "duplicate source id")
            source_ids.add(source["id"])
        actions = {}
        for action in value["actions"]:
            fields(action, "id title prerequisites evidence_terms category", "action")
            for key in ("id", "title", "category"):
                text(action[key], "action." + key)
            strings(action["prerequisites"], "prerequisites")
            strings(action["evidence_terms"], "evidence_terms")
            require(bool(action["evidence_terms"]) and all(tokens(t) for t in action["evidence_terms"]),
                    "actions need searchable evidence_terms")
            require(action["id"] not in actions, "duplicate action id")
            actions[action["id"]] = action
        strings(value["completed"], "completed")
        require(set(value["completed"]) <= set(actions), "unknown completed action")
        for action in actions.values():
            require(set(action["prerequisites"]) <= set(actions), "unknown prerequisite")
        visiting, visited = set(), set()

        def visit(action_id):
            require(action_id not in visiting, "cyclic prerequisites")
            if action_id in visited:
                return
            visiting.add(action_id)
            for dependency in actions[action_id]["prerequisites"]:
                visit(dependency)
            visiting.remove(action_id)
            visited.add(action_id)

        for action_id in actions:
            visit(action_id)
        for action_id in value["completed"]:
            require(set(actions[action_id]["prerequisites"]) <= set(value["completed"]),
                    "completed actions must include their prerequisites")
        fields(value["ticket"], "customer_id text severity", "ticket")
        for key in ("customer_id", "text"):
            text(value["ticket"][key], "ticket." + key)
        require(value["ticket"]["severity"] in ("minor", "standard", "major", "critical"),
                "unsupported ticket severity")
        routing = value["routing"]
        fields(routing, "categories fallback_category priority_by_severity", "routing")
        require(isinstance(routing["categories"], dict) and 0 < len(routing["categories"]) <= 100,
                "routing categories must be a nonempty bounded object")
        for category, rule in routing["categories"].items():
            text(category, "category")
            fields(rule, "keywords team owner", "routing rule")
            strings(rule["keywords"], "keywords")
            text(rule["team"], "team")
            text(rule["owner"], "owner")
        require(isinstance(routing["fallback_category"], str)
                and routing["fallback_category"] in routing["categories"], "unknown fallback category")
        priorities = routing["priority_by_severity"]
        fields(priorities, "minor standard major critical", "priority_by_severity")
        require(all(p in PRIORITIES for p in priorities.values()), "invalid priority")
        require(all(action["category"] in routing["categories"] for action in actions.values()),
                "unknown action category")
        return value

    require(document is not None, "stage validation requires the input document")
    if stage == "research":
        fields(value, "question status evidence confidence limitations", stage)
        require(value["question"] == document["question"], "research question changed")
        require(isinstance(value["evidence"], list), "evidence must be a list")
        sources = {s["id"]: s for s in document["sources"]}
        seen = set()
        scores = []
        for evidence in value["evidence"]:
            fields(evidence, "source_id title excerpt relevance reliability", "evidence")
            source_id = evidence["source_id"]
            require(isinstance(source_id, str) and source_id in sources and source_id not in seen,
                    "unknown or repeated citation")
            seen.add(source_id)
            source = sources[source_id]
            relevance = round(len(tokens(document["question"]) & tokens(source["text"])) /
                              len(tokens(document["question"])), 6)
            require(relevance > 0 and source["reliability"] > 0, "unsupported evidence")
            require(evidence["title"] == source["title"] and evidence["excerpt"] == source["text"],
                    "evidence must quote the source exactly")
            require(type(evidence["relevance"]) in (int, float)
                    and evidence["relevance"] == relevance
                    and evidence["reliability"] == source["reliability"], "invalid evidence score")
            scores.append(relevance * source["reliability"])
        expected_sources = {s["id"] for s in document["sources"]
                            if s["reliability"] > 0 and tokens(document["question"]) & tokens(s["text"])}
        require(seen == expected_sources, "research omitted eligible evidence")
        require(value["status"] == ("supported" if seen else "insufficient"), "invalid research status")
        require(type(value["confidence"]) in (int, float)
                and value["confidence"] == round(max(scores, default=0), 6), "invalid confidence")
        strings(value["limitations"], "limitations")
        require(bool(value["limitations"]), "research must state limitations")
    elif stage == "journey":
        validate("research", previous, document)
        fields(value, "status next_actions steps evidence_ids reason", stage)
        actions = {a["id"]: a for a in document["actions"]}
        citations = [e["source_id"] for e in previous["evidence"]]
        require(value["evidence_ids"] == citations, "journey lost research citations")
        strings(value["next_actions"], "next_actions")
        strings(value["steps"], "steps")
        supported = supported_actions(document, previous)
        completed = set(document["completed"])
        available = {a for a in supported if a not in completed
                     and set(actions[a]["prerequisites"]) <= completed}
        require(set(value["next_actions"]) == available, "invalid next actions")
        require(value["status"] in ("ready", "blocked"), "invalid journey status")
        require(len(value["steps"]) == (2 if value["status"] == "ready" else 0),
                "a ready journey must contain exactly two steps")
        for action_id in value["steps"]:
            require(action_id in supported and action_id not in completed, "invalid journey action")
            require(set(actions[action_id]["prerequisites"]) <= completed, "unmet prerequisite")
            completed.add(action_id)
        if value["status"] == "blocked":
            initial = set(document["completed"])
            require(not any(second != first and second not in initial
                            and set(actions[second]["prerequisites"]) <= initial | {first}
                            for first in available for second in supported),
                    "blocked journey has a valid two-step path")
        text(value["reason"], "journey reason")
    elif stage == "triage":
        fields(value, "customer_id category priority team owner journey_status action_ids evidence_ids reasons", stage)
        require(isinstance(previous, dict) and set(previous) == {"research", "journey"},
                "triage requires both validated handoffs")
        validate("journey", previous["journey"], document, previous["research"])
        journey = previous["journey"]
        category = value["category"]
        rules = document["routing"]["categories"]
        require(isinstance(category, str) and category in rules, "unknown routed category")
        require(value["team"] == rules[category]["team"] and value["owner"] == rules[category]["owner"],
                "routing must identify the configured accountable team and owner")
        require(value["customer_id"] == document["ticket"]["customer_id"], "customer identity changed")
        require(value["journey_status"] == journey["status"]
                and value["action_ids"] == journey["steps"]
                and value["evidence_ids"] == journey["evidence_ids"], "triage lost journey context")
        require(value["priority"] in PRIORITIES, "invalid triage priority")
        require(value["category"] == choose_category(document, journey), "incorrect category")
        require(value["priority"] == choose_priority(document, journey), "incorrect priority")
        strings(value["reasons"], "triage reasons")
        require(bool(value["reasons"]), "triage requires reasons")
    elif stage == "output":
        fields(value, "schema_version status fixture_label research journey triage", stage)
        require(value["schema_version"] == 1 and value["status"] == "ok"
                and value["fixture_label"] == document["fixture_label"], "invalid output envelope")
        validate("research", value["research"], document)
        validate("journey", value["journey"], document, value["research"])
        validate("triage", value["triage"], document,
                 {"research": value["research"], "journey": value["journey"]})
    else:
        raise ValidationError("unknown validation stage")
    return value


def research(document):
    validate("input", document)
    question_terms = tokens(document["question"])
    evidence = []
    for source in document["sources"]:
        overlap = question_terms & tokens(source["text"])
        if overlap and source["reliability"] > 0:
            evidence.append({"source_id": source["id"], "title": source["title"],
                             "excerpt": source["text"],
                             "relevance": round(len(overlap) / len(question_terms), 6),
                             "reliability": source["reliability"]})
    evidence.sort(key=lambda e: (-e["relevance"] * e["reliability"], e["source_id"]))
    result = {"question": document["question"], "status": "supported" if evidence else "insufficient",
              "evidence": evidence,
              "confidence": round(max((e["relevance"] * e["reliability"] for e in evidence), default=0), 6),
              "limitations": ["Lexical relevance is not fact verification.",
                              "Source reliability is supplied by the caller; contradictions are not resolved."]}
    return validate("research", result, document)


def supported_actions(document, evidence):
    corpus = set().union(*(tokens(e["excerpt"]) for e in evidence["evidence"]))
    return {a["id"]: sum(bool(tokens(term) & corpus) for term in a["evidence_terms"])
            for a in document["actions"]
            if any(tokens(term) & corpus for term in a["evidence_terms"])}


def journey(document, evidence):
    validate("research", evidence, document)
    actions = {a["id"]: a for a in document["actions"]}
    supported = supported_actions(document, evidence)
    ranked = sorted(supported, key=lambda a: (-supported[a], a))
    completed = set(document["completed"])
    available = [a for a in ranked if a not in completed and set(actions[a]["prerequisites"]) <= completed]
    steps = []
    for first in available:
        seconds = [a for a in ranked if a not in completed | {first}
                   and set(actions[a]["prerequisites"]) <= completed | {first}]
        if seconds:
            steps = [first, seconds[0]]
            break
    result = {"status": "ready" if steps else "blocked", "next_actions": available, "steps": steps,
              "evidence_ids": [e["source_id"] for e in evidence["evidence"]],
              "reason": ("Two evidence-supported actions satisfy prerequisites in sequence." if steps else
                         "No evidence-supported two-step path exists; human assistance is required.")}
    return validate("journey", result, document, evidence)


def choose_category(document, plan):
    rules = document["routing"]["categories"]
    actions = {a["id"]: a for a in document["actions"]}
    words = tokens(document["ticket"]["text"])
    scores = {category: sum(bool(tokens(keyword) & words) for keyword in rule["keywords"])
              for category, rule in rules.items()}
    for action_id in plan["steps"]:
        scores[actions[action_id]["category"]] += 1
    return (sorted(scores, key=lambda category: (-scores[category], category))[0]
            if any(scores.values()) else document["routing"]["fallback_category"])


def choose_priority(document, plan):
    base = document["routing"]["priority_by_severity"][document["ticket"]["severity"]]
    return PRIORITIES[max(PRIORITIES.index(base), 2 if plan["status"] == "blocked" else 0)]


def triage(document, evidence, plan):
    validate("journey", plan, document, evidence)
    category = choose_category(document, plan)
    rule = document["routing"]["categories"][category]
    result = {"customer_id": document["ticket"]["customer_id"], "category": category,
              "priority": choose_priority(document, plan), "team": rule["team"], "owner": rule["owner"],
              "journey_status": plan["status"], "action_ids": list(plan["steps"]),
              "evidence_ids": list(plan["evidence_ids"]),
              "reasons": ["Category uses ticket keyword matches plus one vote per journey action; ties use category name.",
                          "Configured severity priority applies; blocked journeys have a high-priority floor."]}
    return validate("triage", result, document, {"research": evidence, "journey": plan})


def run_pipeline(document):
    document = copy.deepcopy(validate("input", document))
    evidence = research(document)
    plan = journey(document, evidence)
    routed = triage(document, evidence, plan)
    return validate("output", {"schema_version": 1, "status": "ok",
                              "fixture_label": document["fixture_label"], "research": evidence,
                              "journey": plan, "triage": routed}, document)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], "rb") as handle:
            raw = handle.read(1_000_001)
        require(len(raw) <= 1_000_000, "input exceeds 1000000 bytes")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        result = run_pipeline(document)
        exit_code = 0
    except (ValueError, OSError, UnicodeError, TypeError, KeyError, RecursionError) as error:
        result = {"status": "error", "error": str(error)}
        exit_code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
