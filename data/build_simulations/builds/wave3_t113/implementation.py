"""Deterministic ticket triage. Usage: python -B implementation.py INPUT.json.

Category keyword matches are case-insensitive substrings of subject and body.
The first matching configured category wins; otherwise fallback_category wins.
Priority is the maximum of the category baseline and all matching priority rules.
Every category, including the fallback, has an accountable owner and team.
"""

import json
import sys


PRIORITIES = ("low", "normal", "high", "urgent")


class ValidationError(ValueError):
    pass


def require(condition, path, message):
    if not condition:
        raise ValidationError(f"{path}: {message}")


def fields(value, expected, path):
    require(isinstance(value, dict), path, "must be an object")
    require(set(value) == set(expected), path,
            f"expected exactly these fields: {', '.join(expected)}")


def text(value, path, allow_empty=False):
    require(isinstance(value, str), path, "must be a string")
    require(allow_empty or bool(value.strip()), path, "must not be blank")
    require(len(value) <= 100000, path, "must be at most 100000 characters")


def sequence(value, path, maximum=10000):
    require(isinstance(value, list), path, "must be an array")
    require(len(value) <= maximum, path, f"must contain at most {maximum} items")


def priority(value, path):
    require(isinstance(value, str) and value in PRIORITIES, path,
            "must be low, normal, high, or urgent")


def keywords(value, path):
    sequence(value, path, 1000)
    seen = set()
    for index, word in enumerate(value):
        location = f"{path}[{index}]"
        text(word, location)
        require(word == word.strip(), location, "must not have surrounding whitespace")
        require(word.casefold() not in seen, location, "duplicate keyword")
        seen.add(word.casefold())


def validate(payload):
    """Single shared validation boundary for CLI and Python callers."""
    fields(payload, ("schema_version", "dataset_label", "config", "tickets"), "$")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "$.schema_version", "must be integer 1")
    text(payload["dataset_label"], "$.dataset_label")
    config = payload["config"]
    fields(config, ("categories", "fallback_category", "priority_rules"), "$.config")
    sequence(config["categories"], "$.config.categories", 1000)
    require(bool(config["categories"]), "$.config.categories", "must not be empty")
    ids = set()
    for index, category in enumerate(config["categories"]):
        path = f"$.config.categories[{index}]"
        fields(category, ("id", "keywords", "base_priority", "team", "owner"), path)
        for name in ("id", "team", "owner"):
            text(category[name], f"{path}.{name}")
        require(category["id"] not in ids, f"{path}.id", "duplicate category ID")
        ids.add(category["id"])
        keywords(category["keywords"], f"{path}.keywords")
        priority(category["base_priority"], f"{path}.base_priority")
    text(config["fallback_category"], "$.config.fallback_category")
    require(config["fallback_category"] in ids, "$.config.fallback_category",
            "must reference a configured category")
    sequence(config["priority_rules"], "$.config.priority_rules", 1000)
    for index, rule in enumerate(config["priority_rules"]):
        path = f"$.config.priority_rules[{index}]"
        fields(rule, ("keywords", "priority"), path)
        keywords(rule["keywords"], f"{path}.keywords")
        require(bool(rule["keywords"]), f"{path}.keywords", "must not be empty")
        priority(rule["priority"], f"{path}.priority")
    sequence(payload["tickets"], "$.tickets")
    ids = set()
    for index, ticket in enumerate(payload["tickets"]):
        path = f"$.tickets[{index}]"
        fields(ticket, ("id", "subject", "body"), path)
        text(ticket["id"], f"{path}.id")
        text(ticket["subject"], f"{path}.subject", allow_empty=True)
        text(ticket["body"], f"{path}.body", allow_empty=True)
        require(ticket["id"] not in ids, f"{path}.id", "duplicate ticket ID")
        ids.add(ticket["id"])
    return payload


def matching(words, content):
    return [word for word in words if word.casefold() in content]


def triage(payload):
    validate(payload)
    config = payload["config"]
    fallback = next(c for c in config["categories"]
                    if c["id"] == config["fallback_category"])
    results = []
    counts = {name: 0 for name in PRIORITIES}
    for ticket in payload["tickets"]:
        content = (ticket["subject"] + "\n" + ticket["body"]).casefold()
        chosen, hits = fallback, []
        for category in config["categories"]:
            category_hits = matching(category["keywords"], content)
            if category_hits:
                chosen, hits = category, category_hits
                break
        level = PRIORITIES.index(chosen["base_priority"])
        evidence = []
        for index, rule in enumerate(config["priority_rules"]):
            rule_hits = matching(rule["keywords"], content)
            if rule_hits:
                level = max(level, PRIORITIES.index(rule["priority"]))
                evidence.append({"rule_index": index, "keywords": rule_hits,
                                 "priority": rule["priority"]})
        final_priority = PRIORITIES[level]
        counts[final_priority] += 1
        results.append({
            "ticket_id": ticket["id"],
            "category": chosen["id"],
            "priority": final_priority,
            "route": {"team": chosen["team"], "owner": chosen["owner"]},
            "reason": {
                "category_selection": "keyword" if hits else "fallback",
                "category_keywords": hits,
                "base_priority": chosen["base_priority"],
                "priority_rules": evidence,
            },
        })
    return {"status": "ok", "schema_version": 1,
            "dataset_label": payload["dataset_label"],
            "results": results,
            "summary": {"total": len(results), "by_priority": counts}}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSON: duplicate object key {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"JSON: non-finite number {value} is not permitted")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "arguments",
                "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as source:
            payload = json.load(source, object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        result = triage(payload)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
