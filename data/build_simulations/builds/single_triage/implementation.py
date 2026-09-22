"""Deterministic, stdlib-only ticket triage. See build_manifest.json for contract."""

import argparse
import json
import math
import sys


class ValidationError(ValueError):
    """Invalid input configuration, ticket, or classifier result."""


SEVERITIES = ("low", "medium", "high", "critical")
URGENCIES = ("low", "medium", "high")
PRIORITIES = ("P0", "P1", "P2", "P3")


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def fields(value, allowed, required, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) <= set(allowed), label + " contains unknown fields")
    require(set(required) <= set(value), label + " is missing required fields")


def choices(value, allowed, label):
    require(isinstance(value, list) and bool(value), label + " must be a nonempty array")
    for item in value:
        text(item, label + " item")
        if allowed is not None:
            require(item in allowed, label + " contains an unknown value: " + item)
    require(len(value) == len(set(value)), label + " contains duplicate values")


def validate(document):
    fields(document, ("config", "tickets"), ("config", "tickets"), "input")
    config = document["config"]
    fields(config, ("categories", "owners", "rules"),
           ("categories", "owners", "rules"), "config")
    owners = config["owners"]
    choices(owners, None, "owners")
    categories = config["categories"]
    require(isinstance(categories, dict) and bool(categories),
            "categories must be a nonempty category-to-owner object")
    for category, owner in categories.items():
        text(category, "category")
        require(owner is None or (isinstance(owner, str) and owner in owners),
                "category owner must be configured or null")
    rules = config["rules"]
    require(isinstance(rules, list), "rules must be an array")
    seen = set()
    for rule in rules:
        fields(rule, ("id", "category", "owner", "keywords", "severity", "urgency", "priority"),
               ("id", "category"), "rule")
        rule_id = text(rule["id"], "rule id")
        require(rule_id not in seen, "duplicate rule id: " + rule_id)
        seen.add(rule_id)
        category = text(rule["category"], "rule category")
        require(category in categories, "rule category is not configured")
        if "owner" in rule:
            require(isinstance(rule["owner"], str) and rule["owner"] in owners,
                    "rule owner is not configured")
        require(any(key in rule for key in ("keywords", "severity", "urgency")),
                "rule must specify at least one matching condition")
        for key, allowed in (("keywords", None), ("severity", SEVERITIES), ("urgency", URGENCIES)):
            if key in rule:
                choices(rule[key], allowed, "rule " + key)
        if "priority" in rule:
            require(isinstance(rule["priority"], str) and rule["priority"] in PRIORITIES,
                    "rule priority must be P0, P1, P2, or P3")
    tickets = document["tickets"]
    require(isinstance(tickets, list), "tickets must be an array")
    seen = set()
    for ticket in tickets:
        fields(ticket, ("id", "subject", "body", "severity", "urgency"),
               ("id", "subject"), "ticket")
        ticket_id = text(ticket["id"], "ticket id")
        require(ticket_id not in seen, "duplicate ticket id: " + ticket_id)
        seen.add(ticket_id)
        require(isinstance(ticket["subject"], str), "ticket subject must be text")
        require(isinstance(ticket.get("body", ""), str), "ticket body must be text")
        require(isinstance(ticket.get("severity", "medium"), str)
                and ticket.get("severity", "medium") in SEVERITIES, "invalid ticket severity")
        require(isinstance(ticket.get("urgency", "medium"), str)
                and ticket.get("urgency", "medium") in URGENCIES, "invalid ticket urgency")
    return config, tickets


def base_priority(severity, urgency):
    if severity == "critical" or (severity == "high" and urgency == "high"):
        return "P0"
    if severity == "high" or urgency == "high":
        return "P1"
    if severity == "medium" or urgency == "medium":
        return "P2"
    return "P3"


def classify(ticket, classifier, config):
    try:
        result = classifier(dict(ticket))
    except Exception as exc:
        raise ValidationError("classifier failed for ticket " + ticket["id"]) from exc
    fields(result, ("category", "owner", "confidence"), ("category", "confidence"),
           "classifier result")
    category = result["category"]
    require(isinstance(category, str) and category in config["categories"],
            "classifier category is not configured")
    confidence = result["confidence"]
    require(type(confidence) in (int, float), "classifier confidence must be numeric")
    # Checking the bounded range first also safely rejects arbitrarily large integers.
    require(0 <= confidence <= 1 and math.isfinite(confidence),
            "classifier confidence must be finite and between 0 and 1")
    owner = result.get("owner", config["categories"][category])
    if "owner" in result:
        require(isinstance(owner, str) and owner in config["owners"],
                "classifier owner is not configured")
    return category, owner, confidence


def triage(document, classifier=None):
    """Validate the entire batch, then return results in ticket input order.

    classifier(ticket_copy) is optional and runs only when no rule matched.
    Its category and confidence are required; owner is optional.
    """
    config, tickets = validate(document)
    require(classifier is None or callable(classifier), "classifier must be callable")
    results = []
    for original in tickets:
        ticket = dict(original)
        ticket.setdefault("body", "")
        ticket.setdefault("severity", "medium")
        ticket.setdefault("urgency", "medium")
        severity, urgency = ticket["severity"], ticket["urgency"]
        priority = base_priority(severity, urgency)
        baseline = priority
        haystack = (ticket["subject"] + "\n" + ticket["body"]).casefold()
        matched = []
        for rule in config["rules"]:
            keywords = [word for word in rule.get("keywords", []) if word.casefold() in haystack]
            if "keywords" in rule and not keywords:
                continue
            if "severity" in rule and severity not in rule["severity"]:
                continue
            if "urgency" in rule and urgency not in rule["urgency"]:
                continue
            owner = rule.get("owner", config["categories"][rule["category"]])
            conditions = {}
            if "keywords" in rule:
                conditions["keywords"] = keywords
            if "severity" in rule:
                conditions["severity"] = severity
            if "urgency" in rule:
                conditions["urgency"] = urgency
            matched.append({"rule_id": rule["id"], "category": rule["category"],
                            "owner": owner, "conditions": conditions,
                            "priority": rule.get("priority")})
            if "priority" in rule:
                priority = min(priority, rule["priority"], key=PRIORITIES.index)
        candidate_categories = [
            category for category in config["categories"]
            if any(match["category"] == category for match in matched)
        ]
        candidate_owners = [
            owner for owner in config["owners"]
            if any(match["owner"] == owner for match in matched)
        ]
        category = owner = confidence = None
        source = "rules" if matched else "none"
        if len(candidate_categories) > 1:
            status, reason = "ambiguous", "multiple_categories"
        elif matched:
            category = candidate_categories[0]
            distinct_owners = {match["owner"] for match in matched}
            if len(distinct_owners) > 1:
                status, reason = "ambiguous", "conflicting_owners"
            else:
                owner = matched[0]["owner"]
                status = "assigned" if owner is not None else "unassigned"
                reason = "matched_rules" if owner is not None else "category_has_no_owner"
        elif classifier is not None:
            category, owner, confidence = classify(ticket, classifier, config)
            candidate_categories = [category]
            candidate_owners = [] if owner is None else [owner]
            source = "classifier"
            status = "assigned" if owner is not None else "unassigned"
            reason = "classifier_result" if owner is not None else "category_has_no_owner"
        else:
            status, reason = "unassigned", "no_matching_rule"
        results.append({
            "id": ticket["id"], "category": category, "priority": priority, "owner": owner,
            "status": status, "reason": reason, "source": source, "confidence": confidence,
            "candidate_categories": candidate_categories, "candidate_owners": candidate_owners,
            "matched_rules": matched,
            "priority_explanation": {"severity": severity, "urgency": urgency,
                                     "baseline": baseline, "final": priority},
        })
    return {"results": results}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="UTF-8 JSON path, or - for stdin")
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            document = json.load(sys.stdin, object_pairs_hook=unique_object,
                                 parse_constant=reject_constant)
        else:
            with open(args.input, encoding="utf-8") as stream:
                document = json.load(stream, object_pairs_hook=unique_object,
                                     parse_constant=reject_constant)
        output = triage(document)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False, indent=2))
        return 0
    except (ValidationError, json.JSONDecodeError, OSError, UnicodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
