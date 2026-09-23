"""Synthetic reference: deterministic, first-match customer ticket triage."""

import json
import sys


PRIORITIES = ("low", "normal", "high", "urgent")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), path="value"):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(required) <= value.keys(), f"{path} missing required fields")
    require(value.keys() <= set(required) | set(optional), f"{path} has unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonblank text")


def strings(value, path, allow_empty=False):
    require(isinstance(value, list), f"{path} must be an array")
    require(allow_empty or bool(value), f"{path} must not be empty")
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), f"{path} must contain unique values")


def validate(value, kind="input", config=None):
    """Shared boundary validation for input and all routing decisions."""
    if kind == "decision":
        fields(value, ("category", "priority", "team"), path="decision")
        for key in ("category", "priority", "team"):
            text(value[key], f"decision.{key}")
        require(value["category"] in config["categories"], "Unknown category")
        require(value["priority"] in PRIORITIES, "Unknown priority")
        require(value["team"] in config["teams"], "Unknown team")
        return
    require(kind == "input", "Unknown validation kind")
    fields(value, ("schema_version", "config", "tickets"), path="input")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "schema_version must be integer 1")
    cfg = value["config"]
    fields(cfg, ("categories", "teams", "rules", "default"), path="config")
    strings(cfg["categories"], "categories")
    require(isinstance(cfg["teams"], dict) and bool(cfg["teams"]), "teams must be a nonempty object")
    for team, owner in cfg["teams"].items():
        text(team, "team name")
        text(owner, "accountable owner")
    validate(cfg["default"], "decision", cfg)
    require(isinstance(cfg["rules"], list), "rules must be an array")
    seen = set()
    for rule in cfg["rules"]:
        fields(rule, ("id", "keywords", "decision"), ("channels",), "rule")
        text(rule["id"], "rule.id")
        require(rule["id"] not in seen, "Duplicate rule id")
        seen.add(rule["id"])
        strings(rule["keywords"], "rule.keywords")
        if "channels" in rule:
            strings(rule["channels"], "rule.channels")
        validate(rule["decision"], "decision", cfg)
    require(isinstance(value["tickets"], list), "tickets must be an array")
    seen = set()
    for ticket in value["tickets"]:
        fields(ticket, ("id", "subject", "body", "channel"), path="ticket")
        for key in ("id", "subject", "channel"):
            text(ticket[key], f"ticket.{key}")
        require(isinstance(ticket["body"], str), "ticket.body must be text")
        require(ticket["id"] not in seen, "Duplicate ticket id")
        seen.add(ticket["id"])


def triage(payload, classifier=None):
    """Classifiers are optional local callables, only used for unmatched tickets.

    The callable receives a ticket copy and must return exactly category, priority,
    and team. No provider is imported or contacted.
    """
    validate(payload)
    require(classifier is None or callable(classifier), "classifier must be callable")
    cfg = payload["config"]
    assignments = []
    for ticket in payload["tickets"]:
        content = (ticket["subject"] + "\n" + ticket["body"]).casefold()
        decision = dict(cfg["default"])
        source = "default"
        rule_id = None
        matched_keywords = []
        for rule in cfg["rules"]:
            if "channels" in rule and ticket["channel"] not in rule["channels"]:
                continue
            hits = [word for word in rule["keywords"] if word.casefold() in content]
            if hits:
                decision = dict(rule["decision"])
                source, rule_id, matched_keywords = "rule", rule["id"], hits
                break
        if source == "default" and classifier is not None:
            try:
                decision = classifier(dict(ticket))
            except Exception as exc:
                raise ValidationError("Injected classifier failed") from exc
            source = "injected_callable"
        validate(decision, "decision", cfg)
        assignments.append({
            "ticket_id": ticket["id"],
            **decision,
            "accountable_owner": cfg["teams"][decision["team"]],
            "source": source,
            "rule_id": rule_id,
            "matched_keywords": matched_keywords,
        })
    return {
        "schema_version": 1,
        "status": "ok",
        "assignments": assignments,
        "summary": {
            "total": len(assignments),
            "by_priority": {
                priority: sum(a["priority"] == priority for a in assignments)
                for priority in PRIORITIES
            },
            "by_team": {
                team: sum(a["team"] == team for a in assignments)
                for team in cfg["teams"]
            },
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"Nonstandard JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        result = triage(payload)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
