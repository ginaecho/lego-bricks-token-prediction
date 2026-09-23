"""Synthetic, deterministic document-review-to-ticket pipeline; no certification."""

import copy
import json
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(names.split()), path + " has missing or unknown fields")


def text(value, path):
    require(type(value) is str and bool(value.strip()), path + " must be nonempty text")


def array(value, path):
    require(type(value) is list, path + " must be an array")


def priority(value, path):
    require(type(value) is int and 1 <= value <= 4, path + " must be integer 1..4")


def strings(value, path):
    array(value, path)
    require(bool(value), path + " must not be empty")
    for item in value:
        text(item, path)
    require(len(set(s.casefold() for s in value)) == len(value),
            path + " must not contain duplicates")


def exact_match(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys() and
                all(exact_match(actual[key], value) for key, value in expected.items()))
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            exact_match(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate(envelope, stage="input"):
    """Validate the same envelope at each boundary, including derived traceability."""
    require(stage in ("input", "review", "complete"), "unknown validation stage")
    fields(envelope, "schema_version synthetic input review triage status", "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "schema_version must be 1")
    require(envelope["synthetic"] is True, "fixtures must be labeled synthetic")
    require(envelope["status"] == stage, "status does not match validation stage")
    data = envelope["input"]
    fields(data, "document requirements evidence routing", "input")
    doc = data["document"]
    fields(doc, "id text", "document")
    text(doc["id"], "document.id")
    text(doc["text"], "document.text")
    array(data["requirements"], "requirements")
    requirement_ids = set()
    for req in data["requirements"]:
        fields(req, "id description terms severity", "requirement")
        text(req["id"], "requirement.id")
        require(req["id"] not in requirement_ids, "duplicate requirement id")
        requirement_ids.add(req["id"])
        text(req["description"], "requirement.description")
        strings(req["terms"], "requirement.terms")
        require(type(req["severity"]) is str and
                req["severity"] in ("low", "medium", "high", "critical"),
                "invalid requirement severity")
    array(data["evidence"], "evidence")
    evidence_ids = set()
    for ev in data["evidence"]:
        fields(ev, "id requirement_id quote", "evidence")
        for key in ("id", "requirement_id", "quote"):
            text(ev[key], "evidence." + key)
        require(ev["id"] not in evidence_ids, "duplicate evidence id")
        evidence_ids.add(ev["id"])
        require(ev["requirement_id"] in requirement_ids, "unknown evidence requirement_id")
    routing = data["routing"]
    fields(routing, "severity_priority categories rules default_category", "routing")
    fields(routing["severity_priority"], "low medium high critical", "severity_priority")
    for value in routing["severity_priority"].values():
        priority(value, "severity priority")
    categories = routing["categories"]
    require(type(categories) is dict and bool(categories), "categories must be a nonempty object")
    for name, route in categories.items():
        text(name, "category name")
        fields(route, "owner team priority_floor", "route")
        text(route["owner"], "route.owner")
        text(route["team"], "route.team")
        priority(route["priority_floor"], "route.priority_floor")
    text(routing["default_category"], "default_category")
    require(routing["default_category"] in categories, "unknown default category")
    array(routing["rules"], "routing.rules")
    for rule in routing["rules"]:
        fields(rule, "keywords category", "rule")
        strings(rule["keywords"], "rule.keywords")
        text(rule["category"], "rule.category")
        require(rule["category"] in categories, "rule uses unknown category")
    if stage == "input":
        require(envelope["review"] is None and envelope["triage"] is None,
                "input cannot contain derived output")
    else:
        # Recompute the bounded deterministic contract to reject forged or stale handoffs.
        require(exact_match(envelope["review"], _review(data)), "invalid review handoff")
        if stage == "review":
            require(envelope["triage"] is None, "review cannot contain triage output")
        else:
            require(exact_match(envelope["triage"], _triage(data, envelope["review"])),
                    "invalid triage handoff")
    return envelope


def _review(data):
    checks, gaps = [], []
    for index, req in enumerate(data["requirements"], 1):
        evidence = [ev for ev in data["evidence"] if ev["requirement_id"] == req["id"]]
        accepted = [ev for ev in evidence if ev["quote"] in data["document"]["text"]]
        rejected = [ev["id"] for ev in evidence if ev not in accepted]
        missing = [term for term in req["terms"]
                   if not any(term.casefold() in ev["quote"].casefold() for ev in accepted)]
        state = ("satisfied" if not missing else
                 "missing_evidence" if not evidence else
                 "invalid_evidence" if not accepted else "insufficient_evidence")
        check = {
            "requirement_id": req["id"], "outcome": state,
            "accepted_evidence_ids": [ev["id"] for ev in accepted],
            "rejected_evidence_ids": rejected, "missing_terms": missing,
        }
        checks.append(check)
        if state != "satisfied":
            gaps.append({
                "id": "gap-" + str(index), "document_id": data["document"]["id"],
                "requirement_id": req["id"], "description": req["description"],
                "severity": req["severity"], "reason": state, "missing_terms": missing,
                "evidence_ids": [ev["id"] for ev in evidence],
            })
    return {
        "notice": "Deterministic evidence screening only; not certification or compliance assurance.",
        "checks": checks, "gaps": gaps,
    }


def _triage(data, review):
    config = data["routing"]
    tickets = []
    for gap in review["gaps"]:
        search = (gap["description"] + " " + " ".join(gap["missing_terms"])).casefold()
        category = config["default_category"]
        match = None
        for index, rule in enumerate(config["rules"]):
            if any(keyword.casefold() in search for keyword in rule["keywords"]):
                category, match = rule["category"], index
                break
        route = config["categories"][category]
        tickets.append({
            "id": "ticket-" + gap["id"], "gap_id": gap["id"],
            "document_id": gap["document_id"], "requirement_id": gap["requirement_id"],
            "evidence_ids": list(gap["evidence_ids"]), "reason": gap["reason"],
            "missing_terms": list(gap["missing_terms"]), "category": category,
            "priority": min(config["severity_priority"][gap["severity"]], route["priority_floor"]),
            "owner": route["owner"], "team": route["team"], "matched_rule_index": match,
            "state": "open",
        })
    return {"tickets": tickets}


def review_stage(envelope):
    validate(envelope, "input")
    result = copy.deepcopy(envelope)
    result["review"] = _review(result["input"])
    result["status"] = "review"
    return validate(result, "review")


def triage_stage(envelope):
    validate(envelope, "review")
    result = copy.deepcopy(envelope)
    result["triage"] = _triage(result["input"], result["review"])
    result["status"] = "complete"
    return validate(result, "complete")


def run(envelope):
    return triage_stage(review_stage(envelope))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonstandard JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=unique_object,
                              parse_constant=reject_constant)
        result = run(value)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
