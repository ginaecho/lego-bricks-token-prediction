"""Synthetic, offline reference pipeline. No certification or live web access."""

import copy
import ipaddress
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class ValidationError(ValueError):
    pass


STAGES = ("review", "web", "feedback", "journey")
MAX_INPUT_BYTES = 1_000_000


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, context):
    require(type(value) is dict, context + " must be an object")
    require(set(value) == set(names), context + " has missing or unknown fields")


def text(value, context):
    require(type(value) is str and bool(value.strip()), context + " must be nonempty text")
    require(len(value) <= 20000, context + " exceeds text limit")


def array(value, context, nonempty=False):
    require(type(value) is list and len(value) <= 200, context + " must be a bounded array")
    require(not nonempty or bool(value), context + " must not be empty")


def strings(value, context, nonempty=False):
    array(value, context, nonempty)
    for item in value:
        text(item, context)
    require(len(value) == len(set(value)), context + " must not contain duplicates")


def records(value, names, context, nonempty=False):
    array(value, context, nonempty)
    ids = set()
    for record in value:
        fields(record, names, context)
        text(record["id"], context + ".id")
        require(record["id"] not in ids, context + " has duplicate id")
        ids.add(record["id"])
    return ids


def normalized(value):
    return " ".join(value.casefold().split())


def host_name(value):
    text(value, "allowlisted host")
    require(value == value.lower() and len(value) <= 253, "Host must be lowercase DNS")
    require("." in value and not value.endswith("."), "Host must be a qualified DNS name")
    require(all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part)
                for part in value.split(".")), "Invalid DNS host")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return
    raise ValidationError("IP addresses are not allowlisted hosts")


def canonical_url(value, hosts):
    text(value, "source.url")
    require(not any(c.isspace() or ord(c) < 32 for c in value) and "\\" not in value,
            "URL contains forbidden characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(parsed.scheme == "https" and parsed.hostname in hosts,
            "URL must use HTTPS and an exactly allowlisted host")
    require(parsed.username is None and parsed.password is None and port in (None, 443),
            "URL credentials or nonstandard ports forbidden")
    require(not parsed.fragment, "URL fragments forbidden")
    return urlunsplit(("https", parsed.hostname, parsed.path or "/", parsed.query, ""))


def validate_input(value):
    fields(value, ("schema_version", "synthetic", "requirements", "documents",
                   "allowlisted_hosts", "sources", "feedback", "actions",
                   "completed_actions"), "input")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "Unsupported schema_version")
    require(value["synthetic"] is True, "Only clearly labeled synthetic fixtures are supported")
    req_ids = records(value["requirements"], ("id", "description", "terms"), "requirements", True)
    for requirement in value["requirements"]:
        text(requirement["description"], "requirement.description")
        strings(requirement["terms"], "requirement.terms", True)
    records(value["documents"], ("id", "text"), "documents")
    for document in value["documents"]:
        text(document["text"], "document.text")
    strings(value["allowlisted_hosts"], "allowlisted_hosts", True)
    for host in value["allowlisted_hosts"]:
        host_name(host)
    records(value["sources"], ("id", "url", "title", "retrieved_at", "body",
                              "requirement_ids"), "sources")
    urls = set()
    for source in value["sources"]:
        url = canonical_url(source["url"], value["allowlisted_hosts"])
        require(url not in urls, "Duplicate canonical source URL")
        urls.add(url)
        for key in ("title", "body", "retrieved_at"):
            text(source[key], "source." + key)
        from datetime import datetime
        try:
            stamp = datetime.fromisoformat(source["retrieved_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("Invalid retrieved_at timestamp") from exc
        require(stamp.utcoffset() is not None, "retrieved_at requires a timezone")
        strings(source["requirement_ids"], "source.requirement_ids", True)
        require(set(source["requirement_ids"]) <= req_ids, "Unknown source requirement")
    records(value["feedback"], ("id", "text", "requirement_ids"), "feedback")
    for feedback in value["feedback"]:
        text(feedback["text"], "feedback.text")
        strings(feedback["requirement_ids"], "feedback.requirement_ids", True)
        require(set(feedback["requirement_ids"]) <= req_ids, "Unknown feedback requirement")
    action_ids = records(value["actions"], ("id", "title", "requirement_ids", "prerequisites"),
                         "actions", True)
    for action in value["actions"]:
        text(action["title"], "action.title")
        strings(action["requirement_ids"], "action.requirement_ids", True)
        strings(action["prerequisites"], "action.prerequisites")
        require(set(action["requirement_ids"]) <= req_ids, "Unknown action requirement")
        require(set(action["prerequisites"]) <= action_ids, "Unknown action prerequisite")
    strings(value["completed_actions"], "completed_actions")
    require(set(value["completed_actions"]) <= action_ids, "Unknown completed action")
    actions = {action["id"]: action for action in value["actions"]}
    visiting, visited = set(), set()

    def visit(action_id):
        require(action_id not in visiting, "Cyclic action prerequisites")
        if action_id in visited:
            return
        visiting.add(action_id)
        for prerequisite in actions[action_id]["prerequisites"]:
            visit(prerequisite)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in actions:
        visit(action_id)
    completed = set(value["completed_actions"])
    require(all(set(actions[action_id]["prerequisites"]) <= completed for action_id in completed),
            "Completed actions must include their prerequisites")
    return value


def matching_excerpt(body, terms):
    """Return exact source offsets for a literal case-insensitive term match."""
    matches = []
    for term in terms:
        match = re.search(re.escape(term), body, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), match.end()))
    if not matches:
        return None
    start, end = min(matches)
    start, end = max(0, start - 45), min(len(body), end + 75)
    return {"start": start, "end": end, "text": body[start:end]}


def review_data(config):
    checks = []
    for requirement in config["requirements"]:
        evidence = []
        for document in config["documents"]:
            excerpt = matching_excerpt(document["text"], requirement["terms"])
            if excerpt:
                evidence.append({"document_id": document["id"], "excerpt": excerpt})
        checks.append({"requirement_id": requirement["id"],
                       "status": "evidence_found" if evidence else "gap",
                       "evidence": evidence})
    return {"checks": checks,
            "gap_ids": [check["requirement_id"] for check in checks if check["status"] == "gap"],
            "notice": "Keyword evidence screening only; not certification or a compliance conclusion."}


def web_data(config, review):
    requirements = {item["id"]: item for item in config["requirements"]}
    findings = []
    for requirement_id in review["gap_ids"]:
        for source in config["sources"]:
            if requirement_id not in source["requirement_ids"]:
                continue
            excerpt = matching_excerpt(source["body"], requirements[requirement_id]["terms"])
            if excerpt:
                findings.append({
                    "id": "finding-" + str(len(findings) + 1),
                    "requirement_id": requirement_id,
                    "source_id": source["id"],
                    "url": canonical_url(source["url"], config["allowlisted_hosts"]),
                    "title": source["title"],
                    "retrieved_at": source["retrieved_at"],
                    "excerpt": excerpt,
                })
    found = {item["requirement_id"] for item in findings}
    return {"retrieval_mode": "offline_synthetic_url_snapshot",
            "requested_requirement_ids": list(review["gap_ids"]),
            "findings": findings,
            "unresolved_gap_ids": [item for item in review["gap_ids"] if item not in found],
            "notice": "External findings do not close document evidence gaps."}


def feedback_data(config, web):
    groups = {}
    for item in config["feedback"]:
        key = normalized(item["text"])
        if key not in groups:
            groups[key] = {"id": "feedback-group-" + str(len(groups) + 1),
                           "original_ids": [], "requirement_ids": [], "excerpts": []}
        group = groups[key]
        group["original_ids"].append(item["id"])
        group["requirement_ids"] = sorted(set(group["requirement_ids"]) | set(item["requirement_ids"]))
        group["excerpts"].append({"feedback_id": item["id"], "text": item["text"]})
    themes = []
    for requirement_id in web["requested_requirement_ids"]:
        support = [group for group in groups.values() if requirement_id in group["requirement_ids"]]
        findings = [item["id"] for item in web["findings"] if item["requirement_id"] == requirement_id]
        if support:
            themes.append({"requirement_id": requirement_id, "unique_feedback_count": len(support),
                           "supporting_group_ids": [item["id"] for item in support],
                           "finding_ids": findings,
                           "research_status": "supported" if findings else "unresolved"})
    themed = {group_id for theme in themes for group_id in theme["supporting_group_ids"]}
    return {"groups": list(groups.values()), "themes": themes,
            "duplicate_count": len(config["feedback"]) - len(groups),
            "unthemed_group_ids": [group["id"] for group in groups.values() if group["id"] not in themed]}


def journey_data(config, feedback):
    themes = {theme["requirement_id"]: theme for theme in feedback["themes"]}
    completed = set(config["completed_actions"])
    remaining = [action for action in config["actions"] if action["id"] not in completed]

    def score(action):
        return sum(themes[item]["unique_feedback_count"]
                   for item in action["requirement_ids"] if item in themes)

    pairs = []
    for first in remaining:
        if not set(first["prerequisites"]) <= completed:
            continue
        for second in remaining:
            if second["id"] == first["id"]:
                continue
            if set(second["prerequisites"]) <= completed | {first["id"]}:
                pairs.append((first, second))
    require(bool(pairs), "Cannot construct two distinct prerequisite-valid next actions")
    first, second = min(pairs, key=lambda pair: (-(score(pair[0]) + score(pair[1])),
                                               -score(pair[0]), pair[0]["id"], pair[1]["id"]))
    steps = []
    available = set(completed)
    for action in (first, second):
        theme_ids = sorted(set(action["requirement_ids"]) & set(themes))
        steps.append({"action_id": action["id"], "title": action["title"],
                      "prerequisites": list(action["prerequisites"]),
                      "satisfied_by": sorted(set(action["prerequisites"]) & available),
                      "theme_requirement_ids": theme_ids,
                      "finding_ids": sorted({finding for item in theme_ids
                                             for finding in themes[item]["finding_ids"]}),
                      "supporting_group_ids": sorted({group for item in theme_ids
                                                      for group in themes[item]["supporting_group_ids"]}),
                      "score": score(action),
                      "rationale": ("Prioritized by deduplicated feedback."
                                    if theme_ids else "Prerequisite-feasible fallback; no feedback support.")})
        available.add(action["id"])
    return {"steps": steps, "validated": True,
            "ranking": "Maximum total unique-feedback support, then first-step support, then action IDs."}


def validate_envelope(envelope, expected_stage, config):
    """Validate every handoff against the input and the preceding validated payload.

    Exact deterministic recomputation checks references, counts, offsets, provenance
    and journey ordering rather than trusting stage-supplied validity flags.
    """
    validate_input(config)
    require(expected_stage in STAGES, "Unknown stage")
    fields(envelope, ("schema_version", "synthetic", "stage", "status", "data"), "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "Invalid envelope schema_version")
    require(envelope["synthetic"] is True and envelope["status"] == "ok", "Invalid envelope status")
    require(envelope["stage"] == expected_stage, "Wrong handoff stage")
    prefix = STAGES[:STAGES.index(expected_stage) + 1]
    fields(envelope["data"], prefix, "envelope.data")
    expected = {}
    expected["review"] = review_data(config)
    if "web" in prefix:
        expected["web"] = web_data(config, expected["review"])
    if "feedback" in prefix:
        expected["feedback"] = feedback_data(config, expected["web"])
    if "journey" in prefix:
        expected["journey"] = journey_data(config, expected["feedback"])
    # JSON comparison also distinguishes booleans from integers.
    require(json.dumps(envelope["data"], sort_keys=True, ensure_ascii=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, ensure_ascii=True, allow_nan=False),
            "Handoff data failed provenance or deterministic schema validation")
    return envelope


def advance(config, stage, previous=None):
    validate_input(config)
    require(stage in STAGES, "Unknown stage")
    position = STAGES.index(stage)
    if position:
        validate_envelope(previous, STAGES[position - 1], config)
        data = copy.deepcopy(previous["data"])
    else:
        require(previous is None, "Review cannot consume an earlier stage")
        data = {}
    if stage == "review":
        data[stage] = review_data(config)
    elif stage == "web":
        data[stage] = web_data(config, data["review"])
    elif stage == "feedback":
        data[stage] = feedback_data(config, data["web"])
    else:
        data[stage] = journey_data(config, data["feedback"])
    result = {"schema_version": 1, "synthetic": True, "stage": stage, "status": "ok", "data": data}
    return validate_envelope(result, stage, config)


def run_pipeline(config):
    validate_input(config)
    current = None
    for stage in STAGES:
        current = advance(config, stage, current)
    return current


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with Path(argv[0]).open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        require(len(raw) <= MAX_INPUT_BYTES, "Input exceeds byte limit")
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
        output = run_pipeline(config)
    except (ValidationError, OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)},
                         ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
