"""Synthetic, offline triage -> adaptive onboarding -> research reference CLI.

Run: python -B implementation.py example_input.json
No URL is fetched: source bodies are explicitly supplied synthetic snapshots.
run_pipeline(payload, classifier=None) optionally accepts a local classifier
callable(ticket_copy, category_names) returning exactly category and priority.
"""

import copy
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class ValidationError(ValueError):
    pass


PRIORITIES = ("low", "normal", "high", "urgent")
EXPERIENCE = ("beginner", "intermediate", "advanced")
FORMATS = ("text", "video", "interactive")
STAGES = ("input", "triage", "adaptive", "web")


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, fields, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(fields), path + " has missing or unknown fields")


def text(value, path, limit=10000):
    require(type(value) is str and bool(value.strip()) and len(value) <= limit,
            path + " must be a nonempty bounded string")
    return value


def sequence(value, path, nonempty=False, limit=100):
    require(type(value) is list and len(value) <= limit, path + " must be a bounded array")
    require(not nonempty or bool(value), path + " must not be empty")
    return value


def strings(value, path, nonempty=False):
    sequence(value, path, nonempty)
    for item in value:
        text(item, path, 200)
    require(len(set(value)) == len(value), path + " must not contain duplicates")
    return value


def integer(value, path, minimum, maximum):
    require(type(value) is int and minimum <= value <= maximum,
            path + " must be an integer in range")


def choice(value, options, path):
    require(type(value) is str and value in options, path + " contains an invalid choice")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def canonical_url(value, hosts):
    text(value, "source.url", 2048)
    require(not any(c.isspace() or ord(c) < 32 for c in value) and "\\" not in value,
            "URL contains forbidden characters")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ValidationError("invalid URL") from exc
    require(parts.scheme == "https" and parts.hostname in hosts,
            "URL must use HTTPS and an exact allowlisted host")
    require(parts.username is None and parts.password is None and port in (None, 443),
            "URL credentials and non-HTTPS ports are forbidden")
    require(not parts.fragment, "URL fragments are forbidden")
    return urlunsplit(("https", parts.hostname, parts.path or "/", parts.query, ""))


def validate_input(data):
    obj(data, ("schema_version", "synthetic", "ticket", "profile", "config", "sources"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "all inputs must be explicitly synthetic")
    ticket = data["ticket"]
    obj(ticket, ("id", "subject", "body", "severity"), "ticket")
    for field in ("id", "subject", "body"):
        text(ticket[field], "ticket." + field)
    choice(ticket["severity"], PRIORITIES, "ticket.severity")
    profile = data["profile"]
    obj(profile, ("experience", "preferred_format", "completed_steps"), "profile")
    choice(profile["experience"], EXPERIENCE, "profile.experience")
    choice(profile["preferred_format"], FORMATS, "profile.preferred_format")
    strings(profile["completed_steps"], "profile.completed_steps")
    config = data["config"]
    obj(config, ("categories", "fallback_category", "priority_rules", "steps",
                 "allowlisted_hosts", "max_findings"), "config")
    sequence(config["categories"], "categories", True)
    names = []
    for category in config["categories"]:
        obj(category, ("name", "keywords", "team", "owner", "default_priority",
                       "onboarding_steps", "research_terms"), "category")
        for field in ("name", "team", "owner"):
            text(category[field], "category." + field, 200)
        for field in ("keywords", "onboarding_steps", "research_terms"):
            strings(category[field], "category." + field)
        choice(category["default_priority"], PRIORITIES, "category.default_priority")
        names.append(category["name"])
    require(len(set(names)) == len(names), "duplicate category name")
    choice(config["fallback_category"], names, "fallback_category")
    sequence(config["priority_rules"], "priority_rules")
    for rule in config["priority_rules"]:
        obj(rule, ("keywords", "priority"), "priority_rule")
        strings(rule["keywords"], "priority_rule.keywords", True)
        choice(rule["priority"], PRIORITIES, "priority_rule.priority")
    sequence(config["steps"], "steps")
    steps = {}
    for step in config["steps"]:
        obj(step, ("id", "title", "prerequisites", "min_experience", "formats",
                   "research_terms"), "step")
        text(step["id"], "step.id", 200)
        text(step["title"], "step.title", 200)
        require(step["id"] not in steps, "duplicate step id")
        strings(step["prerequisites"], "step.prerequisites")
        strings(step["formats"], "step.formats", True)
        strings(step["research_terms"], "step.research_terms")
        for fmt in step["formats"]:
            choice(fmt, FORMATS, "step.formats")
        choice(step["min_experience"], EXPERIENCE, "step.min_experience")
        steps[step["id"]] = step
    for step in steps.values():
        require(all(p in steps for p in step["prerequisites"]), "unknown prerequisite")
    visiting, visited = set(), set()

    def visit(step_id):
        require(step_id not in visiting, "cyclic prerequisites")
        if step_id in visited:
            return
        visiting.add(step_id)
        for prerequisite in steps[step_id]["prerequisites"]:
            visit(prerequisite)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in steps:
        visit(step_id)
    require(all(s in steps for s in profile["completed_steps"]), "unknown completed step")
    for category in config["categories"]:
        require(all(s in steps for s in category["onboarding_steps"]), "unknown onboarding step")
    hosts = strings(config["allowlisted_hosts"], "allowlisted_hosts", True)
    for host in hosts:
        require(host == host.lower() and len(host) <= 253 and "." in host
                and all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                        for label in host.split(".")), "invalid allowlisted host")
    integer(config["max_findings"], "max_findings", 1, 20)
    sequence(data["sources"], "sources")
    ids, urls = set(), set()
    for source in data["sources"]:
        obj(source, ("id", "url", "title", "content", "synthetic"), "source")
        for field in ("id", "title", "content"):
            text(source[field], "source." + field)
        require(source["synthetic"] is True, "source must be synthetic")
        url = canonical_url(source["url"], hosts)
        require(source["id"] not in ids and url not in urls, "duplicate source id or URL")
        ids.add(source["id"])
        urls.add(url)
    return data


def match_keywords(keywords, body):
    body_tokens = tokens(body)
    return [keyword for keyword in keywords if tokens(keyword)
            and tokens(keyword) <= body_tokens]


def category_for(data, name):
    return next(c for c in data["config"]["categories"] if c["name"] == name)


def triage_result(data, classifier=None):
    ticket, config = data["ticket"], data["config"]
    body = ticket["subject"] + " " + ticket["body"]
    scored = [(len(match_keywords(c["keywords"], body)), c) for c in config["categories"]]
    score, category = max(scored, key=lambda pair: pair[0])
    if score == 0:
        category = category_for(data, config["fallback_category"])
    rationale = "keyword match" if score else "configured fallback"
    proposal = None
    if classifier is not None:
        try:
            proposal = classifier(copy.deepcopy(ticket), tuple(c["name"] for c in config["categories"]))
        except Exception as exc:
            raise ValidationError("injected classifier failed") from exc
        obj(proposal, ("category", "priority"), "classifier output")
        choice(proposal["category"], [c["name"] for c in config["categories"]], "classifier.category")
        choice(proposal["priority"], PRIORITIES, "classifier.priority")
        category = category_for(data, proposal["category"])
        rationale = "validated injected classifier"
    candidates = [ticket["severity"], category["default_priority"]]
    if proposal:
        candidates.append(proposal["priority"])
    rules = [r for r in config["priority_rules"] if match_keywords(r["keywords"], body)]
    candidates.extend(rule["priority"] for rule in rules)
    priority = max(candidates, key=PRIORITIES.index)
    return {"ticket_id": ticket["id"], "category": category["name"], "priority": priority,
            "team": category["team"], "owner": category["owner"],
            "onboarding_steps": list(category["onboarding_steps"]),
            "research_terms": list(category["research_terms"]),
            "reason": rationale + "; severity, category and keyword-rule priority floor applied"}


def adaptive_result(data, triage):
    profile = data["profile"]
    steps = {s["id"]: s for s in data["config"]["steps"]}
    completed = set(profile["completed_steps"])
    done, blocked, order, skipped = set(completed), {}, [], []

    def schedule(step_id):
        if step_id in done:
            if step_id in completed and step_id not in skipped:
                skipped.append(step_id)
            return True
        if step_id in blocked:
            return False
        step = steps[step_id]
        if EXPERIENCE.index(profile["experience"]) < EXPERIENCE.index(step["min_experience"]):
            blocked[step_id] = "requires " + step["min_experience"] + " experience"
            return False
        ready = [schedule(p) for p in step["prerequisites"]]
        if not all(ready):
            blocked[step_id] = "a prerequisite is blocked"
            return False
        preferred = profile["preferred_format"]
        fmt = preferred if preferred in step["formats"] else step["formats"][0]
        reason = ("category recommendation" if step_id in triage["onboarding_steps"]
                  else "required prerequisite")
        reason += "; experience requirement met; "
        reason += "preferred format" if fmt == preferred else "preferred format unavailable; configured fallback"
        order.append({"id": step_id, "title": step["title"], "format": fmt,
                      "prerequisites": list(step["prerequisites"]), "reason": reason})
        done.add(step_id)
        return True

    for step_id in triage["onboarding_steps"]:
        schedule(step_id)
    terms = list(triage["research_terms"])
    for item in order:
        for term in steps[item["id"]]["research_terms"]:
            if term not in terms:
                terms.append(term)
    return {"ticket_id": triage["ticket_id"], "category": triage["category"],
            "priority": triage["priority"], "owner": triage["owner"],
            "plan": order, "skipped_completed": skipped,
            "blocked": [{"id": key, "reason": value} for key, value in blocked.items()],
            "research_terms": terms}


def web_result(data, adaptive):
    query_terms = sorted(tokens(" ".join(adaptive["research_terms"])))
    hits = []
    for source in data["sources"]:
        matched = sorted(set(query_terms) & tokens(source["content"]))
        if not matched:
            continue
        hits.append({"source_id": source["id"],
                     "url": canonical_url(source["url"], data["config"]["allowlisted_hosts"]),
                     "title": source["title"], "finding": source["content"],
                     "provenance": {"start": 0, "end": len(source["content"]),
                                    "synthetic": True, "ingestion": "provided_snapshot"},
                     "matched_terms": matched, "score": len(matched)})
    hits.sort(key=lambda h: (-h["score"], h["source_id"]))
    return {"ticket_id": adaptive["ticket_id"], "category": adaptive["category"],
            "priority": adaptive["priority"], "owner": adaptive["owner"],
            "plan_step_ids": [s["id"] for s in adaptive["plan"]],
            "query_terms": query_terms, "findings": hits[:data["config"]["max_findings"]],
            "status": "found" if hits else "no_results"}


def validate_envelope(envelope, expected_stage):
    """One shared boundary validator; recomputes deterministic derived artifacts."""
    obj(envelope, ("schema_version", "status", "stage", "input", "triage", "adaptive", "web"),
        "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "invalid envelope version")
    require(envelope["status"] == "ok" and envelope["stage"] == expected_stage,
            "invalid pipeline state")
    choice(expected_stage, STAGES, "expected_stage")
    data = validate_input(envelope["input"])
    stage_index = STAGES.index(expected_stage)
    for i, stage in enumerate(STAGES[1:], 1):
        require((envelope[stage] is not None) == (i <= stage_index),
                "invalid stage presence")
    if stage_index >= 1:
        result = envelope["triage"]
        obj(result, ("ticket_id", "category", "priority", "team", "owner",
                     "onboarding_steps", "research_terms", "reason"), "triage output")
        choice(result["category"], [c["name"] for c in data["config"]["categories"]], "triage.category")
        choice(result["priority"], PRIORITIES, "triage.priority")
        category = category_for(data, result["category"])
        require(result["ticket_id"] == data["ticket"]["id"], "ticket identity changed")
        for field in ("team", "owner", "onboarding_steps", "research_terms"):
            require(result[field] == category[field], "triage route or handoff mismatch")
        text(result["reason"], "triage.reason")
        floors = [data["ticket"]["severity"], category["default_priority"]]
        body = data["ticket"]["subject"] + " " + data["ticket"]["body"]
        floors.extend(r["priority"] for r in data["config"]["priority_rules"]
                      if match_keywords(r["keywords"], body))
        require(PRIORITIES.index(result["priority"]) >= max(map(PRIORITIES.index, floors)),
                "priority below configured floor")
    if stage_index >= 2:
        require(envelope["adaptive"] == adaptive_result(data, envelope["triage"]),
                "adaptive output violates handoff or plan")
    if stage_index >= 3:
        require(envelope["web"] == web_result(data, envelope["adaptive"]),
                "research output violates handoff or provenance")
    return envelope


def triage_stage(envelope, classifier=None):
    validate_envelope(envelope, "input")
    result = copy.deepcopy(envelope)
    result["triage"] = triage_result(result["input"], classifier)
    result["stage"] = "triage"
    return validate_envelope(result, "triage")


def adaptive_stage(envelope):
    validate_envelope(envelope, "triage")
    result = copy.deepcopy(envelope)
    result["adaptive"] = adaptive_result(result["input"], result["triage"])
    result["stage"] = "adaptive"
    return validate_envelope(result, "adaptive")


def web_stage(envelope):
    validate_envelope(envelope, "adaptive")
    result = copy.deepcopy(envelope)
    result["web"] = web_result(result["input"], result["adaptive"])
    result["stage"] = "web"
    return validate_envelope(result, "web")


def initial_envelope(data):
    return validate_envelope({"schema_version": 1, "status": "ok", "stage": "input",
                              "input": copy.deepcopy(data), "triage": None,
                              "adaptive": None, "web": None}, "input")


def run_pipeline(data, classifier=None):
    return web_stage(adaptive_stage(triage_stage(initial_envelope(data), classifier)))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonstandard JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_text(encoding="utf-8")
        require(len(raw) <= 2_000_000, "input file exceeds size limit")
        data = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
