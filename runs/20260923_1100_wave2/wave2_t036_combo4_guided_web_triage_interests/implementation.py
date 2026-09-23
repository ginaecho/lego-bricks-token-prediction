"""Synthetic, offline reference pipeline. Python 3 standard library only.

Usage: python -B implementation.py example_input.json
Retrieval uses supplied fixture documents exclusively; no network is performed.
All four stages use the same versioned envelope and validation primitives.
"""

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


VERSION = "1.0"
PRIORITY = {"low": 1, "normal": 2, "high": 3, "urgent": 4}


class ValidationError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def require(condition, message, details=None):
    if not condition:
        raise ValidationError(message, details)


def obj(value, path, required, optional=()):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(required) <= set(value), f"{path}: missing required fields")
    require(set(value) <= set(required) | set(optional),
            f"{path}: unknown fields")
    return value


def text(value, path, maximum=20000):
    require(isinstance(value, str) and bool(value.strip()),
            f"{path}: expected nonempty string")
    require(len(value) <= maximum, f"{path}: text too long")
    return value


def sequence(value, path, nonempty=False):
    require(isinstance(value, list), f"{path}: expected array")
    require(len(value) <= 1000, f"{path}: too many entries")
    require(not nonempty or bool(value), f"{path}: array cannot be empty")
    return value


def strings(value, path, nonempty=False):
    sequence(value, path, nonempty)
    for entry in value:
        text(entry, path, 2000)
    require(len(value) == len(set(value)), f"{path}: duplicate values")
    return value


def integer(value, path, lower, upper):
    require(type(value) is int and lower <= value <= upper,
            f"{path}: expected integer {lower}..{upper}")
    return value


def identifier(value, path):
    text(value, path, 80)
    require(bool(re.fullmatch(r"[a-zA-Z0-9_-]+", value)),
            f"{path}: invalid identifier")
    return value


def records(value, path, required, optional=(), nonempty=False):
    sequence(value, path, nonempty)
    seen = set()
    for record in value:
        obj(record, path, required, optional)
        key = identifier(record["id"], path + ".id")
        require(key not in seen, f"{path}: duplicate id {key}")
        seen.add(key)
    return value


def previous(envelope, stage):
    obj(envelope, "output", ["schema_version", "fixture_label", "status", "stages"])
    require(envelope["schema_version"] == VERSION, "unsupported output schema")
    require(isinstance(envelope["stages"], dict), "invalid stage container")
    result = envelope["stages"].get(stage)
    require(isinstance(result, dict) and result.get("status") == "complete",
            f"{stage}: validated completed predecessor required")
    return result


def allowed_url(value, hosts):
    text(value, "URL", 2000)
    require(not any(c.isspace() or ord(c) < 32 for c in value)
            and "\\" not in value, "URL: whitespace/control/backslash forbidden")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValidationError("URL: malformed authority") from exc
    require(parsed.scheme == "https" and parsed.hostname in hosts,
            "URL: HTTPS and exact allowlisted host required")
    require(parsed.username is None and parsed.password is None
            and port in (None, 443) and not parsed.fragment,
            "URL: credentials, non-HTTPS ports and fragments forbidden")
    return value


def hits(content, keywords):
    return [keyword for keyword in keywords
            if re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
                         content, re.IGNORECASE)]


def guided(config):
    obj(config, "guided", ["steps", "responses"])
    steps = records(config["steps"], "guided.steps",
                    ["id", "requires"], nonempty=True)
    ids = {step["id"] for step in steps}
    require({"research_consent", "support_consent", "personalization_consent"} <= ids,
            "guided: all three consent steps are required")
    require(isinstance(config["responses"], dict), "guided.responses: expected object")
    require(set(config["responses"]) <= ids, "guided: unknown response")
    for value in config["responses"].values():
        require(type(value) is bool, "guided: responses must be booleans")
    for step in steps:
        deps = strings(step["requires"], "guided.requires")
        require(set(deps) <= ids and step["id"] not in deps,
                "guided: unknown or self prerequisite")
    order = []
    remaining = list(steps)
    while remaining:
        ready = [step for step in remaining if set(step["requires"]) <= set(order)]
        require(bool(ready), "guided: cyclic prerequisites")
        for step in ready:
            order.append(step["id"])
            remaining.remove(step)
    by_id = {step["id"]: step for step in steps}
    completed = set()
    progress = []
    for key in order:
        prerequisites_met = set(by_id[key]["requires"]) <= completed
        status = ("blocked" if not prerequisites_met else
                  "complete" if config["responses"].get(key, False) else "pending")
        if status == "complete":
            completed.add(key)
        progress.append({"id": key, "status": status,
                         "requires": by_id[key]["requires"]})
    result = {"status": "complete" if len(completed) == len(steps) else "incomplete",
              "steps": progress, "completed": len(completed), "total": len(steps),
              "percent": round(100 * len(completed) / len(steps), 2)}
    require(result["status"] == "complete", "guided: onboarding incomplete", result)
    return result


def research(config, envelope):
    setup = previous(envelope, "guided")
    require(any(step["id"] == "research_consent" and step["status"] == "complete"
                for step in setup["steps"]), "research: consent required")
    obj(config, "web", ["allowlisted_hosts", "urls", "documents", "topics"])
    hosts = strings(config["allowlisted_hosts"], "web.allowlisted_hosts", True)
    for host in hosts:
        require(bool(re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+",
            host)), "web: host must be a lowercase domain")
    urls = strings(config["urls"], "web.urls", True)
    require(isinstance(config["documents"], dict), "web.documents: expected object")
    require(set(config["documents"]) == set(urls),
            "web: fixture documents must exactly match ingested URLs")
    topics = config["topics"]
    require(isinstance(topics, dict) and bool(topics), "web.topics: nonempty object required")
    for topic, keywords in topics.items():
        identifier(topic, "web.topic")
        strings(keywords, "web.topic.keywords", True)
    findings = []
    for index, url in enumerate(urls, 1):
        allowed_url(url, hosts)
        document = obj(config["documents"][url], "web.document", ["title", "text"])
        title = text(document["title"], "web.document.title", 300)
        evidence = text(document["text"], "web.document.text")
        findings.append({
            "id": f"F{index}", "url": url, "title": title, "evidence": evidence,
            "sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
            "retrieval": "synthetic_fixture",
            "topics": sorted(topic for topic, words in topics.items()
                             if hits(evidence, words)),
        })
    return {"status": "complete", "consumed_stage": "guided",
            "onboarding_percent": setup["percent"], "findings": findings}


def route(config, envelope):
    web = previous(envelope, "web")
    obj(config, "triage", ["tickets", "rules", "routes", "fallback",
                           "urgent_keywords"])
    rules = records(config["rules"], "triage.rules",
                    ["id", "category", "keywords", "priority"])
    categories = set()
    for rule in rules:
        identifier(rule["category"], "triage.category")
        strings(rule["keywords"], "triage.keywords", True)
        require(isinstance(rule["priority"], str) and rule["priority"] in PRIORITY,
                "triage: invalid priority")
        categories.add(rule["category"])
    fallback = obj(config["fallback"], "triage.fallback", ["category", "priority"])
    identifier(fallback["category"], "triage.fallback.category")
    require(isinstance(fallback["priority"], str) and fallback["priority"] in PRIORITY,
            "triage: invalid fallback priority")
    categories.add(fallback["category"])
    require(isinstance(config["routes"], dict)
            and set(config["routes"]) == categories,
            "triage: exactly one accountable route required per category")
    for routing in config["routes"].values():
        obj(routing, "triage.route", ["queue", "owner"])
        text(routing["queue"], "triage.route.queue", 200)
        text(routing["owner"], "triage.route.owner", 200)
    urgent = strings(config["urgent_keywords"], "triage.urgent_keywords")
    tickets = records(config["tickets"], "triage.tickets",
                      ["id", "subject", "body", "source_urls"], nonempty=True)
    findings_by_url = {finding["url"]: finding for finding in web["findings"]}
    routed = []
    for ticket in tickets:
        subject = text(ticket["subject"], "triage.ticket.subject", 300)
        body = text(ticket["body"], "triage.ticket.body")
        urls = strings(ticket["source_urls"], "triage.source_urls", True)
        require(set(urls) <= set(findings_by_url),
                "triage: ticket references unvalidated research")
        findings = [findings_by_url[url] for url in urls]
        ticket_content = subject + "\n" + body
        content = ticket_content + "\n" + "\n".join(f["evidence"] for f in findings)
        scored = [(len(hits(content, rule["keywords"])), -index, rule)
                  for index, rule in enumerate(rules)]
        selected = max(scored, key=lambda entry: entry[:2]) if scored else None
        chosen = selected[2] if selected and selected[0] else fallback
        category = chosen["category"]
        urgent_matches = hits(ticket_content, urgent)
        priority = "urgent" if urgent_matches else chosen["priority"]
        routed.append({
            "id": ticket["id"], "category": category, "priority": priority,
            "route": dict(config["routes"][category]),
            "rule_id": chosen.get("id"),
            "matched_keywords": hits(content, chosen.get("keywords", [])),
            "urgent_matches": urgent_matches,
            "finding_ids": [f["id"] for f in findings],
            "source_urls": urls,
            "topics": sorted({topic for f in findings for topic in f["topics"]}),
        })
    return {"status": "complete", "consumed_stage": "web",
            "tickets": routed, "finding_ids": [f["id"] for f in web["findings"]]}


def interests(config, envelope):
    triage = previous(envelope, "triage")
    web = previous(envelope, "web")
    obj(config, "interests", ["preferences", "excluded_topics", "excluded_item_ids",
                              "limit", "items"])
    preferences = config["preferences"]
    require(isinstance(preferences, dict), "interests.preferences: expected object")
    for topic, weight in preferences.items():
        identifier(topic, "interests.preference")
        integer(weight, "interests.preference.weight", 1, 5)
    excluded_topics = strings(config["excluded_topics"], "interests.excluded_topics")
    excluded_ids = strings(config["excluded_item_ids"], "interests.excluded_item_ids")
    limit = integer(config["limit"], "interests.limit", 0, 100)
    items = records(config["items"], "interests.items",
                    ["id", "title", "topics", "category", "source_urls"])
    by_url = {f["url"]: f for f in web["findings"]}
    ranked = []
    exclusions = []
    for item in items:
        text(item["title"], "interests.item.title", 300)
        identifier(item["category"], "interests.item.category")
        topics = strings(item["topics"], "interests.item.topics", True)
        urls = strings(item["source_urls"], "interests.item.source_urls", True)
        require(set(urls) <= set(by_url), "interests: unvalidated source URL")
        supported_topics = {t for url in urls for t in by_url[url]["topics"]}
        require(set(topics) <= supported_topics, "interests: ungrounded item topics")
        if item["id"] in excluded_ids or set(topics) & set(excluded_topics):
            exclusions.append({"id": item["id"], "reason": "explicit_exclusion"})
            continue
        matching_preferences = {t: preferences[t] for t in sorted(topics)
                                if t in preferences}
        related = [ticket for ticket in triage["tickets"]
                   if ticket["category"] == item["category"]
                   and set(ticket["source_urls"]) & set(urls)
                   and set(ticket["topics"]) & set(topics)]
        if not matching_preferences or not related:
            exclusions.append({"id": item["id"], "reason": "no_grounded_interest_match"})
            continue
        preference_score = sum(matching_preferences.values())
        urgency_score = max(PRIORITY[ticket["priority"]] for ticket in related)
        ranked.append({
            "id": item["id"], "title": item["title"],
            "score": preference_score * 10 + urgency_score,
            "explanation": {
                "matched_preferences": matching_preferences,
                "preference_score": preference_score,
                "urgency_score": urgency_score,
                "formula": "10 * preference_score + urgency_score",
                "ticket_ids": [ticket["id"] for ticket in related],
                "accountable_owners": sorted({t["route"]["owner"] for t in related}),
                "finding_ids": [by_url[url]["id"] for url in urls],
                "source_urls": urls,
            },
        })
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return {"status": "complete", "consumed_stage": "triage",
            "recommendations": ranked[:limit], "eligible_count": len(ranked),
            "excluded": exclusions}


def run_pipeline(data):
    obj(data, "input", ["schema_version", "fixture_label", "guided", "web",
                       "triage", "interests"])
    require(data["schema_version"] == VERSION, "unsupported schema_version")
    require(isinstance(data["fixture_label"], str)
            and data["fixture_label"].startswith("SYNTHETIC"),
            "fixture_label must explicitly begin with SYNTHETIC")
    text(data["fixture_label"], "fixture_label", 300)
    envelope = {"schema_version": VERSION, "fixture_label": data["fixture_label"],
                "status": "ok", "stages": {}}
    envelope["stages"]["guided"] = guided(data["guided"])
    envelope["stages"]["web"] = research(data["web"], envelope)
    envelope["stages"]["triage"] = route(data["triage"], envelope)
    envelope["stages"]["interests"] = interests(data["interests"], envelope)
    return envelope


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON: duplicate key {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"JSON: nonfinite constant {value} forbidden")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: implementation.py INPUT.json")
        raw = Path(argv[0]).read_text(encoding="utf-8")
        data = json.loads(raw, object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        result = run_pipeline(data)
    except (ValueError, OSError, RecursionError) as exc:
        result = {"schema_version": VERSION, "status": "error", "error": str(exc)}
        if isinstance(exc, ValidationError) and exc.details is not None:
            result["details"] = exc.details
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
