"""Synthetic, offline reference pipeline. Run: python -B implementation.py INPUT.json.

The web stage retrieves exact URLs from supplied snapshot fixtures, never a network.
All stages share the same envelope; each validates the preceding stage before use.
"""

import copy
import datetime as dt
import json
import math
import re
import sys
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = "1.0"
STAGES = ("input", "triage", "web", "feedback", "behavior")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, name, fields):
    require(isinstance(value, dict), name + " must be an object")
    require(set(value) == set(fields), name + " has missing or unknown fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def array(value, name):
    require(isinstance(value, list), name + " must be an array")
    return value


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            name + " must be a finite number >= " + str(minimum))
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO timestamp") from exc
    require(result.tzinfo is not None, "timestamp must include a timezone")
    return result


def unique_records(records, name):
    array(records, name)
    ids = [text(r.get("id"), name + ".id") if isinstance(r, dict)
           else text(None, name + ".id") for r in records]
    require(len(ids) == len(set(ids)), name + " ids must be unique")


def words(value):
    return re.findall(r"\w+", value.casefold())


def matches(value, keyword):
    haystack, needle = words(value), words(keyword)
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                               for i in range(len(haystack) - len(needle) + 1))


def canonical_url(value):
    text(value, "URL")
    require(not any(c.isspace() or ord(c) < 32 for c in value), "invalid URL whitespace")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ValidationError("malformed URL") from exc
    require(parts.scheme.lower() == "https" and bool(parts.hostname), "URL must be HTTPS")
    require(parts.username is None and parts.password is None and port in (None, 443),
            "URL credentials and non-HTTPS ports are forbidden")
    host = parts.hostname.lower()
    require(bool(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host)), "invalid URL host")
    return urlunsplit(("https", host, parts.path or "/", parts.query, ""))


def validate_input(data):
    obj(data, "input", ("schema_version", "synthetic", "tickets", "triage_config",
                       "web_config", "feedback", "behavior"))
    require(data["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    require(data["synthetic"] is True, "reference input must be labeled synthetic")
    cfg = data["triage_config"]
    obj(cfg, "triage_config", ("categories", "default_category", "urgent_keywords"))
    array(cfg["categories"], "categories")
    require(bool(cfg["categories"]), "at least one category is required")
    names = []
    for category in cfg["categories"]:
        obj(category, "category", ("name", "keywords", "owner", "priority"))
        names.append(text(category["name"], "category.name"))
        text(category["owner"], "category.owner")
        require(category["priority"] in ("low", "normal", "high"), "invalid category priority")
        array(category["keywords"], "keywords")
        for keyword in category["keywords"]:
            require(bool(words(text(keyword, "keyword"))), "keyword must contain words")
    require(len(names) == len(set(names)), "category names must be unique")
    require(cfg["default_category"] in names, "default category must exist")
    for keyword in array(cfg["urgent_keywords"], "urgent_keywords"):
        require(bool(words(text(keyword, "urgent keyword"))), "urgent keyword must contain words")

    unique_records(data["tickets"], "tickets")
    for ticket in data["tickets"]:
        obj(ticket, "ticket", ("id", "text", "urls"))
        text(ticket["text"], "ticket.text")
        for url in array(ticket["urls"], "ticket.urls"):
            canonical_url(url)
    web = data["web_config"]
    obj(web, "web_config", ("allowed_hosts", "snapshots", "max_excerpt_chars"))
    hosts = array(web["allowed_hosts"], "allowed_hosts")
    require(bool(hosts), "at least one allowed host is required")
    for host in hosts:
        text(host, "host")
        require(bool(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host)),
                "allowlist requires lowercase exact host names")
    require(len(set(hosts)) == len(hosts), "duplicate allowed hosts")
    require(type(web["max_excerpt_chars"]) is int and 1 <= web["max_excerpt_chars"] <= 2000,
            "max_excerpt_chars must be an integer in [1, 2000]")
    urls = []
    for snapshot in array(web["snapshots"], "snapshots"):
        obj(snapshot, "snapshot", ("url", "title", "content", "retrieved_at"))
        url = canonical_url(snapshot["url"])
        require(urlsplit(url).hostname in hosts, "snapshot host is not allowlisted")
        urls.append(url)
        text(snapshot["title"], "snapshot.title")
        text(snapshot["content"], "snapshot.content")
        timestamp(snapshot["retrieved_at"])
    require(len(urls) == len(set(urls)), "duplicate canonical snapshot URL")

    ticket_ids = {t["id"] for t in data["tickets"]}
    unique_records(data["feedback"], "feedback")
    for item in data["feedback"]:
        obj(item, "feedback item", ("id", "ticket_id", "text"))
        require(item["ticket_id"] in ticket_ids, "feedback references an unknown ticket")
        require(bool(words(text(item["text"], "feedback.text"))), "feedback needs words")
    behavior = data["behavior"]
    obj(behavior, "behavior", ("now", "user_id", "half_life_days", "limit",
                              "catalog", "events"))
    now = timestamp(behavior["now"])
    text(behavior["user_id"], "user_id")
    number(behavior["half_life_days"], "half_life_days")
    require(behavior["half_life_days"] > 0, "half_life_days must be positive")
    require(type(behavior["limit"]) is int and behavior["limit"] > 0, "limit must be positive integer")
    unique_records(behavior["catalog"], "catalog")
    for item in behavior["catalog"]:
        obj(item, "catalog item", ("id", "title", "themes", "popularity"))
        text(item["title"], "catalog.title")
        for theme in array(item["themes"], "catalog.themes"):
            require(theme in names, "unknown catalog theme")
        require(len(item["themes"]) == len(set(item["themes"])), "duplicate catalog themes")
        number(item["popularity"], "popularity")
    catalog_ids = {item["id"] for item in behavior["catalog"]}
    unique_records(behavior["events"], "events")
    for event in behavior["events"]:
        obj(event, "event", ("id", "user_id", "item_id", "kind", "at"))
        text(event["user_id"], "event.user_id")
        require(event["item_id"] in catalog_ids, "unknown event item")
        require(event["kind"] in ("browse", "purchase"), "invalid event kind")
        require(timestamp(event["at"]) <= now, "future behavior event")
    for snapshot in web["snapshots"]:
        require(timestamp(snapshot["retrieved_at"]) <= now, "future snapshot")


def _triage(data):
    config = data["triage_config"]
    categories = config["categories"]
    default = next(c for c in categories if c["name"] == config["default_category"])
    result = []
    for ticket in data["tickets"]:
        counts = [sum(matches(ticket["text"], k) for k in c["keywords"]) for c in categories]
        best = max(counts)
        category = categories[counts.index(best)] if best else default
        urgent = any(matches(ticket["text"], k) for k in config["urgent_keywords"])
        result.append({
            "ticket_id": ticket["id"], "category": category["name"],
            "priority": "high" if urgent else category["priority"], "owner": category["owner"],
            "matched_keywords": [k for k in category["keywords"] if matches(ticket["text"], k)],
            "urgent": urgent, "urls": sorted({canonical_url(u) for u in ticket["urls"]}),
        })
    return {"tickets": result}


def _web(data, triage):
    config = data["web_config"]
    snapshots = {canonical_url(s["url"]): s for s in config["snapshots"]}
    findings, unresolved = [], []
    for ticket in triage["tickets"]:
        for url in ticket["urls"]:
            if urlsplit(url).hostname not in config["allowed_hosts"]:
                reason = "host_not_allowlisted"
            elif url not in snapshots:
                reason = "snapshot_not_found"
            else:
                snapshot = snapshots[url]
                excerpt = snapshot["content"][:config["max_excerpt_chars"]]
                findings.append({
                    "id": "finding-" + str(len(findings) + 1),
                    "ticket_id": ticket["ticket_id"], "category": ticket["category"],
                    "owner": ticket["owner"], "priority": ticket["priority"],
                    "url": url, "title": snapshot["title"],
                    "retrieved_at": snapshot["retrieved_at"],
                    "excerpt": excerpt, "start": 0, "end": len(excerpt),
                })
                continue
            unresolved.append({"ticket_id": ticket["ticket_id"], "url": url, "reason": reason})
    return {"findings": findings, "unresolved": unresolved}


def _feedback(data, web):
    groups = {}
    for record in data["feedback"]:
        normalized = " ".join(words(record["text"]))
        groups.setdefault(normalized, []).append(record)
    deduplicated = []
    themes = {}
    categories = data["triage_config"]["categories"]
    for records in groups.values():
        record_ids = [r["id"] for r in records]
        ticket_ids = sorted({r["ticket_id"] for r in records})
        findings = [f for f in web["findings"] if f["ticket_id"] in ticket_ids]
        theme_names = {c["name"] for c in categories
                       if any(matches(records[0]["text"], k) for k in c["keywords"])}
        theme_names.update(f["category"] for f in findings)
        if not theme_names:
            theme_names.add(data["triage_config"]["default_category"])
        group_id = records[0]["id"]
        support = [{"feedback_id": r["id"], "ticket_id": r["ticket_id"],
                    "excerpt": r["text"], "start": 0, "end": len(r["text"])} for r in records]
        finding_ids = [f["id"] for f in findings]
        deduplicated.append({
            "id": group_id, "record_ids": record_ids, "ticket_ids": ticket_ids,
            "themes": sorted(theme_names), "finding_ids": finding_ids, "support": support,
        })
        for name in sorted(theme_names):
            theme = themes.setdefault(name, {"name": name, "count": 0,
                                            "feedback_group_ids": [], "finding_ids": []})
            theme["count"] += 1
            theme["feedback_group_ids"].append(group_id)
            theme["finding_ids"] = sorted(set(theme["finding_ids"]) | set(finding_ids))
    return {"groups": deduplicated, "themes": [themes[name] for name in sorted(themes)]}


def _behavior(data, feedback):
    config = data["behavior"]
    now = timestamp(config["now"])
    events = [e for e in config["events"] if e["user_id"] == config["user_id"]]
    affinities = {theme["name"]: theme for theme in feedback["themes"]}
    rankings = []
    for item in config["catalog"]:
        relevant = [e for e in events if e["item_id"] == item["id"]]
        event_score = sum((3.0 if e["kind"] == "purchase" else 1.0) *
                          2.0 ** (-((now - timestamp(e["at"])).total_seconds() / 86400) /
                                  config["half_life_days"]) for e in relevant)
        used_themes = sorted(t for t in item["themes"] if t in affinities)
        feedback_score = sum(affinities[t]["count"] for t in used_themes)
        popularity_score = math.log1p(item["popularity"]) * 0.1
        score = event_score + feedback_score + popularity_score
        rankings.append({
            "item_id": item["id"], "score": score, "event_score": event_score,
            "feedback_score": feedback_score, "popularity_score": popularity_score,
            "event_ids": [e["id"] for e in relevant], "themes": used_themes,
            "feedback_group_ids": sorted({g for t in used_themes
                                          for g in affinities[t]["feedback_group_ids"]}),
            "finding_ids": sorted({f for t in used_themes for f in affinities[t]["finding_ids"]}),
        })
    rankings.sort(key=lambda r: (-r["score"], r["item_id"]))
    return {"user_id": config["user_id"], "cold_start": not events,
            "strategy": "feedback_and_popularity" if not events else "recency_feedback_popularity",
            "rankings": rankings[:config["limit"]]}


def validate(envelope, expected_stage):
    """One shared boundary validator, including deterministic provenance checks.

    Recomputing bounded stage payloads verifies exact fields, scores, references,
    routing and excerpts instead of trusting a merely shape-correct handoff.
    """
    obj(envelope, "envelope", ("schema_version", "status", "stage", "data", "results"))
    require(envelope["schema_version"] == SCHEMA_VERSION, "unsupported envelope schema")
    require(envelope["status"] == "ok", "invalid success status")
    require(expected_stage in STAGES and envelope["stage"] == expected_stage, "unexpected stage")
    validate_input(envelope["data"])
    completed = STAGES[1:STAGES.index(expected_stage) + 1]
    obj(envelope["results"], "results", completed)
    expected = {}
    for stage in completed:
        if stage == "triage":
            result = _triage(envelope["data"])
        elif stage == "web":
            result = _web(envelope["data"], expected["triage"])
        elif stage == "feedback":
            result = _feedback(envelope["data"], expected["web"])
        else:
            result = _behavior(envelope["data"], expected["feedback"])
        # JSON comparison is type-sensitive (unlike Python's True == 1).
        require(json.dumps(envelope["results"][stage], sort_keys=True, allow_nan=False) ==
                json.dumps(result, sort_keys=True, allow_nan=False),
                "invalid " + stage + " result or provenance")
        expected[stage] = result
    return envelope


def advance(envelope, stage):
    require(stage in STAGES[1:], "unknown next stage")
    validate(envelope, STAGES[STAGES.index(stage) - 1])
    output = copy.deepcopy(envelope)
    data, results = output["data"], output["results"]
    if stage == "triage":
        results[stage] = _triage(data)
    elif stage == "web":
        results[stage] = _web(data, results["triage"])
    elif stage == "feedback":
        results[stage] = _feedback(data, results["web"])
    else:
        results[stage] = _behavior(data, results["feedback"])
    output["stage"] = stage
    return validate(output, stage)


def run_pipeline(data):
    envelope = {"schema_version": SCHEMA_VERSION, "status": "ok",
                "stage": "input", "data": copy.deepcopy(data), "results": {}}
    validate(envelope, "input")
    for stage in STAGES[1:]:
        envelope = advance(envelope, stage)
    return envelope


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


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
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
        rendered = json.dumps(output, ensure_ascii=True, sort_keys=True, allow_nan=False)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error", "error": str(exc)}))
        return 2
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
