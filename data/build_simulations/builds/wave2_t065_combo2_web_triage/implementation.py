"""Offline, deterministic synthetic web-research -> support-triage reference CLI."""

import hashlib
import json
import re
import sys
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, where):
    require(isinstance(value, dict), where + " must be an object")
    require(set(value) == set(names.split()), where + " has missing or unknown fields")


def text(value, where, allow_empty=False):
    require(isinstance(value, str), where + " must be a string")
    require(allow_empty or bool(value.strip()), where + " must not be blank")


def array(value, where):
    require(isinstance(value, list), where + " must be an array")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def allowed_url(value, hosts):
    text(value, "URL")
    require(not any(c.isspace() or ord(c) < 32 for c in value), "URL contains whitespace")
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(url.scheme == "https" and url.hostname in hosts, "URL is not allowlisted HTTPS")
    require(url.username is None and url.password is None, "URL credentials forbidden")
    require(port in (None, 443) and not url.fragment and "\\" not in value,
            "URL port, fragment or backslash forbidden")
    return value


def validate(kind, value, context=None):
    """One validation boundary shared by ingestion, research, routing and output."""
    if kind == "input":
        fields(value, "schema_version synthetic allowed_hosts pages sources tickets routing", kind)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        require(value["synthetic"] is True, "Fixtures must be explicitly synthetic")
        hosts = value["allowed_hosts"]
        array(hosts, "allowed_hosts")
        require(bool(hosts), "At least one allowlisted host required")
        for host in hosts:
            text(host, "host")
            require(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host) is not None,
                    "Hosts must be lowercase DNS names")
        require(len(set(hosts)) == len(hosts), "Duplicate allowed host")
        require(isinstance(value["pages"], dict), "pages must be an object")
        for url, body in value["pages"].items():
            allowed_url(url, hosts)
            text(body, "page content")
        array(value["sources"], "sources")
        ids = set()
        urls = set()
        for source in value["sources"]:
            fields(source, "id url", "source")
            text(source["id"], "source.id")
            url = allowed_url(source["url"], hosts)
            require(source["id"] not in ids and url not in urls, "Duplicate source")
            require(url in value["pages"], "Source has no offline retrieval fixture")
            ids.add(source["id"])
            urls.add(url)
        array(value["tickets"], "tickets")
        ticket_ids = set()
        for ticket in value["tickets"]:
            fields(ticket, "id subject body source_ids", "ticket")
            for key in ("id", "subject"):
                text(ticket[key], "ticket." + key)
            text(ticket["body"], "ticket.body", allow_empty=True)
            require(ticket["id"] not in ticket_ids, "Duplicate ticket id")
            ticket_ids.add(ticket["id"])
            array(ticket["source_ids"], "source_ids")
            for source_id in ticket["source_ids"]:
                text(source_id, "source_id")
                require(source_id in ids, "Unknown source reference")
            require(len(set(ticket["source_ids"])) == len(ticket["source_ids"]),
                    "Duplicate source reference")
        routing = value["routing"]
        fields(routing, "rules fallback", "routing")
        array(routing["rules"], "rules")
        categories = set()
        for rule in routing["rules"]:
            fields(rule, "category keywords priority team owner", "rule")
            validate("route", {k: rule[k] for k in ("category", "priority", "team", "owner")})
            require(rule["category"] not in categories, "Duplicate category")
            categories.add(rule["category"])
            array(rule["keywords"], "keywords")
            require(bool(rule["keywords"]), "Rule keywords cannot be empty")
            for keyword in rule["keywords"]:
                text(keyword, "keyword")
        validate("route", routing["fallback"])
    elif kind == "route":
        fields(value, "category priority team owner", kind)
        for key in ("category", "team", "owner"):
            text(value[key], key)
        require(type(value["priority"]) is int and 1 <= value["priority"] <= 4,
                "priority must be integer 1 (highest) through 4")
    elif kind == "research":
        fields(value, "schema_version synthetic findings", kind)
        require(value["schema_version"] == 1 and value["synthetic"] is True,
                "Invalid research envelope")
        array(value["findings"], "findings")
        require(len(value["findings"]) == len(context["sources"]), "Missing findings")
        for finding, source in zip(value["findings"], context["sources"]):
            fields(finding, "source_id url content sha256 retrieval", "finding")
            require(finding["source_id"] == source["id"] and finding["url"] == source["url"],
                    "Finding provenance mismatch")
            allowed_url(finding["url"], context["allowed_hosts"])
            text(finding["content"], "finding.content")
            require(finding["content"] == context["pages"][source["url"]],
                    "Retrieved content does not match fixture")
            require(finding["sha256"] == digest(finding["content"]), "Finding digest mismatch")
            require(finding["retrieval"] == "synthetic-offline-fixture", "Invalid retrieval mode")
    elif kind == "output":
        fields(value, "status schema_version synthetic research tickets", kind)
        require(value["status"] == "ok" and value["schema_version"] == 1
                and value["synthetic"] is True, "Invalid output envelope")
        validate("research", value["research"], context)
        array(value["tickets"], "output tickets")
        require(len(value["tickets"]) == len(context["tickets"]), "Missing triaged tickets")
        index = {f["source_id"]: f for f in value["research"]["findings"]}
        for result, ticket in zip(value["tickets"], context["tickets"]):
            fields(result, "id subject route matched_keywords citations", "triaged ticket")
            require(result["id"] == ticket["id"] and result["subject"] == ticket["subject"],
                    "Ticket identity mismatch")
            validate("route", result["route"])
            expected = classify(ticket, index, context["routing"])
            require(result == expected, "Triage output or provenance mismatch")
    else:
        raise ValidationError("Unknown validation kind")
    return value


def research(request):
    validate("input", request)
    findings = []
    for source in request["sources"]:
        content = request["pages"][source["url"]]
        findings.append({
            "source_id": source["id"], "url": source["url"], "content": content,
            "sha256": digest(content), "retrieval": "synthetic-offline-fixture",
        })
    return validate("research", {"schema_version": 1, "synthetic": True,
                                 "findings": findings}, request)


def classify(ticket, index, routing):
    evidence = [index[source_id] for source_id in ticket["source_ids"]]
    haystack = "\n".join([ticket["subject"], ticket["body"]]
                         + [finding["content"] for finding in evidence]).casefold()
    candidates = []
    for position, rule in enumerate(routing["rules"]):
        matches = [word for word in rule["keywords"] if word.casefold() in haystack]
        if matches:
            candidates.append((rule["priority"], position, rule, matches))
    if candidates:
        _, _, selected, matches = min(candidates, key=lambda row: row[:2])
    else:
        selected, matches = routing["fallback"], []
    return {
        "id": ticket["id"], "subject": ticket["subject"],
        "route": {key: selected[key] for key in ("category", "priority", "team", "owner")},
        "matched_keywords": matches,
        "citations": [{"source_id": f["source_id"], "url": f["url"], "sha256": f["sha256"]}
                      for f in evidence],
    }


def triage(request, researched):
    validate("input", request)
    validate("research", researched, request)
    index = {finding["source_id"]: finding for finding in researched["findings"]}
    output = {
        "status": "ok", "schema_version": 1, "synthetic": True, "research": researched,
        "tickets": [classify(ticket, index, request["routing"]) for ticket in request["tickets"]],
    }
    return validate("output", output, request)


def run_pipeline(request):
    validate("input", request)
    return triage(request, research(request))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            request = json.load(handle, object_pairs_hook=unique_object)
        result = run_pipeline(request)
    except (ValidationError, OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
