"""Synthetic, offline adaptive-onboarding -> provenance-preserving research CLI.

Run: python -B implementation.py example_input.json
Only fixture documents are ingested; this module never performs network I/O.
"""

import ipaddress
import json
import re
import sys
from urllib.parse import urlsplit, urlunsplit


class ValidationError(ValueError):
    pass


STEPS = ("basics", "source_evaluation", "research")
PREREQUISITES = {
    "basics": [],
    "source_evaluation": ["basics"],
    "research": ["source_evaluation"],
}
EXPLANATIONS = {
    "basics": "Learn how a research question becomes searchable terms.",
    "source_evaluation": "Check the allowlist and distinguish evidence from claims.",
    "research": "Retrieve matching passages and retain their source attribution.",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(expected), location + " has missing or unknown fields")


def text(value, location, limit=20000):
    require(isinstance(value, str) and bool(value.strip()), location + " must be nonempty text")
    require(len(value) <= limit, location + " is too long")


def host(value):
    text(value, "allowlisted host", 253)
    require(value == value.lower(), "allowlisted hosts must be lowercase")
    require(re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value) is not None,
            "invalid host")
    labels = value.split(".")
    require(len(labels) >= 2 and all(
        1 <= len(label) <= 63 and not label.startswith("-") and not label.endswith("-")
        for label in labels), "invalid domain labels")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise ValidationError("IP literal hosts are not supported")
    return value


def canonical_url(value, allowlist):
    text(value, "source URL", 2048)
    require(not any(c.isspace() or ord(c) < 32 for c in value) and "\\" not in value,
            "URL contains forbidden characters")
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
    except ValueError as exc:
        raise ValidationError("malformed URL") from exc
    require(parts.scheme == "https" and hostname in allowlist, "URL is not allowlisted HTTPS")
    require(parts.netloc.lower() == hostname, "credentials and explicit ports are forbidden")
    require(not parts.fragment, "URL fragments are forbidden")
    return urlunsplit(("https", hostname, parts.path or "/", parts.query, ""))


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def validate(stage, value):
    """Shared validation entry point for input and both inter-stage/output schemas."""
    if stage == "input":
        fields(value, ("schema_version", "synthetic", "profile", "research"), "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        require(value["synthetic"] is True, "only explicitly synthetic fixtures are supported")
        profile = value["profile"]
        fields(profile, ("experience", "preference", "topic", "completed"), "profile")
        require(profile["experience"] in ("novice", "intermediate", "expert"),
                "invalid experience")
        require(profile["preference"] in ("brief", "detailed"), "invalid preference")
        text(profile["topic"], "topic", 300)
        require(bool(tokens(profile["topic"])), "topic requires searchable terms")
        completed = profile["completed"]
        require(isinstance(completed, list) and all(isinstance(s, str) and s in STEPS
                for s in completed), "invalid completed steps")
        require(len(set(completed)) == len(completed), "duplicate completed steps")
        credits = credited_steps(profile["experience"])
        for step in completed:
            require(set(PREREQUISITES[step]) <= set(completed) | set(credits),
                    "completed step is missing a prerequisite")
        research = value["research"]
        fields(research, ("allowlisted_hosts", "sources", "fixture_documents", "max_results"),
               "research")
        hosts = research["allowlisted_hosts"]
        require(isinstance(hosts, list) and 0 < len(hosts) <= 20, "invalid allowlist")
        for item in hosts:
            host(item)
        require(len(set(hosts)) == len(hosts), "duplicate hosts")
        sources = research["sources"]
        require(isinstance(sources, list) and 0 < len(sources) <= 50, "invalid sources")
        normalized = [canonical_url(url, hosts) for url in sources]
        require(len(set(normalized)) == len(normalized), "duplicate canonical source URL")
        docs = research["fixture_documents"]
        require(isinstance(docs, dict) and set(docs) == set(normalized),
                "fixture document keys must exactly match canonical sources")
        for url, document in docs.items():
            fields(document, ("fixture_id", "title", "text"), "fixture document")
            for key in document:
                text(document[key], "document " + key)
        ids = [doc["fixture_id"] for doc in docs.values()]
        require(len(ids) == len(set(ids)), "fixture IDs must be unique")
        limit = research["max_results"]
        require(type(limit) is int and 1 <= limit <= 20, "max_results must be 1..20")
    elif stage == "onboarding":
        fields(value, ("schema_version", "synthetic", "query", "preference",
                       "credited", "already_completed", "steps", "ready", "retrieval"),
               "onboarding")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1
                and value["synthetic"] is True, "invalid onboarding schema")
        text(value["query"], "onboarding query", 300)
        require(bool(tokens(value["query"])), "query requires searchable terms")
        require(value["preference"] in ("brief", "detailed"), "invalid preference")
        for key in ("credited", "already_completed"):
            require(isinstance(value[key], list) and all(
                isinstance(s, str) and s in STEPS for s in value[key]), "invalid step credit")
            require(len(set(value[key])) == len(value[key]), "duplicate step credit")
        done = set(value["credited"]) | set(value["already_completed"])
        for step in done:
            require(set(PREREQUISITES[step]) <= done, "invalid prerequisite credit")
        require(isinstance(value["steps"], list), "steps must be a list")
        for step in value["steps"]:
            fields(step, ("id", "prerequisites", "explanation"), "step")
            name = step["id"]
            require(isinstance(name, str) and name in STEPS and name not in done,
                    "invalid or duplicate planned step")
            require(step["prerequisites"] == PREREQUISITES[name]
                    and set(PREREQUISITES[name]) <= done, "unmet planned prerequisite")
            text(step["explanation"], "step explanation")
            done.add(name)
        require(value["ready"] is True and set(STEPS) <= done, "incomplete onboarding plan")
        # Reuse the input contract to validate the entire retrieval handoff.
        validate("input", {
            "schema_version": 1, "synthetic": True,
            "profile": {"experience": "novice", "preference": value["preference"],
                        "topic": value["query"], "completed": []},
            "research": value["retrieval"],
        })
    elif stage == "output":
        fields(value, ("schema_version", "synthetic", "status", "onboarding", "research"),
               "output")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1
                and value["synthetic"] is True and value["status"] == "ok",
                "invalid output envelope")
        validate("onboarding", value["onboarding"])
        research = value["research"]
        fields(research, ("query", "findings", "ingested_sources", "unmatched_sources"),
               "research output")
        plan = value["onboarding"]
        require(research["query"] == plan["query"], "research query changed")
        urls = plan["retrieval"]["sources"]
        require(research["ingested_sources"] == urls, "source lineage changed")
        expected = retrieve(plan)
        require(research == expected, "findings or provenance do not match validated fixtures")
    else:
        raise ValidationError("unknown validation stage")
    return value


def credited_steps(experience):
    return list(STEPS[:{"novice": 0, "intermediate": 1, "expert": 2}[experience]])


def adaptive_onboarding(payload):
    validate("input", payload)
    profile = payload["profile"]
    credits = credited_steps(profile["experience"])
    done = set(credits) | set(profile["completed"])
    steps = []
    for name in STEPS:
        if name not in done:
            explanation = EXPLANATIONS[name]
            if profile["preference"] == "detailed":
                explanation += " Prerequisites: " + (
                    ", ".join(PREREQUISITES[name]) or "none") + "."
            steps.append({"id": name, "prerequisites": list(PREREQUISITES[name]),
                          "explanation": explanation})
            done.add(name)
    retrieval = json.loads(json.dumps(payload["research"]))
    retrieval["sources"] = [
        canonical_url(url, retrieval["allowlisted_hosts"]) for url in retrieval["sources"]]
    return validate("onboarding", {
        "schema_version": 1, "synthetic": True, "query": profile["topic"].strip(),
        "preference": profile["preference"], "credited": credits,
        "already_completed": list(profile["completed"]), "steps": steps, "ready": True,
        "retrieval": retrieval,
    })


def retrieve(plan):
    query_tokens = tokens(plan["query"])
    settings = plan["retrieval"]
    findings = []
    unmatched = []
    for url in settings["sources"]:
        doc = settings["fixture_documents"][url]
        passages = []
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]|(?=\n|$))", doc["text"]):
            raw = match.group()
            quote = raw.strip()
            score = len(query_tokens & tokens(quote))
            if quote and score:
                start = match.start() + len(raw) - len(raw.lstrip())
                passages.append({
                    "score": score, "quote": quote, "start": start,
                    "end": start + len(quote), "url": url, "title": doc["title"],
                    "fixture_id": doc["fixture_id"], "synthetic": True,
                })
        passages.sort(key=lambda p: (-p["score"], p["start"]))
        if not passages:
            unmatched.append(url)
        findings.extend(passages[:1] if plan["preference"] == "brief" else passages)
    findings.sort(key=lambda p: (-p["score"], settings["sources"].index(p["url"]), p["start"]))
    return {
        "query": plan["query"], "findings": findings[:settings["max_results"]],
        "ingested_sources": list(settings["sources"]), "unmatched_sources": unmatched,
    }


def web_research(onboarding):
    validate("onboarding", onboarding)
    return retrieve(onboarding)


def run(payload):
    onboarding = adaptive_onboarding(payload)
    result = {"schema_version": 1, "synthetic": True, "status": "ok",
              "onboarding": onboarding, "research": web_research(onboarding)}
    return validate("output", result)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=reject_duplicate_keys)
        result = run(payload)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
