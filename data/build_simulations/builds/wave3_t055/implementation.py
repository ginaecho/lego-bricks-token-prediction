"""Synthetic, offline support -> interests -> research reference pipeline.

All stages use the same versioned envelope and validate their incoming and outgoing
state. Research ingests supplied page snapshots, never fetching a live URL.
Critical identifiers are replaced before any customer-facing stage is constructed.
These safety/protection rules are demonstrations, not compliance certification.
"""

import copy
import csv
import hashlib
import io
import ipaddress
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit


SCHEMA_VERSION = 1
MAX_FILE_BYTES = 1_000_000
SAFETY_RULES = {
    "P1": "life_safety or critical_service",
    "P2": "otherwise customers_affected >= 100",
    "P3": "otherwise",
}
STAGES = ("input", "support", "interests", "web")
STOP_WORDS = set("a an and are can do for how i in is it my of on or the to with".split())


class ValidationError(ValueError):
    """A deliberately non-sensitive validation failure."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), label + ": invalid fields")


def text(value, label, limit=20000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            label + ": expected bounded nonempty text")
    require(not any(ord(c) < 32 and c not in "\n\r\t" for c in value),
            label + ": control characters forbidden")
    return value


def sequence(value, label, limit=100):
    require(isinstance(value, list) and len(value) <= limit, label + ": invalid list")
    return value


def number(value, label, minimum=0, maximum=1e9, integer=False):
    require(type(value) in (int, float) and math.isfinite(value),
            label + ": expected finite number")
    require(minimum <= value <= maximum and (not integer or type(value) is int),
            label + ": number out of range")
    return value


def boolean(value, label):
    require(type(value) is bool, label + ": expected boolean")


def timestamp(value):
    text(value, "timestamp", 64)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("timestamp: invalid ISO 8601 time") from None
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "timestamp: timezone required")
    return parsed


def tags(value, label):
    sequence(value, label, 30)
    for tag in value:
        text(tag, label, 60)
        require(re.fullmatch(r"[a-z][a-z0-9_-]*", tag) is not None,
                label + ": invalid tag")
    require(len(value) == len(set(value)), label + ": duplicate tag")


def unique_ids(items, label):
    sequence(items, label)
    ids = []
    for item in items:
        require(isinstance(item, dict) and "id" in item, label + ": missing ID")
        text(item["id"], label + " ID", 100)
        ids.append(item["id"])
    require(len(ids) == len(set(ids)), label + ": duplicate ID")
    return set(ids)


def canonical_url(value):
    text(value, "URL", 2000)
    try:
        parts = urlsplit(value)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise ValidationError("URL: malformed") from None
    require(parts.scheme.lower() == "https" and bool(host), "URL: HTTPS host required")
    require(parts.username is None and parts.password is None and port in (None, 443),
            "URL: credentials and nonstandard ports forbidden")
    require(not parts.fragment and not re.search(r"\s|\\", value),
            "URL: fragments, whitespace and backslashes forbidden")
    require(re.fullmatch(r"[a-zA-Z0-9.-]+", host) is not None and "." in host
            and not host.endswith(".") and not host.endswith(".localhost"),
            "URL: invalid public-style host")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValidationError("URL: IP literals forbidden")
    decoded = unquote(parts.path)
    require(not any(part in (".", "..") for part in decoded.split("/"))
            and not re.search(r"[\x00-\x20\\]", decoded), "URL: unsafe path")
    return urlunsplit(("https", host.lower(), parts.path or "/", parts.query, ""))


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOP_WORDS


def priority(report):
    if report["life_safety"] or report["critical_service"]:
        return "P1"
    return "P2" if report["customers_affected"] >= 100 else "P3"


def parse_meter_csv(raw, asset_ids):
    text(raw, "meter_interval_csv", 100000)
    columns = ["meter_id", "timestamp", "interval_minutes", "consumption_kwh", "season", "asset_id"]
    try:
        reader = csv.DictReader(io.StringIO(raw), strict=True)
        require(reader.fieldnames == columns, "meter CSV: unexpected headers")
        records = list(reader)
    except csv.Error:
        raise ValidationError("meter CSV: malformed") from None
    require(0 < len(records) <= 1000, "meter CSV: expected 1..1000 intervals")
    readings = []
    spans = {}
    for row in records:
        require(set(row) == set(columns) and all(isinstance(v, str) for v in row.values()),
                "meter CSV: invalid row")
        text(row["meter_id"], "meter ID", 100)
        start = timestamp(row["timestamp"])
        require(re.fullmatch(r"\d+", row["interval_minutes"]) is not None,
                "meter CSV: invalid interval")
        minutes = int(row["interval_minutes"])
        require(minutes in (15, 30, 60), "meter CSV: interval must be 15, 30 or 60 minutes")
        try:
            consumption = float(row["consumption_kwh"])
        except ValueError:
            raise ValidationError("meter CSV: invalid consumption") from None
        number(consumption, "meter consumption", maximum=10000)
        require(row["season"] in ("winter", "spring", "summer", "autumn"),
                "meter CSV: invalid season")
        require(row["asset_id"] in asset_ids, "meter CSV: unknown grid asset")
        span = (start.timestamp(), start.timestamp() + minutes * 60)
        previous = spans.setdefault(row["meter_id"], [])
        require(not any(span[0] < end and begin < span[1] for begin, end in previous),
                "meter CSV: duplicate or overlapping intervals")
        previous.append(span)
        readings.append({**row, "interval_minutes": minutes, "consumption_kwh": consumption})
    return readings


def validate_context(context):
    fields(context, ("request", "preferences", "knowledge_base", "catalog", "sources", "energy"),
           "context")
    request = context["request"]
    fields(request, ("question", "team_online"), "request")
    text(request["question"], "question", 2000)
    boolean(request["team_online"], "team_online")
    prefs = context["preferences"]
    fields(prefs, ("weights", "excluded_ids", "excluded_tags", "limit"), "preferences")
    require(isinstance(prefs["weights"], dict) and len(prefs["weights"]) <= 30,
            "preferences: invalid weights")
    tags(list(prefs["weights"]), "preference tags")
    for weight in prefs["weights"].values():
        number(weight, "preference weight", maximum=10)
    tags(prefs["excluded_tags"], "excluded tags")
    sequence(prefs["excluded_ids"], "excluded IDs")
    for excluded in prefs["excluded_ids"]:
        text(excluded, "excluded ID", 100)
    require(len(prefs["excluded_ids"]) == len(set(prefs["excluded_ids"])),
            "preferences: duplicate exclusions")
    number(prefs["limit"], "recommendation limit", maximum=20, integer=True)

    kb = context["knowledge_base"]
    kb_ids = unique_ids(kb, "knowledge base")
    for item in kb:
        fields(item, ("id", "title", "text", "tags", "url"), "knowledge")
        text(item["title"], "knowledge title", 200)
        text(item["text"], "knowledge text")
        tags(item["tags"], "knowledge tags")
        canonical_url(item["url"])
    catalog = context["catalog"]
    unique_ids(catalog, "catalog")
    for item in catalog:
        fields(item, ("id", "title", "tags", "knowledge_ids", "url"), "catalog item")
        text(item["title"], "catalog title", 200)
        tags(item["tags"], "catalog tags")
        sequence(item["knowledge_ids"], "catalog knowledge IDs")
        require(item["knowledge_ids"] and all(isinstance(k, str) and k in kb_ids
                                             for k in item["knowledge_ids"]),
                "catalog: missing grounding")
        require(len(item["knowledge_ids"]) == len(set(item["knowledge_ids"])),
                "catalog: duplicate grounding")
        canonical_url(item["url"])
    source = context["sources"]
    fields(source, ("allowlisted_urls", "documents"), "sources")
    sequence(source["allowlisted_urls"], "URL allowlist")
    allowed = [canonical_url(url) for url in source["allowlisted_urls"]]
    require(len(allowed) == len(set(allowed)), "URL allowlist: duplicate")
    sequence(source["documents"], "documents")
    document_urls = []
    for doc in source["documents"]:
        fields(doc, ("url", "title", "content", "source_sha256"), "document")
        url = canonical_url(doc["url"])
        require(url in allowed, "document: URL is not allowlisted")
        document_urls.append(url)
        text(doc["title"], "document title", 200)
        text(doc["content"], "document content")
        require(doc["source_sha256"] is None or (
            isinstance(doc["source_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", doc["source_sha256"]) is not None),
            "document: invalid source digest")
    require(len(document_urls) == len(set(document_urls)), "documents: duplicate URL")

    energy = context["energy"]
    fields(energy, ("meter_interval_csv", "meter_readings", "grid_assets", "scada_telemetry",
                    "outage_reports", "emissions", "safety_rules"), "energy")
    require(energy["safety_rules"] == SAFETY_RULES, "energy: safety rules must match declared rules")
    assets = energy["grid_assets"]
    asset_ids = unique_ids(assets, "grid assets")
    for asset in assets:
        fields(asset, ("id", "kind", "critical", "connected_to"), "grid asset")
        require(re.fullmatch(r"(SYN-[A-Z0-9-]{5,80}|ASSET-\d{3})", asset["id"]) is not None,
                "grid asset: synthetic or protected ID required")
        require(asset["kind"] in ("substation", "feeder", "transformer"), "grid asset: invalid kind")
        boolean(asset["critical"], "critical")
        sequence(asset["connected_to"], "topology edges")
        require(all(isinstance(link, str) and link in asset_ids and link != asset["id"]
                    for link in asset["connected_to"]), "topology: invalid endpoint")
        require(len(asset["connected_to"]) == len(set(asset["connected_to"])),
                "topology: duplicate edge")
    readings = parse_meter_csv(energy["meter_interval_csv"], asset_ids)
    require(isinstance(energy["meter_readings"], list), "meter_readings: expected list")
    telemetry = energy["scada_telemetry"]
    fields(telemetry, ("protocol", "sampled_at", "points"), "SCADA")
    require(telemetry["protocol"] == "SCADA-SYNTHETIC-1", "SCADA: unexpected protocol")
    timestamp(telemetry["sampled_at"])
    sequence(telemetry["points"], "SCADA points")
    for point in telemetry["points"]:
        fields(point, ("asset_id", "metric", "value", "unit", "quality"), "SCADA point")
        require(isinstance(point["asset_id"], str) and point["asset_id"] in asset_ids,
                "SCADA: unknown asset")
        require(isinstance(point["metric"], str), "SCADA: invalid metric")
        require({"voltage": "kV", "load": "MW", "frequency": "Hz"}.get(point["metric"])
                == point["unit"] and point["unit"] is not None, "SCADA: metric/unit mismatch")
        number(point["value"], "SCADA value")
        require(point["quality"] in ("good", "uncertain", "bad"), "SCADA: invalid quality")
    unique_ids(energy["outage_reports"], "outage reports")
    for report in energy["outage_reports"]:
        fields(report, ("id", "asset_id", "reported_at", "life_safety", "critical_service",
                        "customers_affected", "priority", "description"), "outage report")
        require(isinstance(report["asset_id"], str) and report["asset_id"] in asset_ids,
                "outage: unknown asset")
        timestamp(report["reported_at"])
        boolean(report["life_safety"], "life_safety")
        boolean(report["critical_service"], "critical_service")
        number(report["customers_affected"], "customers_affected", integer=True)
        text(report["description"], "outage description", 2000)
        require(report["priority"] == priority(report), "outage: safety priority mismatch")
    unique_ids(energy["emissions"], "emissions")
    for emission in energy["emissions"]:
        fields(emission, ("id", "value", "unit", "source"), "emissions figure")
        number(emission["value"], "emissions value")
        require(emission["unit"] in ("kgCO2e", "tCO2e", "gCO2e/kWh"),
                "emissions: explicit supported unit required")
        require(isinstance(emission["source"], str) and emission["source"] in kb_ids,
                "emissions: known knowledge source required")
    return readings


def protected_context(context):
    result = copy.deepcopy(context)
    for doc in result["sources"]["documents"]:
        doc["source_sha256"] = hashlib.sha256(doc["content"].encode("utf-8")).hexdigest()
    # Protect every grid ID, not just assets marked critical. Keep topology referential.
    identifiers = sorted(asset["id"] for asset in result["energy"]["grid_assets"])
    require(all(identifier.startswith("SYN-") for identifier in identifiers),
            "input: grid IDs must be synthetic, not pre-protected")
    mapping = {identifier: "ASSET-%03d" % (i + 1) for i, identifier in enumerate(identifiers)}
    pattern = re.compile("|".join(re.escape(i) for i in sorted(identifiers, key=len, reverse=True)),
                         re.IGNORECASE) if identifiers else None
    lookup = {key.lower(): value for key, value in mapping.items()}

    def walk(value):
        if isinstance(value, str):
            if not pattern:
                return value
            # Encoded asset IDs cannot become a public provenance URL.
            decoded = value
            for _ in range(4):
                decoded = unquote(decoded)
            if pattern.search(decoded) and re.search(r"https?://", value, re.IGNORECASE):
                raise ValidationError("protection: asset identifier in URL-bearing text")
            if pattern.search(decoded):
                return pattern.sub(lambda match: lookup[match.group(0).lower()], decoded)
            return value
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {walk(key): walk(item) for key, item in value.items()}
        return value

    result = walk(result)
    result["energy"]["meter_readings"] = parse_meter_csv(
        result["energy"]["meter_interval_csv"],
        {asset["id"] for asset in result["energy"]["grid_assets"]})
    return result


def build_support(context):
    query = tokens(context["request"]["question"])
    candidates = []
    for item in context["knowledge_base"]:
        overlap = query & tokens(" ".join([item["title"], item["text"], *item["tags"]]))
        if overlap:
            candidates.append((len(overlap), item))
    matched = [item for _, item in sorted(candidates, key=lambda x: (-x[0], x[1]["id"]))[:3]]
    energy = context["energy"]
    outages = [{"outage_id": report["id"], "priority": report["priority"],
                "rule": SAFETY_RULES[report["priority"]]}
               for report in sorted(energy["outage_reports"], key=lambda r: (r["priority"], r["id"]))]
    evidence = [{"knowledge_id": item["id"], "url": canonical_url(item["url"]),
                 "excerpt": item["text"][:600]} for item in matched]
    answer = "\n".join("[" + item["knowledge_id"] + "] " + item["excerpt"] for item in evidence)
    if not matched:
        answer = "The supplied knowledge does not establish an answer. A human follow-up is needed."
    if any(item["priority"] == "P1" for item in outages):
        answer += ("\n[declared-safety-rules:P1] Immediate safety concern: stay away from damaged "
                   "equipment and contact local emergency services if anyone is in danger.")
    return {
        "answer": answer,
        "matched_knowledge_ids": [item["id"] for item in matched],
        "intent_tags": sorted({tag for item in matched for tag in item["tags"]}),
        "evidence": evidence,
        "handoff": {
            "needed": not matched or any(item["priority"] == "P1" for item in outages),
            "channel": "live_queue" if context["request"]["team_online"] else "async_ticket",
            "reason": "unanswered" if not matched else ("safety" if any(
                item["priority"] == "P1" for item in outages) else "grounded_answer"),
        },
        "outage_priorities": outages,
        "meter_summary": {
            "interval_count": len(energy["meter_readings"]),
            "consumption": round(sum(row["consumption_kwh"] for row in energy["meter_readings"]), 6),
            "unit": "kWh",
            "source": "energy.meter_readings",
            "seasons": sorted({row["season"] for row in energy["meter_readings"]}),
        },
        "emissions": copy.deepcopy(energy["emissions"]),
    }


def build_interests(context, support):
    prefs = context["preferences"]
    knowledge = {item["id"]: item for item in context["knowledge_base"]}
    recommendations = []
    for item in context["catalog"]:
        if item["id"] in prefs["excluded_ids"] or set(item["tags"]) & set(prefs["excluded_tags"]):
            continue
        weighted = {tag: prefs["weights"][tag] for tag in sorted(item["tags"])
                    if prefs["weights"].get(tag, 0) > 0}
        support_tags = sorted(set(item["tags"]) & set(support["intent_tags"]))
        shared_evidence = sorted(set(item["knowledge_ids"]) & set(support["matched_knowledge_ids"]))
        score = round(sum(weighted.values()) + 2 * len(support_tags) + len(shared_evidence), 6)
        if score <= 0:
            continue
        urls = {canonical_url(item["url"])}
        urls.update(canonical_url(knowledge[k]["url"]) for k in item["knowledge_ids"])
        recommendations.append({
            "id": item["id"], "title": item["title"], "score": score,
            "explanation": {
                "preference_contributions": weighted,
                "support_tags": support_tags,
                "support_tag_weight": 2,
                "shared_support_knowledge_ids": shared_evidence,
                "shared_knowledge_weight": 1,
                "grounding": [{"knowledge_id": k, "excerpt": knowledge[k]["text"][:600],
                              "url": canonical_url(knowledge[k]["url"])}
                             for k in sorted(item["knowledge_ids"])],
            },
            "research_urls": sorted(urls),
            "research_terms": sorted(tokens(context["request"]["question"])
                                     | set(item["tags"]) | set(support_tags)),
        })
    recommendations.sort(key=lambda item: (-item["score"], item["id"]))
    return {
        "consumed_support_knowledge_ids": list(support["matched_knowledge_ids"]),
        "recommendations": recommendations[:prefs["limit"]],
        "exclusions_applied": {"ids": list(prefs["excluded_ids"]), "tags": list(prefs["excluded_tags"])},
    }


def build_web(context, interests):
    sources = context["sources"]
    allowed = {canonical_url(url) for url in sources["allowlisted_urls"]}
    docs = {canonical_url(doc["url"]): doc for doc in sources["documents"]}
    findings, blocked, missing, irrelevant = [], set(), set(), set()
    for item in interests["recommendations"]:
        for url in item["research_urls"]:
            if url not in allowed:
                blocked.add(url)
                continue
            if url not in docs:
                missing.add(url)
                continue
            doc = docs[url]
            overlap = sorted(set(item["research_terms"]) & tokens(doc["content"]))
            if not overlap:
                irrelevant.add(url)
                continue
            # Quote a passage that actually contains a retrieval term.
            match = next(re.finditer(r"[a-z0-9]+", doc["content"], re.IGNORECASE))
            for possible in re.finditer(r"[a-z0-9]+", doc["content"], re.IGNORECASE):
                if possible.group(0).lower() in overlap:
                    match = possible
                    break
            offset = max(0, match.start() - 80)
            quote = doc["content"][offset:offset + 400]
            findings.append({
                "recommendation_id": item["id"], "url": url, "title": doc["title"],
                "quote": quote, "quote_offset": offset,
                "sha256": hashlib.sha256(doc["content"].encode("utf-8")).hexdigest(),
                "source_sha256": doc["source_sha256"],
                "matched_terms": overlap,
                "provenance": "synthetic supplied snapshot; identifier-protected quote; no live fetch",
            })
    return {
        "consumed_recommendation_ids": [item["id"] for item in interests["recommendations"]],
        "findings": findings, "blocked_urls": sorted(blocked), "missing_urls": sorted(missing),
        "irrelevant_urls": sorted(irrelevant), "network_used": False,
    }


def validate(state, expected_stage=None):
    fields(state, ("schema_version", "synthetic", "status", "stage", "context",
                   "support", "interests", "web"), "envelope")
    require(type(state["schema_version"]) is int and state["schema_version"] == SCHEMA_VERSION,
            "envelope: unsupported schema version")
    require(state["synthetic"] is True and state["status"] == "ok",
            "envelope: synthetic fixture and ok status required")
    require(state["stage"] in STAGES, "envelope: invalid stage")
    if expected_stage is not None:
        require(state["stage"] == expected_stage, "envelope: invalid handoff order")
    readings = validate_context(state["context"])
    stage = STAGES.index(state["stage"])
    if stage == 0:
        require(all(doc["source_sha256"] is None
                    for doc in state["context"]["sources"]["documents"]),
                "input: source digests must initially be null")
        require(state["context"]["energy"]["meter_readings"] == [],
                "input: normalized readings must initially be empty")
        require(all(asset["id"].startswith("SYN-")
                    for asset in state["context"]["energy"]["grid_assets"]),
                "input: synthetic raw grid IDs required")
    else:
        require(all(doc["source_sha256"] is not None
                    for doc in state["context"]["sources"]["documents"]),
                "handoff: source digest required")
        require(state["context"]["energy"]["meter_readings"] == readings,
                "handoff: meter readings do not match interval CSV")
        require(all(re.fullmatch(r"ASSET-\d{3}", asset["id"]) is not None
                    for asset in state["context"]["energy"]["grid_assets"]),
                "handoff: unprotected grid IDs")
    for index, name in enumerate(STAGES[1:], 1):
        if stage < index:
            require(state[name] is None, "envelope: future stage must be null")
        else:
            if name == "support":
                expected = build_support(state["context"])
            elif name == "interests":
                expected = build_interests(state["context"], state["support"])
            else:
                expected = build_web(state["context"], state["interests"])
            require(state[name] == expected, "handoff: invalid " + name + " output")
    return state


def support_stage(state):
    validate(state, "input")
    result = copy.deepcopy(state)
    result["context"] = protected_context(state["context"])
    result["support"] = build_support(result["context"])
    result["stage"] = "support"
    return validate(result, "support")


def interests_stage(state):
    validate(state, "support")
    result = copy.deepcopy(state)
    result["interests"] = build_interests(result["context"], result["support"])
    result["stage"] = "interests"
    return validate(result, "interests")


def web_stage(state):
    validate(state, "interests")
    result = copy.deepcopy(state)
    result["web"] = build_web(result["context"], result["interests"])
    result["stage"] = "web"
    return validate(result, "web")


def run_pipeline(state):
    return web_stage(interests_stage(support_stage(state)))


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON: duplicate key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("JSON: non-finite number forbidden")


def load_input(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    require(len(raw) <= MAX_FILE_BYTES, "input: file exceeds size limit")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=reject_constant)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        result = run_pipeline(load_input(args[0]))
    except ValidationError as exc:
        result = {"status": "error", "error": str(exc)}
        code = 2
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, TypeError, OverflowError):
        result = {"status": "error", "error": "Invalid or unreadable input file"}
        code = 2
    else:
        code = 0
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
