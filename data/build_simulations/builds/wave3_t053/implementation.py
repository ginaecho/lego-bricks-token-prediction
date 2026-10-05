"""Synthetic, deterministic search -> support -> research reference pipeline.

Python 3 standard library only. No live systems, providers, or network access.
Critical identifiers are replaced with run-local aliases, not encryption. All
input source statements are treated as supplied evidence, not verified truth.
"""

import copy
import csv
import difflib
import io
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


VERSION = "1.0"
PRIORITY_RULES = {
    "life_safety": "P1",
    "critical_service": "P2",
    "widespread": "P2",
    "routine": "P3",
    "widespread_customer_threshold": 1000,
}
FIXTURE_SOURCES = {
    "fixture-meter": "Synthetic smart-meter interval CSV",
    "fixture-grid": "Synthetic grid topology and SCADA telemetry",
    "fixture-outage": "Synthetic outage reports and declared safety rules",
}
SYNONYMS = (
    {"bill", "billing", "cost", "price", "pricing"},
    {"usage", "consumption", "energy"},
    {"outage", "blackout", "interruption", "broken"},
    {"efficient", "efficiency", "saving", "savings"},
    {"cheap", "affordable", "budget", "inexpensive"},
    {"phone", "mobile", "smartphone"},
    {"laptop", "notebook"},
    {"solar", "renewable"},
)
STOP_WORDS = {"a", "an", "the", "to", "and", "for", "my", "is", "of", "i",
              "what", "how", "with", "can", "me", "in", "on", "please"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, name):
    require(isinstance(value, dict), name + " must be an object")
    return value


def seq(value, name, nonempty=False):
    require(isinstance(value, list), name + " must be an array")
    require(not nonempty or bool(value), name + " must not be empty")
    return value


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    require(len(value) <= 100000, name + " is too long")
    return value.strip()


def number(value, name, minimum=0, maximum=None, integer=False):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite numeric")
    require(not integer or type(value) is int, name + " must be an integer")
    require(value >= minimum and (maximum is None or value <= maximum), name + " out of range")
    return value


def boolean(value, name):
    require(type(value) is bool, name + " must be boolean")
    return value


def timestamp(value):
    value = text(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("invalid ISO 8601 timestamp") from None
    require(parsed.tzinfo is not None, "timestamp must include timezone")
    return value, parsed


def keyed(items, name):
    result = {}
    for item in seq(items, name):
        obj(item, name + " item")
        identifier = text(item.get("id"), name + " id")
        require(identifier not in result, "duplicate " + name + " id")
        result[identifier] = item
    return result


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOP_WORDS


def expanded(value):
    terms = tokens(value)
    for group in SYNONYMS:
        if terms & group:
            terms |= group
    return terms


def relevance(query, document):
    terms, words = expanded(query), tokens(document)
    return sum(3 if term in words else
               int(any(difflib.SequenceMatcher(None, term, word).ratio() >= .84
                       for word in words if len(term) >= 4 and len(word) >= 4))
               for term in terms)


def priority(outage, rules):
    if outage["life_safety"] or outage["downed_line"]:
        return "P1"
    if outage["critical_service"] or outage["customers_affected"] >= rules["widespread_customer_threshold"]:
        return "P2"
    return "P3"


def normalize(raw, canonical=False):
    obj(raw, "input")
    require(raw.get("schema_version") == VERSION, "unsupported schema_version")
    require(raw.get("synthetic") is True, "this reference accepts clearly labeled synthetic data only")
    queries = {name: text(raw.get(name), name)
               for name in ("query", "support_question", "research_question")}
    top_k = number(raw.get("top_k", 5), "top_k", 1, 10, integer=True)
    sources = keyed(raw.get("sources"), "sources")
    require(not set(sources) & set(FIXTURE_SOURCES), "reserved source id")
    for source in sources.values():
        text(source.get("title"), "source title")
        text(source.get("text"), "source text")
    source_records = [{"id": key, "title": value["title"], "text": value["text"]}
                      for key, value in sources.items()]
    source_records += [{"id": key, "title": title, "text": title}
                       for key, title in FIXTURE_SOURCES.items()]

    assets = keyed(raw.get("grid_assets"), "grid assets")
    require(bool(assets), "at least one grid asset required")
    aliases = {}
    for index, identifier in enumerate(sorted(assets), 1):
        aliases[identifier] = "asset-%03d" % index
    for identifier in sorted(assets):
        asset = assets[identifier]
        station = text(asset.get("substation_id"), "substation id")
        require(station not in assets, "substation and asset identifiers must differ")
        if station not in aliases:
            aliases[station] = "station-%03d" % (1 + sum(v.startswith("station-") for v in aliases.values()))
        text(asset.get("kind"), "asset kind")
        boolean(asset.get("critical"), "asset critical")
    topology = []
    for edge in seq(raw.get("topology"), "topology"):
        require(isinstance(edge, list) and len(edge) == 2, "topology edge requires two endpoints")
        require(all(isinstance(x, str) and x in assets for x in edge), "unknown topology endpoint")
        require(edge[0] != edge[1], "self-loop topology edge")
        topology.append([aliases[x] for x in edge])
    rules = obj(raw.get("safety_rules"), "safety_rules")
    require(set(rules) == set(PRIORITY_RULES), "declare all safety rules")
    for key, value in PRIORITY_RULES.items():
        if key == "widespread_customer_threshold":
            number(rules[key], key, 1, integer=True)
        else:
            require(rules[key] == value, "unsafe priority rule")

    intervals, seen = [], set()
    csv_text = text(raw.get("meter_interval_csv"), "meter_interval_csv")
    try:
        reader = csv.DictReader(io.StringIO(csv_text), strict=True)
        require(reader.fieldnames == ["meter_id", "timestamp", "kwh"],
                "CSV header must be meter_id,timestamp,kwh")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "malformed CSV row")
            meter = text(row["meter_id"], "meter id")
            require(meter not in assets and meter not in
                    [a["substation_id"] for a in assets.values()], "meter identifier collides with grid")
            if meter not in aliases:
                aliases[meter] = "meter-%03d" % (1 + sum(v.startswith("meter-") for v in aliases.values()))
            when, parsed = timestamp(row["timestamp"])
            key = (meter, parsed)
            require(key not in seen, "duplicate meter interval")
            seen.add(key)
            try:
                kwh = float(row["kwh"])
            except ValueError:
                raise ValidationError("invalid interval kwh") from None
            number(kwh, "interval kwh")
            intervals.append({"meter_ref": aliases[meter], "timestamp": when, "kwh": kwh})
    except csv.Error:
        raise ValidationError("invalid interval CSV") from None
    require(bool(intervals), "meter CSV requires readings")

    telemetry = []
    scada = obj(raw.get("telemetry"), "telemetry")
    require(scada.get("format") == "scada-style-v1", "unsupported telemetry format")
    seen_telemetry = set()
    for sample in seq(scada.get("samples"), "telemetry samples", True):
        obj(sample, "telemetry sample")
        identifier = text(sample.get("asset_id"), "telemetry asset")
        require(identifier in assets, "telemetry references unknown asset")
        when, parsed = timestamp(sample.get("timestamp"))
        require((identifier, parsed) not in seen_telemetry, "duplicate telemetry sample")
        seen_telemetry.add((identifier, parsed))
        require(sample.get("status") in ("normal", "warning", "offline"), "invalid telemetry status")
        telemetry.append({
            "asset_ref": aliases[identifier], "timestamp": when,
            "voltage_kv": number(sample.get("voltage_kv"), "voltage_kv", 0, 1000),
            "load_pct": number(sample.get("load_pct"), "load_pct", 0, 200),
            "status": sample["status"],
        })

    outages = []
    for identifier, outage in keyed(raw.get("outage_reports"), "outages").items():
        asset = text(outage.get("asset_id"), "outage asset")
        require(asset in assets, "outage references unknown asset")
        record = {
            "id": identifier, "asset_ref": aliases[asset],
            "timestamp": timestamp(outage.get("timestamp"))[0],
            "customers_affected": number(outage.get("customers_affected"), "customers_affected", integer=True),
        }
        for flag in ("life_safety", "downed_line", "critical_service"):
            record[flag] = boolean(outage.get(flag), flag)
        record["priority"] = priority(record, rules)
        require(outage.get("priority") == record["priority"], "outage priority violates declared safety rules")
        outages.append(record)
    outages.sort(key=lambda item: (item["priority"], item["id"]))

    catalog = []
    for identifier, offering in keyed(raw.get("catalog"), "catalog").items():
        source = text(offering.get("source_id"), "offering source")
        require(source in sources, "unknown offering source")
        catalog.append({
            "id": identifier, "name": text(offering.get("name"), "offering name"),
            "description": text(offering.get("description"), "offering description"),
            "tags": [text(tag, "tag") for tag in seq(offering.get("tags"), "tags")],
            "source_id": source,
        })
    knowledge = []
    for identifier, item in keyed(raw.get("knowledge"), "knowledge").items():
        source = text(item.get("source_id"), "knowledge source")
        require(source in sources, "unknown knowledge source")
        statement = text(item.get("text"), "knowledge text")
        require(statement in sources[source]["text"], "knowledge must quote supplied source text")
        knowledge.append({"id": identifier, "text": statement, "source_id": source})
    emissions = []
    for item in seq(raw.get("emissions"), "emissions"):
        obj(item, "emissions figure")
        source = text(item.get("source_id"), "emissions source")
        require(source in sources, "unknown emissions source")
        require(item.get("unit") in ("kgCO2e", "tCO2e"), "emissions require kgCO2e or tCO2e units")
        emissions.append({
            "value": number(item.get("value"), "emissions value"),
            "unit": item["unit"], "source_id": source,
            "period": text(item.get("period"), "emissions period"),
        })
    context = {
        **queries, "top_k": top_k, "sources": source_records, "catalog": catalog,
        "knowledge": knowledge, "emissions": emissions, "safety_rules": dict(rules),
        "grid_assets": [{"asset_ref": aliases[key], "station_ref": aliases[value["substation_id"]],
                         "kind": value["kind"], "critical": value["critical"]}
                        for key, value in assets.items()],
        "topology": topology, "meter_readings": intervals, "telemetry": telemetry,
        "outage_reports": outages,
        "protection": "Run-local aliases; raw asset, station and meter identifiers withheld. Not CIP certification.",
    }
    if not canonical:
        require(not any(re.fullmatch(r"(asset|station|meter)-\d+", identifier) for identifier in aliases),
                "raw identifiers must not use reserved output alias syntax")
    # Redact free text too, including supplied source material and questions.
    pattern = re.compile(r"(?<![A-Za-z0-9_-])(?:" +
                         "|".join(re.escape(key) for key in sorted(aliases, key=len, reverse=True)) +
                         r")(?![A-Za-z0-9_-])")
    protected_values = set(aliases.values())
    def redact(value):
        if isinstance(value, str):
            if value in protected_values:
                return value
            return pattern.sub(lambda match: aliases[match.group()], value)
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value
    return redact(context)


def expected_matches(context):
    matches = []
    for product in context["catalog"]:
        score = (2 * relevance(context["query"], product["name"] + " " + " ".join(product["tags"]))
                 + relevance(context["query"], product["description"]))
        if score:
            matches.append({**product, "score": score})
    return sorted(matches, key=lambda item: (-item["score"], item["id"]))[:context["top_k"]]


def expected_claims(state):
    context = state["context"]
    claims = []
    def add(kind, value, citations):
        claims.append({"id": "claim-%03d" % (len(claims) + 1),
                       "kind": kind, "text": value, "citations": citations})
    for match in state["search"]["matches"][:2]:
        add("offering", match["name"] + ": " + match["description"], [match["source_id"]])
    question = context["support_question"]
    ranked = sorted(context["knowledge"],
                    key=lambda k: (-relevance(question, k["text"]), k["id"]))
    for item in [k for k in ranked if relevance(question, k["text"]) > 0][:3]:
        add("knowledge", item["text"], [item["source_id"]])
    total = math.fsum(reading["kwh"] for reading in context["meter_readings"])
    add("meter", "Observed synthetic intervals total %.3f kWh; not a full billing-period estimate." % total,
        ["fixture-meter"])
    for outage in context["outage_reports"]:
        caution = (" Stay clear of downed lines; contact local emergency services for immediate danger."
                   if outage["priority"] == "P1" else " Use the utility outage reporting channel.")
        add("outage", "%s at %s: %s, %d customers affected.%s" %
            (outage["id"], outage["asset_ref"], outage["priority"], outage["customers_affected"], caution),
            ["fixture-outage"])
    # Use only the latest sample per asset, not stale alarm states.
    latest = {}
    for sample in context["telemetry"]:
        key = sample["asset_ref"]
        if key not in latest or timestamp(sample["timestamp"])[1] > timestamp(latest[key]["timestamp"])[1]:
            latest[key] = sample
    for key in sorted(latest):
        sample = latest[key]
        if sample["status"] != "normal" or sample["load_pct"] > 100:
            add("telemetry", "%s latest synthetic telemetry: %s, %.1f%% load; operator review required." %
                (key, sample["status"], sample["load_pct"]), ["fixture-grid"])
    for figure in context["emissions"]:
        add("emissions", "Reported emissions: %s %s for %s (supplied source, not independently verified)." %
            (figure["value"], figure["unit"], figure["period"]), [figure["source_id"]])
    return claims


def validate(state, expected_stage):
    """Shared validation for every stage boundary, including deterministic provenance."""
    obj(state, "pipeline state")
    require(state.get("schema_version") == VERSION and state.get("synthetic") is True,
            "invalid envelope version or fixture label")
    require(state.get("status") == "ok" and state.get("stage") == expected_stage, "invalid stage handoff")
    required = {"schema_version", "synthetic", "status", "stage", "context", "search"}
    if expected_stage in ("support", "research"):
        required.add("support")
    if expected_stage == "research":
        required.add("research")
    require(expected_stage in ("search", "support", "research") and set(state) == required,
            "unexpected envelope fields")
    context = obj(state.get("context"), "context")
    # Re-normalize a reconstructed input to check the complete shared domain schema.
    rebuilt = {
        "schema_version": VERSION, "synthetic": True,
        **{key: context.get(key) for key in
           ("query", "support_question", "research_question", "top_k", "safety_rules",
            "catalog", "knowledge", "emissions", "topology")},
        "sources": [s for s in seq(context.get("sources"), "sources")
                    if obj(s, "source").get("id") not in FIXTURE_SOURCES],
        "grid_assets": [{"id": a.get("asset_ref"), "substation_id": a.get("station_ref"),
                         "kind": a.get("kind"), "critical": a.get("critical")}
                        for a in seq(context.get("grid_assets"), "grid_assets") if obj(a, "asset")],
        "outage_reports": [{**o, "asset_id": o.get("asset_ref")}
                           for o in seq(context.get("outage_reports"), "outages") if obj(o, "outage")],
        "telemetry": {"format": "scada-style-v1", "samples":
                      [{**s, "asset_id": s.get("asset_ref")}
                       for s in seq(context.get("telemetry"), "samples") if obj(s, "sample")]},
    }
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["meter_id", "timestamp", "kwh"])
    for reading in seq(context.get("meter_readings"), "readings"):
        obj(reading, "reading")
        writer.writerow([reading.get("meter_ref"), reading.get("timestamp"), reading.get("kwh")])
    rebuilt["meter_interval_csv"] = buffer.getvalue()
    normalized = normalize(rebuilt, canonical=True)
    require(normalized == context, "context is not canonical or contains unprotected identifiers")
    search_result = obj(state.get("search"), "search")
    matches = expected_matches(context)
    require(search_result == {"interpreted_terms": sorted(expanded(context["query"])),
                              "matches": matches, "no_match": not bool(matches)},
            "search handoff does not match validated context")
    if expected_stage in ("support", "research"):
        support_result = obj(state.get("support"), "support")
        require(support_result == support_payload(state), "support handoff is not grounded in search output")
    if expected_stage == "research":
        require(state.get("research") == research_payload(state), "research evidence provenance mismatch")
    try:
        json.dumps(state, allow_nan=False)
    except (TypeError, ValueError):
        raise ValidationError("state must be finite JSON") from None
    return state


def support_payload(state):
    context = state["context"]
    claims = expected_claims(state)
    answer = " ".join(claim["text"] for claim in claims)
    gaps = []
    if state["search"]["no_match"]:
        gaps.append("No catalog match; clarify intended product or service.")
    if not any(c["kind"] == "knowledge" for c in claims):
        gaps.append("No directly relevant supplied support policy; contact a human for unresolved questions.")
    gaps.append("No verified restoration time or live grid status is available.")
    escalate = any(o["priority"] in ("P1", "P2") for o in context["outage_reports"])
    return {"consumes": "validated:search", "question": context["support_question"],
            "matched_product_ids": [p["id"] for p in state["search"]["matches"]],
            "claims": claims, "answer": answer, "limitations": gaps,
            "human_escalation_required": escalate or not any(c["kind"] == "knowledge" for c in claims)}


def research_payload(state):
    context, support_result = state["context"], state["support"]
    question = context["research_question"]
    ranked = sorted(support_result["claims"],
                    key=lambda claim: (-relevance(question, claim["text"]), claim["id"]))
    findings = [{"claim_id": claim["id"], "finding": claim["text"],
                 "citations": list(claim["citations"]),
                 "relevance_score": relevance(question, claim["text"]),
                 "evidence_status": "supplied-synthetic"}
                for claim in ranked]
    if any(o["priority"] in ("P1", "P2") for o in context["outage_reports"]):
        decision = "Escalate safety/service outages to the utility operator before commercial recommendations."
    elif support_result["human_escalation_required"]:
        decision = "Ask a human support representative to resolve unsupported questions before deciding."
    else:
        decision = "Compare matched offerings against the supplied policies; verify eligibility with the utility."
    relevant = any(finding["relevance_score"] > 0 for finding in findings)
    cited = sorted({source for finding in findings for source in finding["citations"]})
    return {
        "consumes": "validated:support", "question": question,
        "findings": findings, "decision": decision,
        "answerability": "partial" if relevant else "insufficient-evidence",
        "recommended_product_ids": list(support_result["matched_product_ids"]),
        "citations": [source for source in context["sources"] if source["id"] in cited],
        "limitations": list(support_result["limitations"]) + [
            "Synthetic supplied evidence only; no external corroboration, causal inference or compliance certification.",
            "Seasonal interval samples do not establish annual consumption or emissions savings."],
        "next_steps": ["Verify current outage status with the utility operator.",
                       "Obtain complete interval history and sourced comparable emissions periods."],
    }


def search(raw):
    context = normalize(raw)
    matches = expected_matches(context)
    return validate({
        "schema_version": VERSION, "synthetic": True, "status": "ok", "stage": "search",
        "context": context, "search": {"interpreted_terms": sorted(expanded(context["query"])),
                                     "matches": matches, "no_match": not bool(matches)},
    }, "search")


def support(search_state):
    validate(search_state, "search")
    state = copy.deepcopy(search_state)
    state["support"] = support_payload(state)
    state["stage"] = "support"
    return validate(state, "support")


def research(support_state):
    validate(support_state, "support")
    state = copy.deepcopy(support_state)
    state["research"] = research_payload(state)
    state["stage"] = "research"
    return validate(state, "research")


def run_pipeline(raw):
    return research(support(search(raw)))


def reject_constant(value):
    raise ValidationError("non-finite JSON constant")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        raw = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                         parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(raw)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Do not echo paths, source text or potentially critical identifiers on failure.
        print(json.dumps({"status": "error", "error": "Invalid input or unreadable file.",
                          "schema_version": VERSION}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
