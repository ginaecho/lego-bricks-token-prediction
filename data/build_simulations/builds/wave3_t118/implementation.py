"""Synthetic energy evidence pipeline. Standard library only; no certification claims.

Every boundary uses the same versioned envelope and validate() function. Evidence
means supplied text matching declared terms, not independently verified truth.
Critical identifiers are demonstratively pseudonymized, not access-controlled.
"""

import copy
import csv
import datetime as dt
import io
import json
import math
import re
import sys


STAGES = ("input", "review", "adaptive", "research", "faq")
RULES = {"life_safety": "P1", "essential_service": "P2", "routine": "P3"}
PREREQUISITES = {
    "asset_protection": [],
    "safety": ["asset_protection"],
    "meter_data": ["asset_protection"],
    "research": ["safety", "meter_data"],
}
STOP = set("a an the what which how is are was were to of in on for and or "
           "does do with about evidence supports supplied".split())


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 12000,
            label + " must be nonempty bounded text")


def number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    require(valid,
            label + " must be finite and nonnegative")


def validate_json_tree(value, depth=0):
    require(depth <= 40, "JSON nesting exceeds supported depth")
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "JSON keys must be text")
        for item in value.values():
            validate_json_tree(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            validate_json_tree(item, depth + 1)
    elif type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        require(finite, "JSON numbers must be finite")
    else:
        require(value is None or type(value) in (str, bool), "unsupported JSON value")


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("timestamp must be ISO 8601") from None
    require(parsed.tzinfo is not None, "timestamp must include timezone")
    return parsed


def records(value, label, nonempty=False):
    require(isinstance(value, list) and len(value) <= 1000, label + " must be a bounded list")
    require(not nonempty or bool(value), label + " must not be empty")
    require(all(isinstance(row, dict) for row in value), label + " entries must be objects")
    return value


def indexed(rows, label):
    result = {}
    for row in rows:
        text(row.get("id"), label + ".id")
        require(row["id"] not in result, label + " IDs must be unique")
        result[row["id"]] = row
    return result


def meter_rows(context):
    value = context.get("meter_csv")
    text(value, "meter_csv")
    try:
        reader = csv.DictReader(io.StringIO(value), strict=True)
        require(reader.fieldnames == [
            "timestamp", "meter_id", "asset_id", "consumption_kwh", "season"
        ], "meter CSV header mismatch")
        rows = list(reader)
    except csv.Error:
        raise ValidationError("malformed meter CSV") from None
    require(0 < len(rows) <= 1000, "meter CSV must contain bounded interval readings")
    assets = {row["id"] for row in context["grid_assets"]}
    seen = set()
    for row in rows:
        require(None not in row and all(v is not None for v in row.values()),
                "meter CSV row width mismatch")
        instant = timestamp(row["timestamp"])
        text(row["meter_id"], "meter_id")
        require(row["asset_id"] in assets, "meter references unknown asset")
        require(row["season"] in ("winter", "spring", "summer", "autumn"),
                "meter season is invalid")
        try:
            consumption = float(row["consumption_kwh"])
        except ValueError:
            raise ValidationError("meter consumption must be numeric") from None
        number(consumption, "meter consumption")
        key = (row["meter_id"], instant)
        require(key not in seen, "duplicate meter interval")
        seen.add(key)
        row["consumption_kwh"] = consumption
    return rows


def validate_context(context, input_stage):
    require(isinstance(context, dict), "context must be an object")
    require(context.get("synthetic") is True, "fixtures must be explicitly synthetic")
    require(context.get("identifier_policy") == "redact",
            "critical identifier policy must be redact")
    require(context.get("safety_rules") == RULES, "declared safety rules mismatch")
    assets = indexed(records(context.get("grid_assets"), "grid_assets", True), "grid_assets")
    devices = set()
    for asset in assets.values():
        require(type(asset.get("critical")) is bool, "asset critical flag must be boolean")
        text(asset.get("device_id"), "asset.device_id")
        require(asset["device_id"] not in devices, "device IDs must be unique")
        devices.add(asset["device_id"])
        if input_stage:
            require(not asset["id"].startswith("protected-") and
                    not asset["device_id"].startswith("protected-"),
                    "input identifiers use reserved protection prefix")
        if asset["critical"]:
            require(len(asset["id"]) >= 6 and len(asset["device_id"]) >= 6,
                    "critical identifiers must be at least six characters")
            if not input_stage:
                require(asset["id"].startswith("protected-asset-") and
                        asset["device_id"].startswith("protected-device-"),
                        "critical identifiers must be protected")
    require(not (set(assets) & devices), "asset and device identifier namespaces must differ")
    edges = context.get("topology")
    require(isinstance(edges, list) and len(edges) <= 1000, "topology must be a bounded list")
    for edge in edges:
        require(isinstance(edge, list) and len(edge) == 2 and
                all(isinstance(item, str) and item in assets for item in edge) and
                edge[0] != edge[1], "invalid topology edge")
    meter_rows(context)
    telemetry = context.get("telemetry")
    require(isinstance(telemetry, dict), "SCADA telemetry must be an object")
    for obs in records(telemetry.get("observations"), "telemetry.observations", True):
        timestamp(obs.get("timestamp"))
        require(isinstance(obs.get("asset_id"), str) and obs["asset_id"] in assets,
                "telemetry references unknown asset")
        require(obs.get("device_id") == assets[obs["asset_id"]]["device_id"],
                "telemetry device must belong to asset")
        require(obs.get("signal") in ("load", "voltage"), "unknown telemetry signal")
        require(obs.get("unit") == {"load": "kW", "voltage": "V"}[obs["signal"]],
                "telemetry unit mismatch")
        number(obs.get("value"), "telemetry.value")
    sources = indexed(records(context.get("sources"), "sources"), "sources")
    for source in sources.values():
        text(source.get("title"), "source.title")
        text(source.get("text"), "source.text")
        require(source.get("kind") in ("primary", "secondary"), "source kind is invalid")
        claims = source.get("claims", {})
        require(isinstance(claims, dict), "source claims must be an object")
        for key, value in claims.items():
            text(key, "claim key")
            text(value, "claim value")
    for emission in records(context.get("emissions"), "emissions"):
        number(emission.get("value"), "emissions.value")
        require(emission.get("unit") in ("kgCO2e", "tCO2e"), "emissions require explicit units")
        require(isinstance(emission.get("source_id"), str) and
                emission["source_id"] in sources, "emissions require a supplied source")
    outages = indexed(records(context.get("outages"), "outages"), "outages")
    for outage in outages.values():
        require(isinstance(outage.get("asset_id"), str) and outage["asset_id"] in assets,
                "outage references unknown asset")
        flags = outage.get("safety")
        require(isinstance(flags, dict) and set(flags) == {"life_safety", "essential_service"}
                and all(type(v) is bool for v in flags.values()), "invalid outage safety flags")
        expected = RULES["life_safety"] if flags["life_safety"] else (
            RULES["essential_service"] if flags["essential_service"] else RULES["routine"])
        require(outage.get("priority") == expected, "outage priority violates declared safety rules")
        text(outage.get("description"), "outage.description")
    requirements = indexed(records(context.get("requirements"), "requirements", True), "requirements")
    for item in requirements.values():
        text(item.get("description"), "requirement.description")
        refs = item.get("source_ids")
        require(isinstance(refs, list) and all(isinstance(ref, str) and ref in sources for ref in refs),
                "requirement references unknown source")
        terms = item.get("required_terms")
        require(isinstance(terms, list) and 0 < len(terms) <= 30,
                "requirement must declare bounded evidence terms")
        for term in terms:
            text(term, "requirement term")
        require(item.get("topic") in PREREQUISITES, "unknown requirement topic")
    profile = context.get("profile")
    require(isinstance(profile, dict), "profile must be an object")
    require(profile.get("experience") in ("beginner", "intermediate", "expert"),
            "invalid experience")
    require(profile.get("preference") in ("concise", "detailed"), "invalid preference")
    completed = profile.get("completed_topics")
    require(isinstance(completed, list) and all(isinstance(t, str) and t in PREREQUISITES
            for t in completed), "unknown completed topic")
    questions = indexed(records(context.get("questions"), "questions", True), "questions")
    for question in questions.values():
        text(question.get("text"), "question.text")
        require(not question["id"].startswith("gap:"), "question ID uses reserved gap prefix")


def protect(context):
    replacements = {}
    for index, asset in enumerate(sorted(context["grid_assets"], key=lambda a: a["id"]), 1):
        if asset["critical"]:
            replacements[asset["id"]] = "protected-asset-%04d" % index
            replacements[asset["device_id"]] = "protected-device-%04d" % index
    pattern = re.compile("|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))) \
        if replacements else None

    def visit(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: replacements[match.group()], value) if pattern else value
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {visit(key): visit(item) for key, item in value.items()}
        return value
    return visit(context)


def review_result(context, previous):
    sources = {source["id"]: source for source in context["sources"]}
    checks = []
    for requirement in context["requirements"]:
        evidence, missing = [], []
        for term in requirement["required_terms"]:
            matches = [sid for sid in requirement["source_ids"]
                       if term.casefold() in sources[sid]["text"].casefold()]
            if matches:
                evidence.extend({"term": term, "source_id": sid,
                                 "quote": sources[sid]["text"]} for sid in matches)
            else:
                missing.append(term)
        checks.append({"requirement_id": requirement["id"], "topic": requirement["topic"],
                       "status": "gap" if missing else "supported",
                       "missing_terms": missing, "evidence": evidence})
    readings = meter_rows(context)
    seasonal = {}
    for row in readings:
        seasonal[row["season"]] = round(seasonal.get(row["season"], 0) + row["consumption_kwh"], 6)
        number(seasonal[row["season"]], "seasonal consumption total")
    return {"checks": checks, "gaps": [row["requirement_id"] for row in checks if row["missing_terms"]],
            "meter_summary": {"intervals": len(readings), "seasonal_consumption": seasonal, "unit": "kWh"},
            "outage_priorities": [{"outage_id": row["id"], "priority": row["priority"]}
                                  for row in context["outages"]],
            "emissions": copy.deepcopy(context["emissions"]),
            "limitations": ["Term presence is not verification or certification.",
                            "Critical identifiers are pseudonymized; this is not a NERC CIP audit."]}


def adaptive_result(context, previous):
    review = previous["review"]
    profile = context["profile"]
    completed = set(profile["completed_topics"])
    ordered = []

    def add(topic):
        if topic in completed or topic in ordered:
            return
        for dependency in PREREQUISITES[topic]:
            add(dependency)
        ordered.append(topic)

    for check in review["checks"]:
        if check["status"] == "gap":
            add(check["topic"])
    add("research")
    depth = {"beginner": "foundational", "intermediate": "guided", "expert": "accelerated"}[
        profile["experience"]]
    modules = []
    for topic in ordered:
        related = [check["requirement_id"] for check in review["checks"]
                   if check["status"] == "gap" and check["topic"] == topic]
        explanation = "Required to prepare " + topic.replace("_", " ") + "."
        if profile["preference"] == "detailed":
            explanation += " Complete its listed prerequisites before using supplied evidence; gaps remain unverified."
        modules.append({"topic": topic, "prerequisites": PREREQUISITES[topic],
                        "depth": depth, "explanation": explanation, "gap_ids": related})
    requirements = {row["id"]: row for row in context["requirements"]}
    questions = [{"id": q["id"], "text": q["text"], "origin": "user"} for q in context["questions"]]
    for check in review["checks"]:
        if check["status"] == "gap":
            questions.append({"id": "gap:" + check["requirement_id"],
                              "text": requirements[check["requirement_id"]]["description"] +
                              " " + " ".join(check["missing_terms"]),
                              "origin": check["requirement_id"]})
    return {"modules": modules, "completed_topics": sorted(completed),
            "review_gap_ids": review["gaps"], "research_questions": questions}


def tokens(value):
    return {word for word in re.findall(r"[a-z0-9]+", value.casefold())
            if len(word) > 2 and word not in STOP}


def research_result(context, previous):
    adaptive = previous["adaptive"]
    findings = []
    for question in adaptive["research_questions"]:
        query = tokens(question["text"])
        ranked = []
        for source in context["sources"]:
            overlap = query & tokens(source["text"])
            if query and len(overlap) >= min(2, len(query)):
                ranked.append((len(overlap), source))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        # Check all retrieved sources for contradictions before limiting answer excerpts.
        claim_values = {}
        for _, source in ranked:
            for key, value in source.get("claims", {}).items():
                claim_values.setdefault(key, set()).add(value.casefold())
        conflicts = sorted(key for key, values in claim_values.items() if len(values) > 1)
        evidence = [{"source_id": source["id"], "quote": source["text"],
                     "kind": source["kind"], "matched_terms": sorted(query & tokens(source["text"]))}
                    for _, source in ranked[:3]]
        findings.append({"question_id": question["id"], "question": question["text"],
                         "origin": question["origin"],
                         "status": "conflicting" if conflicts else ("supported" if evidence else "insufficient"),
                         "evidence": evidence, "conflicts": conflicts,
                         "decision": "Resolve conflicting claims before deciding." if conflicts else (
                             "Use cited excerpts as bounded evidence." if evidence else
                             "Obtain additional source material before deciding.")})
    return {"findings": findings, "onboarding_topics": [m["topic"] for m in adaptive["modules"]],
            "review_gap_ids": adaptive["review_gap_ids"],
            "limitations": ["Only supplied sources were searched.",
                            "Lexical retrieval and declared claim conflicts are not semantic verification."]}


def faq_result(context, previous):
    research = previous["research"]
    answers = []
    for finding in research["findings"]:
        supported = finding["status"] == "supported"
        answers.append({"question_id": finding["question_id"], "question": finding["question"],
                        "status": "answered" if supported else "abstained",
                        "answer": "\n".join("[" + e["source_id"] + "] " + e["quote"]
                                            for e in finding["evidence"]) if supported else
                                  "I cannot provide a grounded answer: " + finding["status"] + " evidence.",
                        "citations": [e["source_id"] for e in finding["evidence"]] if supported else [],
                        "reason": "Verbatim supplied-source excerpts." if supported else finding["decision"]})
    return {"answers": answers, "unresolved_review_gaps": research["review_gap_ids"],
            "onboarding_topics": research["onboarding_topics"],
            "notice": "Synthetic demonstration only; no compliance certification or operational dispatch."}


BUILDERS = {"review": review_result, "adaptive": adaptive_result,
            "research": research_result, "faq": faq_result}


def validate(envelope, expected_stage=None):
    validate_json_tree(envelope)
    require(isinstance(envelope, dict), "envelope must be an object")
    require(set(envelope) == {"schema_version", "status", "stage", "context", "results"},
            "envelope fields mismatch")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "unsupported schema_version")
    require(envelope["status"] == "ok", "pipeline envelope status must be ok")
    stage = envelope["stage"]
    require(isinstance(stage, str) and stage in STAGES, "unknown pipeline stage")
    require(expected_stage is None or stage == expected_stage, "unexpected handoff stage")
    validate_context(envelope["context"], stage == "input")
    require(isinstance(envelope["results"], dict), "results must be an object")
    expected = {}
    # Bounded deterministic replay verifies all prior outputs and their provenance,
    # not merely their shape. No unvalidated stage data is consumed.
    for name in STAGES[1:STAGES.index(stage) + 1]:
        expected[name] = BUILDERS[name](envelope["context"], expected)
    require(envelope["results"] == expected, "stage results fail deterministic provenance validation")
    return envelope


def advance(envelope, destination):
    require(destination in BUILDERS, "invalid destination")
    validate(envelope, STAGES[STAGES.index(destination) - 1])
    result = copy.deepcopy(envelope)
    if destination == "review":
        result["context"] = protect(result["context"])
    result["stage"] = destination
    result["results"][destination] = BUILDERS[destination](result["context"], result["results"])
    return validate(result, destination)


def review(envelope):
    return advance(envelope, "review")


def adaptive(envelope):
    return advance(envelope, "adaptive")


def research(envelope):
    return advance(envelope, "research")


def faq(envelope):
    return advance(envelope, "faq")


def run(envelope):
    return faq(research(adaptive(review(envelope))))


def parse_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, "duplicate JSON key")
            result[key] = item
        return result

    def constant(value):
        raise ValidationError("nonfinite JSON number")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json (or - for stdin)")
        if argv[0] == "-":
            raw = sys.stdin.read(1000001)
        else:
            with open(argv[0], "r", encoding="utf-8") as handle:
                raw = handle.read(1000001)
        require(len(raw) <= 1000000, "input exceeds one million characters")
        result = run(parse_json(raw))
    except (OSError, UnicodeError):
        result = {"status": "error", "error": "Input file could not be read as UTF-8."}
    except json.JSONDecodeError:
        result = {"status": "error", "error": "Malformed JSON input."}
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Never echo supplied identifiers or source material in error diagnostics.
        result = {"status": "error", "error": "Input or handoff validation failed."}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 2 if result["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
