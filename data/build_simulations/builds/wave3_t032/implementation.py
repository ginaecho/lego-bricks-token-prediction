"""Synthetic energy onboarding -> exact-passage research; Python standard library."""
import csv
import io
import json
import math
import sys
from datetime import datetime

TOPICS = ("consumption", "outages", "emissions")
RULES = {"life_safety": "P1", "essential_service": "P2", "routine": "P3"}
SCHEMA = "synthetic-energy/v1"


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def text(value):
    return isinstance(value, str) and bool(value.strip())


def validate(kind, value, context=None):
    """Shared validation boundary for input and both pipeline handoffs."""
    require(isinstance(value, dict), "Object required")
    if kind == "input":
        require(value.get("schema_version") == SCHEMA, "Unsupported schema")
        require(value.get("synthetic") is True, "Only labeled synthetic fixtures accepted")
        profile = value.get("profile")
        require(isinstance(profile, dict), "Profile required")
        require(profile.get("experience") in ("novice", "experienced"), "Invalid experience")
        require(profile.get("detail") in ("brief", "guided"), "Invalid detail preference")
        require(isinstance(profile.get("topics"), list) and profile["topics"]
                and all(t in TOPICS for t in profile["topics"])
                and len(set(profile["topics"])) == len(profile["topics"]), "Invalid topics")
        require(isinstance(profile.get("completed"), list)
                and all(t in ("energy_basics", "asset_protection", "safety_rules", "emissions_units")
                        for t in profile["completed"]), "Invalid completed prerequisites")
        require(value.get("safety_rules") == RULES, "Declared safety rules must match demonstration policy")
        assets = value.get("grid_assets")
        require(isinstance(assets, list) and assets, "Grid assets required")
        ids, aliases = set(), set()
        for asset in assets:
            require(isinstance(asset, dict), "Invalid asset")
            ident, alias = asset.get("internal_id"), asset.get("alias")
            require(text(ident) and len(ident) >= 8 and ident.startswith("SYN-"),
                    "Synthetic internal asset identifier required")
            require(isinstance(alias, str) and alias.startswith("asset-")
                    and alias[6:].isdigit(), "Public asset alias required")
            require(ident not in ids and alias not in aliases, "Duplicate asset")
            require(type(asset.get("critical")) is bool, "Critical flag required")
            ids.add(ident)
            aliases.add(alias)
        links = value.get("topology")
        require(isinstance(links, list), "Synthetic topology required")
        for link in links:
            require(isinstance(link, list) and len(link) == 2
                    and all(isinstance(x, str) and x in ids for x in link)
                    and link[0] != link[1], "Invalid topology edge")
        sources = value.get("passages")
        require(isinstance(sources, list) and sources, "Passages required")
        source_ids = set()
        for source in sources:
            require(isinstance(source, dict)
                    and all(text(source.get(k)) for k in ("source_id", "title", "text")),
                    "Invalid passage")
            require(source["source_id"] not in source_ids, "Duplicate source citation")
            require(isinstance(source.get("topics"), list) and source["topics"]
                    and all(t in TOPICS for t in source["topics"]), "Invalid passage topics")
            source_ids.add(source["source_id"])
            # Citations remain verbatim: reject sensitive source text rather than rewrite it.
            require(not any(i in json.dumps(source, ensure_ascii=False) for i in ids),
                    "Protected asset identifier in public source")
        telemetry = value.get("telemetry")
        require(isinstance(telemetry, dict) and telemetry.get("format") == "SCADA-synthetic",
                "SCADA-style telemetry required")
        samples = telemetry.get("samples")
        require(isinstance(samples, list) and samples, "Telemetry samples required")
        for sample in samples:
            require(isinstance(sample, dict) and sample.get("device_id") in ids
                    and number(sample.get("load")) and sample.get("load_unit") == "kW",
                    "Invalid telemetry sample")
            timestamp(sample.get("timestamp"))
            emission = sample.get("emissions")
            require(isinstance(emission, dict) and number(emission.get("value"))
                    and emission.get("unit") in ("kgCO2e", "tCO2e")
                    and emission.get("source_id") in source_ids,
                    "Emissions require nonnegative value, units and known source")
        readings = parse_readings(value.get("meter_csv"), ids)
        outages = value.get("outage_reports")
        require(isinstance(outages, list), "Outage reports required")
        outage_ids = set()
        for outage in outages:
            require(isinstance(outage, dict) and text(outage.get("report_id"))
                    and outage.get("asset_id") in ids
                    and type(outage.get("life_safety")) is bool
                    and type(outage.get("essential_service")) is bool, "Invalid outage")
            require(outage["report_id"] not in outage_ids, "Duplicate outage report")
            outage_ids.add(outage["report_id"])
            priority = "P1" if outage["life_safety"] else (
                "P2" if outage["essential_service"] else "P3")
            require(outage.get("priority") == priority, "Outage priority violates declared safety rules")
            require(not any(i in outage["report_id"] for i in ids), "Protected identifier in report ID")
        require(readings, "Meter readings required")
    elif kind == "onboarding":
        require(context is not None and value.get("schema_version") == SCHEMA,
                "Invalid onboarding schema")
        profile = context["profile"]
        require(value.get("topics") == profile["topics"], "Topic handoff mismatch")
        require(value.get("detail") == profile["detail"], "Preference handoff mismatch")
        required = prerequisites(profile)
        require(value.get("required") == required, "Prerequisite handoff mismatch")
        pending = [x for x in required if x not in profile["completed"]]
        require(value.get("pending") == pending and value.get("ready") is (not pending),
                "Readiness handoff mismatch")
        require(value.get("lessons") == lessons(profile, pending), "Explanation handoff mismatch")
    elif kind == "research":
        require(context is not None and value.get("schema_version") == SCHEMA,
                "Invalid research schema")
        onboarding, source_input = context
        require(value.get("topics") == onboarding["topics"], "Research topic mismatch")
        require(value.get("status") == ("complete" if onboarding["ready"] else "blocked"),
                "Research readiness mismatch")
        expected = extract(onboarding, source_input)
        require(value.get("findings") == expected, "Findings or exact citations invalid")
        require(value.get("unanswered_topics") == (
            [t for t in onboarding["topics"] if not any(f["topic"] == t for f in expected)]
            if onboarding["ready"] else []), "Unanswered topics mismatch")
    else:
        raise ValidationError("Unknown validation boundary")
    return value


def timestamp(value):
    require(text(value), "Timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.tzinfo is not None, "Timezone required")
        return result
    except ValueError:
        raise ValidationError("Invalid timezone-aware timestamp") from None


def parse_readings(raw, ids):
    require(isinstance(raw, str), "Smart-meter CSV required")
    reader = csv.DictReader(io.StringIO(raw), strict=True)
    require(reader.fieldnames == ["meter_id", "timestamp", "interval_minutes", "consumption", "unit"],
            "Invalid smart-meter CSV header")
    rows, seen = [], {}
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
        require(row["meter_id"] in ids and row["unit"] == "kWh", "Invalid meter or consumption unit")
        try:
            interval, consumption = int(row["interval_minutes"]), float(row["consumption"])
        except ValueError:
            raise ValidationError("Invalid meter numeric value") from None
        require(interval in (15, 30, 60) and number(consumption), "Invalid meter interval or consumption")
        when = timestamp(row["timestamp"])
        last = seen.get(row["meter_id"])
        require(last is None or (when - last[0]).total_seconds() >= last[1] * 60,
                "Overlapping or unordered meter intervals")
        seen[row["meter_id"]] = (when, interval)
        rows.append({"meter_id": row["meter_id"], "timestamp": row["timestamp"],
                     "interval_minutes": interval, "consumption": consumption, "unit": "kWh"})
    return rows


def prerequisites(profile):
    result = ["asset_protection", "safety_rules"]
    if profile["experience"] == "novice":
        result.insert(0, "energy_basics")
    if "emissions" in profile["topics"]:
        result.append("emissions_units")
    return result


def lessons(profile, pending):
    explanations = {
        "energy_basics": "Distinguish interval energy in kWh from instantaneous load in kW.",
        "asset_protection": "Use public aliases; never publish internal critical asset identifiers.",
        "safety_rules": "Life safety is P1; essential service is P2; all other outages are P3.",
        "emissions_units": "Keep each emissions value together with its unit and source citation.",
    }
    return [{"prerequisite": key, "explanation": explanations[key],
             "action": ("Review the explanation and explicitly mark completion before rerunning."
                        if profile["detail"] == "guided" else "Review and mark complete.")}
            for key in pending]


def onboard(source_input):
    validate("input", source_input)
    profile = source_input["profile"]
    required = prerequisites(profile)
    pending = [p for p in required if p not in profile["completed"]]
    result = {"schema_version": SCHEMA, "topics": list(profile["topics"]),
              "detail": profile["detail"], "required": required, "pending": pending,
              "ready": not pending, "lessons": lessons(profile, pending)}
    return validate("onboarding", result, source_input)


def extract(onboarding, source_input):
    if not onboarding["ready"]:
        return []
    findings = []
    for topic in onboarding["topics"]:
        matches = sorted((p for p in source_input["passages"] if topic in p["topics"]),
                         key=lambda p: p["source_id"])
        # Curated topic tags define bounded retrieval; brief mode returns one passage per topic.
        for passage in matches[:1] if onboarding["detail"] == "brief" else matches:
            findings.append({"topic": topic, "finding": passage["text"],
                             "citation": {"source_id": passage["source_id"],
                                          "title": passage["title"], "start": 0,
                                          "end": len(passage["text"]), "quote": passage["text"]}})
    return findings


def research(onboarding, source_input):
    validate("input", source_input)
    validate("onboarding", onboarding, source_input)
    findings = extract(onboarding, source_input)
    result = {"schema_version": SCHEMA, "topics": list(onboarding["topics"]),
              "status": "complete" if onboarding["ready"] else "blocked",
              "findings": findings,
              "unanswered_topics": [t for t in onboarding["topics"]
                                    if not any(f["topic"] == t for f in findings)]
              if onboarding["ready"] else []}
    return validate("research", result, (onboarding, source_input))


def run(source_input):
    guided = onboard(source_input)
    findings = research(guided, source_input)
    aliases = {a["internal_id"]: a["alias"] for a in source_input["grid_assets"]}
    readings = parse_readings(source_input["meter_csv"], set(aliases))
    result = {
        "schema_version": SCHEMA, "synthetic": True,
        "status": "ok" if guided["ready"] else "needs_prerequisites",
        "notice": "Synthetic demonstration only; not NERC CIP compliance certification.",
        "onboarding": guided, "research": findings,
        "entities": {
            "grid_assets": [{"alias": a["alias"], "critical": a["critical"]}
                            for a in source_input["grid_assets"]],
            "topology": [[aliases[x] for x in edge] for edge in source_input["topology"]],
            "meter_readings": [{**r, "meter_id": aliases[r["meter_id"]]} for r in readings],
            "outage_reports": [{**o, "asset_id": aliases[o["asset_id"]]}
                               for o in source_input["outage_reports"]],
            "telemetry": [{k: v for k, v in s.items() if k in
                           ("timestamp", "load", "load_unit", "emissions")} |
                          {"device_id": aliases[s["device_id"]]}
                          for s in source_input["telemetry"]["samples"]],
        },
    }
    # Defense in depth includes unexpected nested fields, errors never echo input.
    require(not any(i in json.dumps(result, ensure_ascii=False) for i in aliases),
            "Protected identifier in output")
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            source_input = json.load(handle)
        result = run(source_input)
        print(json.dumps(result, allow_nan=False, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, csv.Error, OverflowError):
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable file."}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
