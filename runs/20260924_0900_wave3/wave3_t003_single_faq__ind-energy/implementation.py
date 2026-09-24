"""Synthetic utility FAQ reference. Standard library only; no provider calls."""

import csv
import io
import json
import math
import re
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


SAFETY_RULES = {"life_safety": "critical", "essential_service": "high",
                "routine": "normal"}
STOPWORDS = set("a an the is are what how why when do does can i my to of for "
                "and in on with please tell me about".split())


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required):
    require(isinstance(value, dict) and set(value) == set(required),
            "Object has missing or unexpected fields")


def text(value):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 20000,
            "Expected bounded nonempty text")
    return value


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "Expected finite nonnegative number")


def timestamp(value):
    text(value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("Invalid timestamp") from None
    require(parsed.tzinfo is not None, "Timestamp must include timezone")
    return parsed


def records(value, allow_empty=False):
    require(isinstance(value, list) and (allow_empty or bool(value)),
            "Expected record array")


def validate(data):
    """Single validation boundary for all entities and domain constraints."""
    fields(data, ["schema_version", "synthetic", "question", "grid_assets",
                  "smart_meter_interval_csv", "telemetry", "outage_reports",
                  "safety_rules", "knowledge_base"])
    require(data["schema_version"] == "1.0" and data["synthetic"] is True,
            "Only explicitly synthetic schema 1.0 is supported")
    text(data["question"])
    require(data["safety_rules"] == SAFETY_RULES, "Safety rules cannot be overridden")
    records(data["grid_assets"])
    assets = {}
    for asset in data["grid_assets"]:
        fields(asset, ["asset_id", "kind", "critical", "connected_to"])
        identifier = text(asset["asset_id"])
        require(re.fullmatch(r"SYN-[A-Z0-9-]+", identifier) is not None,
                "Asset identifiers must be fictitious SYN identifiers")
        require(identifier not in assets, "Duplicate grid asset")
        require(asset["kind"] in ("substation", "device"), "Invalid asset kind")
        require(type(asset["critical"]) is bool, "Critical flag must be boolean")
        records(asset["connected_to"], allow_empty=True)
        for connection in asset["connected_to"]:
            text(connection)
        require(len(set(asset["connected_to"])) == len(asset["connected_to"]),
                "Duplicate topology connection")
        assets[identifier] = asset
    for asset in assets.values():
        require(all(c in assets and c != asset["asset_id"]
                    for c in asset["connected_to"]), "Invalid topology reference")

    raw_csv = text(data["smart_meter_interval_csv"])
    reader = csv.DictReader(io.StringIO(raw_csv), strict=True)
    require(reader.fieldnames == ["timestamp", "meter_id", "asset_id", "consumption_kwh"],
            "Invalid interval CSV header")
    readings, previous = [], {}
    try:
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()),
                    "Malformed interval CSV row")
            time = timestamp(row["timestamp"])
            meter = text(row["meter_id"])
            require(re.fullmatch(r"SYN-METER-[A-Z0-9-]+", meter) is not None,
                    "Meter identifiers must be synthetic")
            require(row["asset_id"] in assets, "Unknown meter grid asset")
            try:
                consumption = float(row["consumption_kwh"])
            except ValueError:
                raise ValidationError("Invalid interval consumption") from None
            number(consumption)
            require(meter not in previous or time > previous[meter],
                    "Meter timestamps must strictly increase")
            previous[meter] = time
            readings.append({**row, "consumption_kwh": consumption})
    except csv.Error:
        raise ValidationError("Malformed interval CSV") from None
    require(bool(readings), "At least one interval is required")

    records(data["telemetry"])
    for point in data["telemetry"]:
        fields(point, ["timestamp", "asset_id", "voltage_kv", "load_mw",
                       "status", "emissions"])
        timestamp(point["timestamp"])
        require(isinstance(point["asset_id"], str) and point["asset_id"] in assets,
                "Unknown telemetry grid asset")
        number(point["voltage_kv"])
        number(point["load_mw"])
        require(point["status"] in ("online", "offline", "alarm"), "Invalid SCADA status")
        fields(point["emissions"], ["value", "unit", "source"])
        number(point["emissions"]["value"])
        require(point["emissions"]["unit"] in ("kgCO2e", "tCO2e", "gCO2e/kWh"),
                "Emissions require supported units")
        text(point["emissions"]["source"])

    records(data["outage_reports"], allow_empty=True)
    outage_ids = set()
    for report in data["outage_reports"]:
        fields(report, ["report_id", "asset_id", "timestamp", "hazard", "priority"])
        rid = text(report["report_id"])
        require(rid not in outage_ids, "Duplicate outage report")
        outage_ids.add(rid)
        require(isinstance(report["asset_id"], str) and report["asset_id"] in assets,
                "Unknown outage grid asset")
        timestamp(report["timestamp"])
        hazard = text(report["hazard"])
        require(hazard in SAFETY_RULES and report["priority"] == SAFETY_RULES[hazard],
                "Outage priority violates declared safety rules")

    records(data["knowledge_base"], allow_empty=True)
    ids = set()
    for article in data["knowledge_base"]:
        fields(article, ["id", "title", "text", "tags"])
        aid = text(article["id"])
        require(aid not in ids, "Duplicate knowledge article")
        ids.add(aid)
        text(article["title"])
        text(article["text"])
        records(article["tags"])
        for tag in article["tags"]:
            text(tag)
    return assets, readings


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOPWORDS


def redact(value, identifiers):
    """Redact registered critical identifiers everywhere, including citations."""
    if isinstance(value, str):
        for identifier in sorted(identifiers, key=len, reverse=True):
            value = re.sub(re.escape(identifier), "[PROTECTED-ASSET]", value,
                           flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [redact(item, identifiers) for item in value]
    if isinstance(value, dict):
        return {redact(key, identifiers): redact(item, identifiers)
                for key, item in value.items()}
    return value


def answer_request(data, answer_callable=None):
    """An optional local callable may select exact excerpts, never invent prose."""
    assets, readings = validate(data)
    protected = [a["asset_id"] for a in assets.values() if a["critical"]]
    query = tokens(data["question"])
    ranked = []
    for article in data["knowledge_base"]:
        searchable = tokens(article["title"] + " " + " ".join(article["tags"]))
        overlap = query & searchable
        # Require substantial query coverage, not merely any common word.
        if query and overlap and len(overlap) / len(query) >= 0.5:
            ranked.append((len(overlap), article["id"], article))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    evidence = redact([item[2] for item in ranked[:3]], protected)
    result = {
        "schema_version": "1.0",
        "synthetic": True,
        "status": "abstained",
        "answer": None,
        "abstention_reason": "No sufficiently relevant knowledge-base evidence.",
        "citations": [],
        "entities": {
            "grid_assets": data["grid_assets"],
            "meter_readings": readings,
            "telemetry": data["telemetry"],
            "outage_reports": data["outage_reports"],
        },
        "protection": "Registered critical asset identifiers redacted; demonstration only.",
    }
    if evidence:
        selected = evidence
        if answer_callable is not None:
            context = {"question": redact(data["question"], protected),
                       "evidence": evidence}
            try:
                # Isolate the evidence used for validation from callable mutation.
                generated = answer_callable(json.loads(json.dumps(context)))
            except Exception:
                raise ValidationError("Injected answer callable failed") from None
            fields(generated, ["answer", "evidence_ids"])
            records(generated["evidence_ids"])
            for aid in generated["evidence_ids"]:
                text(aid)
            require(len(set(generated["evidence_ids"])) == len(generated["evidence_ids"]),
                    "Duplicate answer citation")
            by_id = {a["id"]: a for a in evidence}
            require(all(aid in by_id for aid in generated["evidence_ids"]),
                    "Answer cites unavailable evidence")
            selected = [by_id[aid] for aid in generated["evidence_ids"]]
            require(generated["answer"] == "\n".join(a["text"] for a in selected),
                    "Answer must contain only exact cited excerpts")
        result.update(status="answered", answer="\n".join(a["text"] for a in selected),
                      abstention_reason=None,
                      citations=[{"id": a["id"], "title": a["title"],
                                  "excerpt": a["text"]} for a in selected])
    return redact(result, protected)


def no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "Duplicate JSON key")
        obj[key] = value
    return obj


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=no_duplicates)
        result = answer_request(data)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Never echo paths, supplied values or exception text through the CLI.
        print(json.dumps({"schema_version": "1.0", "status": "error",
                          "error": "Input file or schema validation failed."}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
