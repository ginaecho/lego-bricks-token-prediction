"""Synthetic utilities triage reference. Python standard library; no providers.

Input is a versioned envelope containing topology, smart-meter CSV, SCADA JSON,
emissions and tickets. Outputs use input-order opaque references, never asset IDs.
Safety floors cannot be weakened by category defaults or keyword configuration.
This illustrates identifier minimization, not NERC CIP certification or security
for storage/transport: protect the raw input separately.
"""

import csv
import io
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


CATEGORIES = ("outage", "grid", "meter", "general")
PRIORITIES = ("P1", "P2", "P3", "P4")
SAFETY_FLOORS = {"life_threatening": "P1", "downed_line": "P1",
                 "critical_service": "P2"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            label + ": invalid fields")


def text(value, label, limit=160):
    require(isinstance(value, str) and 0 < len(value) <= limit
            and value.strip() == value and not any(ord(c) < 32 for c in value),
            label + ": invalid text")


def identifier(value, label):
    require(isinstance(value, str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,63}", value) is not None,
            label + ": invalid identifier")


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            label + ": expected finite nonnegative number")


def timestamp(value):
    require(isinstance(value, str), "timestamp: expected text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.utcoffset() is not None, "timestamp: timezone required")
        return parsed
    except (ValueError, OverflowError):
        raise ValidationError("timestamp: invalid ISO 8601 value") from None


def sequence(value, label, maximum=1000):
    require(isinstance(value, list) and len(value) <= maximum,
            label + ": expected bounded list")


def earlier(*priorities):
    return min(priorities, key=PRIORITIES.index)


def validate(payload):
    """Single validation boundary shared by library calls and the CLI."""
    fields(payload, ("schema_version", "synthetic", "season", "assets", "meters",
                     "meter_readings_csv", "telemetry", "emissions", "tickets",
                     "config"), "input")
    require(type(payload["schema_version"]) is int
            and payload["schema_version"] == 1, "schema_version: expected 1")
    require(payload["synthetic"] is True, "synthetic: must be true")
    require(payload["season"] in ("winter", "spring", "summer", "autumn"),
            "season: invalid value")
    assets = {}
    sequence(payload["assets"], "assets")
    for item in payload["assets"]:
        fields(item, ("asset_id", "kind", "critical", "connected_to"), "asset")
        identifier(item["asset_id"], "asset")
        require(item["asset_id"] not in assets, "asset: duplicate identifier")
        require(item["kind"] in ("substation", "transformer", "feeder"),
                "asset: invalid kind")
        require(type(item["critical"]) is bool, "asset: critical must be boolean")
        sequence(item["connected_to"], "connections")
        for neighbor in item["connected_to"]:
            identifier(neighbor, "connection")
        require(len(set(item["connected_to"])) == len(item["connected_to"]),
                "connections: duplicates")
        assets[item["asset_id"]] = item
    require(bool(assets), "assets: at least one required")
    for asset in assets.values():
        require(all(n in assets and n != asset["asset_id"]
                    for n in asset["connected_to"]), "connections: invalid reference")
    meters = {}
    sequence(payload["meters"], "meters")
    for item in payload["meters"]:
        fields(item, ("meter_id", "asset_id"), "meter")
        identifier(item["meter_id"], "meter")
        require(item["meter_id"] not in meters, "meter: duplicate identifier")
        require(isinstance(item["asset_id"], str) and item["asset_id"] in assets,
                "meter: unknown asset")
        meters[item["meter_id"]] = item["asset_id"]
    raw = payload["meter_readings_csv"]
    require(isinstance(raw, str) and len(raw) <= 1_000_000, "CSV: invalid size")
    readings, seen = [], set()
    try:
        reader = csv.DictReader(io.StringIO(raw), strict=True)
        require(reader.fieldnames == ["meter_id", "timestamp", "interval_minutes",
                                      "consumption_kwh"], "CSV: invalid header")
        for row in reader:
            require(len(readings) < 10000 and None not in row
                    and all(v is not None for v in row.values()), "CSV: malformed row")
            require(row["meter_id"] in meters, "CSV: unknown meter")
            instant = timestamp(row["timestamp"])
            require(row["interval_minutes"] in ("15", "30", "60"),
                    "CSV: interval must be 15, 30 or 60 minutes")
            amount = float(row["consumption_kwh"])
            number(amount, "CSV consumption")
            key = (row["meter_id"], instant)
            require(key not in seen, "CSV: duplicate interval")
            seen.add(key)
            readings.append((row["meter_id"], amount))
    except (csv.Error, ValueError, OverflowError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError("CSV: malformed numeric or record value") from None
    sequence(payload["telemetry"], "telemetry")
    for item in payload["telemetry"]:
        fields(item, ("asset_id", "timestamp", "voltage_v", "status"), "telemetry")
        require(isinstance(item["asset_id"], str) and item["asset_id"] in assets,
                "telemetry: unknown asset")
        timestamp(item["timestamp"])
        number(item["voltage_v"], "telemetry voltage")
        require(item["status"] in ("normal", "alarm", "offline"),
                "telemetry: invalid status")
    sequence(payload["emissions"], "emissions")
    for item in payload["emissions"]:
        fields(item, ("asset_id", "value", "unit", "source"), "emissions")
        require(isinstance(item["asset_id"], str) and item["asset_id"] in assets,
                "emissions: unknown asset")
        number(item["value"], "emissions value")
        require(item["unit"] in ("kgCO2e", "tCO2e"), "emissions: invalid unit")
        text(item["source"], "emissions source")
    config = payload["config"]
    fields(config, ("keywords", "category_priority", "routes", "safety_rules"), "config")
    for section in ("keywords", "category_priority", "routes"):
        fields(config[section], CATEGORIES, section)
    for category in CATEGORIES:
        words = config["keywords"][category]
        sequence(words, "keywords", 30)
        for word in words:
            text(word, "keyword", 40)
        require(config["category_priority"][category] in PRIORITIES,
                "category_priority: invalid priority")
        route = config["routes"][category]
        fields(route, ("team", "accountable_owner"), "route")
        text(route["team"], "route team", 80)
        text(route["accountable_owner"], "route owner", 80)
    fields(config["safety_rules"], SAFETY_FLOORS, "safety_rules")
    for flag, floor in SAFETY_FLOORS.items():
        priority = config["safety_rules"][flag]
        require(priority in PRIORITIES and PRIORITIES.index(priority)
                <= PRIORITIES.index(floor), "safety_rules: unsafe priority floor")
    sequence(payload["tickets"], "tickets")
    ticket_ids = set()
    for item in payload["tickets"]:
        fields(item, ("ticket_id", "text", "asset_ids", "meter_ids", "outage_report"),
               "ticket")
        identifier(item["ticket_id"], "ticket")
        require(item["ticket_id"] not in ticket_ids, "ticket: duplicate identifier")
        ticket_ids.add(item["ticket_id"])
        text(item["text"], "ticket text", 2000)
        for field, known in (("asset_ids", assets), ("meter_ids", meters)):
            sequence(item[field], "ticket references")
            require(all(isinstance(i, str) and i in known for i in item[field]),
                    "ticket: unknown reference")
            require(len(set(item[field])) == len(item[field]), "ticket: duplicate reference")
        outage = item["outage_report"]
        if outage is not None:
            fields(outage, ("reported_at", "affected_customers", "safety"), "outage")
            timestamp(outage["reported_at"])
            require(type(outage["affected_customers"]) is int
                    and outage["affected_customers"] >= 0, "outage: invalid customer count")
            fields(outage["safety"], SAFETY_FLOORS, "outage safety")
            require(all(type(v) is bool for v in outage["safety"].values()),
                    "outage: safety values must be boolean")
    return assets, meters, readings


def triage(payload):
    assets, meters, readings = validate(payload)
    asset_refs = {key: f"asset-{index:03d}"
                  for index, key in enumerate(assets, 1)}
    meter_refs = {key: f"meter-{index:03d}"
                  for index, key in enumerate(meters, 1)}
    config = payload["config"]
    results = []
    for index, ticket in enumerate(payload["tickets"], 1):
        linked = set(ticket["asset_ids"]) | {meters[m] for m in ticket["meter_ids"]}
        outage = ticket["outage_report"]
        matches = [candidate for candidate in CATEGORIES
                   if any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)",
                                    ticket["text"], re.IGNORECASE)
                          for word in config["keywords"][candidate])]
        reason = "fallback"
        category = "general"
        if outage is not None:
            category, reason = "outage", "structured_outage"
        elif "outage" in matches:
            category, reason = "outage", "configured_keyword"
        elif any(t["asset_id"] in linked and t["status"] != "normal"
                 for t in payload["telemetry"]):
            category, reason = "grid", "telemetry_alert"
        elif matches:
            category, reason = matches[0], "configured_keyword"
        priority = config["category_priority"][category]
        applied = []
        if outage is not None:
            for flag, active in outage["safety"].items():
                if active:
                    priority = earlier(priority, config["safety_rules"][flag])
                    applied.append(flag)
        elif category == "outage":
            # Free-text outage reports lack a safety assessment: never de-prioritize.
            priority = "P1"
            applied.append("unassessed_outage")
        results.append({
            "ticket_ref": f"ticket-{index:03d}", "category": category,
            "priority": priority, "routing": dict(config["routes"][category]),
            "reason": reason, "safety_rules_applied": sorted(applied),
            "asset_refs": [ref for key, ref in asset_refs.items() if key in linked],
            "meter_refs": [meter_refs[m] for m in ticket["meter_ids"]],
            "outage_report": outage,
        })
    totals = {meter: 0.0 for meter in meters}
    for meter, amount in readings:
        totals[meter] += amount
        number(totals[meter], "aggregate consumption")
    output = {
        "status": "ok", "schema_version": 1, "synthetic": True,
        "season": payload["season"], "tickets": results,
        "meter_readings": [{"meter_ref": meter_refs[m], "asset_ref": asset_refs[a],
                            "consumption": {"value": round(totals[m], 6), "unit": "kWh"}}
                           for m, a in meters.items()],
        "grid_assets": [{"asset_ref": asset_refs[key], "kind": asset["kind"],
                         "critical": asset["critical"],
                         "connected_to": [asset_refs[n] for n in asset["connected_to"]]}
                        for key, asset in assets.items()],
        "telemetry": [{"asset_ref": asset_refs[t["asset_id"]],
                       **{k: v for k, v in t.items() if k != "asset_id"}}
                      for t in payload["telemetry"]],
        "emissions": [{"asset_ref": asset_refs[e["asset_id"]],
                       **{k: v for k, v in e.items() if k != "asset_id"}}
                      for e in payload["emissions"]],
        "protection": "Opaque per-envelope references; raw identifiers omitted.",
    }
    # Scrub configurable display strings too; critical identifiers never leave this boundary.
    protected = sorted((a for a in assets if assets[a]["critical"]), key=len, reverse=True)

    def redact(value):
        if isinstance(value, str):
            for asset_id in protected:
                value = re.sub(re.escape(asset_id), "[protected-asset]", value,
                               flags=re.IGNORECASE)
            return value
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value

    return redact(output)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON: duplicate field")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: implementation.py INPUT.json")
        path = Path(args[0])
        require(path.stat().st_size <= 2_000_000, "input: size limit exceeded")
        payload = json.loads(path.read_text(encoding="utf-8"),
                             object_pairs_hook=no_duplicates)
        output = triage(payload)
        encoded = json.dumps(output, allow_nan=False, sort_keys=True)
    except ValidationError as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 2
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable file"}))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
