"""Synthetic utilities customer insights. Standard library; not a compliance product.

Schema v1 has a shared envelope and validated input/output entity collections.
CSV consumption is per 30-minute interval, in kWh. Telemetry voltage is in V.
Identifiers are replaced with request-local aliases; free text is never echoed.
Sentiment uses a fixed lexicon with immediate-token negation, not an LLM.
"""

import csv
import io
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


LEXICON = {"good": 1, "great": 2, "reliable": 1, "thanks": 1,
           "bad": -1, "poor": -1, "angry": -2, "unsafe": -2,
           "dangerous": -2, "outage": -1, "slow": -1}
SAFETY_RULES = {"fire": 1, "downed_line": 1, "medical_dependency": 1,
                "active_outage": 2, "other": 3}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected),
            "Invalid object fields")


def text(value):
    require(type(value) is str and 0 < len(value.strip()) <= 2000,
            "Invalid text field")


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value)
            and value >= minimum, "Invalid numeric field")


def timestamp(value):
    text(value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.utcoffset() is not None, "Timestamp requires timezone")
        return parsed
    except ValueError:
        raise ValidationError("Invalid timestamp") from None


def sentiment(description):
    tokens = re.findall(r"[a-z]+", description.lower())
    matches = []
    for index, token in enumerate(tokens):
        if token in LEXICON:
            negated = index > 0 and tokens[index - 1] in {"not", "never", "no"}
            weight = LEXICON[token] * (-1 if negated else 1)
            matches.append({"term": token, "weight": weight, "negated": negated})
    score = sum(item["weight"] for item in matches)
    return {"score": score, "label": "positive" if score > 0 else
            "negative" if score < 0 else "neutral", "matches": matches}


def priority(report):
    flags = report["safety_flags"]
    if flags:
        return 1, "declared_safety_flag"
    if report["outage_active"]:
        return 2, "active_outage"
    return 3, "other"


def parse_meters(raw, asset_ids):
    require(type(raw) is str, "Meter CSV must be text")
    try:
        reader = csv.DictReader(io.StringIO(raw), strict=True)
        require(reader.fieldnames == ["timestamp", "device_id", "consumption_kwh", "season"],
                "Invalid meter CSV header")
        rows, seen = [], set()
        for row in reader:
            require(None not in row and None not in row.values(), "Invalid meter CSV row")
            instant = timestamp(row["timestamp"])
            require(instant.minute in (0, 30) and instant.second == 0
                    and instant.microsecond == 0, "Meters require half-hour boundaries")
            require(row["device_id"] in asset_ids, "Unknown meter asset")
            require(row["season"] in {"winter", "spring", "summer", "autumn"},
                    "Invalid season")
            value = float(row["consumption_kwh"])
            number(value)
            key = (row["device_id"], instant)
            require(key not in seen, "Duplicate meter interval")
            seen.add(key)
            rows.append({**row, "consumption_kwh": value})
        return rows
    except (csv.Error, ValueError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError("Invalid meter CSV") from None


def validate(document, direction="input"):
    """One validation boundary for both sides of the v1 envelope."""
    require(direction in {"input", "output"}, "Invalid validation direction")
    common = ["schema_version", "synthetic", "grid_assets", "outage_reports", "emissions"]
    keys(document, common + (["safety_rules", "meter_interval_csv", "scada_telemetry"]
                             if direction == "input" else
                             ["status", "meter_readings", "telemetry", "method"]))
    require(type(document["schema_version"]) is int and document["schema_version"] == 1,
            "Unsupported schema version")
    require(document["synthetic"] is True, "Only labeled synthetic data is supported")
    for field in ("grid_assets", "outage_reports", "emissions"):
        require(type(document[field]) is list, "Entity collection must be a list")
    for emission in document["emissions"]:
        keys(emission, ["value", "unit", "source"])
        number(emission["value"])
        require(emission["unit"] in ("kgCO2e", "tCO2e"), "Unsupported emissions unit")
        text(emission["source"])
    if direction == "output":
        require(document["status"] == "ok", "Invalid output status")
        for report in document["outage_reports"]:
            keys(report, ["report_ref", "asset_ref", "customers_affected", "outage_active",
                          "safety_flags", "sentiment", "priority", "priority_reason"])
            expected, reason = priority(report)
            require(report["priority"] == expected and report["priority_reason"] == reason,
                    "Output violates declared safety priority")
            score = report["sentiment"]
            require(score["score"] == sum(item["weight"] for item in score["matches"]),
                    "Invalid sentiment score")
        return document
    rules = document["safety_rules"]
    keys(rules, SAFETY_RULES)
    require(all(type(rules[k]) is int and rules[k] == v for k, v in SAFETY_RULES.items()),
            "Safety rules must preserve mandatory priority tiers")
    assets = {}
    for asset in document["grid_assets"]:
        keys(asset, ["asset_id", "kind", "critical", "upstream_id"])
        text(asset["asset_id"])
        require(re.fullmatch(r"SYN-[A-Z0-9-]+", asset["asset_id"]) is not None,
                "Asset identifier must use the fictitious SYN- namespace")
        require(asset["asset_id"] not in assets, "Duplicate grid asset")
        require(asset["kind"] in ("substation", "meter", "feeder"), "Invalid asset kind")
        require(type(asset["critical"]) is bool, "Critical flag must be boolean")
        if asset["upstream_id"] is not None:
            text(asset["upstream_id"])
        assets[asset["asset_id"]] = asset
    for asset_id in assets:
        visited, current = set(), asset_id
        while current is not None:
            require(current in assets, "Unknown upstream asset")
            require(current not in visited, "Grid topology contains a cycle")
            visited.add(current)
            current = assets[current]["upstream_id"]
    meter_ids = {key for key, value in assets.items() if value["kind"] == "meter"}
    meters = parse_meters(document["meter_interval_csv"], meter_ids)
    telemetry = document["scada_telemetry"]
    require(type(telemetry) is list, "Telemetry must be a list")
    seen_telemetry = set()
    for reading in telemetry:
        keys(reading, ["asset_id", "timestamp", "voltage_v", "state"])
        text(reading["asset_id"])
        require(reading["asset_id"] in assets, "Unknown telemetry asset")
        instant = timestamp(reading["timestamp"])
        number(reading["voltage_v"])
        require(reading["state"] in ("normal", "alarm", "offline"), "Invalid telemetry state")
        key = (reading["asset_id"], instant)
        require(key not in seen_telemetry, "Duplicate telemetry reading")
        seen_telemetry.add(key)
    seen_reports = set()
    for report in document["outage_reports"]:
        keys(report, ["report_id", "asset_id", "description", "customers_affected",
                      "outage_active", "safety_flags"])
        text(report["report_id"])
        text(report["asset_id"])
        text(report["description"])
        require(report["report_id"] not in seen_reports, "Duplicate outage report")
        seen_reports.add(report["report_id"])
        require(report["asset_id"] in assets, "Unknown outage asset")
        require(type(report["customers_affected"]) is int and
                report["customers_affected"] >= 0, "Invalid affected customer count")
        require(type(report["outage_active"]) is bool, "Outage active must be boolean")
        flags = report["safety_flags"]
        require(type(flags) is list and all(type(flag) is str and flag in
                ("fire", "downed_line", "medical_dependency") for flag in flags),
                "Invalid safety flags")
        require(len(flags) == len(set(flags)), "Duplicate safety flag")
    return assets, meters


def run(document):
    assets, meters = validate(document)
    aliases = {identifier: f"asset-{index:04d}"
               for index, identifier in enumerate(sorted(assets), 1)}
    # Sources retain provenance without exposing a known asset identifier.
    def protect_source(source):
        for identifier in sorted(aliases, key=len, reverse=True):
            source = re.sub(re.escape(identifier), "[protected-asset]", source,
                            flags=re.IGNORECASE)
        return source

    reports = []
    for index, report in enumerate(document["outage_reports"], 1):
        tier, reason = priority(report)
        reports.append({"report_ref": f"report-{index:04d}",
                        "asset_ref": aliases[report["asset_id"]],
                        "customers_affected": report["customers_affected"],
                        "outage_active": report["outage_active"],
                        "safety_flags": sorted(report["safety_flags"]),
                        "sentiment": sentiment(report["description"]),
                        "priority": tier, "priority_reason": reason})
    reports.sort(key=lambda r: (r["priority"], -r["customers_affected"],
                               r["sentiment"]["score"], r["report_ref"]))
    output = {
        "schema_version": 1, "synthetic": True, "status": "ok",
        "method": {"sentiment": "fixed lexicon sum; immediate-token negation",
                   "ranking": "safety tier, affected customers descending, sentiment ascending, report_ref",
                   "identifier_protection": "request-local aliases; descriptions omitted",
                   "notice": "Synthetic demonstration only; no NERC CIP certification"},
        "grid_assets": [{"asset_ref": aliases[k], "kind": assets[k]["kind"],
                         "critical": assets[k]["critical"],
                         "upstream_ref": aliases.get(assets[k]["upstream_id"])}
                        for k in sorted(assets)],
        "meter_readings": [{"asset_ref": aliases[r["device_id"]], "timestamp": r["timestamp"],
                            "consumption": {"value": r["consumption_kwh"], "unit": "kWh"},
                            "interval_minutes": 30, "season": r["season"]} for r in meters],
        "telemetry": [{"asset_ref": aliases[r["asset_id"]], "timestamp": r["timestamp"],
                       "voltage": {"value": r["voltage_v"], "unit": "V"}, "state": r["state"]}
                      for r in document["scada_telemetry"]],
        "outage_reports": reports,
        "emissions": [{**e, "source": protect_source(e["source"])}
                      for e in document["emissions"]]}
    validate(output, "output")
    return output


def reject_constant(value):
    raise ValidationError("Non-finite JSON constant")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Expected one JSON input path")
        with Path(args[0]).open(encoding="utf-8") as stream:
            document = json.load(stream, parse_constant=reject_constant,
                                 object_pairs_hook=unique_object)
        output = run(document)
        print(json.dumps(output, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        # Never echo file paths, identifiers, descriptions, or parser internals.
        print(json.dumps({"status": "error", "error": "Invalid input or unreadable file"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
