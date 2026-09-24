"""Synthetic guided onboarding reference; Python standard library only.

Input schema version 1 contains profile, smart-meter CSV, SCADA telemetry and
outages. All validation precedes planning. Identifiers are pseudonymized in
output, not encrypted; this demonstrative control is not NERC certification.
"""

import csv
import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime, timedelta


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, label):
    require(isinstance(value, dict) and set(value) == set(required),
            label + " has missing or unsupported fields")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 200,
            label + " must be nonempty text of at most 200 characters")
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            label + " must be finite and nonnegative")
    return value


def synthetic_id(value, label):
    text(value, label)
    require(re.fullmatch(r"SYN-[A-Z][A-Z0-9-]*", value) is not None,
            label + " must be a synthetic SYN- prefixed identifier")
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("timestamp must be ISO 8601") from None
    require(result.tzinfo is not None and result.utcoffset() == timedelta(0),
            "timestamp must include UTC timezone")
    return result


def token(kind, identifier):
    return kind + "-" + hashlib.sha256((kind + ":" + identifier).encode()).hexdigest()[:16]


def modules(profile):
    rows = [
        ("protection", [], "Handle critical asset identifiers only in authorized systems.",
         "Identifiers are pseudonymized in this output; source fixtures still contain them."),
    ]
    meter_prereq = ["protection"]
    if profile["experience"] == "novice":
        rows.append(("seasonal_basics", ["protection"],
                     "Learn how seasons affect electricity consumption.",
                     "Synthetic Northern Hemisphere data can peak during winter heating or summer cooling."))
        meter_prereq.append("seasonal_basics")
    rows.extend([
        ("meter_readings", meter_prereq, "Interpret interval energy readings.",
         "Consumption is energy in kWh for each declared interval, not instantaneous power."),
        ("grid_assets", ["protection"], "Inspect the synthetic grid topology.",
         "SCADA voltage uses kV; emissions require an explicit unit and traceable source."),
        ("safety_rules", ["protection"], "Learn the declared outage safety rules.",
         "P1: life threat; otherwise P2: critical service; otherwise P3: at least 100 customers; otherwise P4."),
        ("outage_triage", ["grid_assets", "safety_rules"], "Apply safety-first outage prioritization.",
         "Experience and presentation preferences never change priority or waive safety prerequisites."),
    ])
    return rows


def validate(data):
    """Single validation boundary shared by the CLI and Python API."""
    keys(data, ["schema_version", "synthetic", "profile", "meter_readings",
                "grid_assets", "outage_reports"], "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "only explicitly synthetic fixtures are supported")
    profile = data["profile"]
    keys(profile, ["experience", "preference", "completed"], "profile")
    require(profile["experience"] in ("novice", "intermediate", "expert"),
            "unsupported experience")
    require(profile["preference"] in ("concise", "detailed"), "unsupported preference")
    completed = profile["completed"]
    require(isinstance(completed, list) and all(isinstance(x, str) for x in completed),
            "completed must be a list of module names")
    require(len(completed) == len(set(completed)), "completed contains duplicates")
    catalog = modules(profile)
    require(set(completed) <= {r[0] for r in catalog}, "completed contains unknown modules")
    for name, prerequisites, _, _ in catalog:
        if name in completed:
            require(set(prerequisites) <= set(completed),
                    "completed modules must include all prerequisites")

    scada = data["grid_assets"]
    keys(scada, ["format", "synthetic", "assets"], "grid_assets")
    require(scada["format"] == "SCADA-style telemetry JSON" and scada["synthetic"] is True,
            "grid_assets must be labeled synthetic SCADA-style telemetry JSON")
    assets = scada["assets"]
    require(isinstance(assets, list) and 0 < len(assets) <= 1000,
            "assets must contain 1 to 1000 entries")
    asset_ids = set()
    for asset in assets:
        keys(asset, ["asset_id", "critical", "neighbors", "timestamp", "voltage_kv",
                     "emissions"], "asset")
        identifier = synthetic_id(asset["asset_id"], "asset_id")
        require(identifier not in asset_ids, "duplicate asset identifier")
        asset_ids.add(identifier)
        require(type(asset["critical"]) is bool, "critical must be boolean")
        require(isinstance(asset["neighbors"], list)
                and all(isinstance(n, str) for n in asset["neighbors"]),
                "neighbors must be identifier strings")
        require(len(asset["neighbors"]) == len(set(asset["neighbors"])),
                "duplicate neighbors")
        timestamp(asset["timestamp"])
        number(asset["voltage_kv"], "voltage_kv")
        keys(asset["emissions"], ["value", "unit", "source"], "emissions")
        number(asset["emissions"]["value"], "emissions value")
        require(asset["emissions"]["unit"] in ("kgCO2e", "tCO2e"),
                "emissions unit must be kgCO2e or tCO2e")
        text(asset["emissions"]["source"], "emissions source")
    adjacency = {a["asset_id"]: set(a["neighbors"]) for a in assets}
    for identifier, neighbors in adjacency.items():
        require(neighbors <= asset_ids and identifier not in neighbors,
                "topology contains unknown or self-linked assets")
        require(all(identifier in adjacency[n] for n in neighbors),
                "synthetic topology links must be reciprocal")

    meter = data["meter_readings"]
    keys(meter, ["format", "interval_minutes", "csv"], "meter_readings")
    require(meter["format"] == "smart-meter interval CSV", "unsupported meter format")
    require(type(meter["interval_minutes"]) is int
            and meter["interval_minutes"] in (15, 30, 60), "unsupported meter interval")
    require(isinstance(meter["csv"], str) and len(meter["csv"]) <= 1000000,
            "meter csv must be text of at most 1000000 characters")
    reader = csv.DictReader(io.StringIO(meter["csv"]), strict=True)
    fields = ["timestamp", "meter_id", "asset_id", "consumption_kwh", "season"]
    rows = []
    last = {}
    meter_asset = {}
    try:
        require(reader.fieldnames == fields, "meter CSV header is invalid")
        for row in reader:
            require(set(row) == set(fields) and all(v is not None for v in row.values()),
                    "meter CSV row shape is invalid")
            when = timestamp(row["timestamp"])
            meter_id = synthetic_id(row["meter_id"], "meter_id")
            require(row["asset_id"] in asset_ids, "meter references an unknown asset")
            try:
                consumption = float(row["consumption_kwh"])
            except ValueError:
                raise ValidationError("consumption_kwh must be numeric") from None
            number(consumption, "consumption_kwh")
            expected_season = ("winter" if when.month in (12, 1, 2) else
                               "spring" if when.month in (3, 4, 5) else
                               "summer" if when.month in (6, 7, 8) else "autumn")
            require(row["season"] == expected_season,
                    "season must match the synthetic Northern Hemisphere calendar")
            if meter_id in last:
                require(when - last[meter_id] == timedelta(minutes=meter["interval_minutes"]),
                        "each meter must have consecutive ordered intervals")
                require(meter_asset[meter_id] == row["asset_id"],
                        "a meter cannot change assets within the fixture")
            last[meter_id] = when
            meter_asset[meter_id] = row["asset_id"]
            rows.append(dict(row, consumption_kwh=consumption))
    except csv.Error:
        raise ValidationError("meter CSV is malformed") from None
    require(bool(rows), "meter CSV must contain readings")

    outages = data["outage_reports"]
    require(isinstance(outages, list) and len(outages) <= 1000,
            "outage_reports must be a list with at most 1000 entries")
    outage_ids = set()
    for report in outages:
        keys(report, ["outage_id", "asset_id", "life_threat", "critical_service",
                      "customers_affected", "priority"], "outage report")
        oid = synthetic_id(report["outage_id"], "outage_id")
        require(oid not in outage_ids, "duplicate outage identifier")
        outage_ids.add(oid)
        require(report["asset_id"] in asset_ids, "outage references an unknown asset")
        require(type(report["life_threat"]) is bool
                and type(report["critical_service"]) is bool, "safety flags must be boolean")
        require(type(report["customers_affected"]) is int
                and report["customers_affected"] >= 0, "customers_affected must be nonnegative integer")
        expected = priority(report)
        require(report["priority"] == expected, "outage priority violates declared safety rules")
    return profile, assets, rows, outages, catalog


def priority(report):
    return ("P1" if report["life_threat"] else
            "P2" if report["critical_service"] else
            "P3" if report["customers_affected"] >= 100 else "P4")


def run(data):
    profile, assets, readings, outages, catalog = validate(data)
    completed = set(profile["completed"])
    steps = []
    for name, prerequisites, short, detail in catalog:
        missing = [p for p in prerequisites if p not in completed]
        state = "completed" if name in completed else "locked" if missing else "available"
        explanation = short + (" " + detail if profile["preference"] == "detailed" else "")
        steps.append({"module": name, "prerequisites": prerequisites,
                      "unmet_prerequisites": missing, "status": state,
                      "explanation": explanation})
    next_module = next((s["module"] for s in steps if s["status"] == "available"), None)
    identifiers = {a["asset_id"] for a in assets}
    identifiers.update(r["meter_id"] for r in readings)
    identifiers.update(r["outage_id"] for r in outages)

    def safe_source(source):
        # Free-text provenance can itself contain asset identifiers.
        for identifier in sorted(identifiers, key=lambda x: (-len(x), x)):
            source = source.replace(identifier, "[protected identifier]")
        return source

    normalized_assets = [
        {"asset_ref": token("asset", a["asset_id"]), "critical": a["critical"],
         "neighbors": [token("asset", n) for n in a["neighbors"]],
         "timestamp": a["timestamp"], "voltage_kv": a["voltage_kv"],
         "emissions": dict(a["emissions"], source=safe_source(a["emissions"]["source"]))}
        for a in assets
    ]
    return {
        "schema_version": 1, "status": "ok", "synthetic": True,
        "protection": "Demonstrative pseudonymization only; not NERC CIP certification or encryption.",
        "onboarding": {"experience": profile["experience"], "preference": profile["preference"],
                       "steps": steps, "next_module": next_module,
                       "ready": all(s["status"] == "completed" for s in steps)},
        "entities": {
            "meter_readings": [
                {"meter_ref": token("meter", r["meter_id"]),
                 "asset_ref": token("asset", r["asset_id"]), "timestamp": r["timestamp"],
                 "consumption": {"value": r["consumption_kwh"], "unit": "kWh"},
                 "interval_minutes": data["meter_readings"]["interval_minutes"],
                 "season": r["season"]} for r in readings],
            "grid_assets": normalized_assets,
            "outage_reports": [
                {"outage_ref": token("outage", r["outage_id"]),
                 "asset_ref": token("asset", r["asset_id"]),
                 "priority": priority(r), "customers_affected": r["customers_affected"],
                 "safety_basis": ("life threat" if r["life_threat"] else
                                  "critical service" if r["critical_service"] else
                                  "100 or more customers" if r["customers_affected"] >= 100 else
                                  "routine interruption")}
                for r in sorted(outages, key=lambda r: (r["priority"], r["outage_id"]))]
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON contains duplicate fields")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object)
        result = run(data)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError):
        result = {"schema_version": 1, "status": "error", "message": "Cannot read valid UTF-8 JSON input"}
    except ValidationError as exc:
        result = {"schema_version": 1, "status": "error", "message": str(exc)}
    except (TypeError, ValueError, OverflowError, RecursionError):
        result = {"schema_version": 1, "status": "error", "message": "Invalid input structure or value"}
    print(json.dumps(result, sort_keys=True))
    return 2


if __name__ == "__main__":
    sys.exit(main())
