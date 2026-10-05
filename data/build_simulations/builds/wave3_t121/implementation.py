"""Synthetic energy onboarding reference. No compliance certification is implied."""

import csv
import io
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


STEPS = (
    "protect_identifiers",
    "register_grid",
    "import_meter_readings",
    "import_telemetry",
    "review_outages",
    "finish",
)
SAFETY_RULES = {
    "danger_to_life": "critical",
    "essential_service": "high",
    "ordinary_interruption": "normal",
}
CSV_FIELDS = ["meter_id", "asset_id", "timestamp", "interval_minutes", "energy_kwh"]
MAX_BYTES = 1_000_000


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, where):
    require(isinstance(value, dict) and set(value) == set(names),
            where + ": invalid fields")


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 200,
            where + ": expected nonempty bounded text")


def number(value, where, minimum=0, maximum=1e9):
    require(type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum, where + ": invalid number")


def timestamp(value, where):
    require(isinstance(value, str), where + ": invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
                where + ": timezone required")
        return parsed
    except (ValueError, OverflowError):
        raise ValidationError(where + ": invalid timestamp") from None


def identifier(value, pattern, where):
    require(isinstance(value, str) and re.fullmatch(pattern, value) is not None,
            where + ": expected fictitious identifier")


def validate(payload):
    """One validation boundary for CLI and Python callers; return normalized entities."""
    fields(payload, ["schema_version", "synthetic", "protection", "safety_rules",
                     "grid_assets", "meter_interval_csv", "telemetry",
                     "outage_reports", "completed_steps", "actions"], "input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "unsupported schema version")
    require(payload["synthetic"] is True, "synthetic fixtures are required")
    fields(payload["protection"], ["critical_identifier_policy", "acknowledged"],
           "protection")
    require(payload["protection"]["critical_identifier_policy"] == "pseudonymize",
            "critical identifiers must be protected")
    require(type(payload["protection"]["acknowledged"]) is bool,
            "protection acknowledgement must be boolean")
    require(payload["safety_rules"] == SAFETY_RULES,
            "declared safety rules must match the demonstrative safety policy")
    for name in ("grid_assets", "telemetry", "outage_reports", "completed_steps", "actions"):
        require(isinstance(payload[name], list), name + ": expected list")

    assets = {}
    for asset in payload["grid_assets"]:
        fields(asset, ["asset_id", "kind", "critical", "parent_id"], "grid asset")
        aid = asset["asset_id"]
        identifier(aid, r"(SUB|DEV)-SYN-[0-9]{3}", "grid asset")
        require(aid not in assets, "duplicate grid asset")
        require(asset["kind"] in ("substation", "device"), "invalid grid asset kind")
        require(aid.startswith("SUB-" if asset["kind"] == "substation" else "DEV-"),
                "grid asset kind and identifier disagree")
        require(type(asset["critical"]) is bool, "critical flag must be boolean")
        parent = asset["parent_id"]
        require(parent is None or isinstance(parent, str), "invalid grid parent")
        require((asset["kind"] == "substation" and parent is None)
                or (asset["kind"] == "device" and parent is not None),
                "substations must be roots and devices must have a parent")
        assets[aid] = asset
    for aid, asset in assets.items():
        parent = asset["parent_id"]
        if parent is not None:
            require(parent in assets and assets[parent]["kind"] == "substation",
                    "device must reference an existing substation")
    aliases = {aid: "asset_%03d" % i for i, aid in enumerate(sorted(assets), 1)}

    raw_csv = payload["meter_interval_csv"]
    require(isinstance(raw_csv, str) and len(raw_csv) <= MAX_BYTES, "invalid meter CSV")
    readings, seen, meter_assets = [], set(), {}
    try:
        reader = csv.DictReader(io.StringIO(raw_csv), strict=True)
        require(reader.fieldnames == CSV_FIELDS, "invalid meter CSV header")
        for row in reader:
            require(set(row) == set(CSV_FIELDS) and all(v is not None for v in row.values()),
                    "invalid meter CSV row")
            identifier(row["meter_id"], r"MTR-SYN-[0-9]{3}", "meter")
            require(row["asset_id"] in assets, "meter references unknown grid asset")
            moment = timestamp(row["timestamp"], "meter")
            require(row["interval_minutes"] in ("15", "30", "60"),
                    "meter interval must be 15, 30 or 60 minutes")
            interval = int(row["interval_minutes"])
            require(moment.minute % interval == 0 and moment.second == 0
                    and moment.microsecond == 0, "meter timestamp is not interval aligned")
            try:
                energy = float(row["energy_kwh"])
            except ValueError:
                raise ValidationError("invalid meter energy") from None
            number(energy, "meter energy", maximum=10000)
            key = (row["meter_id"], moment)
            require(key not in seen, "duplicate meter interval")
            seen.add(key)
            previous = meter_assets.setdefault(row["meter_id"], row["asset_id"])
            require(previous == row["asset_id"], "meter cannot change grid asset")
            readings.append({"meter_id": row["meter_id"], "asset_id": row["asset_id"],
                             "timestamp": moment.isoformat(), "interval_minutes": interval,
                             "energy_kwh": energy})
    except csv.Error:
        raise ValidationError("invalid meter CSV syntax") from None

    telemetry = []
    telemetry_keys = set()
    for item in payload["telemetry"]:
        fields(item, ["asset_id", "timestamp", "voltage_kv", "emissions"], "telemetry")
        require(isinstance(item["asset_id"], str) and item["asset_id"] in assets,
                "telemetry references unknown grid asset")
        moment = timestamp(item["timestamp"], "telemetry")
        key = (item["asset_id"], moment)
        require(key not in telemetry_keys, "duplicate telemetry sample")
        telemetry_keys.add(key)
        number(item["voltage_kv"], "voltage", maximum=1000)
        emission = item["emissions"]
        fields(emission, ["value", "unit", "source"], "emissions")
        number(emission["value"], "emissions")
        require(emission["unit"] in ("kgCO2e", "tCO2e"), "unsupported emissions unit")
        text(emission["source"], "emissions source")
        require(emission["source"] == "synthetic-seasonal-model",
                "emissions source must identify the synthetic model")
        telemetry.append(item)

    outages, outage_ids = [], set()
    for report in payload["outage_reports"]:
        fields(report, ["report_id", "asset_id", "danger_to_life",
                        "essential_service", "priority"], "outage report")
        identifier(report["report_id"], r"OUT-SYN-[0-9]{3}", "outage")
        require(report["report_id"] not in outage_ids, "duplicate outage report")
        outage_ids.add(report["report_id"])
        require(isinstance(report["asset_id"], str) and report["asset_id"] in assets,
                "outage references unknown grid asset")
        require(type(report["danger_to_life"]) is bool
                and type(report["essential_service"]) is bool,
                "outage safety flags must be boolean")
        reason = ("danger_to_life" if report["danger_to_life"] else
                  "essential_service" if report["essential_service"] else
                  "ordinary_interruption")
        require(report["priority"] == SAFETY_RULES[reason],
                "outage priority violates declared safety rules")
        outages.append(dict(report, safety_reason=reason))

    completed = payload["completed_steps"]
    require(len(completed) <= len(STEPS) and completed == list(STEPS[:len(completed)]),
            "completed steps must be an ordered prerequisite-satisfied prefix")
    requirements = [
        payload["protection"]["acknowledged"], bool(assets), bool(readings),
        bool(telemetry), True, True,
    ]
    for index in range(len(completed)):
        require(requirements[index], "completed step lacks its validated prerequisite data")
    for action in payload["actions"]:
        require(isinstance(action, str) and action in STEPS, "unknown onboarding action")
    return assets, aliases, readings, telemetry, outages, requirements


def run(payload):
    assets, aliases, readings, telemetry, outages, requirements = validate(payload)
    completed = list(payload["completed_steps"])
    transitions = []
    for action in payload["actions"]:
        require(action not in completed, "onboarding action is already completed")
        require(len(completed) < len(STEPS) and action == STEPS[len(completed)],
                "onboarding action has unmet prerequisites")
        require(requirements[len(completed)], "onboarding action lacks required data")
        completed.append(action)
        transitions.append({"step": action, "state": "completed"})
    next_step = STEPS[len(completed)] if len(completed) < len(STEPS) else None
    meter_aliases = {mid: "meter_%03d" % i for i, mid in
                     enumerate(sorted({r["meter_id"] for r in readings}), 1)}
    public_readings = [dict(r, asset_id=aliases[r["asset_id"]],
                            meter_id=meter_aliases[r["meter_id"]]) for r in readings]
    return {
        "schema_version": 1,
        "status": "complete" if next_step is None else "in_progress",
        "synthetic": True,
        "notice": "Synthetic demonstration; not NERC CIP certification. "
                  "Aliases are output minimization, not encryption or access control.",
        "protection": {"critical_identifier_policy": "pseudonymize"},
        "safety_rules": dict(SAFETY_RULES),
        "progress": {
            "completed_steps": completed, "completed_count": len(completed),
            "total_steps": len(STEPS), "percent": round(100 * len(completed) / len(STEPS), 2),
            "next_step": next_step,
            "next_step_ready": next_step is not None and requirements[len(completed)],
            "steps": [{"step": step, "prerequisites": list(STEPS[:i]),
                       "state": "completed" if i < len(completed) else
                       "available" if i == len(completed) and requirements[i] else "blocked"}
                      for i, step in enumerate(STEPS)],
            "transitions": transitions,
        },
        "entities": {
            "grid_assets": [{"asset_id": aliases[aid], "kind": asset["kind"],
                             "critical": asset["critical"],
                             "parent_id": aliases.get(asset["parent_id"])}
                            for aid, asset in sorted(assets.items())],
            "meter_readings": public_readings,
            "telemetry": [dict(item, asset_id=aliases[item["asset_id"]])
                          for item in telemetry],
            "outage_reports": [dict(report, asset_id=aliases[report["asset_id"]])
                               for report in outages],
        },
    }


def reject_constant(_):
    raise ValidationError("nonfinite JSON numbers are forbidden")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with Path(args[0]).open("rb") as source:
            raw = source.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "input exceeds size limit")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = run(payload)
    except ValidationError as error:
        result = {"status": "error", "error": str(error)}
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        result = {"status": "error", "error": "Unable to read input or invalid input structure"}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 2 if result["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
