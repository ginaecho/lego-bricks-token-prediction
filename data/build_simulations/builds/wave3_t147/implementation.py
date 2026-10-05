"""Synthetic utility-product comparison. Standard library; no external services."""
import csv
import hashlib
import hmac
import io
import json
import math
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


METRICS = {
    "monthly_cost_usd": "min",
    "consumption_kwh": "min",
    "emissions_kgCO2e": "min",
    "availability_percent": "max",
    "outage_risk": "min",
}
SAFETY_RULES = {
    "life_safety": "emergency",
    "essential_service": "urgent",
    "otherwise": "routine",
}
RISK = {"routine": 1, "urgent": 2, "emergency": 3}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(obj, required, optional=()):
    require(isinstance(obj, dict), "Expected an object")
    require(set(required) <= set(obj) <= set(required) | set(optional),
            "Missing or unsupported schema fields")


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected nonempty text")
    return value.strip()


def number(value, upper=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Expected a finite nonnegative number")
    require(upper is None or value <= upper, "Number exceeds allowed range")
    return float(value)


def array(value, nonempty=False):
    require(isinstance(value, list) and (value or not nonempty), "Expected a list")
    return value


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result

    def bad_constant(_):
        raise ValidationError("Nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def normalize(candidate, key, interval_minutes):
    fields(candidate, ("name", "monthly_cost_usd", "grid_assets", "topology",
                       "meter_csv", "telemetry_json", "outage_reports"))
    name = text(candidate["name"])
    cost = number(candidate["monthly_cost_usd"])
    assets = {}
    for asset in array(candidate["grid_assets"], True):
        fields(asset, ("asset_id", "kind", "critical"))
        aid = text(asset["asset_id"])
        require(aid not in assets, "Duplicate grid asset")
        require(asset["kind"] in ("substation", "meter", "device"), "Invalid asset kind")
        require(type(asset["critical"]) is bool, "Critical flag must be boolean")
        assets[aid] = asset
    # All identifiers, not only critical ones, are replaced before serialization.
    def public_id(aid):
        return "asset_" + hmac.new(key.encode(), aid.encode(), hashlib.sha256).hexdigest()[:24]

    topology = []
    edges = set()
    for edge in array(candidate["topology"]):
        fields(edge, ("from", "to"))
        left, right = text(edge["from"]), text(edge["to"])
        require(left in assets and right in assets and left != right, "Invalid topology reference")
        pair = tuple(sorted((left, right)))
        require(pair not in edges, "Duplicate topology connection")
        edges.add(pair)
        topology.append({"from": public_id(left), "to": public_id(right)})

    raw_csv = text(candidate["meter_csv"])
    reader = csv.DictReader(io.StringIO(raw_csv), strict=True)
    require(reader.fieldnames == ["device_id", "timestamp", "consumption", "unit"],
            "Invalid smart-meter CSV header")
    totals, timelines, readings = [], {}, []
    factors = {"Wh": 0.001, "kWh": 1.0, "MWh": 1000.0}
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
        aid = text(row["device_id"])
        require(aid in assets and assets[aid]["kind"] == "meter", "Unknown meter reference")
        stamp = datetime.fromisoformat(text(row["timestamp"]).replace("Z", "+00:00"))
        require(stamp.utcoffset() is not None, "Meter timestamps need a timezone")
        epoch = stamp.timestamp()
        require(row["unit"] in factors, "Unsupported consumption unit")
        energy = number(float(row["consumption"])) * factors[row["unit"]]
        number(energy)
        timeline = timelines.setdefault(aid, set())
        require(epoch not in timeline, "Duplicate meter interval")
        timeline.add(epoch)
        totals.append(energy)
        readings.append({"device_ref": public_id(aid), "timestamp": stamp.isoformat(),
                         "consumption_kwh": energy})
    meters = {aid for aid, asset in assets.items() if asset["kind"] == "meter"}
    require(meters and set(timelines) == meters, "Every meter needs interval readings")
    timeline = sorted(next(iter(timelines.values())))
    require(all(sorted(t) == timeline for t in timelines.values()), "Meter coverage must match")
    require(all(b - a == interval_minutes * 60 for a, b in zip(timeline, timeline[1:])),
            "Meter intervals must be contiguous")

    telemetry = strict_json(text(candidate["telemetry_json"]))
    fields(telemetry, ("synthetic", "samples"))
    require(telemetry["synthetic"] is True, "Telemetry must be labeled synthetic")
    availability, emissions, sources, seen = [], [], [], set()
    emission_factors = {"gCO2e": 0.001, "kgCO2e": 1.0, "tCO2e": 1000.0}
    for sample in array(telemetry["samples"], True):
        fields(sample, ("asset_id", "availability_percent", "emissions"))
        aid = text(sample["asset_id"])
        require(aid in assets and aid not in seen, "Invalid telemetry asset reference")
        seen.add(aid)
        availability.append(number(sample["availability_percent"], 100))
        emission = sample["emissions"]
        fields(emission, ("value", "unit", "source"))
        require(emission["unit"] in emission_factors, "Unsupported emissions unit")
        value = number(emission["value"]) * emission_factors[emission["unit"]]
        number(value)
        emissions.append(value)
        sources.append({"asset_ref": public_id(aid), "value": value,
                        "unit": "kgCO2e", "source": text(emission["source"])})
    require(seen == set(assets), "Telemetry must cover every grid asset")
    outages = []
    for outage in array(candidate["outage_reports"]):
        fields(outage, ("asset_id", "life_safety", "essential_service", "priority"))
        aid = text(outage["asset_id"])
        require(aid in assets, "Unknown outage asset reference")
        require(type(outage["life_safety"]) is bool and type(outage["essential_service"]) is bool,
                "Safety flags must be boolean")
        expected = ("emergency" if outage["life_safety"] else
                    "urgent" if outage["essential_service"] else "routine")
        require(outage["priority"] == expected, "Outage priority violates declared safety rules")
        outages.append({"asset_ref": public_id(aid), "priority": expected,
                        "life_safety": outage["life_safety"],
                        "essential_service": outage["essential_service"]})
    metrics = {"monthly_cost_usd": cost, "consumption_kwh": math.fsum(totals),
               "emissions_kgCO2e": math.fsum(emissions),
               "availability_percent": math.fsum(availability) / len(availability),
               "outage_risk": max((RISK[o["priority"]] for o in outages), default=0)}
    for value in metrics.values():
        number(value)
    result = {"name": name, "attributes": metrics, "meter_readings": readings,
              "grid_assets": [{"asset_ref": public_id(aid), "kind": a["kind"],
                               "critical": a["critical"]} for aid, a in assets.items()],
              "topology": topology, "outage_reports": outages, "emissions": sources}
    # Free-form labels/sources must not accidentally expose any known raw identifier or key.
    serialized = json.dumps(result)
    require(key not in serialized and all(aid not in serialized for aid in assets),
            "Public text contains a protected identifier or secret")
    return result, timeline, set(assets)


def compare(data):
    fields(data, ("schema_version", "synthetic", "protection_key", "interval_minutes",
                  "safety_rules", "preferences", "candidates"), ("query",))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Unsupported schema version")
    require(data["synthetic"] is True, "Input must be labeled synthetic")
    key = text(data["protection_key"])
    require(len(key) >= 16, "Protection key must contain at least 16 characters")
    require(type(data["interval_minutes"]) is int and 1 <= data["interval_minutes"] <= 1440,
            "Invalid interval length")
    require(data["safety_rules"] == SAFETY_RULES, "Unsupported declared safety rules")
    weights = data["preferences"]
    require(isinstance(weights, dict) and weights and set(weights) <= set(METRICS),
            "Preferences must name supported metrics")
    weights = {k: number(v) for k, v in weights.items()}
    weight_sum = math.fsum(weights.values())
    require(math.isfinite(weight_sum) and weight_sum > 0, "At least one positive preference required")
    rows, names, reference, protected = [], set(), None, set()
    for candidate in array(data["candidates"], True):
        row, timeline, identifiers = normalize(candidate, key, data["interval_minutes"])
        require(row["name"].casefold() not in names, "Product names must be unique")
        names.add(row["name"].casefold())
        require(reference is None or reference == timeline, "Product observation windows must match")
        reference = timeline
        rows.append(row)
        protected.update(identifiers)
    query = data.get("query", "")
    require(isinstance(query, str), "Query must be text")
    rows = [r for r in rows if query.strip().casefold() in r["name"].casefold()]
    comparison = {metric: [{"name": row["name"], "value": row["attributes"][metric]}
                           for row in rows] for metric in METRICS}
    ranking = []
    for row in rows:
        components = {}
        for metric, weight in weights.items():
            values = [r["attributes"][metric] for r in rows]
            low, high = min(values), max(values)
            utility = 1.0 if low == high else (row["attributes"][metric] - low) / (high - low)
            if high != low and METRICS[metric] == "min":
                utility = 1 - utility
            components[metric] = utility * (weight / weight_sum)
        ranking.append({"name": row["name"], "score": round(math.fsum(components.values()), 8),
                        "weighted_utilities": components})
    ranking.sort(key=lambda r: (-r["score"], r["name"].casefold()))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    result = {"status": "ok", "schema_version": 1, "synthetic": True,
              "products": rows, "side_by_side": comparison, "ranking": ranking,
              "safety_rules": SAFETY_RULES,
              "method": "Weighted min-max utility; ties alphabetical; equal metrics have utility 1",
              "notice": "Synthetic demonstration only; no NERC CIP compliance certification"}
    serialized = json.dumps(result, allow_nan=False)
    require(key not in serialized and all(aid not in serialized for aid in protected),
            "Public output contains protected data")
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Expected one input JSON file")
        with open(argv[0], encoding="utf-8") as handle:
            result = compare(strict_json(handle.read()))
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, OverflowError, csv.Error, RecursionError):
        # Never echo input values, raw identifiers, keys, file contents or paths.
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable file"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
