"""Deterministic synthetic energy workflow; Python standard library only.

Run: python -B implementation.py example_input.json
Asset aliases are demonstrative data minimization, not access control or NERC
certification. All stages exchange and validate the same cumulative envelope.
"""

import copy
import csv
import io
import json
import math
import re
import sys
from datetime import datetime


VERSION = "1.0"
STAGES = ("normalized", "research", "search", "documents", "insights")
PRIORITIES = ("P1", "P2", "P3")
SYNONYMS = (
    {"power", "electricity", "energy"},
    {"outage", "outages", "blackout", "interruption"},
    {"meter", "meters", "metering"},
    {"usage", "consumption", "demand"},
    {"repair", "restore", "restoration", "recovery"},
)
STOP = {"a", "an", "the", "and", "to", "for", "of", "in", "is", "with", "how"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be text")
    return value


def number(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            label + " must be a finite nonnegative number")
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid timestamp") from exc
    require(parsed.tzinfo is not None, "timestamps require timezone")
    return parsed


def records(value, label, nonempty=False):
    require(isinstance(value, list), label + " must be an array")
    require(not nonempty or bool(value), label + " cannot be empty")
    require(all(isinstance(item, dict) for item in value), label + " needs objects")
    return value


def indexed(items, key, label):
    result = {}
    for item in items:
        identity = text(item.get(key), label + " identifier")
        require(identity not in result, label + " identifiers must be unique")
        result[identity] = item
    return result


def priority(outage):
    if outage["life_safety"]:
        return "P1"
    if outage["critical_service"] or outage["customers_affected"] >= 100:
        return "P2"
    return "P3"


def emissions(value):
    require(isinstance(value, dict), "emissions must be an object")
    number(value.get("value"), "emissions value")
    require(value.get("unit") in ("kgCO2e", "tCO2e"), "unsupported emissions unit")
    text(value.get("source"), "emissions source")


def terms(value):
    words = set(re.findall(r"[a-z0-9]+", value.lower())) - STOP
    for group in SYNONYMS:
        if words & group:
            words |= group
    return words


def validate_dataset(data):
    require(isinstance(data, dict), "data must be an object")
    require(data.get("synthetic") is True, "fixtures must be labeled synthetic")
    text(data.get("question"), "question")
    assets = indexed(records(data.get("assets"), "assets", True), "asset_id", "assets")
    for asset in assets.values():
        text(asset.get("kind"), "asset kind")
        require(type(asset.get("critical")) is bool, "critical must be boolean")
        parent = asset.get("parent_id")
        require(parent is None or parent in assets, "unknown topology parent")
        seen = {asset["asset_id"]}
        while parent is not None:
            require(parent not in seen, "cyclic grid topology")
            seen.add(parent)
            parent = assets[parent].get("parent_id")
            require(parent is None or parent in assets, "unknown topology parent")
    sources = indexed(records(data.get("sources"), "sources"), "id", "sources")
    for source in sources.values():
        text(source.get("text"), "source text")
        if "emissions" in source:
            emissions(source["emissions"])
    products = indexed(records(data.get("products"), "products"), "id", "products")
    for product in products.values():
        text(product.get("name"), "product name")
        text(product.get("description"), "product description")
        for field in ("tags", "asset_kinds", "priorities"):
            require(isinstance(product.get(field), list) and
                    all(isinstance(v, str) and v.strip() for v in product[field]),
                    "product " + field + " must be text arrays")
        require(bool(product["asset_kinds"]), "product needs asset kinds")
        require(bool(product["priorities"]) and
                set(product["priorities"]) <= set(PRIORITIES), "invalid product priorities")
    seen = set()
    for reading in records(data.get("meter_readings"), "meter readings"):
        require(reading.get("asset_id") in assets, "unknown meter asset")
        text(reading.get("meter_id"), "meter id")
        key = (reading["meter_id"], timestamp(reading.get("timestamp")))
        require(key not in seen, "duplicate meter interval")
        seen.add(key)
        number(reading.get("kwh"), "meter kwh")
    seen = set()
    for sample in records(data.get("samples"), "samples"):
        require(sample.get("asset_id") in assets, "unknown telemetry asset")
        key = (sample["asset_id"], timestamp(sample.get("timestamp")))
        require(key not in seen, "duplicate telemetry sample")
        seen.add(key)
        number(sample.get("voltage_v"), "voltage")
        number(sample.get("load_kw"), "load")
        emissions(sample.get("emissions"))
    outages = indexed(records(data.get("outages"), "outages"), "id", "outages")
    for outage in outages.values():
        require(outage.get("asset_id") in assets, "unknown outage asset")
        text(outage.get("description"), "outage description")
        for field in ("life_safety", "critical_service"):
            require(type(outage.get(field)) is bool, field + " must be boolean")
        require(type(outage.get("customers_affected")) is int and
                outage["customers_affected"] >= 0, "invalid customer count")
        require(outage.get("declared_priority") == priority(outage),
                "outage priority violates declared safety rules")
    feedback = indexed(records(data.get("feedback"), "feedback"), "id", "feedback")
    for item in feedback.values():
        text(item.get("text"), "feedback text")
        require(item.get("outage_id") is None or item["outage_id"] in outages,
                "unknown feedback outage")


def validate(envelope, expected):
    """Shared structural, domain and cross-stage integrity boundary."""
    require(isinstance(envelope, dict), "envelope must be an object")
    require(expected in STAGES and envelope.get("stage") == expected, "wrong stage")
    require(envelope.get("schema_version") == VERSION, "unsupported schema version")
    data = envelope.get("data")
    validate_dataset(data)
    assets = {a["asset_id"]: a for a in data["assets"]}
    require(all(re.fullmatch(r"ASSET-[0-9]{4,}", key) for key in assets),
            "outputs require protected asset aliases")
    sources = {s["id"]: s for s in data["sources"]}
    products = {p["id"]: p for p in data["products"]}
    outages = {o["id"]: o for o in data["outages"]}
    index = STAGES.index(expected)
    results = envelope.get("results")
    require(isinstance(results, dict) and set(results) == set(STAGES[1:index + 1]),
            "invalid cumulative results")
    if index >= 1:
        result = results["research"]
        text(result.get("query"), "research query")
        evidence = records(result.get("evidence"), "evidence")
        indexed(evidence, "source_id", "evidence")
        for item in evidence:
            require(item["source_id"] in sources, "unknown evidence citation")
            require(item.get("excerpt") in sources[item["source_id"]]["text"],
                    "evidence must quote its source")
            text(item.get("excerpt"), "evidence excerpt")
            number(item.get("score"), "evidence score")
        require(result.get("evidence_status") == ("supported" if evidence else "insufficient"),
                "invalid evidence status")
    if index >= 2:
        matches = records(results["search"].get("matches"), "matches")
        indexed(matches, "product_id", "matches")
        citations = [item["source_id"] for item in results["research"]["evidence"]]
        for match in matches:
            require(match["product_id"] in products, "unknown matched product")
            number(match.get("score"), "search score")
            require(match.get("evidence_ids") == citations, "search lost research evidence")
            eligible = match.get("outage_ids")
            require(isinstance(eligible, list) and len(eligible) == len(set(eligible)),
                    "invalid matched outage list")
            for oid in eligible:
                require(oid in outages, "unknown matched outage")
                outage = outages[oid]
                product = products[match["product_id"]]
                require(priority(outage) in product["priorities"] and
                        assets[outage["asset_id"]]["kind"] in product["asset_kinds"],
                        "unsafe or incompatible product match")
    if index >= 3:
        docs = results["documents"]
        rows = records(docs.get("outage_rows"), "outage rows")
        require(set(indexed(rows, "outage_id", "document outages")) == set(outages),
                "documents must cover every outage")
        matches = results["search"]["matches"]
        for row in rows:
            outage = outages[row["outage_id"]]
            require(row.get("priority") == priority(outage) and
                    row.get("asset_id") == outage["asset_id"], "document outage mismatch")
            choices = [m["product_id"] for m in matches if outage["id"] in m["outage_ids"]]
            require(row.get("recommended_product_id") == (choices[0] if choices else None),
                    "document recommendation must follow search")
        require(docs.get("meter_totals") == meter_totals(data["meter_readings"]),
                "meter totals mismatch")
        require(docs.get("telemetry") == data["samples"], "telemetry provenance lost")
        require(docs.get("outage_csv") == outage_csv(rows), "document CSV mismatch")
    if index >= 4:
        require(results["insights"] == summarize_feedback(data, results["documents"]),
                "insights must follow validated documents and feedback")
    return envelope


def normalize(raw):
    require(isinstance(raw, dict), "input must be an object")
    require(raw.get("schema_version") == VERSION, "unsupported schema version")
    telemetry = raw.get("telemetry")
    require(isinstance(telemetry, dict), "telemetry must be an object")
    raw_csv = raw.get("meter_csv")
    require(isinstance(raw_csv, str), "meter_csv must be CSV text")
    reader = csv.DictReader(io.StringIO(raw_csv), strict=True)
    require(reader.fieldnames == ["timestamp", "meter_id", "asset_id", "kwh"],
            "meter CSV header must be timestamp,meter_id,asset_id,kwh")
    readings = []
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()),
                "malformed meter CSV row")
        try:
            row["kwh"] = float(row["kwh"])
        except ValueError as exc:
            raise ValidationError("meter kwh must be numeric") from exc
        readings.append(row)
    data = {key: copy.deepcopy(raw.get(key)) for key in
            ("synthetic", "question", "sources", "products", "outages", "feedback")}
    data.update(assets=copy.deepcopy(telemetry.get("assets")),
                samples=copy.deepcopy(telemetry.get("samples")), meter_readings=readings)
    validate_dataset(data)
    aliases = {asset: f"ASSET-{i:04d}" for i, asset in
               enumerate(sorted(a["asset_id"] for a in data["assets"]), 1)}
    meters = sorted({r["meter_id"] for r in readings})
    require(not (set(meters) & set(aliases)), "meter and asset identifiers must be distinct")
    aliases.update({meter: f"METER-{i:04d}" for i, meter in enumerate(meters, 1)})
    # Replace identifiers in free text as well as reference fields, in one pass.
    pattern = re.compile("|".join(re.escape(k) for k in sorted(aliases, key=len, reverse=True)))

    def protect(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: aliases[m.group()], value)
        if isinstance(value, list):
            return [protect(item) for item in value]
        if isinstance(value, dict):
            return {protect(key): protect(item) for key, item in value.items()}
        return value

    envelope = {"schema_version": VERSION, "stage": "normalized",
                "data": protect(data), "results": {}}
    return validate(envelope, "normalized")


def advance(previous, old, new, result):
    validate(previous, old)
    envelope = copy.deepcopy(previous)
    envelope["stage"] = new
    envelope["results"][new] = result
    return validate(envelope, new)


def research(previous):
    validate(previous, "normalized")
    data = previous["data"]
    query_words = terms(data["question"])
    evidence = []
    for source in data["sources"]:
        score = len(query_words & terms(source["text"]))
        if score:
            evidence.append({"source_id": source["id"], "score": score,
                             "excerpt": source["text"][:360]})
    evidence.sort(key=lambda item: (-item["score"], item["source_id"]))
    evidence = evidence[:5]
    query = data["question"] + " " + " ".join(item["excerpt"] for item in evidence)
    return advance(previous, "normalized", "research",
                   {"query": query.strip(), "evidence": evidence,
                    "evidence_status": "supported" if evidence else "insufficient",
                    "decision_note": "Review cited excerpts; lexical relevance is not factual verification."})


def search(previous):
    validate(previous, "research")
    data = previous["data"]
    evidence = previous["results"]["research"]
    query_words = terms(evidence["query"])
    assets = {a["asset_id"]: a for a in data["assets"]}
    matches = []
    for product in data["products"]:
        matched = query_words & terms(" ".join(
            [product["name"], product["description"]] + product["tags"]))
        eligible = [o["id"] for o in data["outages"]
                    if priority(o) in product["priorities"] and
                    assets[o["asset_id"]]["kind"] in product["asset_kinds"]]
        if matched and (eligible or not data["outages"]):
            matches.append({"product_id": product["id"], "score": len(matched),
                            "matched_terms": sorted(matched), "outage_ids": sorted(eligible),
                            "evidence_ids": [e["source_id"] for e in evidence["evidence"]]})
    matches.sort(key=lambda item: (-item["score"], item["product_id"]))
    return advance(previous, "research", "search",
                   {"matches": matches, "status": "matched" if matches else "no_match"})


def meter_totals(readings):
    grouped = {}
    for row in readings:
        key = (row["meter_id"], row["asset_id"])
        grouped.setdefault(key, []).append(row["kwh"])
    totals = []
    for (meter, asset), values in sorted(grouped.items()):
        total = math.fsum(values)
        number(total, "aggregated consumption")
        totals.append({"meter_id": meter, "asset_id": asset,
                       "value": round(total, 6), "unit": "kWh", "intervals": len(values)})
    return totals


def safe_cell(value):
    value = "" if value is None else str(value)
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def outage_csv(rows):
    output = io.StringIO(newline="")
    fields = ["outage_id", "asset_id", "priority", "recommended_product_id"]
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([safe_cell(row[field]) for field in fields])
    return output.getvalue()


def documents(previous):
    validate(previous, "search")
    data = previous["data"]
    rows = []
    for outage in sorted(data["outages"], key=lambda o: (priority(o), o["id"])):
        choices = [m["product_id"] for m in previous["results"]["search"]["matches"]
                   if outage["id"] in m["outage_ids"]]
        rows.append({"outage_id": outage["id"], "asset_id": outage["asset_id"],
                     "priority": priority(outage),
                     "recommended_product_id": choices[0] if choices else None})
    return advance(previous, "search", "documents",
                   {"outage_rows": rows, "outage_csv": outage_csv(rows),
                    "meter_totals": meter_totals(data["meter_readings"]),
                    "telemetry": copy.deepcopy(data["samples"])})


THEMES = {
    "billing": ({"bill", "billing", "charge", "cost"}, "Review billing and meter intervals"),
    "communication": ({"update", "updates", "notice", "communication"}, "Improve outage updates"),
    "reliability": ({"outage", "blackout", "interruption", "reliable"}, "Review restoration and reliability"),
    "safety": ({"unsafe", "danger", "safety", "sparks"}, "Escalate to safety response team"),
}
NEGATIVE = {"bad", "angry", "poor", "unsafe", "slow", "expensive", "wrong", "failed"}
POSITIVE = {"good", "great", "helpful", "fast", "thanks"}


def summarize_feedback(data, docs):
    rows = {row["outage_id"]: row for row in docs["outage_rows"]}
    buckets = {}
    for feedback in data["feedback"]:
        words = terms(feedback["text"])
        names = [name for name, (keywords, _) in THEMES.items() if words & keywords] or ["other"]
        sentiment = "negative" if words & NEGATIVE else "positive" if words & POSITIVE else "neutral"
        row = rows.get(feedback.get("outage_id"))
        for name in names:
            bucket = buckets.setdefault(name, {"theme": name, "feedback_ids": [],
                "sentiment_counts": {"negative": 0, "positive": 0, "neutral": 0},
                "priority": "P3", "outage_ids": [], "product_ids": [],
                "action": THEMES[name][1] if name in THEMES else "Review uncategorized feedback"})
            bucket["feedback_ids"].append(feedback["id"])
            bucket["sentiment_counts"][sentiment] += 1
            if row:
                bucket["priority"] = min(bucket["priority"], row["priority"])
                bucket["outage_ids"].append(row["outage_id"])
                if row["recommended_product_id"]:
                    bucket["product_ids"].append(row["recommended_product_id"])
    themes = list(buckets.values())
    for bucket in themes:
        for key in ("feedback_ids", "outage_ids", "product_ids"):
            bucket[key] = sorted(set(bucket[key]))
        bucket["count"] = len(bucket["feedback_ids"])
    themes.sort(key=lambda b: (b["priority"], -b["sentiment_counts"]["negative"], b["theme"]))
    return {"feedback_count": len(data["feedback"]), "themes": themes,
            "limitation": "Deterministic keyword themes; sentiment does not understand negation. "
                          "Feedback may occur in multiple themes; feedback never changes outage priority."}


def insights(previous):
    validate(previous, "documents")
    return advance(previous, "documents", "insights",
                   summarize_feedback(previous["data"], previous["results"]["documents"]))


def run(raw):
    state = normalize(raw)
    for stage in (research, search, documents, insights):
        state = stage(state)
    return {"status": "ok", **state}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            raw = json.load(handle)
        result = run(raw)
        output = json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError,
            RecursionError, csv.Error):
        # Do not echo file paths, rejected payloads or critical identifiers.
        print(json.dumps({"status": "error", "message": "Invalid input, stage data, or unreadable file"}))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
