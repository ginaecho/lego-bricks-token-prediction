"""Deterministic synthetic energy pipeline. Python standard library only.

Run: python -B implementation.py example_input.json
Protection rules are demonstrations, not NERC CIP compliance certification.
Schema fields use literal labels, not caller-supplied regular expressions.
"""

import copy
import csv
import datetime
import hashlib
import io
import json
import math
import re
import sys


VERSION = "energy-reference/1"
STAGES = ("input", "deep", "interests", "feedback", "extract")
SAFETY = {"fire": "critical", "live_wire": "critical",
          "medical_dependency": "critical", "sustained_loss": "high",
          "default": "routine"}
LEVELS = {"routine": 0, "high": 1, "critical": 2}
UNITS = {"kgCO2e", "tCO2e"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def unique(items, field, context):
    require(isinstance(items, list), context + " must be a list")
    ids = []
    for item in items:
        require(isinstance(item, dict), context + " contains a non-object")
        value = item.get(field)
        require(isinstance(value, str) and bool(value.strip()),
                context + " needs nonempty identifiers")
        ids.append(value)
    require(len(ids) == len(set(ids)), context + " identifiers must be unique")
    return set(ids)


def priority(flags, rules):
    require(isinstance(flags, list) and all(
        isinstance(f, str) and f in rules and f != "default" for f in flags),
        "Unknown safety flag")
    return max([rules["default"]] + [rules[f] for f in flags],
               key=LEVELS.__getitem__)


def span(text, quote):
    start = text.find(quote)
    require(bool(quote) and start >= 0, "Evidence quote not found in source")
    return {"start": start, "end": start + len(quote), "quote": quote}


def validate(envelope, expected=None):
    """One validation boundary used for input and every stage handoff."""
    require(isinstance(envelope, dict), "Envelope must be an object")
    require(envelope.get("schema_version") == VERSION, "Unsupported schema version")
    stage = envelope.get("stage")
    require(stage in STAGES and (expected is None or stage == expected),
            "Unexpected pipeline stage")
    require(envelope.get("status") == "ok", "Invalid envelope status")
    data = envelope.get("data")
    require(isinstance(data, dict), "Missing data object")
    source = data.get("source")
    require(isinstance(source, dict) and source.get("synthetic") is True,
            "Input must be explicitly synthetic")
    assets = unique(source.get("assets"), "id", "assets")
    require(assets, "At least one grid asset is required")
    for asset in source["assets"]:
        require(type(asset.get("critical")) is bool, "Asset critical flag required")
    links = source.get("topology")
    require(isinstance(links, list), "Topology must be a list")
    for link in links:
        require(isinstance(link, list) and len(link) == 2
                and all(a in assets for a in link) and link[0] != link[1],
                "Topology contains an invalid edge")
    docs = unique(source.get("documents"), "id", "documents")
    for doc in source["documents"]:
        require(doc.get("asset_id") in assets and isinstance(doc.get("text"), str),
                "Document has invalid asset or text")
        require(isinstance(doc.get("claims"), list), "Document claims required")
        for claim in doc["claims"]:
            require(isinstance(claim, dict)
                    and isinstance(claim.get("topic"), str)
                    and bool(claim["topic"].strip()), "Invalid claim")
            require(isinstance(claim.get("value"), (str, int, float))
                    and not isinstance(claim.get("value"), bool), "Invalid claim value")
            if isinstance(claim["value"], (int, float)):
                require(number(claim["value"]), "Nonfinite claim value")
            require(isinstance(claim.get("quote"), str), "Claim quote required")
            span(doc["text"], claim["quote"])
            if claim["topic"] == "emissions":
                require(number(claim["value"]) and claim["value"] >= 0
                        and claim.get("unit") in UNITS
                        and claim.get("source") == doc["id"],
                        "Emissions require nonnegative value, units and source")
    prefs = source.get("preferences")
    require(isinstance(prefs, dict) and isinstance(prefs.get("weights"), dict),
            "Preference weights required")
    require(all(isinstance(k, str) and k and number(v) and v >= 0
                for k, v in prefs["weights"].items()), "Invalid preference weights")
    require(isinstance(prefs.get("exclude_assets"), list)
            and all(a in assets for a in prefs["exclude_assets"]),
            "Invalid exclusions")
    require(source.get("safety_rules") == SAFETY, "Declared safety rules must match policy")
    outages = unique(source.get("outages"), "id", "outages")
    for outage in source["outages"]:
        require(outage.get("asset_id") in assets, "Unknown outage asset")
        require(outage.get("priority") == priority(outage.get("safety_flags"), SAFETY),
                "Outage priority violates declared safety rules")
    unique(source.get("feedback"), "id", "feedback")
    for feedback in source["feedback"]:
        require(feedback.get("asset_id") in assets
                and isinstance(feedback.get("text"), str)
                and feedback["text"].strip(), "Invalid feedback")
        require(feedback.get("outage_id") in outages, "Unknown feedback outage")
        outage = next(o for o in source["outages"] if o["id"] == feedback["outage_id"])
        require(outage["asset_id"] == feedback["asset_id"], "Feedback outage asset mismatch")
    fields = source.get("extraction_schema")
    unique(fields, "name", "extraction fields")
    labels = []
    for field in fields:
        require(isinstance(field.get("label"), str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,39}", field["label"])
                and field.get("type") in ("string", "number")
                and type(field.get("required")) is bool, "Invalid extraction field")
        require(field.get("semantic", "generic") in ("generic", "emissions", "priority"),
                "Invalid extraction semantic")
        if field.get("semantic") == "emissions":
            require(field["type"] == "number" and field.get("unit") in UNITS,
                    "Emissions extraction needs a numeric value and units")
        labels.append(field["label"].casefold())
    require(len(labels) == len(set(labels)), "Duplicate extraction labels")
    require(isinstance(source.get("meter_csv"), str), "Meter CSV required")
    require(isinstance(source.get("telemetry"), dict), "SCADA telemetry object required")
    parse_measurements(source)
    if stage == "input":
        return envelope
    research = data.get("research")
    require(isinstance(research, dict), "Research handoff missing")
    evidence_ids = unique(research.get("evidence"), "id", "evidence")
    doc_index = {d["id"]: d for d in source["documents"]}
    for item in research["evidence"]:
        require(item.get("document_id") in docs, "Unknown evidence document")
        document = doc_index[item["document_id"]]
        require(item.get("asset_id") == document["asset_id"], "Evidence asset mismatch")
        verify_span(document["text"], item.get("span"))
        require(any(item.get("topic") == c["topic"] and item.get("value") == c["value"]
                    and item["span"]["quote"] == c["quote"]
                    and (c["topic"] != "emissions" or (
                        item.get("unit") == c["unit"] and item.get("source") == c["source"]))
                    for c in document["claims"]), "Evidence changed source claim")
    require(isinstance(research.get("disagreements"), list)
            and isinstance(research.get("unresolved_questions"), list),
            "Research synthesis incomplete")
    if STAGES.index(stage) >= 2:
        ranked = data.get("recommendations")
        unique(ranked, "asset_id", "recommendations")
        for recommendation in ranked:
            require(recommendation["asset_id"] in assets
                    and recommendation["asset_id"] not in prefs["exclude_assets"],
                    "Recommendation violates exclusion")
            require(number(recommendation.get("score"))
                    and recommendation.get("evidence_ids")
                    and all(e in evidence_ids for e in recommendation["evidence_ids"]),
                    "Recommendation must be grounded")
            require(all(next(e for e in research["evidence"] if e["id"] == eid)["asset_id"]
                        == recommendation["asset_id"] for eid in recommendation["evidence_ids"]),
                    "Recommendation evidence asset mismatch")
    if STAGES.index(stage) >= 3:
        insight = data.get("feedback_analysis")
        require(isinstance(insight, dict), "Feedback handoff missing")
        unique(insight.get("records"), "id", "feedback records")
        allowed = {r["asset_id"] for r in data["recommendations"]}
        feedback_index = {f["id"]: f for f in source["feedback"]}
        for record in insight["records"]:
            require(record.get("asset_id") in allowed, "Feedback not recommended")
            require(isinstance(record.get("supports"), list) and record["supports"],
                    "Feedback support missing")
            for support in record["supports"]:
                require(support.get("feedback_id") in feedback_index, "Unknown feedback support")
                original = feedback_index[support["feedback_id"]]
                require(original["asset_id"] == record["asset_id"]
                        and original["outage_id"] == record["outage_id"],
                        "Feedback support context mismatch")
                verify_span(original["text"], support.get("span"))
                require(" ".join(original["text"].casefold().split())
                        == " ".join(record["text"].casefold().split()),
                        "Deduplicated feedback text mismatch")
            require(record["text"] == feedback_index[
                record["supports"][0]["feedback_id"]]["text"],
                "Canonical feedback must preserve first source")
        require(isinstance(insight.get("themes"), list), "Feedback themes missing")
        record_index = {r["id"]: r for r in insight["records"]}
        for theme in insight["themes"]:
            require(isinstance(theme.get("record_ids"), list)
                    and all(r in record_index for r in theme["record_ids"])
                    and theme.get("count") == len(theme["record_ids"])
                    and theme.get("supporting_excerpts") == [
                        record_index[r]["supports"][0] for r in theme["record_ids"]],
                    "Theme support mismatch")
    if stage == "extract":
        extraction = data.get("extraction")
        require(isinstance(extraction, list), "Extraction handoff missing")
        records = {r["id"]: r for r in data["feedback_analysis"]["records"]}
        require(len(extraction) == len(records)
                and {r.get("record_id") for r in extraction} == set(records),
                "Extraction must cover feedback records")
        for result in extraction:
            require(set(result.get("fields", {})) == {f["name"] for f in fields},
                    "Extracted fields do not match schema")
            record = records[result["record_id"]]
            missing = []
            for field in fields:
                value = result["fields"][field["name"]]
                if value is None:
                    missing.append(field["name"])
                    continue
                require(value.get("source") == record["supports"][0]["feedback_id"],
                        "Extraction provenance mismatch")
                verify_span(record["text"], value.get("span"))
                require((field["type"] == "string" and isinstance(value.get("value"), str))
                        or (field["type"] == "number" and number(value.get("value"))),
                        "Extraction type mismatch")
                if field.get("semantic") == "emissions":
                    require(value.get("unit") == field["unit"] and value["value"] >= 0,
                            "Extracted emissions require units and source")
                if field.get("semantic") == "priority":
                    expected_priority = next(o["priority"] for o in source["outages"]
                                             if o["id"] == record["outage_id"])
                    require(value["value"] == expected_priority,
                            "Extracted priority violates safety rules")
            require(result.get("missing_fields") == missing
                    and result.get("missing_required") == [
                        f["name"] for f in fields if f["required"] and f["name"] in missing],
                    "Missing field report mismatch")
    return envelope


def verify_span(text, value):
    require(isinstance(value, dict), "Source span required")
    start, end = value.get("start"), value.get("end")
    require(type(start) is int and type(end) is int
            and 0 <= start < end <= len(text)
            and text[start:end] == value.get("quote"), "Invalid source span")


def timestamp(value):
    require(isinstance(value, str), "Timestamp required")
    try:
        result = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("Invalid timestamp") from None
    require(result.tzinfo is not None, "Timestamp timezone required")
    return result


def parse_measurements(source):
    assets = {a["id"] for a in source["assets"]}
    reader = csv.DictReader(io.StringIO(source["meter_csv"]))
    require(reader.fieldnames == ["meter_id", "asset_id", "timestamp", "consumption_kwh"],
            "Unexpected smart-meter CSV columns")
    readings, seen = [], set()
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()),
                "Malformed meter CSV row")
        require(row["meter_id"].strip() and row["asset_id"] in assets, "Invalid meter asset")
        instant = timestamp(row["timestamp"])
        try:
            value = float(row["consumption_kwh"])
        except ValueError:
            raise ValidationError("Invalid meter consumption") from None
        require(number(value) and value >= 0, "Consumption must be finite and nonnegative")
        key = (row["meter_id"], instant)
        require(key not in seen, "Duplicate meter interval")
        seen.add(key)
        readings.append(dict(row, consumption_kwh=value, unit="kWh"))
    telemetry = source["telemetry"]
    require(telemetry.get("format") == "synthetic-scada-v1"
            and isinstance(telemetry.get("devices"), list), "Invalid telemetry format")
    unique(telemetry["devices"], "device_id", "SCADA devices")
    for device in telemetry["devices"]:
        require(device.get("asset_id") in assets
                and number(device.get("voltage_kv")) and device["voltage_kv"] >= 0
                and device.get("state") in ("energized", "isolated"),
                "Invalid SCADA telemetry")
        timestamp(device.get("timestamp"))
    return readings


def transition(envelope, before, after, key, value):
    validate(envelope, before)
    result = copy.deepcopy(envelope)
    result["stage"] = after
    result["data"][key] = value
    return validate(result, after)


def deep(envelope):
    validate(envelope, "input")
    source = envelope["data"]["source"]
    evidence, grouped = [], {}
    for document in source["documents"]:
        for claim in document["claims"]:
            item = {"id": "evidence-%04d" % (len(evidence) + 1),
                    "document_id": document["id"], "asset_id": document["asset_id"],
                    "topic": claim["topic"], "value": claim["value"],
                    "span": span(document["text"], claim["quote"])}
            if claim["topic"] == "emissions":
                item.update(unit=claim["unit"], source=claim["source"])
            evidence.append(item)
            grouped.setdefault((item["asset_id"], item["topic"]), []).append(item)
    disagreements, questions = [], []
    for (asset, topic), items in sorted(grouped.items()):
        def canonical(item):
            value = item["value"]
            if topic == "emissions":
                value *= 1000 if item["unit"] == "tCO2e" else 1
            return json.dumps(value, sort_keys=True) if not number(value) else str(float(value))
        if len({canonical(i) for i in items}) > 1:
            disagreements.append({"asset_id": asset, "topic": topic,
                                  "evidence_ids": [i["id"] for i in items]})
            questions.append("Reconcile conflicting %s evidence for %s." % (topic, asset))
    for asset in source["assets"]:
        if not any(e["asset_id"] == asset["id"] for e in evidence):
            questions.append("Obtain documentary evidence for %s." % asset["id"])
    for topic in sorted(source["preferences"]["weights"]):
        if not any(e["topic"] == topic for e in evidence):
            questions.append("Obtain evidence about %s." % topic)
    return transition(envelope, "input", "deep", "research",
                      {"evidence": evidence, "disagreements": disagreements,
                       "unresolved_questions": questions,
                       "meter_readings": parse_measurements(source),
                       "summary": {"documents": len(source["documents"]),
                                   "claims": len(evidence),
                                   "disagreements": len(disagreements)}})


def interests(envelope):
    validate(envelope, "deep")
    source, research = envelope["data"]["source"], envelope["data"]["research"]
    prefs = source["preferences"]
    ranked = []
    for asset in source["assets"]:
        if asset["id"] in prefs["exclude_assets"]:
            continue
        evidence = [e for e in research["evidence"] if e["asset_id"] == asset["id"]]
        if not evidence:
            continue
        topics = sorted({e["topic"] for e in evidence})
        score = sum(prefs["weights"].get(t, 0) for t in topics)
        if score <= 0:
            continue
        conflicts = [d for d in research["disagreements"] if d["asset_id"] == asset["id"]]
        ranked.append({"asset_id": asset["id"], "score": score,
                       "evidence_ids": [e["id"] for e in evidence],
                       "explanation": "Matched weighted topics: " + ", ".join(
                           t for t in topics if prefs["weights"].get(t, 0) > 0),
                       "uncertainty": conflicts})
    ranked.sort(key=lambda r: (-r["score"], r["asset_id"]))
    return transition(envelope, "deep", "interests", "recommendations", ranked)


def feedback(envelope):
    validate(envelope, "interests")
    allowed = {r["asset_id"] for r in envelope["data"]["recommendations"]}
    records, grouped = [], {}
    for item in envelope["data"]["source"]["feedback"]:
        if item["asset_id"] not in allowed:
            continue
        normalized = " ".join(item["text"].casefold().split())
        key = (item["asset_id"], item["outage_id"], normalized)
        if key not in grouped:
            record = {"id": "feedback-%04d" % (len(records) + 1),
                      "asset_id": item["asset_id"], "outage_id": item["outage_id"],
                      "text": item["text"], "supports": []}
            grouped[key] = record
            records.append(record)
        grouped[key]["supports"].append({"feedback_id": item["id"],
                                         "span": span(item["text"], item["text"])})
    themes = []
    keywords = {"safety": ("fire", "wire", "medical"),
                "reliability": ("outage", "interruption", "restored"),
                "consumption": ("consumption", "winter", "summer"),
                "emissions": ("emissions",)}
    for name, words in keywords.items():
        matches = [r for r in records if any(re.search(r"\b" + word + r"\b",
                                                     r["text"], re.I) for word in words)]
        if matches:
            themes.append({"name": name, "record_ids": [r["id"] for r in matches],
                           "count": len(matches),
                           "supporting_excerpts": [r["supports"][0] for r in matches]})
    return transition(envelope, "interests", "feedback", "feedback_analysis",
                      {"records": records, "themes": themes,
                       "duplicates_removed": sum(len(r["supports"]) - 1 for r in records),
                       "excluded_feedback_count": sum(
                           f["asset_id"] not in allowed
                           for f in envelope["data"]["source"]["feedback"])})


def extract(envelope):
    validate(envelope, "feedback")
    fields = envelope["data"]["source"]["extraction_schema"]
    results = []
    for record in envelope["data"]["feedback_analysis"]["records"]:
        values, missing = {}, []
        for field in fields:
            pattern = r"(?:^|[;\n])\s*" + re.escape(field["label"]) + r"\s*:\s*([^;\n]+)"
            matches = list(re.finditer(pattern, record["text"], re.I))
            require(len(matches) <= 1, "Ambiguous repeated extraction label")
            if not matches:
                values[field["name"]] = None
                missing.append(field["name"])
                continue
            match = matches[0]
            raw = match.group(1).strip()
            start = match.start(1) + len(match.group(1)) - len(match.group(1).lstrip())
            value = raw
            if field["type"] == "number":
                unit = field.get("unit")
                numeric = raw
                if unit:
                    require(raw.endswith(" " + unit), "Extracted number has wrong or missing unit")
                    numeric = raw[:-len(unit)].strip()
                require(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", numeric),
                        "Invalid numeric extraction")
                value = float(numeric)
                require(number(value), "Nonfinite extracted number")
            extracted = {"value": value, "source": record["supports"][0]["feedback_id"],
                         "span": {"start": start, "end": start + len(raw), "quote": raw}}
            if field.get("unit"):
                extracted["unit"] = field["unit"]
            values[field["name"]] = extracted
        results.append({"record_id": record["id"], "fields": values,
                        "missing_fields": missing,
                        "missing_required": [f["name"] for f in fields
                                             if f["required"] and f["name"] in missing]})
    return transition(envelope, "feedback", "extract", "extraction", results)


def protect(source):
    """Pseudonymize all asset, meter and SCADA identifiers before synthesis.

    All text and quotes are transformed together; reported character offsets
    address the protected source retained in the envelope, not raw private input.
    The reverse mapping is deliberately never returned.
    """
    identifiers = {a["id"] for a in source["assets"]}
    identifiers.update(d["device_id"] for d in source["telemetry"]["devices"])
    identifiers.update(r["meter_id"] for r in parse_measurements(source))
    replacements = {value: "protected-" + hashlib.sha256(value.encode()).hexdigest()[:16]
                    for value in identifiers}
    pattern = re.compile("|".join(re.escape(v) for v in sorted(identifiers, key=lambda v: (-len(v), v))))

    def walk(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: replacements[m.group()], value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            result = {walk(k): walk(v) for k, v in value.items()}
            require(len(result) == len(value), "Protected keys collided")
            return result
        return value

    return walk(source)


def pipeline(payload):
    require(isinstance(payload, dict), "Input must be an object")
    raw = {"schema_version": VERSION, "status": "ok", "stage": "input",
           "data": {"source": copy.deepcopy(payload)}}
    validate(raw, "input")
    raw["data"]["source"] = protect(payload)
    validate(raw, "input")
    return extract(feedback(interests(deep(raw))))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Expected exactly one input JSON path")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle)
        result = pipeline(payload)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError,
            OverflowError, csv.Error):
        # Never echo attacker-controlled text, file paths, or critical identifiers.
        print(json.dumps({"status": "error",
                          "error": "Input/file validation failed; check the documented schema."}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
