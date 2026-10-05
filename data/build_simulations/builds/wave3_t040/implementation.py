"""Synthetic telecom comparison -> personalized discovery, Python standard library only."""

import csv
import io
import json
import re
import sys
from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


class ValidationError(ValueError):
    pass


TAGS = {"streaming", "travel", "work", "gaming", "budget"}
CSV_FIELDS = [
    "record_id", "subscriber_id", "residency_tag", "phone_number",
    "imei", "call_minutes", "data_mb",
]


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value, label, minimum=0, maximum=10**9):
    require(not isinstance(value, bool) and isinstance(value, (int, float, str)),
            f"{label}: expected a number")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError(f"{label}: invalid number") from None
    require(result.is_finite() and minimum <= result <= maximum,
            f"{label}: out of range")
    return result


def integer(value, label, minimum=0, maximum=10**9):
    result = number(value, label, minimum, maximum)
    require(result == result.to_integral_value(), f"{label}: expected integer")
    return int(result)


def stage_integer(value, label, minimum=0, maximum=10**9):
    require(type(value) is int, f"{label}: expected normalized integer")
    return integer(value, label, minimum, maximum)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 10000,
            f"{label}: expected nonblank text")
    return value


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_-]{0,39}", value),
            "Invalid synthetic record identifier")
    return value


def object_value(value, label):
    require(isinstance(value, dict), f"{label}: expected object")
    return value


def tags(value):
    require(isinstance(value, list) and all(isinstance(x, str) and x in TAGS for x in value),
            "Unknown interest tag")
    require(len(value) == len(set(value)), "Duplicate interest tag")
    return value


def record(value, residency, key, seen):
    object_value(value, "record")
    require(value.get("residency_tag") == residency, "Record residency mismatch")
    identifier(value.get(key))
    require(value[key] not in seen, "Duplicate record identifier")
    seen.add(value[key])
    return value


def validate(payload, kind="input"):
    """One validation boundary for external input and both stage contracts."""
    object_value(payload, kind)
    if kind == "input":
        require(type(payload.get("schema_version")) is int and payload["schema_version"] == 1,
                "Unsupported schema version")
        require(payload.get("synthetic_fixture") is True, "Synthetic fixture marker required")
        residency = payload.get("residency_tag")
        require(residency in ("EU", "US"), "Unsupported residency tag")
        sub = record(payload.get("subscriber"), residency, "subscriber_id", set())
        require(sub.get("consent_personalization") is True,
                "Subscriber personalization consent required")
        require(sub.get("purpose") == "plan_recommendation", "Unsupported processing purpose")
        preferences = object_value(sub.get("preferences"), "preferences")
        budget = integer(preferences.get("budget_cents"), "budget", 1)
        minimum_data = integer(preferences.get("minimum_data_mb"), "minimum data", 1)
        tags(preferences.get("interests"))
        excluded = preferences.get("excluded_offer_ids")
        require(isinstance(excluded, list), "Exclusions must be a list")
        for value in excluded:
            identifier(value)
        require(len(excluded) == len(set(excluded)), "Duplicate exclusion")
        top_n = integer(preferences.get("top_n"), "top_n", 1, 50)
        offers = payload.get("offers")
        require(isinstance(offers, list) and 1 <= len(offers) <= 100, "Expected 1-100 offers")
        normalized = []
        seen = set()
        for offer in offers:
            record(offer, residency, "offer_id", seen)
            require(offer.get("currency") == "USD", "Only USD prices supported")
            price = number(offer.get("monthly_price"), "monthly price")
            require(price * 100 == (price * 100).to_integral_value(),
                    "Price must have whole cents")
            unit = offer.get("data_unit")
            require(unit in ("MB", "GB"), "Data unit must be MB or GB")
            data = number(offer.get("data_allowance"), "data allowance")
            data *= 1000 if unit == "GB" else 1
            data_mb = integer(str(data), "normalized data")
            quality = integer(offer.get("network_quality"), "network quality", 0, 100)
            normalized.append({
                "offer_id": offer["offer_id"], "residency_tag": residency,
                "monthly_price_cents": int(price * 100), "data_mb": data_mb,
                "network_quality": quality, "tags": tags(offer.get("tags")),
            })
        cdr = text(payload.get("call_detail_records_csv"), "call detail CSV")
        try:
            reader = csv.DictReader(io.StringIO(cdr), strict=True)
            require(reader.fieldnames == CSV_FIELDS, "Unexpected call detail CSV headers")
            usage = list(reader)
        except csv.Error:
            raise ValidationError("Malformed call detail CSV") from None
        seen = set()
        for row in usage:
            require(None not in row and all(v is not None for v in row.values()),
                    "Malformed call detail CSV row")
            record(row, residency, "record_id", seen)
            require(row["subscriber_id"] == sub["subscriber_id"], "Usage subscriber mismatch")
            require(re.fullmatch(r"\+1-202-555-01\d{2}", row["phone_number"]) is not None,
                    "Only fictional reserved-range phone numbers accepted")
            require(re.fullmatch(r"SYNTH-IMEI-\d{6}", row["imei"]) is not None,
                    "Only synthetic IMEI markers accepted")
            row["call_minutes"] = integer(row["call_minutes"], "call minutes")
            row["data_mb"] = integer(row["data_mb"], "usage data")
        collections = {}
        for name, key in (("network_fault_tickets", "ticket_id"),
                          ("support_chat_transcripts", "chat_id"),
                          ("billing_adjustments", "adjustment_id")):
            rows = payload.get(name)
            require(isinstance(rows, list), f"{name}: expected list")
            seen = set()
            for row in rows:
                record(row, residency, key, seen)
                require(row.get("subscriber_id") == sub["subscriber_id"],
                        "Related record subscriber mismatch")
                if name == "network_fault_tickets":
                    require(row.get("status") in ("open", "closed"), "Invalid fault status")
                    text(row.get("summary"), "fault summary")
                elif name == "support_chat_transcripts":
                    text(row.get("transcript"), "support transcript")
                else:
                    integer(row.get("amount_cents"), "adjustment amount", -10**9)
                    text(row.get("reason"), "Billing adjustment reason")
            collections[name] = rows
        ticket_ids = {x["ticket_id"] for x in collections["network_fault_tickets"]}
        for row in collections["support_chat_transcripts"]:
            identifier(row.get("ticket_id"))
            require(row.get("ticket_id") in ticket_ids, "Chat must reference a known fault ticket")
        return {
            "residency_tag": residency, "offers": normalized,
            "preferences": {
                "residency_tag": residency, "budget_cents": budget,
                "minimum_data_mb": minimum_data, "interests": preferences["interests"],
                "excluded_offer_ids": excluded, "top_n": top_n,
            },
            "context": {
                "residency_tag": residency,
                "observed_data_mb": sum(x["data_mb"] for x in usage),
                "observed_call_minutes": sum(x["call_minutes"] for x in usage),
                "open_fault_count": sum(x["status"] == "open" for x in collections["network_fault_tickets"]),
                "support_chat_count": len(collections["support_chat_transcripts"]),
                "billing_adjustment_count": len(collections["billing_adjustments"]),
            },
        }
    require(kind in ("comparison", "recommendations"), "Unknown validation contract")
    require(type(payload.get("schema_version")) is int and payload["schema_version"] == 1
            and payload.get("synthetic_fixture") is True,
            "Invalid stage schema")
    residency = payload.get("residency_tag")
    require(residency in ("EU", "US"), "Invalid stage residency")
    context = object_value(payload.get("context"), "context")
    prefs = object_value(payload.get("preferences"), "preferences")
    require(set(context) == {"residency_tag", "observed_data_mb", "observed_call_minutes",
                             "open_fault_count", "support_chat_count", "billing_adjustment_count"},
            "Unexpected stage context fields")
    require(set(prefs) == {"residency_tag", "budget_cents", "minimum_data_mb", "interests",
                           "excluded_offer_ids", "top_n"}, "Unexpected preference fields")
    require(context.get("residency_tag") == prefs.get("residency_tag") == residency,
            "Stage context residency mismatch")
    for key in ("observed_data_mb", "observed_call_minutes", "open_fault_count",
                "support_chat_count", "billing_adjustment_count"):
        stage_integer(context.get(key), key)
    stage_integer(prefs.get("budget_cents"), "budget", 1)
    stage_integer(prefs.get("minimum_data_mb"), "minimum data", 1)
    stage_integer(prefs.get("top_n"), "top_n", 1, 50)
    tags(prefs.get("interests"))
    exclusions = prefs.get("excluded_offer_ids")
    require(isinstance(exclusions, list), "Invalid stage exclusions")
    for value in exclusions:
        identifier(value)
    require(len(exclusions) == len(set(exclusions)), "Duplicate stage exclusion")
    rows = payload.get("rows")
    require(isinstance(rows, list), "Stage rows must be a list")
    if kind == "comparison":
        require(bool(rows), "Comparison rows must not be empty")
        require(payload.get("columns") == [
            "monthly_price_cents", "data_mb", "network_quality", "preference_score"
        ], "Invalid comparison columns")
    else:
        require(len(rows) <= prefs["top_n"], "Too many recommendations")
    seen = set()
    previous = None
    for rank, row in enumerate(rows, 1):
        record(row, residency, "offer_id", seen)
        fields = {"offer_id", "residency_tag", "monthly_price_cents", "data_mb",
                  "network_quality", "tags", "preference_score", "rank"}
        if kind == "recommendations":
            fields |= {"recommendation_score", "explanations"}
        require(set(row) == fields, "Unexpected stage row fields")
        for key in ("monthly_price_cents", "data_mb"):
            stage_integer(row.get(key), key)
        stage_integer(row.get("network_quality"), "quality", 0, 100)
        stage_integer(row.get("preference_score"), "score", 0, 10000)
        tags(row.get("tags"))
        require(stage_integer(row.get("rank"), "rank", 1) == rank, "Invalid rank sequence")
        if kind == "recommendations":
            require(row["offer_id"] not in exclusions, "Excluded offer leaked")
            stage_integer(row.get("recommendation_score"), "recommendation score", 0, 12000)
            require(row.get("explanations") == explain(row, prefs), "Ungrounded explanation")
            require(row["recommendation_score"] == interest_score(row, prefs),
                    "Invalid recommendation score")
        require(row["preference_score"] == preference_score(row, prefs, context),
                "Invalid preference score")
        key = (-(row["preference_score"] if kind == "comparison" else row["recommendation_score"]),
               row["offer_id"])
        require(previous is None or previous <= key, "Stage ranking is not sorted")
        previous = key
    return deepcopy(payload)


def preference_score(offer, prefs, context):
    demand = max(prefs["minimum_data_mb"], context["observed_data_mb"])
    data_fit = min(Decimal(offer["data_mb"]) / demand, Decimal(1))
    cost_fit = min(Decimal(prefs["budget_cents"]) /
                   max(offer["monthly_price_cents"], 1), Decimal(1))
    quality = Decimal(offer["network_quality"]) / 100
    quality_weight = 40 if context["open_fault_count"] else 20
    score = 40 * data_fit + (60 - quality_weight) * cost_fit + quality_weight * quality
    return int((score * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def interest_score(row, prefs):
    return row["preference_score"] + 400 * len(set(row["tags"]) & set(prefs["interests"]))


def explain(row, prefs):
    matches = sorted(set(row["tags"]) & set(prefs["interests"]))
    return [
        "Matched interests: " + (", ".join(matches) if matches else "none"),
        f"Monthly price: {row['monthly_price_cents']} USD cents; budget: {prefs['budget_cents']} USD cents.",
        f"Data allowance: {row['data_mb']} MB; network quality: {row['network_quality']}/100.",
        f"Comparison preference score: {row['preference_score']}/10000.",
    ]


def compare(payload):
    data = validate(payload)
    rows = deepcopy(data["offers"])
    for row in rows:
        row["preference_score"] = preference_score(row, data["preferences"], data["context"])
    rows.sort(key=lambda row: (-row["preference_score"], row["offer_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return validate({
        "schema_version": 1, "synthetic_fixture": True,
        "residency_tag": data["residency_tag"],
        "context": data["context"], "preferences": data["preferences"],
        "columns": ["monthly_price_cents", "data_mb", "network_quality", "preference_score"],
        "rows": rows,
    }, "comparison")


def recommend(comparison):
    validated = validate(comparison, "comparison")
    prefs = validated["preferences"]
    rows = [row for row in validated["rows"] if row["offer_id"] not in prefs["excluded_offer_ids"]]
    for row in rows:
        row["recommendation_score"] = interest_score(row, prefs)
        row["explanations"] = explain(row, prefs)
    rows.sort(key=lambda row: (-row["recommendation_score"], row["offer_id"]))
    rows = rows[:prefs["top_n"]]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return validate({
        "schema_version": 1, "synthetic_fixture": True,
        "residency_tag": validated["residency_tag"], "context": validated["context"],
        "preferences": prefs, "rows": rows,
    }, "recommendations")


def run(payload):
    comparison = compare(payload)
    recommendations = recommend(comparison)
    return {"status": "ok", "schema_version": 1, "synthetic_fixture": True,
            "comparison": comparison, "recommendations": recommendations}


def reject_constant(value):
    raise ValidationError("Non-finite JSON numeric constant")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as source:
            payload = json.load(source, parse_constant=reject_constant)
        result = run(payload)
    except (OSError, UnicodeError):
        result = {"status": "error", "message": "Unable to read input file"}
    except json.JSONDecodeError:
        result = {"status": "error", "message": "Invalid JSON input"}
    except ValidationError as error:
        result = {"status": "error", "message": str(error)}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
