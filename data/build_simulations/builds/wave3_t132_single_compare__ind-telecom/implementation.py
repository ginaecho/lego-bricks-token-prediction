"""Synthetic telecom plan comparison; local demonstration, not compliance certification."""
import csv
import io
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            name + " must be a finite number in range")
    return value


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required fields")
    require(value.keys() <= set(required) | set(optional), "Unknown fields")


def quantity(value, kind):
    fields(value, ("value", "unit"))
    scales = {"data": {"MB": .001, "GB": 1, "TB": 1000},
              "voice": {"minutes": 1, "hours": 60},
              "price": {"USD": 1, "cents": .01}}
    require(isinstance(value["unit"], str) and value["unit"] in scales[kind],
            "Unsupported " + kind + " unit")
    result = number(value["value"], kind) * scales[kind][value["unit"]]
    require(math.isfinite(result), "Normalized quantity is too large")
    return result


def validate_and_normalize(payload):
    """The only input boundary; returns private normalized state for ranking."""
    fields(payload, ("schema_version", "synthetic", "residency", "privacy", "subscriber",
                     "usage_csv", "fault_tickets", "support_chats", "products", "preferences"))
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "Unsupported schema_version")
    require(payload["synthetic"] is True, "Only clearly labeled synthetic inputs are accepted")
    region = payload["residency"]
    require(region in ("EU", "US"), "Unsupported residency")
    privacy = payload["privacy"]
    fields(privacy, ("purpose", "consent", "do_not_sell"))
    require(privacy["purpose"] == "product_comparison" and privacy["consent"] is True
            and privacy["do_not_sell"] is True, "Subscriber privacy authorization required")
    account = payload["subscriber"]
    fields(account, ("id", "phone", "imei", "residency", "synthetic"))
    text(account["id"], "subscriber id")
    require(isinstance(account["phone"], str)
            and re.fullmatch(r"\+1-202-555-01\d{2}", account["phone"]) is not None,
            "Use a fictitious +1-202-555-01xx phone")
    require(isinstance(account["imei"], str)
            and re.fullmatch(r"00000000000\d{4}", account["imei"]) is not None,
            "Use a synthetic zero-prefixed 15-digit IMEI")

    def record(item, linked=False):
        require(item["residency"] == region, "Record residency mismatch")
        require(item["synthetic"] is True, "Record must be labeled synthetic")
        if linked:
            require(item["subscriber_id"] == account["id"], "Unknown subscriber reference")

    record(account)
    expected = ["record_id", "subscriber_id", "residency", "synthetic", "period",
                "voice_minutes", "data_mb"]
    require(isinstance(payload["usage_csv"], str), "usage_csv must be CSV text")
    try:
        reader = csv.DictReader(io.StringIO(payload["usage_csv"]), strict=True)
        require(reader.fieldnames == expected, "Invalid CDR CSV header")
        usage = []
        seen = set()
        for row in reader:
            require(None not in row and None not in row.values(), "Invalid CDR row width")
            identifier = text(row["record_id"], "CDR id")
            require(identifier not in seen, "Duplicate CDR id")
            seen.add(identifier)
            require(row["synthetic"] == "true", "CDR must be labeled synthetic")
            row["synthetic"] = True
            record(row, linked=True)
            require(re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", row["period"]) is not None,
                    "Invalid CDR monthly period")
            usage.append({"period": row["period"],
                          "voice": number(float(row["voice_minutes"]), "CDR minutes"),
                          "data": number(float(row["data_mb"]), "CDR MB") / 1000})
    except (csv.Error, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("Invalid numeric or CSV CDR content") from None
    require(len({row["period"] for row in usage}) <= 1, "CDRs must cover one monthly period")
    totals = {key: number(sum(row[key] for row in usage), "Usage total")
              for key in ("data", "voice")}

    for key in ("fault_tickets", "support_chats", "products"):
        require(isinstance(payload[key], list), key + " must be a list")
        ids = set()
        for item in payload[key]:
            require(isinstance(item, dict), "Record must be an object")
            identifier = text(item.get("id"), "record id")
            require(identifier not in ids, "Duplicate record id")
            ids.add(identifier)
    for ticket in payload["fault_tickets"]:
        fields(ticket, ("id", "subscriber_id", "residency", "synthetic", "status", "summary"),
               ("billing_adjustment_cents", "billing_reason"))
        record(ticket, linked=True)
        require(ticket["status"] in ("open", "resolved"), "Invalid fault status")
        text(ticket["summary"], "fault summary")
        if "billing_adjustment_cents" in ticket:
            amount = ticket["billing_adjustment_cents"]
            require(type(amount) is int and abs(amount) <= 1000000,
                    "Billing adjustment must be integer cents within demonstration limits")
            text(ticket.get("billing_reason"), "billing adjustment reason")
        elif "billing_reason" in ticket:
            raise ValidationError("Billing reason requires an adjustment")
    for chat in payload["support_chats"]:
        fields(chat, ("id", "subscriber_id", "residency", "synthetic", "messages"))
        record(chat, linked=True)
        require(isinstance(chat["messages"], list), "Chat messages must be a list")
        for message in chat["messages"]:
            fields(message, ("speaker", "text"))
            require(message["speaker"] in ("subscriber", "agent"), "Invalid chat speaker")
            text(message["text"], "chat text")
    products = []
    require(bool(payload["products"]), "At least one product is required")
    for product in payload["products"]:
        fields(product, ("id", "residency", "synthetic", "monthly_price", "monthly_data",
                         "monthly_voice", "reliability_percent"))
        record(product)
        require(re.fullmatch(r"PLAN-[A-Z0-9-]+", product["id"]) is not None,
                "Product ids must be nonpersonal PLAN identifiers")
        reliability = number(product["reliability_percent"], "reliability")
        require(reliability <= 100, "Reliability must not exceed 100")
        products.append({"product_id": product["id"], "residency": region, "synthetic": True,
                         "monthly_price_usd": quantity(product["monthly_price"], "price"),
                         "monthly_data_gb": quantity(product["monthly_data"], "data"),
                         "monthly_voice_minutes": quantity(product["monthly_voice"], "voice"),
                         "reliability_percent": reliability})
    prefs = payload["preferences"]
    fields(prefs, ("max_monthly_price_usd", "minimum_data_gb", "minimum_voice_minutes", "weights"))
    for key in ("max_monthly_price_usd", "minimum_data_gb", "minimum_voice_minutes"):
        number(prefs[key], key)
    weights = prefs["weights"]
    fields(weights, ("price", "data", "voice", "reliability"))
    for key, value in weights.items():
        number(value, key + " weight")
    require(math.isfinite(sum(weights.values())) and sum(weights.values()) > 0,
            "Weights must have a positive finite total")
    return products, prefs, totals


def compare(payload):
    products, prefs, usage = validate_and_normalize(payload)
    needs = {"data": max(usage["data"], prefs["minimum_data_gb"]),
             "voice": max(usage["voice"], prefs["minimum_voice_minutes"])}
    weight_total = sum(prefs["weights"].values())
    weights = {key: value / weight_total for key, value in prefs["weights"].items()}
    comparisons = []
    for product in products:
        price = product["monthly_price_usd"]
        budget = prefs["max_monthly_price_usd"]
        # A budget is a preference, not a filter; unsuitable offers remain visible.
        utility = {"price": 1 if price == 0 else min(1, budget / price),
                   "data": 1 if needs["data"] == 0 else min(1, product["monthly_data_gb"] / needs["data"]),
                   "voice": 1 if needs["voice"] == 0 else min(1, product["monthly_voice_minutes"] / needs["voice"]),
                   "reliability": product["reliability_percent"] / 100}
        gaps = []
        if price > budget:
            gaps.append("over_budget")
        if product["monthly_data_gb"] < needs["data"]:
            gaps.append("insufficient_data")
        if product["monthly_voice_minutes"] < needs["voice"]:
            gaps.append("insufficient_voice")
        score = round(100 * sum(weights[key] * utility[key] for key in weights), 6)
        comparisons.append(dict(product, score=score, preference_gaps=gaps,
                                utility={key: round(value, 6) for key, value in utility.items()}))
    comparisons.sort(key=lambda row: (-row["score"], row["monthly_price_usd"], row["product_id"]))
    for rank, row in enumerate(comparisons, 1):
        row["rank"] = rank
    return {"status": "ok", "schema_version": 1, "synthetic": True,
            "residency": payload["residency"],
            "privacy": {"identifiers_and_transcripts_omitted": True, "no_external_processing": True},
            "usage_summary": dict(usage, residency=payload["residency"], synthetic=True),
            "required_allowances": dict(needs, residency=payload["residency"], synthetic=True),
            "support_summary": {"fault_ticket_count": len(payload["fault_tickets"]),
                                "chat_count": len(payload["support_chats"]),
                                "residency": payload["residency"], "synthetic": True},
            "comparison": comparisons, "recommended_product_id": comparisons[0]["product_id"],
            "notice": "Synthetic demonstration only; no compliance certification or billing execution."}


def reject_constant(_):
    raise ValidationError("Nonfinite JSON constants are not accepted")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py input.json")
        with open(args[0], encoding="utf-8") as source:
            payload = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = compare(payload)
        print(json.dumps(output, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        # Error responses deliberately omit input fragments, paths and subscriber data.
        print(json.dumps({"status": "error", "error": "Invalid input or unreadable input file"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
