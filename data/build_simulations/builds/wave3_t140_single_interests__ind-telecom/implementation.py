"""Deterministic synthetic telecom discovery; standard-library reference only."""
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


def obj(value, keys, path, optional=()):
    require(isinstance(value, dict), path + " must be an object")
    require(set(keys) <= value.keys() <= set(keys) | set(optional),
            path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def number(value, path, minimum=0, maximum=1_000_000):
    require(type(value) in (int, float) and minimum <= value <= maximum
            and math.isfinite(value), path + " must be a finite in-range number")


def strings(value, path):
    require(isinstance(value, list), path + " must be an array")
    for item in value:
        text(item, path + " entry")
    require(len(value) == len(set(value)), path + " must not contain duplicates")


def validate(payload):
    """One boundary for all records, including records later excluded by ranking."""
    obj(payload, ("schema_version", "synthetic", "residency", "subscriber_account",
                  "call_detail_records_csv", "network_fault_tickets", "catalog",
                  "exclusions", "top_k"), "input")
    require(payload["schema_version"] == "1.0", "unsupported schema_version")
    require(payload["synthetic"] is True, "only clearly labeled synthetic data is accepted")
    residency = payload["residency"]
    require(residency in ("EU", "US"), "residency must be EU or US")
    account = payload["subscriber_account"]
    obj(account, ("id", "residency", "synthetic", "phone", "imei", "privacy", "preferences"),
        "subscriber_account")
    seen = set()

    def record(value, path, linked=False):
        text(value["id"], path + ".id")
        require(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value["id"]) is not None,
                path + ".id must be a safe pseudonymous identifier")
        require(value["id"] not in seen, "duplicate record id")
        seen.add(value["id"])
        require(value["residency"] == residency, path + " violates data residency")
        require(value["synthetic"] is True, path + " must be synthetic")
        if linked:
            require(value["subscriber_id"] == account["id"], path + " references another subscriber")

    record(account, "subscriber_account")
    require(isinstance(account["phone"], str)
            and re.fullmatch(r"\+000\d{9}", account["phone"]) is not None,
            "phone must be fictitious +000 followed by nine digits")
    require(isinstance(account["imei"], str)
            and re.fullmatch(r"000000\d{9}", account["imei"]) is not None,
            "IMEI must be fictitious 000000 followed by nine digits")
    privacy = account["privacy"]
    obj(privacy, ("personalization_consent", "sale_share_opt_out", "deletion_requested",
                  "purpose"), "privacy")
    for key in ("personalization_consent", "sale_share_opt_out", "deletion_requested"):
        require(type(privacy[key]) is bool, "privacy flags must be boolean")
    require(privacy["personalization_consent"] and not privacy["sale_share_opt_out"]
            and not privacy["deletion_requested"],
            "privacy choices prohibit personalization")
    require(privacy["purpose"] == "personalized_discovery", "purpose is not permitted")
    require(isinstance(account["preferences"], dict), "preferences must be an object")
    for tag, weight in account["preferences"].items():
        text(tag, "preference tag")
        number(weight, "preference weight", 0, 100)
    exclusions = payload["exclusions"]
    obj(exclusions, ("ids", "tags"), "exclusions")
    strings(exclusions["ids"], "excluded ids")
    strings(exclusions["tags"], "excluded tags")
    require(type(payload["top_k"]) is int and 1 <= payload["top_k"] <= 50,
            "top_k must be an integer from 1 to 50")

    source = payload["call_detail_records_csv"]
    require(isinstance(source, str), "call_detail_records_csv must be text")
    headers = ["id", "subscriber_id", "residency", "synthetic", "phone", "imei",
               "data_mb", "call_minutes"]
    usage = []
    try:
        reader = csv.DictReader(io.StringIO(source), strict=True)
        require(reader.fieldnames == headers, "CSV header does not match shared schema")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()),
                    "CSV row has wrong column count")
            require(row["synthetic"] == "true", "CSV record must be synthetic")
            row["synthetic"] = True
            record(row, "usage record", linked=True)
            require(row["phone"] == account["phone"] and row["imei"] == account["imei"],
                    "usage identifiers do not match subscriber")
            for key in ("data_mb", "call_minutes"):
                try:
                    row[key] = float(row[key])
                except (ValueError, OverflowError):
                    raise ValidationError("CSV usage volume must be numeric") from None
                number(row[key], "CSV " + key)
            usage.append(row)
    except csv.Error:
        raise ValidationError("malformed call-detail CSV") from None

    tickets = payload["network_fault_tickets"]
    require(isinstance(tickets, list), "network_fault_tickets must be an array")
    for ticket in tickets:
        obj(ticket, ("id", "subscriber_id", "residency", "synthetic", "status",
                     "support_chat_transcript"), "ticket")
        record(ticket, "ticket", linked=True)
        require(ticket["status"] in ("open", "resolved"), "invalid ticket status")
        chat = ticket["support_chat_transcript"]
        require(isinstance(chat, list) and len(chat) > 0, "transcript must contain messages")
        for message in chat:
            obj(message, ("speaker", "text", "residency", "synthetic"), "chat message")
            require(message["speaker"] in ("subscriber", "agent"), "invalid chat speaker")
            text(message["text"], "chat text")
            require(message["residency"] == residency, "chat violates data residency")
            require(message["synthetic"] is True, "chat must be synthetic")

    require(isinstance(payload["catalog"], list), "catalog must be an array")
    for item in payload["catalog"]:
        obj(item, ("id", "residency", "synthetic", "tags", "kind"),
            "catalog record", optional=("billing_adjustment",))
        record(item, "catalog record")
        strings(item["tags"], "catalog tags")
        require(item["kind"] in ("plan", "support", "billing_adjustment"), "invalid catalog kind")
        if item["kind"] == "billing_adjustment":
            require("billing_adjustment" in item, "billing adjustment requires details")
            adjustment = item["billing_adjustment"]
            obj(adjustment, ("amount", "currency", "reason"), "billing adjustment")
            number(adjustment["amount"], "adjustment amount", -10000, 10000)
            require(adjustment["currency"] in ("EUR", "USD"), "unsupported currency")
            text(adjustment["reason"], "billing adjustment reason")
        else:
            require("billing_adjustment" not in item, "non-billing item has adjustment details")
    return account, usage, tickets


def recommend(payload):
    account, usage, tickets = validate(payload)
    residency = payload["residency"]
    signals = {}
    for tag, weight in sorted(account["preferences"].items()):
        if weight > 0:
            signals.setdefault(tag, []).append({
                "source": "explicit_preference", "record_ids": [account["id"]],
                "points": weight, "fact": "subscriber-assigned interest weight",
            })
    if sum(row["data_mb"] for row in usage) >= 1000:
        signals.setdefault("data", []).append({
            "source": "usage_threshold", "record_ids": sorted(row["id"] for row in usage),
            "points": 2, "fact": "aggregate synthetic data usage is at least 1000 MB",
        })
    open_ids = sorted(ticket["id"] for ticket in tickets if ticket["status"] == "open")
    if open_ids:
        signals.setdefault("reliability", []).append({
            "source": "open_network_fault", "record_ids": open_ids,
            "points": 3, "fact": "subscriber has an unresolved synthetic network ticket",
        })
    excluded = payload["exclusions"]
    ranked = []
    for item in payload["catalog"]:
        if item["id"] in excluded["ids"] or set(item["tags"]) & set(excluded["tags"]):
            continue
        evidence = []
        for tag in sorted(item["tags"]):
            for signal in signals.get(tag, []):
                evidence.append(dict(signal, tag=tag, residency=residency, synthetic=True))
        if not evidence:
            continue
        ranked.append({
            "id": item["id"], "residency": residency, "synthetic": True,
            "kind": item["kind"], "score": sum(e["points"] for e in evidence),
            "explanations": evidence,
        })
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return {
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "residency": residency, "recommendations": ranked[:payload["top_k"]],
        "policy": {
            "demonstration_only": True, "data_minimized": True,
            "billing_adjustments_executed": False,
            "ranking": "sum of matched preference and grounded usage/ticket points; ID tie-break",
        },
    }


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "provide exactly one input JSON file")
        with open(args[0], encoding="utf-8") as handle:
            payload = json.load(handle)
        result = recommend(payload)
    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
        message = str(exc) if isinstance(exc, ValidationError) else "input file cannot be read as JSON"
        print(json.dumps({"schema_version": "1.0", "status": "error", "error": message}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
